import os
import numpy as np
import concurrent.futures
from multiprocessing import shared_memory
from scipy.fft import fftn, fftshift, ifft2, ifftshift, fft2
from scipy.ndimage import map_coordinates, zoom
from skimage.restoration import unwrap_phase
from PySide6.QtCore import QThread, Signal

def _center_crop_pad(arr, target_size):
    """ Helper: Fast numpy crop or pad to an even square target_size """
    curr_size = arr.shape[0]
    if curr_size == target_size:
        return arr
    res = np.zeros((target_size, target_size), dtype=arr.dtype)
    if curr_size > target_size:
        start = (curr_size - target_size) // 2
        res[:, :] = arr[start:start+target_size, start:start+target_size]
    else:
        start = (target_size - curr_size) // 2
        res[start:start+curr_size, start:start+curr_size] = arr[:, :]
    return res

def _process_single_projection(j, step_data, kx_beam, ky_beam, kn, 
                               NKP, dkP, Nx, dx_crop, dk_crop, n_orig, 
                               approx_mode, interp_order, shm_name, ko_shape, ko_dtype):
    """
    Top-level function required for Windows multiprocessing.
    Attaches to the shared memory block, processes one projection, and returns the result.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # Reconstruct the 3D KO array from shared memory (Zero RAM duplication)
        KO = np.ndarray(ko_shape, dtype=ko_dtype, buffer=shm.buf)

        # [FIX 6]: Correct meshgrid orientation. 'ij' guarantees U is axis 0, V is axis 1
        # matching Python's array index order arr[dim0, dim1, dim2].
        u = (np.arange(NKP) - NKP // 2) * dkP
        U, V = np.meshgrid(u, u, indexing='ij')

        kz_beam = np.sqrt(max(0, kn**2 - kx_beam**2 - ky_beam**2))

        tx, ty, tz = step_data['phantom_tx'], step_data['phantom_ty'], step_data['phantom_tz']
        rx, ry, rz = step_data['phantom_rx'], step_data['phantom_ry'], step_data['phantom_rz']

        U_scat = U + kx_beam
        V_scat = V + ky_beam

        term = kn**2 - U_scat**2 - V_scat**2
        kzz = np.zeros_like(U)
        valid_mask = term >= 0
        kzz[valid_mask] = np.sqrt(term[valid_mask])

        # Avoid Ewald sphere singularity ring (domain definition)
        domain = kzz > (4 * dkP)
        W = np.zeros_like(U)
        W[domain] = kzz[domain] - kz_beam

        # Z-axis Rotation matrix (Kinematics application)
        rad_x, rad_y, rad_z = np.radians(-rx), np.radians(-ry), np.radians(-rz)
        cx, sx = np.cos(rad_x), np.sin(rad_x)
        cy, sy = np.cos(rad_y), np.sin(rad_y)
        cz, sz = np.cos(rad_z), np.sin(rad_z)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        coords_3d = np.vstack((U.ravel(), V.ravel(), W.ravel()))
        rotated_coords = R @ coords_3d
        coords_indices = (rotated_coords / dkP) + (NKP // 2)

        Fp_real = map_coordinates(KO.real, coords_indices, order=interp_order).reshape(NKP, NKP)
        Fp_imag = map_coordinates(KO.imag, coords_indices, order=interp_order).reshape(NKP, NKP)
        Fp = (Fp_real + 1j * Fp_imag)
        Fp[~domain] = 0

        # Translation application
        if tx != 0 or ty != 0 or tz != 0:
            Fp *= np.exp(-1j * 2 * np.pi * (U * tx + V * ty + W * tz))

        Fp_norm = np.zeros_like(Fp)
        Fp_norm[domain] = Fp[domain] / (1j * 4 * np.pi * kzz[domain])

        # [FIX 1.2]: Inverse FFT scaling step 1 (Physical normalization)
        Up = fftshift(ifft2(ifftshift(Fp_norm))) / (dx_crop**2)

        # [FIX 3.1]: Anti-aliasing spatial crop
        pad_drop = (NKP - Nx) // 2
        Up_crop = Up[pad_drop : pad_drop+Nx, pad_drop : pad_drop+Nx]

        # [FIX 3.2]: Back to K-Space for spectral wrapping
        Fp_crop = fftshift(fft2(ifftshift(Up_crop))) * (dx_crop**2)
        
        # [FIX 2]: Carrier wave centering / Spectral Shift
        shift_x = int(np.round(kx_beam / dk_crop))
        shift_y = int(np.round(ky_beam / dk_crop))
        Fp_centered = np.roll(Fp_crop, shift=(shift_x, shift_y), axis=(0, 1))

        # Size matching to target n_orig (MATLAB's zip=0 logic for GUI compliance)
        Fp_final = _center_crop_pad(Fp_centered, n_orig)
        Fp_final = np.roll(Fp_final, shift=(-shift_x, -shift_y), axis=(0, 1))

        # [FIX 1.3]: Final Spatial Scaling
        dxp = 1.0 / (dk_crop * n_orig)
        Up_final = ifft2(ifftshift(Fp_final)) / (dxp**2)

        # [FIX 5]: Physical Phase Unwrapping (Miguel 2D MATLAB replacement)
        if approx_mode == 'Rytov':
            amp = np.exp(Up_final.real)
            ph = unwrap_phase(Up_final.imag)
        else:
            Up_born = Up_final + 1
            amp = np.abs(Up_born)
            ph = unwrap_phase(np.angle(Up_born))

        return j, amp.astype(np.float32), ph.astype(np.float32)

    finally:
        shm.close()


class SimulationWorker(QThread):
    progress_updated = Signal(int)
    finished_signal = Signal(np.ndarray, np.ndarray)

    def __init__(self, phantom, metadata, rayXY, motion_sequence, approx_mode='Rytov', interp_order=1):
        super().__init__()
        self.phantom = phantom.astype(np.complex64)
        self.metadata = metadata
        self.rayXY = rayXY
        self.motion_sequence = motion_sequence
        self.approx_mode = approx_mode
        self.interp_order = interp_order
        self.is_running = True

    def _get_spherical_mask_3d(self, n):
        z, y, x = np.ogrid[-n/2:n/2, -n/2:n/2, -n/2:n/2]
        r = np.sqrt(x**2 + y**2 + z**2)
        max_r = n / 2.0
        transition = 10.0
        inner_r = max_r - transition
        
        mask = np.ones((n, n, n), dtype=np.float32)
        trans_zone = (r > inner_r) & (r <= max_r)
        mask[trans_zone] = 0.5 * (1 + np.cos(np.pi * (r[trans_zone] - inner_r) / transition))
        mask[r > max_r] = 0.0
        return mask

    def run(self):
        n_orig = self.phantom.shape[0]
        dxo = float(self.metadata['dx'])
        lam = float(self.metadata['lambda'])
        n_imm = float(self.metadata['n_immersion'])
        kn = n_imm / lam
        dko = 1.0 / (dxo * n_orig)

        # [FIX 4]: Dynamic Zero-Padding and Interpoaltion replacing hardcoded n_pad=600
        Bk0 = 2 * kn
        k0_max = np.max(np.abs(self.rayXY))
        Bk = Bk0 + 2 * k0_max
        NKo = int(2 + np.round(Bk / dko / 2.0) * 2) # Ensures spectrum perfectly fits Ewald sphere
        dxu = dxo * n_orig / NKo
        
        zpc = 2.0 
        NKP = int(np.round(NKo * zpc / 2.0) * 2) # Final Anti-aliasing padding size
        dkP = 1.0 / (dxu * NKP)

        k_factor = (2 * np.pi * kn) ** 2
        potential_orig = k_factor * (1 - (self.phantom ** 2) / (n_imm ** 2))
        mask_3d = self._get_spherical_mask_3d(n_orig)
        potential_orig *= mask_3d

        # Safely zoom complex numbers 
        zoom_factor = NKo / n_orig
        pot_real = zoom(potential_orig.real, zoom_factor, order=1)
        pot_imag = zoom(potential_orig.imag, zoom_factor, order=1)
        potential_resampled = pot_real + 1j * pot_imag

        pad_width = (NKP - NKo) // 2
        potential_padded = np.pad(potential_resampled, pad_width, mode='constant', constant_values=0)

        # [FIX 1.1]: Forward FFT Scaling (dxu^3)
        KO = fftshift(fftn(ifftshift(potential_padded))) * (dxu ** 3)
        
        # Precalculate spatial tracking variables for worker
        Nx = int(np.round(NKP * dkP / dko / 2.0) * 2)
        dx_crop = 1.0 / (dkP * NKP)
        dk_crop = 1.0 / (dx_crop * Nx)

        num_projections = len(self.motion_sequence)
        SINOamp = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)
        SINOph = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)

        # ---------------------------------------------------------
        # SHARED MEMORY SETUP
        # ---------------------------------------------------------
        shm = shared_memory.SharedMemory(create=True, size=KO.nbytes)
        try:
            KO_shared = np.ndarray(KO.shape, dtype=KO.dtype, buffer=shm.buf)
            np.copyto(KO_shared, KO)

            total_cores = os.cpu_count() or 4
            num_workers = max(1, total_cores - 2)

            # ---------------------------------------------------------
            # MULTIPROCESSING EXECUTION
            # ---------------------------------------------------------
            completed = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for j, step_data in enumerate(self.motion_sequence):
                    if not self.is_running:
                        break
                        
                    future = executor.submit(
                        _process_single_projection,
                        j, step_data, self.rayXY[0, j], self.rayXY[1, j],
                        kn, NKP, dkP, Nx, dx_crop, dk_crop, n_orig,
                        self.approx_mode, self.interp_order, shm.name, KO.shape, KO.dtype
                    )
                    futures[future] = j

                for future in concurrent.futures.as_completed(futures):
                    if not self.is_running:
                        executor.shutdown(wait=False)
                        break
                        
                    try:
                        j, amp, ph = future.result()
                        SINOamp[:, :, j] = amp
                        SINOph[:, :, j] = ph
                        
                        completed += 1
                        progress = int((completed / num_projections) * 100)
                        self.progress_updated.emit(progress)
                    except Exception as exc:
                        print(f"Projection generated an exception: {exc}")

            if self.is_running:
                self.finished_signal.emit(SINOamp, SINOph)

        finally:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def stop(self):
        self.is_running = False