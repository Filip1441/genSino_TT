import os
import numpy as np
import concurrent.futures
from multiprocessing import shared_memory
from scipy.fft import fftn, fftshift, ifft2, ifftshift
from scipy.ndimage import map_coordinates
from PySide6.QtCore import QThread, Signal

def _process_single_projection(j, step_data, kx_beam, ky_beam, kn, dk_pad, n_pad, center_idx_pad, pad_before, n_orig, approx_mode, interp_order, shm_name, ko_shape, ko_dtype, dx):
    """
    Top-level function required for Windows multiprocessing.
    Attaches to the shared memory block, processes one projection, and returns the result.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # Reconstruct the 3D KO array from shared memory (Zero RAM duplication)
        KO = np.ndarray(ko_shape, dtype=ko_dtype, buffer=shm.buf)

        # Recreate coordinate grid locally for this process
        u = np.fft.fftfreq(n_pad, d=dx)
        v = np.fft.fftfreq(n_pad, d=dx)
        U, V = np.meshgrid(np.fft.fftshift(u), np.fft.fftshift(v), indexing='ij')

        kz_beam = np.sqrt(max(0, kn**2 - kx_beam**2 - ky_beam**2))

        tx = step_data['phantom_tx']
        ty = step_data['phantom_ty']
        tz = step_data['phantom_tz']
        rx = step_data['phantom_rx']
        ry = step_data['phantom_ry']
        rz = step_data['phantom_rz']

        U_scat = U + kx_beam
        V_scat = V + ky_beam

        term = kn**2 - U_scat**2 - V_scat**2
        
        kzz = np.zeros_like(U)
        valid_mask = term >= 0
        kzz[valid_mask] = np.sqrt(term[valid_mask])

        domain = kzz > (4 * dk_pad)
        
        W = np.zeros_like(U)
        W[domain] = kzz[domain] - kz_beam

        rad_x = np.radians(-rx)
        rad_y = np.radians(-ry)
        rad_z = np.radians(-rz)

        cx, sx = np.cos(rad_x), np.sin(rad_x)
        cy, sy = np.cos(rad_y), np.sin(rad_y)
        cz, sz = np.cos(rad_z), np.sin(rad_z)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R = Rz @ Ry @ Rx

        coords_3d = np.vstack((U.ravel(), V.ravel(), W.ravel()))
        rotated_coords = R @ coords_3d

        coords_indices = (rotated_coords / dk_pad) + center_idx_pad

        Fp_real = map_coordinates(KO.real, coords_indices, order=interp_order).reshape(n_pad, n_pad)
        Fp_imag = map_coordinates(KO.imag, coords_indices, order=interp_order).reshape(n_pad, n_pad)
        Fp = (Fp_real + 1j * Fp_imag)

        Fp[~domain] = 0

        if tx != 0 or ty != 0 or tz != 0:
            Fp *= np.exp(-1j * 2 * np.pi * (U * tx + V * ty + W * tz))

        Fp_norm = np.zeros_like(Fp)
        Fp_norm[domain] = Fp[domain] / (1j * 4 * np.pi * kzz[domain])

        Up_full = fftshift(ifft2(ifftshift(Fp_norm)))
        Up = Up_full[pad_before : pad_before + n_orig, pad_before : pad_before + n_orig]

        if approx_mode == 'Rytov':
            amp, ph = np.exp(Up.real), Up.imag
        else:
            Up_born = Up + 1
            amp, ph = np.abs(Up_born), np.angle(Up_born)

        return j, amp.astype(np.float32), ph.astype(np.float32)

    finally:
        # Always close the shared memory connection in the worker
        shm.close()


class SimulationWorker(QThread):
    """
    Heavy duty pre-computation worker.
    Core logic rewritten to mirror rigorous MATLAB implementation:
    - Applies a smooth 3D spherical mask to the object BEFORE projection.
    - 3D Zero Padding applied with 'constant' zeros for a perfectly sterile background.
    - Strictly avoids Ewald sphere singularity using the MATLAB hard-limit criterion.
    - Interpolation order can be dynamically changed (1 for speed, 3 for quality).
    - FULLY PARALLELIZED using ProcessPoolExecutor and Shared Memory.
    """
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
        n_pad = 600  
        
        pad_before = (n_pad - n_orig) // 2
        pad_after = n_pad - n_orig - pad_before
        pad_width = ((pad_before, pad_after), (pad_before, pad_after), (pad_before, pad_after))
        
        dx = float(self.metadata['dx'])
        lam = float(self.metadata['lambda'])
        n_imm = float(self.metadata['n_immersion'])
        kn = n_imm / lam
        
        k_factor = (2 * np.pi * kn) ** 2
        potential_orig = k_factor * (1 - (self.phantom ** 2) / (n_imm ** 2))
        
        mask_3d = self._get_spherical_mask_3d(n_orig)
        potential_orig *= mask_3d
        
        potential = np.pad(potential_orig, pad_width, mode='constant', constant_values=0)
        KO = fftshift(fftn(ifftshift(potential)))
        
        dk_pad = 1.0 / (n_pad * dx)
        center_idx_pad = n_pad // 2
        num_projections = len(self.motion_sequence)
        
        SINOamp = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)
        SINOph = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)

        # ---------------------------------------------------------
        # SHARED MEMORY SETUP
        # ---------------------------------------------------------
        # Allocate shared memory block size of KO
        shm = shared_memory.SharedMemory(create=True, size=KO.nbytes)
        try:
            # Create a NumPy array backed by shared memory and copy KO into it
            KO_shared = np.ndarray(KO.shape, dtype=KO.dtype, buffer=shm.buf)
            np.copyto(KO_shared, KO)

            # Auto-detect CPU cores (leave 2 cores free for OS and GUI stability)
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
                        kn, dk_pad, n_pad, center_idx_pad, pad_before, n_orig,
                        self.approx_mode, self.interp_order, shm.name, KO.shape, KO.dtype, dx
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
            # ---------------------------------------------------------
            # CLEANUP
            # ---------------------------------------------------------
            # Crucial to prevent memory leaks in Windows
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def stop(self):
        self.is_running = False