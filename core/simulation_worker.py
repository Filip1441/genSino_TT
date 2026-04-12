import os
import numpy as np
import concurrent.futures
import multiprocessing
from multiprocessing import shared_memory
from scipy.fft import fftn, fftshift, ifft2, ifftshift, fft2, ifftn
from scipy.ndimage import map_coordinates
from skimage.restoration import unwrap_phase
from PySide6.QtCore import QThread, Signal

def _center_crop_pad(arr, target_size):
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
                               approx_mode, interp_order, shm_real_name, shm_imag_name, ko_shape):
    
    shm_real = shared_memory.SharedMemory(name=shm_real_name)
    shm_imag = shared_memory.SharedMemory(name=shm_imag_name)
    try:
        KO_real = np.ndarray(ko_shape, dtype=np.float32, buffer=shm_real.buf)
        KO_imag = np.ndarray(ko_shape, dtype=np.float32, buffer=shm_imag.buf)

        u = (np.arange(NKP) - NKP // 2) * dkP
        U, V = np.meshgrid(u, u, indexing='ij')

        kx0 = kn * kx_beam
        ky0 = kn * ky_beam
        kz_beam = np.sqrt(max(0, kn**2 - kx0**2 - ky0**2))

        tx, ty, tz = step_data['phantom_tx'], step_data['phantom_ty'], step_data['phantom_tz']
        rx, ry, rz = step_data['phantom_rx'], step_data['phantom_ry'], step_data['phantom_rz']

        U_scat = U + kx0
        V_scat = V + ky0

        term = kn**2 - U_scat**2 - V_scat**2
        kzz = np.zeros_like(U)
        valid_mask = term >= 0
        kzz[valid_mask] = np.sqrt(term[valid_mask])

        domain = kzz > (4 * dkP)
        W = np.zeros_like(U)
        W[domain] = kzz[domain] - kz_beam

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

        Fp_r = map_coordinates(KO_real, coords_indices, order=interp_order, prefilter=False).reshape(NKP, NKP)
        Fp_i = map_coordinates(KO_imag, coords_indices, order=interp_order, prefilter=False).reshape(NKP, NKP)
        Fp = (Fp_r + 1j * Fp_i)
        Fp[~domain] = 0

        if tx != 0 or ty != 0 or tz != 0:
            Fp *= np.exp(-1j * 2 * np.pi * (U * tx + V * ty + W * tz))

        Fp_norm = np.zeros_like(Fp)
        Fp_norm[domain] = Fp[domain] / (1j * 4 * np.pi * kzz[domain])

        Up = fftshift(ifft2(ifftshift(Fp_norm))) / (dx_crop**2)

        pad_drop = (NKP - Nx) // 2
        Up_crop = Up[pad_drop : pad_drop+Nx, pad_drop : pad_drop+Nx]

        Fp_crop = fftshift(fft2(ifftshift(Up_crop))) * (dx_crop**2)
        
        shift_x = int(np.round(kx_beam / dk_crop))
        shift_y = int(np.round(ky_beam / dk_crop))
        Fp_centered = np.roll(Fp_crop, shift=(shift_x, shift_y), axis=(0, 1))

        Fp_final = _center_crop_pad(Fp_centered, n_orig)
        Fp_final = np.roll(Fp_final, shift=(-shift_x, -shift_y), axis=(0, 1))

        dxp = 1.0 / (dk_crop * n_orig)
        Up_final = fftshift(ifft2(ifftshift(Fp_final))) / (dxp**2)

        if approx_mode == 'Rytov':
            amp = np.exp(Up_final.real)
            ph = unwrap_phase(Up_final.imag)
        else:
            Up_born = Up_final + 1
            amp = np.abs(Up_born)
            ph = unwrap_phase(np.angle(Up_born))

        return j, amp.astype(np.float32), ph.astype(np.float32)

    finally:
        shm_real.close()
        shm_imag.close()


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
        # Force strict 32-bit floats to prevent 2x RAM explosions
        n_orig = self.phantom.shape[0]
        dxo = np.float32(self.metadata['dx'])
        lam = np.float32(self.metadata['lambda'])
        n_imm = np.float32(self.metadata['n_immersion'])
        kn = n_imm / lam
        dko = np.float32(1.0 / (dxo * n_orig))

        Bk0 = 2 * kn
        k0_max = np.max(np.abs(self.rayXY))
        Bk = Bk0 + 2 * k0_max
        NKo = int(2 + np.round(Bk / dko / 2.0) * 2) 
        dxu = np.float32(dxo * n_orig / NKo)
        
        zpc = 1.0 
        NKP = int(np.round(NKo * zpc / 2.0) * 2)
        dkP = np.float32(1.0 / (dxu * NKP))

        fft_cores = multiprocessing.cpu_count()

        k_factor = np.float32((2 * np.pi * kn) ** 2)
        phantom_sq = self.phantom ** 2
        n_imm_sq = np.float32(n_imm ** 2)
        
        potential_orig = k_factor * (np.float32(1.0) - (phantom_sq / n_imm_sq))
        potential_orig = potential_orig.astype(np.complex64)
        del phantom_sq # Aggressive memory clearing

        mask_3d = self._get_spherical_mask_3d(n_orig)
        potential_orig *= mask_3d
        del mask_3d

        # -------------------------------------------------------------
        # RAM-Optimized Fourier Resampling 
        # -------------------------------------------------------------
        potential_orig = ifftshift(potential_orig)
        tmp_KO = fftn(potential_orig, workers=fft_cores)
        del potential_orig # Clear before allocating padding
        
        tmp_KO = fftshift(tmp_KO)
        pad_resample = (NKo - n_orig) // 2
        tmp_KO = np.pad(tmp_KO, 
                        ((pad_resample, pad_resample), 
                         (pad_resample, pad_resample), 
                         (pad_resample, pad_resample)), mode='constant', constant_values=0)
        
        # Scaling after Fourier padding is required to preserve physical amplitude
        scale_factor = np.float32((NKo / n_orig) ** 3)
        tmp_KO *= scale_factor
        
        tmp_KO = ifftshift(tmp_KO)
        potential_resampled = ifftn(tmp_KO, workers=fft_cores)
        del tmp_KO 

        potential_resampled = fftshift(potential_resampled).astype(np.complex64)
        
        pad_width = (NKP - NKo) // 2
        potential_padded = np.pad(potential_resampled, 
                                  ((pad_width, pad_width), 
                                   (pad_width, pad_width), 
                                   (pad_width, pad_width)), mode='constant', constant_values=0)
        del potential_resampled

        # Final K-Space Generation
        potential_padded = ifftshift(potential_padded)
        KO = fftn(potential_padded, workers=fft_cores)
        del potential_padded
        
        KO = fftshift(KO)
        KO *= np.float32(dxu ** 3)
        
        # Prevent Worker Memory Duplication: Strictly slice real and imaginary floats!
        KO_real = np.ascontiguousarray(KO.real, dtype=np.float32)
        KO_imag = np.ascontiguousarray(KO.imag, dtype=np.float32)
        del KO

        Nx = int(np.round(NKP * dkP / dko / 2.0) * 2)
        dx_crop = np.float32(1.0 / (dkP * NKP))
        dk_crop = np.float32(1.0 / (dx_crop * Nx))

        num_projections = len(self.motion_sequence)
        SINOamp = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)
        SINOph = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)

        shm_real = shared_memory.SharedMemory(create=True, size=KO_real.nbytes)
        shm_imag = shared_memory.SharedMemory(create=True, size=KO_imag.nbytes)
        
        try:
            KO_real_shared = np.ndarray(KO_real.shape, dtype=KO_real.dtype, buffer=shm_real.buf)
            KO_imag_shared = np.ndarray(KO_imag.shape, dtype=KO_imag.dtype, buffer=shm_imag.buf)
            np.copyto(KO_real_shared, KO_real)
            np.copyto(KO_imag_shared, KO_imag)

            del KO_real, KO_imag # Final local cleanup

            num_workers = min(max(1, fft_cores - 2), 16)

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
                        self.approx_mode, self.interp_order, shm_real.name, shm_imag.name, KO_real_shared.shape
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
            shm_real.close()
            shm_imag.close()
            try:
                shm_real.unlink()
                shm_imag.unlink()
            except FileNotFoundError:
                pass

    def stop(self):
        self.is_running = False