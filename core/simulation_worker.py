"""
Holographic Tomography Simulation Engine
----------------------------------------
The core of the "Virtual Lab". Implements a wave-optics forward model based on 
the Rytov/Born approximation. It computes the 3D potential of a phantom and 
extracts slices from the Ewald sphere in parallel using multiprocessing.
"""

import numpy as np
import concurrent.futures
import multiprocessing
from multiprocessing import shared_memory
from scipy.fft import fftn, fftshift, ifft2, ifftshift, fft2, ifftn
from scipy.ndimage import map_coordinates
from skimage.restoration import unwrap_phase
from PySide6.QtCore import QThread, Signal
from core.utils import get_rotation_matrix

# Default physical and optical parameters for the simulation
DEFAULT_METADATA = {
    'M': 48.41,                # Magnification
    'NA': 1.3,                 # Numerical Aperture
    'cam_pix': 3.45,           # Camera pixel size 
    'do_NNC': 1,               # Non Negativity constraint
    'downsampling': 3.2889,    # Resampling factor
    'dx': 0.234375,            # Voxel size (um)
    'fpm': 0,                  # Fourier phase mask
    'geometry': 'fixed',       # System geometry
    'lambda': 0.6328,          # Wavelength 
    'n_immersion': 1.33,       # Refractive index of immersion medium
    'obj_type': 'tech',        # Phantom type
    'omit_reference': 0        # Reference wave toggle
}

def _center_crop_pad(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Helper to crop or pad an array from/to its center."""
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
    """
    Subprocess function to generate a single projection.
    Extracts a 2D Ewald sphere slice from the 3D scattered potential.
    """
    
    # Access the shared memory segments
    shm_real = shared_memory.SharedMemory(name=shm_real_name)
    shm_imag = shared_memory.SharedMemory(name=shm_imag_name)
    try:
        # Wrap shared buffers as numpy arrays
        KO_real = np.ndarray(ko_shape, dtype=np.float32, buffer=shm_real.buf)
        KO_imag = np.ndarray(ko_shape, dtype=np.float32, buffer=shm_imag.buf)

        # Coordinate grid in K-space
        u = (np.arange(NKP) - NKP // 2) * dkP
        U, V = np.meshgrid(u, u, indexing='ij')

        # Wave vector calculation
        kx0 = kn * kx_beam
        ky0 = kn * ky_beam
        kz_beam = np.sqrt(max(0, kn**2 - kx0**2 - ky0**2))

        tx, ty, tz = step_data['phantom_tx'], step_data['phantom_ty'], step_data['phantom_tz']
        rx, ry, rz = step_data['phantom_rx'], step_data['phantom_ry'], step_data['phantom_rz']

        U_scat = U + kx0
        V_scat = V + ky0

        # Ewald sphere surface calculation
        term = kn**2 - U_scat**2 - V_scat**2
        kzz = np.zeros_like(U)
        valid_mask = term >= 0
        kzz[valid_mask] = np.sqrt(term[valid_mask])

        domain = kzz > (4 * dkP) # Bandwidth limitation
        W = np.zeros_like(U)
        W[domain] = kzz[domain] - kz_beam

        # Apply 3D rotation to the extraction coordinates
        R = get_rotation_matrix(-rx, -ry, -rz)
        coords_3d = np.vstack((U.ravel(), V.ravel(), W.ravel()))
        rotated_coords = R @ coords_3d
        coords_indices = (rotated_coords / dkP) + (NKP // 2)

        # Interpolate from the 3D potential (prefilter=False because we do it once in the main thread)
        Fp_r = map_coordinates(KO_real, coords_indices, order=interp_order, prefilter=False).reshape(NKP, NKP)
        Fp_i = map_coordinates(KO_imag, coords_indices, order=interp_order, prefilter=False).reshape(NKP, NKP)
        Fp = (Fp_r + 1j * Fp_i)
        Fp[~domain] = 0

        # Apply shifts (translations)
        if tx != 0 or ty != 0 or tz != 0:
            Fp *= np.exp(-1j * 2 * np.pi * (U * tx + V * ty + W * tz))

        # Scattered field normalization
        Fp_norm = np.zeros_like(Fp)
        Fp_norm[domain] = Fp[domain] / (1j * 4 * np.pi * kzz[domain])

        # Inverse Fourier to real space (projection plane)
        Up = fftshift(ifft2(ifftshift(Fp_norm))) / (dx_crop**2)

        # Crop and center-align the optical field
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

        # Apply approximation model (Rytov or Born)
        if approx_mode == 'Rytov':
            amp = np.exp(Up_final.real)
            ph = unwrap_phase(Up_final.imag)
        else:
            Up_born = Up_final + 1
            amp = np.abs(Up_born)
            ph = unwrap_phase(np.angle(Up_born))

        return j, amp.astype(np.float32), ph.astype(np.float32)

    finally:
        # Avoid resource leaks by closing shm handles
        shm_real.close()
        shm_imag.close()


class SimulationWorker(QThread):
    """
    Main background worker for the high-performance HT simulation.
    Handles the 3D potential calculation and orchestrates the parallel projection generation.
    """
    progress_updated = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal(np.ndarray, np.ndarray, dict)

    def __init__(self, phantom, rayXY, motion_sequence, metadata=None, approx_mode='Rytov', interp_order=1):
        super().__init__()
        self.phantom = phantom.astype(np.complex64)
        self.rayXY = rayXY
        self.motion_sequence = motion_sequence
        self.approx_mode = approx_mode
        self.interp_order = interp_order
        self.is_running = True
        
        # Load and update metadata
        self.metadata = DEFAULT_METADATA.copy()
        if metadata:
            self.metadata.update(metadata)
        
        # Automatic background index extraction if not specified
        if self.metadata['n_immersion'] is None or self.metadata['n_immersion'] == 1.5123:
            self.metadata['n_immersion'] = float(np.real(self.phantom[0, 0, 0]))

    def _get_spherical_mask_3d(self, n: int) -> np.ndarray:
        """Generates a smooth spherical window to prevent Fourier artifacts at boundaries."""
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
        """Orchestrates the massive 3D FFT and the parallel extraction loop."""
        n_orig = self.phantom.shape[0]
        dxo = np.float32(self.metadata['dx'])
        lam = np.float32(self.metadata['lambda'])
        n_imm = np.float32(self.metadata['n_immersion'])
        kn = n_imm / lam
        dko = np.float32(1.0 / (dxo * n_orig))

        # Fourier space padding for sufficient sampling
        Bk0 = 2 * kn
        k0_max = np.max(np.abs(self.rayXY))
        Bk = Bk0 + 2 * k0_max
        NKo = int(2 + np.round(Bk / dko / 2.0) * 2) 
        dxu = np.float32(dxo * n_orig / NKo)
        
        zpc = 1
        NKP = int(np.round(NKo * zpc / 2.0) * 2)
        dkP = np.float32(1.0 / (dxu * NKP))

        fft_cores = multiprocessing.cpu_count()

        # Step 1: Calculate the scattered potential (f = k0^2 * (1 - n^2/n_imm^2))
        k_factor = np.float32((2 * np.pi * kn) ** 2)
        phantom_sq = self.phantom ** 2
        n_imm_sq = np.float32(n_imm ** 2)
        
        potential_orig = k_factor * (np.float32(1.0) - (phantom_sq / n_imm_sq))
        potential_orig = potential_orig.astype(np.complex64)
        del phantom_sq 

        # Apply spherical window
        mask_3d = self._get_spherical_mask_3d(n_orig)
        potential_orig *= mask_3d
        del mask_3d

        # Step 2: 3D FFT to K-space and resampling to padded grid
        potential_orig = ifftshift(potential_orig)
        tmp_KO = fftn(potential_orig, workers=fft_cores)
        del potential_orig 
        
        tmp_KO = fftshift(tmp_KO)
        pad_resample = (NKo - n_orig) // 2
        
        # Use pre-allocation instead of np.pad for better memory management
        tmp_padded = np.zeros((NKo, NKo, NKo), dtype=tmp_KO.dtype)
        tmp_padded[pad_resample:pad_resample+n_orig, pad_resample:pad_resample+n_orig, pad_resample:pad_resample+n_orig] = tmp_KO
        del tmp_KO
        
        scale_factor = np.float32((NKo / n_orig) ** 3)
        tmp_padded *= scale_factor
        
        tmp_padded = ifftshift(tmp_padded)
        potential_resampled = ifftn(tmp_padded, workers=fft_cores)
        del tmp_padded 

        potential_resampled = fftshift(potential_resampled).astype(np.complex64)
        
        # Step 3: Final padding for Ewald sphere resolution - use pre-allocated zeros to avoid multiple copies
        pad_width = (NKP - NKo) // 2
        potential_padded = np.zeros((NKP, NKP, NKP), dtype=np.complex64)
        potential_padded[pad_width:pad_width+NKo, pad_width:pad_width+NKo, pad_width:pad_width+NKo] = potential_resampled
        del potential_resampled

        potential_padded = ifftshift(potential_padded)
        KO = fftn(potential_padded, workers=fft_cores)
        del potential_padded
        
        KO = fftshift(KO)
        KO *= np.float32(dxu ** 3) # Final potential in K-space
        
        # Prepare for shared memory (OPTIMIZED: avoid intermediate contiguous float32 copies)
        shm_size = KO.nbytes // 2
        shm_real = shared_memory.SharedMemory(create=True, size=shm_size)
        shm_imag = shared_memory.SharedMemory(create=True, size=shm_size)
        
        try:
            # Wrap shared buffers as numpy arrays
            KO_real_shared = np.ndarray(KO.shape, dtype=np.float32, buffer=shm_real.buf)
            KO_imag_shared = np.ndarray(KO.shape, dtype=np.float32, buffer=shm_imag.buf)
            
            # Prefilter for cubic interpolation once here, so workers don't have to do it (prevents RAM explosion)
            if self.interp_order > 1:
                from scipy.ndimage import spline_filter
                # Process real part
                coeffs = spline_filter(KO.real, order=self.interp_order, output=np.float32)
                np.copyto(KO_real_shared, coeffs)
                del coeffs
                # Process imag part
                coeffs = spline_filter(KO.imag, order=self.interp_order, output=np.float32)
                np.copyto(KO_imag_shared, coeffs)
                del coeffs
            else:
                # Directly copy from complex64 views (KO.real/imag) into shared memory buffers.
                # This is much more memory efficient than using np.ascontiguousarray first.
                np.copyto(KO_real_shared, KO.real)
                np.copyto(KO_imag_shared, KO.imag)

            # Record shape then delete the massive complex array
            ko_shape_actual = KO.shape
            del KO

            # Allocate sinogram storage AFTER freeing the big volume
            num_projections = len(self.motion_sequence)
            SINOamp = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)
            SINOph = np.zeros((n_orig, n_orig, num_projections), dtype=np.float32)

            Nx = int(np.round(NKP * dkP / dko / 2.0) * 2)
            dx_crop = np.float32(1.0 / (dkP * NKP))
            dk_crop = np.float32(1.0 / (dx_crop * Nx))
            # Step 4: Parallel extraction of projections using a ProcessPoolExecutor
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

                # Monitoring the loop and updating the UI progress
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
                        self.error_signal.emit(f"Projection error: {exc}")

            if self.is_running:
                self.finished_signal.emit(SINOamp, SINOph, self.metadata)

        finally:
            # Cleanup shared memory
            shm_real.close()
            shm_imag.close()
            try:
                shm_real.unlink()
                shm_imag.unlink()
            except FileNotFoundError:
                pass

    def stop(self):
        """Stops the calculation process."""
        self.is_running = False