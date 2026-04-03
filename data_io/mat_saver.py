import scipy.io as sio
import numpy as np

def save_sinogram_mat(output_path, sino_amp, sino_ph, rayXY, detected_ri):
    """
    Saves the generated sinogram data and metadata into a MATLAB .mat file.
    Dynamically applies the detected background refractive index.
    """
    metadata = {
        'M': 48.41,
        'NA': 1.3,
        'cam_pix': 3.45,
        'do_NNC': 1,
        'downsampling': 3.2889,
        'dx': 0.234375, 
        'fpm': 0,
        'geometry': 'fixed',
        'lambda': 0.6328,
        'n_immersion': float(detected_ri),
        'obj_type': 'tech',
        'omit_reference': 0
    }
    
    metadata['SINOamp'] = sino_amp.astype(np.float32)
    metadata['SINOph'] = sino_ph.astype(np.float32)
    metadata['rayXY'] = rayXY.astype(np.float64)
    
    sio.savemat(output_path, metadata)