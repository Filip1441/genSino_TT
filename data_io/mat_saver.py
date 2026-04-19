"""
MATLAB File Export Utility
--------------------------
Handles the storage of generated synthetic sinograms and 
associated beam geometry into .mat files.
"""

import scipy.io as sio
import numpy as np

def save_sinogram_mat(output_path: str, sino_amp: np.ndarray, 
                      sino_ph: np.ndarray, rayXY: np.ndarray, metadata: dict):
    """
    Saves sinogram data and metadata to a MATLAB-compatible .mat file.
    
    Args:
        output_path: Target .mat file path.
        sino_amp: 3D array of amplitude projections.
        sino_ph: 3D array of phase projections (unwrapped).
        rayXY: 2xN array of beam illumination positions.
        metadata: Dictionary containing physical/optical experiment parameters.
    """
    save_dict = metadata.copy()
    
    # Add simulation results to the dictionary
    save_dict['SINOamp'] = sino_amp.astype(np.float32)
    save_dict['SINOph'] = sino_ph.astype(np.float32)
    save_dict['rayXY'] = rayXY.astype(np.float64)
    
    sio.savemat(output_path, save_dict)