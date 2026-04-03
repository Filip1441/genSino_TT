import scipy.io as sio
import numpy as np

def save_sinogram_mat(output_path, sino_amp, sino_ph, rayXY, metadata):
    """
    Saves the generated sinogram data and metadata into a MATLAB .mat file.
    Uses the centralized metadata dictionary provided by the main window.
    """
    # Create a local copy to avoid modifying the original dictionary in the GUI
    save_dict = metadata.copy()
    
    save_dict['SINOamp'] = sino_amp.astype(np.float32)
    save_dict['SINOph'] = sino_ph.astype(np.float32)
    save_dict['rayXY'] = rayXY.astype(np.float64)
    
    sio.savemat(output_path, save_dict)