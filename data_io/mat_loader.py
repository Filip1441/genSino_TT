import os
import numpy as np
import scipy.io as sio

def load_phantom_mat(filepath: str) -> np.ndarray:
    """
    Loads a 3D phantom matrix from a .mat file.
    Expects the variable inside the .mat file to be named 'phantomRI'.
    
    Args:
        filepath (str): Absolute or relative path to the .mat file.
        
    Returns:
        np.ndarray: 3D numpy array containing the phantom data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If 'phantomRI' is not found in the file.
        ValueError: If the loaded variable is not a 3D array.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    phantom_data = None
    
    try:
        # Try loading as a standard MATLAB file (< v7.3)
        mat_data = sio.loadmat(filepath)
        
        if 'phantomRI' not in mat_data:
            raise KeyError(f"Variable 'phantomRI' not found in {filepath}")
            
        phantom_data = mat_data['phantomRI']
        
    except NotImplementedError:
        # Fallback for MATLAB v7.3+ which uses HDF5 format
        import h5py
        with h5py.File(filepath, 'r') as f:
            if 'phantomRI' not in f.keys():
                raise KeyError(f"Variable 'phantomRI' not found in {filepath} (HDF5 format)")
            
            # MATLAB saves HDF5 data in column-major order, 
            # so we transpose it to match standard numpy (row-major) format
            phantom_data = np.array(f['phantomRI']).T
            
    if phantom_data is None:
        raise ValueError("Failed to load data from the file.")
        
    if phantom_data.ndim != 3:
        raise ValueError(f"Expected 3D array, but got {phantom_data.ndim}D array.")
        
    return phantom_data