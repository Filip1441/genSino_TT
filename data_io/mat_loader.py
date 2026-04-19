"""
MATLAB File Loading Engine
--------------------------
Provides robust support for loading 3D phantoms from .mat files.
Supports both legacy MAT-files (v7.2 and older) via scipy and 
HDF5-based MAT-files (v7.3) via h5py.
"""

import os
import numpy as np
import scipy.io as sio

try:
    import h5py
except ImportError:
    h5py = None

def load_phantom_mat(filepath: str) -> np.ndarray:
    """
    Loads a 3D phantom array from a MATLAB variable named 'phantomRI'.
    
    Args:
        filepath: Path to the .mat file.
        
    Returns:
        np.ndarray: 3D array of refractive index data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If an HDF5 file is encountered but h5py is missing.
        KeyError: If 'phantomRI' is missing from the file.
        ValueError: If data structure is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    phantom_data = None
    
    try:
        # Standard MAT file loading
        mat_data = sio.loadmat(filepath)
        
        if 'phantomRI' not in mat_data:
            raise KeyError(f"Variable 'phantomRI' not found in {filepath}")
            
        phantom_data = mat_data['phantomRI']
        
    except NotImplementedError:
        # Fallback for MATLAB v7.3 (HDF5 based)
        if h5py is None:
            raise ImportError("h5py is required to load MATLAB v7.3+ files. Please install it.")
            
        with h5py.File(filepath, 'r') as f:
            if 'phantomRI' not in f.keys():
                raise KeyError(f"Variable 'phantomRI' not found in {filepath} (HDF5 format)")
            
            # Note: HDF5 storage is transposed relative to standard MAT loading
            phantom_data = np.ascontiguousarray(np.array(f['phantomRI']).T)
            
    if phantom_data is None:
        raise ValueError("Failed to load data from the file.")
        
    if phantom_data.ndim != 3:
        raise ValueError(f"Expected 3D array, but got {phantom_data.ndim}D array.")
        
    return phantom_data