"""
Data Processing Workers
-----------------------
This module contains QThread-based workers for background data operations,
such as loading phantoms and saving results, to keep the UI responsive.
"""

import numpy as np
from PySide6.QtCore import QThread, Signal
from data_io.mat_loader import load_phantom_mat
from core.utils import normalize_to_uint8

def preprocess_volume_to_rgba(volume_data: np.ndarray) -> np.ndarray:
    """
    Converts 3D volume data to an RGBA array for 3D visualization.
    Downsamples the data for performance and maps intensities to colors.
    
    Args:
        volume_data: 3D numpy array of refractive index values.
        
    Returns:
        np.ndarray: RGBA array ready for gl.GLVolumeItem.
    """
    # Downsample for faster rendering in the 3D view
    display_data = volume_data[::2, ::2, ::2]
    display_shape = display_data.shape
    bg_value = display_data[0, 0, 0] # Assume the corner is background
    
    rgba = np.zeros(display_shape + (4,), dtype=np.ubyte)
    tolerance = 1e-5
    
    # Identify non-background voxels (the phantom)
    phantom_mask = np.abs(display_data - bg_value) > tolerance
    
    if np.any(phantom_mask):
        phantom_values = display_data[phantom_mask]
        # Normalize intensities to 0-255 for the color channels
        norm_byte = normalize_to_uint8(phantom_values, default_value=255)
        
        # Prepare an RGBA block: use normalized byte for R, G, B; 50 for alpha
        pixel_block = np.empty((norm_byte.size, 4), dtype=np.ubyte)
        pixel_block[:, :3] = norm_byte[:, None]
        pixel_block[:, 3] = 50
        
        rgba[phantom_mask] = pixel_block
        
    return rgba

class DataLoaderWorker(QThread):
    """Worker thread for loading 3D phantom files from disk."""
    finished_signal = Signal(np.ndarray, np.ndarray)
    error_signal = Signal(str)

    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath

    def run(self):
        """Executes the loading and preprocessing in a background thread."""
        try:
            # Load the raw .mat data
            volume_data = load_phantom_mat(self.filepath)
            # Preprocess for the 3D viewer
            rgba_data = preprocess_volume_to_rgba(volume_data)
            self.finished_signal.emit(volume_data, rgba_data)
        except Exception as e:
            self.error_signal.emit(str(e))

class DataSaverWorker(QThread):
    """Generic worker thread for invoking save functions in the background."""
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, save_func, filepath, *args, parent=None):
        super().__init__(parent)
        self.save_func = save_func
        self.filepath = filepath
        self.args = args

    def run(self):
        """Executes the provided save function."""
        try:
            self.save_func(self.filepath, *self.args)
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
