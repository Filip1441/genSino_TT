"""
General Utilities Module
------------------------
Shared mathematical and processing functions used across the GenSino-TT project,
including data normalization and 3D rotation matrix calculation.
"""

import numpy as np

def normalize_to_uint8(arr: np.ndarray, default_value: int = 0) -> np.ndarray:
    """
    Normalizes a numerical array to the 0-255 range and converts it to uint8.
    
    Args:
        arr: The input numpy array to normalize.
        default_value: The value to use if the array is flat (max == min).
        
    Returns:
        np.ndarray: Normalized array as ubyte/uint8.
    """
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        normalized = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        # Avoid division by zero for constant-value arrays
        normalized = np.full(arr.shape, float(default_value))
        
    return normalized.astype(np.uint8)

def get_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Calculates a standard 3D rotation matrix using the Z-Y-X convention.
    
    Args:
        rx_deg: Rotation around the X-axis in degrees.
        ry_deg: Rotation around the Y-axis in degrees.
        rz_deg: Rotation around the Z-axis in degrees.
        
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    rad_x, rad_y, rad_z = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx, sx = np.cos(rad_x), np.sin(rad_x)
    cy, sy = np.cos(rad_y), np.sin(rad_y)
    cz, sz = np.cos(rad_z), np.sin(rad_z)

    # Individual rotation matrices
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx
