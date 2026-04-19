"""
Measurement and Motion Simulation Module
----------------------------------------
Handles the generation of scan trajectories and provides a worker for
real-time playback of measurement results in the GUI.
"""

import numpy as np
import random
from PySide6.QtCore import QThread, Signal

def generate_motion_sequence(num_projections: int, step_deg: float, is_galvo: bool, 
                             illumination_angle: float, kinematics: dict, noise: dict):
    """
    Generates a full sequence of beam and phantom transformations for the scan.
    
    Args:
        num_projections: Total number of angular steps.
        step_deg: Degrees between each projection.
        is_galvo: True for LAT (Galvo) mode, False for TTLAT (Object Rotation).
        illumination_angle: Base illumination angle in degrees.
        kinematics: Dict of bools and values for deterministic wobbles (X, Y, Z).
        noise: Dict of bools and values for randomized noise.
        
    Returns:
        tuple: (List of step dictionaries, 2xN array of beam positions).
    """
    sequence = []
    
    # Calculate beam offset in K-space coordinates
    radius = np.sin(np.radians(illumination_angle))
    if is_galvo:
        # Beam sweeps a circle in LAT mode
        angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)
    else:
        # Beam is stationary relative to rotation in TTLAT mode
        angles = np.zeros(num_projections)
        
    rayXY = np.zeros((2, num_projections))
    rayXY[0, :] = radius * np.cos(angles)
    rayXY[1, :] = radius * np.sin(angles)

    for j in range(num_projections):
        current_angle = j * step_deg
        base_angle_cw = -current_angle
        rad_angle = np.radians(current_angle)
        
        # Calculate deterministic kinematic offsets
        kx = kinematics['x'][1] * np.cos(rad_angle) if kinematics['x'][0] else 0.0
        ky = -kinematics['y'][1] * np.sin(rad_angle) if kinematics['y'][0] else 0.0
        kz = -kinematics['z'][1] * (current_angle / 360.0) if kinematics['z'][0] else 0.0

        # Calculate randomized motion noise
        nx = random.uniform(-noise['Translate X'][1], noise['Translate X'][1]) if noise['Translate X'][0] else 0.0
        ny = random.uniform(-noise['Translate Y'][1], noise['Translate Y'][1]) if noise['Translate Y'][0] else 0.0
        nz = random.uniform(-noise['Translate Z'][1], noise['Translate Z'][1]) if noise['Translate Z'][0] else 0.0
        
        nrx = random.uniform(-noise['Rotation X'][1], noise['Rotation X'][1]) if noise['Rotation X'][0] else 0.0
        nry = random.uniform(-noise['Rotation Y'][1], noise['Rotation Y'][1]) if noise['Rotation Y'][0] else 0.0
        nrz = random.uniform(-noise['Rotation Z'][1], noise['Rotation Z'][1]) if noise['Rotation Z'][0] else 0.0

        if is_galvo:
            # LAT Mode: Beam rotates, phantom stays mostly stationary (except noise)
            beam_rot = base_angle_cw
            p_tx, p_ty, p_tz = kx + nx, ky + ny, kz + nz
            p_rx, p_ry, p_rz = nrx, nry, nrz
            sim_theta = 0.0
        else:
            # TTLAT Mode: Beam stationary, phantom rotates around Z
            beam_rot = 0.0
            p_tx, p_ty, p_tz = kx + nx, ky + ny, kz + nz
            p_rx, p_ry, p_rz = nrx, nry, base_angle_cw + nrz
            sim_theta = base_angle_cw

        sequence.append({
            'beam_illumination_angle': illumination_angle,
            'beam_rotation': beam_rot,
            'phantom_tx': p_tx,
            'phantom_ty': p_ty,
            'phantom_tz': p_tz,
            'phantom_rx': p_rx,
            'phantom_ry': p_ry,
            'phantom_rz': p_rz,
            'sim_theta_z': sim_theta
        })
        
    return sequence, rayXY

class MeasurementWorker(QThread):
    """
    Worker for emulating the hardware measurement process.
    Iterates through the motion sequence and emits signals for UI updates.
    """
    update_beam_signal = Signal(float, float)
    update_phantom_signal = Signal(float, float, float, float, float, float)
    update_images_signal = Signal(np.ndarray, np.ndarray)
    
    def __init__(self, motion_sequence, sino_amp, sino_ph, parent=None):
        super().__init__(parent)
        self.motion_sequence = motion_sequence
        self.sino_amp = sino_amp
        self.sino_ph = sino_ph
        self.is_running = True

    def run(self):
        """Standard playback loop."""
        for j, step_data in enumerate(self.motion_sequence):
            if not self.is_running:
                break
                
            # Update the 3D viewer
            self.update_beam_signal.emit(step_data['beam_illumination_angle'], step_data['beam_rotation'])
            self.update_phantom_signal.emit(
                step_data['phantom_tx'], step_data['phantom_ty'], step_data['phantom_tz'],
                step_data['phantom_rx'], step_data['phantom_ry'], step_data['phantom_rz']
            )
            # Update the 2D projection displays
            if self.sino_amp is not None and self.sino_ph is not None:
                self.update_images_signal.emit(self.sino_amp[:, :, j], self.sino_ph[:, :, j])
                
            self.msleep(50) # frame rate
            
    def stop(self):
        """Safely stops the playback loop."""
        self.is_running = False