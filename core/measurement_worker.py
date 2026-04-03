import numpy as np
import random
from PySide6.QtCore import QThread, Signal

def generate_motion_sequence(num_projections, step_deg, is_galvo, illumination_angle, kinematics, noise):
    """
    Generates a full timeline of transformations for the entire measurement
    and calculates the K-space trajectory (rayXY) depending on the scan mode.
    """
    sequence = []
    
    # Precalculate K-space trajectory based on scan mode
    radius = np.sin(np.radians(illumination_angle))
    if is_galvo:
        angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)
    else:
        angles = np.zeros(num_projections)
        
    rayXY = np.zeros((2, num_projections))
    rayXY[0, :] = radius * np.cos(angles)
    rayXY[1, :] = radius * np.sin(angles)

    for j in range(num_projections):
        current_angle = j * step_deg
        base_angle_cw = -current_angle
        rad_angle = np.radians(current_angle)
        
        # 1. Kinematics
        kx = kinematics['x'][1] * np.cos(rad_angle) if kinematics['x'][0] else 0.0
        ky = -kinematics['y'][1] * np.sin(rad_angle) if kinematics['y'][0] else 0.0
        kz = -kinematics['z'][1] * (current_angle / 360.0) if kinematics['z'][0] else 0.0

        # 2. Translation Noise
        nx = random.uniform(-noise['Translate X'][1], noise['Translate X'][1]) if noise['Translate X'][0] else 0.0
        ny = random.uniform(-noise['Translate Y'][1], noise['Translate Y'][1]) if noise['Translate Y'][0] else 0.0
        nz = random.uniform(-noise['Translate Z'][1], noise['Translate Z'][1]) if noise['Translate Z'][0] else 0.0
        
        # 3. Rotation Noise
        nrx = random.uniform(-noise['Quaternion X'][1], noise['Quaternion X'][1]) if noise['Quaternion X'][0] else 0.0
        nry = random.uniform(-noise['Quaternion Y'][1], noise['Quaternion Y'][1]) if noise['Quaternion Y'][0] else 0.0
        nrz = random.uniform(-noise['Quaternion Z'][1], noise['Quaternion Z'][1]) if noise['Quaternion Z'][0] else 0.0

        # 4. Mode Logic
        if is_galvo:
            beam_rot = base_angle_cw
            p_tx, p_ty, p_tz = kx + nx, ky + ny, kz + nz
            p_rx, p_ry, p_rz = nrx, nry, nrz
            sim_theta = 0.0
        else:
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
    Playback worker. Takes precalculated data and plays it smoothly.
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
        """
        Iterates over the precalculated frames and updates all GUI elements simultaneously.
        """
        for j, step_data in enumerate(self.motion_sequence):
            if not self.is_running:
                break
                
            # Update 3D Viewer
            self.update_beam_signal.emit(step_data['beam_illumination_angle'], step_data['beam_rotation'])
            self.update_phantom_signal.emit(
                step_data['phantom_tx'], step_data['phantom_ty'], step_data['phantom_tz'],
                step_data['phantom_rx'], step_data['phantom_ry'], step_data['phantom_rz']
            )
            
            # Update 2D Live Images
            self.update_images_signal.emit(self.sino_amp[:, :, j], self.sino_ph[:, :, j])
            
            # Playback speed (approx 20 FPS)
            self.msleep(50)
            
    def stop(self):
        self.is_running = False