"""
Main Application Window
-----------------------
The central hub of the GenSino-TT application. Manages the user interface,
input parameters, background worker orchestration, and data visualization.
"""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QRadioButton, QGroupBox, QLabel,
    QDoubleSpinBox, QCheckBox, QFormLayout, QGridLayout,
    QButtonGroup, QSizePolicy, QFileDialog, QMessageBox,
    QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QIcon

from gui.viewer_3d import Viewer3D
from data_io.mat_saver import save_sinogram_mat
from data_io.csv_saver import save_motion_sequence_csv
from core.measurement_worker import MeasurementWorker, generate_motion_sequence
from core.simulation_worker import SimulationWorker
from core.data_worker import DataLoaderWorker, DataSaverWorker
from core.utils import normalize_to_uint8

class MainWindow(QMainWindow):
    """
    GenSino-TT Main Window.
    Handles UI layout, parameter management, and threading for the HT simulator.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenSino-TT")
        self.resize(1280, 720)
        self.setWindowIcon(QIcon("assets/icon.png"))

        # State management variables
        self.phantom_data = None
        self.worker = None
        self.sim_worker = None
        self.motion_sequence = None
        
        # Result caching
        self.current_sino_amp = None
        self.current_sino_ph = None
        self.current_rayXY = None
        self.current_metadata = {}

        # --- Base Layout Construction ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel: Visualization (3D View + Projections)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 3D Setup Viewer
        self.group_3d = QGroupBox("3D Visualization")
        layout_3d = QVBoxLayout(self.group_3d)
        self.viewer_3d = Viewer3D()
        layout_3d.addWidget(self.viewer_3d)
        left_layout.addWidget(self.group_3d, stretch=2)

        # 2D Projections (Amplitude and Phase)
        bottom_row_widget = QWidget()
        bottom_row_layout = QHBoxLayout(bottom_row_widget)
        bottom_row_layout.setContentsMargins(0, 0, 0, 0)

        self.group_amplitude = QGroupBox("Amplitude Image")
        amplitude_layout = QVBoxLayout(self.group_amplitude)
        self.img_amplitude = QLabel("")
        self.img_amplitude.setStyleSheet("background-color: #050505; border: 1px solid #333;")
        self.img_amplitude.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_amplitude.setFixedSize(256, 256)
        amplitude_layout.addWidget(self.img_amplitude, alignment=Qt.AlignmentFlag.AlignCenter)

        self.group_phase = QGroupBox("Phase Image")
        phase_layout = QVBoxLayout(self.group_phase)
        self.img_phase = QLabel("")
        self.img_phase.setStyleSheet("background-color: #050505; border: 1px solid #333;")
        self.img_phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_phase.setFixedSize(256, 256)
        phase_layout.addWidget(self.img_phase, alignment=Qt.AlignmentFlag.AlignCenter)
        
        bottom_row_layout.addWidget(self.group_amplitude)
        bottom_row_layout.addWidget(self.group_phase)
        left_layout.addWidget(bottom_row_widget, stretch=1)

        main_layout.addWidget(left_panel, stretch=1)

        # Right Panel: Control Panel (Scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(420)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        
        right_panel_content = QWidget()
        right_layout = QVBoxLayout(right_panel_content)

        # File Management
        self.btn_load_phantom = QPushButton("Load phantom")
        self.btn_load_phantom.setMinimumHeight(45)
        right_layout.addWidget(self.btn_load_phantom)

        # Scan Mode Configuration
        self.group_scan_mode = QGroupBox("Scan Mode")
        scan_mode_layout = QVBoxLayout()
        self.radio_galvo = QRadioButton("Galvo scan")
        self.radio_galvo.setChecked(True) 
        self.radio_obj_rot = QRadioButton("Object rotation scan")
        self.scan_btn_group = QButtonGroup()
        self.scan_btn_group.addButton(self.radio_galvo)
        self.scan_btn_group.addButton(self.radio_obj_rot)
        scan_mode_layout.addWidget(self.radio_galvo)
        scan_mode_layout.addWidget(self.radio_obj_rot)
        self.group_scan_mode.setLayout(scan_mode_layout)
        right_layout.addWidget(self.group_scan_mode)

        # Optical Settings
        self.group_settings = QGroupBox("Settings")
        settings_layout = QFormLayout()
        self.spin_illumination_angle = QDoubleSpinBox()
        self.spin_illumination_angle.setRange(0.0, 90.0)
        self.spin_illumination_angle.setValue(45.0)
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.01, 360.0)
        self.spin_step.setValue(10.0)
        settings_layout.addRow("Beam illumination angle (deg):", self.spin_illumination_angle)
        settings_layout.addRow("Rotation step (deg):", self.spin_step)
        self.group_settings.setLayout(settings_layout)
        right_layout.addWidget(self.group_settings)

        # Accuracy Toggles
        self.group_accuracy = QGroupBox("Computational Accuracy")
        accuracy_layout = QVBoxLayout()
        self.radio_interp_1 = QRadioButton("Linear")
        self.radio_interp_3 = QRadioButton("Cubic")
        self.radio_interp_1.setChecked(True)
        self.interp_btn_group = QButtonGroup()
        self.interp_btn_group.addButton(self.radio_interp_1)
        self.interp_btn_group.addButton(self.radio_interp_3)
        accuracy_layout.addWidget(self.radio_interp_1)
        accuracy_layout.addWidget(self.radio_interp_3)
        self.group_accuracy.setLayout(accuracy_layout)
        right_layout.addWidget(self.group_accuracy)

        # Kinematics Controls (Mechanical Wobble)
        self.group_kinematics = QGroupBox("Constant Kinematics")
        kinematics_layout = QGridLayout()
        self.check_k_x = QCheckBox("X radius")
        self.spin_k_x = QDoubleSpinBox(); self.spin_k_x.setRange(0, 50); self.spin_k_x.setValue(5); self.spin_k_x.setEnabled(False)
        self.check_k_x.toggled.connect(self.spin_k_x.setEnabled)
        self.check_k_y = QCheckBox("Y radius")
        self.spin_k_y = QDoubleSpinBox(); self.spin_k_y.setRange(0, 50); self.spin_k_y.setValue(5); self.spin_k_y.setEnabled(False)
        self.check_k_y.toggled.connect(self.spin_k_y.setEnabled)
        self.check_k_z = QCheckBox("Z fall range")
        self.spin_k_z = QDoubleSpinBox(); self.spin_k_z.setRange(0, 50); self.spin_k_z.setValue(10); self.spin_k_z.setEnabled(False)
        self.check_k_z.toggled.connect(self.spin_k_z.setEnabled)
        kinematics_layout.addWidget(self.check_k_x, 0, 0); kinematics_layout.addWidget(self.spin_k_x, 0, 1)
        kinematics_layout.addWidget(self.check_k_y, 1, 0); kinematics_layout.addWidget(self.spin_k_y, 1, 1)
        kinematics_layout.addWidget(self.check_k_z, 2, 0); kinematics_layout.addWidget(self.spin_k_z, 2, 1)
        self.group_kinematics.setLayout(kinematics_layout)
        right_layout.addWidget(self.group_kinematics)

        # Motion Noise Controls (Jitter)
        self.group_noise = QGroupBox("Motion Noise")
        noise_layout = QGridLayout()
        noise_rows = [("Translate X", 0.5), ("Translate Y", 0.5), ("Translate Z", 0.5),
                      ("Rotation X", 0.25), ("Rotation Y", 0.25), ("Rotation Z", 0.25)]
        self.noise_controls = {}
        for row_idx, (name, def_val) in enumerate(noise_rows):
            cb = QCheckBox(name)
            spin = QDoubleSpinBox(); spin.setRange(0, 5); spin.setValue(def_val); spin.setEnabled(False)
            cb.toggled.connect(spin.setEnabled)
            noise_layout.addWidget(cb, row_idx, 0); noise_layout.addWidget(spin, row_idx, 1)
            self.noise_controls[name] = {"checkbox": cb, "value": spin}
        self.group_noise.setLayout(noise_layout)
        right_layout.addWidget(self.group_noise)
        
        # Simulation Action Buttons
        self.btn_start_measurement = QPushButton("Start Measurement")
        self.btn_start_measurement.setMinimumHeight(45)
        self.btn_start_measurement.setEnabled(False)
        right_layout.addWidget(self.btn_start_measurement)

        # Progress Messaging
        progress_container = QWidget()
        progress_layout = QHBoxLayout(progress_container)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(True)
        progress_layout.addWidget(self.progress_bar)
        right_layout.addWidget(progress_container)
        
        # Post-Processing Buttons
        self.btn_replay_measurement = QPushButton("Replay Measurement")
        self.btn_replay_measurement.setMinimumHeight(45)
        self.btn_replay_measurement.setEnabled(False)
        right_layout.addWidget(self.btn_replay_measurement)

        self.btn_save_sino = QPushButton("Save Sinogram")
        self.btn_save_sino.setMinimumHeight(45)
        self.btn_save_sino.setEnabled(False)
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_save_csv.setMinimumHeight(45)
        self.btn_save_csv.setEnabled(False)
        right_layout.addWidget(self.btn_save_sino)
        right_layout.addWidget(self.btn_save_csv)

        right_layout.addStretch()
        self.scroll_area.setWidget(right_panel_content)
        main_layout.addWidget(self.scroll_area)

        # --- Signal Connections ---
        self.btn_load_phantom.clicked.connect(self.action_load_phantom)
        self.spin_illumination_angle.valueChanged.connect(self.action_update_illumination_angle)
        self.btn_start_measurement.clicked.connect(self.action_toggle_measurement)
        self.btn_replay_measurement.clicked.connect(self.action_replay_measurement)
        self.btn_save_sino.clicked.connect(self.action_save_sinogram)
        self.btn_save_csv.clicked.connect(self.action_save_csv)
        
        # Final initialization
        self.action_update_illumination_angle(self.spin_illumination_angle.value())

    # --- Action Handlers ---

    def action_load_phantom(self):
        """Triggers the background loading of a .mat phantom file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Phantom", "data", "MAT Files (*.mat)")
        if filepath:
            self.btn_load_phantom.setText("Loading...")
            self.btn_load_phantom.setEnabled(False)
            self.set_ui_enabled(False)
            
            # Use background worker to avoid GUI freeze
            self.data_worker = DataLoaderWorker(filepath)
            self.data_worker.finished_signal.connect(self.on_phantom_loaded)
            self.data_worker.error_signal.connect(self.on_data_error)
            self.data_worker.start()

    def on_phantom_loaded(self, volume_data, rgba_data):
        """Callback when the phantom data is ready."""
        self.phantom_data = volume_data
        self.viewer_3d.set_phantom_data(volume_data, rgba_data)
        
        self.btn_load_phantom.setText("Load phantom")
        self.btn_load_phantom.setEnabled(True)
        self.set_ui_enabled(True)

    def on_data_error(self, message):
        """Standard error display handler."""
        QMessageBox.critical(self, "Error", message)
        self.btn_load_phantom.setText("Load phantom")
        self.btn_load_phantom.setEnabled(True)
        self.set_ui_enabled(True)

    def action_update_illumination_angle(self, value: float):
        """Updates the 3D laser beam orientation based on slider value."""
        self.viewer_3d.update_beam_transform(illumination_angle_deg=value, rotation_deg=0.0)
        
    def action_toggle_measurement(self):
        """Starts or cancels the HT simulation."""
        if self.sim_worker and self.sim_worker.isRunning():
            self.sim_worker.stop()
            self.btn_start_measurement.setText("Canceling...")
            self.btn_start_measurement.setEnabled(False)
        else:
            self.action_start_measurement()

    def action_start_measurement(self):
        """Prepares parameters and launches the parallel simulation worker."""
        self.set_ui_enabled(False)
        self.btn_start_measurement.setText("Cancel Calculation")
        self.btn_start_measurement.setEnabled(True)
        self.progress_bar.setValue(0)

        # Collate motion parameters
        kinematics = {
            'x': (self.check_k_x.isChecked(), self.spin_k_x.value()),
            'y': (self.check_k_y.isChecked(), self.spin_k_y.value()),
            'z': (self.check_k_z.isChecked(), self.spin_k_z.value())
        }
        noise = {name: (ctrl['checkbox'].isChecked(), ctrl['value'].value()) 
                 for name, ctrl in self.noise_controls.items()}

        is_galvo = self.radio_galvo.isChecked()
        illumination_angle = self.spin_illumination_angle.value()
        step = self.spin_step.value()
        num_projs = int(360 / step)
        interp_order = 1 if self.radio_interp_1.isChecked() else 3

        # Generate trajectory
        self.motion_sequence, rayXY = generate_motion_sequence(num_projs, step, is_galvo, illumination_angle, kinematics, noise)
        self.current_rayXY = rayXY

        # Start simulation worker
        self.sim_worker = SimulationWorker(
            self.phantom_data, rayXY, self.motion_sequence, interp_order=interp_order
        )
        self.sim_worker.progress_updated.connect(self.progress_bar.setValue)
        self.sim_worker.error_signal.connect(self.on_data_error)
        self.sim_worker.finished_signal.connect(self.action_start_playback)
        self.sim_worker.finished.connect(self.on_sim_worker_finished)
        self.sim_worker.start()

    def on_sim_worker_finished(self):
        """Cleanup after simulation finishes or is canceled."""
        if not self.sim_worker.is_running:
            self.action_measurement_finished()

    def action_replay_measurement(self):
        """Restarts the playback visualization of previously computed results."""
        self.set_ui_enabled(False)
        self.action_start_playback(self.current_sino_amp, self.current_sino_ph, self.current_metadata)

    def action_start_playback(self, sino_amp: np.ndarray, sino_ph: np.ndarray, metadata: dict):
        """Launches the measurement worker to drive the playback visuals."""
        self.btn_start_measurement.setText("Playing measurement...")
        self.btn_start_measurement.setEnabled(False)
        
        self.current_sino_amp = sino_amp
        self.current_sino_ph = sino_ph
        self.current_metadata = metadata

        # Start playback worker
        self.worker = MeasurementWorker(self.motion_sequence, sino_amp, sino_ph)
        self.worker.update_beam_signal.connect(self.viewer_3d.update_beam_transform)
        self.worker.update_phantom_signal.connect(self.viewer_3d.update_phantom_transform)
        self.worker.update_images_signal.connect(self.update_live_images)
        self.worker.finished.connect(self.action_measurement_finished)
        self.worker.start()
        
    def action_measurement_finished(self):
        """Restores the UI to ready state after playback or simulation."""
        self.set_ui_enabled(True)
        self.btn_start_measurement.setText("Start Measurement")
        self.action_update_illumination_angle(self.spin_illumination_angle.value())
        self.viewer_3d.update_phantom_transform(0, 0, 0, 0, 0, 0)
        
    def action_save_sinogram(self):
        """Background save of sinogram data to .mat file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Sinogram", "data/synthetic_sinogram.mat", "MAT Files (*.mat)"
        )
        if filepath:
            self.reset_button_state(self.btn_save_sino, "Saving...", False)
            self.save_worker = DataSaverWorker(
                save_sinogram_mat, filepath, 
                self.current_sino_amp, self.current_sino_ph, self.current_rayXY, self.current_metadata
            )
            self.save_worker.finished_signal.connect(lambda: self.reset_button_state(self.btn_save_sino, "Save Sinogram", True))
            self.save_worker.error_signal.connect(lambda msg: self.on_save_error(msg, self.btn_save_sino, "Save Sinogram"))
            self.save_worker.start()

    def action_save_csv(self):
        """Background save of mechanical trajectory to .csv file."""
        filepath, _ = QFileDialog.getSaveFileName(self, "Save CSV Log", "data/motion_log.csv", "CSV Files (*.csv)")
        if filepath:
            self.reset_button_state(self.btn_save_csv, "Saving...", False)
            self.save_worker = DataSaverWorker(save_motion_sequence_csv, filepath, self.motion_sequence)
            self.save_worker.finished_signal.connect(lambda: self.reset_button_state(self.btn_save_csv, "Save CSV", True))
            self.save_worker.error_signal.connect(lambda msg: self.on_save_error(msg, self.btn_save_csv, "Save CSV"))
            self.save_worker.start()

    def reset_button_state(self, button, text, enabled):
        """Helper to update button properties."""
        button.setText(text)
        button.setEnabled(enabled)

    def on_save_error(self, message, button, original_text):
        """Helper to handle export errors."""
        self.reset_button_state(button, original_text, True)
        QMessageBox.critical(self, "Error", f"Failed to save file:\n{message}")
        
    def set_ui_enabled(self, state: bool):
        """Updates the enabled state of UI groups based on application context."""
        has_phantom = self.phantom_data is not None
        self.btn_load_phantom.setEnabled(state)
        self.group_scan_mode.setEnabled(state)
        self.group_settings.setEnabled(state)
        self.group_accuracy.setEnabled(state)
        self.group_kinematics.setEnabled(state)
        self.group_noise.setEnabled(state)
        self.btn_start_measurement.setEnabled(state and has_phantom)
        
        has_data = self.current_sino_amp is not None
        self.btn_replay_measurement.setEnabled(state and has_data)
        self.btn_save_sino.setEnabled(state and has_data)
        self.btn_save_csv.setEnabled(state and self.motion_sequence is not None)

    def update_live_images(self, amplitude_data: np.ndarray, phase_data: np.ndarray):
        """Refreshes the 2D amplitude and phase labels with new frame data."""
        amp_pixmap = self._convert_numpy_to_pixmap(amplitude_data)
        ph_pixmap = self._convert_numpy_to_pixmap(phase_data)
        fixed_size = self.img_amplitude.size()
        self.img_amplitude.setPixmap(amp_pixmap.scaled(
            fixed_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        self.img_phase.setPixmap(ph_pixmap.scaled(
            fixed_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def _convert_numpy_to_pixmap(self, arr: np.ndarray) -> QPixmap:
        """Utility for converting raw numpy arrays to Qt Pixmaps for display."""
        data = normalize_to_uint8(arr)
        data = np.ascontiguousarray(data)
        height, width = data.shape
        qimg = QImage(data.data, width, height, width, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def closeEvent(self, event):
        """Ensures all background threads are stopped and joined before closing the app."""
        workers = [
            getattr(self, 'sim_worker', None),
            getattr(self, 'worker', None),
            getattr(self, 'data_worker', None),
            getattr(self, 'save_worker', None)
        ]
        
        for w in workers:
            if w and w.isRunning():
                if hasattr(w, 'stop'):
                    w.stop()
                elif hasattr(w, 'is_running'):
                    w.is_running = False
                w.wait() # Block main thread until worker finishes safely
                
        event.accept()