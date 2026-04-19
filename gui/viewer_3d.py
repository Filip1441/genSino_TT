"""
3D Visualization Suite
----------------------
Provides OpenGL-based 3D visualization of the holographic tomography setup,
including the laser beam and the refractive index phantom.
"""

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt

class LimitedGLViewWidget(gl.GLViewWidget):
    """
    Extensions to pyqtgraph's GLViewWidget that restricts 
    camera zoom distances for better user experience.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_dist = 100
        self.max_dist = 2500

    def wheelEvent(self, ev):
        """Restrict zoom via mouse wheel."""
        super().wheelEvent(ev)
        self.clamp_distance()

    def mouseMoveEvent(self, ev):
        """Restrict zoom via right-click drag."""
        super().mouseMoveEvent(ev)
        if ev.buttons() == Qt.MouseButton.RightButton:
            self.clamp_distance()

    def clamp_distance(self):
        """Forces the camera distance within [min_dist, max_dist] range."""
        if self.opts['distance'] > self.max_dist:
            self.opts['distance'] = self.max_dist
        elif self.opts['distance'] < self.min_dist:
            self.opts['distance'] = self.min_dist
        self.update()

class Viewer3D(QWidget):
    """
    A widget that combines a GLViewWidget with logic to manage 
    3D elements (beam mesh, volume phantom) and their transformations.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Initialize the OpenGL view
        self.gl_view = LimitedGLViewWidget()
        self.gl_view.opts['distance'] = 1000 
        self.gl_view.setBackgroundColor('k') # Black background
        self.layout.addWidget(self.gl_view)
        
        # Ground reference grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=512, y=512)
        self.grid.setSpacing(x=64, y=64) 
        self.gl_view.addItem(self.grid)
        
        self.volume_item = None
        self.original_shape = (0, 0, 0)
        
        # Initialize laser beam
        self.create_beam()

    def create_beam(self):
        """Creates a semi-transparent red cylinder representing the laser beam."""
        mesh_data = gl.MeshData.cylinder(rows=50, cols=50, radius=[150, 150], length=20000)
        
        # Color initialization: Red with 15% opacity
        colors = np.ones((mesh_data.faceCount(), 4), dtype=float)
        colors[:, 0] = 1.0 # Red
        colors[:, 1] = 0.0 # Green
        colors[:, 2] = 0.0 # Blue
        colors[:, 3] = 0.15 # Alpha
        
        self.beam_item = gl.GLMeshItem(
            meshdata=mesh_data, 
            smooth=False, 
            shader=None,
            glOptions='additive' # Additive blending for 'glow' effect
        )
        self.beam_item.setMeshData(vertexes=mesh_data.vertexes(), faces=mesh_data.faces(), faceColors=colors)
        self.gl_view.addItem(self.beam_item)
        
        # Set default position
        self.update_beam_transform(illumination_angle_deg=0.0, rotation_deg=0.0)

    def update_beam_transform(self, illumination_angle_deg: float, rotation_deg: float):
        """Updates the beam mesh orientation based on illumination and rotation angles."""
        self.beam_item.resetTransform()
        self.beam_item.translate(0, 0, -10000) # Center the long cylinder
        self.beam_item.rotate(illumination_angle_deg, 0, 1, 0) # Tilt (Theta)
        self.beam_item.rotate(rotation_deg, 0, 0, 1) # Rotation around Z (Galvo)

    def set_phantom_data(self, volume_data: np.ndarray, rgba_data: np.ndarray):
        """
        Initializes or updates the GLVolumeItem with new phantom data.
        
        Args:
            volume_data: Original volume (to extract shape).
            rgba_data: Preprocessed RGBA ubyte data for the volume renderer.
        """
        if volume_data.ndim != 3:
            return
            
        self.original_shape = volume_data.shape
        rgba = rgba_data
        
        if self.volume_item is None:
            self.volume_item = gl.GLVolumeItem(rgba)
            self.gl_view.addItem(self.volume_item)
        else:
            self.volume_item.setData(rgba)
            
        self.update_phantom_transform(0, 0, 0, 0, 0, 0)

    def update_phantom_transform(self, tx: float, ty: float, tz: float, rx: float, ry: float, rz: float):
        """Updates the 3D volume item with new translation and rotation values."""
        if self.volume_item is None:
            return
            
        self.volume_item.resetTransform()
        # Voxel-to-world scaling (consistent with Grid spacing)
        self.volume_item.scale(2, 2, 2)
        # Center the volume around (0,0,0)
        self.volume_item.translate(-self.original_shape[0]/2, -self.original_shape[1]/2, -self.original_shape[2]/2)
        
        # Apply rotations
        if rx != 0:
            self.volume_item.rotate(rx, 1, 0, 0)
        if ry != 0:
            self.volume_item.rotate(ry, 0, 1, 0)
        if rz != 0:
            self.volume_item.rotate(rz, 0, 0, 1)
        
        # Apply final translation
        self.volume_item.translate(tx, ty, tz)