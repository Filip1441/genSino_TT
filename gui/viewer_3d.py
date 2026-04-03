import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QWidget, QVBoxLayout

class Viewer3D(QWidget):
    """
    Volumetric 3D Viewer for Holographic Tomography.
    Handles the phantom volume rendering and the illumination beam geometry.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Initialize the OpenGL View Widget
        self.gl_view = gl.GLViewWidget()
        # Set camera further away to comfortably fit the 512x512x512 volume
        self.gl_view.opts['distance'] = 1000 
        self.gl_view.setBackgroundColor('k') # Black background
        self.layout.addWidget(self.gl_view)
        
        # Add a coordinate grid matched to 512x512 size
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=512, y=512)
        # Spacing of 64 creates exactly 8 blocks across the 512 length
        self.grid.setSpacing(x=64, y=64) 
        self.gl_view.addItem(self.grid)
        
        self.volume_item = None
        self.original_shape = (0, 0, 0)
        
        # Create and display the illumination beam
        self.create_beam()

    def create_beam(self):
        """
        Creates a transparent cylinder representing the illumination beam.
        Uses additive blending to simulate light without occluding the phantom.
        """
        # Create cylinder mesh: radius 60, length 800
        mesh_data = gl.MeshData.cylinder(rows=50, cols=50, radius=[150, 150], length=1600)
        
        # Define color: visible red
        colors = np.ones((mesh_data.faceCount(), 4), dtype=float)
        colors[:, 0] = 1.0  # Red
        colors[:, 1] = 0.0  # Green
        colors[:, 2] = 0.0  # Blue
        colors[:, 3] = 0.15 # Alpha (transparency level)
        
        self.beam_item = gl.GLMeshItem(
            meshdata=mesh_data, 
            smooth=False, 
            shader=None,           # Disable shader to prevent opaque surface calculations
            glOptions='additive'   # Additive blending prevents Z-fighting and object deletion
        )
        self.beam_item.setMeshData(vertexes=mesh_data.vertexes(), faces=mesh_data.faces(), faceColors=colors)
        self.gl_view.addItem(self.beam_item)
        
        # Initialize default position
        self.update_beam_transform(illumination_angle_deg=0.0, rotation_deg=0.0)

    def update_beam_transform(self, illumination_angle_deg: float, rotation_deg: float):
        """
        Updates the beam spatial position based on illumination_angle and galvo scan rotation.
        """
        self.beam_item.resetTransform()
        
        # 1. Center the cylinder so its middle is at (0,0,0) (half of 800 is 400)
        self.beam_item.translate(0, 0, -800)
        
        # 2. Apply illumination_angle tilt (tilting towards X means rotating around Y)
        self.beam_item.rotate(illumination_angle_deg, 0, 1, 0)
        
        # 3. Apply scan rotation around Z axis (creates a cone if illumination_angle > 0)
        self.beam_item.rotate(rotation_deg, 0, 0, 1)

    def set_phantom_data(self, volume_data: np.ndarray):
        """
        Loads the phantom volume data and displays it.
        Uses visual downsampling for performance while maintaining spatial dimensions.
        """
        if volume_data.ndim != 3:
            return
            
        self.original_shape = volume_data.shape
        
        # Downsample for visualization to prevent UI freezing
        display_data = volume_data[::2, ::2, ::2]
        display_shape = display_data.shape
        bg_value = display_data[0, 0, 0]
        
        rgba = np.zeros(display_shape + (4,), dtype=np.ubyte)
        tolerance = 1e-5
        phantom_mask = np.abs(display_data - bg_value) > tolerance
        
        if np.any(phantom_mask):
            phantom_values = display_data[phantom_mask]
            min_val = phantom_values.min()
            max_val = phantom_values.max()
            
            if max_val > min_val:
                normalized = (phantom_values - min_val) / (max_val - min_val) * 255.0
            else:
                normalized = np.full(phantom_values.shape, 255.0)
                
            norm_byte = normalized.astype(np.ubyte)
            
            rgba[phantom_mask, 0] = norm_byte  # R
            rgba[phantom_mask, 1] = norm_byte  # G
            rgba[phantom_mask, 2] = norm_byte  # B
            rgba[phantom_mask, 3] = 50         # Alpha (semi-transparent)
            
        if self.volume_item is None:
            self.volume_item = gl.GLVolumeItem(rgba)
            self.gl_view.addItem(self.volume_item)
        else:
            self.volume_item.setData(rgba)
            
        # Reset position to center after loading
        self.update_phantom_transform(0, 0, 0, 0, 0, 0)

    def update_phantom_transform(self, tx: float, ty: float, tz: float, rx: float, ry: float, rz: float):
        """
        Updates the phantom spatial position and rotation for all 3 axes.
        
        Args:
            tx, ty, tz: Translation vectors.
            rx, ry, rz: Rotation angles in degrees for X, Y, and Z axes.
        """
        if self.volume_item is None:
            return
            
        self.volume_item.resetTransform()
        
        # Scale due to visual downsampling
        self.volume_item.scale(2, 2, 2)
        
        # 1. Center the volume
        self.volume_item.translate(-self.original_shape[0]/2, -self.original_shape[1]/2, -self.original_shape[2]/2)
        
        # 2. Apply rotations (Order: X, then Y, then Z)
        if rx != 0:
            self.volume_item.rotate(rx, 1, 0, 0)
        if ry != 0:
            self.volume_item.rotate(ry, 0, 1, 0)
        if rz != 0:
            self.volume_item.rotate(rz, 0, 0, 1)
        
        # 3. Apply translations (Kinematics + Noise)
        self.volume_item.translate(tx, ty, tz)