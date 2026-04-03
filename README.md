# Holographic Tomograph Virtual Laboratory

A standalone Python-based virtual laboratory designed to simulate Optical Diffraction Tomography (ODT) measurements. This application acts as a forward model, taking a 3D refractive index phantom and generating realistic, synthetic sinograms while simulating various mechanical imperfections during the scanning process.

## Core Features

* **Interactive GUI:** Built with PySide6, featuring an interactive 3D visualization of the setup (beam and object orientation) alongside real-time 2D diagnostic displays for Amplitude and Phase projections.
* **Dual Scanning Modes:**
  * **Galvo Scan:** Simulates illumination angle variations with a stationary object.
  * **Object Rotation Scan:** Simulates sample rotation explicitly around the Z-axis.
* **Mechanical Imperfection Simulation:** * **Constant Kinematics:** Apply deterministic structural wobbles, such as X/Y radius deviations and Z-axis fall range during the scan.
  * **Motion Noise:** Inject randomized translational and rotational noise to test the robustness of external reconstruction algorithms.
* **Accurate Forward Modeling:** * Implements rigorous Ewald sphere extraction in 3D K-space using `scipy.fft` and `scipy.ndimage.map_coordinates`.
  * Dynamic toggle between Linear and Cubic interpolation orders.
  * Parallelized projection generation using CPU multiprocessing to bypass the GIL and speed up heavy K-space slicing.
* **Data Export:**
  * **MAT Export:** Saves the generated `SINOamp` and `SINOph` arrays alongside precise metadata (including automatically extracted background refractive index) to a MATLAB `.mat` file.
  * **CSV Export:** Logs the exact frame-by-frame mechanical tracking data (translations and rotations) to a `.csv` file for ground-truth comparison during reconstruction.

## Requirements

To run this simulator, you need Python 3 and the following packages:
* `numpy`
* `scipy`
* `PySide6`

## How to Use

1. **Launch:** Run the main application script to open the graphical interface.
2. **Load Phantom:** Click `Load phantom` to select your baseline 3D object in `.mat` format.
3. **Set Parameters:** * Choose the scan mode, rotation step, and beam azimuth.
   * Toggle the required computational accuracy.
   * (Optional) Enable and tune kinematics and random motion noise.
4. **Simulate:** Click `Start Measurement`. The application will compute the 3D FFT and slice the Ewald spheres in parallel.
5. **Review & Save:** Once complete, use `Replay Measurement` to review the generated frames, and use the save buttons to export your synthetic sinogram and motion log.

## Note on Repository Limits
Large `.mat` phantoms, generated sinograms, and `.csv` logs are excluded from this repository via `.gitignore` to comply with standard file size limits.
