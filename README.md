# Holographic Tomograph Virtual Laboratory

A standalone Python-based virtual laboratory designed to simulate holographic tomography measurements. This application acts as a forward model, taking a 3D refractive index phantom and generating realistic, synthetic sinograms while simulating various mechanical imperfections and scanning geometries.

## Core Features

The simulator offers two distinct measurement modes tailored for advanced research in optical tomography:

*   **Limited Angle Tomography (LAT):** Simulates a measurement setup where the illumination angle varies (scanned by a galvanometer mirror) while the object remains stationary in the Z-axis orientation.
*   **TomoTweezers Limited Angle Tomography (TTLAT):** Simulates the TomoTweezers method, where a tilted beam remains constant relative to the object as it is rotated around the Z-axis with optical traps.

**Shared Simulation Capabilities:**
*   **Mechanical Imperfection Simulation:**
    *   **Constant Kinematics:** Apply deterministic structural wobbles, such as X/Y radius deviations and Z-axis fall range during the scan.
    *   **Motion Noise:** Inject randomized translational and rotational noise to simulate real-world mechanical jitter and test reconstruction robustness.
*   **Real-time Monitoring:** An interactive suite with a 3D visualization window allowing you to watch the laser beam scan through the object in real-time, coupled with 2D live previews of the amplitude and phase projections.
*   **Data Export:**
    *   **MAT Export:** Saves the generated `SINOamp` and `SINOph` arrays alongside precise metadata to a MATLAB-ready `.mat` file, fully prepared for immediate reconstruction.
    *   **CSV Export:** Logs the exact frame-by-frame mechanical tracking data to a `.csv` file for ground-truth comparison.

## Frameworks & Optimizations

The application is built with a focus on performance and usability:
*   **GUI Framework:** Developed as a desktop application using **PySide6**, providing a responsive and professional laboratory interface.
*   **Multithreaded Architecture:** Utilizes separate worker threads for data loading, simulation, and hardware-playback emulation to keep the GUI fluid during heavy computations.
*   **Computational Engine:** 
    *   The core logic is based on the **Rytov approximation** and rigorous **Ewald sphere** extraction in 3D Fourier space.
    *   High-performance processing using **CPU multiprocessing** to bypass the Python GIL and parallelize the complex K-space slicing.

## Requirements

The project dependencies are listed in `requirements.txt`:
*   `numpy>=1.26.4`
*   `scipy>=1.13.0`
*   `PySide6>=6.9.0`
*   `pyqtgraph>=0.13.7`
*   `scikit-image>=0.22.0`
*   `h5py>=3.13.0`

Install them via:
```bash
pip install -r requirements.txt
```

## How to Use

1.  **Launch:** Run the main application script `genSino_TT.py`.
2.  **Load Phantom:** Click `Load phantom` to select your 3D RI object. Example objects and phantoms are provided in the **`data`** folder.
3.  **Configure Scan:** 
    *   Select between LAT or TTLAT scanning modes.
    *   Adjust the rotation step, beam illumination angle, and computational accuracy (Linear/Cubic).
    *   Input desired kinematics or noise parameters to match your target laboratory conditions.
4.  **Generate Data:** Click `Start Measurement`. The application will compute the 3D FFT and generate the sinogram projections in parallel.
5.  **Review and Save:** Use the `Replay` button to review the scan results frame-by-frame. Once satisfied, save your synthetic sinogram (`.mat`) and motion trajectory (`.csv`) for further processing.
