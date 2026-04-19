"""
CSV Export Utilities
--------------------
Handles the export of mechanical motion trajectories to CSV files.
"""

import csv

def save_motion_sequence_csv(filepath: str, motion_sequence: list):
    """
    Saves the mechanical trajectory of the phantom to a CSV file.
    Only translation and rotation parameters are logged.
    
    Args:
        filepath: Target CSV file path.
        motion_sequence: List of dictionaries containing frame data.
    """
    if not motion_sequence:
        return

    header = ['Frame', 'phantom_tx', 'phantom_ty', 'phantom_tz', 
              'phantom_rx', 'phantom_ry', 'phantom_rz']

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for i, step in enumerate(motion_sequence):
            writer.writerow([
                i,
                step.get('phantom_tx', 0.0),
                step.get('phantom_ty', 0.0),
                step.get('phantom_tz', 0.0),
                step.get('phantom_rx', 0.0),
                step.get('phantom_ry', 0.0),
                step.get('phantom_rz', 0.0)
            ])