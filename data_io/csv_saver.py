import csv

def save_motion_sequence_csv(filepath, motion_sequence):
    """
    Saves the object motion parameters (translations and rotations) 
    from the motion sequence to a CSV file.
    """
    if not motion_sequence:
        return

    # Define the header based on phantom keys
    header = ['Frame', 'phantom_tx', 'phantom_ty', 'phantom_tz', 
              'phantom_rx', 'phantom_ry', 'phantom_rz']

    try:
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
    except Exception as e:
        raise e