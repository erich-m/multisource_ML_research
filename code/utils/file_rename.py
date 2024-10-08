import os

# Define the base directory
base_dir = 'intermediary_data/extracted_encounters_5'

# Iterate over all participants (assuming participant numbers from 0 to 72)
for p_num in range(73):  # 0 to 72 inclusive
    participant_dir = os.path.join(base_dir, f'participant_{p_num}')
    
    # Check if the participant directory exists
    if not os.path.exists(participant_dir):
        print(f"Participant {p_num} directory not found.")
        continue
    
    # Iterate through all files in the participant directory
    for filename in os.listdir(participant_dir):
        # Look for files that contain 'ltsig' but not 'ltsignal'
        if 'ltsig' in filename and 'ltsignal' not in filename:
            # Construct old and new file paths
            old_path = os.path.join(participant_dir, filename)
            new_filename = filename.replace('ltsig', 'ltsignal')
            new_path = os.path.join(participant_dir, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

print("Renaming complete.")
