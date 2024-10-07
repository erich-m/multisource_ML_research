# implementation of gap fill-in interpolation algorithm, eye position averaging algorithm, moving average algorithm (for noise), and the I-VT implementation is detailed in Tobii I-VT Fixation Filter:
# Available Online: http://www.vinis.co.kr/ivt_filter.pdf

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

# TODO: Iterate over all encounters to process all data
# Set parameters
max_gap_length = float(input("Enter max gap length to interpolate (seconds) = ") or 0.075) #default 75 milliseconds
window_size = int(input("Enter the window size for the moving average (number of samples) = ") or 5)  # default window size is 5 samples for noise reduction filter
velocity_window_ms = float(input("Enter the window length for velocity calculation (milliseconds) = ") or 20)  # default 20 ms for I-VT filter

# I-VT Fixation Filter Parameters
velocity_threshold = float(input("Enter the velocity threshold for I-VT filter (degrees/second) = ") or 30)  # default threshold of 30 degrees/second
min_fixation_duration = float(input("Enter the minimum fixation duration (seconds) = ") or 0.1)  # default minimum duration of 100 ms

# get data summary
data_summary = pd.read_excel('summary_files/data_summary.xlsx')

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary)):
    p_num = summary_row["participant_id"]

    gaze_encounter_df = pd.read_csv(f'intermediary_data/extracted_encounters_5/participant_{p_num}/gaze_encounter_{p_num}.csv')
    gaze_encounter_df['Timestamp'] = gaze_encounter_df['Timestamp'].astype(float)

    columns_to_interpolate = [col for col in gaze_encounter_df.columns if col not in ['Timestamp', 'Type']]

    processed_gaze_df = gaze_encounter_df.copy()

    # Identify gaps in the selected columns by first identifying values where values are missing (a flag is set in a temporary column for each data column, called is_valid)
    processed_gaze_df['is_valid'] = processed_gaze_df[columns_to_interpolate].notna().all(axis=1)
    # Gaps are then found in the non-valid columns (columns that are missing data) by setting a flag in two temporary columns (gap_start) and (gap_end)
    processed_gaze_df['gap_start'] = (~processed_gaze_df['is_valid']) & (processed_gaze_df['is_valid'].shift(1) | (processed_gaze_df.index == 0))
    processed_gaze_df['gap_end'] = (~processed_gaze_df['is_valid']) & (processed_gaze_df['is_valid'].shift(-1) | (processed_gaze_df.index == len(processed_gaze_df) - 1))

    # Process each gap
    # TODO: Correct the interpolation to account for changes in the timestamp period (timestamp does not increase consistently)
    gap_starts = processed_gaze_df.index[processed_gaze_df['gap_start']]
    for start in gap_starts:
        end = processed_gaze_df.index[processed_gaze_df['gap_end'] & (processed_gaze_df.index >= start)][0]
        gap_duration = processed_gaze_df.loc[end, 'Timestamp'] - processed_gaze_df.loc[start, 'Timestamp']
        # Only interpolate if the gap is shorter than max_gap_length (default 0.075 seconds)
        if gap_duration <= max_gap_length:
            for col in columns_to_interpolate:
                last_valid = processed_gaze_df.loc[:start-1, col].last_valid_index()
                next_valid = processed_gaze_df.loc[end+1:, col].first_valid_index()
                
                if last_valid is not None and next_valid is not None:
                    # use np interpolation using the timestamp column 
                    interpolated_values = np.interp(
                        processed_gaze_df.loc[last_valid:next_valid, 'Timestamp'],
                        [processed_gaze_df.loc[last_valid, 'Timestamp'], processed_gaze_df.loc[next_valid, 'Timestamp']],
                        [processed_gaze_df.loc[last_valid, col], processed_gaze_df.loc[next_valid, col]]
                    )
                    
                    processed_gaze_df.loc[last_valid:next_valid, col] = interpolated_values

    # Drop helper columns
    processed_gaze_df = processed_gaze_df.drop(columns=['is_valid', 'gap_start', 'gap_end'])

    # Print summary of interpolation
    """
    original_null_count = gaze_encounter_df[columns_to_interpolate].isnull().sum().sum()
    interpolated_null_count = processed_gaze_df[columns_to_interpolate].isnull().sum().sum()

    print(f"Total null values before interpolation: {original_null_count}")
    print(f"Total null values after interpolation: {interpolated_null_count}")
    print(f"Number of values interpolated: {original_null_count - interpolated_null_count}") 
    """

    # calculate the averages (not strict) between the left and right eye:

    # Calculate average between left and right eyes for gaze origin, gaze direction, and pupil diameter
    processed_gaze_df['Avg_Gazeorigin_X'] = processed_gaze_df[['Data Eyeleft Gazeorigin X', 'Data Eyeright Gazeorigin X']].mean(axis=1, skipna=True)
    processed_gaze_df['Avg_Gazeorigin_Y'] = processed_gaze_df[['Data Eyeleft Gazeorigin Y', 'Data Eyeright Gazeorigin Y']].mean(axis=1, skipna=True)
    processed_gaze_df['Avg_Gazeorigin_Z'] = processed_gaze_df[['Data Eyeleft Gazeorigin Z', 'Data Eyeright Gazeorigin Z']].mean(axis=1, skipna=True)

    processed_gaze_df['Avg_Gazedirection_X'] = processed_gaze_df[['Data Eyeleft Gazedirection X', 'Data Eyeright Gazedirection X']].mean(axis=1, skipna=True)
    processed_gaze_df['Avg_Gazedirection_Y'] = processed_gaze_df[['Data Eyeleft Gazedirection Y', 'Data Eyeright Gazedirection Y']].mean(axis=1, skipna=True)
    processed_gaze_df['Avg_Gazedirection_Z'] = processed_gaze_df[['Data Eyeleft Gazedirection Z', 'Data Eyeright Gazedirection Z']].mean(axis=1, skipna=True)

    processed_gaze_df['Avg_Pupildiameter'] = processed_gaze_df[['Data Eyeleft Pupildiameter', 'Data Eyeright Pupildiameter']].mean(axis=1, skipna=True)

    # Check for any remaining NaN values and fill with single eye data if one eye is missing
    processed_gaze_df['Avg_Gazeorigin_X'] = processed_gaze_df['Avg_Gazeorigin_X'].fillna(processed_gaze_df['Data Eyeleft Gazeorigin X']).fillna(processed_gaze_df['Data Eyeright Gazeorigin X'])
    processed_gaze_df['Avg_Gazeorigin_Y'] = processed_gaze_df['Avg_Gazeorigin_Y'].fillna(processed_gaze_df['Data Eyeleft Gazeorigin Y']).fillna(processed_gaze_df['Data Eyeright Gazeorigin Y'])
    processed_gaze_df['Avg_Gazeorigin_Z'] = processed_gaze_df['Avg_Gazeorigin_Z'].fillna(processed_gaze_df['Data Eyeleft Gazeorigin Z']).fillna(processed_gaze_df['Data Eyeright Gazeorigin Z'])

    processed_gaze_df['Avg_Gazedirection_X'] = processed_gaze_df['Avg_Gazedirection_X'].fillna(processed_gaze_df['Data Eyeleft Gazedirection X']).fillna(processed_gaze_df['Data Eyeright Gazedirection X'])
    processed_gaze_df['Avg_Gazedirection_Y'] = processed_gaze_df['Avg_Gazedirection_Y'].fillna(processed_gaze_df['Data Eyeleft Gazedirection Y']).fillna(processed_gaze_df['Data Eyeright Gazedirection Y'])
    processed_gaze_df['Avg_Gazedirection_Z'] = processed_gaze_df['Avg_Gazedirection_Z'].fillna(processed_gaze_df['Data Eyeleft Gazedirection Z']).fillna(processed_gaze_df['Data Eyeright Gazedirection Z'])

    processed_gaze_df['Avg_Pupildiameter'] = processed_gaze_df['Avg_Pupildiameter'].fillna(processed_gaze_df['Data Eyeleft Pupildiameter']).fillna(processed_gaze_df['Data Eyeright Pupildiameter'])

    # Moving average smoothing for noise reduction
    for col in ['Avg_Gazeorigin_X', 'Avg_Gazeorigin_Y', 'Avg_Gazeorigin_Z', 'Avg_Gazedirection_X', 'Avg_Gazedirection_Y', 'Avg_Gazedirection_Z', 'Avg_Pupildiameter']:
        processed_gaze_df[col] = processed_gaze_df[col].rolling(window=window_size, min_periods=1).mean()

    # I-VT velocity calculation
    sampling_rate = 1 / (processed_gaze_df['Timestamp'].diff().mean())  # Calculate the approximate sampling rate
    window_length_samples = max(1, int((velocity_window_ms / 1000) * sampling_rate))

    processed_gaze_df['Velocity'] = np.nan

    for i in range(window_length_samples, len(processed_gaze_df) - window_length_samples):
        eye_position_middle = processed_gaze_df.loc[i, ['Avg_Gazeorigin_X', 'Avg_Gazeorigin_Y', 'Avg_Gazeorigin_Z']]
        gaze_point_start = processed_gaze_df.loc[i - window_length_samples, ['Avg_Gazedirection_X', 'Avg_Gazedirection_Y', 'Avg_Gazedirection_Z']]
        gaze_point_end = processed_gaze_df.loc[i + window_length_samples, ['Avg_Gazedirection_X', 'Avg_Gazedirection_Y', 'Avg_Gazedirection_Z']]
        
        visual_angle = np.arccos(np.clip(np.dot(gaze_point_start, gaze_point_end) / (np.linalg.norm(gaze_point_start) * np.linalg.norm(gaze_point_end)), -1.0, 1.0))
        time_difference = processed_gaze_df.loc[i + window_length_samples, 'Timestamp'] - processed_gaze_df.loc[i - window_length_samples, 'Timestamp']

        # Check for zero division
        if time_difference != 0:
            processed_gaze_df.loc[i, 'Velocity'] = visual_angle / time_difference
        else:
            processed_gaze_df.loc[i, 'Velocity'] = np.nan  # Assign NaN if time_difference is zero

    # fill the first and last velocity values
    processed_gaze_df['Velocity'] = processed_gaze_df['Velocity'].ffill().bfill()

    # Initialize fixation-related columns
    processed_gaze_df['Is_Fixation'] = False
    processed_gaze_df['Fixation_ID'] = np.nan
    current_fixation_id = 0
    fixation_start_time = None

    # Iterate through the DataFrame to identify fixations based on velocity
    for i in range(len(processed_gaze_df)):
        # Check if the current velocity exceeds the threshold
        if processed_gaze_df.loc[i, 'Velocity'] < velocity_threshold:
            # Mark as a potential fixation
            if fixation_start_time is None:
                fixation_start_time = processed_gaze_df.loc[i, 'Timestamp']
                processed_gaze_df.loc[i, 'Is_Fixation'] = True
                processed_gaze_df.loc[i, 'Fixation_ID'] = current_fixation_id
            else:
                processed_gaze_df.loc[i, 'Is_Fixation'] = True
                processed_gaze_df.loc[i, 'Fixation_ID'] = current_fixation_id
        else:
            # If the velocity exceeds the threshold and we have a fixation
            if fixation_start_time is not None:
                fixation_end_time = processed_gaze_df.loc[i - 1, 'Timestamp']
                fixation_duration = fixation_end_time - fixation_start_time
                
                # Check if the fixation duration is long enough to be considered valid
                if fixation_duration >= min_fixation_duration:
                    current_fixation_id += 1
                
                # Reset fixation tracking
                fixation_start_time = None
                
    # Assign unique Fixation_ID to the remaining fixations
    if fixation_start_time is not None:
        fixation_end_time = processed_gaze_df.iloc[-1]['Timestamp']
        fixation_duration = fixation_end_time - fixation_start_time
        
        # if the fixation duration is larger than the provided parameter, identify the record as a fixation
        if fixation_duration >= min_fixation_duration:
            for j in range(i, len(processed_gaze_df)):
                processed_gaze_df.loc[j, 'Fixation_ID'] = current_fixation_id
    processed_gaze_df['Fixation_ID'] = processed_gaze_df['Fixation_ID'].fillna(value=-1)

    processed_gaze_df.to_csv(f'intermediary_data/processed_eye_tracking/participant_{p_num}/processed_gaze_{p_num}.csv', index=False)

    # plot the velocity data into a chart to visualize and save
    plt.figure(figsize=(12, 6))
    plt.plot(processed_gaze_df['Timestamp'], processed_gaze_df['Velocity'], label='Velocity (degrees/second)', color='blue')
    plt.title('Velocity Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (degrees/second)')
    plt.axhline(y=velocity_threshold, color='red', linestyle='--', label='Velocity Threshold')
    plt.legend()

    plt.savefig(f'intermediary_data/processed_eye_tracking/participant_{p_num}/processed_gaze_chart_{p_num}.png')
