""" This is the second script in the data preprocessing pipeline.
It applies interpolation and the gap fill-in interpolation algorithm to the eye tracking from the extracted encounters
The script calculates the average between the left and right eye, applies a moving average filter to reduce noise, and implements the I-VT fixation filter to identify fixations.
The data files are saved in intermediary_data/processed_eye_tracking"""

# implementation of gap fill-in interpolation algorithm, eye position averaging algorithm, moving average algorithm (for noise), and the I-VT implementation is detailed in Tobii I-VT Fixation Filter:
# Available Online: http://www.vinis.co.kr/ivt_filter.pdf

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

# Set parameters
max_gap_length = float(input("Enter max gap length to interpolate (seconds) = ") or 0.075) #default 75 milliseconds
window_size = int(input("Enter the window size for the moving average (number of samples) = ") or 5)  # default window size is 5 samples for noise reduction filter
velocity_window_ms = float(input("Enter the window length for velocity calculation (milliseconds) = ") or 20)  # default 20 ms for I-VT filter

# I-VT Fixation Filter Parameters
velocity_threshold = float(input("Enter the velocity threshold for I-VT filter (degrees/second) = ") or 30)  # default threshold of 30 degrees/second
min_fixation_duration = float(input("Enter the minimum fixation duration (seconds) = ") or 0.1)  # default minimum duration of 100 ms

# get data summary
data_summary = pd.read_excel('summary_files/data_summary.xlsx')

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary),colour='green'):
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    p_num = summary_row["participant_id"]

    # get the intersection locations for the hazard order
    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx',sheet_name=hazard_order)

    # iterate through each of the 4 hazards
    for current_hazard in range(0,4):
        intersection_type = (list(intersection_locations)[current_hazard*2]).split("_")[2]

        gaze_encounter = pd.read_csv(f'data/intermediary_data/extracted_encounters_5/participant_{p_num}/gaze_encounter_{p_num}_{intersection_type}.csv')
        gaze_encounter['Timestamp'] = gaze_encounter['Timestamp'].astype(float)

        columns_to_interpolate = [col for col in gaze_encounter.columns if col not in ['Timestamp', 'Type']]

        processed_gaze = gaze_encounter.copy()

        # Identify gaps in the selected columns by first identifying values where values are missing (a flag is set in a temporary column for each data column, called is_valid)
        processed_gaze['is_valid'] = processed_gaze[columns_to_interpolate].notna().all(axis=1)
        # Gaps are then found in the non-valid columns (columns that are missing data) by setting a flag in two temporary columns (gap_start) and (gap_end)
        processed_gaze['gap_start'] = (~processed_gaze['is_valid']) & (processed_gaze['is_valid'].shift(1) | (processed_gaze.index == 0))
        processed_gaze['gap_end'] = (~processed_gaze['is_valid']) & (processed_gaze['is_valid'].shift(-1) | (processed_gaze.index == len(processed_gaze) - 1))

        # Process each gap
        gap_starts = processed_gaze.index[processed_gaze['gap_start']]
        for start in gap_starts:
            # Find the end of the gap if it exists
            gap_end_candidates = processed_gaze.index[processed_gaze['gap_end'] & (processed_gaze.index >= start)]
            
            # Ensure that gap_end_candidates is not empty
            if len(gap_end_candidates) > 0:
                end = gap_end_candidates[0]
                gap_duration = processed_gaze.loc[end, 'Timestamp'] - processed_gaze.loc[start, 'Timestamp']
                
                # Only interpolate if the gap is shorter than max_gap_length
                if gap_duration <= max_gap_length:
                    for col in columns_to_interpolate:
                        last_valid = processed_gaze.loc[:start-1, col].last_valid_index()
                        next_valid = processed_gaze.loc[end+1:, col].first_valid_index()
                        
                        if last_valid is not None and next_valid is not None:
                            # Get the timestamps and values for the range we're interpolating
                            timestamps = processed_gaze.loc[last_valid:next_valid, 'Timestamp']
                            start_value = processed_gaze.loc[last_valid, col]
                            end_value = processed_gaze.loc[next_valid, col]
                            start_time = processed_gaze.loc[last_valid, 'Timestamp']
                            end_time = processed_gaze.loc[next_valid, 'Timestamp']
                            
                            # Perform linear interpolation
                            interpolated_values = start_value + (end_value - start_value) * (timestamps - start_time) / (end_time - start_time)
                            
                            # Assign the interpolated values
                            processed_gaze.loc[last_valid:next_valid, col] = interpolated_values
            else:
                # Occurs when end of file is a gap
                pass

        # Drop helper columns
        processed_gaze = processed_gaze.drop(columns=['is_valid', 'gap_start', 'gap_end'])

        # Print summary of interpolation
        """
        original_null_count = gaze_encounter[columns_to_interpolate].isnull().sum().sum()
        interpolated_null_count = processed_gaze[columns_to_interpolate].isnull().sum().sum()

        print(f"Total null values before interpolation: {original_null_count}")
        print(f"Total null values after interpolation: {interpolated_null_count}")
        print(f"Number of values interpolated: {original_null_count - interpolated_null_count}") 
        """

        # calculate the averages (not strict) between the left and right eye:

        # Calculate average between left and right eyes for gaze origin, gaze direction, and pupil diameter
        processed_gaze['Avg_Gazeorigin_X'] = processed_gaze[['Data Eyeleft Gazeorigin X', 'Data Eyeright Gazeorigin X']].mean(axis=1, skipna=True)
        processed_gaze['Avg_Gazeorigin_Y'] = processed_gaze[['Data Eyeleft Gazeorigin Y', 'Data Eyeright Gazeorigin Y']].mean(axis=1, skipna=True)
        processed_gaze['Avg_Gazeorigin_Z'] = processed_gaze[['Data Eyeleft Gazeorigin Z', 'Data Eyeright Gazeorigin Z']].mean(axis=1, skipna=True)

        processed_gaze['Avg_Gazedirection_X'] = processed_gaze[['Data Eyeleft Gazedirection X', 'Data Eyeright Gazedirection X']].mean(axis=1, skipna=True)
        processed_gaze['Avg_Gazedirection_Y'] = processed_gaze[['Data Eyeleft Gazedirection Y', 'Data Eyeright Gazedirection Y']].mean(axis=1, skipna=True)
        processed_gaze['Avg_Gazedirection_Z'] = processed_gaze[['Data Eyeleft Gazedirection Z', 'Data Eyeright Gazedirection Z']].mean(axis=1, skipna=True)

        processed_gaze['Avg_Pupildiameter'] = processed_gaze[['Data Eyeleft Pupildiameter', 'Data Eyeright Pupildiameter']].mean(axis=1, skipna=True)

        # Check for any remaining NaN values and fill with single eye data if one eye is missing
        processed_gaze['Avg_Gazeorigin_X'] = processed_gaze['Avg_Gazeorigin_X'].fillna(processed_gaze['Data Eyeleft Gazeorigin X']).fillna(processed_gaze['Data Eyeright Gazeorigin X'])
        processed_gaze['Avg_Gazeorigin_Y'] = processed_gaze['Avg_Gazeorigin_Y'].fillna(processed_gaze['Data Eyeleft Gazeorigin Y']).fillna(processed_gaze['Data Eyeright Gazeorigin Y'])
        processed_gaze['Avg_Gazeorigin_Z'] = processed_gaze['Avg_Gazeorigin_Z'].fillna(processed_gaze['Data Eyeleft Gazeorigin Z']).fillna(processed_gaze['Data Eyeright Gazeorigin Z'])

        processed_gaze['Avg_Gazedirection_X'] = processed_gaze['Avg_Gazedirection_X'].fillna(processed_gaze['Data Eyeleft Gazedirection X']).fillna(processed_gaze['Data Eyeright Gazedirection X'])
        processed_gaze['Avg_Gazedirection_Y'] = processed_gaze['Avg_Gazedirection_Y'].fillna(processed_gaze['Data Eyeleft Gazedirection Y']).fillna(processed_gaze['Data Eyeright Gazedirection Y'])
        processed_gaze['Avg_Gazedirection_Z'] = processed_gaze['Avg_Gazedirection_Z'].fillna(processed_gaze['Data Eyeleft Gazedirection Z']).fillna(processed_gaze['Data Eyeright Gazedirection Z'])

        processed_gaze['Avg_Pupildiameter'] = processed_gaze['Avg_Pupildiameter'].fillna(processed_gaze['Data Eyeleft Pupildiameter']).fillna(processed_gaze['Data Eyeright Pupildiameter'])

        # Moving average smoothing for noise reduction
        for col in ['Avg_Gazeorigin_X', 'Avg_Gazeorigin_Y', 'Avg_Gazeorigin_Z', 'Avg_Gazedirection_X', 'Avg_Gazedirection_Y', 'Avg_Gazedirection_Z', 'Avg_Pupildiameter']:
            processed_gaze[col] = processed_gaze[col].rolling(window=window_size, min_periods=1).mean()

        # I-VT velocity calculation
        timestamp_diff_mean = processed_gaze['Timestamp'].diff().mean()

        if pd.isna(timestamp_diff_mean) or timestamp_diff_mean == 0:
            sampling_rate = 50 #default sampling rate 50Hz
        else:
            sampling_rate = 1 / timestamp_diff_mean

        window_length_samples = max(1, int((velocity_window_ms / 1000) * sampling_rate))

        processed_gaze['Velocity'] = np.nan

        for i in range(window_length_samples, len(processed_gaze) - window_length_samples):
            eye_position_middle = processed_gaze.loc[i, ['Avg_Gazeorigin_X', 'Avg_Gazeorigin_Y', 'Avg_Gazeorigin_Z']]
            gaze_point_start = processed_gaze.loc[i - window_length_samples, ['Avg_Gazedirection_X', 'Avg_Gazedirection_Y', 'Avg_Gazedirection_Z']]
            gaze_point_end = processed_gaze.loc[i + window_length_samples, ['Avg_Gazedirection_X', 'Avg_Gazedirection_Y', 'Avg_Gazedirection_Z']]
            
            visual_angle = np.arccos(np.clip(np.dot(gaze_point_start, gaze_point_end) / (np.linalg.norm(gaze_point_start) * np.linalg.norm(gaze_point_end)), -1.0, 1.0))
            visual_angle_degrees = np.degrees(visual_angle)
            time_difference = processed_gaze.loc[i + window_length_samples, 'Timestamp'] - processed_gaze.loc[i - window_length_samples, 'Timestamp']

            # Check for zero division
            if time_difference != 0:
                processed_gaze.loc[i, 'Velocity'] = visual_angle_degrees / time_difference
            else:
                processed_gaze.loc[i, 'Velocity'] = np.nan  # Assign NaN if time_difference is zero


        # fill the first and last velocity values
        processed_gaze['Velocity'] = processed_gaze['Velocity'].ffill().bfill()

        # Initialize fixation-related columns
        processed_gaze['Is_Fixation'] = False
        processed_gaze['Fixation_ID'] = np.nan
        current_fixation_id = 0
        fixation_start_time = None

        # Iterate through the DataFrame to identify fixations based on velocity
        for i in range(len(processed_gaze)):
            # Check if the current velocity exceeds the threshold
            if processed_gaze.loc[i, 'Velocity'] < velocity_threshold:
                # Mark as a potential fixation
                if fixation_start_time is None:
                    fixation_start_time = processed_gaze.loc[i, 'Timestamp']
                    processed_gaze.loc[i, 'Is_Fixation'] = True
                    processed_gaze.loc[i, 'Fixation_ID'] = current_fixation_id
                else:
                    processed_gaze.loc[i, 'Is_Fixation'] = True
                    processed_gaze.loc[i, 'Fixation_ID'] = current_fixation_id
            else:
                # If the velocity exceeds the threshold and we have a fixation
                if fixation_start_time is not None:
                    fixation_end_time = processed_gaze.loc[i - 1, 'Timestamp']
                    fixation_duration = fixation_end_time - fixation_start_time
                    
                    # Check if the fixation duration is long enough to be considered valid
                    if fixation_duration >= min_fixation_duration:
                        current_fixation_id += 1
                    
                    # Reset fixation tracking
                    fixation_start_time = None
                    
        # Assign unique Fixation_ID to the remaining fixations
        if fixation_start_time is not None:
            fixation_end_time = processed_gaze.iloc[-1]['Timestamp']
            fixation_duration = fixation_end_time - fixation_start_time
            
            # if the fixation duration is larger than the provided parameter, identify the record as a fixation
            if fixation_duration >= min_fixation_duration:
                for j in range(i, len(processed_gaze)):
                    processed_gaze.loc[j, 'Fixation_ID'] = current_fixation_id
        processed_gaze['Fixation_ID'] = processed_gaze['Fixation_ID'].fillna(value=-1)

        os.makedirs(f'data/intermediary_data/processed_eye_tracking/participant_{p_num}',exist_ok=True)

        # for c, col in enumerate(processed_gaze.columns):
        #     print(c,col)
        processed_gaze.to_csv(f'data/intermediary_data/processed_eye_tracking/participant_{p_num}/processed_gaze_{p_num}_{intersection_type}.csv', index=False)

        # plot the velocity data into a chart to visualize and save
        plt.figure(figsize=(12, 6))
        plt.plot(processed_gaze['Timestamp'], processed_gaze['Velocity'], label='Velocity (degrees/second)', color='blue')
        plt.title('Velocity Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (degrees/second)')
        plt.axhline(y=velocity_threshold, color='red', linestyle='--', label='Velocity Threshold')
        plt.legend()

        plt.savefig(f'data/intermediary_data/processed_eye_tracking/participant_{p_num}/processed_gaze_chart_{p_num}_{intersection_type}.png')
        plt.close()