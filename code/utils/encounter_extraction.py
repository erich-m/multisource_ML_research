""" This script is the first stage of the data preprocessing pipeline. 
It extracts the encounters for each participant and saves them to csv files in the intermediary data folder.
Encounters are extracted based on the closest approach to the intersection where the hazard approaches the driver. 
The script only gets from the identified point of interest and backwards up to a specified length of time.
The data is saved in the intermediary_data folder in extracted_encounters_{encounter_length}"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

# get encounter_length
encounter_length = float(input("encounter length (seconds) = ") or 5)
# get data summary
data_summary = pd.read_excel('summary_files/data_summary.xlsx')

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary), colour='green'):
    # get hazard order id and time offset between driving and eye-tracking
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    time_offset = float(summary_row["Time Difference"])
    p_num = summary_row["participant_id"]

    # get the intersection locations for the hazard order
    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx', sheet_name=hazard_order)

    # obtain current participant driving data
    driving_df = pd.read_csv(f'data/raw_data/participant_{p_num}/driving_{p_num}.csv', sep=';', encoding='ISO-8859-1')
    driving_df.drop(index=0, inplace=True)
    
    # Drop any unnamed columns
    unnamed_cols = [col for col in driving_df.columns if 'Unnamed' in col]
    if unnamed_cols:
        driving_df = driving_df.drop(columns=unnamed_cols)

    # obtain current participant eye tracking data
    eye_tracking_gaze_df = pd.read_excel(f'data/raw_data/participant_{p_num}/eye_tracking_{p_num}.xlsx', sheet_name='Gaze Data')
    eye_tracking_imu_df = pd.read_excel(f'data/raw_data/participant_{p_num}/eye_tracking_{p_num}.xlsx', sheet_name='IMU Data')

    # iterate through each of the 4 hazards
    for current_hazard in range(0,4):
        # Drop only the distance columns from previous iterations
        previous_distance_cols = [f'distance_to_inter_{i}' for i in range(current_hazard)]
        if previous_distance_cols:
            driving_df = driving_df.drop(columns=previous_distance_cols, errors='ignore')
        
        # location of the current intersection
        target_x = intersection_locations.iloc[0,current_hazard*2]
        target_y = intersection_locations.iloc[0,(current_hazard*2)+1]
        intersection_type = (list(intersection_locations)[current_hazard*2]).split("_")[2]
        
        # get driver vehicle position cols
        x_col = driving_df.iloc[:,3].astype(float)
        y_col = driving_df.iloc[:,4].astype(float)

        # calculate distance from driver vehicle to target intersection and find row with min distance
        driving_df['distance_to_inter_' + str(current_hazard)] = np.sqrt((x_col - target_x)**2 + (y_col-target_y)**2)
        closest_approach_index = driving_df['distance_to_inter_' + str(current_hazard)].idxmin()
        closest_approach = driving_df.loc[closest_approach_index]

        # calculate time bounds of encounter
        lower_time_bound = float(closest_approach.iloc[0]) - encounter_length
        upper_time_bound = float(closest_approach.iloc[0])

        # bounds for eye tracking data
        lower_time_bound_offset = lower_time_bound - time_offset
        upper_time_bound_offset = upper_time_bound - time_offset

        # filter for driving encounter (Time is now column 0)
        driving_encounter = driving_df[(driving_df.iloc[:, 0].astype(float) >= lower_time_bound) & (driving_df.iloc[:, 0].astype(float) <= upper_time_bound)]

        # filter for eye tracking encounter (Timestamp is now column 1)
        gaze_encounter = eye_tracking_gaze_df[(eye_tracking_gaze_df.iloc[:, 1].astype(float) >= lower_time_bound_offset) & (eye_tracking_gaze_df.iloc[:, 1].astype(float) <= upper_time_bound_offset)]
        imu_encounter = eye_tracking_imu_df[(eye_tracking_imu_df.iloc[:, 1].astype(float) >= lower_time_bound_offset) & (eye_tracking_imu_df.iloc[:, 1].astype(float) <= upper_time_bound_offset)]

        # Drop any unnamed columns before saving
        driving_cols = [col for col in driving_encounter.columns if 'Unnamed' in col]
        if driving_cols:
            driving_encounter = driving_encounter.drop(columns=driving_cols)
            
        gaze_cols = [col for col in gaze_encounter.columns if 'Unnamed' in col]
        if gaze_cols:
            gaze_encounter = gaze_encounter.drop(columns=gaze_cols)
            
        imu_cols = [col for col in imu_encounter.columns if 'Unnamed' in col]
        if imu_cols:
            imu_encounter = imu_encounter.drop(columns=imu_cols)

        # save the encounters for each participant to csv file in the intermediary data folder
        os.makedirs(f'data/intermediary_data/extracted_encounters_{int(encounter_length)}/participant_{p_num}', exist_ok=True)
        driving_encounter.to_csv(f'data/intermediary_data/extracted_encounters_{int(encounter_length)}/participant_{p_num}/driving_encounter_{p_num}_{intersection_type}.csv', index=False)
        gaze_encounter.to_csv(f'data/intermediary_data/extracted_encounters_{int(encounter_length)}/participant_{p_num}/gaze_encounter_{p_num}_{intersection_type}.csv', index=False)
        imu_encounter.to_csv(f'data/intermediary_data/extracted_encounters_{int(encounter_length)}/participant_{p_num}/imu_encounter_{p_num}_{intersection_type}.csv', index=False)