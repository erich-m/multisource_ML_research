import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

# get encounter_length
encounter_length = float(input("encounter length (seconds) = "))
# get data summary
data_summary = pd.read_excel('summary_files/data_summary.xlsx')

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary)):
    # get hazard order id and time offset between driving and eye-tracking
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    time_offset = float(summary_row["Time Difference"])
    p_num = summary_row["participant_id"]

    # obtain current participant driving data
    driving_df = pd.read_csv(f'raw_data/participant_{p_num}/driving_{p_num}.csv',sep=';',encoding='ISO-8859-1')
    driving_df.drop(index=0,inplace=True)

    # obtain current participant eye tracking data
    eye_tracking_gaze_df = pd.read_excel(f'raw_data/participant_{p_num}/eye_tracking_{p_num}.xlsx',sheet_name='Gaze Data')
    eye_tracking_imu_df = pd.read_excel(f'raw_data/participant_{p_num}/eye_tracking_{p_num}.xlsx',sheet_name='IMU Data')

    # get the intersection locations for the hazard order
    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx',sheet_name=hazard_order)

    # iterate through each of the 4 hazards
    for current_hazard in range(0,4):
        # TODO: Save current hazard as different file. Currently getting one per participant 
        # location of the current intersection
        target_x = intersection_locations.iloc[0,current_hazard*2]
        target_y = intersection_locations.iloc[0,(current_hazard*2)+1]
        
        # get driver vehicle position cols
        x_col = driving_df.iloc[:,3].astype(float)
        y_col = driving_df.iloc[:,4].astype(float)

        # calculate distance from driver vehicle to target intersection and find row with min distance
        driving_df['distance_to_inter_' + str(current_hazard)] = np.sqrt((x_col - target_x)**2 + (y_col-target_y)**2)
        closest_approach_index = driving_df['distance_to_inter_' + str(current_hazard)].idxmin()
        closest_approach = driving_df.loc[closest_approach_index]

        # calculate time bounds of encounter
        lower_time_bound = float(closest_approach.iloc[0]) - encounter_length  # closest_approach['Time'] -> closest_approach.iloc[0]
        upper_time_bound = float(closest_approach.iloc[0])  # closest_approach['Time'] -> closest_approach.iloc[0]

        # bounds for eye tracking data
        lower_time_bound_offset = lower_time_bound - time_offset
        upper_time_bound_offset = upper_time_bound - time_offset

        # filter for driving encounter (Time is now column 0)
        driving_encounter = driving_df[(driving_df.iloc[:, 0].astype(float) >= lower_time_bound) & (driving_df.iloc[:, 0].astype(float) <= upper_time_bound)]

        # filter for eye tracking encounter (Timestamp is now column 1)
        gaze_encounter = eye_tracking_gaze_df[(eye_tracking_gaze_df.iloc[:, 1].astype(float) >= lower_time_bound_offset) & (eye_tracking_gaze_df.iloc[:, 1].astype(float) <= upper_time_bound_offset)]
        imu_encounter = eye_tracking_imu_df[(eye_tracking_imu_df.iloc[:, 1].astype(float) >= lower_time_bound_offset) & (eye_tracking_imu_df.iloc[:, 1].astype(float) <= upper_time_bound_offset)]

        # save the encounters for each participant to csv file in the intermediary data folder
        driving_encounter.to_csv(f'intermediary_data/extracted_encounters_{int(encounter_length)}/participant_{p_num}/driving_encounter_{p_num}.csv', index=False)
        gaze_encounter.to_csv(f'intermediary_data/extracted_encounters_{int(encounter_length)}/participant_{p_num}/gaze_encounter_{p_num}.csv', index=False)
        imu_encounter.to_csv(f'intermediary_data/extracted_encounters_{int(encounter_length)}/participant_{p_num}/imu_encounter_{p_num}.csv', index=False)
