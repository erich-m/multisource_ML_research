import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os
from tqdm import tqdm

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

data_summary = pd.read_excel('summary_files/data_summary.xlsx')

def normalize(v):
    return v / np.linalg.norm(v)

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary),colour='green'):
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    p_num = summary_row["participant_id"]

    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx',sheet_name=hazard_order)

    for current_hazard in range(0,4):
        intersection_type = (list(intersection_locations)[current_hazard*2]).split("_")[2]
        encounter_df = pd.read_csv(f'intermediary_data/encounter_data/participant_{p_num}/encounter_data_{p_num}_{intersection_type}.csv')
        # print(list(encounter_df.columns))

        # Process each row
        for idx in encounter_df.index:
            # Get IMU data
            accel = np.array([
                encounter_df.loc[idx, 'Data Accelerometer X'],
                encounter_df.loc[idx, 'Data Accelerometer Y'],
                encounter_df.loc[idx, 'Data Accelerometer Z']
            ])
            mag = np.array([
                encounter_df.loc[idx, 'Data Magnetometer X'],
                encounter_df.loc[idx, 'Data Magnetometer Y'],
                encounter_df.loc[idx, 'Data Magnetometer Z']
            ])
            
            # Normalize vectors
            down = normalize(accel)  # Assuming accelerometer mainly measures gravity
            east = normalize(np.cross(down, mag))
            north = normalize(np.cross(east, down))
            
            # Create rotation matrix from IMU frame to global frame
            R_imu_to_global = np.vstack((east, down, north)).T
            
            # Get gaze directions for both eyes
            left_gaze = np.array([
                encounter_df.loc[idx, 'Data Eyeleft Gazedirection X'],
                encounter_df.loc[idx, 'Data Eyeleft Gazedirection Y'],
                encounter_df.loc[idx, 'Data Eyeleft Gazedirection Z']
            ])
            right_gaze = np.array([
                encounter_df.loc[idx, 'Data Eyeright Gazedirection X'],
                encounter_df.loc[idx, 'Data Eyeright Gazedirection Y'],
                encounter_df.loc[idx, 'Data Eyeright Gazedirection Z']
            ])
            
            # Transform gaze vectors to global coordinates
            left_gaze_global = R_imu_to_global @ left_gaze
            right_gaze_global = R_imu_to_global @ right_gaze
            
            # Get car's yaw angle (converting to radians)
            yaw = np.radians(encounter_df.loc[idx, 'CoG position/Yaw'])
            
            # Create rotation matrix for car's orientation
            R_car = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            # Project gaze vectors onto horizontal plane (x-z for glasses, x-y for car)
            # and normalize
            left_gaze_horizontal = normalize(np.array([left_gaze_global[0], left_gaze_global[2]]))
            right_gaze_horizontal = normalize(np.array([right_gaze_global[0], right_gaze_global[2]]))
            
            # Transform to car coordinates
            # First adjust for driver position offset (1.5m back, 0.5m right)
            car_position = np.array([
                encounter_df.loc[idx, 'CoG position/X'],
                encounter_df.loc[idx, 'CoG position/Y']
            ])
            
            # Driver position in car coordinates (before rotation)
            driver_offset = np.array([-1.5, -0.5])
            
            # Apply car rotation to driver offset
            driver_offset_rotated = R_car @ driver_offset
            
            # Calculate driver position in global coordinates
            driver_position = car_position + driver_offset_rotated
            
            # Transform gaze vectors to car coordinates
            left_gaze_car = R_car.T @ left_gaze_horizontal
            right_gaze_car = R_car.T @ right_gaze_horizontal
            
            # Store transformed vectors back in DataFrame
            encounter_df.loc[idx, 'left_gaze_car_x'] = left_gaze_car[0]
            encounter_df.loc[idx, 'left_gaze_car_y'] = left_gaze_car[1]
            encounter_df.loc[idx, 'right_gaze_car_x'] = right_gaze_car[0]
            encounter_df.loc[idx, 'right_gaze_car_y'] = right_gaze_car[1]
            
            # Store driver position
            encounter_df.loc[idx, 'driver_position_x'] = driver_position[0]
            encounter_df.loc[idx, 'driver_position_y'] = driver_position[1]

        # Handle yaw angle discontinuity
        encounter_df['yaw_continuous'] = np.unwrap(np.radians(encounter_df['CoG position/Yaw'])) * 180 / np.pi

        os.makedirs(f'intermediary_data/transformed_encounter_data/participant_{p_num}',exist_ok=True)
        encounter_df.to_csv(f'intermediary_data/transformed_encounter_data/participant_{p_num}/transformed_data_{p_num}_{intersection_type}.csv')