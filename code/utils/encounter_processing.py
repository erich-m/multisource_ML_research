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

        for idx in encounter_df.index:
            # acceleration as a 3D vector
            accel = np.array([
                encounter_df.loc[idx, 'Data Accelerometer X'],
                encounter_df.loc[idx, 'Data Accelerometer Y'],
                encounter_df.loc[idx, 'Data Accelerometer Z']
            ])
            # magentometer as a 3D vector
            mag = np.array([
                encounter_df.loc[idx, 'Data Magnetometer X'],
                encounter_df.loc[idx, 'Data Magnetometer Y'],
                encounter_df.loc[idx, 'Data Magnetometer Z']
            ])
            
            # normalize vectors
            down = normalize(accel)  # accelerometer mainly measures gravity
            east = normalize(np.cross(down, mag))
            north = normalize(np.cross(east, down))
            
            # rotation matrix from the IMU directions
            R_imu_to_global = np.vstack((east, down, north)).T
            
            # gaze direction for left eye as a 3D vector
            left_gaze = np.array([
                encounter_df.loc[idx, 'Data Eyeleft Gazedirection X'],
                encounter_df.loc[idx, 'Data Eyeleft Gazedirection Y'],
                encounter_df.loc[idx, 'Data Eyeleft Gazedirection Z']
            ])
            # gaze direction for right eye as a 3D vecctor
            right_gaze = np.array([
                encounter_df.loc[idx, 'Data Eyeright Gazedirection X'],
                encounter_df.loc[idx, 'Data Eyeright Gazedirection Y'],
                encounter_df.loc[idx, 'Data Eyeright Gazedirection Z']
            ])
            
            # matrix multiply the rotation matrix from the IMU and the gaze vectors to get the orientation of the vectors in global coordinates
            left_gaze_global = R_imu_to_global @ left_gaze
            right_gaze_global = R_imu_to_global @ right_gaze
            
            # get the car direction (yaw, where yaw is the rotation around the vertical axis) in radians
            yaw = np.radians(encounter_df.loc[idx, 'CoG position/Yaw'])
            
            # 2x2 rotation matrix for the car from the yaw of the car
            R_car = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            # project gaze vectors onto horizontal plane (x-z for glasses, x-y for car) and normalize
            left_gaze_horizontal = normalize(np.array([left_gaze_global[0], left_gaze_global[2]]))
            right_gaze_horizontal = normalize(np.array([right_gaze_global[0], right_gaze_global[2]]))
            
            # transform to car coordinates and adjust for driver position offset (1.5m back, 0.5m right)
            car_position = np.array([
                encounter_df.loc[idx, 'CoG position/X'],
                encounter_df.loc[idx, 'CoG position/Y']
            ])
            
            # driver position in car coordinates (before rotation)
            driver_offset = np.array([-1.5, -0.5])
            
            # apply car rotation to driver offset
            driver_offset_rotated = R_car @ driver_offset
            
            # calculate driver position in global coordinates
            driver_position = car_position + driver_offset_rotated
            
            # transform gaze vectors to car coordinates
            left_gaze_car = R_car.T @ left_gaze_horizontal
            right_gaze_car = R_car.T @ right_gaze_horizontal
            
            # store transformed vectors back in df
            encounter_df.loc[idx, 'left_gaze_car_x'] = left_gaze_car[0]
            encounter_df.loc[idx, 'left_gaze_car_y'] = left_gaze_car[1]
            encounter_df.loc[idx, 'right_gaze_car_x'] = right_gaze_car[0]
            encounter_df.loc[idx, 'right_gaze_car_y'] = right_gaze_car[1]
            
            # store driver position
            encounter_df.loc[idx, 'driver_position_x'] = driver_position[0]
            encounter_df.loc[idx, 'driver_position_y'] = driver_position[1]

        # handle yaw angle discontinuity (separate from the above calculations)
        encounter_df['yaw_continuous'] = np.unwrap(np.radians(encounter_df['CoG position/Yaw'])) * 180 / np.pi

        os.makedirs(f'intermediary_data/transformed_encounter_data/participant_{p_num}',exist_ok=True)
        encounter_df.to_csv(f'intermediary_data/transformed_encounter_data/participant_{p_num}/transformed_data_{p_num}_{intersection_type}.csv')