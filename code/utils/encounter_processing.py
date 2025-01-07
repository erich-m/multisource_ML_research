""" This script is the fourth script in the data preprocessing pipeline.
In this script, the eye tracking coordinates are converted from the eye tracking coordinate system to the car coordinate system.
The car world gaze vector is then compared with the position of the hazard vehicle to determine if the gaze vector intersects the hazard vehcile within a certain radius.
The results are saved in the transformed_encounter_data folder.
 """

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

data_summary = pd.read_excel('summary_files/data_summary.xlsx')

# define radius for hazard detection in meters
HAZARD_RADIUS = 2.0

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary), colour='green'):
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    hazard_label_order = "order_" + str(summary_row["HazardOrder"]) + "_hazards"
    p_num = summary_row["participant_id"]

    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx', sheet_name=hazard_order)
    hazard_labels = pd.read_excel('summary_files/intersection_locations.xlsx', sheet_name=hazard_label_order)
    for current_hazard in range(0, 4):
        intersect_count = 0
        intersection_type = (list(intersection_locations)[current_hazard*2]).split("_")[2]
        encounter_df = pd.read_csv(f'intermediary_data/encounter_data/participant_{p_num}/encounter_data_{p_num}_{intersection_type}.csv')
        encounter_df.columns = encounter_df.columns.str.strip()

        # Find the vehicle with the minimum distance
        moving_hazard = int((hazard_labels.iloc[0,current_hazard*2]).split('_')[1])+1

        # Get the column names for the hazard x and y positions
        hazard_x_col = f'CoG position/X.{moving_hazard}'
        hazard_y_col = f'CoG position/Y.{moving_hazard}'

        # Add hazard x and y columns to the encounter_df with meaningful labels
        encounter_df[f'hazard_{moving_hazard}_x'] = encounter_df[hazard_x_col]
        encounter_df[f'hazard_{moving_hazard}_y'] = encounter_df[hazard_y_col]
        
        # initialize gaze intersection column
        encounter_df['gaze_intersects_hazard'] = False
        
        # store vectors for timeline visualization
        timeline_data = {
            'glasses_space': [],
            'global_space': [],
            'car_space': [],
            'timestamps': [],
            'driver_positions': [],
            'car_positions': [],
            'car_directions': []
        }
        
        # iterate through each row of the dataframe
        for idx in encounter_df.index:
            # get the acceleration vector of the eye tracker imu
            accel = np.array([
                encounter_df.loc[idx, 'Data Accelerometer X'],
                encounter_df.loc[idx, 'Data Accelerometer Y'],
                encounter_df.loc[idx, 'Data Accelerometer Z']
            ])
            # get the magnetometer vector of the imu
            mag = np.array([
                encounter_df.loc[idx, 'Data Magnetometer X'],
                encounter_df.loc[idx, 'Data Magnetometer Y'],
                encounter_df.loc[idx, 'Data Magnetometer Z']
            ])
            
            # define the positive direction origin axes based on the tobii head unit coordinate system
            down = accel / np.linalg.norm(accel)
            east = np.cross(down, mag)
            east = east / np.linalg.norm(east)
            north = np.cross(east, down)
            north = north / np.linalg.norm(north)
            
            # get the imu rotation matrix 
            R_imu_to_global = np.vstack((east, down, north)).T
            
            # get the vectors of the left and right gaze direction
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
            
            # calculate the normalized average gaze vector
            combined_gaze = (left_gaze + right_gaze) / 2
            combined_gaze = combined_gaze / np.linalg.norm(combined_gaze)
            # align the normalized gaze vector with the global coordinate system
            combined_gaze_global = R_imu_to_global @ combined_gaze
            
            # project to horizontal plane using X and Z components from glasses space
            # since Z is forward and X is left in glasses space, we want to use these for the car's XY plane
            combined_gaze_horizontal = np.array([combined_gaze[2], -combined_gaze[0]])  # [forward, left] components
            combined_gaze_horizontal = combined_gaze_horizontal / np.linalg.norm(combined_gaze_horizontal)
            
            # get the yaw rotation of the car 
            yaw = np.radians(encounter_df.loc[idx, 'CoG position/Yaw'])
            
            # get the rotation matrix of the car coordinate system
            R_car = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            # get the car position in the car world
            car_position = np.array([
                encounter_df.loc[idx, 'CoG position/X'],
                encounter_df.loc[idx, 'CoG position/Y']
            ])
            
            # set the driver offset based on the origin of the vehicle model and the distance to the driver seat
            # the driver sits 1.497m backward and 0.4m to the left of the origin of the car model in the initial state
            # the initial driver vehicle position is (-35.64, -25.06)
            # the initial driver vehicle driver seat is (-37.137,-24.660)
            
            driver_offset = np.array([1.497, -0.4])
            # apply the cars rotation to the drivers position within the car
            driver_offset_rotated = R_car @ driver_offset
            # apply the offset to the car position in space to get the driver position in the car world
            driver_position = car_position + driver_offset_rotated
            
            # the gaze rotation is applied from the car rotation
            combined_gaze_car = R_car.T @ combined_gaze_horizontal
            
            # get hazard vehicle position
            hazard_position = np.array([
                encounter_df.loc[idx, hazard_x_col],
                encounter_df.loc[idx, hazard_y_col]
            ])
            
            # check if gaze vector intersects hazard circle
            # vector from driver to circle center
            to_circle = hazard_position - driver_position

            # project this vector onto the gaze direction
            proj_length = np.dot(to_circle, combined_gaze_car)

            # find the closest point on the ray to the circle center
            closest_point = driver_position + combined_gaze_car * proj_length

            # calculate distance from closest point to circle center
            distance = np.linalg.norm(closest_point - hazard_position)

            # check if distance is within radius
            encounter_df.loc[idx, 'gaze_intersects_hazard'] = distance <= HAZARD_RADIUS
            if distance <= HAZARD_RADIUS:
                intersect_count += 1
            
            # store vectors and positions as intermediates for visualization
            timeline_data['glasses_space'].append(combined_gaze)
            timeline_data['global_space'].append(combined_gaze_horizontal)
            timeline_data['car_space'].append(combined_gaze_car)
            timeline_data['timestamps'].append(encounter_df.loc[idx, 'Timestamp'])
            timeline_data['driver_positions'].append(driver_position)
            timeline_data['car_positions'].append(car_position)
            timeline_data['car_directions'].append(R_car @ np.array([1, 0]))
            
            # store transformed vectors in dataframe
            encounter_df.loc[idx, 'combined_gaze_car_x'] = combined_gaze_car[0]
            encounter_df.loc[idx, 'combined_gaze_car_y'] = combined_gaze_car[1]
            encounter_df.loc[idx, 'driver_position_x'] = driver_position[0]
            encounter_df.loc[idx, 'driver_position_y'] = driver_position[1]

        # create timeline visualization
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)

        # plot 1: glasses space timeline (3d)
        ax1 = plt.subplot(gs[0, 0], projection='3d')
        ax1.set_title('Glasses Space Timeline')
        
        # create color gradient based on time
        timestamps_normalized = np.array(timeline_data['timestamps'])
        timestamps_normalized = (timestamps_normalized - timestamps_normalized.min()) / (timestamps_normalized.max() - timestamps_normalized.min())
        colors = plt.cm.viridis(timestamps_normalized)
        
        for i, (gaze, color) in enumerate(zip(timeline_data['glasses_space'], colors)):
            if i % 10 == 0:  # plot every 10th vector to avoid overcrowding
                ax1.quiver(0, 0, 0, *gaze, color=color, alpha=0.5)
        
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # plot 2: global space timeline
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title('Global Space Timeline')
        
        for i, (gaze, color) in enumerate(zip(timeline_data['global_space'], colors)):
            if i % 10 == 0:
                ax2.arrow(0, 0, gaze[0], gaze[1], color=color, alpha=0.5, head_width=0.05)
        
        ax2.set_xlim([-1.5, 1.5])
        ax2.set_ylim([-1.5, 1.5])
        ax2.grid(True)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # plot 3: car space timeline with path
        ax3 = plt.subplot(gs[1, :])
        ax3.set_title('Car Space Timeline')
        
        # plot car path
        car_positions = np.array(timeline_data['car_positions'])
        ax3.plot(car_positions[:, 0], car_positions[:, 1], 'k-', alpha=0.3, label='Car Path')
        
        # plot gaze vectors at regular intervals
        for i, (gaze, driver_pos, car_dir, color) in enumerate(zip(
            timeline_data['car_space'], 
            timeline_data['driver_positions'],
            timeline_data['car_directions'],
            colors
        )):
            if i % 10 == 0:  # plot every 10th vector
                # plot driver position
                ax3.plot(driver_pos[0], driver_pos[1], 'o', color=color, markersize=3)
                
                # plot car direction
                ax3.arrow(driver_pos[0], driver_pos[1], 
                         car_dir[0]*0.5, car_dir[1]*0.5,
                         color=color, alpha=0.3, head_width=0.1)
                
                # plot gaze vector
                ax3.arrow(driver_pos[0], driver_pos[1],
                         gaze[0]*2, gaze[1]*2,
                         color=color, alpha=0.5, head_width=0.1)

        ax3.grid(True)
        ax3.axis('equal')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        # add colorbar to show time progression
        norm = mcolors.Normalize(vmin=min(timeline_data['timestamps']), vmax=max(timeline_data['timestamps']))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        plt.colorbar(sm, ax=ax3, label='Time (s)')

        os.makedirs(f'intermediary_data/transformed_encounter_data/participant_{p_num}/timeline_visualization', exist_ok=True)

        plt.tight_layout()
        plt.savefig(f'intermediary_data/transformed_encounter_data/participant_{p_num}/timeline_visualization/timeline_visualization_{p_num}_{intersection_type}.png')
        plt.close()

        encounter_df['yaw_continuous'] = np.unwrap(np.radians(encounter_df['CoG position/Yaw'])) * 180 / np.pi
        
        encounter_df.to_csv(f'intermediary_data/transformed_encounter_data/participant_{p_num}/transformed_data_{p_num}_{intersection_type}.csv')
        # print(f"for intersection {intersection_type}, {p_num}. intersect_count={intersect_count}")

        # *some of the graphs do not have the gaze vectors because the gaze is missing for the total duration of the encounter. 
        # ? these will most likely be removed from the final dataset since they are not yielding any useful information towards the eye tracking component