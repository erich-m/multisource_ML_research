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

def normalize(v):
    return v / np.linalg.norm(v)

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary), colour='green'):
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    p_num = summary_row["participant_id"]

    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx', sheet_name=hazard_order)

    for current_hazard in range(0, 4):
        intersection_type = (list(intersection_locations)[current_hazard*2]).split("_")[2]
        encounter_df = pd.read_csv(f'intermediary_data/encounter_data/participant_{p_num}/encounter_data_{p_num}_{intersection_type}.csv')
        
        # Store vectors for timeline visualization
        timeline_data = {
            'glasses_space': [],
            'global_space': [],
            'car_space': [],
            'timestamps': [],
            'driver_positions': [],
            'car_positions': [],
            'car_directions': []
        }
        
        for idx in encounter_df.index:
            # Original transformation code (unchanged until visualization)
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
            
            down = normalize(accel)
            east = normalize(np.cross(down, mag))
            north = normalize(np.cross(east, down))
            
            R_imu_to_global = np.vstack((east, down, north)).T
            
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
            
            combined_gaze = normalize((left_gaze + right_gaze) / 2)
            combined_gaze_global = R_imu_to_global @ combined_gaze
            
            # Modified: Project to horizontal plane using X and Z components from glasses space
            # Since Z is forward and X is left in glasses space, we want to use these for the car's X-Y plane
            combined_gaze_horizontal = normalize(np.array([combined_gaze[2], -combined_gaze[0]]))  # [forward, left] components
            
            yaw = np.radians(encounter_df.loc[idx, 'CoG position/Yaw'])
            
            R_car = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            car_position = np.array([
                encounter_df.loc[idx, 'CoG position/X'],
                encounter_df.loc[idx, 'CoG position/Y']
            ])
            
            driver_offset = np.array([-1.5, -0.5])
            driver_offset_rotated = R_car @ driver_offset
            driver_position = car_position + driver_offset_rotated
            
            combined_gaze_car = R_car.T @ combined_gaze_horizontal
            
            # Store vectors and positions for visualization
            timeline_data['glasses_space'].append(combined_gaze)
            timeline_data['global_space'].append(combined_gaze_horizontal)
            timeline_data['car_space'].append(combined_gaze_car)
            timeline_data['timestamps'].append(encounter_df.loc[idx, 'Timestamp'])
            timeline_data['driver_positions'].append(driver_position)
            timeline_data['car_positions'].append(car_position)
            timeline_data['car_directions'].append(R_car @ np.array([1, 0]))
            
            # Store transformed vectors in DataFrame
            encounter_df.loc[idx, 'combined_gaze_car_x'] = combined_gaze_car[0]
            encounter_df.loc[idx, 'combined_gaze_car_y'] = combined_gaze_car[1]
            encounter_df.loc[idx, 'driver_position_x'] = driver_position[0]
            encounter_df.loc[idx, 'driver_position_y'] = driver_position[1]

        # Create timeline visualization
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)

        # Plot 1: Glasses Space Timeline (3D)
        ax1 = plt.subplot(gs[0, 0], projection='3d')
        ax1.set_title('Glasses Space Timeline')
        
        # Create color gradient based on time
        timestamps_normalized = np.array(timeline_data['timestamps'])
        timestamps_normalized = (timestamps_normalized - timestamps_normalized.min()) / (timestamps_normalized.max() - timestamps_normalized.min())
        colors = plt.cm.viridis(timestamps_normalized)
        
        for i, (gaze, color) in enumerate(zip(timeline_data['glasses_space'], colors)):
            if i % 10 == 0:  # Plot every 10th vector to avoid overcrowding
                ax1.quiver(0, 0, 0, *gaze, color=color, alpha=0.5)
        
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Plot 2: Global Space Timeline
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

        # Plot 3: Car Space Timeline with Path
        ax3 = plt.subplot(gs[1, :])
        ax3.set_title('Car Space Timeline')
        
        # Plot car path
        car_positions = np.array(timeline_data['car_positions'])
        ax3.plot(car_positions[:, 0], car_positions[:, 1], 'k-', alpha=0.3, label='Car Path')
        
        # Plot gaze vectors at regular intervals
        for i, (gaze, driver_pos, car_dir, color) in enumerate(zip(
            timeline_data['car_space'], 
            timeline_data['driver_positions'],
            timeline_data['car_directions'],
            colors
        )):
            if i % 10 == 0:  # Plot every 10th vector
                # Plot driver position
                ax3.plot(driver_pos[0], driver_pos[1], 'o', color=color, markersize=3)
                
                # Plot car direction
                ax3.arrow(driver_pos[0], driver_pos[1], 
                         car_dir[0]*0.5, car_dir[1]*0.5,
                         color=color, alpha=0.3, head_width=0.1)
                
                # Plot gaze vector
                ax3.arrow(driver_pos[0], driver_pos[1],
                         gaze[0]*2, gaze[1]*2,
                         color=color, alpha=0.5, head_width=0.1)

        ax3.grid(True)
        ax3.axis('equal')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        # Add colorbar to show time progression
        norm = mcolors.Normalize(vmin=min(timeline_data['timestamps']), vmax=max(timeline_data['timestamps']))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        plt.colorbar(sm, ax=ax3, label='Time (s)')

        os.makedirs(f'intermediary_data/transformed_encounter_data/participant_{p_num}/timeline_visualization', exist_ok=True)

        plt.tight_layout()
        plt.savefig(f'intermediary_data/transformed_encounter_data/participant_{p_num}/timeline_visualization/timeline_visualization_{p_num}_{intersection_type}.png')
        plt.close()

        encounter_df['yaw_continuous'] = np.unwrap(np.radians(encounter_df['CoG position/Yaw'])) * 180 / np.pi
        
        encounter_df.to_csv(f'intermediary_data/transformed_encounter_data/participant_{p_num}/transformed_data_{p_num}_{intersection_type}.csv')