""" This is the fifth script in the data processing pipeline. 
Once the data has been processed, merged and interpolated, the hazard vehicle and the participant vehicle locations are compared to check if collision occured
If there is a collision, the column called global_collision_flag is set to true to indicate that there is a collision in the encounter
There is also a time to collision that is calculated by taking the time of collision as time 0 and counting down using the timestamp
This label can be used for regression. if there is no collision, the flag previous is set to false and the time_to_collision is set to inf """
import pandas as pd
import numpy as np

import os
import sys
from tqdm import tqdm

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)

data_summary = pd.read_excel('summary_files/data_summary.xlsx')

# define collision threshold for hazard collision detection in meters
COLLISION_DISTANCE = 2.609

# tally the number of collisions
collision_total = 0
ncollision_total = 0

collision_summary = pd.DataFrame(columns=['p_num', 'intersection_type', 'global_collision_flag', 'time_of_collision'])

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary), colour='green'):
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    hazard_label_order = "order_" + str(summary_row["HazardOrder"]) + "_hazards"
    p_num = summary_row["participant_id"]

    participant_folder = f'data/intermediary_data/labeled_data/participant_{p_num}'
    os.makedirs(participant_folder, exist_ok=True)

    # Remove existing labeled data files for the participant
    for file in os.listdir(participant_folder):
        if file.endswith('.csv'):
            os.remove(os.path.join(participant_folder, file))

    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx', sheet_name=hazard_order)

    for current_hazard in range(0, 4):
        first_collision_time = np.inf
        intersect_count = 0
        intersection_type = (list(intersection_locations)[current_hazard * 2]).split("_")[2]
        encounter_df = pd.read_csv(
            f'data/intermediary_data/transformed_encounter_data/participant_{p_num}/transformed_data_{p_num}_{intersection_type}.csv'
        )
        encounter_df.columns = encounter_df.columns.str.strip()

        # global_collision_flag: true for all the records of the encounter if there is a collision between the driver and the hazard vehicle, otherwise false

        # 1. Label data with global collision flag to indicate if driver had collision
        encounter_df['global_collision_flag'] = (
            ((encounter_df['driver_position_x'] - encounter_df[f'CoG position/X.hazard'])**2 +
             (encounter_df['driver_position_y'] - encounter_df[f'CoG position/Y.hazard'])**2)**0.5
            < COLLISION_DISTANCE
        ).any()

        # time_to_collision_flag: if there is collision, set to the time in seconds until the driver vehicle collides with the hazard, otherwise set to inf
        # 2. Label data with regression to first collision flag, and drop rows after the collision point
        if encounter_df['global_collision_flag'].any():
            first_collision_time = encounter_df.loc[
                ((encounter_df['driver_position_x'] - encounter_df[f'CoG position/X.hazard'])**2 +
                 (encounter_df['driver_position_y'] - encounter_df[f'CoG position/Y.hazard'])**2)**0.5
                < COLLISION_DISTANCE, 'Timestamp'
            ].min()

            encounter_df = encounter_df[encounter_df['Timestamp'] <= first_collision_time]
            encounter_df['time_to_collision_flag'] = first_collision_time - encounter_df['Timestamp']
            collision_total += 1
        else:
            encounter_df['time_to_collision_flag'] = sys.float_info.max
            ncollision_total += 1

        file_label = 'c' if encounter_df['global_collision_flag'].any() else 'nc'

        collision_row = {'p_num': p_num, 'intersection_type': intersection_type,
                         'global_collision_flag': encounter_df['global_collision_flag'].any(),
                         'time_of_collision': first_collision_time, 'record_count': len(encounter_df)}
        collision_summary = collision_summary._append(collision_row, ignore_index=True)
        
        encounter_df.drop(columns=['driver_position_x','driver_position_y'],inplace=True)
        # if encounter_df.isna().sum().any():
        #     print(encounter_df.isna().sum())
        #     print("Columns with NA values:", encounter_df.columns[encounter_df.isna().any()].tolist())
        encounter_df.to_csv(f'{participant_folder}/labeled_data_{p_num}_{intersection_type}_{file_label}.csv', index=False)

print(f'collisions={collision_total};ncollisions={ncollision_total}')
collision_summary.to_csv('summary_files/collision_summary.csv',index_label='index')
