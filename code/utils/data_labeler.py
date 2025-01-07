import pandas as pd
import numpy as np
import os
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

collision_summary = pd.DataFrame(columns=['p_num','intersection_type','global_collision_flag','time_of_collision'])

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary), colour='green'):
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    hazard_label_order = "order_" + str(summary_row["HazardOrder"]) + "_hazards"
    p_num = summary_row["participant_id"]

    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx', sheet_name=hazard_order)

    for current_hazard in range(0, 4):
        first_collision_time = np.inf
        intersect_count = 0
        intersection_type = (list(intersection_locations)[current_hazard*2]).split("_")[2]
        encounter_df = pd.read_csv(f'intermediary_data/transformed_encounter_data/participant_{p_num}/transformed_data_{p_num}_{intersection_type}.csv')
        encounter_df.columns = encounter_df.columns.str.strip()

        # 1. Label data with global collision flag to indicate if driver had collision

        # Find which hazard number is present in the columns
        hazard_columns = [col for col in encounter_df.columns if col.startswith('hazard_') and col.endswith('_x')]
        if not hazard_columns:
            raise ValueError("No hazard position columns found in dataframe")

        # Extract the hazard number from the first found column (e.g., 'hazard_1_x' -> '1')
        hazard_num = hazard_columns[0].split('_')[1]

        # Calculate Euclidean distance using the detected hazard number
        encounter_df['global_collision_flag'] = (
            ((encounter_df['driver_position_x'] - encounter_df[f'hazard_{hazard_num}_x'])**2 + 
            (encounter_df['driver_position_y'] - encounter_df[f'hazard_{hazard_num}_y'])**2)**0.5 
            < COLLISION_DISTANCE
        ).any()

        # 2. Label data with regression to first collision flag, and drop rows after the collision point
        if encounter_df['global_collision_flag'].any():
            # Find the first timestamp where collision occurred
            first_collision_time = encounter_df.loc[
                ((encounter_df['driver_position_x'] - encounter_df[f'hazard_{hazard_num}_x'])**2 + 
                (encounter_df['driver_position_y'] - encounter_df[f'hazard_{hazard_num}_y'])**2)**0.5 
                < COLLISION_DISTANCE, 'Timestamp'
            ].min()

            # Keep only rows up to the first collision
            encounter_df = encounter_df[encounter_df['Timestamp'] <= first_collision_time]

            # Calculate time to collision for each row
            # Positive values before collision, zero at collision
            encounter_df['time_to_collision_flag'] = first_collision_time - encounter_df['Timestamp']
            collision_total += 1
        else:
            encounter_df['time_to_collision_flag'] = np.inf
            ncollision_total += 1

        file_label = 'c' if encounter_df['global_collision_flag'].any() else 'nc'
        
        collision_row = {'p_num': p_num, 'intersection_type': intersection_type, 'global_collision_flag': encounter_df['global_collision_flag'].any(), 'time_of_collision': first_collision_time, 'record_count': len(encounter_df)}
        collision_summary = collision_summary._append(collision_row, ignore_index=True)

        os.makedirs(f'intermediary_data/labeled_data/participant_{p_num}', exist_ok=True)
        encounter_df.to_csv(f'intermediary_data/labeled_data/participant_{p_num}/labeled_data_{p_num}_{intersection_type}_{file_label}.csv')

print(f'collisions={collision_total};ncollsions={ncollision_total}')
collision_summary.to_csv(f'summary_files/collision_summary.csv')