import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm

window_size = int(input("Enter the window size: ") or 1)

RANDOM_STATE = 42
BINS = 10

# load collision summary file
collision_summary = pd.read_csv('summary_files/collision_summary.csv')

# bin index increments using modulo and only when participant is added to the bin, respective to c/nc
nc_i = 0
c_i = 0

# create the bin directories in processed data
for b in range(0,BINS):
    os.makedirs(f'data/processed_data/binned_data/bin_{b}',exist_ok=True)

    # cleanup existing files in bins
    for file in os.listdir(f'data/processed_data/binned_data/bin_{b}'):
        os.remove(os.path.join(f'data/processed_data/binned_data/bin_{b}',file))

collision_summary = collision_summary.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

def create_windowed_data(df, window_size):
    """Create windowed data from input dataframe"""
    windowed_rows = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        # Flatten the window into a single row
        flat_row = window.values.flatten()
        windowed_rows.append(flat_row)
    
    # Create column names for the windowed dataframe
    columns = []
    for i in range(window_size):
        for col in df.columns:
            columns.append(f'{col}_t{i}')
    
    return pd.DataFrame(windowed_rows, columns=columns)

# add participants to the bins
for i in tqdm(range(0,len(collision_summary)),colour='green'):
    p_num = collision_summary['p_num'][i]
    global_collision_flag = collision_summary['global_collision_flag'][i]
    intersection_type = collision_summary['intersection_type'][i]

    # if the participant has a collision, add to bin and update the c_i index
    if global_collision_flag:
        src = f'data/intermediary_data/labeled_data/participant_{p_num}/labeled_data_{p_num}_{intersection_type}_c.csv'
        dst = f'data/processed_data/binned_data/bin_{c_i}/labeled_data_{p_num}_{intersection_type}_c.csv'
        
        if os.path.isfile(src):
            df = pd.read_csv(src)
            windowed_df = create_windowed_data(df, window_size=window_size)
            windowed_df.to_csv(dst, index=False)
        c_i = (c_i + 1) % BINS
    else:
        # if the participant has no collision, add to bin and update the nc_i index
        src = f'data/intermediary_data/labeled_data/participant_{p_num}/labeled_data_{p_num}_{intersection_type}_nc.csv'
        dst = f'data/processed_data/binned_data/bin_{nc_i}/labeled_data_{p_num}_{intersection_type}_nc.csv'
        
        if os.path.isfile(src):
            df = pd.read_csv(src)
            windowed_df = create_windowed_data(df, window_size=window_size)
            windowed_df.to_csv(dst, index=False)
        nc_i = (nc_i + 1) % BINS