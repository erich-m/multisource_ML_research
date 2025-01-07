""" This is the sixth script in the data processing pipeline. 
From the output of the labeled data, this script splits the data into bins.
This is done ahead of the machine learning training process to have a consistent train-test split, and keep records for participants together."""

import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm

BINS = 10

# load collision summary file
collision_summary = pd.read_csv('summary_files/collision_summary.csv')

# bin index increments using modulo and only when participant is added to the bin, respective to c/nc
nc_i = 0
c_i = 0

# create the bin directories in intermediary data and processed data
os.makedirs('data/intermediary_data/binned_data',exist_ok=True)
for b in range(0,BINS):
    os.makedirs(f'data/intermediary_data/binned_data/bin_{b}',exist_ok=True)
    os.makedirs(f'data/processed_data/binned_data/bin_{b}',exist_ok=True)

# add participants to the bins
for i in tqdm(range(0,len(collision_summary)),colour='green'):
    p_num = collision_summary['p_num'][i]
    global_collision_flag = collision_summary['global_collision_flag'][i]
    intersection_type = collision_summary['intersection_type'][i]

    # if the participant has a collision, add to bin and update the c_i index
    if global_collision_flag:
        src = f'data/intermediary_data/labeled_data/participant_{p_num}/labeled_data_{p_num}_{intersection_type}_c.csv'
        dst = f'data/intermediary_data/binned_data/bin_{c_i}/labeled_data_{p_num}_{intersection_type}_c.csv'
        
        if os.path.isfile(src):
            shutil.copyfile(src,dst)
        c_i = (c_i + 1) % BINS
    else:
        # if the participant has a collision, add to bin and update the nc_i index
        src = f'data/intermediary_data/labeled_data/participant_{p_num}/labeled_data_{p_num}_{intersection_type}_nc.csv'
        dst = f'data/intermediary_data/binned_data/bin_{nc_i}/labeled_data_{p_num}_{intersection_type}_nc.csv'
        if os.path.isfile(src):
            shutil.copyfile(src,dst)
        nc_i = (nc_i + 1) % BINS