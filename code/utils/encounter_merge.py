""" This script is the third script in the data processing pipeline.
It takes the eye tracking data from processed_eye_tracking and merges it with the driving and imu data from the same participant
After each merge of the files, the data is interpolated using spline interpolation.
Each participant has 4 encounters, one for each intersection type"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# pd.set_option('display.max_columns',None)

splineOrder = input("Spline order = ") or 5

data_summary = pd.read_excel('summary_files/data_summary.xlsx')

for summary_index, summary_row in tqdm(data_summary.iterrows(), total=len(data_summary),colour='green'):
    hazard_order = "order_" + str(summary_row["HazardOrder"])
    p_num = summary_row["participant_id"]

    intersection_locations = pd.read_excel('summary_files/intersection_locations.xlsx',sheet_name=hazard_order)

    for current_hazard in range(0,4):
        intersection_type = (list(intersection_locations)[current_hazard*2]).split("_")[2]
        
        drive_encounter = pd.read_csv(f'data/intermediary_data/extracted_encounters_5/participant_{p_num}/driving_encounter_{p_num}_{intersection_type}.csv')
        drive_encounter.set_index(drive_encounter.columns[0], inplace=True)
        
        imu_encounter = pd.read_csv(f'data/intermediary_data/extracted_encounters_5/participant_{p_num}/imu_encounter_{p_num}_{intersection_type}.csv')
        imu_encounter.set_index(imu_encounter.columns[1], inplace=True)
        imu_encounter.rename(columns={'Type':'Type_imu'},inplace=True)

        gaze_encounter = pd.read_csv(f'data/intermediary_data/processed_eye_tracking/participant_{p_num}/processed_gaze_{p_num}_{intersection_type}.csv')
        gaze_encounter.set_index(gaze_encounter.columns[1], inplace=True)
        gaze_encounter.rename(columns={'Type':'Type_gaze'},inplace=True)

        # merge the three datasets together into a single dataframe using the timestamp as the index
        merged_df = drive_encounter.join([gaze_encounter,imu_encounter], how='outer')
        merged_df = merged_df.infer_objects(copy=False)
        merged_df.index = pd.to_numeric(merged_df.index)
        merged_df.sort_index(inplace=True)

        # interpolate missing values using spline interpolation (default is order 5 spline)
        # fill remaining values (front and end of dataframe columns using linear interpolation)
        # * there are still some columns in some of the dataframes that are entirely empty and should be removed before training any models
        # ! there is a suppressed warning with the interpolation. some of the values are too small and so the spline is approximated
        # print(merged_df.shape)
        # merge the three datasets together into a single dataframe using the timestamp as the index
        merged_df = drive_encounter.join([gaze_encounter,imu_encounter], how='outer')
        merged_df = merged_df.infer_objects(copy=False)

        # Clean the index
        merged_df = merged_df[merged_df.index.notna()]  # Remove rows with NaN indices
        merged_df.index = pd.to_numeric(merged_df.index, errors='coerce')  # Convert index to numeric
        merged_df = merged_df[merged_df.index.notna()]  # Remove any rows where index conversion created NaNs
        merged_df.sort_index(inplace=True)

        # interpolate missing values using spline interpolation (default is order 5 spline)
        merged_df.interpolate(method='spline',order=splineOrder,inplace=True)
        merged_df.interpolate(method='ffill', inplace=True) # fill in remaining NAN values
        merged_df.interpolate(method='bfill', inplace=True) # fill in remaining NAN values
        # print(merged_df.head)
        # print(merged_df.isna().sum())

        os.makedirs(f'data/intermediary_data/encounter_data/participant_{p_num}', exist_ok=True)
        merged_df.index.name = 'Timestamp'

        # for c, col in enumerate(merged_df.columns):
        #     print(c,col)

        merged_df.drop(columns=["Type_imu"],inplace=True) # remove unnessecary columns
        merged_df.to_csv(f'data/intermediary_data/encounter_data/participant_{p_num}/encounter_data_{p_num}_{intersection_type}.csv',index=True)