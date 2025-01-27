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

        # Clean the index
        merged_df = merged_df[merged_df.index.notna()]  # Remove rows with NaN indices
        merged_df.index = pd.to_numeric(merged_df.index, errors='coerce')  # Convert index to numeric
        merged_df = merged_df[merged_df.index.notna()]  # Remove any rows where index conversion created NaNs
        merged_df.sort_index(inplace=True)

        # Separate numeric and categorical columns for appropriate interpolation
        numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
        categorical_columns = merged_df.select_dtypes(exclude=[np.number]).columns

        # interpolate numeric columns using spline interpolation
        merged_df[numeric_columns] = merged_df[numeric_columns].interpolate(method='spline', order=splineOrder)
        merged_df[numeric_columns] = merged_df[numeric_columns].fillna(method='ffill')
        merged_df[numeric_columns] = merged_df[numeric_columns].fillna(method='bfill')

        # Handle categorical columns with forward/backward fill
        merged_df[categorical_columns] = merged_df[categorical_columns].fillna(method='ffill')
        merged_df[categorical_columns] = merged_df[categorical_columns].fillna(method='bfill')

        os.makedirs(f'data/intermediary_data/encounter_data/participant_{p_num}', exist_ok=True)
        merged_df.index.name = 'Timestamp'

        merged_df.drop(columns=["Type_imu"],inplace=True) # remove unnessecary columns
        
        # Check for any remaining null values
        # ! Eye tracking for the following is either missing completely or onyl contains timestamps and nothing else
        # ! This issue is resolved later since the processing is built off the participant summary list and removing participant data will cause issues
        # * Participant 2 - lthalf
        # * Participant 9 - ltsignal, ltminor, lthalf
        # * Participant 19 - all
        # * Participant 26 - lthalf, ltsignal, ltmajor
        # * Participant 49 - ltmajor
        # * Participant 51 - lthalf 
        # if merged_df.isna().sum().any():
        #     print(f"\nRemaining null values for participant {p_num}, intersection {intersection_type}:")
        #     null_columns = merged_df.columns[merged_df.isna().any()].tolist()

        merged_df.to_csv(f'data/intermediary_data/encounter_data/participant_{p_num}/encounter_data_{p_num}_{intersection_type}.csv',index=True)