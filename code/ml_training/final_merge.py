""" This is the final script that gets run before the data is analyzed for the machine learning. 
Once the data has been converted into bins, this script merges the data from each bin and saves it as a new file in processed data"""

import pandas as pd
from tqdm import tqdm
import os

window_size = int(input("Enter the window size: ") or 1)
binned_data_path = f'data/processed_data/binned_data_{window_size}'

if os.path.isdir(binned_data_path):# make sure the path exists
    for b, bin_dir in tqdm(enumerate(os.listdir(binned_data_path)), colour='green'):
        bin_path = os.path.join(binned_data_path, bin_dir)
        if os.path.isdir(bin_path):
            df_list = []
            for file in os.listdir(bin_path):
                file_path = os.path.join(bin_path, file)
                if os.path.isfile(file_path) and file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    df_list.append(df)
            if df_list:
                os.makedirs(f'data/processed_data/merged_data_{window_size}', exist_ok=True)

                merged_df = pd.concat(df_list, ignore_index=True)
                merged_df.to_csv(f'data/processed_data/merged_data_{window_size}/bin_{b}.csv', index=False)

    # Merge bin_0 through bin_6 for training data
    training_dfs = []
    for i in range(7):
        file_path = f'data/processed_data/merged_data_{window_size}/bin_{i}.csv'
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            training_dfs.append(df)
    if training_dfs:
        os.makedirs(f'data/processed_data/final_data_{window_size}', exist_ok=True)
        training_df = pd.concat(training_dfs, ignore_index=True)
        training_df.to_csv(f'data/processed_data/final_data_{window_size}/training.csv', index=False)

    # Use bin_7 for testing data
    testing_file_path = f'data/processed_data/merged_data_{window_size}/bin_7.csv'
    if os.path.isfile(testing_file_path):
        testing_df = pd.read_csv(testing_file_path)
        testing_df.to_csv(f'data/processed_data/final_data_{window_size}/testing.csv', index=False)

    # Merge bin_8 and bin_9 for validation data
    validation_dfs = []
    for i in range(8, 10):
        file_path = f'data/processed_data/merged_data_{window_size}/bin_{i}.csv'
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            validation_dfs.append(df)
    if validation_dfs:
        validation_df = pd.concat(validation_dfs, ignore_index=True)
        validation_df.to_csv(f'data/processed_data/final_data_{window_size}/validation.csv', index=False)

