""" This is the sixth script in the data processing pipeline. 
From the output of the labeled data, this script splits the data into training and testing sets.
This is done ahead of the machine learning training process to have a consistent train-test split, and keep records for participants together."""

# ? Do I need to keep encounters from each participant together or just the data in the encounters together in the train-test split

import pandas as pd
import numpy as np
from tqdm import tqdm

# load collision summary file
collision_summary = pd.read_csv('summary_files/collision_summary.csv')