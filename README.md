# Predicting Vehicle Collisions in Turn-Across-Path Hazard Events From Driving And Eye-Tracking Data

This ML project aims to create and design a ML model capable of predicting vehicle collisions in simulated turn-across-path hazard events, using both driving data obtained from a driving simulator, and eye-tracking data to provide more insight into driving behaviour and patterns.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Data Sources](#data-sources)
- [Data Processing Workflow](#data-processing-workflow)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training & Testing](#model-training--testing)
- [Results](#results)
- [Contributing](#contributing)
- [Future Work](#future-work)
- [License](#license)

## Project Overview  

### Goal

The goal of this project is to predict driver behaviour using eye-tracking and driving data

### Motivation

This project aims to solve several problems in the field of driving. As AI becomes more prevalent, the rate at which decisions are made while driving is increased exponentially. If an AI makes an error that requires human intervention, the time required to act may be too late. Therefore, this project aims to predict driving collisions and hazard impacts before they occur to allow human drivers to override the AI system before consequences arise. This work is also important in vehicles without self-driving capabilities, as it can help enhance driving performance and reduce the number of collisions in these specific turn-across-path hazard scenarios

### Current Status

This project is in-progress as part of research for a MASc. Eng + AI

---

## Directory Structure

    /multisourse_ML_research/  
    ├──data
    │ ├──raw_data/  
    │ ├──intermediary_data/  
    │ └── processed_data/  
    ├── summary_files/  
    ├── code/  
    │ ├── data_processing/  
    │ ├── ml_training/  
    │ └── utils/  
    ├── models/  
    ├── results/  
    ├── documentation/  
    ├── README.md  

- `raw_data/`: Contains the raw data from the driving simulator, and the raw data from the eye-tracking (Oktal Driving Simulator, Tobii Eye Tracking)
- `intermediary_data/`: Contains the data as it flows through the preprocessing stages
- `processed_data/`: Holds data prepared for model training and testing
- `code/`: Python scripts for data processing and machine learning
- `models/`: Trained models and their configurations
- `results/`: Results from training and testing

---

## Data Sources

Current dataset follows the TCPS 2 guidelines on participant privacy

### Data Description

- **Data Sources:** The driving data is from the University of Guelph Oktal driving simulator located in the DRiVE lab. The driving simulator offers 300 degrees of surrounding driving environment complete with rear view display and side view mirrors. The driving environment is created and designed to analyze specific hazard conditions known as turn-across-path hazards. Hazard vehciles drive in-front of the path of the participant either from the opposite side of the lane, or from the right side at intersections. The intersections are comprised of T and 4 way intersections with stop-signs and traffic signals.

The eye-tracking data is from Tobii Pro 3 eye-tracking glasses. The participant drivers were instructed to wear the glasses in order to track eye movement, to ultimately analyze for potential driving behvaiours and patterns that could predict vehicle collisions.

- **Format:** The format of the raw eye-tracking data is .xlsx format. The driving data is in the form of .csv files. Each of the features are contained in a single column withing the various files. The driving data and the eye tracking data are on separate clocks, and they are synchronized to the closest second by human analysis.

### Data Preprocessing  

- **Preprocessing Stages:**

1. Encounters are extracted first to an intermediary dataset (`extracted_encounters_#`). The extracted encounters are determined by calculating the time when the driver vehicle is closest to the intersection of each respective hazard vehicle depending on the hazard order for the given drive. From that closest time, and backwards by an input amount (default is 5 seconds), the length of the input time is extracted and stored separately
2. The eye tracking data is the processed from the encounters. The algorithm that is applied is the I-VT threshold method which determines if the speed of the eye movement is above a certain threshold - the movement is classified as either  afixation or saccade and is labeled accordingly in the data. There are several different parameters that can be adjusted but defaults are set up for optimal use based on literature from Tobii for implementation of the algorithm
3. Next, each of the encoutners that are extracted have the eye tracking and the driving data merged together. First the datasets are aligned in time based on offsets found in `data_summary.xlsx`, and then they are merged together. Next, the missing values from encoutner extractions are filled using spline interpolation with a default spline order of 5 to fill in the gaps. The merged encounters are then stored in `encounter_data`
4. Each hazard encounter is then processed using `encounter_processing.py` to obtain the eye tracking coordinates from the 3D space that the glasses operate in and convert it to 2D coordinates relative to the driver in the driving environment. To do this, the IMU data is applied and the coordinates are translated to where the driver is for a given record. The data is also processed to determine fixations and saccades. The eye gaze direction vector is averaged between the left and right eye, and is then processed to see if the direction vector intersects a radius around the hazard vehicle (direction of the vehicle is not in the dataset to check actual bounding boxes). The output from this script is fed to `transformed_encounter_data` along with visuals to plot the gaze vector and the driver vehicle accordingly
5. The next stage of the preprocessing is to label the dataset. For each encounter file, the default time to collision is set to an infinite value. The distance from the respective hazard vehicle to the driver vehicle is obtained. A threshold is applied to filter against the distance measurement. If the distance is under the threshold, a label column is set to indicate collision for all of the records in the encounter (and no collision if no collision occured). This is defined as the global collision flag. Next, the time at which the vehicles are under the threshold is obtained. A time to collision column is created to count down until this point. Since any information after this first collision will be irrelavent as the collision has now occured, the remaining records are removed from the encounter. This will be the regression label
6. The labeled dataset is then processed into windows, which are configurable depending on the type of windowing desired. This way a signle record has information from multiple timestamps. The labeled and windowed data is split into a default set of 10 bins. This will make training and testing more consistent. The algorithm is simple, and ensures that each bin has a balance of c vs nc. The encounters are also shuffled and maintained using a random_state value for consistency. This way the split into bins is randomized

- **Scripts (respective to stages above):**

1. `encounter_extraction.py`
2. `eye_tracking_processing.py`
3. `encounter_merge.py`
4. `encounter_processing.py`
5. `data_labeler.py`
6. `encounter_windowing.py`

## Data Processing Workflow

Provided is a brief summary of the machine learning pipeline for this project:

1. **Raw Data:** Data is loaded from `raw_data/`.
2. **Cleaning and Transformation:** Performed by scripts in `data_processing/`, `eye_tracking_processing/` and `driving_processing/`
3. **Training/Testing Split:** Data is manually split into training and testing sets.

---

## Machine Learning Pipeline  

### Features Used  

- **Feature Overview:** In-progress
  
### Models

- **Model Types:** In-progress
- **Hyperparameters:** In-progress

### Scripts

- **Training Scripts:** Training scripts are located  in `/code/ml_training/train_model.py`
- **Testing Scripts:** testing scripts are located in `/code/ml_training/test_model.py`

---

## Installation

### Requirements

- **Python Version:** The project is written and tested in Python version **3.12.8**. Other versions may not be supported
- **Dependencies:** This project requires the following lbiraries  to be installed:
  - pandas  
  - numpy
  - os
  - shutil
  - tqdm
  - matplotllib
  - scipy
  - sklearn

## Usage

### How to Run the Project

1. **Data Processing:**  
    Several scripts are written to be used in a specfic order to process the set of data. Refer to the **Data Processing Workflow** header. All of the scripts can be run simply by using `py script_name.py`.
2. **Training the Model:**  
3. **Testing the Model:**  

### Configuration

In-progress

---

## Model Training & Testing

### Training Process

- **Training Script:** In-progress
- **Data Used:** In-progress

### Testing Process

- **Testing Script:** In-progress
- **Metrics:** In-progress

---

## Results

### Model Performance

In-progress

- **Training Results:** In-progress
- **Testing Results:** In-progress

### Comparison of Models

In-progress

---

## Contributing

No external contributions are being accepted at this time.

---

## Future Work

- **Feature Improvements:** In-progress
- **Model Extensions:** In-progress

---

## License

[MIT LICENSE](LICENSE)
