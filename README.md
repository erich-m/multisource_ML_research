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

- **Goal:** The goal of this project is to predict driver behaviour using eye-tracking and driving data

### Motivation

- **Motivation:** This project aims to solve several problems in the field of driving. As AI becomes more prevalent, the rate at which decisions are made while driving is increased exponentially. If an AI makes an error that requires human intervention, the time required to act may be too late. Therefore, this project aims to predict driving collisions and hazard impacts before they occur to allow human drivers to override the AI system before consequences arise. This work is also important in vehicles without self-driving capabilities, as it can help enhance driving performance and reduce the number of collisions in these specific turn-across-path hazard scenarios

### Current Status

- **Current Status:** This project is in-progress as part of research for a MASc. Eng + AI

---

## Directory Structure

/multisourse_ML_research/ ├── raw_data/ ├── processed_data/ ├── summary_files/ ├── code/ │ ├── data_processing/ │ ├── ml_training/ │ └── utils/ ├── models/ ├── results/ ├── documentation/ ├── README.md

- `raw_data/`: Contains the raw data from the driving simulator, and the raw data from the eye-tracking (Oktal Driving Simulator, Tobii Eye Tracking)
- `processed_data/`: Holds data prepared for model training and testing
- `code/`: Python scripts for data processing and machine learning
- `models/`: Trained models and their configurations
- `results/`: Results from training and testing

---

## Data Sources

Current data files are hidden until TCPS guidelines are met.

### Data Description

- **Data Sources:** The driving data is from the University of Guelph Oktal driving simulator located in the DRiVE lab. The driving simulator offers 300 degrees of surrounding driving environment complete with rear view display and side view mirrors. The driving environment is created and designed to analyze specific hazard conditions known as turn-across-path hazards. Hazard vehciles drive in-front of the path of the participant either from the opposite side of the lane, or from the right side at intersections. The intersections are comprised of T and 4 way intersections with stop-signs and traffic signals.

The eye-tracking data is from Tobii Pro 3 eye-tracking glasses. The participant drivers were instructed to wear the glasses in order to track eye movement, to ultimately analyze for potential driving behvaiours and patterns that could predict vehicle collisions.

- **Format:** The format of the raw eye-tracking data is .xlsx format. The driving data is in the form of .csv files. Each of the features are contained in a single column withing the various files. The driving data and the eye tracking data are on separate clocks, and they are synchronized to the closest second by human analysis.

### Data Preprocessing  

- **Preprocessing Steps:** In-progress
- **Scripts:** In-progress

## Data processing Workflow

Provided is a brief summary of the machine learning pipeline for this project:

1. **Raw Data:** Data is loaded from `raw_data/`.
2. **Cleaning and Transformation:** Performed by scripts in `data_processing/`.
3. **Feature Extraction:** Relevant features are extracted and saved in `processed_data/extracted_features/`.
4. **Training/Testing Split:** Data is manually split into training and testing sets.

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

- **Python Version:** The project is written and tested in Python version 3.12.2. Other versions may not be supported
- **Dependencies:** In-progress

## Usage

### How to Run the Project

In-progress

1. **Data Processing:**  
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
