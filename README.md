# Traffic Event Prediction System

This project aims to predict traffic events based on historical traffic data, weather conditions, and daylight information. The system uses machine learning models to forecast potential traffic incidents in different cities.

## Project Overview

The project consists of two main steps:
1. Data collection and preprocessing
2. Event prediction using LSTM deep learning models

## Requirements

- Python 3.12+
- TensorFlow with CUDA support (preferred)
- Various Python packages (see requirements.txt)

## Setup and Execution Instructions

1. Clone the repository to your directory on the cluster
2. Run the project using SLURM:
    ```bash
    sbatch run_files.sh
    ```

The script will:
- Create a virtual environment if it doesn't exist
- Install TensorFlow with CUDA support
- Install all required dependencies
- Download necessary data files
- Execute the prediction notebook

Note: The process may take some time depending on the dataset size and available computing resources. Check the logfile for execution progress and results.

## Data Sources

- **Traffic Events**: Downloaded from [US Traffic 2016-2020 dataset](https://smoosavi.org/datasets/lstw)
- **Sunlight Data**: Downloaded from [sunrise-sunset.org](https://sunrise-sunset.org/)
- **Weather Data**: Retrieved dynamically from Copernicus or Oikolab API (supported but not used by the author) based on geographic coordinates

## Project Structure

- `1_CreateInput.ipynb`: Data collection and preprocessing notebook
- `2_PredictEvent.ipynb`: Event prediction and model training notebook
- `common.py`: Shared configuration settings
- `run_files.sh`: Script to run all tasks
- `requirements.txt`: Python dependencies
- `prod.env`: Environment variables for API access

## Cities Covered

The system currently supports the following cities:
- Austin, TX
- Los Angeles, CA
- New York City, NY

## Model Architecture

The project uses LSTM (Long Short-Term Memory) neural networks for prediction. The model architecture includes:
- LSTM layer with 64 units
- Dense output layer with softmax activation

## Workflow

1. Run `1_CreateInput.ipynb` to:
    - Load traffic event data
    - Fetch sunlight information
    - Download weather data
    - Construct feature vectors for city-geohash pairs

2. Run `2_PredictEvent.ipynb` to:
    - Vectorize the data
    - Train LSTM models
    - Evaluate model performance

## Output

The model produces:
- Trained model files (`.h5`) in the `data/output` directory
- Classification reports in the `data/print` directory

## Analysis

All of my analysis using the 3/1 year dataset can be found under the output folder

## Acknowledgements

- [US Traffic 2016-2020 dataset](https://smoosavi.org/datasets/lstw) for traffic event data
- [sunrise-sunset.org](https://sunrise-sunset.org/) for sunlight data
- Copernicus for weather data
