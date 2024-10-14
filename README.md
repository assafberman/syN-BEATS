# syN-BEATS for Robust Pollutant Forecasting
This repository contains the implementation of the syN-BEATS model for forecasting pollutant concentrations in data-limited environments. The syN-BEATS model is an ensemble adaptation of the N-BEATS architecture, designed to improve forecasting accuracy by integrating weak and strong learners.

This work is based on the research paper:
*syN-BEATS for robust pollutant forecasting in data-limited context published in Environmental Monitoring and Assessment (2024).*

## Table of Contents
* [Project Overview](#Project-Overview)
* [Requirments](#Requirments)
* [Installation](#Installation)
* [Project Structure](#Project-Structure)

# Project Overview
The syN-BEATS model is an ensemble of multiple N-BEATS models with different configurations to improve forecasting robustness in scenarios where data is limited. The model leverages weak learners to capture broad patterns and strong learners to fine-tune the predictions, making it particularly useful for regions with limited air quality and meteorological data.

## Key Features:
Ensemble of weak and strong learners.
Uses Bayesian optimization for fine-tuning ensemble weights.
Designed for pollutant forecasting with minimal input data from monitoring stations.
The datasets used in the paper come from meteorological and air quality stations, with specific pollutants including nitrogen oxide (NO), nitrogen dioxide (NO<sub>2</sub>), nitrogen oxide compounds (NOx), ozone (O<sub>3</sub>), and PM<sub>2.5</sub>.

# Requirements
* Python 3.8+
* The following Python libraries:
  * torch
  * numpy
  * pandas
  * matplotlib
  * darts (time series forecasting library)
  * scikit-learn
 
# Installation
To get started, clone the repository and install the necessary dependencies:
```
git clone https://github.com/assafberman/syn-beats.git
cd syn-beats
pip install -r requirements.txt
```

# Project Structure
```
├── main.py               # Script for training and testing syN-BEATS
├── nbeats.py             # Core implementation of N-BEATS and syN-BEATS
├── data/                 # Directory to store input data
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```
