LightGBM-ST
Spatiotemporal LightGBM-Based Interpolation for GNSS Time Series with Missing Data
This repository provides the implementation of the LGBM-ST model, a spatiotemporal interpolation framework designed to reconstruct missing values in GNSS vertical displacement time series.
The proposed method integrates temporal features and spatial correlations between GNSS stations within a LightGBM-based machine learning model to improve the accuracy and robustness of missing data reconstruction.

This repository contains the datasets and scripts required to reproduce the main experiments presented in the study.

Repository Structure
LightGBM-ST
│
├── data
│   └── real_GNSS
│       ├── GSAX.csv
│       ├── GSDH.csv
│       ├── YNLJ.csv
│       └── YNYS.csv
│
├── src
│   ├── contrast_experiments.py
│   ├── ablation_experiments.py
│   └── spatial_correlation.py
│
└── README.md


Directory description

data/
Contains the GNSS vertical displacement datasets used in the experiments.

src/
Contains Python scripts implementing the interpolation model and experimental procedures.

README.md
Documentation describing the repository and how to reproduce the experiments.

Dataset

The repository includes GNSS vertical displacement time series from several stations used for interpolation experiments:

GSAX
GSDH
YNLJ
YNYS
All datasets are stored in:data/real_GNSS/

Each dataset is stored as a CSV file containing the displacement observations used for model training and evaluation.

Requirements

The implementation is based on Python.
Recommended Python version:Python >= 3.8

Required Python libraries include:
numpy
pandas
scikit-learn
lightgb
matplotlib
scipy

You can install the required packages using pip:
pip install numpy pandas scikit-learn lightgbm matplotlib scipy

Running the Experiments

The scripts provided in the src directory allow reproduction of the main experiments in the paper.

1. Contrast Experiments

To run the comparison experiments between LGBM-ST and other interpolation methods:
python src/contrast_experiments.py

These experiments evaluate the performance of different interpolation models under varying missing data rates.

2. Ablation Experiments

To analyze the contribution of different components in the LGBM-ST framework:
python src/ablation_experiments.py


This script performs ablation studies to assess the impact of spatiotemporal features on model performance.

3. Spatial Correlation Analysis

To compute spatial correlations between GNSS stations:

python src/spatial_correlation.py


This analysis helps quantify spatial relationships used as input features for the interpolation model.

Reproducibility

All scripts and datasets necessary for reproducing the core experimental results of the study are included in this repository.

By running the scripts provided in the src directory, users can reproduce the following analyses described in the manuscript:

interpolation performance comparison

ablation experiments

spatial correlation analysis between GNSS stations

The repository structure is designed to ensure that the results presented in the paper can be reproduced.
