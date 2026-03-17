# LightGBM-ST

Code repository for the paper:

**LGBM-ST: An Interpretable Spatiotemporal Interpolation Model for GNSS Time Series with Missing Data**

This repository provides the implementation of the proposed **LightGBM-ST** framework for interpolating missing values in GNSS vertical displacement time series by incorporating both spatial and temporal information.

---

# Overview

GNSS time series often contain missing observations due to equipment failure, environmental interference, or data transmission issues. Accurate reconstruction of missing values is essential for geophysical analysis.

In this study, we propose **LightGBM-ST**, an interpretable spatiotemporal interpolation framework based on the LightGBM model. The proposed approach integrates:

- Temporal features of GNSS time series
- Spatial correlations between GNSS stations
- Machine learning based regression modeling

The framework is evaluated against several commonly used interpolation approaches and demonstrates improved robustness and prediction accuracy under different missing data scenarios.

---

# Repository Structure

The structure of this repository is organized as follows:

LightGBM-ST
│
├── data
│   └── real_GNSS
│       ├── GSAX.csv
│       ├── GSDH.csv
│       ├── YNLJ.csv
│       ├── YNYS.csv
│       └── README.md
├── results
├── src
│   ├── ablation_experiments.py
│   ├── contrast_experiments.py
│   └── spatial_correlation.py
│
├── README.md
└── requirements.txt

Description of key components:

**src/**  
Contains all scripts required to reproduce the experiments in the paper.

- `contrast_experiments.py`  
  Performs comparison experiments between the proposed LightGBM-ST model and baseline interpolation methods.

- `ablation_experiments.py`  
  Conducts ablation studies to analyze the contribution of different components of the LightGBM-ST framework.

- `spatial_correlation.py`  
  Computes spatial correlations between GNSS stations, which are used as spatial features in the interpolation model.

**data/**  
Contains the GNSS datasets used in the experiments.

**results/**  
Stores generated figures and experimental outputs.

---

# Dataset

The dataset used in this study consists of GNSS vertical displacement time series collected from multiple stations.

Each dataset file contains:

- Observation time
- Vertical displacement values
- Station information

Real GNSS datasets are included to evaluate the robustness of the proposed model.

---

# Requirements

Python environment:

Python >= 3.12


Required Python packages:

numpy
pandas
scikit-learn
scipy
lightgbm
matplotlib
seaborn

Install dependencies using:

pip install -r requirements.txt

---

# Running the Experiments

The experimental pipeline of the proposed LightGBM-ST framework consists of four main steps: spatial correlation analysis, feature construction, comparison experiments, and ablation analysis.

---

### Step 1: Spatial Correlation Analysis

python src/SpatialCorrelation.py

This step calculates the spatial correlations between GNSS stations based on their vertical displacement time series. Stations with strong correlations to the target station are identified and selected as candidate spatial features for the interpolation model.

---

### Step 2: Spatiotemporal Feature Construction

Based on the selected correlated stations, spatial and temporal features are constructed to represent the spatiotemporal characteristics of GNSS time series. These features are then used as input variables for the LightGBM-ST interpolation model.

---

### Step 3: Comparison Experiments

python src/ContrastExperiments.py

This script evaluates the performance of the proposed LightGBM-ST model and several baseline interpolation methods under different missing data scenarios.

The following evaluation metrics are used:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Correlation Coefficient (R)

---

### Step 4: Ablation Experiments

python src/LGBMSTAblation.py

This script performs ablation studies to analyze the contributions of different components in the LightGBM-ST framework, particularly the impact of spatial features derived from correlated GNSS stations.

# Reproducibility

All datasets, scripts, and configurations required to reproduce the experimental results reported in the manuscript are provided in this repository.

The experiments can be reproduced by executing the scripts in the `src` directory following the instructions above.

---

# Citation

If you find this repository useful in your research, please consider citing the corresponding paper:

LGBM-ST: An Interpretable Spatiotemporal Interpolation Model for GNSS Time Series with Missing Data


---

# License

This project is intended for academic research and educational use.
