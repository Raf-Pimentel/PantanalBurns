# PantanalBurns: A Regime-Aware Framework for Wetland Disturbance Monitoring

This repository contains the reproducible open-source Python workflow for the study: **"A Reproducible Unsupervised Framework to Disentangle Fire Disturbance and Flood-Pulse Variability in the Pantanal Wetlands (2020–2025)"**.

## Project Overview

Monitoring disturbances in large floodplains like the Brazilian Pantanal is challenging because hydroclimatic seasonality (flood pulses) can mimic or mask the spectral signals of fire and recovery. This project implements a **regime-aware, unsupervised stratification** framework to distinguish true disturbance-related changes from background seasonal variability in the northern Pantanal.

### Key Features

* 
**Dual-Stream Unsupervised Learning:** Combines spatial K-means clustering () to map eco-spectral regimes with temporal state-space clustering () to identify system conditions: Healthy, Stressed/Recovering, or Severely Impacted.


* 
**Satellite-Only Workflow:** Uses Landsat 8 Surface Reflectance time series (NDVI and NBR) without requiring external field severity data.


* 
**Predictive Modeling:** Includes a Random Forest regressor to model NBR dynamics and project recovery trends into 2026.


* 
**Reproducibility:** A fully documented pipeline from raw raster statistics extraction to final trend visualization.



## Repository Structure

* `PipelinePantanal.py`: Automates the extraction of temporal statistics (mean, std) from NDVI and NBR rasters, handles NoData masking, and generates the consolidated `resultados_temporais.csv`.
* `temporal_analysis.py`: Performs the machine learning analysis, including Random Forest modeling, performance evaluation (, MAE), and future trajectory forecasting.
* `temporal_results.csv`: Contains the processed time-series data (2020–2025) and valid pixel counts for each satellite acquisition.
* `requirements.txt`: Defines the Python environment dependencies, including `rasterio`, `scikit-learn`, `pandas`, and `matplotlib`.
* `.gitignore`: Configured to exclude local virtual environments, cache, and large output directories (`_NBR_OUT/`, `_NDVI_OUT/`).

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Raf-Pimentel/PantanalBurns.git
cd PantanalBurns

```


2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt

```



## Workflow

### 1. Data Processing

The `PipelinePantanal.py` script identifies and processes Landsat 8 scenes. It filters pixels within the physically valid range of  and merges NBR and NDVI data by `scene_id` to ensure temporal consistency.

### 2. Machine Learning & Forecasting

The `temporal_analysis.py` script implements:

* **Feature Engineering:** Creation of time-based features (month, day of year) and lag variables (NBR/NDVI lag-1) to account for system memory.
* **Model Training:** A Random Forest Regressor trained on an 80/20 temporal split.
* **Forecasting:** An autoregressive 12-month forecast for the 2026 cycle.

## Key Results

* 
**Ecosystem States:** The framework identified three robust states: Healthy/High Vigor (~47.9%), Stressed/Recovering (~33.8%), and Severely Impacted (~18.3%).


* 
**Predictive Skill:** The Random Forest model reproduced historical NBR dynamics with high accuracy ().


* 
**Disturbance Detection:** The method accurately captured acute decline episodes during the 2020 wildfires and the 2024 drought crisis.



## Citation

If you use this framework or data in your research, please cite:

> Bazo de Castro, D., Pimentel de Melo, R. R., & Pereira, V. S. (2025). *A Reproducible Unsupervised Framework to Disentangle Fire Disturbance and Flood-Pulse Variability in the Pantanal Wetlands (2020–2025)*.

## License

This project is licensed under the **MIT License**.