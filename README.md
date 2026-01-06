# PantanalBurns: A Regime-Aware Framework for Wetland Disturbance Monitoring

This repository contains the reproducible open-source Python workflow for the study: **"A Reproducible Unsupervised Framework to Disentangle Fire Disturbance and Flood-Pulse Variability in the Pantanal Wetlands (2020–2025)"**.

## Project Overview

Monitoring disturbances in large floodplains like the Brazilian Pantanal is challenging because hydroclimatic seasonality (flood pulses) can mimic or mask spectral signals of fire and recovery. This project implements a **regime-aware, unsupervised stratification** framework to distinguish true disturbance-related changes from background seasonal variability.

### Key Features

* 
**Dual-Stream Unsupervised Learning:** Combines spatial K-means clustering () to map eco-spectral regimes and temporal state-space clustering () to identify system conditions (Healthy, Stressed, or Impacted).


* 
**Satellite-Only Workflow:** Uses Landsat 8 Surface Reflectance time series (NDVI and NBR) without requiring external field severity data.


* 
**Predictive Modeling:** Includes a Random Forest regressor to model NBR dynamics and project recovery trends into 2026.


* 
**Reproducibility:** Fully documented pipeline from raw raster statistics extraction to final trend visualization.



## Repository Structure

* `PipelinePantanal.py`: Extracts temporal statistics (mean, std) from NDVI and NBR rasters, handles NoData masking, and generates the consolidated `resultados_temporais.csv`.
* `temporal_analysis.py`: Performs Random Forest modeling, evaluates predictive performance (, MAE), and generates future forecasts and visualizations.
* `temporal_results.csv`: Contains the processed time-series data (2020–2025) used in the study.
* `requirements.txt`: List of necessary Python dependencies (Geoprocessing, Data Science, and ML libraries).
* `.gitignore`: Prevents local environment files and large raster output directories (`_NBR_OUT/`, `_NDVI_OUT/`) from being tracked.

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

The `PipelinePantanal.py` script processes Landsat 8 scenes. It filters pixels within the physically valid range of  and merges NBR and NDVI data by `scene_id` to ensure temporal consistency.

### 2. Machine Learning Analysis

The `temporal_analysis.py` script implements the following:

* **Feature Engineering:** Creates time-based features (month, day of year) and lag variables to account for system memory.
* **Model Training:** Uses a Random Forest Regressor () with an 80/20 temporal split.
* **Forecasting:** Generates a 12-month autoregressive forecast for NBR trends.

## Key Results

* 
**System States:** The framework identified three robust states: Healthy/High Vigor (~47.9%), Stressed/Recovering (~33.8%), and Severely Impacted (~18.3%).


* 
**Model Accuracy:** The Random Forest model reproduced NBR dynamics with an .


* 
**Crisis Detection:** The method accurately captured acute decline episodes during the historic 2020 wildfires and the 2024 drought crisis.



## Citation

If you use this framework or data in your research, please cite:

> Bazo de Castro, D., Pimentel de Melo, R. R., & Pereira, V. S. (2025). *A Reproducible Unsupervised Framework to Disentangle Fire Disturbance and Flood-Pulse Variability in the Pantanal Wetlands (2020–2025)*. 
> 
> 

## License

This project is licensed under the **CC-BY 4.0 License**.