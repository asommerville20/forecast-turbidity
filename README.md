# Deep Learning for River Suspended Sediment Forecasting

This repository contains a full workflow for building, training, evaluating, and optimizing a Long Short-Term Memory (LSTM) model to forecast daily suspended sediment concentrations (SSC) at a U.S. Geological Survey (USGS) gauging site. The pipeline is designed for reproducibility and scientific transparency, aligning with FAIR data and model practices.

---

## 🔍 Project Overview

Accurate forecasting of riverine suspended sediment is essential for watershed management, aquatic ecosystem protection, and water quality compliance. This project uses publicly available USGS data to:

- Preprocess and align multiple hydrologic time series.
- Train a deep learning model (LSTM) to predict SSC using turbidity and discharge data.
- Perform hyperparameter optimization with Optuna to improve model performance.
- Evaluate model accuracy using metrics commonly used in hydrology (e.g., NSE, WAPE).

---

## 🗂️ Repository Structure

| File                      | Purpose |
|--------------------------|---------|
| `st1_preprocessing_vars.py` | Downloads and preprocesses turbidity, discharge, and SSC data from NWIS for USGS site `11447650`. Aligns datasets to daily resolution and fills gaps using linear interpolation. |
| `st2_train_LSTM.py`      | Builds and trains a PyTorch LSTM model to forecast SSC. Includes a custom `RiverData` class, model evaluation routines, and model visualization utilities. |
| `st3_optimize_LSTM.py`   | Uses Optuna to perform Bayesian optimization of learning rate and weight decay. Includes early stopping and saves best model checkpoint. |

---

## 📊 Data Sources

All input data are programmatically retrieved from the USGS National Water Information System (NWIS) using the `dataretrieval` Python package:

- **Turbidity (FNU)**: Instantaneous values
- **Discharge (cfs)**: Tidally filtered and standard daily values
- **Suspended Sediment Concentration (SSC)**: Daily values

> Site: `11447650` – Sacramento River at Freeport, CA

---

## 🧠 Model Details

- **Architecture**: Single-layer LSTM with fully connected output
- **Features**: Turbidity, discharge, and lagged SSC
- **Sequence Length**: 14 days
- **Prediction Horizon**: 1 day
- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation Metrics**:
  - Nash–Sutcliffe Efficiency (NSE)
  - Weighted Absolute Percentage Error (WAPE)

---

## 📈 Optimization

Hyperparameter tuning is performed using **Optuna** with the following settings:

- Learning Rate: `loguniform(1e-4, 1e-2)`
- Weight Decay: `loguniform(1e-5, 1e-2)`
- Number of Trials: 200
- Early Stopping: 10 epochs of no improvement

The final results are saved and visualized using interactive HTML plots.

---

## 🛠️ Dependencies

- Python ≥ 3.9  
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- `torch` (PyTorch)
- `optuna`
- `dataretrieval`

Install all dependencies using:

```bash
pip install -r requirements.txt

