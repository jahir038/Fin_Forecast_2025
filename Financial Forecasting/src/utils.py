# src/utils.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def compute_metrics_group(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) if np.sum(np.abs(y_true)) != 0 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # 👈 FIX HERE
    r2 = r2_score(y_true, y_pred)
    return mae, wmape, rmse, r2
