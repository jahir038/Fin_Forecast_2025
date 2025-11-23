# -----------------------------
# model_training.py
# -----------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# SARIMA Model Training
# -----------------------------
def train_sarima_model(df_long, order=(1, 1, 1)):
    sarima_models = {}
    for (mep, kpi), group in df_long.groupby(["MEP", "Accounts"]):
        group_sorted = group.sort_values("Quarter")
        series = group_sorted["Value"].values

        if len(series) < 8 or np.isnan(series).any():
            continue

        train_series = series[:-4]  # Train up to FY25 Q1

        try:
            model = SARIMAX(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False)
            sarima_models[(mep, kpi)] = result
        except:
            continue

    return sarima_models

# -----------------------------
# Correlation Model Training
# -----------------------------
def train_correlation_model(df_corr, predictor_cols=["DL_HC", "IDL_HC"]):
    corr_models = {}
    target_kpis = [col for col in df_corr.columns if col not in ["MEP", "Quarter"] + predictor_cols]

    for kpi in target_kpis:
        for mep, group in df_corr.groupby("MEP"):
            group_sorted = group.sort_values("Quarter")

            X = group_sorted[predictor_cols].iloc[:-4].values
            y = group_sorted[kpi].iloc[:-4].values

            if np.isnan(X).any() or np.isnan(y).any():
                continue

            model = LinearRegression()
            model.fit(X, y)
            corr_models[(mep, kpi)] = model

    return corr_models
