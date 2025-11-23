import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Evaluation Metrics
# -----------------------------
def compute_metrics_group(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) if np.sum(np.abs(y_true)) != 0 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, wmape, rmse, r2

# -----------------------------
# Dynamic Quarter Ordering
# -----------------------------
def get_chronological_quarter_order(quarter_series):
    def quarter_key(q):
        year = int(q[2:4])  # e.g., FY25 → 25
        quarter = int(q[-1])  # Q1 → 1
        return (year, quarter)

    unique_quarters = sorted(quarter_series.dropna().unique(), key=quarter_key)
    return unique_quarters

# -----------------------------
# SARIMA Prediction
# -----------------------------
def predict_sarima(df_long, sarima_models):
    results = []

    for (mep, kpi), model in sarima_models.items():
        group = df_long[(df_long["MEP"] == mep) & (df_long["Accounts"] == kpi)].copy()

        # Ensure correct chronological sorting of quarters
        group["Quarter"] = pd.Categorical(group["Quarter"], ordered=True, categories=sorted(df_long["Quarter"].unique()))
        group = group.sort_values("Quarter")

        # Predict last 3 actual quarters
        y_true = group["Value"].values[-3:]
        forecast_quarters = group["Quarter"].values[-3:]
        y_pred = model.forecast(steps=3)

        mae, wmape, rmse, r2 = compute_metrics_group(y_true, y_pred)

        for i in range(3):
            results.append({
                "Model": "SARIMA",
                "MEP": mep,
                "KPI": kpi,
                "Quarter": forecast_quarters[i],
                "Actual": y_true[i],
                "Predicted": y_pred[i],
                "MAE": mae,
                "WMAPE": wmape,
                "RMSE": rmse,
                "R2": r2
            })

    return pd.DataFrame(results)

# -----------------------------
# Correlation Model Prediction
# -----------------------------
def predict_correlation(df_corr, corr_models, predictor_cols=["DL_HC", "IDL_HC"]):
    results = []
    target_kpis = [col for col in df_corr.columns if col not in ["MEP", "Quarter"] + predictor_cols]

    for kpi in target_kpis:
        for mep, group in df_corr.groupby("MEP"):
            key = (mep, kpi)
            if key not in corr_models:
                continue

            group_sorted = group.copy()
            group_sorted["Quarter"] = pd.Categorical(group_sorted["Quarter"], ordered=True, categories=sorted(df_corr["Quarter"].unique()))
            group_sorted = group_sorted.sort_values("Quarter")

            # Last 3 actual quarters
            X_test = group_sorted[predictor_cols].iloc[-3:].values
            y_true = group_sorted[kpi].iloc[-3:].values
            forecast_quarters = group_sorted["Quarter"].values[-3:]

            model = corr_models[key]
            y_pred = model.predict(X_test)

            mae, wmape, rmse, r2 = compute_metrics_group(y_true, y_pred)

            for i in range(3):
                results.append({
                    "Model": "Correlation",
                    "MEP": mep,
                    "KPI": kpi,
                    "Quarter": forecast_quarters[i],
                    "Actual": y_true[i],
                    "Predicted": y_pred[i],
                    "MAE": mae,
                    "WMAPE": wmape,
                    "RMSE": rmse,
                    "R2": r2
                })

    return pd.DataFrame(results)