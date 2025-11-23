# final_forecast.py
import pandas as pd
import numpy as np
import joblib

def predict_next_quarter(df_corr, corr_models, predictor_cols=["DL_HC", "IDL_HC"]):
    # Sort quarters properly: FY22 Q1, FY22 Q2, ...
    def quarter_sort_key(q):
        fy, qtr = q.split()
        return (int(fy[2:]), int(qtr[1]))

    quarter_order = sorted(df_corr["Quarter"].unique(), key=quarter_sort_key)
    latest_quarter = quarter_order[-1]

    results = []

    for kpi in df_corr.columns:
        if kpi in ["MEP", "Quarter"] + predictor_cols:
            continue  # Skip non-KPI columns

        for mep, group in df_corr.groupby("MEP"):
            key = (mep, kpi)
            if key not in corr_models:
                continue

            model = corr_models[key]

            # Get latest input row
            latest_row = group[group["Quarter"] == latest_quarter]
            if latest_row.empty:
                continue

            input_feats = latest_row[predictor_cols].values.reshape(1, -1)
            prediction = model.predict(input_feats)[0]

            # Coefficients & intercept
            coef = model.coef_ if hasattr(model, "coef_") else [np.nan] * len(predictor_cols)
            intercept = model.intercept_ if hasattr(model, "intercept_") else np.nan

            row_data = {
                "MEP": mep,
                "KPI": kpi,
                "Quarter": "FY26 Q1",
                "Predicted_KPI": prediction,
                "Intercept": intercept,
                "DL_HC_Used": latest_row["DL_HC"].values[0],
                "IDL_HC_Used": latest_row["IDL_HC"].values[0]
            }

            for i, col in enumerate(predictor_cols):
                row_data[f"Coef_{col}"] = coef[i]

            # Include all past actual KPI values in wide format
            past_vals = group.pivot(index="MEP", columns="Quarter", values=kpi)
            for qtr in quarter_order:
                val = past_vals.loc[mep, qtr] if qtr in past_vals.columns else np.nan
                row_data[qtr] = val

            results.append(row_data)

    return pd.DataFrame(results)
