import pandas as pd
import numpy as np
import joblib

def load_models(model_path="models/correlation_models.pkl"):
    return joblib.load(model_path)

def predict_next_quarter(df_corr, corr_models, forecast_quarter, predictor_cols=["DL_HC", "IDL_HC"]):
    # Get list of quarters (exclude forecast quarter)
    quarter_order = sorted(df_corr["Quarter"].unique(), key=lambda x: (int(x[2:4]), int(x[-1])))

    results = []

    for kpi in df_corr.columns:
        if kpi in ["MEP", "Quarter"] + predictor_cols:
            continue

        for mep, group in df_corr.groupby("MEP"):
            key = (mep, kpi)
            if key not in corr_models:
                continue

            model = corr_models[key]
            latest_row = group[group["Quarter"] == forecast_quarter]
            if latest_row.empty:
                continue

            input_feats = latest_row[predictor_cols].values.reshape(1, -1)
            prediction = model.predict(input_feats)[0]

            coef = model.coef_ if hasattr(model, "coef_") else [np.nan] * len(predictor_cols)
            intercept = model.intercept_ if hasattr(model, "intercept_") else np.nan

            row_data = {
                "MEP": mep,
                "KPI": kpi,
                "Quarter": forecast_quarter,
                "Predicted_KPI": prediction,
                "Intercept": intercept,
                "DL_HC_Used": latest_row["DL_HC"].values[0],
                "IDL_HC_Used": latest_row["IDL_HC"].values[0],
            }

            for i, col in enumerate(predictor_cols):
                row_data[f"Coef_{col}"] = coef[i]

            # Add historical KPI values — exclude forecast quarter
            past_vals = group.pivot(index="MEP", columns="Quarter", values=kpi)
            for qtr in quarter_order:
                if qtr == forecast_quarter:
                    continue  # Skip forecast quarter in actuals
                val = past_vals.loc[mep, qtr] if qtr in past_vals.columns else np.nan
                row_data[qtr] = val

            results.append(row_data)

    return pd.DataFrame(results)
