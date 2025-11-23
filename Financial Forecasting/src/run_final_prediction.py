# -----------------------------
# run_final_prediction.py
# -----------------------------
import pandas as pd
import joblib
from data_loader import load_and_preprocess_data_sql
from data_preprocessing import preprocess_for_correlation_model
from final_forecast import predict_next_quarter

# Load data and model
server = "localhost"
database = "finance_metric"
table_name = "KPI_Fin_1"

raw_df = load_and_preprocess_data_sql(server, database, table_name)
corr_models = joblib.load("models/corr_models.pkl")

df_corr = preprocess_for_correlation_model(raw_df)

# Predict FY26 Q1
final_forecast_df = predict_next_quarter(df_corr, corr_models)

# Save results
final_forecast_df.to_csv("output/final_forecast_FY26Q1.csv", index=False)
print("✅ Final forecast for FY26 Q1 saved to output/final_forecast_FY26Q1.csv")
