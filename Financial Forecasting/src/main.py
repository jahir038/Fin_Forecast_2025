# -----------------------------
# main.py
# -----------------------------
import pandas as pd
import os
import joblib
from data_loader import load_and_preprocess_data_sql
from data_preprocessing import preprocess_for_sarima, preprocess_for_correlation_model
from model_training import train_sarima_model, train_correlation_model
from forecasting import predict_sarima, predict_correlation

# -----------------------------
# STEP 1: Load Raw Data
# -----------------------------
server = "localhost"
database = "finance_metric"
table_name = "KPI_Fin_1"

raw_df = load_and_preprocess_data_sql(server, database, table_name)


# -----------------------------
# STEP 2: Preprocessing
# -----------------------------

df_sarima = preprocess_for_sarima(raw_df)
df_corr = preprocess_for_correlation_model(raw_df)

# -----------------------------
# STEP 3: Train Models
# -----------------------------

sarima_models = train_sarima_model(df_sarima)
corr_models = train_correlation_model(df_corr)

# -----------------------------
# STEP 4: Predict
# -----------------------------

sarima_results = predict_sarima(df_sarima, sarima_models)
corr_results = predict_correlation(df_corr, corr_models)

# -----------------------------
# STEP 5: Combine & Save Results
# -----------------------------
all_results = pd.concat([sarima_results, corr_results], ignore_index=True)
# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Save the trained models
joblib.dump(corr_models, "models/corr_models.pkl")
print("✅ Correlation model saved to models/corr_models.pkl")

# Save the results in the output folder
all_results.to_csv("output/model_predictions.csv", index=False)
print("✅ Model predictions saved to model_predictions.csv")
