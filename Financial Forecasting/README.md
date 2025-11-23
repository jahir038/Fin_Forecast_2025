Financial KPI Forecasting

This project focuses on forecasting financial KPIs for various MEP-KPI pairs using machine learning and statistical models. It processes historical quarterly data and provides predictions for the next quarter (e.g., FY26 Q1), allowing business users to evaluate performance trends and conduct what-if analyses.

🔍 Features
📁 Data Source: SQL Server table containing MEP-wise quarterly KPI data.

🔧 Models Used:

SARIMA (Seasonal ARIMA)

Correlation-based (Linear Regression)

Random Forest (Optional - deprecated due to lack of generalization)

🔮 Forecasting Objective: Predict the immediate next quarter KPI values (e.g., FY26 Q1).

📈 Evaluation Metrics: MAE, RMSE, R², WMAPE

✏️ What-If Capability: Users can simulate the impact of DL/IDL Headcount changes on KPIs using saved model coefficients.

🗂️ Project Structure

financial-kpi-forecasting/
│
├── src/
│   ├── data_loader.py              # Loads data from SQL server
│   ├── data_preprocessing.py       # Prepares data for each model type
│   ├── model_training.py           # Trains SARIMA & Correlation models
│   ├── forecasting.py              # Forecast logic for each model
|   ├── final_forecast.py        # Predicts next quarter using correlation model
│   ├── run_final_prediction.py     # Main script to run the final prediction module
│   └── main.py                     # Orchestrates full model training, evaluation & prediction
│
├── models/
│   └── correlation_models.pkl      # Saved correlation model for reuse
│ 
│
├── output/
│   └── final_forecast_FY26Q1.csv   # Final predicted output for FY26 Q1
│ 
│
└── README.md

⚙️ How to Run
1. ⚡ Initial Full Pipeline (Train + Predict)
bash
Copy
Edit
python src/main.py
Trains SARIMA and Correlation models.

Evaluates on last 3 quarters (e.g., FY25 Q2, Q3, Q4).

Saves results to model_predictions.csv.

2. 🔮 Final Forecast for FY26 Q1
bash
Copy
Edit
python src/run_final_prediction.py
Loads raw data and saved correlation models.

Predicts FY26 Q1 using entire historical data.

Saves output to output/final_forecast_FY26Q1.csv.


✅ Benefits
Tailored forecast per MEP-KPI

Transparent model coefficients for business users

Flexible architecture for future model upgrades

Designed for extensibility (e.g., dashboard integration)