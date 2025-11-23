import pandas as pd

# Only DL/IDL HC mappings
KPI_MAPPING = {
    "KPI: Direct Headcount - Total": "DL_HC",
    "KPI: Indirect Headcount - Total": "IDL_HC"
}

def transform_uploaded_data(df_raw):
    # Melt wide format to long format
    df_long = df_raw.melt(id_vars=["MEP", "Accounts"], var_name="Quarter", value_name="Value")

    # Clean whitespace
    df_long["Accounts"] = df_long["Accounts"].str.strip()
    df_long["Quarter"] = df_long["Quarter"].str.strip()

    # Get unique quarters and determine forecast quarter
    quarters = sorted(df_long["Quarter"].unique(), key=lambda x: (int(x[2:4]), int(x[-1])))
    latest_quarter = quarters[-1]

    # Forecast for next quarter
    fy_num = int(latest_quarter[2:4])
    qtr_num = int(latest_quarter[-1])
    forecast_quarter = f"FY{fy_num + 1} Q1" if qtr_num == 4 else f"FY{fy_num} Q{qtr_num + 1}"

    # Map DL/IDL, leave rest as KPI names
    df_long["KPI"] = df_long["Accounts"].map(KPI_MAPPING).fillna(
        df_long["Accounts"].str.replace("KPI: ", "", regex=False)
    )

    # Pivot to wide format per quarter
    df_pivot = df_long.pivot_table(index=["MEP", "Quarter"], columns="KPI", values="Value").reset_index()

    # Filter latest quarter for model input
    df_model_input = df_pivot[df_pivot["Quarter"] == latest_quarter].copy()
    df_model_input["Quarter"] = forecast_quarter  # Replace with prediction quarter

    # Create full historical pivot: MEP, KPI, FY22 Q1 ... FY25 Q4
    df_history = df_long.pivot(index=["MEP", "KPI"], columns="Quarter", values="Value").reset_index()

    return df_model_input, df_history, forecast_quarter
