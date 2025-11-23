

# Preprocess the data to feed into required format for trying different models
#We are planning to use the following models:
# 1. SARIMA
# 2. Random Forest
# 3. Correlation based models (Linear Regression)

# 1. For SARIMA Model:Requires the data in long format per MEP,KPI Pair
def preprocess_for_sarima(df):
    df_long = df.melt(id_vars=['MEP', 'Accounts'], var_name='Quarter', value_name='Value')
    return df_long


# 3. For Correlation-based Models: Requires wide format with DL HC and IDL HC as predictors
# and other KPIs as target candidates


def preprocess_for_correlation_model(df):
    """
    Pivot long-format data into wide format for correlation-based modeling.
    Keeps DL HC and IDL HC as predictors, and other KPIs as target candidates.

    Input:
        df_long: DataFrame with ['MEP', 'KPI', 'Quarter', 'Value']

    Output:
        df_corr: DataFrame with ['MEP', 'Quarter', 'DL_HC', 'IDL_HC', ...other KPIs...]
    """
    df_long = df.melt(id_vars=["MEP", "Accounts"], var_name="Quarter", value_name="Value")

    # Step 1: Pivot into wide format
    df_pivot = df_long.pivot_table(
        index=["MEP", "Quarter"],
        columns="Accounts",
        values="Value"
    ).reset_index()

    # Step 2: Rename DL/IDL columns for easier reference
    df_pivot = df_pivot.rename(columns={
        "KPI: Direct Headcount - Total": "DL_HC",
        "KPI: Indirect Headcount - Total": "IDL_HC"
    })

    # Step 3: Drop rows with missing DL or IDL HC
    df_corr = df_pivot.dropna(subset=["DL_HC", "IDL_HC"])

    return df_corr

