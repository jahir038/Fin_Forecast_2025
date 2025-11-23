import streamlit as st
import pandas as pd
import io
from core.forecasting import predict_next_quarter, load_models
from core.data_transformer import transform_uploaded_data

st.set_page_config(page_title="Forecast Simulation", layout="wide")
st.title("📊 Forecast Simulation Tool")

uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df_raw = pd.read_excel(uploaded_file)
        st.sidebar.write("📌 Uploaded Columns:")
        st.sidebar.write(df_raw.columns.tolist())

        # User input for DL/IDL HC adjustment (%)
        st.sidebar.header("🛠️ Adjust DL / IDL HC Change (%)")
        dl_change = st.sidebar.number_input("DL HC Change (%)", value=0, step=1, format="%d")
        idl_change = st.sidebar.number_input("IDL HC Change (%)", value=0, step=1, format="%d")

        # Convert to multiplier (e.g., 10 => 1.10, -5 => 0.95)
        dl_factor = 1 + (dl_change / 100)
        idl_factor = 1 + (idl_change / 100)

        with st.expander("🔍 Preview Uploaded Data"):
            st.dataframe(df_raw)

        # Transform data
        df_model_input, df_full_history, forecast_quarter = transform_uploaded_data(df_raw)

        # Apply headcount changes
        df_model_input["DL_HC"] = df_model_input["DL_HC"] * dl_factor
        df_model_input["IDL_HC"] = df_model_input["IDL_HC"] * idl_factor

        # Load model
        corr_models = load_models()

        # Predict
        forecast_df = predict_next_quarter(df_model_input, corr_models, forecast_quarter)

        if not forecast_df.empty:
            st.success(f"✅ Forecast for {forecast_quarter} Generated!")
            st.dataframe(forecast_df)

            with st.expander("📥 Download Forecast (Excel)"):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    forecast_df.to_excel(writer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="📥 Download Excel",
                    data=buffer,
                    file_name="forecast_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("⚠️ Forecast output is empty. Please check if MEP & KPI combinations match the trained models.")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

else:
    st.info("👈 Upload a `.xlsx` file to begin.")
