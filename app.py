# app.py

import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from groq import Groq

# 🎯 Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI
st.set_page_config(page_title="📈 Revenue Forecast AI Agent", layout="wide")
st.title("📊 AI-Powered Revenue Forecasting")
st.markdown("Upload your Excel file with `Date` and `Revenue` columns.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
forecast_period = st.slider("Select forecast period (in days)", min_value=30, max_value=365, value=90)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Basic validation
        if "Date" not in df.columns or "Revenue" not in df.columns:
            st.error("❌ Columns `Date` and `Revenue` are required in the uploaded file.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.rename(columns={"Date": "ds", "Revenue": "y"})

        st.subheader("📄 Uploaded Data")
        st.dataframe(df.tail())

        # 🔮 Prophet Forecasting
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)

        st.subheader("📉 Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("🔍 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Prepare data for AI Commentary
        merged_df = df.merge(forecast[['ds', 'yhat']], on="ds", how="left")
        json_data = merged_df.to_json(orient="records")

        # 🧠 AI Commentary
        st.subheader("🤖 AI-Generated Forecast Commentary")

        prompt = f"""
        You are the Head of FP&A at a SaaS company. Based on this time series forecast for Revenue, provide:
        - Key trends and insights in revenue.
        - Any seasonality or patterns detected.
        - A CFO-ready summary using the Pyramid Principle.
        - Actionable recommendations for financial planning.

        Here is the dataset in JSON format:
        {json_data}
        """

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an FP&A expert skilled in time series forecasting and SaaS analytics."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        commentary = response.choices[0].message.content
        st.markdown(commentary)

    except Exception as e:
        st.error(f"An error occurred: {e}")
