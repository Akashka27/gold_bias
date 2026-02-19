import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Daily Bias Dashboard", layout="wide")

st.title("📊 AI Daily Bias Dashboard")
st.caption("⏰ Auto fetches daily data from Yahoo Finance (No CSV required)")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model


# ---------------- FETCH MARKET DATA ----------------
@st.cache_data(ttl=3600)  # refresh every 1 hour
def fetch_market_data():
    try:
        # Download more history to support lag features safely
        gold = yf.download("GC=F", period="2y", interval="1d", progress=False)
        usdjpy = yf.download("JPY=X", period="2y", interval="1d", progress=False)

        if gold.empty or usdjpy.empty:
            raise ValueError("Yahoo Finance returned empty data.")

        return gold, usdjpy

    except Exception as e:
        raise ValueError(f"Data fetch error: {e}")


# ---------------- CREATE MODEL-COMPATIBLE FEATURES ----------------
def create_model_features(df, model, symbol_name="Asset"):
    df = df.copy()

    # Clean dataframe
    df = df.dropna()

    if df.empty:
        raise ValueError(f"{symbol_name}: DataFrame is empty after cleaning.")

    # Get how many features the model expects (CRITICAL)
    n_features = model.n_features_in_

    # Ensure enough data exists
    if len(df) <= n_features + 10:
        raise ValueError(
            f"{symbol_name}: Not enough historical data. "
            f"Model needs > {n_features+10} rows, but got {len(df)}."
        )

    # Use CLOSE price for lag feature generation (most common ML training method)
    close_series = df["Close"].values

    # Create lag features dynamically to match model input size
    features = []
    for i in range(1, n_features + 1):
        features.append(close_series[-i])

    # Convert to correct numpy shape (1, n_features)
    features_array = np.array(features).reshape(1, -1)

    return features_array


# ---------------- BIAS PREDICTION FUNCTION ----------------
def get_bias(model, features):
    # Safety reshape (prevents shape errors like (50,1))
    features = np.array(features).reshape(1, -1)

    probs = model.predict_proba(features)[0]
    confidence = round(np.max(probs) * 100, 2)

    if probs[1] > 0.6:
        bias = "Bullish 🟢"
    elif probs[0] > 0.6:
        bias = "Bearish 🔴"
    else:
        bias = "Neutral 🟡"

    return bias, confidence, probs


# ---------------- MAIN EXECUTION ----------------
try:
    # Load models
    gold_model, jpy_model = load_models()

    # Fetch live data
    gold_data, jpy_data = fetch_market_data()

    # Create model-compatible features (auto shape match)
    gold_features = create_model_features(gold_data, gold_model, "Gold")
    jpy_features = create_model_features(jpy_data, jpy_model, "USDJPY")

    # Get bias predictions
    gold_bias, gold_conf, gold_probs = get_bias(gold_model, gold_features)
    jpy_bias, jpy_conf, jpy_probs = get_bias(jpy_model, jpy_features)

    # ---------------- DASHBOARD UI ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟡 XAUUSD (Gold)")
        st.metric("AI Bias", gold_bias)
        st.metric("Confidence", f"{gold_conf}%")
        st.caption("Data Source: Yahoo Finance (GC=F)")

    with col2:
        st.subheader("💴 USDJPY")
        st.metric("AI Bias", jpy_bias)
        st.metric("Confidence", f"{jpy_conf}%")
        st.caption("Data Source: Yahoo Finance (JPY=X)")

    st.divider()

    # Timestamp
    st.write(
        "📅 Last Updated:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Debug Section (KEEP during development)
    with st.expander("🔍 Debug Info (Important)"):
        st.write("Gold Model Expected Features:", gold_model.n_features_in_)
        st.write("Gold Feature Shape:", gold_features.shape)
        st.write("USDJPY Model Expected Features:", jpy_model.n_features_in_)
        st.write("USDJPY Feature Shape:", jpy_features.shape)
        st.write("Gold Data Rows:", len(gold_data))
        st.write("USDJPY Data Rows:", len(jpy_data))

except Exception as e:
    st.error(f"❌ Error loading data or model: {e}")
