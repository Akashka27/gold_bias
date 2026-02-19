import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="AI Bias Dashboard", layout="wide")

st.title("📊 AI Daily Bias Dashboard (Auto Yahoo Finance Data)")
st.caption("⏰ Auto updates daily using live market data")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model

# ---------------- FETCH DATA ----------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    gold = yf.download("GC=F", period="120d", interval="1d")
    usdjpy = yf.download("JPY=X", period="120d", interval="1d")
    return gold, usdjpy

# ---------------- FEATURE ENGINEERING (MODEL SAFE) ----------------
def create_lag_features(df, model):
    """
    Creates lag features from Close price based on model feature size.
    This automatically matches your trained XGBoost input shape.
    """
    df = df.copy()
    df = df.dropna()

    # Get how many features the model expects
    n_features = model.n_features_in_

    # Create lag features dynamically
    for i in range(1, n_features + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)

    df = df.dropna()

    # Take latest row
    feature_cols = [f"lag_{i}" for i in range(1, n_features + 1)]
    latest_row = df[feature_cols].iloc[-1]

    # Convert to correct shape (1, n_features)
    features = latest_row.values.reshape(1, -1)

    return features

# ---------------- BIAS FUNCTION ----------------
def get_bias(model, features):
    # Final safety shape check
    features = np.array(features).reshape(1, -1)

    prob = model.predict_proba(features)[0]
    confidence = round(np.max(prob) * 100, 2)

    if prob[1] > 0.6:
        bias = "Bullish 🟢"
    elif prob[0] > 0.6:
        bias = "Bearish 🔴"
    else:
        bias = "Neutral 🟡"

    return bias, confidence, prob

# ---------------- MAIN LOGIC ----------------
try:
    gold_model, jpy_model = load_models()
    gold_data, jpy_data = fetch_market_data()

    # 🔥 Create features that MATCH model training shape
    gold_features = create_lag_features(gold_data, gold_model)
    jpy_features = create_lag_features(jpy_data, jpy_model)

    gold_bias, gold_conf, gold_prob = get_bias(gold_model, gold_features)
    jpy_bias, jpy_conf, jpy_prob = get_bias(jpy_model, jpy_features)

    # ---------------- DASHBOARD UI ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟡 XAUUSD Bias (Gold)")
        st.metric("Bias", gold_bias)
        st.metric("Confidence", f"{gold_conf}%")
        st.caption("Data Source: Yahoo Finance (GC=F)")

    with col2:
        st.subheader("💴 USDJPY Bias")
        st.metric("Bias", jpy_bias)
        st.metric("Confidence", f"{jpy_conf}%")
        st.caption("Data Source: Yahoo Finance (JPY=X)")

    st.divider()

    st.write("📅 Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Debug section (remove later)
    with st.expander("🔍 Debug Info (Model Shape)"):
        st.write("Gold Model Expected Features:", gold_model.n_features_in_)
        st.write("Gold Feature Shape:", gold_features.shape)
        st.write("JPY Model Expected Features:", jpy_model.n_features_in_)
        st.write("JPY Feature Shape:", jpy_features.shape)

except Exception as e:
    st.error(f"Error loading data or model: {e}")
