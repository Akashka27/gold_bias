import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime

st.set_page_config(page_title="AI Bias Dashboard", layout="wide")

st.title("📊 AI Daily Bias Dashboard (Auto Data from Yahoo Finance)")
st.caption("⏰ Auto updates daily using live market data")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model

# ---------------- FETCH DATA ----------------
@st.cache_data(ttl=3600)  # refresh every 1 hour automatically
def fetch_market_data():
    # Gold (proxy via Futures)
    gold = yf.download("GC=F", period="60d", interval="1d")
    
    # USDJPY
    usdjpy = yf.download("JPY=X", period="60d", interval="1d")
    
    return gold, usdjpy

# ---------------- FEATURE ENGINEERING ----------------
def create_features(df):
    df = df.copy()

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["ema20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()

    df = df.dropna()

    # Take only last row and REMOVE index
    latest = df[["rsi", "ema20", "ema50", "atr"]].iloc[-1]

    # Convert to proper shape (1, n_features)
    return latest.values.reshape(1, -1)

# ---------------- BIAS FUNCTION ----------------
def get_bias(model, features):
    # Ensure features are 2D row (1, n_features)
    if isinstance(features, pd.DataFrame):
        features = features.values

    # Fix shape issue (important)
    features = features.reshape(1, -1)

    prob = model.predict_proba(features)[0]
    confidence = round(np.max(prob) * 100, 2)

    if prob[1] > 0.6:
        bias = "Bullish 🟢"
    elif prob[0] > 0.6:
        bias = "Bearish 🔴"
    else:
        bias = "Neutral 🟡"

    return bias, confidence


# ---------------- MAIN LOGIC ----------------
try:
    gold_model, jpy_model = load_models()
    gold_data, jpy_data = fetch_market_data()

    gold_features = create_features(gold_data)
    jpy_features = create_features(jpy_data)

    gold_bias, gold_conf = get_bias(gold_model, gold_features)
    jpy_bias, jpy_conf = get_bias(jpy_model, jpy_features)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟡 XAUUSD (Gold)")
        st.metric("Bias", gold_bias)
        st.metric("Confidence", f"{gold_conf}%")
        st.caption("Data Source: Yahoo Finance (GC=F)")

    with col2:
        st.subheader("💴 USDJPY")
        st.metric("Bias", jpy_bias)
        st.metric("Confidence", f"{jpy_conf}%")
        st.caption("Data Source: Yahoo Finance (JPY=X)")

    st.divider()
    st.write(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error(f"Error loading data or model: {e}")
