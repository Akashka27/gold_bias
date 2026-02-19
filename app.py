import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime
import pytz

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI M15 Trading Assistant", layout="wide")

st.title("🧠 AI Daily Bias + M15 Execution Assistant")
st.caption("HTF Bias (AI Model) → LTF Execution (M15)")

# ---------------- TIME (IST) ----------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

# ---------------- SESSION + KILL ZONE ----------------
def get_session_and_killzone():
    now = get_ist_time()
    hour = now.hour + now.minute / 60

    if 5.5 <= hour < 12.5:
        return "Asian Session 🟢", "Asian Range (Low Volatility)"
    elif 12.5 <= hour < 15.5:
        return "London Session 🔥", "LONDON KILL ZONE (Best for Gold)"
    elif 15.5 <= hour < 18.5:
        return "London Continuation ⚡", "Momentum Window"
    elif 18.5 <= hour < 21.5:
        return "New York Session 🚀", "NEW YORK KILL ZONE (Reversals & Expansion)"
    else:
        return "Off Market 🌙", "Dead Zone (Low Liquidity)"

session, killzone = get_session_and_killzone()

col_time1, col_time2 = st.columns(2)
with col_time1:
    st.info(f"🕒 Live Session: {session}")
with col_time2:
    st.warning(f"🎯 Kill Zone: {killzone}")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model

# ---------------- FETCH DATA (MATCH TRAINING) ----------------
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    asset = yf.download(ticker, period="5y", interval="1d", progress=False)
    dxy = yf.download("DX-Y.NYB", period="5y", interval="1d", progress=False)
    us10y = yf.download("^TNX", period="5y", interval="1d", progress=False)

    for df in [asset, dxy, us10y]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Merge macro drivers (SAME as training script)
    asset = asset.merge(
        dxy[["Close"]].rename(columns={"Close": "DXY"}),
        left_index=True,
        right_index=True,
        how="inner"
    )

    asset = asset.merge(
        us10y[["Close"]].rename(columns={"Close": "US10Y"}),
        left_index=True,
        right_index=True,
        how="inner"
    )

    asset.dropna(inplace=True)
    return asset

# ---------------- FEATURE ENGINEERING (IDENTICAL TO TRAINING) ----------------
def create_features(df):
    df = df.copy()

    # Trend
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df["Trend_Regime"] = np.where(df["EMA50"] > df["EMA200"], 1, 0)

    # Momentum
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD_hist"] = macd.macd_diff()

    # Volatility
    df["ATR"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )
    df["ATR_pct"] = df["ATR"] / df["Close"]

    df["ATR_Rolling"] = df["ATR_pct"].rolling(100).mean()
    df["Volatility_Regime"] = df["ATR_pct"] / df["ATR_Rolling"]

    # Bollinger
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_width"] = bb.bollinger_wband()

    # Returns
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_5"] = df["Close"].pct_change(5)
    df["Volatility_5"] = df["Close"].pct_change().rolling(5).std()

    # Macro Drivers (CRITICAL)
    df["DXY_change"] = df["DXY"].pct_change()
    df["US10Y_change"] = df["US10Y"].pct_change()

    df.dropna(inplace=True)

    features = [
        "EMA50","EMA200","ADX","Trend_Regime",
        "RSI","MACD_hist",
        "ATR_pct","Volatility_Regime","BB_width",
        "Return_1","Return_3","Return_5","Volatility_5",
        "DXY_change","US10Y_change"
    ]

    latest = df[features].iloc[-1]
    return latest.values.reshape(1, -1), df.iloc[-1]

# ---------------- M15 EXECUTION PLAN ENGINE ----------------
def generate_execution_plan(prob, session_name, killzone_name):
    # Bias classification
    if prob > 0.6:
        bias = "Bullish 🟢"
        execution = "Look for M15 BUY pullbacks into demand/support zones."
        avoid = "Avoid counter-trend shorts unless H1 structure breaks."
    elif prob < 0.4:
        bias = "Bearish 🔴"
        execution = "Look for M15 SELL pullbacks into supply/resistance zones."
        avoid = "Avoid counter-trend buys unless market structure shifts."
    else:
        bias = "Neutral 🟡"
        execution = "Wait for clear M15 structure before entering trades."
        avoid = "Avoid trading in chop or low volatility conditions."

    # Session intelligence (Gold optimized)
    if "Asian" in session_name:
        timing = "Low volatility session. Best to wait for London Kill Zone."
    elif "London" in session_name:
        timing = "High probability session for M15 setups."
    elif "New York" in session_name:
        timing = "Watch for reversals and continuation moves."
    else:
        timing = "Dead liquidity zone. Avoid low-quality entries."

    return bias, execution, avoid, timing

# ---------------- MAIN ----------------
try:
    gold_model, jpy_model = load_models()

    gold_df = fetch_data("GC=F")
    jpy_df = fetch_data("JPY=X")

    gold_X, gold_latest = create_features(gold_df)
    jpy_X, jpy_latest = create_features(jpy_df)

    gold_prob = gold_model.predict_proba(gold_X)[0][1]
    jpy_prob = jpy_model.predict_proba(jpy_X)[0][1]

    gold_bias, gold_exec, gold_avoid, gold_timing = generate_execution_plan(
        gold_prob, session, killzone
    )

    jpy_bias, jpy_exec, jpy_avoid, jpy_timing = generate_execution_plan(
        jpy_prob, session, killzone
    )

    st.markdown("## 🎯 M15 Execution Decision Panel")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟡 XAUUSD (Gold)")
        st.success(f"**AI Daily Bias:** {gold_bias}")
        st.info(f"**Execution Plan:** {gold_exec}")
        st.warning(f"**Avoid:** {gold_avoid}")
        st.caption(f"Session Guidance: {gold_timing}")

    with col2:
        st.subheader("💴 USDJPY")
        st.success(f"**AI Daily Bias:** {jpy_bias}")
        st.info(f"**Execution Plan:** {jpy_exec}")
        st.warning(f"**Avoid:** {jpy_avoid}")
        st.caption(f"Session Guidance: {jpy_timing}")

    st.markdown("---")

    # Macro context (since your model uses DXY & US10Y)
    st.markdown("## 🌍 Macro Context (Model Drivers)")
    macro_col1, macro_col2 = st.columns(2)

    with macro_col1:
        st.write(f"DXY Change: {round(gold_latest['DXY_change'] * 100, 3)}%")
    with macro_col2:
        st.write(f"US10Y Change: {round(gold_latest['US10Y_change'] * 100, 3)}%")

    st.markdown("---")
    st.write("🕒 Last Updated:", get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST"))

except Exception as e:
    st.error(f"❌ Error loading data or model: {e}")
