import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime
import pytz

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI M15 Trading Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- NEON GRADIENT BACKGROUND ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #020617, #020617, #031525, #00111a);
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.block-container {
    padding-top: 1.5rem !important;
    max-width: 1200px;
}

.neon-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #00f7ff;
    text-shadow: 0 0 10px #00f7ff,
                 0 0 30px #00f7ff,
                 0 0 60px #00f7ff;
}

.glass-card {
    background: rgba(15, 23, 42, 0.6);
    padding: 28px;
    border-radius: 18px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(0,255,255,0.15);
    box-shadow: 0 0 25px rgba(0,255,255,0.08);
}

.bullish { color: #00ff9f; font-size: 26px; font-weight: bold; }
.bearish { color: #ff4d4d; font-size: 26px; font-weight: bold; }
.neutral { color: #ffd700; font-size: 26px; font-weight: bold; }

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #38bdf8;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="neon-title">⚡ AI M15 Trading Assistant Terminal</div>',
    unsafe_allow_html=True
)

# ---------------- TIME (IST) ----------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

# ---------------- SESSION + KILL ZONE ----------------
def get_session_and_killzone():
    now = get_ist_time()
    hour = now.hour + now.minute / 60

    if 5.5 <= hour < 12.5:
        return "Asian Session 🟢", "Low Volatility"
    elif 12.5 <= hour < 15.5:
        return "London Session 🔥", "London Kill Zone"
    elif 15.5 <= hour < 18.5:
        return "London Continuation ⚡", "Momentum Window"
    elif 18.5 <= hour < 21.5:
        return "New York Session 🚀", "New York Kill Zone"
    else:
        return "Off Market 🌙", "Dead Liquidity Zone"

session, killzone = get_session_and_killzone()

col1, col2 = st.columns(2)
col1.info(f"🕒 Session: {session}")
col2.warning(f"🎯 Kill Zone: {killzone}")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model

# ---------------- SAFE DATA FETCH ----------------
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    # 10 years ensures enough rows for EMA200 + rolling features
    asset = yf.download(ticker, period="10y", interval="1d", progress=False)
    dxy = yf.download("DX-Y.NYB", period="10y", interval="1d", progress=False)
    us10y = yf.download("^TNX", period="10y", interval="1d", progress=False)

    if asset.empty or dxy.empty or us10y.empty:
        raise ValueError("Market data download failed from Yahoo Finance.")

    for df in [asset, dxy, us10y]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

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

    if len(asset) < 300:
        raise ValueError("Not enough merged macro data for feature generation.")

    return asset

# ---------------- SAFE FEATURE ENGINEERING (MATCHES TRAINING EXACTLY) ----------------
def create_features(df):
    df = df.copy()

    # Trend Features
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

    # Macro Drivers (CRITICAL - from training)
    df["DXY_change"] = df["DXY"].pct_change()
    df["US10Y_change"] = df["US10Y"].pct_change()

    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Feature dataframe became empty after indicators.")

    features = [
        "EMA50","EMA200","ADX","Trend_Regime",
        "RSI","MACD_hist",
        "ATR_pct","Volatility_Regime","BB_width",
        "Return_1","Return_3","Return_5","Volatility_5",
        "DXY_change","US10Y_change"
    ]

    latest = df[features].iloc[-1]
    X = latest.values.reshape(1, -1)

    return X

# ---------------- EXECUTION PLAN ----------------
def generate_plan(prob, session_name):
    if prob > 0.6:
        bias = '<span class="bullish">Bullish 🟢</span>'
        plan = "Focus on M15 BUY pullbacks near support/demand."
        avoid = "Avoid counter-trend shorts."
    elif prob < 0.4:
        bias = '<span class="bearish">Bearish 🔴</span>'
        plan = "Focus on M15 SELL pullbacks near resistance/supply."
        avoid = "Avoid counter-trend buys."
    else:
        bias = '<span class="neutral">Neutral 🟡</span>'
        plan = "Wait for clear M15 structure before entry."
        avoid = "Avoid trading in chop."

    return bias, plan, avoid

# ---------------- MAIN ----------------
try:
    gold_model, jpy_model = load_models()

    gold_df = fetch_data("GC=F")
    jpy_df = fetch_data("JPY=X")

    gold_X = create_features(gold_df)
    jpy_X = create_features(jpy_df)

    gold_prob = gold_model.predict_proba(gold_X)[0][1]
    jpy_prob = jpy_model.predict_proba(jpy_X)[0][1]

    gold_bias, gold_plan, gold_avoid = generate_plan(gold_prob, session)
    jpy_bias, jpy_plan, jpy_avoid = generate_plan(jpy_prob, session)

    st.markdown('<div class="section-title">🎯 M15 Execution Decision Panel</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🟡 XAUUSD (Gold)")
        st.markdown(f"AI Bias: {gold_bias}", unsafe_allow_html=True)
        st.write(f"Execution Plan: {gold_plan}")
        st.write(f"Avoid: {gold_avoid}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💴 USDJPY")
        st.markdown(f"AI Bias: {jpy_bias}", unsafe_allow_html=True)
        st.write(f"Execution Plan: {jpy_plan}")
        st.write(f"Avoid: {jpy_avoid}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.write("🕒 Last Updated:", get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST"))

except Exception as e:
    st.error(f"❌ Error loading data or model: {e}")
