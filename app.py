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

# ---------------- ANIMATED NEON GRADIENT BACKGROUND ----------------
st.markdown("""
<style>

/* Animated Gradient Background (Fast + Professional) */
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

/* Remove extra top spacing */
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1200px;
}

/* Neon Title */
.neon-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #00f7ff;
    text-shadow:
        0 0 10px #00f7ff,
        0 0 30px #00f7ff,
        0 0 60px #00f7ff;
    margin-bottom: 6px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Glass Card (Trading Panel) */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    padding: 28px;
    border-radius: 18px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(0,255,255,0.15);
    box-shadow: 0 0 25px rgba(0,255,255,0.08);
    transition: 0.3s ease-in-out;
}

/* Hover Glow */
.glass-card:hover {
    box-shadow: 0 0 35px rgba(0,255,255,0.25);
    transform: translateY(-2px);
}

/* Bias Glow Styles */
.bullish {
    color: #00ff9f;
    font-size: 28px;
    font-weight: bold;
    text-shadow: 0 0 12px #00ff9f;
}

.bearish {
    color: #ff4d4d;
    font-size: 28px;
    font-weight: bold;
    text-shadow: 0 0 12px #ff4d4d;
}

.neutral {
    color: #ffd700;
    font-size: 28px;
    font-weight: bold;
    text-shadow: 0 0 12px #ffd700;
}

/* Section Title */
.section-title {
    font-size: 30px;
    font-weight: 700;
    color: #38bdf8;
    text-shadow: 0 0 12px rgba(56,189,248,0.7);
    margin-top: 10px;
}

/* Info Text */
.info-text {
    font-size: 16px;
    color: #e2e8f0;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="neon-title">⚡ AI M15 Trading Assistant Terminal</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">HTF AI Bias (Macro Model) → M15 Execution Guidance | Gold & USDJPY</div>',
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

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.info(f"🕒 Live Session: {session}")
with col_s2:
    st.warning(f"🎯 Kill Zone: {killzone}")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model

# ---------------- FETCH DATA (MATCHES YOUR TRAINING) ----------------
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    asset = yf.download(ticker, period="5y", interval="1d", progress=False)
    dxy = yf.download("DX-Y.NYB", period="5y", interval="1d", progress=False)
    us10y = yf.download("^TNX", period="5y", interval="1d", progress=False)

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
    return asset

# ---------------- FEATURE ENGINEERING (IDENTICAL TO TRAINING SCRIPT) ----------------
def create_features(df):
    df = df.copy()

    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df["Trend_Regime"] = np.where(df["EMA50"] > df["EMA200"], 1, 0)

    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD_hist"] = macd.macd_diff()

    df["ATR"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )
    df["ATR_pct"] = df["ATR"] / df["Close"]

    df["ATR_Rolling"] = df["ATR_pct"].rolling(100).mean()
    df["Volatility_Regime"] = df["ATR_pct"] / df["ATR_Rolling"]

    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_width"] = bb.bollinger_wband()

    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_5"] = df["Close"].pct_change(5)
    df["Volatility_5"] = df["Close"].pct_change().rolling(5).std()

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
    return latest.values.reshape(1, -1)

# ---------------- EXECUTION PLAN ENGINE (FOR M15 TRADING) ----------------
def generate_execution_plan(prob, session_name):
    if prob > 0.6:
        bias = '<span class="bullish">Bullish 🟢</span>'
        execution = "Focus on M15 BUY pullbacks into demand/support zones."
        avoid = "Avoid counter-trend shorts unless structure breaks."
    elif prob < 0.4:
        bias = '<span class="bearish">Bearish 🔴</span>'
        execution = "Focus on M15 SELL pullbacks into supply/resistance zones."
        avoid = "Avoid counter-trend buys unless structure shifts."
    else:
        bias = '<span class="neutral">Neutral 🟡</span>'
        execution = "Wait for clear M15 structure before entry."
        avoid = "Avoid trading in chop or low volatility."

    if "Asian" in session_name:
        timing = "Low volatility. Best to wait for London Kill Zone."
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

    gold_X = create_features(gold_df)
    jpy_X = create_features(jpy_df)

    gold_prob = gold_model.predict_proba(gold_X)[0][1]
    jpy_prob = jpy_model.predict_proba(jpy_X)[0][1]

    gold_bias, gold_exec, gold_avoid, gold_timing = generate_execution_plan(
        gold_prob, session
    )
    jpy_bias, jpy_exec, jpy_avoid, jpy_timing = generate_execution_plan(
        jpy_prob, session
    )

    st.markdown('<div class="section-title">🎯 M15 Execution Decision Panel</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🟡 XAUUSD (Gold)")
        st.markdown(f"**AI Bias:** {gold_bias}", unsafe_allow_html=True)
        st.markdown(f"<div class='info-text'><b>Execution Plan:</b> {gold_exec}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-text'><b>Avoid:</b> {gold_avoid}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-text'><b>Session Guidance:</b> {gold_timing}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💴 USDJPY")
        st.markdown(f"**AI Bias:** {jpy_bias}", unsafe_allow_html=True)
        st.markdown(f"<div class='info-text'><b>Execution Plan:</b> {jpy_exec}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-text'><b>Avoid:</b> {jpy_avoid}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-text'><b>Session Guidance:</b> {jpy_timing}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.write("🕒 Last Updated:", get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST"))

except Exception as e:
    st.error(f"❌ Error loading data or model: {e}")
