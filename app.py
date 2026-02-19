import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime
import pytz

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Bias Terminal PRO", layout="wide")

# ---------------- NEON CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.title {
    font-size: 44px;
    text-align: center;
    color: #00eaff;
    text-shadow: 0 0 20px #00eaff, 0 0 40px #00eaff;
}
.neon-card {
    border-radius: 18px;
    padding: 30px;
    margin: 20px 0;
    text-align: center;
    background: #111827;
    box-shadow: 0 0 25px rgba(0,255,255,0.15);
}
.bullish {
    color: #00ff9f;
    font-size: 38px;
    text-shadow: 0 0 15px #00ff9f;
}
.bearish {
    color: #ff4d4d;
    font-size: 38px;
    text-shadow: 0 0 15px #ff4d4d;
}
.neutral {
    color: #ffd700;
    font-size: 38px;
    text-shadow: 0 0 15px #ffd700;
}
.session {
    font-size: 22px;
    text-align: center;
    color: #00eaff;
    text-shadow: 0 0 10px #00eaff;
}
.killzone {
    font-size: 24px;
    text-align: center;
    color: #ff00ff;
    text-shadow: 0 0 15px #ff00ff;
}
.message {
    font-size: 18px;
    margin-top: 12px;
    color: #cbd5e1;
}
.gauge-title {
    font-size: 20px;
    color: #00eaff;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">⚡ AI Institutional Bias Terminal</p>', unsafe_allow_html=True)

# ---------------- TIME + SESSION ----------------
def get_ist_time():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

def get_session_and_killzone():
    now = get_ist_time()
    hour = now.hour + now.minute/60

    if 5.5 <= hour < 12.5:
        return "Asian Session 🟢", "🟢 Asian Range (Low Volatility)"
    elif 12.5 <= hour < 15.5:
        return "London Session 🔥", "🎯 LONDON KILL ZONE (Best for Gold)"
    elif 15.5 <= hour < 18.5:
        return "London Continuation ⚡", "⚡ Momentum Window"
    elif 18.5 <= hour < 21.5:
        return "New York Session 🚀", "🎯 NEW YORK KILL ZONE (Reversals)"
    else:
        return "Off Market 🌙", "💤 Dead Zone (Low Liquidity)"

session, killzone = get_session_and_killzone()

st.markdown(f'<div class="session">🕒 Live Session: {session}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="killzone">{killzone}</div>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model

# ---------------- FETCH DATA ----------------
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

# ---------------- FEATURE ENGINEERING ----------------
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

# ---------------- BIAS + SCORE ----------------
def get_bias_info(model, X):
    prob = model.predict_proba(X)[0][1]
    score = int(prob * 100)

    if prob > 0.6:
        bias = "Bullish 🟢"
        css = "bullish"
        msg = "📈 Buy near M15 demand/support during Kill Zone."
    elif prob < 0.4:
        bias = "Bearish 🔴"
        css = "bearish"
        msg = "📉 Sell near M15 supply/resistance during Kill Zone."
    else:
        bias = "Neutral 🟡"
        css = "neutral"
        msg = "⚠️ Wait for structure + Kill Zone alignment."

    return bias, css, msg, score

# ---------------- MAIN ----------------
try:
    gold_model, jpy_model = load_models()

    gold_df = fetch_data("GC=F")
    jpy_df = fetch_data("JPY=X")

    gold_X = create_features(gold_df)
    jpy_X = create_features(jpy_df)

    gold_bias, gold_css, gold_msg, gold_score = get_bias_info(gold_model, gold_X)
    jpy_bias, jpy_css, jpy_msg, jpy_score = get_bias_info(jpy_model, jpy_X)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="neon-card">
            <div>🟡 XAUUSD (Gold)</div>
            <div class="{gold_css}">{gold_bias}</div>
            <div class="message">{gold_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="gauge-title">Bias Strength Meter</div>', unsafe_allow_html=True)
        st.progress(gold_score)

    with col2:
        st.markdown(f"""
        <div class="neon-card">
            <div>💴 USDJPY</div>
            <div class="{jpy_css}">{jpy_bias}</div>
            <div class="message">{jpy_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="gauge-title">Bias Strength Meter</div>', unsafe_allow_html=True)
        st.progress(jpy_score)

    st.markdown("---")
    st.write("🕒 Last Updated:", get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST"))

except Exception as e:
    st.error(f"❌ Error loading data or model: {e}")
