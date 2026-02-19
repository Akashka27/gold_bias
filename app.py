import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Neon Bias Dashboard", layout="wide")

# ---------------- NEON CSS (🔥 UI UPGRADE) ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.neon-card {
    border-radius: 18px;
    padding: 30px;
    margin: 20px 0;
    text-align: center;
    font-weight: bold;
    box-shadow: 0 0 25px rgba(0,255,255,0.2);
    background: #111827;
}
.bullish {
    color: #00ff9f;
    font-size: 36px;
    text-shadow: 0 0 10px #00ff9f, 0 0 20px #00ff9f;
}
.bearish {
    color: #ff4d4d;
    font-size: 36px;
    text-shadow: 0 0 10px #ff4d4d, 0 0 20px #ff4d4d;
}
.neutral {
    color: #ffd700;
    font-size: 36px;
    text-shadow: 0 0 10px #ffd700, 0 0 20px #ffd700;
}
.message {
    font-size: 18px;
    margin-top: 10px;
    color: #cbd5e1;
}
.title-glow {
    font-size: 42px;
    text-align: center;
    color: #00eaff;
    text-shadow: 0 0 15px #00eaff, 0 0 30px #00eaff;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-glow">⚡ AI Daily Bias Terminal (Neon Edition)</p>', unsafe_allow_html=True)
st.caption("Auto Macro + Regime Bias | Gold & USDJPY")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    gold_model = joblib.load("gold_daily_bias_xgb.pkl")
    jpy_model = joblib.load("usdjpy_daily_bias_xgb.pkl")
    return gold_model, jpy_model


# ---------------- FETCH DATA ----------------
@st.cache_data(ttl=3600)
def fetch_all_data(ticker):
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


# ---------------- FEATURE ENGINEERING (MATCH TRAINING) ----------------
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


# ---------------- BIAS + MESSAGE LOGIC ----------------
def get_bias_and_message(model, X):
    prob = model.predict_proba(X)[0][1]

    if prob > 0.6:
        bias = "Bullish"
        css_class = "bullish"
        message = "📈 Bias is Bullish — Look for BUY entries near M15 demand/support levels."
    elif prob < 0.4:
        bias = "Bearish"
        css_class = "bearish"
        message = "📉 Bias is Bearish — Look for SELL entries near M15 supply/resistance levels."
    else:
        bias = "Neutral"
        css_class = "neutral"
        message = "⚠️ Neutral Bias — Wait for clear market structure before trading."

    return bias, css_class, message


# ---------------- MAIN ----------------
try:
    gold_model, jpy_model = load_models()

    gold_data = fetch_all_data("GC=F")
    jpy_data = fetch_all_data("JPY=X")

    gold_X = create_features(gold_data)
    jpy_X = create_features(jpy_data)

    gold_bias, gold_class, gold_msg = get_bias_and_message(gold_model, gold_X)
    jpy_bias, jpy_class, jpy_msg = get_bias_and_message(jpy_model, jpy_X)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="neon-card">
            <div>🟡 XAUUSD (Gold)</div>
            <div class="{gold_class}">{gold_bias}</div>
            <div class="message">{gold_msg}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="neon-card">
            <div>💴 USDJPY</div>
            <div class="{jpy_class}">{jpy_bias}</div>
            <div class="message">{jpy_msg}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.write(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error(f"❌ Error loading data or model: {e}")
