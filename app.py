import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime
import pytz

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Trading Assistant", layout="wide")

# ---------------- BACKGROUND VIDEO + NEON CSS ----------------
video_url = "https://cdn.coverr.co/videos/coverr-trading-charts-1573/1080p.mp4"

st.markdown(f"""
<style>
/* Fullscreen Video Background */
video {{
    position: fixed;
    right: 0;
    bottom: 0;
    min-width: 100%; 
    min-height: 100%;
    z-index: -2;
    filter: brightness(0.25) blur(2px);
}}

/* Dark overlay for readability */
.main::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(5, 10, 20, 0.75);
    z-index: -1;
}}

/* Neon Title */
.neon-title {{
    font-size: 42px;
    text-align: center;
    color: #00f7ff;
    text-shadow: 0 0 10px #00f7ff,
                 0 0 20px #00f7ff,
                 0 0 40px #00f7ff;
    font-weight: bold;
}}

/* Neon Cards */
.neon-card {{
    background: rgba(15, 23, 42, 0.85);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(0,255,255,0.2);
    box-shadow: 0 0 20px rgba(0,255,255,0.15);
    backdrop-filter: blur(10px);
}}

/* Bias Colors */
.bullish {{
    color: #00ff9f;
    text-shadow: 0 0 10px #00ff9f;
    font-size: 28px;
    font-weight: bold;
}}

.bearish {{
    color: #ff4d4d;
    text-shadow: 0 0 10px #ff4d4d;
    font-size: 28px;
    font-weight: bold;
}}

.neutral {{
    color: #ffd700;
    text-shadow: 0 0 10px #ffd700;
    font-size: 28px;
    font-weight: bold;
}}
</style>

<video autoplay muted loop>
  <source src="{video_url}" type="video/mp4">
</video>
""", unsafe_allow_html=True)

st.markdown('<p class="neon-title">⚡ AI M15 Trading Assistant Terminal</p>', unsafe_allow_html=True)
st.caption("HTF AI Bias + M15 Execution | Gold & USDJPY")
