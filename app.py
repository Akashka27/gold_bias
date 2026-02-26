import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime, timedelta
import pytz
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Neon Bias Dashboard", layout="wide")

# ---------------- NEON CSS ----------------
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
    text-shadow: 0 0 12px #00ff9f, 0 0 25px #00ff9f;
}
.bearish {
    color: #ff4d4d;
    font-size: 36px;
    text-shadow: 0 0 12px #ff4d4d, 0 0 25px #ff4d4d;
}
.refresh-button {
    background: linear-gradient(45deg, #00ff9f, #00ccff);
    color: black;
    border: none;
    padding: 10px 30px;
    border-radius: 25px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px rgba(0,255,255,0.5);
}
.refresh-button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,255,255,0.8);
}
.last-updated {
    color: #888;
    font-size: 14px;
    text-align: right;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE INIT ----------------
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'cache_timestamp' not in st.session_state:
    st.session_state.cache_timestamp = {}

# ---------------- REFRESH FUNCTION ----------------
def refresh_data():
    """Clear cache and force data refresh"""
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
    st.session_state.data_cache = {}
    st.session_state.cache_timestamp = {}
    st.rerun()

# ---------------- DATA LOADING FUNCTION WITH CACHE ----------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_ticker_data(ticker, period="1mo"):
    """Load ticker data with caching"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        return None

# ---------------- HEADER WITH REFRESH ----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
with col2:
    st.title("🤖 AI NEON BIAS DASHBOARD")
    st.markdown("#### Real-time Market Sentiment with Neon Glow")
with col3:
    st.markdown(f"<div class='last-updated'>Last Updated:<br>{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} ET</div>", unsafe_allow_html=True)
    
    # Refresh button with custom styling
    if st.button("🔄 REFRESH DATA", key="refresh_btn", use_container_width=True):
        refresh_data()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    # Auto-refresh options
    st.subheader("Auto-Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.selectbox("Refresh interval", 
                                       ["30 seconds", "1 minute", "5 minutes", "10 minutes"],
                                       index=2)
        # Convert to seconds
        interval_map = {"30 seconds": 30, "1 minute": 60, "5 minutes": 300, "10 minutes": 600}
        
        # Auto-refresh logic
        if st.session_state.last_refresh < datetime.now(pytz.timezone('US/Eastern')) - timedelta(seconds=interval_map[refresh_interval]):
            refresh_data()
    
    st.markdown("---")
    
    # Manual refresh button in sidebar
    if st.button("🔄 Refresh Now", use_container_width=True):
        refresh_data()
    
    st.markdown("---")
    
    # Stock selection
    st.subheader("📊 Stock Selection")
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    selected_tickers = st.multiselect(
        "Choose stocks to track",
        options=default_tickers + ["META", "NFLX", "PYPL", "ADBE", "INTC", "AMD"],
        default=default_tickers[:3]
    )
    
    if not selected_tickers:
        st.warning("Please select at least one stock")
        selected_tickers = ["AAPL"]  # Default
    
    st.markdown("---")
    
    # Analysis period
    period = st.selectbox(
        "Analysis Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=2
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Signal Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

# ---------------- MAIN DASHBOARD ----------------
st.markdown("## 🔥 Live Market Bias")

# Create placeholder for dynamic updates
main_placeholder = st.empty()

with main_placeholder.container():
    # Create columns for each ticker
    cols = st.columns(len(selected_tickers))
    
    # Store predictions for summary
    predictions = []
    
    for idx, ticker in enumerate(selected_tickers):
        with cols[idx]:
            # Load data
            data = load_ticker_data(ticker, period)
            
            if data is not None and not data.empty:
                # Calculate basic indicators
                latest_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else latest_price
                price_change = latest_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
                
                # Simple bias calculation (placeholder - replace with actual ML model)
                # For demo: Bias based on simple moving average
                sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) > 50 else sma_20
                
                if pd.notna(sma_20) and pd.notna(sma_50):
                    if latest_price > sma_20 and sma_20 > sma_50:
                        bias = "BULLISH"
                        confidence = np.random.uniform(0.75, 0.95)  # Placeholder
                    elif latest_price < sma_20 and sma_20 < sma_50:
                        bias = "BEARISH"
                        confidence = np.random.uniform(0.75, 0.95)  # Placeholder
                    else:
                        bias = "NEUTRAL"
                        confidence = np.random.uniform(0.5, 0.7)   # Placeholder
                else:
                    bias = "NEUTRAL"
                    confidence = 0.5
                
                predictions.append({
                    'ticker': ticker,
                    'bias': bias,
                    'confidence': confidence,
                    'price': latest_price,
                    'change': price_change_pct
                })
                
                # Neon card
                card_class = "bullish" if bias == "BULLISH" else "bearish" if bias == "BEARISH" else ""
                
                st.markdown(f"""
                <div class="neon-card">
                    <h2>{ticker}</h2>
                    <div class="{card_class}">{bias}</div>
                    <h3>${latest_price:.2f}</h3>
                    <p style="color: {'#00ff9f' if price_change_pct > 0 else '#ff4d4d'}">
                        {price_change_pct:+.2f}%
                    </p>
                    <p>Confidence: {confidence:.1%}</p>
                    <p style="font-size:12px; color:#666">
                        Vol: {data['Volume'].iloc[-1]:,.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Mini chart
                st.line_chart(data['Close'].tail(30))
            else:
                st.error(f"⚠️ No data for {ticker}")

    # ---------------- SUMMARY SECTION ----------------
    if predictions:
        st.markdown("---")
        st.markdown("## 📈 Market Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bullish_count = sum(1 for p in predictions if p['bias'] == "BULLISH" and p['confidence'] >= confidence_threshold)
            st.metric("Bullish Signals", bullish_count)
        
        with col2:
            bearish_count = sum(1 for p in predictions if p['bias'] == "BEARISH" and p['confidence'] >= confidence_threshold)
            st.metric("Bearish Signals", bearish_count)
        
        with col3:
            neutral_count = sum(1 for p in predictions if p['bias'] == "NEUTRAL")
            st.metric("Neutral", neutral_count)
        
        with col4:
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Detailed table
        st.markdown("### 📊 Detailed Analysis")
        df_summary = pd.DataFrame(predictions)
        df_summary['price'] = df_summary['price'].map('${:.2f}'.format)
        df_summary['change'] = df_summary['change'].map('{:.2f}%'.format)
        df_summary['confidence'] = df_summary['confidence'].map('{:.1%}'.format)
        df_summary.columns = ['Ticker', 'Bias', 'Confidence', 'Price', 'Change']
        st.dataframe(df_summary, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 AI-Powered Market Analysis | Data updates every 5 minutes | Click REFRESH for latest data</p>
    <p style='font-size: 12px;'>⚠️ This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- AUTO-REFRESH SCRIPT ----------------
if auto_refresh:
    time_to_refresh = st.session_state.last_refresh + timedelta(seconds=interval_map[refresh_interval])
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    
    if current_time >= time_to_refresh:
        refresh_data()
    else:
        time_left = time_to_refresh - current_time
        st.sidebar.info(f"⏰ Next refresh in: {time_left.seconds} seconds")
        
