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
st.set_page_config(page_title="AI Neon Bias Dashboard - Gold & USD/JPY", layout="wide")

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
.gold-card {
    background: linear-gradient(145deg, #111827, #1a1f2e);
    border-left: 4px solid #ffd700;
}
.forex-card {
    background: linear-gradient(145deg, #111827, #1a1f2e);
    border-left: 4px solid #00ccff;
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
    background: linear-gradient(45deg, #ffd700, #00ccff);
    color: black;
    border: none;
    padding: 10px 30px;
    border-radius: 25px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px rgba(255,215,0,0.5);
}
.refresh-button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(255,215,0,0.8);
}
.last-updated {
    color: #888;
    font-size: 14px;
    text-align: right;
    padding: 10px;
}
.gold-text {
    color: #ffd700;
    text-shadow: 0 0 10px rgba(255,215,0,0.5);
}
.forex-text {
    color: #00ccff;
    text-shadow: 0 0 10px rgba(0,204,255,0.5);
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
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# ---------------- CONSTANTS ----------------
GOLD_TICKER = "GC=F"
USDJPY_TICKER = "JPY=X"

ASSET_INFO = {
    GOLD_TICKER: {
        "name": "Gold Futures",
        "icon": "🏆",
        "color": "gold-text",
        "card_class": "gold-card"
    },
    USDJPY_TICKER: {
        "name": "USD/JPY",
        "icon": "💱",
        "color": "forex-text",
        "card_class": "forex-card"
    }
}

# ---------------- REFRESH FUNCTION ----------------
def refresh_data():
    """Clear cache and force data refresh"""
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
    st.session_state.data_cache = {}
    st.session_state.cache_timestamp = {}
    st.rerun()

# ---------------- DATA LOADING FUNCTION WITH CACHE ----------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_ticker_data(ticker, period="1mo", interval="1h"):
    """Load ticker data with caching"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        return None

# ---------------- TECHNICAL INDICATORS ----------------
def calculate_technical_indicators(data):
    """Calculate technical indicators for analysis"""
    df = data.copy()
    
    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    df['BB_width'] = df['BB_high'] - df['BB_low']
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Volume (if available)
    if 'Volume' in df.columns:
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
    
    # Support and Resistance Levels
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    
    return df

# ---------------- BIAS CALCULATION ----------------
def calculate_bias(data, ticker):
    """Calculate market bias based on technical indicators"""
    df = calculate_technical_indicators(data)
    latest = df.iloc[-1]
    
    # Initialize signals
    signals = []
    weights = []
    
    # Trend signals
    if not pd.isna(latest['SMA_20']) and not pd.isna(latest['SMA_50']):
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals.append(1)  # Bullish
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            signals.append(-1)  # Bearish
        else:
            signals.append(0)
        weights.append(1.5)
    
    # MACD signals
    if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal']:
            signals.append(1)
        else:
            signals.append(-1)
        weights.append(1.2)
    
    # RSI signals
    if not pd.isna(latest['RSI']):
        if latest['RSI'] < 30:
            signals.append(1)  # Oversold - Bullish
        elif latest['RSI'] > 70:
            signals.append(-1)  # Overbought - Bearish
        else:
            signals.append(0)
        weights.append(1.3)
    
    # Bollinger Bands signals
    if not pd.isna(latest['BB_low']) and not pd.isna(latest['BB_high']):
        if latest['Close'] <= latest['BB_low']:
            signals.append(1)  # At lower band - Bullish
        elif latest['Close'] >= latest['BB_high']:
            signals.append(-1)  # At upper band - Bearish
        else:
            signals.append(0)
        weights.append(1.0)
    
    # Stochastic signals
    if not pd.isna(latest['Stoch_K']) and not pd.isna(latest['Stoch_D']):
        if latest['Stoch_K'] < 20 and latest['Stoch_K'] > latest['Stoch_D']:
            signals.append(1)  # Oversold - Bullish
        elif latest['Stoch_K'] > 80 and latest['Stoch_K'] < latest['Stoch_D']:
            signals.append(-1)  # Overbought - Bearish
        else:
            signals.append(0)
        weights.append(1.1)
    
    # Calculate weighted bias
    if signals:
        weighted_sum = sum(s * w for s, w in zip(signals, weights))
        total_weight = sum(weights)
        
        # Normalize to -1 to 1 range
        bias_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine bias and confidence
        if bias_score > 0.3:
            bias = "BULLISH"
            confidence = min(0.5 + abs(bias_score) * 0.5, 0.95)
        elif bias_score < -0.3:
            bias = "BEARISH"
            confidence = min(0.5 + abs(bias_score) * 0.5, 0.95)
        else:
            bias = "NEUTRAL"
            confidence = 0.5 + abs(bias_score) * 0.3
        
        return bias, confidence, latest
    
    return "NEUTRAL", 0.5, latest

# ---------------- HEADER WITH REFRESH ----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("<h1 style='text-align: center;'>🏆 💱</h1>", unsafe_allow_html=True)
with col2:
    st.title("🤖 GOLD & USD/JPY AI BIAS DASHBOARD")
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
    
    # Analysis settings
    st.subheader("📊 Analysis Settings")
    
    period = st.selectbox(
        "Analysis Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=2,
        help="Time period for analysis"
    )
    
    interval = st.selectbox(
        "Data Interval",
        ["1h", "1d", "1wk"],
        index=0,
        help="Time interval between data points"
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Signal Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence for signal consideration"
    )
    
    st.markdown("---")
    
    # Market information
    st.subheader("ℹ️ Market Info")
    st.markdown("""
    **Gold (GC=F)**
    - Trading Hours: 24/5
    - Contract: 100 troy ounces
    - Exchange: COMEX
    
    **USD/JPY (JPY=X)**
    - Trading Hours: 24/5
    - Pip Value: ~$9.09 per lot
    - Exchange: Forex
    """)

# ---------------- MAIN DASHBOARD ----------------
st.markdown("## 🔥 Live Market Bias")

# Create placeholder for dynamic updates
main_placeholder = st.empty()

with main_placeholder.container():
    # Create two main columns for Gold and USD/JPY
    col1, col2 = st.columns(2)
    
    # Store predictions for summary
    predictions = []
    
    # Process Gold
    with col1:
        st.markdown(f"### 🏆 GOLD (GC=F)")
        data = load_ticker_data(GOLD_TICKER, period, interval)
        
        if data is not None and not data.empty:
            bias, confidence, latest = calculate_bias(data, GOLD_TICKER)
            
            # Calculate price change
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else latest['Close']
            price_change = latest['Close'] - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            predictions.append({
                'ticker': 'GOLD',
                'bias': bias,
                'confidence': confidence,
                'price': latest['Close'],
                'change': price_change_pct
            })
            
            # Gold card
            card_class = "bullish" if bias == "BULLISH" else "bearish" if bias == "BEARISH" else ""
            
            st.markdown(f"""
            <div class="neon-card gold-card">
                <h2 class="gold-text">🏆 GOLD</h2>
                <div class="{card_class}">{bias}</div>
                <h3>${latest['Close']:.2f}</h3>
                <p style="color: {'#00ff9f' if price_change_pct > 0 else '#ff4d4d'}">
                    {price_change_pct:+.2f}%
                </p>
                <p>Confidence: {confidence:.1%}</p>
                <p style="font-size:12px; color:#666">
                    High: ${latest['High']:.2f} | Low: ${latest['Low']:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Technical indicators for Gold
            with st.expander("📊 Gold Technical Indicators"):
                df_tech = calculate_technical_indicators(data)
                latest_tech = df_tech.iloc[-1]
                
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.metric("RSI (14)", f"{latest_tech['RSI']:.1f}" if not pd.isna(latest_tech['RSI']) else "N/A")
                    st.metric("MACD", f"{latest_tech['MACD']:.2f}" if not pd.isna(latest_tech['MACD']) else "N/A")
                with col_metrics2:
                    st.metric("SMA 20", f"${latest_tech['SMA_20']:.2f}" if not pd.isna(latest_tech['SMA_20']) else "N/A")
                    st.metric("SMA 50", f"${latest_tech['SMA_50']:.2f}" if not pd.isna(latest_tech['SMA_50']) else "N/A")
                with col_metrics3:
                    st.metric("Support", f"${latest_tech['Support']:.2f}" if not pd.isna(latest_tech['Support']) else "N/A")
                    st.metric("Resistance", f"${latest_tech['Resistance']:.2f}" if not pd.isna(latest_tech['Resistance']) else "N/A")
            
            # Gold price chart
            st.subheader("📈 Gold Price Movement")
            st.line_chart(data['Close'])
            
            # Volume chart if available
            if 'Volume' in data.columns and data['Volume'].sum() > 0:
                st.subheader("📊 Gold Volume")
                st.bar_chart(data['Volume'])
        else:
            st.error("⚠️ Unable to load Gold data")
    
    # Process USD/JPY
    with col2:
        st.markdown(f"### 💱 USD/JPY")
        data = load_ticker_data(USDJPY_TICKER, period, interval)
        
        if data is not None and not data.empty:
            bias, confidence, latest = calculate_bias(data, USDJPY_TICKER)
            
            # Calculate price change
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else latest['Close']
            price_change = latest['Close'] - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            predictions.append({
                'ticker': 'USD/JPY',
                'bias': bias,
                'confidence': confidence,
                'price': latest['Close'],
                'change': price_change_pct
            })
            
            # USD/JPY card
            card_class = "bullish" if bias == "BULLISH" else "bearish" if bias == "BEARISH" else ""
            
            st.markdown(f"""
            <div class="neon-card forex-card">
                <h2 class="forex-text">💱 USD/JPY</h2>
                <div class="{card_class}">{bias}</div>
                <h3>¥{latest['Close']:.3f}</h3>
                <p style="color: {'#00ff9f' if price_change_pct > 0 else '#ff4d4d'}">
                    {price_change_pct:+.3f}%
                </p>
                <p>Confidence: {confidence:.1%}</p>
                <p style="font-size:12px; color:#666">
                    High: ¥{latest['High']:.3f} | Low: ¥{latest['Low']:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Technical indicators for USD/JPY
            with st.expander("📊 USD/JPY Technical Indicators"):
                df_tech = calculate_technical_indicators(data)
                latest_tech = df_tech.iloc[-1]
                
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.metric("RSI (14)", f"{latest_tech['RSI']:.1f}" if not pd.isna(latest_tech['RSI']) else "N/A")
                    st.metric("MACD", f"{latest_tech['MACD']:.3f}" if not pd.isna(latest_tech['MACD']) else "N/A")
                with col_metrics2:
                    st.metric("SMA 20", f"¥{latest_tech['SMA_20']:.3f}" if not pd.isna(latest_tech['SMA_20']) else "N/A")
                    st.metric("SMA 50", f"¥{latest_tech['SMA_50']:.3f}" if not pd.isna(latest_tech['SMA_50']) else "N/A")
                with col_metrics3:
                    st.metric("Support", f"¥{latest_tech['Support']:.3f}" if not pd.isna(latest_tech['Support']) else "N/A")
                    st.metric("Resistance", f"¥{latest_tech['Resistance']:.3f}" if not pd.isna(latest_tech['Resistance']) else "N/A")
            
            # USD/JPY price chart
            st.subheader("📈 USD/JPY Price Movement")
            st.line_chart(data['Close'])
        else:
            st.error("⚠️ Unable to load USD/JPY data")
    
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
        df_summary['price'] = df_summary.apply(
            lambda x: f"${x['price']:.2f}" if x['ticker'] == 'GOLD' else f"¥{x['price']:.3f}", 
            axis=1
        )
        df_summary['change'] = df_summary['change'].map('{:.3f}%'.format)
        df_summary['confidence'] = df_summary['confidence'].map('{:.1%}'.format)
        df_summary = df_summary[['ticker', 'bias', 'confidence', 'price', 'change']]
        df_summary.columns = ['Asset', 'Bias', 'Confidence', 'Price', 'Change']
        st.dataframe(df_summary, use_container_width=True)

# ---------------- CORRELATION ANALYSIS ----------------
st.markdown("---")
st.markdown("## 🔗 Gold-USD/JPY Correlation Analysis")

col1, col2 = st.columns(2)

with col1:
    # Load both assets for correlation
    gold_data = load_ticker_data(GOLD_TICKER, "1mo", "1h")
    forex_data = load_ticker_data(USDJPY_TICKER, "1mo", "1h")
    
    if gold_data is not None and forex_data is not None and not gold_data.empty and not forex_data.empty:
        # Align data
        common_dates = gold_data.index.intersection(forex_data.index)
        if len(common_dates) > 0:
            gold_prices = gold_data.loc[common_dates, 'Close']
            forex_prices = forex_data.loc[common_dates, 'Close']
            
            # Calculate correlation
            correlation = gold_prices.corr(forex_prices)
            
            st.metric("Price Correlation", f"{correlation:.3f}", 
                     delta="Inverse" if correlation < 0 else "Direct")
            
            # Scatter plot
            corr_df = pd.DataFrame({
                'Gold Price': gold_prices,
                'USD/JPY': forex_prices
            })
            st.scatter_chart(corr_df)
    else:
        st.info("Insufficient data for correlation analysis")

with col2:
    st.markdown("""
    ### 📊 Correlation Insights
    
    **Gold and USD/JPY typically show:**
    
    - **Inverse correlation** when risk sentiment drives markets
    - Gold as safe-haven vs USD/JPY as risk barometer
    
    **Current signals:**
    """)
    
    if len(predictions) == 2:
        if predictions[0]['bias'] == "BULLISH" and predictions[1]['bias'] == "BEARISH":
            st.success("🔮 Classic safe-haven flow: Gold up, USD/JPY down")
        elif predictions[0]['bias'] == "BEARISH" and predictions[1]['bias'] == "BULLISH":
            st.success("📈 Risk-on sentiment: Gold down, USD/JPY up")
        elif predictions[0]['bias'] == predictions[1]['bias']:
            st.info("⚖️ Unusual correlation - special factors may be at play")

# ---------------- AUTO-REFRESH SCRIPT ----------------
if auto_refresh:
    time_to_refresh = st.session_state.last_refresh + timedelta(seconds=interval_map[refresh_interval])
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    
    if current_time >= time_to_refresh:
        refresh_data()
    else:
        time_left = time_to_refresh - current_time
        st.sidebar.info(f"⏰ Next refresh in: {time_left.seconds} seconds")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 AI-Powered Gold & USD/JPY Analysis | Data updates every 5 minutes | Click REFRESH for latest data</p>
    <p style='font-size: 12px;'>⚠️ This is for educational purposes only. Not financial advice.</p>
    <p style='font-size: 12px;'>🏆 Gold: GC=F | 💱 USD/JPY: JPY=X</p>
</div>
""", unsafe_allow_html=True)
