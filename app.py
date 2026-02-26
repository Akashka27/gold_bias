import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from datetime import datetime, timedelta
import pytz
import time
import warnings
warnings.filterwarnings('ignore')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Daily Bias + M15 Levels | Gold & USD/JPY", layout="wide")

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
.daily-bias-card {
    background: linear-gradient(145deg, #111827, #1a1f2e);
    border-left: 4px solid #00ccff;
    margin-bottom: 20px;
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
.sideways {
    color: #ffd700;
    font-size: 36px;
    text-shadow: 0 0 12px #ffd700, 0 0 25px #ffd700;
}
.level-box {
    background: #1e2434;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid;
}
.support {
    border-left-color: #00ff9f;
}
.resistance {
    border-left-color: #ff4d4d;
}
.risk-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 15px;
    font-weight: bold;
    margin: 5px;
}
.risk-low {
    background: #00ff9f20;
    color: #00ff9f;
    border: 1px solid #00ff9f;
}
.risk-medium {
    background: #ffd70020;
    color: #ffd700;
    border: 1px solid #ffd700;
}
.risk-high {
    background: #ff4d4d20;
    color: #ff4d4d;
    border: 1px solid #ff4d4d;
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
.trading-signal-box {
    background: #1a1f2e;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    border: 1px solid #2a2f3e;
}
.pip-value {
    font-family: monospace;
    font-size: 14px;
    color: #888;
}
.timeframe-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    margin: 0 5px;
}
.daily-badge {
    background: #00ccff20;
    color: #00ccff;
    border: 1px solid #00ccff;
}
.m15-badge {
    background: #ffd70020;
    color: #ffd700;
    border: 1px solid #ffd700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE INIT ----------------
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = {}
if 'daily_model' not in st.session_state:
    st.session_state.daily_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# ---------------- CONSTANTS ----------------
GOLD_TICKER = "GC=F"
USDJPY_TICKER = "JPY=X"
DAILY_INTERVAL = "1d"
M15_INTERVAL = "15m"

ASSET_INFO = {
    GOLD_TICKER: {
        "name": "Gold Futures",
        "icon": "🏆",
        "pip_value": 0.1,
        "contract_size": 100,
        "margin_requirement": 0.01,
        "decimal_places": 2,
        "currency": "$",
        "typical_spread": 0.3,
        "daily_range_pips": 1500
    },
    USDJPY_TICKER: {
        "name": "USD/JPY",
        "icon": "💱",
        "pip_value": 0.001,
        "contract_size": 100000,
        "margin_requirement": 0.02,
        "decimal_places": 3,
        "currency": "¥",
        "typical_spread": 0.8,
        "daily_range_pips": 80
    }
}

# ---------------- REFRESH FUNCTION ----------------
def refresh_data():
    """Clear cache and force data refresh"""
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
    st.session_state.data_cache = {}
    st.session_state.model_predictions = {}
    st.rerun()

# ---------------- DATA LOADING FUNCTIONS ----------------
@st.cache_data(ttl=3600)  # Cache for 1 hour for daily data
def load_daily_data(ticker):
    """Load daily timeframe data for model prediction"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="3mo", interval="1d")
        return data
    except Exception as e:
        st.error(f"Error loading daily data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute for M15 data
def load_m15_data(ticker):
    """Load M15 timeframe data for entries/exits"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="15m")
        if data.empty or len(data) < 50:
            data = stock.history(period="1mo", interval="15m")
        return data
    except Exception as e:
        st.error(f"Error loading M15 data for {ticker}: {str(e)}")
        return None

# ---------------- TECHNICAL INDICATORS FOR DAILY MODEL ----------------
def calculate_daily_features(data):
    """Calculate features for daily model prediction"""
    df = data.copy()
    
    # Daily timeframe indicators
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
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    
    # ATR
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Volume (if available)
    if 'Volume' in df.columns:
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price action
    df['Day_Pct_Change'] = df['Close'].pct_change() * 100
    
    return df

# ---------------- LOAD AI MODEL ----------------
@st.cache_resource
def load_daily_model():
    """Load the pre-trained daily bias model"""
    try:
        # In production, load your actual trained model
        # model = joblib.load('daily_bias_model.pkl')
        # scaler = joblib.load('scaler.pkl')
        
        # For demo, we'll simulate the model with rules
        model_info = {
            'type': 'daily_bias_model',
            'version': '1.0',
            'accuracy': 0.75,
            'features': ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'Volume']
        }
        
        return model_info
    except Exception as e:
        st.warning(f"Could not load daily model: {str(e)}. Using technical analysis.")
        return None

# ---------------- DAILY BIAS PREDICTION ----------------
def predict_daily_bias(data, ticker, model):
    """Predict daily bias using the trained model"""
    df = calculate_daily_features(data)
    latest = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    # This is where you'd use your actual trained model
    # For demo, we're using a rule-based system that simulates model output
    
    # Features for model
    features = {
        'price': latest['Close'],
        'sma_20': latest['SMA_20'],
        'sma_50': latest['SMA_50'],
        'ema_12': latest['EMA_12'],
        'ema_26': latest['EMA_26'],
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_signal'],
        'atr': latest['ATR'],
        'volume_ratio': latest.get('Volume_ratio', 1),
        'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']) if not pd.isna(latest['BB_upper']) and not pd.isna(latest['BB_lower']) else 0.5
    }
    
    # Calculate daily bias score (simulating model prediction)
    score = 0
    signals = []
    
    # Trend following (weights based on model importance)
    if not pd.isna(features['sma_20']) and not pd.isna(features['sma_50']):
        if features['price'] > features['sma_20'] and features['sma_20'] > features['sma_50']:
            score += 1.5
            signals.append("Golden Cross (Daily)")
        elif features['price'] < features['sma_20'] and features['sma_20'] < features['sma_50']:
            score -= 1.5
            signals.append("Death Cross (Daily)")
    
    # RSI signals
    if not pd.isna(features['rsi']):
        if features['rsi'] < 30:
            score += 1.2
            signals.append("Oversold (Daily)")
        elif features['rsi'] > 70:
            score -= 1.2
            signals.append("Overbought (Daily)")
    
    # MACD signals
    if not pd.isna(features['macd']) and not pd.isna(features['macd_signal']):
        if features['macd'] > features['macd_signal']:
            score += 1.3
            signals.append("MACD Bullish (Daily)")
        else:
            score -= 1.3
            signals.append("MACD Bearish (Daily)")
    
    # Bollinger Bands
    if not pd.isna(features['bb_position']):
        if features['bb_position'] < 0.2:
            score += 1.0
            signals.append("BB Oversold")
        elif features['bb_position'] > 0.8:
            score -= 1.0
            signals.append("BB Overbought")
    
    # Determine daily bias
    if score > 1.5:
        bias = "BULLISH"
        confidence = min(0.6 + abs(score) * 0.1, 0.9)
        strength = "STRONG" if score > 3 else "MODERATE"
    elif score < -1.5:
        bias = "BEARISH"
        confidence = min(0.6 + abs(score) * 0.1, 0.9)
        strength = "STRONG" if score < -3 else "MODERATE"
    else:
        bias = "SIDEWAYS"
        confidence = 0.5 + abs(score) * 0.1
        strength = "RANGING"
    
    # Get top signals
    signals = list(set(signals))[:3]
    
    # Calculate key daily levels
    daily_high = df['High'].tail(20).max()
    daily_low = df['Low'].tail(20).min()
    daily_range = daily_high - daily_low
    
    return {
        'bias': bias,
        'strength': strength,
        'confidence': min(confidence, 0.95),
        'score': score,
        'signals': signals,
        'daily_high': daily_high,
        'daily_low': daily_low,
        'daily_range': daily_range,
        'previous_close': prev_day['Close'],
        'open': latest['Open'],
        'current_price': latest['Close'],
        'timestamp': datetime.now()
    }

# ---------------- M15 LEVELS DETECTION ----------------
def detect_m15_levels(data, ticker):
    """Detect key support and resistance levels for M15 entries/exits"""
    df = data.copy()
    latest = df.iloc[-1]
    
    # Calculate M15 indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    asset = ASSET_INFO[ticker]
    current_price = latest['Close']
    
    # Find pivot points for M15
    recent_highs = df['High'].rolling(window=10).max()
    recent_lows = df['Low'].rolling(window=10).min()
    
    pivot_highs = []
    pivot_lows = []
    
    for i in range(5, len(df)-5):
        if df['High'].iloc[i] == recent_highs.iloc[i]:
            pivot_highs.append(df['High'].iloc[i])
        if df['Low'].iloc[i] == recent_lows.iloc[i]:
            pivot_lows.append(df['Low'].iloc[i])
    
    # Get nearest levels
    resistance_levels = sorted([p for p in pivot_highs if p > current_price])[:3]
    support_levels = sorted([p for p in pivot_lows if p < current_price], reverse=True)[:3]
    
    # Format levels with distances
    m15_resistance = []
    for price in resistance_levels:
        distance_pips = (price - current_price) / asset['pip_value']
        m15_resistance.append({
            'price': price,
            'distance_pips': distance_pips,
            'distance_pct': ((price - current_price) / current_price) * 100
        })
    
    m15_support = []
    for price in support_levels:
        distance_pips = (current_price - price) / asset['pip_value']
        m15_support.append({
            'price': price,
            'distance_pips': distance_pips,
            'distance_pct': ((current_price - price) / current_price) * 100
        })
    
    return {
        'support': m15_support,
        'resistance': m15_resistance,
        'current_price': current_price,
        'atr': latest['ATR_14'],
        'atr_pips': latest['ATR_14'] / asset['pip_value'],
        'sma_20': latest['SMA_20']
    }

# ---------------- RISK MANAGEMENT ----------------
def calculate_trade_setup(daily_bias, m15_levels, ticker, account_balance=10000, risk_percent=0.02):
    """Calculate trade setup based on daily bias and M15 levels"""
    
    asset = ASSET_INFO[ticker]
    current_price = m15_levels['current_price']
    atr = m15_levels['atr']
    bias = daily_bias['bias']
    confidence = daily_bias['confidence']
    
    risk_amount = account_balance * risk_percent
    
    # Only generate trades if daily bias is clear
    if bias == "SIDEWAYS":
        return {
            'action': 'HOLD',
            'reason': 'Daily bias is sideways',
            'daily_bias': bias
        }
    
    if bias == "BULLISH":
        # Look for buying opportunities on M15
        if len(m15_levels['support']) > 0:
            # Buy near support
            entry_zone_min = m15_levels['support'][0]['price'] * 0.998
            entry_zone_max = m15_levels['support'][0]['price'] * 1.002
            
            # Stop loss below recent low or 1.5x ATR
            stop_loss = min(
                m15_levels['support'][0]['price'] - (atr * 0.5),
                current_price - (atr * 1.2)
            )
            
            # Calculate position
            stop_distance = current_price - stop_loss
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
                if ticker == GOLD_TICKER:
                    position_size = position_size / 100
                elif ticker == USDJPY_TICKER:
                    position_size = position_size / 100000
                
                # Take profits based on daily bias
                tp1 = current_price + (stop_distance * 2)
                tp2 = current_price + (stop_distance * 3)
                
                # Alternative TP at daily high
                tp_alt = daily_bias['daily_high']
                tp_alt_rr = (tp_alt - current_price) / stop_distance if tp_alt > current_price else None
                
                return {
                    'action': 'BUY',
                    'entry_zone': (entry_zone_min, entry_zone_max),
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit_1': tp1,
                    'take_profit_2': tp2,
                    'take_profit_alt': tp_alt,
                    'tp_alt_rr': tp_alt_rr,
                    'position_size': position_size,
                    'position_size_units': f"{position_size:.2f} {'contracts' if ticker == GOLD_TICKER else 'lots'}",
                    'risk_amount': risk_amount,
                    'risk_reward_1': 2.0,
                    'risk_reward_2': 3.0,
                    'stop_distance_pips': stop_distance / asset['pip_value'],
                    'stop_distance_pct': (stop_distance / current_price) * 100,
                    'confidence': confidence,
                    'daily_bias': bias,
                    'trade_rationale': 'Buy on dips with daily bullish bias'
                }
    
    elif bias == "BEARISH":
        # Look for selling opportunities on M15
        if len(m15_levels['resistance']) > 0:
            # Sell near resistance
            entry_zone_min = m15_levels['resistance'][0]['price'] * 0.998
            entry_zone_max = m15_levels['resistance'][0]['price'] * 1.002
            
            # Stop loss above recent high or 1.5x ATR
            stop_loss = max(
                m15_levels['resistance'][0]['price'] + (atr * 0.5),
                current_price + (atr * 1.2)
            )
            
            # Calculate position
            stop_distance = stop_loss - current_price
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
                if ticker == GOLD_TICKER:
                    position_size = position_size / 100
                elif ticker == USDJPY_TICKER:
                    position_size = position_size / 100000
                
                # Take profits based on daily bias
                tp1 = current_price - (stop_distance * 2)
                tp2 = current_price - (stop_distance * 3)
                
                # Alternative TP at daily low
                tp_alt = daily_bias['daily_low']
                tp_alt_rr = (current_price - tp_alt) / stop_distance if tp_alt < current_price else None
                
                return {
                    'action': 'SELL',
                    'entry_zone': (entry_zone_min, entry_zone_max),
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit_1': tp1,
                    'take_profit_2': tp2,
                    'take_profit_alt': tp_alt,
                    'tp_alt_rr': tp_alt_rr,
                    'position_size': position_size,
                    'position_size_units': f"{position_size:.2f} {'contracts' if ticker == GOLD_TICKER else 'lots'}",
                    'risk_amount': risk_amount,
                    'risk_reward_1': 2.0,
                    'risk_reward_2': 3.0,
                    'stop_distance_pips': stop_distance / asset['pip_value'],
                    'stop_distance_pct': (stop_distance / current_price) * 100,
                    'confidence': confidence,
                    'daily_bias': bias,
                    'trade_rationale': 'Sell on rallies with daily bearish bias'
                }
    
    # No valid setup
    return {
        'action': 'HOLD',
        'reason': f'No clear M15 setup for {bias} bias',
        'daily_bias': bias,
        'confidence': confidence
    }

# ---------------- FORMAT PRICE ----------------
def format_price(price, ticker):
    """Format price based on asset type"""
    asset = ASSET_INFO[ticker]
    if ticker == GOLD_TICKER:
        return f"${price:.2f}"
    else:
        return f"¥{price:.3f}"

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("<h1 style='text-align: center;'>🏆 💱</h1>", unsafe_allow_html=True)
with col2:
    st.title("🤖 DAILY BIAS + M15 LEVELS")
    st.markdown("#### AI Daily Direction | M15 Entries | Risk Management")
with col3:
    st.markdown(f"<div class='last-updated'>Last Updated:<br>{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} ET</div>", unsafe_allow_html=True)
    
    if st.button("🔄 REFRESH DATA", key="refresh_btn", use_container_width=True):
        refresh_data()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Trading System Controls")
    
    # Auto-refresh
    st.subheader("🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    if auto_refresh:
        refresh_interval = st.selectbox("Refresh interval", 
                                       ["5 minutes", "15 minutes", "30 minutes"],
                                       index=1)
        interval_map = {"5 minutes": 300, "15 minutes": 900, "30 minutes": 1800}
        
        time_since_refresh = (datetime.now(pytz.timezone('US/Eastern')) - st.session_state.last_refresh).seconds
        if time_since_refresh > interval_map[refresh_interval]:
            refresh_data()
    
    st.markdown("---")
    
    # Account Settings
    st.subheader("💰 Risk Management")
    account_balance = st.number_input("Account Balance ($)", 
                                      min_value=1000, 
                                      max_value=1000000, 
                                      value=10000,
                                      step=1000)
    
    risk_percent = st.slider("Risk per Trade (%)", 
                             min_value=0.5, 
                             max_value=5.0, 
                             value=2.0,
                             step=0.5) / 100
    
    st.markdown("---")
    
    # Model Settings
    st.subheader("🧠 AI Model Settings")
    confidence_threshold = st.slider("Daily Bias Confidence", 
                                     min_value=0.5, 
                                     max_value=0.9, 
                                     value=0.65,
                                     step=0.05,
                                     help="Minimum confidence for daily bias")
    
    min_rr_ratio = st.slider("Minimum Risk-Reward", 
                             min_value=1.0, 
                             max_value=3.0, 
                             value=1.5,
                             step=0.1,
                             help="Minimum risk-reward for M15 entries")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Trading Strategy
    
    1. **Daily Bias** - AI model predicts overall direction
    2. **M15 Levels** - Find entries in direction of daily bias
    3. **Risk Management** - 2% risk per trade, 2:1 RR minimum
    """)

# ---------------- LOAD MODEL ----------------
daily_model = load_daily_model()

# ---------------- MAIN DASHBOARD ----------------
st.markdown("## 📊 Daily Bias + M15 Setups")

# Create two columns
col1, col2 = st.columns(2)

# Store data
daily_predictions = {}
trade_setups = {}

# ---------------- GOLD ANALYSIS ----------------
with col1:
    st.markdown(f"### 🏆 GOLD (GC=F)")
    
    with st.spinner("Analyzing Gold..."):
        # Load daily data for model
        daily_data = load_daily_data(GOLD_TICKER)
        # Load M15 data for entries
        m15_data = load_m15_data(GOLD_TICKER)
        
        if daily_data is not None and m15_data is not None:
            # Get daily bias from model
            daily_bias = predict_daily_bias(daily_data, GOLD_TICKER, daily_model)
            daily_predictions['GOLD'] = daily_bias
            
            # Get M15 levels
            m15_levels = detect_m15_levels(m15_data, GOLD_TICKER)
            
            # Calculate trade setup
            trade_setup = calculate_trade_setup(daily_bias, m15_levels, GOLD_TICKER, account_balance, risk_percent)
            trade_setups['GOLD'] = trade_setup
            
            # Display Daily Bias Card
            bias_class = daily_bias['bias'].lower()
            if daily_bias['bias'] == "BULLISH":
                bias_emoji = "🚀"
            elif daily_bias['bias'] == "BEARISH":
                bias_emoji = "📉"
            else:
                bias_emoji = "⏸️"
            
            st.markdown(f"""
            <div class="neon-card daily-bias-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="timeframe-badge daily-badge">DAILY TIMEFRAME</span>
                    <span class="timeframe-badge m15-badge">M15 EXECUTION</span>
                </div>
                <h2 class="gold-text">🏆 GOLD DAILY BIAS</h2>
                <div class="{bias_class}">{bias_emoji} {daily_bias['bias']} {bias_emoji}</div>
                <p style="color: #888">Strength: {daily_bias['strength']} | Confidence: {daily_bias['confidence']:.1%}</p>
                <p style="color: #888">Daily Range: {format_price(daily_bias['daily_high'], GOLD_TICKER)} - {format_price(daily_bias['daily_low'], GOLD_TICKER)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Daily Signals
            if daily_bias['signals']:
                st.markdown("##### 📊 Daily AI Signals")
                cols = st.columns(len(daily_bias['signals']))
                for i, signal in enumerate(daily_bias['signals']):
                    with cols[i]:
                        st.info(signal)
            
            # M15 Levels
            st.markdown("### 🎯 M15 Key Levels")
            
            m15_col1, m15_col2 = st.columns(2)
            
            with m15_col1:
                st.markdown("##### 🟢 Support Levels")
                if m15_levels['support']:
                    for level in m15_levels['support']:
                        st.markdown(f"""
                        <div class="level-box support">
                            <strong>{format_price(level['price'], GOLD_TICKER)}</strong><br>
                            <small>{level['distance_pips']:.0f} pips away</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No nearby support")
            
            with m15_col2:
                st.markdown("##### 🔴 Resistance Levels")
                if m15_levels['resistance']:
                    for level in m15_levels['resistance']:
                        st.markdown(f"""
                        <div class="level-box resistance">
                            <strong>{format_price(level['price'], GOLD_TICKER)}</strong><br>
                            <small>{level['distance_pips']:.0f} pips away</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No nearby resistance")
            
            # Trade Setup
            st.markdown("### 💼 Trade Setup")
            
            if trade_setup['action'] != 'HOLD' and daily_bias['confidence'] >= confidence_threshold:
                action_color = "#00ff9f" if trade_setup['action'] == "BUY" else "#ff4d4d"
                
                st.markdown(f"""
                <div style="background: #1a1f2e; padding: 20px; border-radius: 10px; border-left: 4px solid {action_color};">
                    <h3 style="color: {action_color}; margin:0">{trade_setup['action']} SIGNAL</h3>
                    <p style="color: #888">{trade_setup['trade_rationale']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Entry Zone
                st.markdown(f"**Entry Zone:** {format_price(trade_setup['entry_zone'][0], GOLD_TICKER)} - {format_price(trade_setup['entry_zone'][1], GOLD_TICKER)}")
                st.markdown(f"**Current Price:** {format_price(trade_setup['current_price'], GOLD_TICKER)}")
                
                # Trade Levels
                col_tp1, col_tp2, col_sl = st.columns(3)
                with col_tp1:
                    st.metric("Stop Loss", format_price(trade_setup['stop_loss'], GOLD_TICKER))
                with col_tp2:
                    st.metric("Take Profit 1", format_price(trade_setup['take_profit_1'], GOLD_TICKER))
                with col_sl:
                    st.metric("Take Profit 2", format_price(trade_setup['take_profit_2'], GOLD_TICKER))
                
                # Position Sizing
                st.markdown("##### 📊 Position Size")
                col_ps1, col_ps2, col_ps3 = st.columns(3)
                with col_ps1:
                    st.metric("Size", trade_setup['position_size_units'])
                with col_ps2:
                    st.metric("Risk", f"${trade_setup['risk_amount']:.2f}")
                with col_ps3:
                    st.metric("RR Ratio", f"1:{trade_setup['risk_reward_1']:.2f}")
                
                # Alternative TP
                if trade_setup['take_profit_alt']:
                    st.info(f"🎯 Daily Target: {format_price(trade_setup['take_profit_alt'], GOLD_TICKER)} (RR: 1:{trade_setup['tp_alt_rr']:.2f})")
                
                # Risk Info
                st.markdown(f"""
                <div style="margin-top: 10px;">
                    <span class="risk-badge risk-medium">Stop Distance: {trade_setup['stop_distance_pips']:.0f} pips</span>
                </div>
                """, unsafe_allow_html=True)
                
            elif daily_bias['bias'] == "SIDEWAYS":
                st.warning("⏸️ Daily bias is sideways. Wait for clearer direction.")
            elif daily_bias['confidence'] < confidence_threshold:
                st.warning(f"⚠️ Daily confidence ({daily_bias['confidence']:.1%}) below threshold")
            else:
                st.info(f"⏸️ No {daily_bias['bias']} setup on M15. Wait for pullback to levels.")
            
            # M15 Chart
            with st.expander("📈 M15 Price Chart"):
                st.line_chart(m15_data['Close'].tail(50))

# ---------------- USD/JPY ANALYSIS ----------------
with col2:
    st.markdown(f"### 💱 USD/JPY")
    
    with st.spinner("Analyzing USD/JPY..."):
        # Load daily data for model
        daily_data = load_daily_data(USDJPY_TICKER)
        # Load M15 data for entries
        m15_data = load_m15_data(USDJPY_TICKER)
        
        if daily_data is not None and m15_data is not None:
            # Get daily bias from model
            daily_bias = predict_daily_bias(daily_data, USDJPY_TICKER, daily_model)
            daily_predictions['USDJPY'] = daily_bias
            
            # Get M15 levels
            m15_levels = detect_m15_levels(m15_data, USDJPY_TICKER)
            
            # Calculate trade setup
            trade_setup = calculate_trade_setup(daily_bias, m15_levels, USDJPY_TICKER, account_balance, risk_percent)
            trade_setups['
