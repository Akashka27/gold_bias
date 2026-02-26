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
st.set_page_config(page_title="AI M15 Gold & USD/JPY Trading System", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE INIT ----------------
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# ---------------- CONSTANTS ----------------
GOLD_TICKER = "GC=F"
USDJPY_TICKER = "JPY=X"
M15_INTERVAL = "15m"

ASSET_INFO = {
    GOLD_TICKER: {
        "name": "Gold Futures",
        "icon": "🏆",
        "pip_value": 0.01,  # $10 per pip typically
        "contract_size": 100,
        "margin_requirement": 0.01  # 1% margin for futures
    },
    USDJPY_TICKER: {
        "name": "USD/JPY",
        "icon": "💱",
        "pip_value": 0.001,  # 1000 JPY per pip standard lot
        "contract_size": 100000,
        "margin_requirement": 0.02  # 2% margin for forex
    }
}

# ---------------- REFRESH FUNCTION ----------------
def refresh_data():
    """Clear cache and force data refresh"""
    st.session_state.last_refresh = datetime.now(pytz.timezone('US/Eastern'))
    st.session_state.data_cache = {}
    st.session_state.model_predictions = {}
    st.rerun()

# ---------------- DATA LOADING FUNCTION ----------------
@st.cache_data(ttl=60)  # Cache for 1 minute only (M15 needs fresh data)
def load_m15_data(ticker, periods=200):
    """Load M15 timeframe data"""
    try:
        stock = yf.Ticker(ticker)
        # Request more data to ensure we have enough for M15
        data = stock.history(period="5d", interval="15m")
        if data.empty or len(data) < 50:
            # Fallback to more data if needed
            data = stock.history(period="1mo", interval="15m")
        return data
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        return None

# ---------------- ADVANCED TECHNICAL INDICATORS ----------------
def calculate_m15_indicators(data):
    """Calculate comprehensive technical indicators for M15"""
    df = data.copy()
    
    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['MACD_histogram'] = macd.macd_diff()
    
    # RSI with multiple periods
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['RSI_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # ATR for volatility and stop losses
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Support and Resistance Levels
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
    df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
    
    # Volume indicators (if available)
    if 'Volume' in df.columns:
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price action features
    df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
    
    # Candlestick patterns (simplified)
    df['Doji'] = abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1
    df['Hammer'] = ((df['Close'] > df['Open']) & 
                    ((df['High'] - df['Close']) < (df['Open'] - df['Low']) * 0.3) &
                    ((df['Close'] - df['Open']) > (df['High'] - df['Low']) * 0.3))
    df['Shooting_Star'] = ((df['Open'] > df['Close']) & 
                           ((df['High'] - df['Open']) > (df['Open'] - df['Low']) * 2) &
                           ((df['Close'] - df['Low']) < (df['Open'] - df['Close']) * 0.5))
    
    return df

# ---------------- LOAD AI MODELS ----------------
@st.cache_resource
def load_ai_models():
    """Load pre-trained AI models for bias prediction"""
    try:
        # In production, load actual trained models
        # For demo, we'll create ensemble of indicators
        
        # Placeholder for model loading
        models = {
            'gold': {
                'type': 'ensemble',
                'version': '1.0',
                'accuracy': 0.72
            },
            'usdjpy': {
                'type': 'ensemble',
                'version': '1.0',
                'accuracy': 0.68
            }
        }
        
        return models
    except Exception as e:
        st.warning(f"Could not load AI models: {str(e)}. Using technical analysis only.")
        return None

# ---------------- AI BIAS PREDICTION ----------------
def predict_bias_with_ai(data, ticker, models):
    """Use AI models to predict market bias"""
    df = calculate_m15_indicators(data)
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Feature engineering for AI model
    features = {
        'rsi_14': latest['RSI_14'],
        'rsi_7': latest['RSI_7'],
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_signal'],
        'bb_position': latest['BB_position'],
        'bb_width': latest['BB_width'],
        'stoch_k': latest['Stoch_K'],
        'stoch_d': latest['Stoch_D'],
        'atr_ratio': latest['ATR_14'] / latest['Close'],
        'volume_ratio': latest.get('Volume_ratio', 1),
        'price_vs_sma20': (latest['Close'] - latest['SMA_20']) / latest['SMA_20'],
        'price_vs_sma50': (latest['Close'] - latest['SMA_50']) / latest['SMA_50'],
        'price_vs_ema9': (latest['Close'] - latest['EMA_9']) / latest['EMA_9'],
        'higher_high': int(latest['Higher_High']),
        'lower_low': int(latest['Lower_Low']),
        'doji': int(latest['Doji']),
        'hammer': int(latest['Hammer']),
        'shooting_star': int(latest['Shooting_Star'])
    }
    
    # Calculate weighted score (simulating AI ensemble)
    weights = {
        'rsi_14': 0.8,
        'macd': 1.2,
        'bb_position': 1.0,
        'stoch_k': 0.7,
        'price_vs_sma20': 1.3,
        'price_vs_ema9': 1.1,
        'higher_high': 1.5,
        'lower_low': 1.5
    }
    
    score = 0
    total_weight = 0
    
    # RSI contribution
    if not pd.isna(features['rsi_14']):
        if features['rsi_14'] < 30:
            score += 1 * weights['rsi_14']
        elif features['rsi_14'] > 70:
            score += -1 * weights['rsi_14']
        total_weight += weights['rsi_14']
    
    # MACD contribution
    if not pd.isna(features['macd']) and not pd.isna(features['macd_signal']):
        if features['macd'] > features['macd_signal']:
            score += 1 * weights['macd']
        else:
            score += -1 * weights['macd']
        total_weight += weights['macd']
    
    # Bollinger Position
    if not pd.isna(features['bb_position']):
        if features['bb_position'] < 0.2:
            score += 1 * weights['bb_position']
        elif features['bb_position'] > 0.8:
            score += -1 * weights['bb_position']
        total_weight += weights['bb_position']
    
    # Price vs SMA
    if not pd.isna(features['price_vs_sma20']):
        if features['price_vs_sma20'] > 0.01:  # 1% above SMA
            score += 1 * weights['price_vs_sma20']
        elif features['price_vs_sma20'] < -0.01:  # 1% below SMA
            score += -1 * weights['price_vs_sma20']
        total_weight += weights['price_vs_sma20']
    
    # Price action signals
    if features['higher_high']:
        score += 1 * weights['higher_high']
        total_weight += weights['higher_high']
    if features['lower_low']:
        score += -1 * weights['lower_low']
        total_weight += weights['lower_low']
    
    # Calculate normalized score (-1 to 1)
    if total_weight > 0:
        normalized_score = score / total_weight
    else:
        normalized_score = 0
    
    # Determine bias
    if normalized_score > 0.3:
        bias = "BULLISH"
        confidence = min(0.5 + abs(normalized_score) * 0.5, 0.95)
        strength = "STRONG" if normalized_score > 0.6 else "MODERATE" if normalized_score > 0.4 else "WEAK"
    elif normalized_score < -0.3:
        bias = "BEARISH"
        confidence = min(0.5 + abs(normalized_score) * 0.5, 0.95)
        strength = "STRONG" if normalized_score < -0.6 else "MODERATE" if normalized_score < -0.4 else "WEAK"
    else:
        bias = "SIDEWAYS"
        confidence = 0.5 + abs(normalized_score) * 0.3
        strength = "CONSOLIDATING"
    
    # Model confidence based on feature availability
    feature_ratio = total_weight / sum(weights.values())
    confidence = confidence * (0.7 + 0.3 * feature_ratio)
    
    return {
        'bias': bias,
        'strength': strength,
        'confidence': confidence,
        'score': normalized_score,
        'features': features,
        'timestamp': datetime.now()
    }

# ---------------- KEY LEVELS DETECTION ----------------
def detect_key_levels(data, ticker):
    """Detect key support and resistance levels for M15"""
    df = calculate_m15_indicators(data)
    latest = df.iloc[-1]
    
    # Recent highs and lows (last 100 candles)
    recent_highs = df['High'].rolling(window=20, center=True).max()
    recent_lows = df['Low'].rolling(window=20, center=True).min()
    
    # Find pivot highs and lows
    pivot_highs = []
    pivot_lows = []
    
    for i in range(20, len(df)-20):
        if df['High'].iloc[i] == recent_highs.iloc[i]:
            pivot_highs.append({
                'price': df['High'].iloc[i],
                'index': i,
                'strength': 'MAJOR' if df['Volume'].iloc[i] > df['Volume_SMA'].iloc[i] * 1.2 else 'MINOR'
            })
        if df['Low'].iloc[i] == recent_lows.iloc[i]:
            pivot_lows.append({
                'price': df['Low'].iloc[i],
                'index': i,
                'strength': 'MAJOR' if df['Volume'].iloc[i] > df['Volume_SMA'].iloc[i] * 1.2 else 'MINOR'
            })
    
    # Get last 5 significant levels
    pivot_highs = sorted(pivot_highs, key=lambda x: x['price'])[-5:]
    pivot_lows = sorted(pivot_lows, key=lambda x: x['price'])[:5]
    
    # Calculate distance from current price
    current_price = latest['Close']
    
    resistance_levels = []
    for level in pivot_highs:
        if level['price'] > current_price:
            distance = ((level['price'] - current_price) / current_price) * 100
            resistance_levels.append({
                'price': level['price'],
                'distance': distance,
                'strength': level['strength']
            })
    
    support_levels = []
    for level in pivot_lows:
        if level['price'] < current_price:
            distance = ((current_price - level['price']) / current_price) * 100
            support_levels.append({
                'price': level['price'],
                'distance': distance,
                'strength': level['strength']
            })
    
    # Sort by distance
    resistance_levels = sorted(resistance_levels, key=lambda x: x['distance'])[:3]
    support_levels = sorted(support_levels, key=lambda x: x['distance'])[:3]
    
    # Add Fibonacci levels
    if len(support_levels) > 0 and len(resistance_levels) > 0:
        high = resistance_levels[0]['price']
        low = support_levels[0]['price']
        range_price = high - low
        
        fib_levels = {
            '0.236': low + 0.236 * range_price,
            '0.382': low + 0.382 * range_price,
            '0.5': low + 0.5 * range_price,
            '0.618': low + 0.618 * range_price,
            '0.786': low + 0.786 * range_price
        }
    else:
        fib_levels = {}
    
    return {
        'support': support_levels,
        'resistance': resistance_levels,
        'fibonacci': fib_levels,
        'current_price': current_price,
        'atr': latest['ATR_14']
    }

# ---------------- RISK MANAGEMENT CALCULATION ----------------
def calculate_risk_management(levels, bias, account_balance=10000, risk_percent=0.02):
    """Calculate entry, stop loss, and take profit levels with proper risk management"""
    
    current_price = levels['current_price']
    atr = levels['atr']
    
    risk_amount = account_balance * risk_percent
    
    if bias == "BULLISH":
        # Long position
        if len(levels['support']) > 0:
            # Use nearest support as stop loss
            stop_loss = levels['support'][0]['price']
            
            # Calculate position size based on risk
            stop_distance = current_price - stop_loss
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
                
                # Take profit levels (2:1 and 3:1 risk-reward)
                tp1 = current_price + (stop_distance * 2)
                tp2 = current_price + (stop_distance * 3)
                
                # Alternative: Use resistance levels for take profit
                if len(levels['resistance']) > 0:
                    tp_alt = levels['resistance'][0]['price']
                else:
                    tp_alt = None
            else:
                position_size = 0
                tp1 = tp2 = None
        else:
            # Use ATR for stop loss if no support level
            stop_loss = current_price - (atr * 1.5)
            stop_distance = atr * 1.5
            position_size = risk_amount / stop_distance
            tp1 = current_price + (atr * 3)
            tp2 = current_price + (atr * 5)
            tp_alt = None
    
    elif bias == "BEARISH":
        # Short position
        if len(levels['resistance']) > 0:
            # Use nearest resistance as stop loss
            stop_loss = levels['resistance'][0]['price']
            
            # Calculate position size based on risk
            stop_distance = stop_loss - current_price
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
                
                # Take profit levels (2:1 and 3:1 risk-reward)
                tp1 = current_price - (stop_distance * 2)
                tp2 = current_price - (stop_distance * 3)
                
                # Alternative: Use support levels for take profit
                if len(levels['support']) > 0:
                    tp_alt = levels['support'][0]['price']
                else:
                    tp_alt = None
            else:
                position_size = 0
                tp1 = tp2 = None
        else:
            # Use ATR for stop loss if no resistance level
            stop_loss = current_price + (atr * 1.5)
            stop_distance = atr * 1.5
            position_size = risk_amount / stop_distance
            tp1 = current_price - (atr * 3)
            tp2 = current_price - (atr * 5)
            tp_alt = None
    else:
        # Sideways - no trade recommendation
        return {
            'action': 'NO TRADE',
            'reason': 'Market is sideways',
            'levels': levels
        }
    
    # Calculate risk-reward ratios
    if bias == "BULLISH" and tp1 and stop_loss:
        rr1 = (tp1 - current_price) / (current_price - stop_loss)
        rr2 = (tp2 - current_price) / (current_price - stop_loss) if tp2 else None
    elif bias == "BEARISH" and tp1 and stop_loss:
        rr1 = (current_price - tp1) / (stop_loss - current_price)
        rr2 = (current_price - tp2) / (stop_loss - current_price) if tp2 else None
    else:
        rr1 = rr2 = None
    
    # Determine risk level
    if stop_distance / current_price < 0.002:  # Less than 0.2%
        risk_level = "HIGH - Tight stop"
    elif stop_distance / current_price < 0.005:  # Less than 0.5%
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW - Wide stop"
    
    return {
        'action': f'{"BUY" if bias == "BULLISH" else "SELL" if bias == "BEARISH" else "HOLD"}',
        'bias': bias,
        'current_price': current_price,
        'stop_loss': stop_loss,
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'take_profit_alt': tp_alt,
        'position_size': position_size,
        'risk_amount': risk_amount,
        'risk_reward_1': rr1,
        'risk_reward_2': rr2,
        'risk_level': risk_level,
        'stop_distance': stop_distance / current_price * 100,  # as percentage
        'levels': levels
    }

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("<h1 style='text-align: center;'>🏆 💱</h1>", unsafe_allow_html=True)
with col2:
    st.title("🤖 M15 GOLD & USD/JPY AI TRADING SYSTEM")
    st.markdown("#### AI-Powered Bias Detection | Key Levels | Risk Management")
with col3:
    st.markdown(f"<div class='last-updated'>Last Updated:<br>{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} ET</div>", unsafe_allow_html=True)
    
    if st.button("🔄 REFRESH M15 DATA", key="refresh_btn", use_container_width=True):
        refresh_data()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Trading System Controls")
    
    # Auto-refresh
    st.subheader("🔄 Auto-Refresh (M15)")
    auto_refresh = st.checkbox("Sync with M15 candles", value=True)
    if auto_refresh:
        refresh_interval = st.selectbox("Refresh interval", 
                                       ["15 minutes", "5 minutes", "1 minute"],
                                       index=0)
        interval_map = {"15 minutes": 900, "5 minutes": 300, "1 minute": 60}
        
        if st.session_state.last_refresh < datetime.now(pytz.timezone('US/Eastern')) - timedelta(seconds=interval_map[refresh_interval]):
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
    confidence_threshold = st.slider("Minimum Confidence", 
                                     min_value=0.5, 
                                     max_value=0.9, 
                                     value=0.65,
                                     step=0.05)
    
    use_ensemble = st.checkbox("Use Ensemble Models", value=True)
    
    st.markdown("---")
    
    # Market Hours
    st.subheader("⏰ Market Hours")
    now = datetime.now(pytz.timezone('US/Eastern'))
    market_open = now.replace(hour=9, minute=30) <= now <= now.replace(hour=16, minute=0)
    forex_24h = True
    
    st.markdown(f"**Gold (COMEX):** {'🟢 Open' if market_open else '🔴 Closed'}")
    st.markdown(f"**USD/JPY (Forex):** {'🟢 24/5' if forex_24h else '🔴 Weekend'}")

# ---------------- LOAD MODELS ----------------
models = load_ai_models()

# ---------------- MAIN DASHBOARD ----------------
st.markdown("## 📊 M15 AI Analysis & Trading Signals")

# Create two columns
col1, col2 = st.columns(2)

# Store predictions
predictions = {}

# Process Gold
with col1:
    st.markdown(f"### 🏆 GOLD (GC=F) - M15")
    
    with st.spinner("Analyzing Gold M15 data..."):
        data = load_m15_data(GOLD_TICKER)
        
        if data is not None and not data.empty:
            # Get AI prediction
            prediction = predict_bias_with_ai(data, GOLD_TICKER, models)
            predictions['GOLD'] = prediction
            
            # Detect key levels
            levels = detect_key_levels(data, GOLD_TICKER)
            
            # Calculate risk management
            risk_mgmt = calculate_risk_management(levels, prediction['bias'], account_balance, risk_percent)
            
            # Display bias with appropriate styling
            bias_class = prediction['bias'].lower()
            if prediction['bias'] == "BULLISH":
                bias_emoji = "🚀"
                bg_color = "#00ff9f20"
            elif prediction['bias'] == "BEARISH":
                bias_emoji = "📉"
                bg_color = "#ff4d4d20"
            else:
                bias_emoji = "⏸️"
                bg_color = "#ffd70020"
            
            st.markdown(f"""
            <div class="neon-card gold-card">
                <h2 class="gold-text">🏆 GOLD M15</h2>
                <div class="{bias_class}">{bias_emoji} {prediction['bias']} {bias_emoji}</div>
                <h3>${levels['current_price']:.2f}</h3>
                <p style="color: #888">Strength: {prediction['strength']} | Confidence: {prediction['confidence']:.1%}</p>
                <p style="color: #888">ATR (14): ${levels['atr']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Levels
            st.subheader("🎯 Key Support & Resistance")
            
            col_level1, col_level2 = st.columns(2)
            
            with col_level1:
                st.markdown("##### 🟢 Support Levels")
                for level in levels['support']:
                    distance_pips = (levels['current_price'] - level['price']) / ASSET_INFO[GOLD_TICKER]['pip_value']
                    st.markdown(f"""
                    <div class="level-box support">
                        <strong>${level['price']:.2f}</strong><br>
                        <small>Distance: {level['distance']:.1f}% ({distance_pips:.0f} pips)</small><br>
                        <span class="risk-badge risk-{'low' if level['strength'] == 'MAJOR' else 'medium'}">{level['strength']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_level2:
                st.markdown("##### 🔴 Resistance Levels")
                for level in levels['resistance']:
                    distance_pips = (level['price'] - levels['current_price']) / ASSET_INFO[GOLD_TICKER]['pip_value']
                    st.markdown(f"""
                    <div class="level-box resistance">
                        <strong>${level['price']:.2f}</strong><br>
                        <small>Distance: {level['distance']:.1f}% ({distance_pips:.0f} pips)</small><br>
                        <span class="risk-badge risk-{'low' if level['strength'] == 'MAJOR' else 'medium'}">{level['strength']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Trading Signal with Risk Management
            st.subheader("💼 Trading Signal")
            
            if risk_mgmt['action'] != 'NO TRADE' and prediction['confidence'] >= confidence_threshold:
                # Action box
                action_color = "#00ff9f" if risk_mgmt['action'] == "BUY" else "#ff4d4d"
                st.markdown(f"""
                <div style="background: {bg_color}; padding: 20px; border-radius: 10px; border-left: 4px solid {action_color};">
                    <h3 style="color: {action_color}; margin:0">{risk_mgmt['action']} SIGNAL</h3>
                    <p style="color: #888">Based on AI {prediction['strength']} {prediction['bias']} bias</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Entry and Levels
                col_entry1, col_entry2, col_entry3 = st.columns(3)
                with col_entry1:
                    st.metric("Entry", f"${risk_mgmt['current_price']:.2f}")
                with col_entry2:
                    st.metric("Stop Loss", f"${risk_mgmt['stop_loss']:.2f}", 
                             delta=f"{-risk_mgmt['stop_distance']:.2f}%" if risk_mgmt['action'] == "BUY" else f"+{risk_mgmt['stop_distance']:.2f}%")
                with col_entry3:
                    st.metric("Take Profit 1", f"${risk_mgmt['take_profit_1']:.2f}" if risk_mgmt['take_profit_1'] else "N/A")
                
                # Position Sizing
                st.markdown("##### 📊 Position Sizing")
                col_pos1, col_pos2, col_pos3 = st.columns(3)
                with col_pos1:
                    st.metric("Position Size", f"{risk_mgmt['position_size']:.4f} units")
                with col_pos2:
                    st.metric("Risk Amount", f"${risk_mgmt['risk_amount']:.2f}")
                with col_pos3:
                    st.metric("Risk-Reward 1", f"1:{risk_mgmt['risk_reward_1']:.2f}" if risk_mgmt['risk_reward_1'] else "N/A")
                
                # Risk Warning
                risk_class = risk_mgmt['risk_level'].split()[0].lower()
                st.markdown(f"""
                <div style="margin-top: 10px;">
                    <span class="risk-badge risk-{risk_class}">{risk_mgmt['risk_level']}</span>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                if prediction['bias'] == "SIDEWAYS":
                    st.info("⏸️ Market is sideways. No clear signal. Wait for breakout.")
                elif prediction['confidence'] < confidence_threshold:
                    st.warning(f"⚠️ Confidence ({prediction['confidence']:.1%}) below threshold ({confidence_threshold:.1%})")
            
            # M15 Chart
            st.subheader("📈 M15 Price Action")
            
            # Create a simple price chart with levels
            chart_data = data[['Close']].copy()
            chart_data.columns = ['Price']
            
            # Add levels to chart data
            for i, level in enumerate(levels['support'][:2]):
                chart_data[f'Sup{i+1}'] = level['price']
            for i, level in enumerate(levels['resistance'][:2]):
                chart_data[f'Res{i+1}'] = level['price']
            
            st.line_chart(chart_data)
            
        else:
            st.error
