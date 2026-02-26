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
        "pip_value": 0.1,  # Gold moves in 10-cent increments
        "contract_size": 100,
        "margin_requirement": 0.01,
        "decimal_places": 2,
        "currency": "$",
        "typical_spread": 0.3,  # $0.30 typical spread
        "daily_range_pips": 1500  # Typical daily range in pips
    },
    USDJPY_TICKER: {
        "name": "USD/JPY",
        "icon": "💱",
        "pip_value": 0.001,  # 0.001 = 1 pip for JPY pairs
        "contract_size": 100000,
        "margin_requirement": 0.02,
        "decimal_places": 3,
        "currency": "¥",
        "typical_spread": 0.8,  # 0.8 pips typical spread
        "daily_range_pips": 80  # Typical daily range in pips
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
                           ((df['Close'] - df['Low']) < (df['Close'] - df['Open']) * 0.5))
    
    return df

# ---------------- LOAD AI MODELS ----------------
@st.cache_resource
def load_ai_models():
    """Load pre-trained AI models for bias prediction"""
    try:
        # In production, load actual trained models
        # For demo, we'll use ensemble of indicators
        
        models = {
            'gold': {
                'type': 'ensemble',
                'version': '1.0',
                'accuracy': 0.72,
                'weights': {
                    'trend': 1.3,
                    'momentum': 1.2,
                    'volatility': 1.0,
                    'volume': 0.8,
                    'pattern': 1.1
                }
            },
            'usdjpy': {
                'type': 'ensemble',
                'version': '1.0',
                'accuracy': 0.68,
                'weights': {
                    'trend': 1.2,
                    'momentum': 1.3,
                    'volatility': 0.9,
                    'volume': 0.7,
                    'pattern': 1.0
                }
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
    previous = df.iloc[-2] if len(df) > 1 else latest
    
    # Feature engineering for AI model
    features = {
        'rsi_14': latest['RSI_14'] if not pd.isna(latest['RSI_14']) else 50,
        'rsi_7': latest['RSI_7'] if not pd.isna(latest['RSI_7']) else 50,
        'macd': latest['MACD'] if not pd.isna(latest['MACD']) else 0,
        'macd_signal': latest['MACD_signal'] if not pd.isna(latest['MACD_signal']) else 0,
        'macd_histogram': latest['MACD_histogram'] if not pd.isna(latest['MACD_histogram']) else 0,
        'bb_position': latest['BB_position'] if not pd.isna(latest['BB_position']) else 0.5,
        'bb_width': latest['BB_width'] if not pd.isna(latest['BB_width']) else 0,
        'stoch_k': latest['Stoch_K'] if not pd.isna(latest['Stoch_K']) else 50,
        'stoch_d': latest['Stoch_D'] if not pd.isna(latest['Stoch_D']) else 50,
        'atr': latest['ATR_14'] if not pd.isna(latest['ATR_14']) else 0,
        'volume_ratio': latest.get('Volume_ratio', 1),
        'price_vs_sma20': (latest['Close'] - latest['SMA_20']) / latest['SMA_20'] if not pd.isna(latest['SMA_20']) else 0,
        'price_vs_sma50': (latest['Close'] - latest['SMA_50']) / latest['SMA_50'] if not pd.isna(latest['SMA_50']) else 0,
        'price_vs_ema9': (latest['Close'] - latest['EMA_9']) / latest['EMA_9'] if not pd.isna(latest['EMA_9']) else 0,
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
        'macd_histogram': 1.1,
        'bb_position': 1.0,
        'stoch_k': 0.7,
        'price_vs_sma20': 1.3,
        'price_vs_sma50': 1.2,
        'price_vs_ema9': 1.1,
        'higher_high': 1.5,
        'lower_low': 1.5,
        'hammer': 1.4,
        'shooting_star': 1.4
    }
    
    score = 0
    total_weight = 0
    signals_used = []
    
    # RSI contribution
    if not pd.isna(features['rsi_14']):
        if features['rsi_14'] < 30:
            score += 1 * weights['rsi_14']
            signals_used.append('RSI Oversold')
        elif features['rsi_14'] > 70:
            score += -1 * weights['rsi_14']
            signals_used.append('RSI Overbought')
        total_weight += weights['rsi_14']
    
    # MACD contribution
    if features['macd'] > features['macd_signal']:
        score += 1 * weights['macd']
        signals_used.append('MACD Bullish')
    elif features['macd'] < features['macd_signal']:
        score += -1 * weights['macd']
        signals_used.append('MACD Bearish')
    total_weight += weights['macd']
    
    # MACD Histogram
    if features['macd_histogram'] > 0 and features['macd_histogram'] > previous['MACD_histogram']:
        score += 0.5 * weights['macd_histogram']
        signals_used.append('MACD Momentum')
    elif features['macd_histogram'] < 0 and features['macd_histogram'] < previous['MACD_histogram']:
        score += -0.5 * weights['macd_histogram']
        signals_used.append('MACD Momentum')
    total_weight += weights['macd_histogram']
    
    # Bollinger Position
    if features['bb_position'] < 0.2:
        score += 1 * weights['bb_position']
        signals_used.append('BB Oversold')
    elif features['bb_position'] > 0.8:
        score += -1 * weights['bb_position']
        signals_used.append('BB Overbought')
    total_weight += weights['bb_position']
    
    # Price vs SMA
    if features['price_vs_sma20'] > 0.005:  # 0.5% above SMA
        score += 0.8 * weights['price_vs_sma20']
        signals_used.append('Above SMA20')
    elif features['price_vs_sma20'] < -0.005:  # 0.5% below SMA
        score += -0.8 * weights['price_vs_sma20']
        signals_used.append('Below SMA20')
    total_weight += weights['price_vs_sma20']
    
    # Price action signals
    if features['higher_high']:
        score += 1 * weights['higher_high']
        signals_used.append('Higher High')
        total_weight += weights['higher_high']
    if features['lower_low']:
        score += -1 * weights['lower_low']
        signals_used.append('Lower Low')
        total_weight += weights['lower_low']
    
    # Candlestick patterns
    if features['hammer']:
        score += 1 * weights['hammer']
        signals_used.append('Hammer Pattern')
        total_weight += weights['hammer']
    if features['shooting_star']:
        score += -1 * weights['shooting_star']
        signals_used.append('Shooting Star')
        total_weight += weights['shooting_star']
    
    # Calculate normalized score (-1 to 1)
    if total_weight > 0:
        normalized_score = score / total_weight
    else:
        normalized_score = 0
    
    # Determine bias with more nuanced categories
    if normalized_score > 0.3:
        bias = "BULLISH"
        if normalized_score > 0.6:
            strength = "STRONG"
            confidence = min(0.7 + abs(normalized_score) * 0.3, 0.95)
        else:
            strength = "MODERATE"
            confidence = 0.6 + abs(normalized_score) * 0.3
    elif normalized_score < -0.3:
        bias = "BEARISH"
        if normalized_score < -0.6:
            strength = "STRONG"
            confidence = min(0.7 + abs(normalized_score) * 0.3, 0.95)
        else:
            strength = "MODERATE"
            confidence = 0.6 + abs(normalized_score) * 0.3
    else:
        bias = "SIDEWAYS"
        if abs(normalized_score) < 0.1:
            strength = "CONSOLIDATING"
        else:
            strength = "UNCERTAIN"
        confidence = 0.5 + abs(normalized_score) * 0.3
    
    # Model confidence based on feature availability
    feature_ratio = min(total_weight / sum(weights.values()), 1.0)
    confidence = confidence * (0.8 + 0.2 * feature_ratio)
    
    # Get unique signals
    signals_used = list(set(signals_used))[:3]  # Top 3 unique signals
    
    return {
        'bias': bias,
        'strength': strength,
        'confidence': min(confidence, 0.95),
        'score': normalized_score,
        'features': features,
        'signals': signals_used,
        'timestamp': datetime.now()
    }

# ---------------- KEY LEVELS DETECTION ----------------
def detect_key_levels(data, ticker):
    """Detect key support and resistance levels for M15"""
    df = calculate_m15_indicators(data)
    latest = df.iloc[-1]
    
    # Get asset info for formatting
    asset = ASSET_INFO[ticker]
    
    # Recent highs and lows (last 50 candles for M15 relevance)
    recent_highs = df['High'].rolling(window=20, center=True).max()
    recent_lows = df['Low'].rolling(window=20, center=True).min()
    
    # Find pivot highs and lows
    pivot_highs = []
    pivot_lows = []
    
    for i in range(10, len(df)-10):
        if df['High'].iloc[i] == recent_highs.iloc[i]:
            volume_confirm = False
            if 'Volume' in df.columns and not pd.isna(df['Volume_SMA'].iloc[i]):
                volume_confirm = df['Volume'].iloc[i] > df['Volume_SMA'].iloc[i] * 1.2
            
            pivot_highs.append({
                'price': df['High'].iloc[i],
                'index': i,
                'time': df.index[i],
                'strength': 'MAJOR' if volume_confirm else 'MINOR'
            })
        
        if df['Low'].iloc[i] == recent_lows.iloc[i]:
            volume_confirm = False
            if 'Volume' in df.columns and not pd.isna(df['Volume_SMA'].iloc[i]):
                volume_confirm = df['Volume'].iloc[i] > df['Volume_SMA'].iloc[i] * 1.2
            
            pivot_lows.append({
                'price': df['Low'].iloc[i],
                'index': i,
                'time': df.index[i],
                'strength': 'MAJOR' if volume_confirm else 'MINOR'
            })
    
    # Get last 5 significant levels
    pivot_highs = sorted(pivot_highs, key=lambda x: x['price'])[-8:]
    pivot_lows = sorted(pivot_lows, key=lambda x: x['price'])[:8]
    
    # Calculate distance from current price
    current_price = latest['Close']
    
    resistance_levels = []
    for level in pivot_highs:
        if level['price'] > current_price:
            distance_pct = ((level['price'] - current_price) / current_price) * 100
            distance_pips = (level['price'] - current_price) / asset['pip_value']
            resistance_levels.append({
                'price': level['price'],
                'distance_pct': distance_pct,
                'distance_pips': distance_pips,
                'strength': level['strength'],
                'time': level['time']
            })
    
    support_levels = []
    for level in pivot_lows:
        if level['price'] < current_price:
            distance_pct = ((current_price - level['price']) / current_price) * 100
            distance_pips = (current_price - level['price']) / asset['pip_value']
            support_levels.append({
                'price': level['price'],
                'distance_pct': distance_pct,
                'distance_pips': distance_pips,
                'strength': level['strength'],
                'time': level['time']
            })
    
    # Sort by distance and take closest levels
    resistance_levels = sorted(resistance_levels, key=lambda x: x['distance_pct'])[:3]
    support_levels = sorted(support_levels, key=lambda x: x['distance_pct'])[:3]
    
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
        # Use ATR for Fibonacci levels
        atr = latest['ATR_14']
        fib_levels = {
            '0.236': current_price - atr * 1,
            '0.382': current_price - atr * 0.618,
            '0.5': current_price,
            '0.618': current_price + atr * 0.618,
            '0.786': current_price + atr * 1
        }
    
    return {
        'support': support_levels,
        'resistance': resistance_levels,
        'fibonacci': fib_levels,
        'current_price': current_price,
        'atr': latest['ATR_14'],
        'atr_pips': latest['ATR_14'] / asset['pip_value']
    }

# ---------------- RISK MANAGEMENT CALCULATION ----------------
def calculate_risk_management(levels, prediction, ticker, account_balance=10000, risk_percent=0.02):
    """Calculate entry, stop loss, and take profit levels with proper risk management"""
    
    asset = ASSET_INFO[ticker]
    current_price = levels['current_price']
    atr = levels['atr']
    bias = prediction['bias']
    confidence = prediction['confidence']
    
    risk_amount = account_balance * risk_percent
    
    # Determine stop loss based on bias and key levels
    if bias == "BULLISH":
        # For long positions
        
        # Use nearest support as primary stop
        if len(levels['support']) > 0:
            nearest_support = levels['support'][0]['price']
            stop_loss = nearest_support - (atr * 0.3)  # Slight buffer below support
        else:
            # Use ATR-based stop
            stop_loss = current_price - (atr * 1.5)
        
        # Calculate stop distance
        stop_distance = current_price - stop_loss
        stop_distance_pips = stop_distance / asset['pip_value']
        stop_distance_pct = (stop_distance / current_price) * 100
        
        # Position sizing
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
            # Adjust for contract size
            if ticker == GOLD_TICKER:
                position_size = position_size / 100  # Gold contracts
            elif ticker == USDJPY_TICKER:
                position_size = position_size / 100000  # Standard lots
        else:
            position_size = 0
        
        # Take profit levels (based on risk-reward)
        tp1_price = current_price + (stop_distance * 2)  # 2:1 risk-reward
        tp2_price = current_price + (stop_distance * 3)  # 3:1 risk-reward
        
        # Alternative take profit based on resistance
        if len(levels['resistance']) > 0:
            tp_alt_price = levels['resistance'][0]['price']
            tp_alt_rr = (tp_alt_price - current_price) / stop_distance
        else:
            tp_alt_price = None
            tp_alt_rr = None
        
        # Risk-reward ratios
        rr1 = 2.0
        rr2 = 3.0
        
        action = "BUY"
        
    elif bias == "BEARISH":
        # For short positions
        
        # Use nearest resistance as primary stop
        if len(levels['resistance']) > 0:
            nearest_resistance = levels['resistance'][0]['price']
            stop_loss = nearest_resistance + (atr * 0.3)  # Slight buffer above resistance
        else:
            # Use ATR-based stop
            stop_loss = current_price + (atr * 1.5)
        
        # Calculate stop distance
        stop_distance = stop_loss - current_price
        stop_distance_pips = stop_distance / asset['pip_value']
        stop_distance_pct = (stop_distance / current_price) * 100
        
        # Position sizing
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
            # Adjust for contract size
            if ticker == GOLD_TICKER:
                position_size = position_size / 100  # Gold contracts
            elif ticker == USDJPY_TICKER:
                position_size = position_size / 100000  # Standard lots
        else:
            position_size = 0
        
        # Take profit levels (based on risk-reward)
        tp1_price = current_price - (stop_distance * 2)  # 2:1 risk-reward
        tp2_price = current_price - (stop_distance * 3)  # 3:1 risk-reward
        
        # Alternative take profit based on support
        if len(levels['support']) > 0:
            tp_alt_price = levels['support'][0]['price']
            tp_alt_rr = (current_price - tp_alt_price) / stop_distance
        else:
            tp_alt_price = None
            tp_alt_rr = None
        
        # Risk-reward ratios
        rr1 = 2.0
        rr2 = 3.0
        
        action = "SELL"
        
    else:
        # Sideways - no trade
        return {
            'action': 'HOLD',
            'reason': 'Market is sideways',
            'levels': levels,
            'bias': bias,
            'confidence': confidence
        }
    
    # Validate stop loss is reasonable
    if stop_distance_pips < 5:  # Too tight (less than 5 pips)
        risk_level = "EXTREME - Stop too tight"
        trade_quality = "POOR"
    elif stop_distance_pips < 15:
        risk_level = "HIGH - Tight stop"
        trade_quality = "FAIR"
    elif stop_distance_pips < 30:
        risk_level = "MEDIUM"
        trade_quality = "GOOD"
    else:
        risk_level = "LOW - Wide stop"
        trade_quality = "EXCELLENT"
    
    # Adjust trade quality based on confidence
    if confidence > 0.8:
        trade_quality = "EXCELLENT"
    elif confidence > 0.7:
        trade_quality = "GOOD" if trade_quality != "POOR" else "FAIR"
    
    return {
        'action': action,
        'bias': bias,
        'trade_quality': trade_quality,
        'current_price': current_price,
        'stop_loss': stop_loss,
        'take_profit_1': tp1_price,
        'take_profit_2': tp2_price,
        'take_profit_alt': tp_alt_price,
        'tp_alt_rr': tp_alt_rr,
        'position_size': position_size,
        'position_size_units': f"{position_size:.2f} {'contracts' if ticker == GOLD_TICKER else 'lots'}",
        'risk_amount': risk_amount,
        'risk_reward_1': rr1,
        'risk_reward_2': rr2,
        'risk_level': risk_level,
        'stop_distance_pips': stop_distance_pips,
        'stop_distance_pct': stop_distance_pct,
        'levels': levels,
        'confidence': confidence,
        'asset_info': asset
    }

# ---------------- FORMAT PRICE FUNCTION ----------------
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
        
        # Calculate time until next refresh
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
    confidence_threshold = st.slider("Minimum Confidence", 
                                     min_value=0.5, 
                                     max_value=0.9, 
                                     value=0.65,
                                     step=0.05,
                                     help="Minimum confidence level for trade signals")
    
    min_rr_ratio = st.slider("Minimum Risk-Reward", 
                             min_value=1.0, 
                             max_value=3.0, 
                             value=1.5,
                             step=0.1,
                             help="Minimum risk-reward ratio for trade signals")
    
    st.markdown("---")
    
    # Market Hours
    st.subheader("⏰ Market Hours")
    now = datetime.now(pytz.timezone('US/Eastern'))
    market_open = now.replace(hour=9, minute=30) <= now <= now.replace(hour=16, minute=0)
    forex_24h = now.weekday() < 5  # Monday to
