
import os, time, hmac, hashlib, requests
from urllib.parse import urlencode
from datetime import datetime
import threading
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")
TOKEN      = os.getenv("TELEGRAM_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")
if not API_SECRET:
    raise RuntimeError("BINANCE_SECRET is missing')")
SYMBOL = "DOGEUSDT"
INTERVAL = "1m"
QUANTITY = 15
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.05
RSI_THRESHOLD = 30
AI_THRESHOLD = 0.65
TEST_MODE = False

BASE = "https://api.binance.com"

def create_xgboost_model():
    """Create an advanced XGBoost model for crypto prediction"""
    np.random.seed(42)
    n_samples = 5000
    
    # Enhanced features for crypto trading
    # price_change, rsi, sma_fast, sma_slow, volume_ratio, macd, bb_position, momentum
    X = np.random.rand(n_samples, 8)
    X[:, 0] = (np.random.rand(n_samples) - 0.5) * 0.1  # price_change (-5% to +5%)
    X[:, 1] *= 100  # rsi (0-100)
    X[:, 2] *= 1  # sma_fast ratio
    X[:, 3] *= 1  # sma_slow ratio
    X[:, 4] = np.random.exponential(1, n_samples)  # volume_ratio
    X[:, 5] = (np.random.rand(n_samples) - 0.5) * 0.02  # macd
    X[:, 6] *= 1  # bollinger_band_position (0-1)
    X[:, 7] = (np.random.rand(n_samples) - 0.5) * 0.05  # momentum
    
    # Advanced rule-based labels for XGBoost training
    y = []
    for i in range(n_samples):
        price_chg, rsi, sma_f, sma_s, vol, macd, bb_pos, momentum = X[i]
        
        score = 0
        # Multiple signal confirmation
        if rsi < 30 and vol > 1.5: score += 2  # Oversold + high volume
        if sma_f > sma_s and macd > 0: score += 2  # Bullish trend
        if bb_pos < 0.2 and momentum > 0: score += 1  # Near lower BB + positive momentum
        if price_chg < -0.02 and vol > 2: score += 1  # Price dip + very high volume
        if rsi < 25: score += 1  # Very oversold
        
        # Counter signals
        if rsi > 70: score -= 2  # Overbought
        if vol < 0.5: score -= 1  # Low volume
        if macd < -0.01: score -= 1  # Bearish MACD
        
        y.append(1 if score >= 3 else 0)
    
    # Train XGBoost model with optimal parameters for crypto trading
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    model.fit(X, y)
    
    # Save model
    with open('xgboost_crypto_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Load or create XGBoost model
try:
    with open('xgboost_crypto_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ XGBoost model loaded successfully")
except:
    print("🔧 Creating new XGBoost model...")
    model = create_xgboost_model()
    print("✅ XGBoost model created and saved")

def tg(msg):
    if TOKEN and CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          json={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        except:
            pass

def signed(method, path, params=None):
    if params is None: params = {}
    params["timestamp"] = int(time.time()*1000)
    qs = urlencode(params, True)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE}{path}?{qs}&signature={sig}"
    headers={"X-MBX-APIKEY": API_KEY}
    return requests.request(method, url, headers=headers, timeout=15).json()

def get_klines(limit=100):
    r = requests.get(f"{BASE}/api/v3/klines", params={
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit
    }, timeout=10).json()
    return [(float(x[4]), float(x[5])) for x in r]  # (close, volume)

def sma(vals, n): 
    return sum(vals[-n:])/n if len(vals) >= n else vals[-1]

def ema(vals, n):
    if len(vals) < n:
        return vals[-1]
    alpha = 2 / (n + 1)
    ema_val = vals[0]
    for val in vals[1:]:
        ema_val = alpha * val + (1 - alpha) * ema_val
    return ema_val

def rsi(vals, n=14):
    if len(vals) < n+1:
        return 50
    deltas = [vals[i] - vals[i-1] for i in range(1, min(n+1, len(vals)))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains)/len(deltas) if gains else 0
    avg_loss = sum(losses)/len(deltas) if losses else 1e-9
    rs = avg_gain/avg_loss
    return 100 - 100/(1 + rs)

def macd(vals, fast=12, slow=26):
    if len(vals) < slow:
        return 0
    ema_fast = ema(vals, fast)
    ema_slow = ema(vals, slow)
    return ema_fast - ema_slow

def bollinger_position(vals, price, n=20):
    if len(vals) < n:
        return 0.5
    sma_val = sma(vals, n)
    std_dev = (sum([(x - sma_val)**2 for x in vals[-n:]]) / n) ** 0.5
    upper = sma_val + 2 * std_dev
    lower = sma_val - 2 * std_dev
    if upper == lower:
        return 0.5
    return (price - lower) / (upper - lower)

def momentum(vals, n=10):
    if len(vals) < n:
        return 0
    return (vals[-1] - vals[-n]) / vals[-n]

def get_xgboost_prediction(closes, volumes, price):
    """Get XGBoost AI prediction with advanced features"""
    try:
        if len(closes) < 30:
            return 0.5
            
        # Calculate advanced features
        price_change = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
        rsi_val = rsi(closes) / 100  # normalize to 0-1
        sma_fast = sma(closes, 5) / price
        sma_slow = sma(closes, 20) / price
        volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10) if len(volumes) >= 10 else 1.0
        macd_val = macd(closes) / price
        bb_pos = bollinger_position(closes, price)
        momentum_val = momentum(closes)
        
        # Create feature array
        features = np.array([[
            price_change,
            rsi_val * 100,  # back to 0-100 scale for model
            sma_fast,
            sma_slow,
            volume_ratio,
            macd_val,
            bb_pos,
            momentum_val
        ]])
        
        # Get XGBoost prediction
        prob = model.predict_proba(features)[0, 1]
        return prob
        
    except Exception as e:
        print(f"XGBoost prediction error: {e}")
        return 0.5

def place(side, qty):
    tg(f"{'TEST' if TEST_MODE else 'LIVE'} {side} {qty}")
    if TEST_MODE: 
        return
    try:
        result = signed("POST", "/api/v3/order", {
            "symbol": SYMBOL,
            "side": side,
            "type": "MARKET",
            "quantity": qty
        })
        print(f"Order result: {result}")
    except Exception as e:
        print(f"Order error: {e}")
        tg(f"Order error: {e}")

# Keep alive function
def keep_alive():
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "🤖 XGBoost AI Crypto Trading Bot is alive!"
    
    def run():
        app.run(host='0.0.0.0', port=8080)
    
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

# Start keep alive
keep_alive()

print("🤖 XGBoost AI Trading Bot started", "(TEST)" if TEST_MODE else "(LIVE)")
tg("🤖 XGBoost AI Trading Bot started (TEST=%s)" % TEST_MODE)

in_pos = False
entry = 0
peak = 0

while True:
    try:
        klines_data = get_klines()
        closes = [x[0] for x in klines_data]
        volumes = [x[1] for x in klines_data]
        
        price = closes[-1]
        
        # Get XGBoost AI prediction
        ai_prob = get_xgboost_prediction(closes, volumes, price)
        
        # Technical indicators for confirmation
        sma_fast = sma(closes, 5)
        sma_slow = sma(closes, 20)
        rsi_val = rsi(closes)
        volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10) if len(volumes) >= 10 else 1.0
        
        print(f"P={price:.4f} XGB={ai_prob:.3f} RSI={rsi_val:.1f} SMA_ratio={sma_fast/sma_slow:.3f} Vol={volume_ratio:.2f}")
        
        # Enhanced buy condition with XGBoost
        if (not in_pos and 
            ai_prob >= AI_THRESHOLD and 
            sma_fast > sma_slow and 
            rsi_val < RSI_THRESHOLD and
            volume_ratio > 1.2):
            
            place("BUY", QUANTITY)
            in_pos = True
            entry = price
            peak = price
            tg(f"🟢 XGBoost BUY {price:.4f} (prob={ai_prob:.3f}, RSI={rsi_val:.1f}, Vol={volume_ratio:.2f})")
        
        # Position management with XGBoost insights
        if in_pos:
            peak = max(peak, price)
            
            # Stop loss
            if price <= peak * (1 - STOP_LOSS_PCT):
                place("SELL", QUANTITY)
                in_pos = False
                profit_pct = ((price - entry) / entry) * 100
                tg(f"🔴 STOP LOSS at {price:.4f} (P&L: {profit_pct:.2f}%)")
                
            # Take profit
            elif price >= entry * (1 + TAKE_PROFIT_PCT):
                place("SELL", QUANTITY)
                in_pos = False
                profit_pct = ((price - entry) / entry) * 100
                tg(f"🟢 TAKE PROFIT at {price:.4f} (P&L: {profit_pct:.2f}%)")
                
            # XGBoost suggests sell or technical reversal
            elif ai_prob < 0.35 or (sma_fast < sma_slow and rsi_val > 75):
                place("SELL", QUANTITY)
                in_pos = False
                profit_pct = ((price - entry) / entry) * 100
                tg(f"🔄 XGBoost SELL at {price:.4f} (P&L: {profit_pct:.2f}%, prob={ai_prob:.3f})")
                
    except Exception as e:
        print("ERROR:", e)
        tg(f"⚠️ Error: {e}")
    
    time.sleep(60)
import os

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
