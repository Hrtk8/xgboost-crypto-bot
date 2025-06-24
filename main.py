import os, time, hmac, hashlib, requests
from urllib.parse import urlencode
from datetime import datetime
import threading
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import json

API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")
TOKEN      = os.getenv("TELEGRAM_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")
if not API_SECRET:
    raise RuntimeError("BINANCE_SECRET is missing')")

SYMBOL = "DOGEUSDT"
INTERVAL = "1m"

# CONSERVATIVE SETTINGS FOR SMALL BALANCE ($2.8 USDT)
MIN_USDT_ORDER = 2.0  # Minimum order size
MAX_BALANCE_PCT = 0.85  # Use only 85% of balance for safety
RESERVE_BUFFER = 0.3  # Keep $0.30 as buffer for fees

# CONSERVATIVE RISK MANAGEMENT
DYNAMIC_STOP_LOSS = False  # Disable trailing stop for stability
BASE_STOP_LOSS_PCT = 0.05  # 5% stop loss (wider for small accounts)
TAKE_PROFIT_PCT = 0.08  # 8% take profit target

# SINGLE TARGET APPROACH (no partial sells for small balance)
MULTI_LEVEL_PROFITS = False

# CONSERVATIVE AI THRESHOLDS
AI_BUY_THRESHOLD = 0.80   # Very high confidence required
AI_SELL_THRESHOLD = 0.25  # Earlier exit signals
RSI_OVERSOLD = 20         # More extreme oversold
RSI_OVERBOUGHT = 80       # More extreme overbought
VOLUME_SURGE_MULTIPLIER = 1.8  # Moderate volume requirement

# SMALL BALANCE PROTECTION
MIN_PROFIT_TO_SELL = 0.15  # Minimum $0.15 profit before selling
MAX_TRADES_PER_DAY = 5     # Limit trades to reduce fees
COOL_DOWN_PERIOD = 1800    # 30 min cooldown between trades

# DAILY LIMITS FOR SMALL ACCOUNTS
PROFIT_TARGET_DAILY = 0.15  # 15% daily target (realistic for small balance)
MAX_DAILY_LOSS = 0.20       # 20% maximum daily loss

TEST_MODE = True  # Start in test mode for safety

BASE = "https://api.binance.com"

# Global tracking variables
daily_pnl = 0.0
starting_portfolio_value = 0.0
total_trades = 0
winning_trades = 0
last_trade_time = 0
daily_trade_count = 0

def create_conservative_xgboost_model():
    """Create a conservative XGBoost model focused on high-probability trades"""
    np.random.seed(42)
    n_samples = 5000  # Moderate training data
    
    # Conservative feature set - focus on strong signals only
    X = np.random.rand(n_samples, 8)  # Reduced features for simplicity
    X[:, 0] = (np.random.rand(n_samples) - 0.5) * 0.10  # price_change
    X[:, 1] *= 100  # rsi
    X[:, 2] *= 1    # sma_ratio
    X[:, 3] = np.random.exponential(1, n_samples)  # volume_ratio
    X[:, 4] = (np.random.rand(n_samples) - 0.5) * 0.02  # macd
    X[:, 5] *= 1    # bollinger_position
    X[:, 6] = (np.random.rand(n_samples) - 0.5) * 0.05  # momentum
    X[:, 7] = np.random.rand(n_samples) * 0.05   # volatility
    
    # Conservative labeling - only very strong signals
    y = []
    for i in range(n_samples):
        features = X[i]
        price_chg, rsi, sma_ratio, vol, macd, bb_pos, momentum, volatility = features
        
        score = 0
        
        # Very strong buy signals only
        if rsi < 20 and vol > 2.5 and momentum > 0.03: score += 4
        if sma_ratio > 1.02 and macd > 0.015 and vol > 2.0: score += 3
        if bb_pos < 0.1 and price_chg < -0.04 and vol > 3.0: score += 4
        if rsi < 25 and vol > 2.0 and macd > 0.01: score += 2
        
        # Strong negative signals
        if rsi > 80: score -= 4
        if vol < 0.8: score -= 3
        if macd < -0.015: score -= 3
        if momentum < -0.03: score -= 3
        if volatility > 0.04: score -= 2  # Avoid high volatility
        
        # Very high threshold for conservative approach
        y.append(1 if score >= 5 else 0)
    
    # Conservative XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=150,  # Fewer trees for stability
        max_depth=4,       # Shallow trees
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    model.fit(X, y)
    
    with open('conservative_xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Load or create model
try:
    with open('conservative_xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Conservative XGBoost model loaded")
except:
    print("🔧 Creating conservative XGBoost model...")
    model = create_conservative_xgboost_model()
    print("✅ Conservative model created")

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

def get_portfolio_status():
    """Get current portfolio status"""
    try:
        account = signed("GET", "/api/v3/account")
        doge_balance = usdt_balance = 0.0
        
        for b in account["balances"]:
            if b["asset"] == "DOGE":
                doge_balance = float(b["free"]) + float(b["locked"])
            elif b["asset"] == "USDT":
                usdt_balance = float(b["free"]) + float(b["locked"])
        
        price_resp = requests.get(f"{BASE}/api/v3/ticker/price", params={"symbol": SYMBOL})
        current_price = float(price_resp.json()["price"])
        
        portfolio_value = (doge_balance * current_price) + usdt_balance
        
        return {
            'doge_balance': doge_balance,
            'usdt_balance': usdt_balance,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'doge_value': doge_balance * current_price
        }
    except Exception as e:
        print(f"❌ Portfolio error: {e}")
        return None

def can_trade():
    """Check if we can make a trade based on limits and cooldowns"""
    global last_trade_time, daily_trade_count
    
    current_time = time.time()
    
    # Check cooldown period
    if current_time - last_trade_time < COOL_DOWN_PERIOD:
        return False, "Cooldown period active"
    
    # Check daily trade limit
    if daily_trade_count >= MAX_TRADES_PER_DAY:
        return False, "Daily trade limit reached"
    
    # Check daily P&L limits
    if starting_portfolio_value > 0:
        daily_pnl_pct = daily_pnl / starting_portfolio_value
        if daily_pnl_pct >= PROFIT_TARGET_DAILY:
            return False, "Daily profit target reached"
        if daily_pnl_pct <= -MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
    
    return True, "OK"

def get_conservative_prediction(closes, volumes, price):
    """Conservative prediction focusing on strong signals only"""
    try:
        if len(closes) < 30:
            return 0.3  # Conservative default
        
        # Calculate conservative features
        price_change = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
        rsi_val = rsi(closes)
        sma_fast = sma(closes, 5)
        sma_slow = sma(closes, 20)
        sma_ratio = sma_fast / sma_slow if sma_slow > 0 else 1
        volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10) if len(volumes) >= 10 else 1.0
        macd_val = macd(closes) / price if price > 0 else 0
        bb_pos = bollinger_position(closes, price)
        momentum_val = momentum(closes)
        volatility = calculate_volatility(closes)
        
        features = np.array([[
            price_change, rsi_val, sma_ratio, volume_ratio, 
            macd_val, bb_pos, momentum_val, volatility
        ]])
        
        prob = model.predict_proba(features)[0, 1]
        return prob
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.3

def calculate_volatility(closes, period=14):
    """Calculate price volatility"""
    if len(closes) < period:
        return 0.02
    
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, min(period, len(closes)))]
    volatility = np.std(returns)
    return volatility

# Technical analysis functions
def sma(vals, n): 
    return sum(vals[-n:])/n if len(vals) >= n else vals[-1] if vals else 0

def ema(vals, n):
    if len(vals) < n:
        return vals[-1] if vals else 0
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

def get_klines(limit=50):
    try:
        r = requests.get(f"{BASE}/api/v3/klines", params={
            "symbol": SYMBOL, "interval": INTERVAL, "limit": limit
        }, timeout=10).json()
        return [(float(x[4]), float(x[5])) for x in r]
    except Exception as e:
        print(f"❌ Klines error: {e}")
        return []

def place_conservative_buy_order(portfolio):
    """Place conservative buy order for small balance"""
    global total_trades, daily_trade_count, last_trade_time
    
    try:
        # Calculate safe order size
        available_usdt = portfolio['usdt_balance'] - RESERVE_BUFFER
        usdt_to_use = min(available_usdt * MAX_BALANCE_PCT, available_usdt)
        current_price = portfolio['current_price']
        
        if usdt_to_use < MIN_USDT_ORDER:
            return None
        
        quantity = int(usdt_to_use / current_price)  # Use int to avoid decimal issues
        
        if quantity <= 0:
            return None
        
        tg(f"🐌 CONSERVATIVE BUY: {quantity} DOGE @ ${current_price:.4f} (${usdt_to_use:.2f})")
        
        if TEST_MODE:
            print(f"TEST MODE: Would buy {quantity} DOGE at ${current_price:.4f}")
            result = {"quantity": quantity, "price": current_price}
        else:
            result = signed("POST", "/api/v3/order", {
                "symbol": SYMBOL,
                "side": "BUY", 
                "type": "MARKET",
                "quantity": str(quantity)
            })
        
        total_trades += 1
        daily_trade_count += 1
        last_trade_time = time.time()
        return result
        
    except Exception as e:
        tg(f"❌ Buy error: {e}")
        return None

def place_conservative_sell_order(doge_balance, current_price, entry_price):
    """Place conservative sell order"""
    global daily_pnl, total_trades, winning_trades, daily_trade_count, last_trade_time
    
    try:
        quantity = int(doge_balance)  # Use int to avoid decimal issues
        if quantity <= 0:
            return None
        
        profit_usd = (current_price - entry_price) * quantity
        
        if TEST_MODE:
            print(f"TEST MODE: Would sell {quantity} DOGE at ${current_price:.4f} (Profit: ${profit_usd:.3f})")
            result = {"quantity": quantity, "price": current_price}
        else:
            result = signed("POST", "/api/v3/order", {
                "symbol": SYMBOL,
                "side": "SELL",
                "type": "MARKET", 
                "quantity": str(quantity)
            })
        
        daily_pnl += profit_usd
        total_trades += 1
        daily_trade_count += 1
        last_trade_time = time.time()
        
        if profit_usd > 0:
            winning_trades += 1
        
        return result
        
    except Exception as e:
        tg(f"❌ Sell error: {e}")
        return None

# Keep alive function
def keep_alive():
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        portfolio = get_portfolio_status()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        if portfolio:
            status = f"""
            🐌 Conservative DOGE Bot (Small Balance)
            
            📊 Portfolio: ${portfolio['portfolio_value']:.3f}
            💰 Daily P&L: ${daily_pnl:.3f}
            📈 Win Rate: {win_rate:.1f}%
            🔄 Trades Today: {daily_trade_count}/{MAX_TRADES_PER_DAY}
            💎 DOGE: {portfolio['doge_balance']:.2f}
            💵 USDT: ${portfolio['usdt_balance']:.3f}
            🧪 Test Mode: {TEST_MODE}
            """
        else:
            status = "🔄 Loading portfolio data..."
        
        return status
    
    def run():
        app.run(host='0.0.0.0', port=8080)
    
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

# Start system
keep_alive()

# Initialize tracking
initial_portfolio = get_portfolio_status()
if initial_portfolio:
    starting_portfolio_value = initial_portfolio['portfolio_value']

print("🐌 CONSERVATIVE DOGE BOT STARTED FOR SMALL BALANCE")
print(f"💰 Starting with: ${starting_portfolio_value:.3f}")
print(f"🧪 Test Mode: {TEST_MODE}")
tg(f"🐌 Conservative Bot Started (${starting_portfolio_value:.3f})")

# Reset daily counters at start
daily_trade_count = 0
last_trade_time = 0

# Main trading loop
in_position = False
entry_price = 0
position_size = 0

while True:
    try:
        # Check if we can trade
        can_trade_now, reason = can_trade()
        if not can_trade_now:
            print(f"⏸️ Trading paused: {reason}")
            time.sleep(300)  # Wait 5 minutes
            continue
        
        portfolio = get_portfolio_status()
        if not portfolio:
            time.sleep(60)
            continue
        
        current_price = portfolio['current_price']
        doge_balance = portfolio['doge_balance']
        
        # Update position status
        if doge_balance > 0 and not in_position:
            in_position = True
            position_size = doge_balance
        elif doge_balance <= 0:
            in_position = False
        
        # Get market data
        klines_data = get_klines(50)
        if not klines_data:
            time.sleep(60)
            continue
        
        closes = [k[0] for k in klines_data]
        volumes = [k[1] for k in klines_data]
        
        # Get conservative prediction
        ai_prob = get_conservative_prediction(closes, volumes, current_price)
        
        # Calculate indicators
        rsi_val = rsi(closes)
        sma_fast = sma(closes, 5)
        sma_slow = sma(closes, 20)
        volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10) if len(volumes) >= 10 else 1.0
        volatility = calculate_volatility(closes)
        
        print(f"🐌 ${current_price:.4f} | AI: {ai_prob:.3f} | RSI: {rsi_val:.1f} | Vol: {volume_ratio:.2f} | P&L: ${daily_pnl:.3f} | Trades: {daily_trade_count}")
        
        # CONSERVATIVE BUY CONDITIONS - Very strict
        if (not in_position and 
            portfolio['usdt_balance'] >= (MIN_USDT_ORDER + RESERVE_BUFFER) and
            ai_prob >= AI_BUY_THRESHOLD and
            rsi_val <= RSI_OVERSOLD and
            sma_fast > sma_slow and
            volume_ratio >= VOLUME_SURGE_MULTIPLIER and
            volatility < 0.04):  # Avoid high volatility
            
            result = place_conservative_buy_order(portfolio)
            if result:
                in_position = True
                entry_price = current_price
                position_size = result['quantity']
                tg(f"🟢 CONSERVATIVE ENTRY @ ${current_price:.4f} (AI: {ai_prob:.3f}, RSI: {rsi_val:.1f})")
        
        # CONSERVATIVE SELL CONDITIONS
        if in_position and doge_balance > 0:
            profit_pct = (current_price - entry_price) / entry_price
            profit_usd = (current_price - entry_price) * position_size
            
            # Take profit at target
            if profit_pct >= TAKE_PROFIT_PCT and profit_usd >= MIN_PROFIT_TO_SELL:
                result = place_conservative_sell_order(doge_balance, current_price, entry_price)
                if result:
                    in_position = False
                    tg(f"🟢 PROFIT TARGET @ ${current_price:.4f} (+${profit_usd:.3f}, +{profit_pct*100:.1f}%)")
            
            # Stop loss
            elif profit_pct <= -BASE_STOP_LOSS_PCT:
                result = place_conservative_sell_order(doge_balance, current_price, entry_price)
                if result:
                    in_position = False
                    tg(f"🔴 STOP LOSS @ ${current_price:.4f} (${profit_usd:.3f}, {profit_pct*100:.1f}%)")
            
            # AI suggests exit or RSI overbought
            elif ai_prob <= AI_SELL_THRESHOLD or rsi_val >= RSI_OVERBOUGHT:
                if profit_usd >= MIN_PROFIT_TO_SELL or profit_pct <= -0.03:  # Exit if profitable or losing too much
                    result = place_conservative_sell_order(doge_balance, current_price, entry_price)
                    if result:
                        in_position = False
                        tg(f"🔄 CONSERVATIVE EXIT @ ${current_price:.4f} (${profit_usd:.3f}, {profit_pct*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        tg(f"⚠️ Error: {e}")
    
    time.sleep(120)  # Longer sleep for conservative approach

