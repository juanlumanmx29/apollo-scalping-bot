#!/usr/bin/env python3
"""
Apollo Scalping Bot - Main FastAPI Application
"""
import os
import sys
import logging
import warnings
import base64
import json
import joblib
import numpy as np
import asyncio
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import jwt
from google.cloud import firestore
from cryptography.fernet import Fernet
import requests
from features import get_features_and_predict, get_current_price_simple
import uvicorn

# Load environment variables
load_dotenv()

# Suppress ML warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', message='.*pickle.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(title="Apollo Scalping Bot API")

# Firebase Configuration
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "apollo-7c7f6")
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
FIREBASE_AUTH_URL = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"

# Encryption setup
FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    FERNET_KEY = base64.urlsafe_b64encode(os.urandom(32)).decode()
fernet = Fernet(FERNET_KEY)

security = HTTPBearer()

# Load ML Model
MODEL_PATHS = [
    'models/ensemble_model.joblib',
    'backend/models/ensemble_model.joblib',
    '/app/models/ensemble_model.joblib'
]

model = None
for MODEL_PATH in MODEL_PATHS:
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ ML Model loaded from: {MODEL_PATH}")
            break
    except Exception as e:
        logger.debug(f"Could not load from {MODEL_PATH}: {e}")
        continue

if model is None:
    logger.warning("⚠️ ML Model not found - using fallback predictions")

# Add CORS
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173", 
    "https://apollo.irondevz.com",
    "https://apollo-scalping-bot.netlify.app",
    "https://*.railway.app",
    "https://*.up.railway.app"
]

if os.getenv("FRONTEND_URL"):
    allowed_origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase token verification
def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        jwt.get_unverified_header(token)
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        user_id = decoded_token.get("user_id", "unknown")
        
        # Verify issuer and audience
        iss_ok = decoded_token.get("iss", "").endswith(FIREBASE_PROJECT_ID)
        aud_ok = decoded_token.get("aud") == FIREBASE_PROJECT_ID or decoded_token.get("aud") == FIREBASE_AUTH_URL
        
        if not iss_ok:
            raise HTTPException(status_code=401, detail="Token inválido (issuer)")
        if not aud_ok:
            raise HTTPException(status_code=401, detail="Token inválido (audiencia)")
            
        return decoded_token
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        raise HTTPException(status_code=401, detail="Token inválido")

# Basic endpoints
@app.get("/")
def root():
    return {"message": "Apollo Scalping Bot API funcionando 🚀", "status": "healthy", "version": "v2.0-debug", "timestamp": datetime.now().isoformat()}

@app.get("/health") 
def health():
    return {"status": "healthy", "service": "apollo-backend"}

@app.get("/test-cors")
def test_cors():
    return {"message": "CORS funcionando correctamente"}

@app.get("/debug-binance")
def debug_binance():
    """Debug endpoint to test Binance API from Railway"""
    import socket
    import urllib.parse
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "network_tests": {},
        "api_tests": {},
        "error_details": []
    }
    
    # Test 1: DNS Resolution
    try:
        ip = socket.gethostbyname("api.binance.com")
        results["network_tests"]["dns_resolution"] = f"✅ api.binance.com -> {ip}"
    except Exception as e:
        results["network_tests"]["dns_resolution"] = f"❌ DNS failed: {e}"
    
    # Test 2: Basic connectivity
    try:
        response = requests.get("https://httpbin.org/ip", timeout=5)
        results["network_tests"]["external_access"] = f"✅ External access: {response.json()}"
    except Exception as e:
        results["network_tests"]["external_access"] = f"❌ External access failed: {e}"
    
    # Test 3: Binance API endpoints
    binance_urls = [
        "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT",
        "https://api1.binance.com/api/v3/ticker/price?symbol=ETHUSDT",
        "https://api.binance.com/api/v3/ping"
    ]
    
    for url in binance_urls:
        try:
            response = requests.get(url, timeout=10)
            results["api_tests"][url] = f"✅ {response.status_code}: {response.text[:100]}"
        except Exception as e:
            results["api_tests"][url] = f"❌ {type(e).__name__}: {str(e)}"
    
    # Test 4: Our functions
    try:
        price_result = get_current_price_simple("ETHUSDT")
        results["our_functions"] = {
            "simple_price": price_result,
            "price_success": price_result is not None
        }
    except Exception as e:
        results["our_functions"] = {"error": str(e)}
    
    return results

# Utility functions
def get_firestore_client():
    return firestore.Client()

# Price and prediction endpoints
@app.get("/price")
def get_price(user=Depends(verify_firebase_token)):
    """Get current ETH/USDT price from Binance"""
    try:
        urls = [
            "https://api.binance.com/api/v3/ticker/price",
            "https://api1.binance.com/api/v3/ticker/price",
            "https://api2.binance.com/api/v3/ticker/price"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in urls:
            try:
                response = requests.get(url, params={"symbol": "ETHUSDT"}, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                price = float(data["price"])
                return {"symbol": data["symbol"], "price": price}
            except Exception as e:
                logger.debug(f"Failed to get price from {url}: {e}")
                continue
        
        raise Exception("All price endpoints failed")
    except Exception as e:
        logger.error(f"Error getting price: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting price: {str(e)}")

@app.get("/probability")
def get_probability(user=Depends(verify_firebase_token)):
    """Get ML model probability prediction"""
    try:
        _, _, prob = get_features_and_predict(model)
        logger.info(f"Probability calculated: {prob:.3f}")
        return {"probability": prob}
    except Exception as e:
        logger.error(f"Error calculating probability: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating probability: {str(e)}")

# Trading state management
paper_trading_status = {}
paper_trading_logs = {}
paper_trading_trades = {}
paper_trading_tasks = {}

@app.get("/paper-trading-status")
def get_paper_trading_status(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    status = paper_trading_status.get(user_id, "pausado")
    return {"status": status}

@app.get("/trades")
def get_trades(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    trades = paper_trading_trades.get(user_id, [])
    return {"trades": trades}

@app.get("/paper-trading-logs")
def get_paper_trading_logs(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    logs = paper_trading_logs.get(user_id, ["🚀 Apollo Bot Ready"])
    return {"logs": logs}

@app.post("/start-paper-trading") 
def start_paper_trading(background_tasks: BackgroundTasks, user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    if paper_trading_status.get(user_id) == "activo":
        return {"status": "ya activo"}
    
    paper_trading_status[user_id] = "activo"
    logger.info(f"Starting paper trading for user: {user_id}")
    
    # Start background trading task
    background_tasks.add_task(run_paper_trading, user_id, model)
    return {"status": "iniciado"}

@app.post("/stop-paper-trading")
def stop_paper_trading(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    paper_trading_status[user_id] = "pausado"
    logger.info(f"Stopping paper trading for user: {user_id}")
    return {"status": "detenido"}

# Apollo Trading Algorithm
async def run_paper_trading(user_id, model, threshold=0.73):
    """Complete Apollo scalping strategy implementation"""
    try:
        logs = []
        trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        capital = 10000
        
        # Apollo Strategy Parameters
        commission_rate = 0.00075  # 0.075% per transaction
        minimum_sell_percentage = 0.0016  # 0.16% minimum profit
        stop_loss_percentage = 0.01  # 1% stop loss
        trailing_stop_activation = 0.0016  # Trailing stop at 0.16%
        trailing_stop_distance = 0.002  # 0.2% below max price
        
        # Trailing stop variables
        trailing_stop_active = False
        max_price_since_entry = 0
        minimum_sell_price = 0
        
        total_trades = 0
        winning_trades = 0
        
        while paper_trading_status.get(user_id) == "activo":
            now = datetime.now().strftime("%H:%M:%S")
            
            try:
                current_price, _, proba = get_features_and_predict(model)
                if current_price is None:
                    logger.warning("Error getting Binance data")
                    logs.insert(0, f"[{now}] ❌ Error getting Binance data")
                    await asyncio.sleep(5)
                    paper_trading_logs[user_id] = logs[:100]
                    continue
                    
                signal = 1 if proba >= threshold else 0
                
                # APOLLO TRADING LOGIC
                if in_position:
                    # Update max price for trailing stop
                    if current_price > max_price_since_entry:
                        max_price_since_entry = current_price
                    
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Activate trailing stop at 0.16% profit
                    if not trailing_stop_active and pnl_pct >= trailing_stop_activation:
                        trailing_stop_active = True
                        logs.insert(0, f"[{now}] 🔄 TRAILING STOP ACTIVATED | Max: ${max_price_since_entry:.2f}")
                    
                    # Calculate stop prices
                    trailing_stop_price = max_price_since_entry * (1 - trailing_stop_distance)
                    stop_loss_price = entry_price * (1 - stop_loss_percentage)
                    
                    should_sell = False
                    sell_reason = ""
                    
                    # Exit conditions
                    if current_price <= stop_loss_price:
                        should_sell = True
                        sell_reason = "STOP_LOSS"
                    elif trailing_stop_active and current_price <= trailing_stop_price and current_price >= minimum_sell_price:
                        should_sell = True
                        sell_reason = "TRAILING_STOP"
                    
                    if should_sell:
                        # Execute sell
                        in_position = False
                        
                        # Calculate P&L with commissions
                        sell_commission = current_price * commission_rate
                        buy_commission = entry_price * commission_rate
                        total_commission = buy_commission + sell_commission
                        
                        net_profit = current_price - entry_price - total_commission
                        pnl_pct = net_profit / entry_price
                        
                        capital += net_profit
                        total_trades += 1
                        
                        if net_profit > 0:
                            winning_trades += 1
                        
                        trades.append({
                            'entry_time': entry_time.isoformat() if entry_time else None,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl_pct': pnl_pct * 100,
                            'net_profit': net_profit,
                            'total_commission': total_commission,
                            'capital': capital,
                            'reason': sell_reason,
                            'timestamp': datetime.now().isoformat(),
                            'max_price': max_price_since_entry if trailing_stop_active else None
                        })
                        
                        logs.insert(0, f"[{now}] 🔴 SELL ({sell_reason}): ${current_price:.2f} | P&L: {pnl_pct*100:.2f}% | Capital: ${capital:.2f}")
                        
                        # Reset variables
                        entry_price = 0
                        entry_time = None
                        trailing_stop_active = False
                        max_price_since_entry = 0
                        minimum_sell_price = 0
                    else:
                        # Position monitoring
                        status = f"TS: {'✅' if trailing_stop_active else '❌'}"
                        if trailing_stop_active:
                            status += f" | TSP: ${trailing_stop_price:.2f}"
                        logs.insert(0, f"[{now}] 📊 IN POSITION: ${current_price:.2f} | P&L: {pnl_pct*100:.2f}% | {status}")
                        
                elif signal == 1 and not in_position:
                    # BUY SIGNAL
                    in_position = True
                    entry_price = current_price
                    entry_time = datetime.now()
                    
                    # Calculate minimum sell price
                    minimum_sell_price = entry_price * (1 + minimum_sell_percentage)
                    max_price_since_entry = current_price
                    
                    # Deduct buy commission
                    buy_commission = entry_price * commission_rate
                    capital -= buy_commission
                    
                    logs.insert(0, f"[{now}] 🟢 BUY: ${current_price:.2f} | Min Sell: ${minimum_sell_price:.2f} | Capital: ${capital:.2f}")
                else:
                    # Monitoring mode
                    logs.insert(0, f"[{now}] 📈 MONITORING: ${current_price:.2f} | Prob: {proba:.3f} | Signal: {'🚀' if signal else '⏳'}")
                
                # Update global state
                paper_trading_logs[user_id] = logs[:100]
                paper_trading_trades[user_id] = trades[-100:]
                
            except Exception as e:
                logs.insert(0, f"[{now}] ❌ ERROR: {str(e)}")
                paper_trading_logs[user_id] = logs[:100]
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"Paper trading error for user {user_id}: {e}")
        paper_trading_status[user_id] = "error"

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Apollo Scalping Bot API started")
    logger.info(f"📊 ML Model available: {'Yes' if model else 'No'}")
    logger.info(f"🔥 Firebase Project ID: {FIREBASE_PROJECT_ID}")
    logger.info("💹 REAL BINANCE DATA MODE - NO SIMULATION")
    logger.info("="*50)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Apollo Scalping Bot on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)