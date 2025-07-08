from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv
import jwt
from google.cloud import firestore
from cryptography.fernet import Fernet
import base64
import json
import joblib
import numpy as np
from features import get_features_and_predict
import asyncio
from datetime import datetime
import warnings
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir warnings específicos de sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', message='.*pickle.*')

load_dotenv()

app = FastAPI(title="Apollo Scalping Bot API")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Apollo Scalping Bot API iniciado")
    logger.info(f"📊 Modelo ML disponible: {'Sí' if model else 'No'}")
    logger.info(f"🔥 Firebase Project ID: {FIREBASE_PROJECT_ID}")
    logger.info("="*50)

# Middleware de CORS (debe ir antes de cualquier endpoint)
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://apollo.irondevz.com",  # Production domain
    "https://apollo-scalping-bot.netlify.app",  # Netlify fallback
    "https://your-frontend-domain.vercel.app"  # Vercel fallback
]

# Add production frontend domain if environment variable is set
if os.getenv("FRONTEND_URL"):
    allowed_origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de Firebase
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
FIREBASE_AUTH_URL = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"

# Clave de cifrado Fernet (en entorno real, almacena esto seguro)
FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    # Generar una clave si no existe (solo para desarrollo)
    FERNET_KEY = base64.urlsafe_b64encode(os.urandom(32)).decode()
fernet = Fernet(FERNET_KEY)

security = HTTPBearer()

# Función para verificar el token JWT de Firebase
def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Verificar que el token tiene un formato válido
        jwt.get_unverified_header(token)
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        # Solo logear el user_id para reducir ruido en logs
        user_id = decoded_token.get("user_id", "unknown")
        logger.debug(f"Token verificado para usuario: {user_id}")
        
        # Relajar verificación de issuer y audiencia
        iss_ok = decoded_token.get("iss", "").endswith(FIREBASE_PROJECT_ID)
        aud_ok = decoded_token.get("aud") == FIREBASE_PROJECT_ID or decoded_token.get("aud") == FIREBASE_AUTH_URL
        if not iss_ok:
            logger.warning(f"Issuer no válido: {decoded_token.get('iss')}")
            raise HTTPException(status_code=401, detail="Token inválido (issuer)")
        if not aud_ok:
            logger.warning(f"Audiencia no válida: {decoded_token.get('aud')}")
            raise HTTPException(status_code=401, detail="Token inválido (audiencia)")
        return decoded_token
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verificando token: {e}")
        raise HTTPException(status_code=401, detail="Token inválido")

@app.get("/")
def root():
    return {"message": "API de Apollo Scalping Bot funcionando 🚀"}

@app.get("/protected")
def protected_route(user=Depends(verify_firebase_token)):
    return {"message": f"¡Acceso autorizado! Usuario: {user.get('user_id', 'desconocido')}"}

# Endpoint para guardar credenciales encriptadas
def get_firestore_client():
    return firestore.Client()

@app.post("/save-credentials")
async def save_credentials(request: Request, user=Depends(verify_firebase_token)):
    data = await request.json()
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    if not api_key or not api_secret:
        raise HTTPException(status_code=400, detail="Faltan credenciales")
    # Encriptar
    payload = json.dumps({"api_key": api_key, "api_secret": api_secret}).encode()
    encrypted = fernet.encrypt(payload).decode()
    # Guardar en Firestore
    db = get_firestore_client()
    user_id = user.get("user_id") or user.get("uid")
    db.collection("users").document(user_id).set({"binance_credentials": encrypted}, merge=True)
    return {"message": "Credenciales guardadas correctamente"}

@app.get("/get-credentials")
def get_credentials(user=Depends(verify_firebase_token)):
    db = get_firestore_client()
    user_id = user.get("user_id") or user.get("uid")
    doc = db.collection("users").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="No hay credenciales guardadas")
    encrypted = doc.to_dict().get("binance_credentials")
    if not encrypted:
        raise HTTPException(status_code=404, detail="No hay credenciales guardadas")
    # Desencriptar
    decrypted = fernet.decrypt(encrypted.encode()).decode()
    return json.loads(decrypted)

@app.get("/test-cors")
def test_cors():
    return {"message": "CORS funcionando correctamente"}

@app.get("/test-binance")
def test_binance():
    """Test Binance connectivity from server"""
    try:
        import requests
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        if response.status_code == 200:
            return {"status": "Binance API accessible", "response": response.json()}
        else:
            return {"status": "Binance API failed", "status_code": response.status_code}
    except Exception as e:
        return {"status": "Binance API error", "error": str(e)}

@app.get("/debug")
def debug_endpoint(user=Depends(verify_firebase_token)):
    """Endpoint para debug - verifica que la autenticación funciona"""
    user_id = user.get("user_id", "unknown")
    logger.info(f"Debug endpoint llamado por usuario: {user_id}")
    return {"message": "Autenticación funcionando", "user_id": user_id}

@app.get("/price")
def get_price(_=Depends(verify_firebase_token)):
    """
    Devuelve el precio actual de ETH/USDT desde Binance.
    """
    try:
        # Try multiple endpoints with better headers
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
                logger.info(f"Precio ETH/USDT obtenido: ${price:.2f}")
                return {"symbol": data["symbol"], "price": price}
            except Exception as e:
                logger.debug(f"Failed to get price from {url}: {e}")
                continue
        
        raise Exception("All price endpoints failed")
    except Exception as e:
        logger.error(f"Error obteniendo precio: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener el precio: {str(e)}")

# Cargar el modelo ML al iniciar
# Try multiple possible paths for different deployment environments
MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), '../models/ensemble_model.joblib'),  # Local development
    os.path.join(os.getcwd(), 'models/ensemble_model.joblib'),  # Render deployment
    '/app/models/ensemble_model.joblib',  # Docker deployment
    'models/ensemble_model.joblib'  # Direct path
]

model = None
for MODEL_PATH in MODEL_PATHS:
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Modelo ML cargado correctamente desde: {MODEL_PATH}")
            break
    except Exception as e:
        logger.debug(f"No se pudo cargar desde {MODEL_PATH}: {e}")
        continue

if model is None:
    logger.warning("⚠️  No se pudo cargar el modelo ML desde ninguna ubicación")
    logger.info("🔄 Funcionando en modo sin modelo - usando probabilidades por defecto")

# Función para predecir probabilidad (simulación de features)
def predict_probability():
    # Aquí deberías obtener los features reales de Binance y procesarlos
    # Por ahora, simulamos un vector de features aleatorio del tamaño esperado por el modelo
    X = np.random.rand(1, model.n_features_in_) if model else [[0]*10]
    prob = float(model.predict_proba(X)[0,1]) if model else 0.5
    return prob

@app.get("/probability")
def get_probability(user=Depends(verify_firebase_token)):
    """
    Devuelve la probabilidad predicha por el modelo ML usando datos actuales de Binance.
    """
    try:
        _, _, prob = get_features_and_predict(model)
        logger.info(f"Probabilidad calculada: {prob:.3f}")
        return {"probability": prob}
    except Exception as e:
        logger.error(f"Error calculando probabilidad: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir probabilidad: {str(e)}")

# Estructuras en memoria
paper_trading_tasks = {}
paper_trading_status = {}
paper_trading_logs = {}
paper_trading_trades = {}

async def paper_trading_loop(user_id, model, threshold=0.73):
    logs = []
    trades = []
    in_position = False
    entry_price = 0
    entry_time = None
    capital = 10000
    
    # ORIGINAL APOLLO STRATEGY PARAMETERS
    commission_rate = 0.00075  # Binance 0.075% per transaction
    minimum_sell_percentage = 0.0016  # 0.16% minimum profit to cover commissions
    stop_loss_percentage = 0.01  # 1% stop loss
    trailing_stop_activation = 0.0016  # Trailing stop activates at 0.16% profit
    trailing_stop_distance = 0.002  # 0.2% below max price
    
    # Trailing stop variables
    trailing_stop_active = False
    max_price_since_entry = 0
    minimum_sell_price = 0
    
    total_trades = 0
    winning_trades = 0
    try:
        while paper_trading_status.get(user_id) == "activo":
            now = datetime.now().strftime("%H:%M:%S")
            current_price, _, proba = get_features_and_predict(model)
            if current_price is None:
                logger.warning("Error obteniendo datos de Binance")
                logs.insert(0, f"[{now}] ❌ Error obteniendo datos de Binance")
                await asyncio.sleep(5)
                paper_trading_logs[user_id] = logs[:100]
                continue
            signal = 1 if proba >= threshold else 0
            
            # ORIGINAL APOLLO TRADING LOGIC
            if in_position:
                # Update max price for trailing stop
                if current_price > max_price_since_entry:
                    max_price_since_entry = current_price
                
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Activate trailing stop when price exceeds 0.16% profit
                if not trailing_stop_active and pnl_pct >= trailing_stop_activation:
                    trailing_stop_active = True
                    logs.insert(0, f"[{now}] 🔄 TRAILING STOP ACTIVADO | Max: ${max_price_since_entry:.2f}")
                
                # Calculate trailing stop price (0.2% below max)
                trailing_stop_price = max_price_since_entry * (1 - trailing_stop_distance)
                
                # Calculate stop loss price (1% below entry)
                stop_loss_price = entry_price * (1 - stop_loss_percentage)
                
                should_sell = False
                sell_reason = ""
                
                # Check exit conditions in order of priority:
                
                # 1. Stop Loss (1% below entry price)
                if current_price <= stop_loss_price:
                    should_sell = True
                    sell_reason = "STOP_LOSS"
                
                # 2. Trailing Stop (if active and price below trailing stop, but above minimum sell)
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
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct * 100,
                        'net_profit': net_profit,
                        'total_commission': total_commission,
                        'capital': capital,
                        'reason': sell_reason,
                        'timestamp': datetime.now(),
                        'max_price': max_price_since_entry if trailing_stop_active else None
                    })
                    
                    logs.insert(0, f"[{now}] 🔴 VENTA ({sell_reason}): ${current_price:.2f} | P&L: {pnl_pct*100:.2f}% | Capital: ${capital:.2f}")
                    
                    # Reset variables
                    entry_price = 0
                    entry_time = None
                    trailing_stop_active = False
                    max_price_since_entry = 0
                    minimum_sell_price = 0
                else:
                    # Position monitoring log
                    status = f"TS: {'✅' if trailing_stop_active else '❌'}"
                    if trailing_stop_active:
                        status += f" | TSP: ${trailing_stop_price:.2f}"
                    logs.insert(0, f"[{now}] 📊 EN POSICIÓN: ${current_price:.2f} | P&L: {pnl_pct*100:.2f}% | {status}")
                    
            elif signal == 1 and not in_position:
                # BUY SIGNAL - Enter position
                in_position = True
                entry_price = current_price
                entry_time = datetime.now()
                
                # Calculate minimum sell price (0.16% above entry to cover commissions)
                minimum_sell_price = entry_price * (1 + minimum_sell_percentage)
                max_price_since_entry = current_price
                
                # Deduct buy commission
                buy_commission = entry_price * commission_rate
                capital -= buy_commission
                
                logs.insert(0, f"[{now}] 🟢 COMPRA: ${current_price:.2f} | Min Venta: ${minimum_sell_price:.2f} | Capital: ${capital:.2f}")
            else:
                # No position, monitoring
                logs.insert(0, f"[{now}] 📈 MONITOREO: ${current_price:.2f} | Prob: {proba:.3f} | Señal: {'🚀' if signal else '⏳'}")
            # Limitar logs y trades
            paper_trading_logs[user_id] = logs[:100]
            paper_trading_trades[user_id] = trades[-100:]
            await asyncio.sleep(5)
    except Exception as e:
        logs.insert(0, f"[ERROR] {str(e)}")
        paper_trading_logs[user_id] = logs[:100]

def run_paper_trading_sync(user_id, model, threshold=0.73):
    """Wrapper function to run async paper trading in background"""
    try:
        asyncio.run(paper_trading_loop(user_id, model, threshold))
    except Exception as e:
        logger.error(f"Error en paper trading para usuario {user_id}: {e}")
        paper_trading_status[user_id] = "error"

@app.post("/start-paper-trading")
def start_paper_trading(background_tasks: BackgroundTasks, user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    if paper_trading_status.get(user_id) == "activo":
        return {"status": "ya activo"}
    paper_trading_status[user_id] = "activo"
    logger.info(f"Iniciando paper trading para usuario: {user_id}")
    # Usar BackgroundTasks para lanzar el ciclo de trading en segundo plano
    background_tasks.add_task(run_paper_trading_sync, user_id, model)
    return {"status": "iniciado"}

@app.post("/stop-paper-trading")
def stop_paper_trading(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    paper_trading_status[user_id] = "pausado"
    logger.info(f"Deteniendo paper trading para usuario: {user_id}")
    return {"status": "detenido"}

@app.get("/paper-trading-status")
def get_paper_trading_status(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    status = paper_trading_status.get(user_id, "pausado")
    return {"status": status}

@app.get("/paper-trading-logs")
def get_paper_trading_logs(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    logs = paper_trading_logs.get(user_id, [])
    return {"logs": logs}

@app.get("/trades")
def get_trades(user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    trades = paper_trading_trades.get(user_id, [])
    return {"trades": trades}