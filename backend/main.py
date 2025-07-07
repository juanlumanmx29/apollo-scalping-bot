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
    "https://your-frontend-domain.vercel.app"  # Replace with your actual Vercel domain
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
        response = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "ETHUSDT"})
        response.raise_for_status()
        data = response.json()
        price = float(data["price"])
        logger.info(f"Precio ETH/USDT obtenido: ${price:.2f}")
        return {"symbol": data["symbol"], "price": price}
    except Exception as e:
        logger.error(f"Error obteniendo precio: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener el precio: {str(e)}")

# Cargar el modelo ML al iniciar
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/ensemble_model.joblib')
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Modelo ML cargado correctamente")
except Exception as e:
    logger.warning(f"⚠️  No se pudo cargar el modelo ML: {e}")
    logger.info("🔄 Funcionando en modo sin modelo - usando probabilidades por defecto")
    model = None

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
    commission_rate = 0.001
    profit_target = 0.015
    stop_loss = 0.008
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
            logs.insert(0, f"[{now}] Precio: ${current_price:.2f} | Prob: {proba:.3f} | Señal: {signal}")
            # Simular lógica de trading
            if in_position:
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct >= profit_target:
                    # Vender por take profit
                    in_position = False
                    pnl_amount = capital * pnl_pct
                    exit_value = capital * (1 + pnl_pct)
                    commission = exit_value * commission_rate
                    capital += pnl_amount - commission
                    total_trades += 1
                    if pnl_amount > 0:
                        winning_trades += 1
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'commission': commission,
                        'capital': capital,
                        'reason': 'PROFIT_TARGET',
                        'timestamp': datetime.now()
                    })
                    logs.insert(0, f"[{now}] 🔴 VENTA (TP): ${current_price:.2f} | P&L: {pnl_pct*100:.2f}% | Capital: ${capital:.2f}")
                    entry_price = 0
                    entry_time = None
                elif pnl_pct <= -stop_loss:
                    # Vender por stop loss
                    in_position = False
                    pnl_amount = capital * pnl_pct
                    exit_value = capital * (1 + pnl_pct)
                    commission = exit_value * commission_rate
                    capital += pnl_amount - commission
                    total_trades += 1
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'commission': commission,
                        'capital': capital,
                        'reason': 'STOP_LOSS',
                        'timestamp': datetime.now()
                    })
                    logs.insert(0, f"[{now}] 🔴 VENTA (SL): ${current_price:.2f} | P&L: {pnl_pct*100:.2f}% | Capital: ${capital:.2f}")
                    entry_price = 0
                    entry_time = None
                else:
                    # Mantener posición
                    pass
            elif signal == 1 and not in_position:
                # Comprar
                in_position = True
                entry_price = current_price
                entry_time = datetime.now()
                commission = capital * commission_rate
                capital -= commission
                logs.insert(0, f"[{now}] 🟢 COMPRA: ${current_price:.2f} | Capital: ${capital:.2f}")
            # Limitar logs y trades
            paper_trading_logs[user_id] = logs[:100]
            paper_trading_trades[user_id] = trades[-100:]
            await asyncio.sleep(5)
    except Exception as e:
        logs.insert(0, f"[ERROR] {str(e)}")
        paper_trading_logs[user_id] = logs[:100]

@app.post("/start-paper-trading")
def start_paper_trading(background_tasks: BackgroundTasks, user=Depends(verify_firebase_token)):
    user_id = user.get("user_id") or user.get("uid")
    if paper_trading_status.get(user_id) == "activo":
        return {"status": "ya activo"}
    paper_trading_status[user_id] = "activo"
    logger.info(f"Iniciando paper trading para usuario: {user_id}")
    # Usar BackgroundTasks para lanzar el ciclo de trading en segundo plano
    background_tasks.add_task(asyncio.run, paper_trading_loop(user_id, model))
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