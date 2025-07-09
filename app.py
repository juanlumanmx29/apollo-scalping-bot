#!/usr/bin/env python3
"""
Apollo Scalping Bot - Main FastAPI Application
"""
import os
import sys
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add backend directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(title="Apollo Scalping Bot API")

# Add CORS
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173", 
    "https://apollo.irondevz.com",
    "https://apollo-scalping-bot.netlify.app",
    "https://*.railway.app",
    "https://*.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic endpoints
@app.get("/")
def root():
    return {"message": "Apollo Scalping Bot API funcionando 🚀", "status": "healthy"}

@app.get("/health") 
def health():
    return {"status": "healthy", "service": "apollo-backend"}

@app.get("/test-cors")
def test_cors():
    return {"message": "CORS funcionando correctamente"}

# Mock trading endpoints for now (will implement full functionality later)
@app.get("/paper-trading-status")
def paper_trading_status():
    return {"status": "pausado"}

@app.get("/trades")
def trades():
    return {"trades": []}

@app.get("/paper-trading-logs")
def paper_trading_logs():
    return {"logs": ["✅ Railway deployment successful!"]}

@app.post("/start-paper-trading") 
def start_paper_trading():
    return {"status": "Railway version - trading not yet implemented"}

@app.post("/stop-paper-trading")
def stop_paper_trading():
    return {"status": "Railway version - trading not yet implemented"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 NUCLEAR: Starting minimal Apollo API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)