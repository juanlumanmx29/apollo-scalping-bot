#!/usr/bin/env python3
"""
Simple debug server to test Railway deployment
"""
import os
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="DEBUG Apollo API")

@app.get("/")
def root():
    return {"message": "DEBUG: Apollo API is running", "port": os.environ.get("PORT", "unknown")}

@app.get("/health")
def health():
    return {"status": "healthy", "debug": True}

@app.get("/debug-env")
def debug_env():
    return {
        "PORT": os.environ.get("PORT", "NOT_SET"),
        "FIREBASE_PROJECT_ID": os.environ.get("FIREBASE_PROJECT_ID", "NOT_SET"),
        "RAILWAY_ENVIRONMENT": os.environ.get("RAILWAY_ENVIRONMENT", "NOT_SET"),
        "PWD": os.environ.get("PWD", "NOT_SET")
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🔧 DEBUG: Starting simple server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)