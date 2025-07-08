# 🚂 Railway Deployment Guide

## Apollo Scalping Bot - Railway Setup

### Backend Deployment
1. **Connect GitHub** to Railway
2. **Deploy from repo** - select `backend` directory
3. **Add Environment Variables:**
   ```env
   FIREBASE_PROJECT_ID=apollo-7c7f6
   FIREBASE_API_KEY=AIzaSyDGXADaGQX14BnXl8lGs4kLEPRGXnYYlQg
   ```

### Configuration Files
- `nixpacks.toml` - Railway build configuration
- `Procfile` - Start command
- `runtime.txt` - Python version

### Railway URL
Backend: `https://apollo-scalping-bot-backend-production.up.railway.app`

### Features
- ✅ Binance API access (no 451 errors)
- ✅ ML model deployment
- ✅ Firebase authentication
- ✅ Real-time paper trading
- ✅ Apollo scalping strategy