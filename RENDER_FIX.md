# 🔧 Render Deployment Fix

## Issue Fixed
The error was: `"/requirements.txt": not found`

## Solution Applied
✅ Copied `requirements.txt` to `backend/` directory
✅ Updated `render.yaml` configuration

## Deploy to Render (Updated Steps)

### Method 1: Using render.yaml (Recommended)
1. Go to https://render.com
2. Click "New" → "Web Service"
3. Connect GitHub: `apollo-scalping-bot`
4. Render will automatically use `render.yaml` configuration

### Method 2: Manual Configuration
1. Go to https://render.com
2. Click "New" → "Web Service"
3. Connect GitHub: `apollo-scalping-bot`
4. Configure:
   - **Name**: apollo-scalping-bot-backend
   - **Region**: Oregon
   - **Branch**: master
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Set Environment Variables in Render:
```
FIREBASE_PROJECT_ID=apollo-7c7f6
FIREBASE_API_KEY=AIzaSyDGXADaGQX14BnXl8lGs4kLEPRGXnYYlQg
FERNET_KEY=n5Rl5kkYfyCTv-2wvt5YnO0prRvzqcpAdvZkwdpNEaE=
BINANCE_API_KEY=66Jk1XMTUwXVHBnMuszBHClBGnIKw1hK9euECOXRMX8Ws35kI2THnHOTjUPuHQL9
BINANCE_API_SECRET=r1Zece1L4qiT0ngE5nHoLON0OsmHgpictoj0JFdIifD1jjW5MCllkH7iRsaSQtU9
FRONTEND_URL=https://apollo.irondevz.com
```

## Expected Result
✅ Backend will deploy successfully
✅ Available at: `https://apollo-scalping-bot-backend.onrender.com`
✅ API endpoints working
✅ Ready for frontend connection