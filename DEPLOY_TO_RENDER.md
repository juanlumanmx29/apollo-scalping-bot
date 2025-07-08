# Deploy Apollo Scalping Bot to Render

## Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up/Sign in with GitHub

## Step 2: Create Web Service
1. Click "New" → "Web Service"
2. Connect your GitHub repository: `apollo-scalping-bot`
3. Configure the service:
   - **Name**: `apollo-scalping-bot-backend`
   - **Region**: Oregon (US West) or closest to you
   - **Branch**: `master`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`

## Step 3: Set Environment Variables
Add these in Render dashboard:
- `FIREBASE_PROJECT_ID`: `apollo-7c7f6`
- `FIREBASE_API_KEY`: `AIzaSyDGXADaGQX14BnXl8lGs4kLEPRGXnYYlQg`
- `FERNET_KEY`: (generate new one or use existing)

## Step 4: Deploy
1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Get the Render URL (e.g., `https://apollo-scalping-bot-backend.onrender.com`)

## Step 5: Update Frontend
Update frontend `.env.production` with new Render URL:
```
VITE_BACKEND_URL=https://apollo-scalping-bot-backend.onrender.com
```

## Why Render Works Better:
- ✅ Better FastAPI support
- ✅ Automatic PORT environment variable handling
- ✅ No configuration file conflicts
- ✅ More reliable deployment process