# 🚀 Complete Deployment Instructions

## Your Current Credentials
```
FIREBASE_PROJECT_ID=apollo-7c7f6
FIREBASE_API_KEY=AIzaSyDGXADaGQX14BnXl8lGs4kLEPRGXnYYlQg
FERNET_KEY=n5Rl5kkYfyCTv-2wvt5YnO0prRvzqcpAdvZkwdpNEaE=
BINANCE_API_KEY=66Jk1XMTUwXVHBnMuszBHClBGnIKw1hK9euECOXRMX8Ws35kI2THnHOTjUPuHQL9
BINANCE_API_SECRET=r1Zece1L4qiT0ngE5nHoLON0OsmHgpictoj0JFdIifD1jjW5MCllkH7iRsaSQtU9
```

## Step 1: Deploy Backend to Render

### 1.1 Create Render Account & Service
1. Go to https://render.com and create account
2. Click "New +" → "Web Service"
3. Connect GitHub: https://github.com/juanlumanmx29/apollo-scalping-bot
4. Configure:
   - **Name**: apollo-scalping-bot-backend
   - **Region**: Oregon (US West)
   - **Branch**: master
   - **Root Directory**: backend
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 1.2 Set Environment Variables in Render
In your Render service settings → Environment tab, add:

```
FIREBASE_PROJECT_ID=apollo-7c7f6
FIREBASE_API_KEY=AIzaSyDGXADaGQX14BnXl8lGs4kLEPRGXnYYlQg
FERNET_KEY=n5Rl5kkYfyCTv-2wvt5YnO0prRvzqcpAdvZkwdpNEaE=
BINANCE_API_KEY=66Jk1XMTUwXVHBnMuszBHClBGnIKw1hK9euECOXRMX8Ws35kI2THnHOTjUPuHQL9
BINANCE_API_SECRET=r1Zece1L4qiT0ngE5nHoLON0OsmHgpictoj0JFdIifD1jjW5MCllkH7iRsaSQtU9
```

### 1.3 Deploy
Click "Create Web Service" and wait for deployment.
Your backend URL will be: `https://apollo-scalping-bot-backend.onrender.com`

## Step 2: Deploy Frontend to Vercel

### 2.1 Install Vercel CLI
```bash
npm install -g vercel
```

### 2.2 Deploy Frontend
```bash
cd frontend
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? (your account)
# - Link to existing project? No
# - Project name: apollo-scalping-bot-frontend
# - Directory: ./
# - Override settings? No
```

### 2.3 Set Backend URL in Vercel
After deployment, in Vercel dashboard:
1. Go to your project settings
2. Environment Variables tab
3. Add: `VITE_BACKEND_URL` = `https://your-render-service-name.onrender.com`
4. Redeploy: `vercel --prod`

## Step 3: Update CORS Settings

### 3.1 Get Your Vercel Domain
Note your frontend URL: `https://apollo-scalping-bot-frontend.vercel.app`

### 3.2 Add Frontend URL to Backend CORS
In Render dashboard, add environment variable:
```
FRONTEND_URL=https://your-vercel-domain.vercel.app
```

## Step 4: Handle Model Issue

The ML model is too large for GitHub. Options:

### Option A: Deploy Without Model (Easiest)
- App will work with default probabilities (0.5)
- Good for testing deployment

### Option B: Upload Model Manually
1. Train model locally: `python scripts/ensemble_models.py`
2. Upload `models/ensemble_model.joblib` to Render via SSH/file upload
3. Place in `/app/models/ensemble_model.joblib`

### Option C: Cloud Storage
1. Upload model to Google Drive/Dropbox
2. Add download logic to backend startup
3. Cache model locally

## Step 5: Test Deployment

1. Visit your Vercel frontend URL
2. Sign in with Google (Firebase Auth)
3. Test paper trading features
4. Check backend logs in Render dashboard

## Troubleshooting

### Common Issues:

**CORS Error:**
- Check FRONTEND_URL is set correctly in Render
- Verify Vercel domain matches exactly

**Firebase Auth Error:**
- Check Firebase environment variables
- Verify Firebase project is configured

**Model Loading Error:**
- Expected for first deployment
- App works without model (default probabilities)

**API Connection Error:**
- Check VITE_BACKEND_URL in Vercel
- Verify backend is running in Render

### Check Logs:
- **Render**: Service logs tab
- **Vercel**: Functions tab
- **Browser**: Developer console

## Alternative: One-Click Deploy

### Render (Backend):
Use render.yaml in your repo - just connect GitHub and deploy

### Vercel (Frontend):
Import project from GitHub dashboard

## Cost Information

- **Render**: 750 hours/month free tier
- **Vercel**: 100GB bandwidth/month free tier
- **Firebase**: 1GB storage free tier

## Local Testing Before Deploy

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8001

# Frontend
cd frontend
npm install
npm run dev
```

## Final Notes

- Both platforms auto-deploy on git push
- Check deployment status in dashboards
- Monitor resource usage to stay within free tiers
- Consider upgrading if you need more resources

Your repository is ready - just follow these steps and you'll have a live trading bot!