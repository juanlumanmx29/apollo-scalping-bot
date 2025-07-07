# 🚀 Apollo Scalping Bot Deployment Guide

## Prerequisites

1. **GitHub Repository**: ✅ Code is now on GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)

## Step 1: Deploy Backend to Render

### 1.1 Create Render Web Service
1. Go to [render.com](https://render.com) and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `https://github.com/juanlumanmx29/apollo-scalping-bot`
4. Configure the service:
   - **Name**: `apollo-scalping-bot-backend`
   - **Region**: Oregon (US West)
   - **Branch**: `master`
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 1.2 Set Environment Variables in Render
In your Render service settings, add these environment variables:

⚠️ **SECURITY NOTICE**: Never commit secrets to code. Set these in Render dashboard only.

```env
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_API_KEY=your-firebase-api-key
FERNET_KEY=your-generated-fernet-key
BINANCE_API_KEY=your-binance-api-key
BINANCE_API_SECRET=your-binance-api-secret
```

**To generate a new FERNET_KEY:**
```python
import base64
import os
print(base64.urlsafe_b64encode(os.urandom(32)).decode())
```

### 1.3 Deploy
Click "Create Web Service" and Render will automatically deploy your backend.

Your backend will be deployed at: `https://your-service-name.onrender.com`

## Step 2: Deploy Frontend to Vercel

### 2.1 Install Vercel CLI
```bash
npm install -g vercel
```

### 2.2 Deploy Frontend
```bash
# Navigate to frontend directory
cd frontend

# Deploy to Vercel
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? (your account)
# - Link to existing project? No
# - Project name: apollo-scalping-bot-frontend
# - Directory: ./
# - Override settings? No
```

### 2.3 Set Environment Variable in Vercel
After deployment, go to your Vercel dashboard and set:

```env
VITE_BACKEND_URL=https://your-service-name.onrender.com
```

Then redeploy:
```bash
vercel --prod
```

## Step 3: Update CORS Settings

### 3.1 Get Your Vercel Domain
After deployment, note your Vercel domain (e.g., `https://apollo-scalping-bot-frontend.vercel.app`)

### 3.2 Update Backend CORS
In Render dashboard, add environment variable:
```env
FRONTEND_URL=https://your-vercel-domain.vercel.app
```

## Step 4: Test Deployment

1. **Visit your frontend URL**
2. **Sign in with Google**
3. **Test the paper trading functionality**
4. **Verify real-time price updates**

## Important: Model Setup

⚠️ **The ML model is too large for GitHub and needs to be uploaded separately.**

See `MODEL_SETUP.md` for detailed instructions on model deployment.

## Alternative: One-Click Deployment

### Backend (Render)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/juanlumanmx29/apollo-scalping-bot)

### Frontend (Vercel)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/juanlumanmx29/apollo-scalping-bot&project-name=apollo-scalping-bot-frontend)

## Troubleshooting

### Common Issues

1. **CORS Error**: Make sure `FRONTEND_URL` is set correctly in Render
2. **Firebase Auth Error**: Verify Firebase environment variables
3. **Model Loading Error**: The app will work without the model (see MODEL_SETUP.md)
4. **API Connection Error**: Check that `VITE_BACKEND_URL` points to your Render domain

### Logs

- **Render**: Check logs in Render dashboard
- **Vercel**: Check function logs in Vercel dashboard
- **Browser**: Check console for frontend errors

## Production Considerations

1. **Security**: 
   - Use environment variables for all secrets
   - Enable HTTPS everywhere
   - Review Firebase security rules

2. **Performance**:
   - Monitor API rate limits (Binance)
   - Consider caching strategies
   - Monitor memory usage

3. **Monitoring**:
   - Set up error tracking (Sentry)
   - Monitor uptime
   - Track trading performance

## Scaling

- **Render**: Automatically scales with traffic
- **Vercel**: Serverless functions scale automatically
- **Database**: Consider upgrading Firebase plan if needed

## Cost Estimation

- **Render**: Free tier includes 750 hours/month
- **Vercel**: Free tier includes 100GB bandwidth/month
- **Firebase**: Free tier includes 1GB storage

Both platforms offer generous free tiers perfect for this application.