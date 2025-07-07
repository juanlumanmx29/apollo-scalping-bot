# 🌐 Netlify Deployment with Custom Domain

## Step 1: Deploy to Netlify

### Option A: GitHub Integration (Recommended)
1. Go to https://netlify.com and sign up/login
2. Click "Add new site" → "Import an existing project"
3. Choose "Deploy with GitHub"
4. Select your repository: `apollo-scalping-bot`
5. Configure build settings:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
6. Click "Deploy site"

### Option B: Drag & Drop
1. Build locally: `cd frontend && npm run build`
2. Drag the `dist` folder to Netlify dashboard

## Step 2: Set Up Custom Domain

### 2.1 Add Domain in Netlify
1. Go to Site settings → Domain management
2. Click "Add custom domain"
3. Enter: `apollo.irondevz.com`
4. Netlify will show you DNS records to configure

### 2.2 Configure DNS (at your domain provider)
Add these DNS records at your domain provider (where you bought irondevz.com):

**For subdomain apollo.irondevz.com:**
```
Type: CNAME
Name: apollo
Value: your-netlify-site.netlify.app
```

**OR if you want apex domain:**
```
Type: A
Name: @
Value: 75.2.60.5 (Netlify's load balancer)
```

### 2.3 Enable HTTPS
1. In Netlify: Site settings → Domain management
2. Scroll to HTTPS section
3. Click "Verify DNS configuration"
4. Once verified, enable "Force HTTPS"

## Step 3: Set Environment Variables

In Netlify dashboard:
1. Go to Site settings → Environment variables
2. Add:
```
VITE_BACKEND_URL=https://apollo-scalping-bot-backend.onrender.com
```

## Step 4: Update Backend CORS

In your Render backend, add environment variable:
```
FRONTEND_URL=https://apollo.irondevz.com
```

## DNS Configuration Examples

### Cloudflare:
- Type: CNAME
- Name: apollo
- Target: your-netlify-site.netlify.app
- Proxy status: DNS only (gray cloud)

### Namecheap/GoDaddy:
- Type: CNAME Record
- Host: apollo
- Value: your-netlify-site.netlify.app

### Google Domains:
- Type: CNAME
- Name: apollo
- Data: your-netlify-site.netlify.app

## Troubleshooting

**Domain not working:**
- Check DNS propagation: https://dnschecker.org
- Wait up to 24 hours for DNS changes
- Verify CNAME record is correct

**Build failing:**
- Check build logs in Netlify dashboard
- Ensure Node.js version is compatible
- Verify package.json is in frontend directory

**Backend connection error:**
- Verify VITE_BACKEND_URL is set correctly
- Check CORS settings in backend
- Test backend URL directly

## Complete Setup Checklist

✅ Netlify site deployed from GitHub
✅ Custom domain `apollo.irondevz.com` added
✅ DNS CNAME record configured
✅ HTTPS enabled
✅ Environment variables set
✅ Backend CORS updated
✅ Domain propagated and working

Your trading bot will be live at: https://apollo.irondevz.com