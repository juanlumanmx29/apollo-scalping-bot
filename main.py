#!/usr/bin/env python3
"""
Apollo Scalping Bot - Minimal Railway Deployment
Simple entry point that Railway will execute without auto-detection.
"""

import os
import sys
import subprocess

def install_deps():
    """Install dependencies"""
    deps = [
        "fastapi==0.100.0",
        "uvicorn[standard]==0.23.0", 
        "PyJWT==2.7.0",
        "cryptography==41.0.0",
        "google-cloud-firestore==2.11.0",
        "requests==2.31.0",
        "pandas==2.0.3",
        "numpy==1.24.0",
        "scikit-learn==1.3.0",
        "joblib==1.3.0",
        "python-dotenv==1.0.0",
        "python-multipart==0.0.6"
    ]
    
    for dep in deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass

def start_server():
    """Start the Apollo server"""
    # Get port
    port = os.environ.get("PORT", "8000")
    print(f"🚀 Apollo starting on port {port}")
    
    # Install dependencies
    install_deps()
    
    # Import after installation
    sys.path.insert(0, "backend")
    os.chdir("backend")
    
    try:
        from main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(port))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()