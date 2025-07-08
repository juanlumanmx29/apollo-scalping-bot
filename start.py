#!/usr/bin/env python3
"""
Apollo Scalping Bot - Simple Startup Script
No configuration files - just pure Python startup.
"""

import os
import sys
import subprocess

def main():
    # Get port from Railway
    port = os.environ.get("PORT")
    if not port:
        print("❌ PORT not set")
        sys.exit(1)
    
    print(f"🚀 Starting Apollo Bot on port {port}")
    
    # Install dependencies if needed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])
    except:
        pass
    
    # Change to backend directory
    os.chdir("backend")
    
    # Start server using Python subprocess
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", str(port)
    ]
    
    print(f"🔧 Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()