#!/usr/bin/env python3
"""
Simple Railway startup - bypass all import issues
"""
import os
import sys
import subprocess

def main():
    # Get port
    port = os.environ.get("PORT", "8000")
    print(f"🚀 Apollo Bot starting on port {port}")
    
    # Install deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"], 
                   capture_output=True)
    
    # Run uvicorn directly on backend
    os.chdir("backend")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", port
    ])

if __name__ == "__main__":
    main()