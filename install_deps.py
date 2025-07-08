#!/usr/bin/env python3
"""
Install dependencies for Apollo Scalping Bot
This script installs dependencies without triggering Railway's auto-detection.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install dependencies from backend/requirements.txt"""
    print("📦 Installing Apollo Scalping Bot dependencies...")
    
    # Install dependencies from backend directory
    requirements_path = "backend/requirements.txt"
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_path
            ])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        print(f"❌ Requirements file not found: {requirements_path}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()