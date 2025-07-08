#!/usr/bin/env python3
"""
Apollo Scalping Bot - Direct Server Launch
This script bypasses Railway's auto-detection completely.
"""

import os
import sys
import subprocess
import signal
import time

def get_port():
    """Get port from Railway environment variable"""
    port = os.environ.get("PORT")
    if not port:
        print("❌ ERROR: PORT environment variable not set")
        sys.exit(1)
    
    try:
        port_num = int(port)
        print(f"✅ PORT detected: {port_num}")
        return port_num
    except ValueError:
        print(f"❌ ERROR: Invalid PORT value: {port}")
        sys.exit(1)

def start_server():
    """Start the Apollo Scalping Bot server"""
    print("🚀 Apollo Scalping Bot - Starting Server")
    print("=" * 50)
    
    # Get the port
    port = get_port()
    
    # Set environment variables
    os.environ["PYTHONPATH"] = "/app/backend"
    
    # Change to backend directory
    backend_dir = "/app/backend"
    if os.path.exists(backend_dir):
        os.chdir(backend_dir)
        print(f"📁 Working directory: {backend_dir}")
    else:
        # Local development
        backend_dir = os.path.join(os.path.dirname(__file__), "backend")
        if os.path.exists(backend_dir):
            os.chdir(backend_dir)
            print(f"📁 Working directory: {backend_dir}")
        else:
            print("❌ ERROR: Backend directory not found")
            sys.exit(1)
    
    # Start the server using Python directly
    print(f"🔧 Starting uvicorn server on 0.0.0.0:{port}")
    
    try:
        # Import and run the app
        sys.path.insert(0, ".")
        from main import app
        import uvicorn
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_server()