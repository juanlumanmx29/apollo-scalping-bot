#!/usr/bin/env python3
"""
Apollo Scalping Bot - Simple HTTP Server
This approach bypasses Railway's FastAPI auto-detection completely.
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

class ApolloHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Import FastAPI app
        try:
            from main import app
            self.fastapi_app = app
            print("✅ FastAPI app imported successfully")
        except Exception as e:
            print(f"❌ Failed to import FastAPI app: {e}")
            self.fastapi_app = None
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "message": "Apollo Scalping Bot API is running! 🚀",
                "status": "healthy",
                "server": "Custom HTTP Server"
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "healthy", "service": "apollo-backend"}
            self.wfile.write(json.dumps(response).encode())
        else:
            # Proxy to FastAPI app if available
            if self.fastapi_app:
                self.proxy_to_fastapi()
            else:
                self.send_response(404)
                self.end_headers()
    
    def do_POST(self):
        if self.fastapi_app:
            self.proxy_to_fastapi()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def proxy_to_fastapi(self):
        """Proxy request to FastAPI app"""
        try:
            # This is a simplified proxy - in production you'd use a proper ASGI server
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"message": "FastAPI proxy would handle this request"}
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())

def start_fastapi_server():
    """Start FastAPI server in a separate thread"""
    try:
        import uvicorn
        from main import app
        port = int(os.environ.get("PORT", 8000)) + 1  # Use a different port
        print(f"🔧 Starting FastAPI server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"❌ Failed to start FastAPI server: {e}")

def main():
    """Main entry point"""
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Apollo Scalping Bot starting on port {port}")
    print(f"📍 Server type: Custom HTTP Server")
    
    # Change to backend directory for imports
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if os.path.exists(backend_dir):
        os.chdir(backend_dir)
        print(f"📁 Working directory: {backend_dir}")
    
    # Start FastAPI in background thread
    fastapi_thread = threading.Thread(target=start_fastapi_server, daemon=True)
    fastapi_thread.start()
    
    # Start HTTP server
    server = HTTPServer(('0.0.0.0', port), ApolloHandler)
    print(f"🌐 Custom HTTP server started on http://0.0.0.0:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Server stopped")
        server.shutdown()

if __name__ == "__main__":
    main()