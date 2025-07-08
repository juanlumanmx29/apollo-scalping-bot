#!/usr/bin/env python3
"""
Apollo Scalping Bot - Railway Deployment Root Entry Point
This script ensures Railway uses our custom startup logic.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Railway deployment from root directory"""
    try:
        # Get port from environment variable (Railway sets this)
        port = os.environ.get("PORT")
        if not port:
            logger.error("❌ PORT environment variable not set by Railway")
            sys.exit(1)
            
        # Validate port is numeric
        try:
            port_int = int(port)
            logger.info(f"🚀 Apollo Scalping Bot starting on Railway port {port_int}")
        except ValueError:
            logger.error(f"❌ Invalid PORT value: {port}")
            sys.exit(1)
        
        # Set working directory to backend
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        os.chdir(backend_dir)
        logger.info(f"📁 Changed working directory to: {backend_dir}")
        
        # Import and run the FastAPI app
        sys.path.insert(0, backend_dir)
        from main import app
        import uvicorn
        
        logger.info("🔧 Starting uvicorn server with Railway configuration...")
        logger.info(f"📍 Host: 0.0.0.0, Port: {port_int}")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port_int,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Apollo Scalping Bot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()