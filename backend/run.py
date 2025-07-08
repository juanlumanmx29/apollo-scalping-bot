#!/usr/bin/env python3
"""
Apollo Scalping Bot - Railway Deployment Entry Point
This script ensures proper PORT environment variable handling for Railway deployment.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Railway deployment"""
    try:
        # Get port from environment variable
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"🚀 Apollo Scalping Bot starting on port {port}")
        logger.info(f"📍 Environment: {'Production' if os.environ.get('RAILWAY_ENVIRONMENT') else 'Development'}")
        
        # Import and run the FastAPI app
        from main import app
        import uvicorn
        
        logger.info("🔧 Starting uvicorn server...")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()