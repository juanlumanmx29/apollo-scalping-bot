#!/bin/bash
set -e

echo "🚀 Apollo Scalping Bot - Starting backend"
echo "📍 PORT: $PORT"
echo "📁 Current directory: $(pwd)"

# List files to debug
echo "📋 Files in root:"
ls -la

# Change to backend directory
cd backend
echo "📁 Changed to backend directory: $(pwd)"
echo "📋 Files in backend:"
ls -la

# Start the FastAPI server
echo "🔧 Starting uvicorn server on port $PORT..."
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"