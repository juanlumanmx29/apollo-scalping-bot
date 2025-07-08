#!/bin/bash
cd backend
echo "Starting Apollo Scalping Bot backend on port $PORT"
uvicorn main:app --host 0.0.0.0 --port "$PORT"