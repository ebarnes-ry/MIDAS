#!/bin/bash

# MIDAS V2 Development Startup Script
# This script starts both the backend API and frontend development servers

set -e

echo "Starting MIDAS V2 Development Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "run_server.py" ]; then
    echo "Error: Please run this script from the MIDAS_V2 directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please create one first:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Fix macOS SSL: point Python to certifi's CA bundle so marker-pdf can
# download font files without hitting certificate verification errors.
CERTIFI_PATH=$(.venv/bin/python -c "import certifi; print(certifi.where())")
export SSL_CERT_FILE="$CERTIFI_PATH"
export REQUESTS_CA_BUNDLE="$CERTIFI_PATH"

# Check if required services are available
echo "Checking service availability..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Warning: Ollama is not running on localhost:11434"
    echo "   Please start Ollama: ollama serve"
    echo "   Or install models: ollama pull qwen2.5vl:7b"
fi

# Check if required models are available
echo "Checking for required models..."
if command -v ollama &> /dev/null; then
    if ! ollama list | grep -q "qwen2.5vl:7b"; then
        echo "Warning: qwen2.5vl:7b model not found"
        echo "   Install with: ollama pull qwen2.5vl:7b"
    fi
    if ! ollama list | grep -q "phi4-mini-reasoning:latest"; then
        echo "Warning: phi4-mini-reasoning:latest model not found"
        echo "   Install with: ollama pull phi4-mini-reasoning:latest"
    fi
fi

# Start backend server
echo "Starting backend API server..."
echo "   Backend will be available at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo ""

# Start backend in background
python run_server.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/health/ > /dev/null 2>&1; then
    echo "Backend failed to start. Check the logs above."
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo "Backend started successfully"

# Start frontend
echo "Starting frontend development server..."
echo "   Frontend will be available at: http://localhost:3000"
echo ""

cd midas-frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend
npm start &
FRONTEND_PID=$!

cd ..

echo ""
echo "Development environment started!"
echo "=================================="
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
