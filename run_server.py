#!/usr/bin/env python3
"""
Development server launcher for MIDAS Vision API.

This script starts the FastAPI server with appropriate settings for development.
For production, you'd use a proper ASGI server deployment.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY and any other vars from .env

import certifi

# On macOS, Python venvs don't inherit the system keychain, so external HTTPS
# requests (e.g. marker-pdf downloading font files) fail with cert errors.
# Point SSL to certifi's up-to-date CA bundle before any other imports touch it.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import uvicorn
import sys
from pathlib import Path

# Add src to Python path so imports work correctly
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    print("Starting MIDAS Vision API Development Server")
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    print("\n" + "="*50 + "\n")
    
    # Start the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",  # Accept connections from any IP
        port=8000,
        reload=True,     # Auto-reload on code changes (development only)
        reload_dirs=[str(src_path)],  # Only watch src directory
        log_level="info"
    )