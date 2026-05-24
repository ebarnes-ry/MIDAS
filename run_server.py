#!/usr/bin/env python3
"""
Server launcher for MIDAS API.

Uses Railway's PORT in production and defaults to 8000 locally.
Reload is enabled only when MIDAS_RELOAD=true.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import certifi
import uvicorn

# Keep this before app/model imports so HTTPS downloads/API calls use certifi.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

project_root = Path(__file__).parent
src_path = project_root / "src"

# Keep project root importable so `src.api.main:app` resolves consistently.
sys.path.insert(0, str(project_root))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("MIDAS_RELOAD", "false").lower() == "true"

    print("Starting MIDAS API server")
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Host: 0.0.0.0")
    print(f"Port: {port}")
    print(f"Reload: {reload_enabled}")
    print(f"Docs: http://localhost:{port}/docs")
    print("\n" + "=" * 50 + "\n")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
        reload_dirs=[str(src_path)] if reload_enabled else None,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )