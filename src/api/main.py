from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os

from .routers import vision, health, reasoning, codegen, verification, trajectories
from src.models.manager import ModelManager

# ── Rate limiter (shared across the app) ─────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── CORS origins ─────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:4321",               # Astro portfolio dev
    "https://emmabarnes.xyz",              # portfolio — update if domain changes
    "https://www.emmabarnes.xyz",
    "https://midas-frontend.vercel.app",   # Vercel deploy — update after deploy
]

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("1. Starting MIDAS API server...")
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    model_manager = ModelManager(config_path=config_path)
    app_state["model_manager"] = model_manager
    print("2. ModelManager initialized successfully")
    print("3. API server ready to accept requests")
    yield
    print("4. Shutting down MIDAS API server...")
    app_state.clear()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MIDAS API",
        description="Mathematical Intelligence with Deductive, Algebraic Synthesis",
        version="2.1.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(vision.router, prefix="/api/v1/vision", tags=["vision"])
    app.include_router(reasoning.router, prefix="/api/v1/reasoning", tags=["reasoning"])
    app.include_router(codegen.router, prefix="/api/v1/codegen", tags=["codegen"])
    app.include_router(verification.router, prefix="/api/v1/verification", tags=["verification"])
    app.include_router(trajectories.router, prefix="/api/v1/trajectories", tags=["trajectories"])

    return app


app = create_app()


@app.get("/")
async def root():
    return {"name": "MIDAS API", "version": "2.1.0", "status": "operational"}


# Railway / container health probe (no auth, no dependencies)
@app.get("/health")
async def health_probe():
    return {"status": "ok", "version": "2.1.0"}
