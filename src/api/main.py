from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os

from .routers import vision, health, reasoning, codegen, verification, trajectories
from src.config.profiles import resolve_config_path, validate_config_environment
from src.models.manager import ModelManager

# ── Rate limiter (shared across the app) ─────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── CORS origins ─────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:4321",               # Astro portfolio dev
    "https://emmabarnes.xyz",
    "https://www.emmabarnes.xyz",
    "https://midas-fawn-theta.vercel.app",
    "https://midas-frontend.vercel.app",
]

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("1. Starting MIDAS API server...")
    config_path = resolve_config_path()
    validate_config_environment(config_path)
    print(f"1a. Using model config: {config_path}")
    model_manager = ModelManager(config_path=config_path)
    app_state["model_manager"] = model_manager
    print("2. ModelManager initialized successfully")

    # Pre-warm Marker so OCR models load at startup, not on the first user request.
    # Without this, the first upload times out while PyTorch/surya downloads weights.
    print("3. Pre-warming Marker OCR models...")
    try:
        import io, tempfile, os
        from PIL import Image as PILImage
        warmup_img = PILImage.new("RGB", (64, 64), color=(255, 255, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            warmup_img.save(f, format="PNG")
            warmup_path = f.name
        try:
            model_manager.marker.convert_document(warmup_path)
        except Exception:
            pass  # warmup may fail on tiny image — models are still loaded into memory
        finally:
            os.unlink(warmup_path)
        print("3. Marker models ready")
    except Exception as e:
        print(f"3. Marker warmup skipped: {e}")

    print("4. API server ready to accept requests")
    yield
    print("5. Shutting down MIDAS API server...")
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

    @app.get("/")
    async def root():
        return {
            "name": "MIDAS API",
            "version": "2.1.0",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "vision": "/api/v1/vision",
                "reasoning": "/api/v1/reasoning",
                "codegen": "/api/v1/codegen",
                "verification": "/api/v1/verification",
                "trajectories": "/api/v1/trajectories",
            },
        }

    @app.get("/health")
    async def health_probe():
        return {"status": "ok", "version": "2.1.0"}

    return app


app = create_app()
