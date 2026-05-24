from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from src.api.limiting import limiter
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os

from .routers import vision, health, reasoning, codegen, verification, trajectories, demo
from src.config.profiles import resolve_config_path, validate_config_environment
from src.models.manager import ModelManager

# ── Rate limiter (shared across the app) ─────────────────────────────────────


# ── CORS origins ─────────────────────────────────────────────────────────────
def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or [
        "http://localhost:3000",
        "http://localhost:4321",
    ]


ALLOWED_ORIGINS = _parse_allowed_origins()

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

    if os.getenv("MIDAS_PREWARM_MARKER", "false").lower() == "true":
        # Optional because hosted demo users mostly use cached examples. Loading
        # Marker/Surya/PyTorch at startup consumes memory even if no fresh upload
        # is processed.
        print("3. Pre-warming Marker OCR models...")
        try:
            import tempfile
            from PIL import Image as PILImage
            warmup_img = PILImage.new("RGB", (64, 64), color=(255, 255, 255))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                warmup_img.save(f, format="PNG")
                warmup_path = f.name
            try:
                model_manager.marker.convert_document(warmup_path)
            except Exception:
                pass
            finally:
                os.unlink(warmup_path)
            print("3. Marker models ready")
        except Exception as e:
            print(f"3. Marker warmup skipped: {e}")
    else:
        print("3. Marker prewarm disabled")

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
    app.include_router(demo.router, prefix="/api/v1/demo", tags=["demo"])

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
