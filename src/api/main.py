"""
FastAPI application entry point.

This is the main FastAPI application that coordinates all API routes and middleware.
It serves as the bridge between HTTP requests and your existing pipeline logic.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from .routers import vision, health, reasoning, codegen, verification, trajectories
from src.models.manager import ModelManager

# Global application state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    This handles startup and shutdown events for the FastAPI app.
    We initialize expensive resources (like models) once at startup
    and clean them up at shutdown.
    """
    # Startup: Initialize ModelManager and other expensive resources
    print("1. Starting MIDAS API server...")
    
    # Initialize your existing ModelManager with config path
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    model_manager = ModelManager(config_path=config_path)
    app_state["model_manager"] = model_manager
    
    print("2. ModelManager initialized successfully")
    print("3. API server ready to accept requests")
    
    yield  # Server runs here
    
    # Shutdown: Clean up resources
    print("4. Shutting down MIDAS API server...")
    # Add any cleanup logic here if needed
    app_state.clear()

def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    
    This approach allows for easy testing and configuration management.
    """
    
    app = FastAPI(
        title="MIDAS Vision API",
        description="API for mathematical document analysis and visual understanding",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configure CORS middleware for frontend communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Common frontend ports
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(vision.router, prefix="/api/v1/vision", tags=["vision"])
    app.include_router(reasoning.router, prefix="/api/v1/reasoning", tags=["reasoning"])
    app.include_router(codegen.router, prefix="/api/v1/codegen", tags=["codegen"])
    app.include_router(verification.router, prefix="/api/v1/verification", tags=["verification"])
    app.include_router(trajectories.router, prefix="/api/v1/trajectories", tags=["trajectories"])
    
    return app

# Create the FastAPI app instance
app = create_app()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": "MIDAS Vision API", 
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "vision": "/api/v1/vision",
            "reasoning": "/api/v1/reasoning",
            "codegen": "/api/v1/codegen",
            "verification": "/api/v1/verification",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }