"""
FastAPI Application - Main server setup.

This module creates and configures the FastAPI application
with WebSocket support for real-time AI conversation.
"""

import yaml
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .websocket import websocket_router


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    - Startup: Load models, initialize connections
    - Shutdown: Cleanup resources
    """
    # Startup
    print("🚀 Starting Local AI Companion Server...")
    config = load_config()
    app.state.config = config
    app.state.character = config.get("character", {})
    
    print(f"   Character: {app.state.character.get('name', 'AI')}")
    print(f"   LLM: {config['llm']['ollama']['model']}")
    print("   Models will be loaded on first request (lazy loading)")
    print()
    
    yield
    
    # Shutdown
    print("\n👋 Shutting down server...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    config = load_config()
    character = config.get("character", {})
    
    app = FastAPI(
        title=f"{character.get('name', 'AI')} - Local AI Companion",
        description="A 100% local and private AI assistant with voice capabilities",
        version="0.2.0",
        lifespan=lifespan,
    )
    
    # CORS middleware - allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(router, prefix="/api")
    app.include_router(websocket_router)
    
    # Serve static files (frontend)
    frontend_path = Path(__file__).parent.parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    
    return app


# Create app instance for uvicorn
app = create_app()
