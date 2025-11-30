"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings
from app.routers import remove_bg, models, stats
from app.models.manager import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    settings = get_settings()

    # Startup: Load default model if configured
    if settings.default_model:
        logger.info(f"Loading default model: {settings.default_model}")
        try:
            model_manager.load_model(settings.default_model)
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
    else:
        logger.warning("No default model configured. Set DEFAULT_MODEL in .env file.")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down OpenBackground API")
    model_manager.unload_all()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Self-hosted background removal microservice using Hugging Face models.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(remove_bg.router, prefix="/api/v1", tags=["Background Removal"])
    app.include_router(models.router, prefix="/api/v1", tags=["Models"])
    app.include_router(stats.router, prefix="/api/v1", tags=["Statistics"])

    # Health check endpoint (public)
    @app.get("/api/v1/health", tags=["Health"])
    async def health_check():
        """Health check endpoint - no authentication required."""
        return {
            "status": "healthy",
            "version": settings.api_version,
            "current_model": model_manager.current_model_name,
            "models_loaded": list(model_manager.loaded_models.keys()),
        }

    # Mount static files for frontend
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    # Serve frontend index.html at root
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the frontend dashboard."""
        return FileResponse("frontend/index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=True,
    )

