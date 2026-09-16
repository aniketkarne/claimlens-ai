"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api.routes import router as api_router


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    # Startup: Ensure directories exist
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.claims_log_path.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown: Clean up if needed
    pass


app = FastAPI(
    title="ClaimLens AI",
    description="Local Multimodal Insurance Claims Intake System - A POC demonstrating how small/local AI models can fit into an insurance company-level architecture for claims intake and triage.",
    version=settings.pipeline_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
# Note: allow_credentials must be False when allow_origins is "*".
# This API has no auth — leave credentials off so the wildcard origin works.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the main frontend page."""
        from fastapi.responses import FileResponse
        return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/")
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "ClaimLens AI - Local Multimodal Insurance Claims Intake System",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
