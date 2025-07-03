"""Main FastAPI application initialization."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings
from app.core.database import Neo4jDriver
from app.services.graphiti_service import graphiti_service
from app.api.routes import health, episodes, search, graph

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await graphiti_service.get_client()
    yield
    # Shutdown
    Neo4jDriver.close()
    await graphiti_service.close()


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(episodes.router, tags=["episodes"])
app.include_router(search.router, tags=["search"])
app.include_router(graph.router, tags=["graph"])