"""Health check endpoints."""
from fastapi import APIRouter, Depends
from neo4j import Driver
from app.core.database import get_driver
from app.services.neo4j_service import Neo4jService

router = APIRouter()


@router.get("/health")
async def root():
    """Basic health check."""
    return {"message": "Backend is running"}

