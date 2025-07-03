"""Health check endpoints."""
from fastapi import APIRouter, Depends
from neo4j import Driver
from app.core.database import get_driver
from app.services.neo4j_service import Neo4jService

router = APIRouter()


@router.get("/")
async def root():
    """Basic health check."""
    return {"message": "Backend is running"}


@router.get("/test-neo4j")
async def test_neo4j(driver: Driver = Depends(get_driver)):
    """Test Neo4j connection."""
    service = Neo4jService(driver)
    message = service.test_connection()
    return {"message": message}