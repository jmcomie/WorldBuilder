"""Database connection and management."""
from neo4j import GraphDatabase, Driver
from typing import Optional
from app.config import get_settings


class Neo4jDriver:
    """Singleton Neo4j driver manager."""
    
    _driver: Optional[Driver] = None
    
    @classmethod
    def get_driver(cls) -> Driver:
        """Get or create Neo4j driver instance."""
        if cls._driver is None:
            settings = get_settings()
            cls._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
        return cls._driver
    
    @classmethod
    def close(cls):
        """Close the driver connection."""
        if cls._driver:
            cls._driver.close()
            cls._driver = None


def get_driver() -> Driver:
    """Dependency to get Neo4j driver."""
    return Neo4jDriver.get_driver()