"""Configuration management for the Worldbuilder backend."""
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""
    
    # Neo4j configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # OpenAI configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # CORS configuration
    cors_origins: list = ["http://localhost:3000"]
    
    # API configuration
    api_title: str = "Worldbuilder API"
    api_version: str = "1.0.0"
    
    # Pagination defaults
    default_limit: int = 100
    max_limit: int = 1000
    
    # Search defaults
    default_search_results: int = 10
    max_search_results: int = 100


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()