"""Graphiti integration service."""
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone
from typing import Optional, List
from app.config import get_settings
from app.core.exceptions import GraphitiException


class GraphitiService:
    """Service for managing Graphiti operations."""
    
    _instance: Optional[Graphiti] = None
    
    @classmethod
    async def get_client(cls) -> Graphiti:
        """Get or create Graphiti client instance."""
        if cls._instance is None:
            settings = get_settings()
            cls._instance = Graphiti(
                settings.neo4j_uri,
                settings.neo4j_username,
                settings.neo4j_password
            )
            await cls._instance.build_indices_and_constraints()
        return cls._instance
    
    @classmethod
    async def close(cls):
        """Close Graphiti client."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None
    
    async def add_episode(
        self,
        name: str,
        content: str,
        source_description: str = "User input"
    ):
        """Add a new episode to the knowledge graph."""
        client = await self.get_client()
        try:
            result = await client.add_episode(
                name=name,
                episode_body=content,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=datetime.now(timezone.utc)
            )
            return result
        except Exception as e:
            raise GraphitiException(f"Failed to add episode: {str(e)}")
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        center_node_uuid: Optional[str] = None
    ):
        """Search the knowledge graph."""
        client = await self.get_client()
        try:
            if center_node_uuid:
                results = await client.search(
                    query=query,
                    center_node_uuid=center_node_uuid,
                    num_results=num_results
                )
            else:
                results = await client.search(
                    query,
                    num_results=num_results
                )
            return results
        except Exception as e:
            raise GraphitiException(f"Search failed: {str(e)}")


# Create a singleton instance
graphiti_service = GraphitiService()