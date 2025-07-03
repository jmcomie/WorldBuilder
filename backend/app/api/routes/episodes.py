"""Episode management endpoints."""
from fastapi import APIRouter, HTTPException
from app.models.episode import EpisodeRequest, EpisodeResponse
from app.services.graphiti_service import graphiti_service
from app.core.exceptions import handle_graphiti_error

router = APIRouter()


@router.post("/episodes", response_model=EpisodeResponse)
async def add_episode(episode: EpisodeRequest):
    """Add a new episode to the knowledge graph."""
    try:
        result = await graphiti_service.add_episode(
            name=episode.name,
            content=episode.content,
            source_description=episode.source_description
        )
        return EpisodeResponse(
            success=True,
            message="Episode added successfully",
            episode_id=str(result.uuid) if hasattr(result, 'uuid') else None
        )
    except Exception as e:
        print(f"Error adding episode: {str(e)}")
        raise handle_graphiti_error(e)