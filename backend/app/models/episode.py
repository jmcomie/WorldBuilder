"""Episode-related Pydantic models."""
from pydantic import BaseModel
from typing import Optional


class EpisodeRequest(BaseModel):
    """Request model for creating a new episode."""
    name: str
    content: str
    source_description: Optional[str] = "User input"


class EpisodeResponse(BaseModel):
    """Response model for episode operations."""
    success: bool
    message: str
    episode_id: Optional[str] = None