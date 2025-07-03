"""Search-related Pydantic models."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class SearchRequest(BaseModel):
    """Request model for searching the knowledge graph."""
    query: str
    num_results: Optional[int] = 10


class SearchNodeInfo(BaseModel):
    """Node information in search results."""
    uuid: str
    name: Optional[str] = None


class SearchResult(BaseModel):
    """Individual search result."""
    uuid: str
    fact: str
    source_node: SearchNodeInfo
    target_node: SearchNodeInfo
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search operations."""
    success: bool
    results: List[SearchResult]
    count: int