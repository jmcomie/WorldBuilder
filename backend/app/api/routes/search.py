"""Search endpoints."""
from fastapi import APIRouter, HTTPException
from app.models.search import SearchRequest, SearchResponse, SearchResult, SearchNodeInfo
from app.services.graphiti_service import graphiti_service

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_graph(search: SearchRequest):
    """Search the knowledge graph using Graphiti."""
    try:
        results = await graphiti_service.search(
            query=search.query,
            num_results=search.num_results
        )
        
        # Format results for frontend
        formatted_results = []
        for edge in results:
            formatted_results.append(SearchResult(
                uuid=str(edge.uuid),
                fact=edge.fact,
                source_node=SearchNodeInfo(
                    uuid=str(edge.source_node_uuid),
                    name=edge.source_node.name if hasattr(edge, 'source_node') else None
                ),
                target_node=SearchNodeInfo(
                    uuid=str(edge.target_node_uuid),
                    name=edge.target_node.name if hasattr(edge, 'target_node') else None
                ),
                valid_at=edge.valid_at.isoformat() if edge.valid_at else None,
                invalid_at=edge.invalid_at.isoformat() if edge.invalid_at else None
            ))
        
        return SearchResponse(
            success=True,
            results=formatted_results,
            count=len(formatted_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))