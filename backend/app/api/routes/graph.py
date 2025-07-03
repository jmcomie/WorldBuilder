"""Graph visualization and manipulation endpoints."""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from neo4j import Driver
from app.core.database import get_driver
from app.models.graph import (
    GraphResponse, GraphElements, GraphStatsResponse,
    NodeDistancesResponse, NodeDistanceInfo, CytoscapeNode, CytoscapeEdge
)
from app.services.neo4j_service import Neo4jService
from app.services.graph_service import GraphService
from app.services.graphiti_service import graphiti_service
from app.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/graph", response_model=GraphResponse)
async def get_graph_data(
    limit: int = Query(default=settings.default_limit, le=settings.max_limit),
    offset: int = Query(default=0, ge=0),
    node_type: Optional[str] = None,
    driver: Driver = Depends(get_driver)
):
    """Get graph data in Cytoscape-compatible format."""
    try:
        service = GraphService(driver)
        nodes, edges = service.get_graph_data(limit, offset, node_type)
        
        return GraphResponse(
            success=True,
            elements=GraphElements(
                nodes=[CytoscapeNode(data=node["data"]) for node in nodes],
                edges=[CytoscapeEdge(data=edge["data"]) for edge in edges]
            )
        )
    except Exception as e:
        print(f"Error in get_graph_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/with-facts")
async def get_graph_with_facts(
    limit: int = Query(default=settings.default_limit, le=settings.max_limit),
    offset: int = Query(default=0, ge=0),
    driver: Driver = Depends(get_driver)
):
    """Get graph data with Graphiti facts emphasized."""
    try:
        service = GraphService(driver)
        nodes, edges, stats = service.get_graph_with_facts(limit, offset)
        
        return {
            "success": True,
            "elements": {
                "nodes": nodes,
                "edges": edges
            },
            "stats": stats
        }
    except Exception as e:
        print(f"Error in get_graph_with_facts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats(driver: Driver = Depends(get_driver)):
    """Get graph statistics."""
    try:
        service = Neo4jService(driver)
        stats = service.get_graph_stats()
        
        return GraphStatsResponse(
            success=True,
            **stats
        )
    except Exception as e:
        print(f"Error in get_graph_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/nodes")
async def get_graph_nodes(
    node_type: Optional[str] = None,
    limit: int = Query(default=settings.default_limit, le=settings.max_limit),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = None,
    driver: Driver = Depends(get_driver)
):
    """Get nodes with filtering options."""
    try:
        service = Neo4jService(driver)
        nodes = service.get_nodes(node_type, limit, offset, search)
        
        return {
            "success": True,
            "nodes": [{"data": node} for node in nodes],
            "count": len(nodes)
        }
    except Exception as e:
        print(f"Error in get_graph_nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/edges")
async def get_graph_edges(
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
    edge_type: Optional[str] = None,
    limit: int = Query(default=settings.default_limit, le=settings.max_limit),
    driver: Driver = Depends(get_driver)
):
    """Get relationships with filtering options."""
    try:
        service = Neo4jService(driver)
        edges = service.get_edges(source_id, target_id, edge_type, limit)
        
        return {
            "success": True,
            "edges": [{"data": edge} for edge in edges],
            "count": len(edges)
        }
    except Exception as e:
        print(f"Error in get_graph_edges: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/node-distances/{center_node_uuid}", response_model=NodeDistancesResponse)
async def get_node_distances(
    center_node_uuid: str,
    driver: Driver = Depends(get_driver)
):
    """Calculate graph distances from a center node using Graphiti."""
    try:
        # Method 1: Use Graphiti search with center node for relevance
        search_results = await graphiti_service.search(
            query="",  # Empty query to get all related nodes
            center_node_uuid=center_node_uuid,
            num_results=50
        )
        
        # Build a map of node scores from search results
        node_scores = {}
        for i, edge in enumerate(search_results):
            # Use ranking position as a proxy for distance/relevance
            score = 1.0 - (i / len(search_results))  # Higher score = closer/more relevant
            if edge.source_node_uuid not in node_scores:
                node_scores[edge.source_node_uuid] = score
            if edge.target_node_uuid not in node_scores:
                node_scores[edge.target_node_uuid] = score
        
        # Method 2: Get actual graph distances from Neo4j
        service = Neo4jService(driver)
        nodes = service.get_node_distances(center_node_uuid)
        
        # Combine with relevance scores
        node_infos = []
        for node in nodes:
            node_infos.append(NodeDistanceInfo(
                uuid=node["uuid"],
                name=node["name"],
                labels=node["labels"],
                distance=node["distance"],
                relevance_score=node_scores.get(node["uuid"], 0.0)
            ))
        
        return NodeDistancesResponse(
            success=True,
            center_node_uuid=center_node_uuid,
            nodes=node_infos,
            count=len(node_infos)
        )
    except Exception as e:
        print(f"Error in get_node_distances: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))