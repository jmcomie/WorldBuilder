"""Graph-related Pydantic models."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class CytoscapeNode(BaseModel):
    """Cytoscape node format."""
    data: Dict[str, Any]


class CytoscapeEdge(BaseModel):
    """Cytoscape edge format."""
    data: Dict[str, Any]


class GraphElements(BaseModel):
    """Graph elements containing nodes and edges."""
    nodes: List[CytoscapeNode]
    edges: List[CytoscapeEdge]


class GraphResponse(BaseModel):
    """Response model for graph data."""
    success: bool
    elements: GraphElements


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""
    success: bool
    nodeCount: int
    edgeCount: int
    nodeTypes: List[str]
    nodeTypeCounts: Dict[str, int]
    edgeTypes: List[str]


class NodeDistanceInfo(BaseModel):
    """Node distance information."""
    uuid: str
    name: str
    labels: List[str]
    distance: int
    relevance_score: float


class NodeDistancesResponse(BaseModel):
    """Response model for node distances."""
    success: bool
    center_node_uuid: str
    nodes: List[NodeDistanceInfo]
    count: int