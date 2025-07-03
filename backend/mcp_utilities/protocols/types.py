"""MCP protocol type definitions."""
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum


class MCPErrorCode(Enum):
    """Standard MCP error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class MCPError:
    """MCP error structure."""
    code: int
    message: str
    data: Optional[Any] = None


@dataclass
class MCPRequest:
    """MCP request structure."""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict[str, Any] = None
    id: Optional[Union[str, int]] = None


@dataclass
class MCPResponse:
    """MCP response structure."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[MCPError] = None
    id: Optional[Union[str, int]] = None


@dataclass
class GraphNode:
    """Graph node representation."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    """Graph edge representation."""
    id: str
    source_id: str
    target_id: str
    type: str
    fact: Optional[str] = None
    properties: Dict[str, Any] = None


@dataclass
class GraphQuery:
    """Graph query parameters."""
    query: str
    limit: int = 100
    offset: int = 0
    node_type: Optional[str] = None
    include_facts: bool = True