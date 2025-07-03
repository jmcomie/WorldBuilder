"""Utility functions for graph data transformation."""
from typing import Dict, Any, Optional


def format_cytoscape_node(node: Any) -> Optional[Dict[str, Any]]:
    """Format a Neo4j node for Cytoscape visualization."""
    if not node:
        return None
    
    node_id = node.get("uuid", node.get("name", str(node.id)))
    return {
        "data": {
            "id": node_id,
            "label": node.get("name", "Unknown"),
            "type": list(node.labels)[0] if node.labels else "Unknown",
            **dict(node)
        }
    }


def format_cytoscape_edge(source: Any, target: Any, rel: Any) -> Optional[Dict[str, Any]]:
    """Format a Neo4j relationship for Cytoscape visualization."""
    if not source or not target or not rel:
        return None
    
    source_id = source.get("uuid", source.get("name", str(source.id)))
    target_id = target.get("uuid", target.get("name", str(target.id)))
    
    edge_data = {
        "id": f"{source_id}-{target_id}-{rel.type}",
        "source": source_id,
        "target": target_id,
        "type": rel.type,
        "label": rel.type
    }
    
    # Add fact if it exists
    if hasattr(rel, "fact") and rel.get("fact"):
        edge_data["fact"] = rel.get("fact")
    elif "fact" in dict(rel):
        edge_data["fact"] = dict(rel)["fact"]
    
    # Add other properties
    edge_data.update(dict(rel))
    
    return {"data": edge_data}