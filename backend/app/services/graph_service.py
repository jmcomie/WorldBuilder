"""Graph processing and transformation service."""
from typing import Dict, List, Optional, Any, Set, Tuple
from neo4j import Driver
from app.core.database import get_driver
from app.utils.graph_utils import format_cytoscape_node, format_cytoscape_edge


class GraphService:
    """Service for graph data processing and transformation."""
    
    def __init__(self, driver: Driver):
        self.driver = driver
    
    def get_graph_data(
        self,
        limit: int = 100,
        offset: int = 0,
        node_type: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """Get graph data in Cytoscape-compatible format."""
        with self.driver.session() as session:
            # Build the query with optional node type filter
            node_filter = ""
            if node_type:
                node_filter = f":{node_type}"
            
            # First get nodes
            nodes_result = session.run(f"""
                MATCH (n{node_filter})
                WHERE n.uuid IS NOT NULL
                RETURN DISTINCT n
                SKIP $offset LIMIT $limit
            """, offset=offset, limit=limit)
            
            nodes = list(nodes_result)
            node_ids = [record["n"]["uuid"] for record in nodes if "uuid" in record["n"]]
            
            # Then get edges with facts
            edges_result = session.run("""
                MATCH (source)-[r]->(target)
                WHERE source.uuid IN $node_ids 
                   OR target.uuid IN $node_ids
                RETURN source, r, target
                LIMIT 200
            """, node_ids=node_ids)
            
            edges = list(edges_result)
            
            # Process results into Cytoscape format
            cytoscape_nodes = []
            cytoscape_edges = []
            seen_nodes: Set[str] = set()
            
            # Process nodes
            for record in nodes:
                node = record["n"]
                node_data = format_cytoscape_node(node)
                if node_data and node_data["data"]["id"] not in seen_nodes:
                    seen_nodes.add(node_data["data"]["id"])
                    cytoscape_nodes.append(node_data)
            
            # Process edges with facts (excluding MENTIONS edges)
            for record in edges:
                source = record["source"]
                target = record["target"]
                rel = record["r"]
                
                # Skip MENTIONS edges
                if rel.type == "MENTIONS":
                    continue
                
                # Add source and target nodes if not seen
                source_node = format_cytoscape_node(source)
                target_node = format_cytoscape_node(target)
                
                if source_node and source_node["data"]["id"] not in seen_nodes:
                    seen_nodes.add(source_node["data"]["id"])
                    cytoscape_nodes.append(source_node)
                
                if target_node and target_node["data"]["id"] not in seen_nodes:
                    seen_nodes.add(target_node["data"]["id"])
                    cytoscape_nodes.append(target_node)
                
                # Create edge
                edge_data = format_cytoscape_edge(source, target, rel)
                if edge_data:
                    cytoscape_edges.append(edge_data)
            
            return cytoscape_nodes, cytoscape_edges
    
    def get_graph_with_facts(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """Get graph data with Graphiti facts emphasized."""
        with self.driver.session() as session:
            # Query specifically for nodes and fact relationships
            result = session.run("""
                // First get nodes that have UUIDs (Graphiti nodes)
                MATCH (n)
                WHERE n.uuid IS NOT NULL
                WITH n
                SKIP $offset LIMIT $limit
                
                // Get all relationships involving these nodes
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE m.uuid IS NOT NULL
                
                // Return nodes and relationships
                RETURN 
                    collect(DISTINCT {
                        id: COALESCE(n.uuid, toString(id(n))),
                        name: n.name,
                        labels: labels(n),
                        properties: properties(n)
                    }) as nodes,
                    collect(DISTINCT {
                        id: COALESCE(r.uuid, toString(id(r))),
                        source: COALESCE(startNode(r).uuid, toString(id(startNode(r)))),
                        target: COALESCE(endNode(r).uuid, toString(id(endNode(r)))),
                        type: type(r),
                        fact: r.fact,
                        properties: properties(r)
                    }) as relationships,
                    collect(DISTINCT {
                        id: COALESCE(m.uuid, toString(id(m))),
                        name: m.name,
                        labels: labels(m),
                        properties: properties(m)
                    }) as connected_nodes
            """, offset=offset, limit=limit)
            
            # Process results
            cytoscape_nodes = []
            cytoscape_edges = []
            seen_nodes: Set[str] = set()
            
            for record in result:
                # Add all nodes
                all_nodes = record["nodes"] + record["connected_nodes"]
                for node in all_nodes:
                    if node["id"] not in seen_nodes:
                        seen_nodes.add(node["id"])
                        cytoscape_nodes.append({
                            "data": {
                                "id": node["id"],
                                "label": node.get("name", "Unknown"),
                                "type": node["labels"][0] if node["labels"] else "Unknown",
                                **node.get("properties", {})
                            }
                        })
                
                # Add relationships with facts (excluding MENTIONS edges)
                for rel in record["relationships"]:
                    if rel["source"] and rel["target"] and rel["type"] != "MENTIONS":
                        edge_data = {
                            "id": rel["id"],
                            "source": rel["source"],
                            "target": rel["target"],
                            "type": rel["type"],
                            "label": rel["type"]
                        }
                        
                        # Prioritize fact property
                        if rel.get("fact"):
                            edge_data["fact"] = rel["fact"]
                        
                        # Add other properties
                        if rel.get("properties"):
                            edge_data.update(rel["properties"])
                        
                        cytoscape_edges.append({"data": edge_data})
            
            # Calculate statistics
            stats = {
                "nodeCount": len(cytoscape_nodes),
                "edgeCount": len(cytoscape_edges),
                "factsCount": sum(1 for edge in cytoscape_edges if edge["data"].get("fact"))
            }
            
            return cytoscape_nodes, cytoscape_edges, stats