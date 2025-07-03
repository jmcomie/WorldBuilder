"""Direct Neo4j database operations service."""
from typing import Dict, List, Optional, Any
from neo4j import Driver
from app.core.database import get_driver


class Neo4jService:
    """Service for direct Neo4j database operations."""
    
    def __init__(self, driver: Driver):
        self.driver = driver
    
    def test_connection(self) -> str:
        """Test Neo4j connection."""
        with self.driver.session() as session:
            result = session.run("RETURN 'Connected to Neo4j' AS message")
            return result.single()["message"]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WITH count(n) as nodeCount, labels(n) as nodeLabels
                WITH nodeCount, collect(nodeLabels) as allLabels
                MATCH ()-[r]->()
                WITH nodeCount, allLabels, count(r) as edgeCount, type(r) as relType
                RETURN 
                    nodeCount,
                    collect(DISTINCT relType) as relationshipTypes,
                    edgeCount,
                    reduce(labels = [], labelList in allLabels | labels + labelList) as allNodeLabels
            """)
            
            record = result.single()
            if record:
                # Count occurrences of each label
                label_counts = {}
                for labels in record["allNodeLabels"]:
                    if isinstance(labels, list):
                        for label in labels:
                            label_counts[label] = label_counts.get(label, 0) + 1
                    elif labels:
                        label_counts[labels] = label_counts.get(labels, 0) + 1
                
                return {
                    "nodeCount": record["nodeCount"],
                    "edgeCount": record["edgeCount"],
                    "nodeTypes": list(label_counts.keys()),
                    "nodeTypeCounts": label_counts,
                    "edgeTypes": record["relationshipTypes"]
                }
            else:
                return {
                    "nodeCount": 0,
                    "edgeCount": 0,
                    "nodeTypes": [],
                    "nodeTypeCounts": {},
                    "edgeTypes": []
                }
    
    def get_nodes(
        self,
        node_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get nodes with filtering options."""
        with self.driver.session() as session:
            # Build the query with filters
            where_clauses = []
            params = {"limit": limit, "offset": offset}
            
            # Base query
            query = "MATCH (n"
            if node_type:
                query += f":{node_type}"
            query += ")"
            
            # Add search filter
            if search:
                where_clauses.append("n.name CONTAINS $search")
                params["search"] = search
            
            # Add WHERE clause if needed
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            # Add return clause
            query += """
                RETURN elementId(n) as id, n.name as name, 
                       labels(n) as labels, properties(n) as properties
                SKIP $offset LIMIT $limit
            """
            
            result = session.run(query, **params)
            
            nodes = []
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "label": record.get("name", "Unknown"),
                    "type": record["labels"][0] if record["labels"] else "Unknown",
                    **record.get("properties", {})
                })
            
            return nodes
    
    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get relationships with filtering options."""
        with self.driver.session() as session:
            # Build the query with filters
            params = {"limit": limit}
            
            # Base query
            query = "MATCH (n)-[r"
            if edge_type:
                query += f":{edge_type}"
            query += "]->(m)"
            
            # Add filters
            where_clauses = []
            if source_id:
                where_clauses.append("elementId(n) = $source_id")
                params["source_id"] = source_id
            if target_id:
                where_clauses.append("elementId(m) = $target_id")
                params["target_id"] = target_id
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += """
                RETURN elementId(r) as id, elementId(n) as source, 
                       elementId(m) as target, type(r) as type,
                       properties(r) as properties
                LIMIT $limit
            """
            
            result = session.run(query, **params)
            
            edges = []
            for record in result:
                edges.append({
                    "id": record["id"],
                    "source": record["source"],
                    "target": record["target"],
                    "label": record["type"],
                    **record.get("properties", {})
                })
            
            return edges
    
    def get_node_distances(self, center_node_uuid: str) -> List[Dict[str, Any]]:
        """Calculate distances from a center node."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (center {uuid: $center_uuid})
                MATCH path = shortestPath((center)-[*0..4]-(node))
                WHERE node.uuid IS NOT NULL AND node.uuid <> $center_uuid
                WITH node, length(path) as distance
                RETURN DISTINCT node.uuid as uuid, node.name as name, 
                       labels(node) as labels, distance
                ORDER BY distance, node.name
                LIMIT 100
            """, center_uuid=center_node_uuid)
            
            nodes = []
            for record in result:
                nodes.append({
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "labels": record["labels"],
                    "distance": record["distance"]
                })
            
            # Also get the center node
            center_result = session.run("""
                MATCH (center {uuid: $center_uuid})
                RETURN center.uuid as uuid, center.name as name, labels(center) as labels
            """, center_uuid=center_node_uuid)
            
            center_record = center_result.single()
            if center_record:
                nodes.insert(0, {
                    "uuid": center_record["uuid"],
                    "name": center_record["name"],
                    "labels": center_record["labels"],
                    "distance": 0
                })
            
            return nodes