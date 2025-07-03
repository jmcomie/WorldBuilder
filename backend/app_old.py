from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j connection
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(uri, auth=(username, password))

# Initialize Graphiti
graphiti_client = None

async def get_graphiti():
    global graphiti_client
    if graphiti_client is None:
        graphiti_client = Graphiti(uri, username, password)
        await graphiti_client.build_indices_and_constraints()
    return graphiti_client

# Pydantic models
class EpisodeRequest(BaseModel):
    name: str
    content: str
    source_description: Optional[str] = "User input"
    
class SearchRequest(BaseModel):
    query: str
    num_results: Optional[int] = 10

@app.get("/")
async def root():
    return {"message": "Backend is running"}

@app.get("/test-neo4j")
async def test_neo4j():
    with driver.session() as session:
        result = session.run("RETURN 'Connected to Neo4j' AS message")
        return {"message": result.single()["message"]}

@app.post("/episodes")
async def add_episode(episode: EpisodeRequest):
    """Add a new episode to the knowledge graph"""
    try:
        client = await get_graphiti()
        result = await client.add_episode(
            name=episode.name,
            episode_body=episode.content,
            source=EpisodeType.text,
            source_description=episode.source_description,
            reference_time=datetime.now(timezone.utc)
        )
        return {
            "success": True,
            "message": "Episode added successfully",
            "episode_id": str(result.uuid) if hasattr(result, 'uuid') else None
        }
    except Exception as e:
        error_message = str(e)
        print(f"Error adding episode: {error_message}")
        
        # Check for rate limit errors
        if "rate_limit_exceeded" in error_message.lower() or "rate limit" in error_message.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        elif "api_key" in error_message.lower() or "authentication" in error_message.lower():
            raise HTTPException(
                status_code=401,
                detail="OpenAI API authentication failed. Please check your API key."
            )
        else:
            raise HTTPException(status_code=500, detail=error_message)

@app.post("/search")
async def search_graph(search: SearchRequest):
    """Search the knowledge graph using Graphiti"""
    try:
        client = await get_graphiti()
        results = await client.search(
            search.query,
            num_results=search.num_results
        )
        
        # Format results for frontend
        formatted_results = []
        for edge in results:
            formatted_results.append({
                "uuid": str(edge.uuid),
                "fact": edge.fact,
                "source_node": {
                    "uuid": str(edge.source_node_uuid),
                    "name": edge.source_node.name if hasattr(edge, 'source_node') else None
                },
                "target_node": {
                    "uuid": str(edge.target_node_uuid),
                    "name": edge.target_node.name if hasattr(edge, 'target_node') else None
                },
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None
            })
        
        return {
            "success": True,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
async def get_graph_data(
    limit: int = 100,
    offset: int = 0,
    node_type: Optional[str] = None
):
    """Get graph data in Cytoscape-compatible format"""
    try:
        with driver.session() as session:
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
            seen_nodes = set()
            
            # Process nodes
            for record in nodes:
                node = record["n"]
                node_id = node.get("uuid", node.get("name", str(node.id)))
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    cytoscape_nodes.append({
                        "data": {
                            "id": node_id,
                            "label": node.get("name", "Unknown"),
                            "type": list(node.labels)[0] if node.labels else "Unknown",
                            **dict(node)
                        }
                    })
            
            # Process edges with facts (excluding MENTIONS edges)
            for record in edges:
                source = record["source"]
                target = record["target"]
                rel = record["r"]
                
                # Skip MENTIONS edges
                if rel.type == "MENTIONS":
                    continue
                
                # Add source and target nodes if not seen
                source_id = source.get("uuid", source.get("name", str(source.id)))
                target_id = target.get("uuid", target.get("name", str(target.id)))
                
                if source_id not in seen_nodes:
                    seen_nodes.add(source_id)
                    cytoscape_nodes.append({
                        "data": {
                            "id": source_id,
                            "label": source.get("name", "Unknown"),
                            "type": list(source.labels)[0] if source.labels else "Unknown",
                            **dict(source)
                        }
                    })
                
                if target_id not in seen_nodes:
                    seen_nodes.add(target_id)
                    cytoscape_nodes.append({
                        "data": {
                            "id": target_id,
                            "label": target.get("name", "Unknown"),
                            "type": list(target.labels)[0] if target.labels else "Unknown",
                            **dict(target)
                        }
                    })
                
                # Create edge with fact if available
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
                
                cytoscape_edges.append({
                    "data": edge_data
                })
            
            return {
                "success": True,
                "elements": {
                    "nodes": cytoscape_nodes,
                    "edges": cytoscape_edges
                }
            }
    except Exception as e:
        print(f"Error in get_graph_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/with-facts")
async def get_graph_with_facts(
    limit: int = 100,
    offset: int = 0
):
    """Get graph data with Graphiti facts emphasized"""
    try:
        with driver.session() as session:
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
            seen_nodes = set()
            
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
            
            return {
                "success": True,
                "elements": {
                    "nodes": cytoscape_nodes,
                    "edges": cytoscape_edges
                },
                "stats": {
                    "nodeCount": len(cytoscape_nodes),
                    "edgeCount": len(cytoscape_edges),
                    "factsCount": sum(1 for edge in cytoscape_edges if edge["data"].get("fact"))
                }
            }
    except Exception as e:
        print(f"Error in get_graph_with_facts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/stats")
async def get_graph_stats():
    """Get graph statistics"""
    try:
        with driver.session() as session:
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
                    "success": True,
                    "nodeCount": record["nodeCount"],
                    "edgeCount": record["edgeCount"],
                    "nodeTypes": list(label_counts.keys()),
                    "nodeTypeCounts": label_counts,
                    "edgeTypes": record["relationshipTypes"]
                }
            else:
                return {
                    "success": True,
                    "nodeCount": 0,
                    "edgeCount": 0,
                    "nodeTypes": [],
                    "nodeTypeCounts": {},
                    "edgeTypes": []
                }
    except Exception as e:
        print(f"Error in get_graph_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/nodes")
async def get_graph_nodes(
    node_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    search: Optional[str] = None
):
    """Get nodes with filtering options"""
    try:
        with driver.session() as session:
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
                    "data": {
                        "id": record["id"],
                        "label": record.get("name", "Unknown"),
                        "type": record["labels"][0] if record["labels"] else "Unknown",
                        **record.get("properties", {})
                    }
                })
            
            return {
                "success": True,
                "nodes": nodes,
                "count": len(nodes)
            }
    except Exception as e:
        print(f"Error in get_graph_nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/edges")
async def get_graph_edges(
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
    edge_type: Optional[str] = None,
    limit: int = 100
):
    """Get relationships with filtering options"""
    try:
        with driver.session() as session:
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
                    "data": {
                        "id": record["id"],
                        "source": record["source"],
                        "target": record["target"],
                        "label": record["type"],
                        **record.get("properties", {})
                    }
                })
            
            return {
                "success": True,
                "edges": edges,
                "count": len(edges)
            }
    except Exception as e:
        print(f"Error in get_graph_edges: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/node-distances/{center_node_uuid}")
async def get_node_distances(center_node_uuid: str):
    """Calculate graph distances from a center node using Graphiti"""
    try:
        # Method 1: Use Graphiti search with center node for relevance
        client = await get_graphiti()
        
        # Search for related nodes from center
        search_results = await client.search(
            query="",  # Empty query to get all related nodes
            center_node_uuid=center_node_uuid,
            num_results=50
        )
        
        # Build a map of node distances from search results
        node_scores = {}
        for i, edge in enumerate(search_results):
            # Use ranking position as a proxy for distance/relevance
            score = 1.0 - (i / len(search_results))  # Higher score = closer/more relevant
            if edge.source_node_uuid not in node_scores:
                node_scores[edge.source_node_uuid] = score
            if edge.target_node_uuid not in node_scores:
                node_scores[edge.target_node_uuid] = score
        
        # Method 2: Direct Neo4j query for actual graph distances
        with driver.session() as session:
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
                node_data = {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "labels": record["labels"],
                    "distance": record["distance"],
                    "relevance_score": node_scores.get(record["uuid"], 0.0)
                }
                nodes.append(node_data)
        
            # Also include the center node
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
                    "distance": 0,
                    "relevance_score": 1.0
                })
        
        return {
            "success": True,
            "center_node_uuid": center_node_uuid,
            "nodes": nodes,
            "count": len(nodes)
        }
    except Exception as e:
        print(f"Error in get_node_distances: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize Graphiti on startup"""
    await get_graphiti()

@app.on_event("shutdown")
async def shutdown():
    driver.close()
    global graphiti_client
    if graphiti_client:
        await graphiti_client.close()