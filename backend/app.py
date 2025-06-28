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
            
            # Query to get nodes and relationships with pagination
            result = session.run(f"""
                MATCH (n{node_filter})
                WITH n SKIP $offset LIMIT $limit
                OPTIONAL MATCH (n)-[r]->(m)
                WITH n, collect({{
                    id: elementId(r),
                    source: elementId(n),
                    target: elementId(m),
                    type: type(r),
                    properties: properties(r)
                }}) as rels, collect(m) as targets
                RETURN 
                    collect(DISTINCT {{
                        id: elementId(n),
                        name: n.name,
                        labels: labels(n),
                        properties: properties(n)
                    }}) as nodes,
                    rels,
                    targets
            """, offset=offset, limit=limit)
            
            # Process results into Cytoscape format
            cytoscape_nodes = []
            cytoscape_edges = []
            seen_nodes = set()
            
            for record in result:
                # Process source nodes
                for node in record["nodes"]:
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
                
                # Process relationships and target nodes
                for i, rel_group in enumerate(record["rels"]):
                    if isinstance(rel_group, list):
                        for rel in rel_group:
                            if rel.get("id"):
                                cytoscape_edges.append({
                                    "data": {
                                        "id": rel["id"],
                                        "source": rel["source"],
                                        "target": rel["target"],
                                        "label": rel["type"],
                                        **rel.get("properties", {})
                                    }
                                })
                    
                # Process target nodes
                for target_group in record["targets"]:
                    if isinstance(target_group, list):
                        for target in target_group:
                            if target and target.get("id") not in seen_nodes:
                                seen_nodes.add(target["id"])
                                cytoscape_nodes.append({
                                    "data": {
                                        "id": target["id"],
                                        "label": target.get("name", "Unknown"),
                                        "type": target["labels"][0] if target.get("labels") else "Unknown",
                                        **target.get("properties", {})
                                    }
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