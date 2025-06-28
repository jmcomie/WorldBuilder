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
async def get_graph_data():
    """Get graph data for visualization using Neo4j directly"""
    try:
        with driver.session() as session:
            # Query to get nodes and relationships
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN 
                    collect(DISTINCT {
                        id: elementId(n), 
                        name: n.name, 
                        labels: labels(n),
                        properties: properties(n)
                    }) as nodes,
                    collect(DISTINCT {
                        id: elementId(r),
                        source: elementId(n), 
                        target: elementId(m), 
                        type: type(r),
                        properties: properties(r)
                    }) as relationships
            """)
            
            record = result.single()
            if record:
                nodes = record["nodes"]
                relationships = [r for r in record["relationships"] if r["id"] is not None]
                
                # Remove duplicates
                unique_nodes = {node["id"]: node for node in nodes}.values()
                
                return {
                    "success": True,
                    "nodes": list(unique_nodes),
                    "relationships": relationships
                }
            else:
                return {
                    "success": True,
                    "nodes": [],
                    "relationships": []
                }
    except Exception as e:
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