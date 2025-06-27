from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

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

@app.get("/")
async def root():
    return {"message": "Backend is running"}

@app.get("/test-neo4j")
async def test_neo4j():
    with driver.session() as session:
        result = session.run("RETURN 'Connected to Neo4j' AS message")
        return {"message": result.single()["message"]}

@app.on_event("shutdown")
async def shutdown():
    driver.close()