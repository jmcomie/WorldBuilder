# App Directory Structure

This directory contains the main application code organized in a modular structure.

## Structure Overview

```
app/
├── main.py           # FastAPI application initialization
├── config.py         # Configuration management
├── core/            # Core functionality
├── models/          # Pydantic models
├── api/             # API layer
├── services/        # Business logic
└── utils/           # Utility functions
```

## Modules

### main.py
- FastAPI application initialization
- CORS configuration
- Lifecycle management (startup/shutdown)
- Router registration

### config.py
- Environment variable management
- Application settings
- Configuration defaults

### core/
Core functionality including:
- `database.py`: Neo4j driver management
- `exceptions.py`: Custom exception handling

### models/
Pydantic models for:
- `episode.py`: Episode request/response models
- `search.py`: Search request/response models
- `graph.py`: Graph data models (Cytoscape format)

### api/routes/
API endpoints organized by domain:
- `health.py`: Health check endpoints
- `episodes.py`: Episode management
- `search.py`: Knowledge graph search
- `graph.py`: Graph visualization and queries

### services/
Business logic services:
- `graphiti_service.py`: Graphiti integration
- `neo4j_service.py`: Direct Neo4j operations
- `graph_service.py`: Graph data processing

### utils/
Utility functions:
- `graph_utils.py`: Graph data transformation helpers

## Running the Application

Start the application with:
```bash
uvicorn app.main:app --reload
```

Or with the full path:
```bash
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```