# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Worldbuilder is a full-stack application using:
- **Backend**: Python/FastAPI with Neo4j graph database
- **Frontend**: React with TypeScript and Vite
- **Infrastructure**: Docker Compose for local development

## Development Commands

### Quick Start
```bash
# Start all services (Neo4j, backend, frontend)
docker-compose up

# Start in detached mode
docker-compose up -d
```

### Backend Development
```bash
cd backend

# Install dependencies (using uv package manager)
uv sync

# Run backend server with hot-reload
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run backend inside Docker (from root directory)
docker-compose up backend
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Lint code
npm run lint

# Preview production build
npm run preview
```

### Database Access
- Neo4j Browser: http://localhost:7475
- Default credentials: neo4j/WorldBuilder2025! (configured in docker-compose.yml)

## Architecture Overview

### Backend Structure
- `backend/app/`: Main application package (modular architecture)
  - `main.py`: FastAPI application initialization
  - `config.py`: Configuration and environment management
  - `core/`: Core functionality (database, exceptions)
  - `models/`: Pydantic models for API contracts
  - `api/routes/`: API endpoints organized by domain
  - `services/`: Business logic and integrations
  - `utils/`: Utility functions
- `backend/mcp/`: MCP (Model Context Protocol) implementation
  - `servers/`: MCP server implementations
  - `client/`: Reusable MCP client library
  - `protocols/`: Protocol type definitions
- `backend/scratch/`: Testing and experimentation scripts
- Knowledge graph endpoints:
  - `POST /episodes`: Add new episodes to the knowledge graph
  - `POST /search`: Search the knowledge graph using Graphiti
  - `GET /graph`: Retrieve graph data for visualization
  - `GET /graph/stats`: Get graph statistics
  - `GET /graph/nodes`: Get filtered nodes
  - `GET /graph/edges`: Get filtered edges
  - `GET /graph/with-facts`: Get graph with Graphiti facts
  - `GET /graph/node-distances/{uuid}`: Calculate distances from a node

### Frontend Structure
- `frontend/src/App.tsx`: Main React component with routing
- `frontend/src/api.ts`: API client for backend communication
- Uses Vite for fast development and building
- Main views: Home, Write (with Episode/Ideation/Ontology modes), Graph, Play
- Overlays: Settings (General, Appearance, API Keys, MCP Servers, Advanced), Help

### Service Communication
- Frontend (port 3000, internally 5173) → Backend (port 8000) → Neo4j (port 7688)
- All services run in Docker containers with networking configured

### API Endpoints

#### Graphiti Integration
- `POST /episodes`: Add a new episode to the knowledge graph
  - Request body: `{ name: string, content: string, source_description?: string }`
- `POST /search`: Search the knowledge graph
  - Request body: `{ query: string, num_results?: number }`
- `GET /graph`: Get graph data in Cytoscape-compatible format
  - Query params: `limit`, `offset`, `node_type`
- `GET /graph/stats`: Get graph statistics (node/edge counts, types)
- `GET /graph/nodes`: Get filtered nodes
  - Query params: `node_type`, `limit`, `offset`, `search`
- `GET /graph/edges`: Get filtered edges
  - Query params: `source_id`, `target_id`, `edge_type`, `limit`
- `GET /graph/with-facts`: Get graph data with Graphiti fact details
  - Query params: `limit`, `offset`
- `GET /graph/node-distances/{center_node_uuid}`: Calculate distances from a center node
  - Path param: `center_node_uuid` - The UUID of the center node

### Graph Visualization
The Graph view uses Cytoscape.js for interactive graph visualization:
- Multiple layout algorithms (force-directed, circle, grid, etc.)
- Interactive controls (zoom, pan, fit, export)
- Node and edge styling based on types
- Real-time graph statistics
- Export graph as PNG image

## Environment Configuration

### Backend
Create a `.env` file in the `backend/` directory (use `.env-template` as reference):
```
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=WorldBuilder2025!
OPENAI_API_KEY=your_openai_api_key_here
```

Note: Inside Docker, Neo4j is accessible at `neo4j:7687`, outside Docker at `localhost:7688`
The OPENAI_API_KEY is required for Graphiti's LLM and embedding capabilities

## Key Development Considerations

1. **Hot Reloading**: Both frontend and backend support hot-reloading. Backend uses `--reload` flag with uvicorn, frontend uses Vite's built-in HMR.
   - **IMPORTANT**: The frontend dev server runs continuously at http://localhost:3000 with automatic hot-reload
   - **DO NOT** manually start/stop the frontend dev server with `npm run dev`
   - **DO NOT** test changes by starting new servers
   - Changes to frontend code are reflected immediately at http://localhost:3000
   - If you need to restart services, use Docker commands: `docker-compose restart frontend`

2. **CORS Configuration**: Backend is configured to accept requests from `http://localhost:3000`. Modify this in `app.py` if frontend URL changes.

3. **Database Connections**: The backend manages Neo4j connections. Check `/` or `/test-neo4j` endpoints to verify database connectivity.

4. **Docker Networking**: Services communicate using Docker service names (e.g., `neo4j`, `backend`) internally, but are exposed on different ports for external access.

## Testing Guidelines

### Browser Testing with Playwright
- Use Playwright MCP tools (`mcp__playwright__*`) for browser testing and visual verification
- Available tools include:
  - `mcp__playwright__browser_navigate`: Navigate to URLs
  - `mcp__playwright__browser_take_screenshot`: Capture visual state
  - `mcp__playwright__browser_snapshot`: Get accessibility tree
  - Other browser automation tools for interaction testing
- Use these tools to verify UI changes, test user interactions, and ensure visual quality