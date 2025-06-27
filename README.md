# worldbuilder

Worldbuilder: full-stack app built with Python/FastAPI backend, React/TypeScript frontend, and Neo4j graph database. Containerized using Docker for easy deployment and development. Backend uses uv for Python package management and includes a CORS configuration for frontend-backend communication. The frontend is built with Vite.

## Running the Services

1. **Set up environment variables:**
   ```bash
   cp backend/.env-template backend/.env
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the applications:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Neo4j Browser: http://localhost:7475

4. **Default Neo4j credentials:**
   - Username: `neo4j`
   - Password: `WorldBuilder2025!`

5. **Stop all services:**
   ```bash
   docker-compose down
   ```

6. **Stop all services and delete data volumes:**
   ```bash
   docker-compose down -v
   ```

## Development

Supports hot-reloading for both frontend and backend during development. Any changes to the source code will automatically reflect in the running containers.
