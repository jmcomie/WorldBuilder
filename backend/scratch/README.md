# Scratch Directory

This directory is for testing and experimentation. Files here are not part of the main application and can be used for:

- Testing new features
- Experimenting with APIs
- Running one-off scripts
- Debugging issues

## Available Test Scripts


### Test Interactive MCP Client

#### Example
```bash
python mcp_interactive_client.py ../mcp_utilities/servers/demo_server.py
```

### test_mcp.py
Tests MCP (Model Context Protocol) server and client implementations.

```bash
python test_mcp.py
```

### test_graphiti.py
Tests Graphiti integration and knowledge graph operations.

```bash
python test_graphiti.py
```

### test_neo4j.py
Tests direct Neo4j database operations and queries.

```bash
python test_neo4j.py
```

## Usage

All test scripts can be run from the scratch directory:

```bash
cd backend/scratch
python <script_name>.py
```

Scripts automatically add the parent directory to the Python path, so they can import from the main application modules.

## Important Notes

- Files in this directory should not be imported by the main application
- Test data created here may need manual cleanup
- Always use test/development databases when running these scripts
