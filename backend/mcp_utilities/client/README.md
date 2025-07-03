# MCP Client Utilities

This directory contains client implementations for interacting with MCP servers.

## Available Clients

### mcp_use_blender_3d.py

A client that demonstrates using `mcp-use` to interact with the Blender MCP server for 3D modeling tasks.

**Dependencies:**
- `mcp-use` - The MCP client framework
- `langchain-anthropic` - For using Claude models
- `python-dotenv` - For environment variable management
- `blender-mcp` - The Blender MCP server (automatically installed via uvx)

**Setup:**
1. Ensure you have a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Run the script:
   ```bash
   python mcp_use_blender_3d.py
   ```

**What it does:**
- Connects to a Blender MCP server
- Uses Claude to interpret natural language requests
- Executes Blender operations to create 3D objects
- Example: Creates an inflatable cube with soft material and a ground plane

**Note:** The Blender MCP server will be automatically downloaded and run via `uvx` when the script executes.

## Adding New Clients

When creating new MCP clients:
1. Use `mcp-use` for simplified client implementation
2. Configure the appropriate MCP servers in the config
3. Use appropriate LLM providers from LangChain
4. Handle cleanup properly in finally blocks
5. Document dependencies and setup requirements