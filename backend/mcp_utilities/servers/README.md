# MCP Servers

This directory contains Model Context Protocol (MCP) server implementations for the Worldbuilder application.

## Available Servers

### Demo Server (`demo_server.py`)

A demonstration MCP server showcasing FastMCP features without performing any mutations. Perfect for testing and learning.

**Features:**
- 4 Tools for worldbuilding analysis and generation
- 3 Resources providing templates and statistics  
- 3 Prompts for creative writing assistance

**Running the server:**

```bash
# Run directly
python demo_server.py

# Test configuration
python demo_server.py --test

# Run with specific transport
python demo_server.py  # Uses STDIO by default
fastmcp run demo_server.py:mcp --transport http --port 8000
```

**Using with a client:**

```python
from fastmcp import Client

async with Client("demo_server.py") as client:
    # List available tools
    tools = await client.list_tools()
    
    # Call a tool
    result = await client.call_tool("generate_story_seed", {
        "parameters": {
            "genre": "fantasy",
            "tone": "dark",
            "complexity": 7
        }
    })
```

## Creating New Servers

To add a new MCP server:

1. Create a new Python file in this directory
2. Import and initialize FastMCP:
   ```python
   from fastmcp import FastMCP
   mcp = FastMCP("Your Server Name")
   ```
3. Add tools, resources, and prompts using decorators
4. Add the server to `__init__.py` exports
5. Include proper documentation

## Best Practices

- Always include comprehensive docstrings
- Use type hints for all parameters
- Handle errors gracefully
- Provide mock/safe implementations for demos
- Test with `--test` flag before deployment
- Consider authentication for production servers