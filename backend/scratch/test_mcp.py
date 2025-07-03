"""Test script for MCP functionality."""
import asyncio
import sys
sys.path.append('..')

from mcp.client.worldbuilder_client import WorldbuilderMCPClient
from mcp.servers.worldbuilder_mcp import WorldbuilderMCPServer, MCPRequest


async def test_mcp_server():
    """Test the MCP server."""
    print("Testing MCP Server...")
    server = WorldbuilderMCPServer()
    
    # Test graph query
    request = MCPRequest(
        method="graph.query",
        params={"query": "test query", "limit": 10},
        id="test-1"
    )
    response = await server.handle_request(request)
    print(f"Graph query response: {response}")
    
    # Test episode addition
    request = MCPRequest(
        method="episode.add",
        params={"name": "Test Episode", "content": "Test content"},
        id="test-2"
    )
    response = await server.handle_request(request)
    print(f"Episode add response: {response}")
    
    # Test search
    request = MCPRequest(
        method="search.execute",
        params={"query": "test search", "num_results": 5},
        id="test-3"
    )
    response = await server.handle_request(request)
    print(f"Search response: {response}")


async def test_mcp_client():
    """Test the MCP client."""
    print("\nTesting MCP Client...")
    
    async with WorldbuilderMCPClient("http://localhost:8080") as client:
        # Test graph query
        result = await client.query_graph("test query", limit=10)
        print(f"Graph query result: {result}")
        
        # Test episode addition
        result = await client.add_episode("Test Episode", "Test content")
        print(f"Episode add result: {result}")
        
        # Test search
        result = await client.search("test search", num_results=5)
        print(f"Search result: {result}")


async def main():
    """Run all tests."""
    await test_mcp_server()
    await test_mcp_client()


if __name__ == "__main__":
    asyncio.run(main())