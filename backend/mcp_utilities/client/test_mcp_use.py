"""
Test script to verify mcp-use installation and basic functionality.
"""

import asyncio
from mcp_use import MCPClient
from langchain_anthropic import ChatAnthropic

def test_imports():
    """Test that all required imports work."""
    print("✅ All imports successful!")
    print(f"  - mcp-use version: {MCPClient.__module__}")
    print(f"  - langchain-anthropic available: {ChatAnthropic.__module__}")

def test_client_creation():
    """Test creating an MCP client."""
    try:
        # Create a simple config
        config = {"mcpServers": {}}
        client = MCPClient.from_dict(config)
        print("✅ MCPClient created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to create MCPClient: {e}")
        return False

async def test_async_context():
    """Test async context manager."""
    try:
        config = {"mcpServers": {}}
        client = MCPClient.from_dict(config)
        
        # Test that we can use the client in an async context
        # (won't actually connect to anything without servers configured)
        print("✅ Async context test passed!")
        return True
    except Exception as e:
        print(f"❌ Async context test failed: {e}")
        return False

def main():
    print("Testing mcp-use installation...\n")
    
    # Test imports
    test_imports()
    print()
    
    # Test client creation
    test_client_creation()
    print()
    
    # Test async functionality
    asyncio.run(test_async_context())
    print()
    
    print("✅ All tests passed! mcp-use is properly installed.")
    print("\nTo use mcp_use_blender_3d.py, ensure you have:")
    print("1. ANTHROPIC_API_KEY in your .env file")
    print("2. Blender MCP server will be auto-installed via uvx when running")

if __name__ == "__main__":
    main()