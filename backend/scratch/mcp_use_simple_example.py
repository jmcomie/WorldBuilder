"""
Simple example showing how mcp-use simplifies MCP client development

This demonstrates the key differences between the traditional MCP client
and using mcp-use for the same functionality.
"""

import asyncio
import os
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_anthropic import ChatAnthropic

load_dotenv()

async def main():
    # Path to our demo server
    server_path = os.path.abspath("../mcp_utilities/servers/demo_server.py")
    
    print("🚀 MCP-Use Simple Example")
    print("=" * 50)
    
    # 1. Create configuration - much simpler than manual setup
    config = {
        "mcpServers": {
            "worldbuilder": {
                "command": "python",
                "args": [server_path]
            }
        }
    }
    
    # 2. Create client and LLM
    client = MCPClient.from_dict(config)
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    # 3. Create agent - this handles all the complex interaction logic
    agent = MCPAgent(llm=llm, client=client, max_steps=10)
    
    try:
        # 4. Simple queries - the agent handles tool calling automatically
        print("\n📝 Example 1: Analyzing a world element")
        result = await agent.run(
            "Analyze a character named 'Elena the Wise' who is a magical scholar"
        )
        print(f"Result: {result}\n")
        
        print("\n📝 Example 2: Generating a story seed")
        result = await agent.run(
            "Generate a detailed fantasy story seed with high complexity"
        )
        print(f"Result: {result}\n")
        
        print("\n📝 Example 3: Using multiple tools")
        result = await agent.run(
            "Create a story seed for a sci-fi romance, then calculate the narrative "
            "distance between the protagonist and antagonist"
        )
        print(f"Result: {result}\n")
        
    finally:
        # 5. Cleanup is simple
        await client.close_all_sessions()
    
    print("\n✅ Demo complete!")
    
    # Show the key differences
    print("\n📚 Key Differences from Traditional MCP Client:")
    print("1. No manual session management needed")
    print("2. No manual tool discovery and formatting")
    print("3. Agent handles multi-step tool interactions automatically")
    print("4. Natural language queries instead of direct tool calls")
    print("5. Built-in LLM integration for intelligent tool selection")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found")
        print("Please add it to your .env file")
    else:
        asyncio.run(main())