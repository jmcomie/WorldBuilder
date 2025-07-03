"""
MCP-Use Interactive Client

An interactive command-line interface for interacting with MCP (Model Context Protocol) servers
using natural language through an AI agent powered by Claude.

This client provides:
- Natural language interaction with MCP servers via Claude AI
- Discovery of available tools, resources, and prompts from connected servers
- Automatic server management for efficient resource usage
- Clean session handling and graceful exits

Usage:
    python mcp_use_interactive_client.py

Requires:
    - A valid multi_server_config.json file with MCP server configurations
    - ANTHROPIC_API_KEY environment variable (can be set in .env file)
    - Configured MCP servers (e.g., filesystem, playwright, etc.)

Commands:
    /help      - Show help message
    /tools     - List all available tools from connected servers
    /resources - List all available resources from connected servers
    /prompts   - List all available prompts from connected servers
    /exit      - Exit the program
    /quit      - Exit the program

Example queries:
    - "Create a file called test.txt with 'Hello World' content"
    - "Search for Python tutorials on Google"
    - "What tools do you have available?"
    - "Read the contents of README.md"
"""

import asyncio
import sys
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_anthropic import ChatAnthropic

# Load environment variables
load_dotenv()

def print_welcome():
    """Print welcome message and instructions"""
    print("\n" + "=" * 60)
    print("🤖 MCP-Use Interactive Client")
    print("=" * 60)
    print("Type your queries below. The agent will use available MCP tools.")
    print("Commands:")
    print("  /help      - Show this help message")
    print("  /tools     - List available tools")
    print("  /resources - List available resources") 
    print("  /prompts   - List available prompts")
    print("  /exit      - Exit the program")
    print("  /quit      - Exit the program")
    print("=" * 60 + "\n")

async def list_tools(client):
    """List all available tools from connected servers"""
    try:
        all_tools = []
        server_tools = {}
        
        # Get tools from each connected server
        for server_name, session in client.sessions.items():
            if session and hasattr(session, 'list_tools'):
                try:
                    result = await session.list_tools()
                    if hasattr(result, 'tools'):
                        server_tools[server_name] = result.tools
                        all_tools.extend(result.tools)
                except Exception as e:
                    print(f"Warning: Could not get tools from {server_name}: {e}")
        
        if all_tools:
            print("\n📦 Available Tools:")
            print("=" * 60)
            
            # Group by server
            for server_name, tools in server_tools.items():
                if tools:
                    print(f"\n🖥️  Server: {server_name}")
                    print("-" * 40)
                    for tool in tools:
                        print(f"  • {tool.name}")
                        if hasattr(tool, 'description') and tool.description:
                            # Wrap long descriptions
                            desc_lines = [tool.description[i:i+55] for i in range(0, len(tool.description), 55)]
                            for line in desc_lines:
                                print(f"    {line}")
            
            print("\n" + "=" * 60)
            print(f"Total tools available: {len(all_tools)}")
            print("=" * 60 + "\n")
        else:
            print("\n⚠️  No tools available. Make sure servers are properly configured and running.\n")
    except Exception as e:
        print(f"\n❌ Error listing tools: {e}\n")
        print("Tip: Tools will be available after the first query when using server_manager.\n")

async def list_resources(client):
    """List all available resources from connected servers"""
    try:
        all_resources = []
        server_resources = {}
        
        # Get resources from each connected server
        for server_name, session in client.sessions.items():
            if session and hasattr(session, 'list_resources'):
                try:
                    result = await session.list_resources()
                    if hasattr(result, 'resources'):
                        server_resources[server_name] = result.resources
                        all_resources.extend(result.resources)
                except Exception as e:
                    print(f"Warning: Could not get resources from {server_name}: {e}")
        
        if all_resources:
            print("\n📚 Available Resources:")
            print("=" * 60)
            
            # Group by server
            for server_name, resources in server_resources.items():
                if resources:
                    print(f"\n🖥️  Server: {server_name}")
                    print("-" * 40)
                    for resource in resources:
                        print(f"  • {resource.uri}")
                        if hasattr(resource, 'name') and resource.name:
                            print(f"    Name: {resource.name}")
                        if hasattr(resource, 'description') and resource.description:
                            print(f"    {resource.description}")
            
            print("\n" + "=" * 60)
            print(f"Total resources available: {len(all_resources)}")
            print("=" * 60 + "\n")
        else:
            print("\n⚠️  No resources available from connected servers.\n")
    except Exception as e:
        print(f"\n❌ Error listing resources: {e}\n")

async def list_prompts(client):
    """List all available prompts from connected servers"""
    try:
        all_prompts = []
        server_prompts = {}
        
        # Get prompts from each connected server
        for server_name, session in client.sessions.items():
            if session and hasattr(session, 'list_prompts'):
                try:
                    result = await session.list_prompts()
                    if hasattr(result, 'prompts'):
                        server_prompts[server_name] = result.prompts
                        all_prompts.extend(result.prompts)
                except Exception as e:
                    print(f"Warning: Could not get prompts from {server_name}: {e}")
        
        if all_prompts:
            print("\n💬 Available Prompts:")
            print("=" * 60)
            
            # Group by server
            for server_name, prompts in server_prompts.items():
                if prompts:
                    print(f"\n🖥️  Server: {server_name}")
                    print("-" * 40)
                    for prompt in prompts:
                        print(f"  • {prompt.name}")
                        if hasattr(prompt, 'description') and prompt.description:
                            print(f"    {prompt.description}")
                        if hasattr(prompt, 'arguments') and prompt.arguments:
                            print(f"    Arguments: {', '.join([arg.name for arg in prompt.arguments])}")
            
            print("\n" + "=" * 60)
            print(f"Total prompts available: {len(all_prompts)}")
            print("=" * 60 + "\n")
        else:
            print("\n⚠️  No prompts available from connected servers.\n")
    except Exception as e:
        print(f"\n❌ Error listing prompts: {e}\n")

async def process_query(agent, query):
    """Process a user query with the agent"""
    try:
        print("\n🔄 Processing your request...\n")
        result = await agent.run(query)
        print("\n✅ Result:")
        print("-" * 40)
        print(result)
        print("-" * 40 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

async def interactive_loop(agent, client):
    """Main interactive loop"""
    print_welcome()
    
    while True:
        try:
            # Get user input
            query = input("You: ").strip()
            
            # Handle empty input
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['/exit', '/quit']:
                print("\n👋 Goodbye!")
                break
            elif query.lower() == '/help':
                print_welcome()
                continue
            elif query.lower() == '/tools':
                await list_tools(client)
                continue
            elif query.lower() == '/resources':
                await list_resources(client)
                continue
            elif query.lower() == '/prompts':
                await list_prompts(client)
                continue
            
            # Process regular query
            await process_query(agent, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")

async def main():
    """Main function to set up and run the interactive client"""
    client = None
    
    try:
        # Create client with multiple servers
        print("🚀 Initializing MCP client...")
        client = MCPClient.from_config_file("multi_server_config.json")
        
        # Create agent with the client
        print("🤖 Setting up AI agent...")
        agent = MCPAgent(
            llm=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            client=client,
            use_server_manager=True,  # Enable the Server Manager
            max_steps=15
        )
        
        print("✅ Ready!\n")
        
        # Run interactive loop
        await interactive_loop(agent, client)
        
    except FileNotFoundError:
        print("❌ Error: multi_server_config.json not found!")
        print("Please create a configuration file with your MCP servers.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    finally:
        # Clean up all sessions
        if client:
            print("\n🧹 Cleaning up connections...")
            await client.close_all_sessions()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

