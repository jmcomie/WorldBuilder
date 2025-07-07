"""
MCP-Use Interactive Client

An interactive command-line interface for interacting with MCP (Model Context Protocol) servers
using natural language through an AI agent powered by Claude.

This client provides:
- Natural language interaction with MCP servers via Claude AI
- Discovery of available tools, resources, and prompts from connected servers
- Flexible server configuration via config files or command-line arguments
- Configurable server management and execution parameters
- Clean session handling and graceful exits

Usage:
    python mcp_use_interactive_client.py [OPTIONS]

Options:
    --config PATH                  Path to server config JSON file
    --servers TEXT                 Comma-delimited list of servers (dot notation)
    --use-server-manager          Enable server manager (default: False)
    --max-steps INTEGER           Maximum steps for agent execution (default: 15)
    --help                        Show this message and exit

At least one of --config or --servers must be specified.

Server specification:
    - Config file: Standard MCP JSON configuration
    - Dot notation: "demo_server" or "ontology.ontology_entity_graph_outline_server"
      Maps to backend/mcp_utilities/servers/[path].py

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
import os
import json
from pathlib import Path
import click
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

def parse_server_path(server_spec):
    """
    Parse a dot-notation server path into a Python module path.
    
    Args:
        server_spec: String like "demo_server" or "ontology.ontology_entity_graph_outline_server"
        
    Returns:
        Full path to the Python server file
    """
    # Get the backend directory path
    backend_dir = Path(__file__).parent.parent
    mcp_utilities_dir = backend_dir / "mcp_utilities" / "servers"
    
    # Convert dots to path separators
    path_parts = server_spec.split('.')
    server_path = mcp_utilities_dir / Path(*path_parts).with_suffix('.py')
    
    if not server_path.exists():
        raise ValueError(f"Server file not found: {server_path}")
    
    return str(server_path)

def create_server_config(server_spec):
    """
    Create a server configuration for a Python-based MCP server.
    
    Args:
        server_spec: Dot-notation server specification
        
    Returns:
        Dictionary with server configuration
    """
    server_path = parse_server_path(server_spec)
    
    # Get the Python executable from the current virtual environment
    python_executable = sys.executable
    
    # Create a simple name from the spec
    server_name = server_spec.replace('.', '_')
    
    return {
        server_name: {
            "command": python_executable,
            "args": [server_path]
        }
    }

def load_servers_from_specs(server_specs):
    """
    Load server configurations from comma-delimited server specifications.
    
    Args:
        server_specs: Comma-delimited string of server specifications
        
    Returns:
        Dictionary with mcpServers configuration
    """
    if not server_specs:
        return {"mcpServers": {}}
    
    servers = {}
    for spec in server_specs.split(','):
        spec = spec.strip()
        if spec:
            try:
                server_config = create_server_config(spec)
                servers.update(server_config)
            except Exception as e:
                click.echo(f"⚠️  Warning: Failed to load server '{spec}': {e}", err=True)
    
    return {"mcpServers": servers}

def merge_server_configs(config1, config2):
    """
    Merge two server configuration dictionaries.
    
    Args:
        config1: First configuration dictionary
        config2: Second configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = {"mcpServers": {}}
    
    if "mcpServers" in config1:
        merged["mcpServers"].update(config1["mcpServers"])
    
    if "mcpServers" in config2:
        merged["mcpServers"].update(config2["mcpServers"])
    
    return merged

@click.command()
@click.option('--config', type=click.Path(exists=True), help='Path to server config JSON file')
@click.option('--servers', type=str, help='Comma-delimited list of servers (dot notation)')
@click.option('--use-server-manager/--no-use-server-manager', default=False, help='Enable server manager')
@click.option('--max-steps', type=int, default=15, help='Maximum steps for agent execution')
def cli(config, servers, use_server_manager, max_steps):
    """
    MCP-Use Interactive Client - Natural language interface for MCP servers.
    
    At least one of --config or --servers must be specified.
    """
    # Validate that at least one server source is provided
    if not config and not servers:
        click.echo("❌ Error: At least one of --config or --servers must be specified.", err=True)
        click.echo("Use --help for more information.", err=True)
        sys.exit(1)
    
    # Run the async main function with the provided parameters
    asyncio.run(main(config, servers, use_server_manager, max_steps))

async def main(config_path, server_specs, use_server_manager, max_steps):
    """Main function to set up and run the interactive client"""
    client = None
    
    try:
        # Load configurations
        config_dict = {"mcpServers": {}}
        
        # Load from config file if provided
        if config_path:
            print(f"📄 Loading configuration from {config_path}...")
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                config_dict = merge_server_configs(config_dict, file_config)
            except Exception as e:
                print(f"❌ Error loading config file: {e}")
                sys.exit(1)
        
        # Load from server specs if provided
        if server_specs:
            print(f"🔧 Loading servers from command line...")
            spec_config = load_servers_from_specs(server_specs)
            config_dict = merge_server_configs(config_dict, spec_config)
        
        # Verify we have at least one server
        if not config_dict.get("mcpServers"):
            print("❌ Error: No servers configured!")
            sys.exit(1)
        
        # Display loaded servers
        print(f"📋 Loaded {len(config_dict['mcpServers'])} server(s):")
        for server_name in config_dict['mcpServers']:
            print(f"   • {server_name}")
        
        # Create client with the merged configuration
        print("\n🚀 Initializing MCP client...")
        client = MCPClient.from_dict(config_dict)
        
        # Create agent with the client
        print("🤖 Setting up AI agent...")
        agent = MCPAgent(
            llm=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            client=client,
            use_server_manager=use_server_manager,
            max_steps=max_steps
        )
        
        print(f"⚙️  Configuration: server_manager={'enabled' if use_server_manager else 'disabled'}, max_steps={max_steps}")
        print("✅ Ready!\n")
        
        # Run interactive loop
        await interactive_loop(agent, client)
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    finally:
        # Clean up all sessions
        if client:
            print("\n🧹 Cleaning up connections...")
            await client.close_all_sessions()

if __name__ == "__main__":
    cli()

