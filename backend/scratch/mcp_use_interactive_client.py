import asyncio
from mcp_use import MCPClient, MCPAgent
from langchain_anthropic import ChatAnthropic

# TODO load_dotenv

async def main():
    # Create client with multiple servers
    client = MCPClient.from_config_file("multi_server_config.json")

    # Create agent with the client
    agent = MCPAgent(
        llm=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
        client=client,
        use_server_manager=True,  # Enable the Server Manager
        max_steps=15
    )

    try:
        # Run a query that uses tools from multiple servers
        result = await agent.run(
            "Make a world and tell me what tools are available and try to run a tool for testing"
        )
        print(result)
    finally:
        # Clean up all sessions
        await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main())

