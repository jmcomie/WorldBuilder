"""Test script for Graphiti experiments."""
import asyncio
import sys
sys.path.append('..')

from app.services.graphiti_service import graphiti_service
from app.config import get_settings


async def test_graphiti_connection():
    """Test Graphiti connection and basic operations."""
    print("Testing Graphiti connection...")
    
    try:
        # Get client
        client = await graphiti_service.get_client()
        print("✓ Graphiti client initialized")
        
        # Test adding an episode
        print("\nTesting episode addition...")
        result = await graphiti_service.add_episode(
            name="Test Episode from Scratch",
            content="This is a test episode created from the scratch directory.",
            source_description="Scratch test script"
        )
        print(f"✓ Episode added: {result}")
        
        # Test search
        print("\nTesting search...")
        search_results = await graphiti_service.search(
            query="test episode",
            num_results=5
        )
        print(f"✓ Found {len(search_results)} search results")
        for i, edge in enumerate(search_results[:3]):
            print(f"  {i+1}. {edge.fact}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    finally:
        await graphiti_service.close()


async def test_advanced_search():
    """Test advanced search capabilities."""
    print("\n\nTesting advanced search...")
    
    try:
        # Search with center node (if available)
        print("Searching for related nodes...")
        results = await graphiti_service.search(
            query="",
            num_results=20
        )
        
        if results:
            # Use first result's source node as center
            center_uuid = results[0].source_node_uuid
            print(f"\nSearching around node: {center_uuid}")
            
            centered_results = await graphiti_service.search(
                query="",
                center_node_uuid=center_uuid,
                num_results=10
            )
            
            print(f"✓ Found {len(centered_results)} related edges")
            
    except Exception as e:
        print(f"✗ Error in advanced search: {str(e)}")


async def main():
    """Run all Graphiti tests."""
    settings = get_settings()
    
    print("=== Graphiti Test Suite ===")
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"OpenAI API Key: {'*' * 10 if settings.openai_api_key else 'NOT SET'}")
    print()
    
    await test_graphiti_connection()
    await test_advanced_search()
    
    print("\n=== Tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())