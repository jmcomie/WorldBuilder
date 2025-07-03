"""Test script for Neo4j queries and operations."""
import sys
sys.path.append('..')

from app.core.database import get_driver
from app.services.neo4j_service import Neo4jService
from app.services.graph_service import GraphService


def test_neo4j_connection():
    """Test basic Neo4j connection."""
    print("Testing Neo4j connection...")
    
    driver = get_driver()
    service = Neo4jService(driver)
    
    try:
        message = service.test_connection()
        print(f"✓ {message}")
    except Exception as e:
        print(f"✗ Connection failed: {str(e)}")


def test_graph_stats():
    """Test graph statistics retrieval."""
    print("\nTesting graph statistics...")
    
    driver = get_driver()
    service = Neo4jService(driver)
    
    try:
        stats = service.get_graph_stats()
        print(f"✓ Graph Statistics:")
        print(f"  - Nodes: {stats['nodeCount']}")
        print(f"  - Edges: {stats['edgeCount']}")
        print(f"  - Node Types: {', '.join(stats['nodeTypes'])}")
        print(f"  - Edge Types: {', '.join(stats['edgeTypes'])}")
    except Exception as e:
        print(f"✗ Failed to get stats: {str(e)}")


def test_cypher_queries():
    """Test custom Cypher queries."""
    print("\nTesting custom Cypher queries...")
    
    driver = get_driver()
    
    # Test query 1: Count nodes by type
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n) as labels, count(n) as count
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print("✓ Node counts by label:")
        for record in result:
            labels = record["labels"]
            count = record["count"]
            if labels:
                print(f"  - {labels[0]}: {count}")


def test_graph_visualization_data():
    """Test graph data formatting for visualization."""
    print("\nTesting graph visualization data...")
    
    driver = get_driver()
    service = GraphService(driver)
    
    try:
        nodes, edges = service.get_graph_data(limit=10)
        print(f"✓ Graph data retrieved:")
        print(f"  - Nodes: {len(nodes)}")
        print(f"  - Edges: {len(edges)}")
        
        if nodes:
            print(f"  - Sample node: {nodes[0]['data']['label']}")
        if edges:
            print(f"  - Sample edge: {edges[0]['data']['type']}")
            
    except Exception as e:
        print(f"✗ Failed to get graph data: {str(e)}")


def test_node_search():
    """Test node search functionality."""
    print("\nTesting node search...")
    
    driver = get_driver()
    service = Neo4jService(driver)
    
    try:
        # Search for nodes containing "test"
        nodes = service.get_nodes(search="test", limit=5)
        print(f"✓ Found {len(nodes)} nodes containing 'test'")
        
        for node in nodes[:3]:
            print(f"  - {node['label']} ({node['type']})")
            
    except Exception as e:
        print(f"✗ Search failed: {str(e)}")


def main():
    """Run all Neo4j tests."""
    print("=== Neo4j Test Suite ===\n")
    
    test_neo4j_connection()
    test_graph_stats()
    test_cypher_queries()
    test_graph_visualization_data()
    test_node_search()
    
    print("\n=== Tests completed ===")


if __name__ == "__main__":
    main()