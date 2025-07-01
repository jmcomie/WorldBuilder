# Graphiti Facts Display Update Plan

## Problem
The graph visualization is missing Graphiti's core feature: fact edges. Graphiti stores extracted facts as edges between entities with descriptive labels, but these are not being displayed.

## Understanding Graphiti's Data Model

### EntityEdge Structure
```
- UUID: Unique identifier for the edge
- Fact: The extracted fact from the episode (e.g., "John works at Acme Corp")
- Valid at/invalid at: Time period when fact was true
- Source/target node UUIDs: Connections between entities
```

### How Graphiti Stores Facts
1. When episodes are added, Graphiti extracts entities and relationships
2. Entities become nodes (Person, Organization, Location, Event)
3. Facts become labeled edges connecting entities
4. Facts are the actual descriptive text, not just relationship types

## Current Issue
The `/graph` endpoint queries Neo4j directly using a generic pattern that returns nodes but no meaningful edges. Graphiti's fact edges are stored differently and need specific queries.

## Simple Implementation Plan

### 1. Backend Changes (30 minutes)
**File**: `backend/app.py`

#### Option A: Direct Neo4j Query for Graphiti Data
```python
@app.get("/graph")
async def get_graph_data():
    # Query for nodes with Graphiti labels
    nodes_query = """
    MATCH (n)
    WHERE n.uuid IS NOT NULL
    RETURN n
    LIMIT 100
    """
    
    # Query for fact edges
    edges_query = """
    MATCH (source)-[r]->(target)
    WHERE source.uuid IS NOT NULL 
    AND target.uuid IS NOT NULL
    AND r.fact IS NOT NULL
    RETURN source.uuid as source_id, 
           target.uuid as target_id, 
           r.fact as fact,
           r.uuid as edge_id,
           type(r) as relationship_type
    LIMIT 200
    """
```

#### Option B: Hybrid Approach (Preferred)
Use Graphiti client for search to get facts, then enhance with Neo4j structure:
```python
@app.get("/graph/with-facts")
async def get_graph_with_facts():
    client = await get_graphiti()
    
    # Get nodes from Neo4j
    # Get edges with facts using Graphiti search
    # Combine into Cytoscape format
```

### 2. Frontend Changes (15 minutes)
**File**: `frontend/src/components/GraphVisualization/GraphVisualizationSimple.tsx`

#### Display Fact Labels on Edges
```javascript
{
    selector: 'edge',
    style: {
        'label': 'data(fact)', // Changed from 'data(label)'
        'text-wrap': 'wrap',
        'text-max-width': '200px',
        'font-size': 10,
        'text-rotation': 'autorotate',
        'text-margin-y': -10
    }
}
```

### 3. Data Format
Ensure edges include fact data:
```javascript
{
    "data": {
        "id": "edge-uuid",
        "source": "source-node-uuid",
        "target": "target-node-uuid",
        "fact": "John Smith works at Acme Corp as CTO",
        "type": "HAS_FACT"
    }
}
```

## Quick Implementation Steps

1. **Check Neo4j Schema** (5 min)
   - Connect to Neo4j browser at http://localhost:7475
   - Run: `MATCH (n)-[r]->(m) WHERE r.fact IS NOT NULL RETURN n, r, m LIMIT 10`
   - Verify how Graphiti stores facts

2. **Update Backend Endpoint** (20 min)
   - Modify `/graph` endpoint to include fact edges
   - Ensure edge data includes the `fact` property
   - Test with curl/Postman

3. **Update Frontend Display** (10 min)
   - Change edge label from `data(label)` to `data(fact)`
   - Add text wrapping for long facts
   - Test visualization

## Success Criteria
- Edges display descriptive facts (e.g., "works at", "located in", "attended")
- Facts are readable (proper text size, wrapping)
- Graph shows the knowledge relationships extracted by Graphiti

## Future Enhancements
- Filter edges by fact type
- Show/hide fact labels
- Temporal filtering (valid_at/invalid_at)
- Fact confidence scores
- Edge bundling for multiple facts between same nodes