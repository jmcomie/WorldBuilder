# Graph Layout Optimization Plan

## Problem Statement
The current graph visualization shows a spaghetti-like layout with nodes positioned without consideration for their fact relationships. The edges (facts) cross over each other unnecessarily, making it difficult to understand the relationships between entities.

## Research Findings

### Graphiti's Built-in Capabilities

Graphiti provides several features that can help optimize graph layout:

1. **Center Node Distance Ranking**: 
   - The `search()` function accepts a `center_node_uuid` parameter
   - Results are ranked based on their graph distance from the center node
   - This provides a natural hierarchy for layout algorithms

2. **Reranking/Scoring**:
   - Graphiti supports `cross_encoder` with `OpenAIRerankerClient`
   - Can provide relevance scores for relationships
   - These scores can be used as edge weights

3. **Search Configurations**:
   - `NODE_HYBRID_SEARCH_EPISODE_MENTIONS` and other recipes
   - Combine semantic similarity with graph structure
   - Results include distance/relevance metrics

### Using Graphiti's Distance Metrics for Layout

We can leverage Graphiti's search capabilities to:
1. Query node distances from a central node
2. Use these distances to create concentric layout rings
3. Apply relevance scores as edge weights
4. Group semantically similar nodes

### 1. Cytoscape.js Layout Capabilities

#### Force-Directed Layouts with Edge Weighting
- **fcose (fast Compound Spring Embedder)**: Currently using this, but not optimized
- **cose (Compound Spring Embedder)**: Supports edge weights and clustering
- **cola**: Constraint-based layout with better edge routing

#### Key Layout Properties for Relationship Optimization:
```javascript
{
  name: 'fcose',
  // Edge-specific parameters
  idealEdgeLength: function(edge) {
    // Can vary based on edge properties
    return edge.data('weight') || 50;
  },
  edgeElasticity: function(edge) {
    // Higher elasticity for less important edges
    return edge.data('fact') ? 0.45 : 0.8;
  },
  nodeRepulsion: 4500,
  
  // Clustering support
  nestingFactor: 0.1,
  gravity: 0.25,
  
  // Quality vs performance
  quality: 'proof', // 'draft' | 'default' | 'proof'
  numIter: 2500,
  
  // Edge routing
  tile: true, // Better edge bundling
  tilingPaddingVertical: 10,
  tilingPaddingHorizontal: 10
}
```

#### Edge Bundling Options:
- **curve-style**: 'bundled-bezier' - Groups parallel edges
- **control-point-distances**: Control bezier curve shapes
- **edge-distances**: Define edge routing behavior

### 2. Graphiti-Specific Considerations

From Graphiti documentation:
- Facts are stored as edge properties with semantic meaning
- Edges have temporal validity (valid_at/invalid_at)
- Facts connect entities with specific relationships
- No native layout algorithms, but data structure supports weighted graphs

### 3. Layout Optimization Strategies

#### A. Semantic Clustering
Group nodes based on:
1. **Entity type**: Person, Organization, Location, Event
2. **Fact density**: Nodes with many shared facts should be closer
3. **Temporal relationships**: Recent facts could influence proximity

#### B. Edge Weight Calculation
Create dynamic edge weights based on:
1. **Fact importance**: Longer facts might indicate stronger relationships
2. **Fact type**: Different relationship types (e.g., "works at" vs "knows")
3. **Temporal relevance**: More recent facts get higher weight
4. **Fact frequency**: Multiple facts between nodes increase weight

#### C. Visual Hierarchy
1. **Primary relationships**: Direct facts with high weight
2. **Secondary relationships**: Indirect connections through shared nodes
3. **Peripheral nodes**: Entities with few connections

## Implementation Plan

### Phase 0: Integrate Graphiti's Distance Metrics (NEW - 1 hour)

1. **Add Backend Endpoint for Distance Calculations**
   ```python
   @app.get("/graph/node-distances/{center_node_uuid}")
   async def get_node_distances(center_node_uuid: str):
       """Calculate graph distances from a center node using Graphiti"""
       # Method 1: Use Graphiti search with center node
       client = await get_graphiti()
       
       # Search for related nodes from center
       search_results = await client.search(
           query="",  # Empty query to get all related nodes
           center_node_uuid=center_node_uuid,
           num_results=100
       )
       
       # Method 2: Direct Neo4j query for graph distances
       with driver.session() as session:
           result = session.run("""
               MATCH (center {uuid: $center_uuid})
               MATCH (center)-[*0..3]-(node)
               WHERE node.uuid IS NOT NULL
               WITH node, min(length((center)-[*]-(node))) as distance
               RETURN node.uuid as uuid, node.name as name, distance
               ORDER BY distance
               LIMIT 100
           """, center_uuid=center_node_uuid)
           
       return {"nodes": [{"uuid": r["uuid"], "name": r["name"], "distance": r["distance"]} for r in result]}
   ```

2. **Frontend Integration**
   ```javascript
   // Fetch distance data
   const distances = await api.getNodeDistances(centerNodeId);
   
   // Use distances for layout positioning
   const concentricLayout = {
     name: 'concentric',
     concentric: function(node) {
       // Position based on distance from center
       const distance = distances[node.id()] || Infinity;
       return 1 / (distance + 1); // Closer nodes have higher values
     },
     levelWidth: function(nodes) {
       return 2; // Number of concentric levels
     }
   };
   ```

3. **Use Graphiti's Relevance Scores**
   - Query edges with relevance scores
   - Apply scores as edge weights
   - Influence node positioning based on relationship strength

### Phase 1: Enhanced Edge Weight System with Graphiti Scores (Quick Win - 1 hour)

1. **Calculate Edge Weights**
   ```javascript
   function calculateEdgeWeight(edge) {
     const fact = edge.data('fact');
     if (!fact) return 1;
     
     // Base weight on fact length (importance proxy)
     let weight = Math.min(fact.length / 50, 3);
     
     // Boost for certain relationship types
     if (fact.includes('Member of') || fact.includes('works at')) {
       weight *= 1.5;
     }
     
     // Consider temporal relevance if available
     const validAt = edge.data('valid_at');
     if (validAt) {
       const age = Date.now() - new Date(validAt).getTime();
       const ageInDays = age / (1000 * 60 * 60 * 24);
       weight *= Math.max(0.5, 1 - (ageInDays / 365)); // Decay over a year
     }
     
     return weight;
   }
   ```

2. **Apply Weights to Layout**
   - Modify fcose layout configuration
   - Use weights for idealEdgeLength and edgeElasticity

### Phase 2: Advanced Layout Algorithm (2 hours)

1. **Implement Cola Layout**
   ```javascript
   cytoscape.use(cola); // Need to add cola extension
   
   const colaLayout = {
     name: 'cola',
     
     // Better for directed graphs with constraints
     flow: { axis: 'y', minSeparation: 30 },
     
     // Edge length based on relationship strength
     edgeLength: function(edge) {
       const weight = calculateEdgeWeight(edge);
       return 100 / weight; // Inverse - stronger relationships are shorter
     },
     
     // Constraints for better organization
     alignment: function() {
       // Align nodes of same type horizontally
       return cy.nodes('[type="Person"]');
     },
     
     // Avoid overlaps
     avoidOverlap: true,
     handleDisconnected: true
   };
   ```

2. **Hierarchical Layout for Certain Relationships**
   - Detect hierarchical patterns (e.g., organizational structures)
   - Apply dagre or breadthfirst layout to subgraphs

### Phase 3: Edge Bundling and Routing (1 hour)

1. **Implement Edge Bundling**
   ```javascript
   // Style modifications
   {
     selector: 'edge[fact]',
     style: {
       'curve-style': 'bundled-bezier',
       'control-point-distances': function(edge) {
         // Dynamic control points based on node positions
         return calculateControlPoints(edge);
       },
       'edge-distances': 'node-position'
     }
   }
   ```

2. **Smart Edge Routing**
   - Detect crossing edges
   - Adjust control points to minimize crossings
   - Use taxi or segments curve-style for orthogonal routing

### Phase 4: Interactive Optimization (1 hour)

1. **User Controls**
   - Layout strength slider (tight vs loose)
   - Edge bundling toggle
   - Relationship type filters
   - Time-based filtering

2. **Progressive Layout**
   - Start with fast draft quality
   - Progressively refine to proof quality
   - Allow manual node pinning

### Phase 5: Community Detection (Optional - 2 hours)

1. **Implement Clustering Algorithms**
   ```javascript
   // Markov clustering based on fact relationships
   const clusters = cy.elements().markovClustering({
     attributes: [
       function(edge) { 
         return edge.data('fact') ? calculateEdgeWeight(edge) : 0;
       }
     ]
   });
   ```

2. **Visual Clustering**
   - Color code communities
   - Add subtle backgrounds for clusters
   - Compound nodes for strong communities

## Quick Implementation Priority (Revised with Graphiti Integration)

1. **Immediate (30 min)**:
   - Query Graphiti for node importance/centrality
   - Use Graphiti's center_node_uuid for hierarchical layout
   - Apply distance-based positioning

2. **Next (1 hour)**:
   - Integrate Graphiti's search scores as edge weights
   - Implement concentric or hierarchical layout based on graph distance
   - Add UI to select center node for layout

3. **Later (2+ hours)**:
   - Use Graphiti's semantic search for node clustering
   - Implement dynamic reranking based on user focus
   - Add temporal filtering using valid_at/invalid_at

## Testing Considerations

- Performance with 100+ nodes
- Visual clarity at different zoom levels
- Mobile responsiveness
- Animation smoothness during layout transitions

## Success Metrics

1. Reduced edge crossings by 50%+
2. Clear visual grouping of related entities
3. Readable fact labels without overlap
4. Smooth interaction with 100+ nodes
5. User can understand relationships at a glance