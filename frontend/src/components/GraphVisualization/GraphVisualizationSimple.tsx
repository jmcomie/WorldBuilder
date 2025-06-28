import React, { useRef, useEffect, useState } from 'react';
import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';
import { api } from '../../api';
import './GraphVisualization.css';

// Register the fcose layout
cytoscape.use(fcose);

interface NodeDetails {
    id: string;
    label: string;
    type: string;
    properties: any;
}

export const GraphVisualizationSimple: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedNode, setSelectedNode] = useState<NodeDetails | null>(null);
    const [currentLayout, setCurrentLayout] = useState<'circle' | 'cose'>('circle');
    const cyRef = useRef<any>(null);

    useEffect(() => {
        const fetchAndRenderGraph = async () => {
            try {
                console.log('Fetching graph data...');
                const response = await api.getGraphWithFacts({ limit: 100 });
                
                if (!response.success || !containerRef.current) {
                    setError('Failed to load graph data');
                    setLoading(false);
                    return;
                }

                console.log('Graph data received:', response.elements);

                // Wait for container to have dimensions
                const waitForContainer = () => {
                    if (!containerRef.current) return;
                    
                    const rect = containerRef.current.getBoundingClientRect();
                    const width = rect.width;
                    const height = rect.height;
                    
                    console.log('Container dimensions:', { width, height, rect });
                    
                    if (width > 0 && height > 0 && !cyRef.current) {
                        // Ensure container has full dimensions
                        containerRef.current.style.width = '100%';
                        containerRef.current.style.height = '100%';
                        containerRef.current.style.position = 'relative';
                        
                        const cy = cytoscape({
                        container: containerRef.current,
                        elements: [...response.elements.nodes, ...response.elements.edges],
                        style: [
                            {
                                selector: 'core',
                                style: {
                                    'active-bg-color': '#fff',
                                    'active-bg-opacity': 0
                                }
                            },
                            {
                                selector: 'node',
                                style: {
                                    'background-color': '#64748b',
                                    'label': 'data(label)',
                                    'color': '#ffffff',
                                    'width': 40,
                                    'height': 40,
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'font-size': 14,
                                    'font-weight': 'bold',
                                    'border-width': 2,
                                    'border-color': '#475569',
                                    'text-background-color': '#64748b',
                                    'text-background-opacity': 0.8,
                                    'text-background-padding': 3
                                }
                            },
                            {
                                selector: 'node[type="Person"]',
                                style: {
                                    'background-color': '#3b82f6',
                                    'border-color': '#1e40af',
                                    'text-background-color': '#3b82f6'
                                }
                            },
                            {
                                selector: 'node[type="Organization"]',
                                style: {
                                    'background-color': '#dc2626',
                                    'border-color': '#991b1b',
                                    'text-background-color': '#dc2626'
                                }
                            },
                            {
                                selector: 'node[type="Location"]',
                                style: {
                                    'background-color': '#16a34a',
                                    'border-color': '#14532d',
                                    'text-background-color': '#16a34a'
                                }
                            },
                            {
                                selector: 'node[type="Event"]',
                                style: {
                                    'background-color': '#9333ea',
                                    'border-color': '#6b21a8',
                                    'text-background-color': '#9333ea'
                                }
                            },
                            {
                                selector: 'edge',
                                style: {
                                    'width': 3,
                                    'line-color': '#64748b',
                                    'target-arrow-color': '#64748b',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'label': 'data(label)',
                                    'font-size': 10,
                                    'text-rotation': 'autorotate',
                                    'text-margin-y': -10
                                }
                            },
                            {
                                selector: 'edge[fact]',
                                style: {
                                    'label': 'data(fact)',
                                    'text-wrap': 'wrap',
                                    'text-max-width': '200px',
                                    'font-size': 12,
                                    'color': '#1f2937',
                                    'text-background-color': '#ffffff',
                                    'text-background-opacity': 0.9,
                                    'text-background-padding': 4,
                                    'text-border-color': '#d1d5db',
                                    'text-border-width': 1,
                                    'text-border-opacity': 1,
                                    'line-color': '#3b82f6',
                                    'target-arrow-color': '#3b82f6',
                                    'width': 4
                                }
                            },
                            {
                                selector: 'node:active',
                                style: {
                                    'overlay-opacity': 0
                                }
                            }
                        ],
                        layout: {
                            name: 'circle',
                            fit: true,
                            padding: 50
                        },
                        minZoom: 0.1,
                        maxZoom: 5,
                        wheelSensitivity: 0.2  // Reduce scroll wheel sensitivity
                    });
                        
                        // Add event handlers
                        cy.on('tap', 'node', (evt: any) => {
                            const node = evt.target;
                            setSelectedNode({
                                id: node.id(),
                                label: node.data('label'),
                                type: node.data('type'),
                                properties: node.data()
                            });
                        });
                        
                        cy.on('mouseover', 'node', (evt: any) => {
                            const node = evt.target;
                            node.style({
                                'width': 50,
                                'height': 50,
                                'z-index': 999
                            });
                        });
                        
                        cy.on('mouseout', 'node', (evt: any) => {
                            const node = evt.target;
                            node.style({
                                'width': 40,
                                'height': 40,
                                'z-index': 1
                            });
                        });
                        
                        cyRef.current = cy;
                        
                        // Force resize to ensure proper rendering
                        setTimeout(() => {
                            cy.resize();
                            cy.center();
                            cy.fit();
                        }, 100);
                        
                        setLoading(false);
                    } else if (!cyRef.current) {
                        // Retry after a short delay
                        setTimeout(waitForContainer, 100);
                    }
                };
                
                waitForContainer();
            } catch (err) {
                console.error('Error loading graph:', err);
                setError(err instanceof Error ? err.message : 'Unknown error');
                setLoading(false);
            }
        };

        fetchAndRenderGraph();
        
        // Handle window resize
        const handleResize = () => {
            if (cyRef.current) {
                cyRef.current.resize();
            }
        };
        
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            if (cyRef.current) {
                cyRef.current.destroy();
                cyRef.current = null;
            }
        };
    }, []);
    
    // Control functions
    const handleZoomIn = () => {
        if (cyRef.current) {
            const currentZoom = cyRef.current.zoom();
            cyRef.current.zoom({
                level: currentZoom * 1.1,  // Reduced from 1.2
                renderedPosition: { x: cyRef.current.width() / 2, y: cyRef.current.height() / 2 }
            });
        }
    };
    
    const handleZoomOut = () => {
        if (cyRef.current) {
            const currentZoom = cyRef.current.zoom();
            cyRef.current.zoom({
                level: currentZoom * 0.9,  // Increased from 0.8
                renderedPosition: { x: cyRef.current.width() / 2, y: cyRef.current.height() / 2 }
            });
        }
    };
    
    const handleFit = () => {
        if (cyRef.current) {
            cyRef.current.fit(undefined, 50);
        }
    };
    
    const handleLayoutChange = () => {
        if (cyRef.current) {
            const newLayout = currentLayout === 'circle' ? 'fcose' : 'circle';
            setCurrentLayout(newLayout as 'circle' | 'cose');
            
            const layoutOptions = newLayout === 'fcose' ? {
                name: 'fcose',
                animate: true,
                animationDuration: 1000,
                fit: true,
                padding: 50,
                nodeRepulsion: 4500,
                idealEdgeLength: 50,
                edgeElasticity: 0.45,
                nestingFactor: 0.1,
                gravity: 0.25,
                numIter: 2500,
                tile: true,
                randomize: false
            } : {
                name: 'circle',
                fit: true,
                padding: 50,
                animate: true,
                animationDuration: 500
            };
            
            cyRef.current.layout(layoutOptions).run();
        }
    };

    return (
        <div className="graph-visualization">
            <div className="graph-header">
                <h2>Knowledge Graph</h2>
                <div className="graph-controls">
                    <button onClick={handleZoomIn} title="Zoom In">🔍+</button>
                    <button onClick={handleZoomOut} title="Zoom Out">🔍-</button>
                    <button onClick={handleFit} title="Fit to Screen">⟲</button>
                    <button onClick={handleLayoutChange} title="Toggle Layout">
                        {currentLayout === 'circle' ? '○ Circle' : '⚡ Force'}
                    </button>
                </div>
            </div>
            <div className="graph-container" ref={containerRef}>
                {loading && (
                    <div className="loading-overlay" style={{ background: 'rgba(255, 255, 255, 0.9)' }}>
                        <div className="loading-spinner"></div>
                        <p>Loading graph data...</p>
                    </div>
                )}
                {error && (
                    <div className="error-overlay">
                        <p>Error: {error}</p>
                    </div>
                )}
            </div>
            {selectedNode && (
                <div className="node-details">
                    <h3>{selectedNode.label}</h3>
                    <p><strong>Type:</strong> {selectedNode.type}</p>
                    <p><strong>ID:</strong> {selectedNode.id}</p>
                    {Object.entries(selectedNode.properties).map(([key, value]) => {
                        if (key !== 'id' && key !== 'label' && key !== 'type' && value) {
                            return (
                                <p key={key}>
                                    <strong>{key}:</strong> {String(value)}
                                </p>
                            );
                        }
                        return null;
                    })}
                    <button onClick={() => setSelectedNode(null)}>Close</button>
                </div>
            )}
        </div>
    );
};