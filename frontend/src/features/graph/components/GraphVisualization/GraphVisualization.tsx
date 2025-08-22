import { useRef, useState, useEffect, useCallback } from 'react';
import { api } from '../../../../shared/api/client';
import { useCytoscape } from '../../hooks/useCytoscape';
import { GraphControls } from './GraphControls';
import { GraphStats } from './GraphStats';
import type { LayoutType, GraphStats as GraphStatsType } from './types';
import './GraphVisualization.css';

export const GraphVisualization = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [elements, setElements] = useState({ nodes: [], edges: [] });
  const [layout, setLayout] = useState<LayoutType>('force-directed');
  const [stats, setStats] = useState<GraphStatsType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  console.log('GraphVisualization rendering, elements:', elements);

  const {
    cy,
    loading: cytoscapeLoading,
    runLayout,
    fit,
    reset,
    zoomIn,
    zoomOut,
    exportImage,
  } = useCytoscape(containerRef, {
    elements: [...elements.nodes, ...elements.edges],
    layout,
  });

  // Fetch graph data
  useEffect(() => {
    const fetchGraphData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await api.getGraph({ limit: 1000 });
        if (response.success) {
          setElements(response.elements);
        }
      } catch (err) {
        console.error('Error fetching graph data:', err);
        setError(
          err instanceof Error ? err.message : 'Failed to load graph data'
        );
      } finally {
        setLoading(false);
      }
    };

    fetchGraphData();
  }, []);

  // Fetch graph statistics
  useEffect(() => {
    const fetchStats = async () => {
      setStatsLoading(true);
      try {
        const response = await api.getGraphStats();
        if (response.success) {
          setStats({
            nodeCount: response.nodeCount,
            edgeCount: response.edgeCount,
            nodeTypes: response.nodeTypes,
            nodeTypeCounts: response.nodeTypeCounts,
            edgeTypes: response.edgeTypes,
          });
        }
      } catch (err) {
        console.error('Error fetching graph stats:', err);
      } finally {
        setStatsLoading(false);
      }
    };

    fetchStats();
  }, []);

  const handleLayoutChange = useCallback(
    (newLayout: LayoutType) => {
      setLayout(newLayout);
      if (cy) {
        runLayout(newLayout);
      }
    },
    [cy, runLayout]
  );

  const handleExport = useCallback(() => {
    if (!cy) return;

    const base64Image = exportImage('png');
    if (base64Image) {
      const link = document.createElement('a');
      link.href = base64Image;
      link.download = `worldbuilder-graph-${new Date().getTime()}.png`;
      link.click();
    }
  }, [cy, exportImage]);

  try {
    return (
      <div className="graph-visualization">
        <div className="graph-header">
          <GraphControls
            cy={cy}
            layout={layout}
            onLayoutChange={handleLayoutChange}
            onZoomIn={zoomIn}
            onZoomOut={zoomOut}
            onFit={fit}
            onReset={reset}
            onExport={handleExport}
          />
          <GraphStats stats={stats} loading={statsLoading} />
        </div>

        <div className="graph-container" ref={containerRef}>
          {loading && (
            <div className="loading-overlay">
              <div className="loading-spinner"></div>
              <p>Loading graph data...</p>
            </div>
          )}
          {error && (
            <div className="error-overlay">
              <p>Error: {error}</p>
              <button onClick={() => window.location.reload()}>Retry</button>
            </div>
          )}
          {!loading && !error && elements.nodes.length === 0 && (
            <div className="empty-state">
              <p>No graph data available yet.</p>
              <p>Start by adding episodes to build your knowledge graph.</p>
            </div>
          )}
        </div>

        {cytoscapeLoading &&
          !loading &&
          !error &&
          elements.nodes.length > 0 && (
            <div className="layout-loading">
              <div className="loading-spinner small"></div>
              <p>Applying layout...</p>
            </div>
          )}
      </div>
    );
  } catch (err) {
    console.error('GraphVisualization render error:', err);
    return (
      <div className="graph-visualization">
        <div className="error-overlay">
          <p>
            Error rendering graph:{' '}
            {err instanceof Error ? err.message : 'Unknown error'}
          </p>
        </div>
      </div>
    );
  }
};
