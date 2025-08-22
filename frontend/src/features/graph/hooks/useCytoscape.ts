import cytoscape from 'cytoscape';
import type { Core, ElementDefinition } from 'cytoscape';
import fcose from 'cytoscape-fcose';
import { useEffect, useRef, useState } from 'react';
import {
  cytoscapeStylesheet,
  cytoscapeLayoutOptions,
} from '../components/GraphVisualization/cytoscape-config';
import type { LayoutType } from '../components/GraphVisualization/types';

// Register the fcose layout
cytoscape.use(fcose);

interface UseCytoscapeOptions {
  elements: ElementDefinition[];
  layout?: LayoutType;
  style?: cytoscape.Stylesheet[];
}

export const useCytoscape = (
  containerRef: React.RefObject<HTMLDivElement>,
  options: UseCytoscapeOptions
) => {
  const [cy, setCy] = useState<Core | null>(null);
  const [loading, setLoading] = useState(true);
  const layoutRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    setLoading(true);

    try {
      console.log('Initializing Cytoscape with elements:', options.elements);

      // Initialize Cytoscape
      const cytoscapeInstance = cytoscape({
        container: containerRef.current,
        elements: options.elements,
        style: options.style || cytoscapeStylesheet,
        layout: { name: 'preset' },
        // wheelSensitivity: 0.2, // Removed to use default value
        minZoom: 0.1,
        maxZoom: 5,
        boxSelectionEnabled: true,
        selectionType: 'single',
        autounselectify: false,
      });

      // Apply layout with error handling
      try {
        const layoutType = options.layout || 'force-directed';
        const layoutOptions =
          layoutType === 'force-directed'
            ? { name: 'circle', fit: true, padding: 50 } // Use circle as fallback for now
            : cytoscapeLayoutOptions[layoutType];

        layoutRef.current = cytoscapeInstance.layout(layoutOptions);

        layoutRef.current.run();
        layoutRef.current.on('layoutstop', () => {
          setLoading(false);
        });
      } catch (layoutError) {
        console.error('Error applying layout:', layoutError);
        // Fallback to preset layout
        cytoscapeInstance.layout({ name: 'preset' }).run();
        setLoading(false);
      }

      // Set up event handlers
      cytoscapeInstance.on('tap', 'node', (evt) => {
        const node = evt.target;
        console.log('Node clicked:', node.data());
      });

      cytoscapeInstance.on('tap', 'edge', (evt) => {
        const edge = evt.target;
        console.log('Edge clicked:', edge.data());
      });

      // Hover effects
      cytoscapeInstance.on('mouseover', 'node', (evt) => {
        const node = evt.target;
        node.addClass('highlighted');
        node.neighborhood().addClass('highlighted');
      });

      cytoscapeInstance.on('mouseout', 'node', (evt) => {
        const node = evt.target;
        node.removeClass('highlighted');
        node.neighborhood().removeClass('highlighted');
      });

      setCy(cytoscapeInstance);

      // Cleanup
      return () => {
        cytoscapeInstance.destroy();
      };
    } catch (error) {
      console.error('Error initializing Cytoscape:', error);
      setLoading(false);
    }
  }, [containerRef]); // Only depend on containerRef to avoid recreating on every render

  // Update elements when they change
  useEffect(() => {
    if (!cy || options.elements.length === 0) return;

    cy.batch(() => {
      cy.elements().remove();
      cy.add(options.elements);
    });

    // Re-run layout
    const layoutOptions =
      cytoscapeLayoutOptions[options.layout || 'force-directed'];
    layoutRef.current = cy.layout(layoutOptions);
    layoutRef.current.run();
  }, [cy, options.elements, options.layout]);

  const runLayout = (layoutType: LayoutType) => {
    if (!cy) return;

    // Stop current layout if running
    if (layoutRef.current) {
      layoutRef.current.stop();
    }

    const layoutOptions = cytoscapeLayoutOptions[layoutType];
    layoutRef.current = cy.layout(layoutOptions);
    layoutRef.current.run();
  };

  const fit = () => {
    if (!cy) return;
    cy.fit(undefined, 50);
  };

  const reset = () => {
    if (!cy) return;
    cy.reset();
    fit();
  };

  const zoomIn = () => {
    if (!cy) return;
    cy.zoom(cy.zoom() * 1.2);
  };

  const zoomOut = () => {
    if (!cy) return;
    cy.zoom(cy.zoom() * 0.8);
  };

  const exportImage = (_format: 'png' | 'jpg' = 'png') => {
    if (!cy) return null;
    return cy.png({ full: true, scale: 2 });
  };

  const getSelectedElements = () => {
    if (!cy) return { nodes: [], edges: [] };
    return {
      nodes: cy.$('node:selected').toArray(),
      edges: cy.$('edge:selected').toArray(),
    };
  };

  const highlightElements = (nodeIds: string[], edgeIds: string[]) => {
    if (!cy) return;

    cy.batch(() => {
      // Reset all elements
      cy.elements().removeClass('highlighted faded');

      // If nothing to highlight, return
      if (nodeIds.length === 0 && edgeIds.length === 0) return;

      // Fade all elements
      cy.elements().addClass('faded');

      // Highlight specified nodes and their edges
      nodeIds.forEach((id) => {
        const node = cy.$(`#${id}`);
        node.removeClass('faded');
        node.connectedEdges().removeClass('faded');
      });

      // Highlight specified edges and their connected nodes
      edgeIds.forEach((id) => {
        const edge = cy.$(`#${id}`);
        edge.removeClass('faded');
        edge.connectedNodes().removeClass('faded');
      });
    });
  };

  return {
    cy,
    loading,
    runLayout,
    fit,
    reset,
    zoomIn,
    zoomOut,
    exportImage,
    getSelectedElements,
    highlightElements,
  };
};
