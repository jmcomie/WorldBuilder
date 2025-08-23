/**
 * Cytoscape Hook (Stubbed)
 * TODO: Reimplement cytoscape functionality
 */

import { RefObject } from 'react';

interface UseCytoscapeOptions {
  elements?: any[];
  layout?: string;
  style?: any[];
}

export const useCytoscape = (
  _containerRef: RefObject<HTMLDivElement>,
  _options: UseCytoscapeOptions
) => {
  // Return stubbed interface
  return {
    cy: null,
    loading: false,
    runLayout: () => {},
    fit: () => {},
    reset: () => {},
    zoomIn: () => {},
    zoomOut: () => {},
    exportImage: () => {},
  };
};

export default useCytoscape;
