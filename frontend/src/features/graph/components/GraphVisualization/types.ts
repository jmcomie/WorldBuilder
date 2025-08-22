import type { Core, ElementDefinition, NodeSingular, EdgeSingular } from 'cytoscape';

export interface GraphStats {
    nodeCount: number;
    edgeCount: number;
    nodeTypes: string[];
    nodeTypeCounts: Record<string, number>;
    edgeTypes: string[];
}

export interface GraphElements {
    nodes: ElementDefinition[];
    edges: ElementDefinition[];
}

export interface GraphResponse {
    success: boolean;
    elements: GraphElements;
}

export interface GraphStatsResponse {
    success: boolean;
    nodeCount: number;
    edgeCount: number;
    nodeTypes: string[];
    nodeTypeCounts: Record<string, number>;
    edgeTypes: string[];
}

export interface GraphFilter {
    nodeTypes: string[];
    edgeTypes: string[];
    searchQuery: string;
    showIsolatedNodes: boolean;
}

export type LayoutType = 'force-directed' | 'circle' | 'grid' | 'concentric' | 'breadthfirst';

export interface GraphControlsProps {
    cy: Core | null;
    layout: LayoutType;
    onLayoutChange: (layout: LayoutType) => void;
    onZoomIn: () => void;
    onZoomOut: () => void;
    onFit: () => void;
    onReset: () => void;
    onExport: () => void;
}

export interface GraphLegendProps {
    nodeTypes: string[];
    nodeTypeCounts: Record<string, number>;
    edgeTypes: string[];
    selectedNodeTypes: string[];
    selectedEdgeTypes: string[];
    onNodeTypeToggle: (nodeType: string) => void;
    onEdgeTypeToggle: (edgeType: string) => void;
}

export interface GraphFiltersProps {
    filter: GraphFilter;
    nodeTypes: string[];
    edgeTypes: string[];
    onFilterChange: (filter: GraphFilter) => void;
}

export interface GraphStatsProps {
    stats: GraphStats | null;
    loading: boolean;
}

export interface NodeData {
    id: string;
    label: string;
    type: string;
    [key: string]: any;
}

export interface EdgeData {
    id: string;
    source: string;
    target: string;
    label: string;
    [key: string]: any;
}

export interface CytoscapeNode extends NodeSingular {
    data(): NodeData;
}

export interface CytoscapeEdge extends EdgeSingular {
    data(): EdgeData;
}