import React from 'react';
// import { GraphVisualization } from '../../components/GraphVisualization/GraphVisualization';
import { GraphVisualizationSimple } from '../../components/GraphVisualization/GraphVisualizationSimple';
import './Graph.css';

const Graph: React.FC = () => {
  return (
    <div className="view-container graph-view">
      <GraphVisualizationSimple />
    </div>
  );
};

export default Graph;