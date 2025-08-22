// import { GraphVisualization } from '../../features/graph/components/GraphVisualization/GraphVisualization';
import { GraphVisualizationSimple } from '../../features/graph/components/GraphVisualization/GraphVisualizationSimple';
import './Graph.css';

const Graph = () => {
  return (
    <div className="view-container graph-view">
      <GraphVisualizationSimple />
    </div>
  );
};

export default Graph;