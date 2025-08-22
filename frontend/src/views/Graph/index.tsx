// import { GraphVisualization } from '../../components/GraphVisualization/GraphVisualization';
import { GraphVisualizationSimple } from '../../components/GraphVisualization/GraphVisualizationSimple';
import './Graph.css';

const Graph = () => {
  return (
    <div className="view-container graph-view">
      <GraphVisualizationSimple />
    </div>
  );
};

export default Graph;