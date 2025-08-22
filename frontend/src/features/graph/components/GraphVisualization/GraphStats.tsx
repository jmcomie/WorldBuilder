import type { GraphStatsProps } from './types';
import './GraphStats.css';

export const GraphStats = ({ stats, loading }: GraphStatsProps) => {
    if (loading) {
        return (
            <div className="graph-stats loading">
                <div className="stat-item">Loading statistics...</div>
            </div>
        );
    }

    if (!stats) {
        return null;
    }

    return (
        <div className="graph-stats">
            <div className="stat-item">
                <span className="stat-label">Nodes:</span>
                <span className="stat-value">{stats.nodeCount.toLocaleString()}</span>
            </div>
            <div className="stat-item">
                <span className="stat-label">Edges:</span>
                <span className="stat-value">{stats.edgeCount.toLocaleString()}</span>
            </div>
            <div className="stat-item">
                <span className="stat-label">Node Types:</span>
                <span className="stat-value">{stats.nodeTypes.length}</span>
            </div>
            <div className="stat-item">
                <span className="stat-label">Edge Types:</span>
                <span className="stat-value">{stats.edgeTypes.length}</span>
            </div>
        </div>
    );
};