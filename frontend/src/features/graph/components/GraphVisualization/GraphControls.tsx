import type { GraphControlsProps, LayoutType } from './types';
import './GraphControls.css';

const LAYOUT_OPTIONS: { value: LayoutType; label: string }[] = [
  { value: 'force-directed', label: 'Force Directed' },
  { value: 'circle', label: 'Circle' },
  { value: 'grid', label: 'Grid' },
  { value: 'concentric', label: 'Concentric' },
  { value: 'breadthfirst', label: 'Breadth First' },
];

export const GraphControls = ({
  cy,
  layout,
  onLayoutChange,
  onZoomIn,
  onZoomOut,
  onFit,
  onReset,
  onExport,
}: GraphControlsProps) => {
  return (
    <div className="graph-controls">
      <div className="control-group">
        <label htmlFor="layout-select">Layout:</label>
        <select
          id="layout-select"
          value={layout}
          onChange={(e) => onLayoutChange(e.target.value as LayoutType)}
          className="layout-select"
        >
          {LAYOUT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group zoom-controls">
        <button
          onClick={onZoomIn}
          className="control-button"
          title="Zoom In"
          disabled={!cy}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
            <line x1="11" y1="8" x2="11" y2="14" />
            <line x1="8" y1="11" x2="14" y2="11" />
          </svg>
        </button>
        <button
          onClick={onZoomOut}
          className="control-button"
          title="Zoom Out"
          disabled={!cy}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
            <line x1="8" y1="11" x2="14" y2="11" />
          </svg>
        </button>
        <button
          onClick={onFit}
          className="control-button"
          title="Fit to Screen"
          disabled={!cy}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <polyline points="15 3 21 3 21 9" />
            <polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" />
            <line x1="3" y1="21" x2="10" y2="14" />
          </svg>
        </button>
        <button
          onClick={onReset}
          className="control-button"
          title="Reset View"
          disabled={!cy}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
        </button>
      </div>

      <div className="control-group">
        <button
          onClick={onExport}
          className="control-button export-button"
          title="Export as Image"
          disabled={!cy}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Export
        </button>
      </div>
    </div>
  );
};
