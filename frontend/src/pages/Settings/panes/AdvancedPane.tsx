import './PaneStyles.css';

const AdvancedPane = () => {
  return (
    <div className="settings-pane">
      <h3 className="settings-pane-title">Advanced Settings</h3>

      <section className="settings-section">
        <h4 className="settings-section-title">Database</h4>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Neo4j connection</span>
            <span className="settings-label-description">
              Database connection string
            </span>
          </label>
          <input
            type="text"
            className="settings-input settings-input-monospace"
            defaultValue="bolt://localhost:7688"
            disabled
          />
        </div>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Connection timeout</span>
            <span className="settings-label-description">
              Maximum time to wait for database connection (ms)
            </span>
          </label>
          <input
            type="number"
            className="settings-input"
            defaultValue={5000}
            min={1000}
            max={30000}
            step={1000}
          />
        </div>
      </section>

      <section className="settings-section">
        <h4 className="settings-section-title">Performance</h4>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Graph render limit</span>
            <span className="settings-label-description">
              Maximum nodes to display at once
            </span>
          </label>
          <input
            type="number"
            className="settings-input"
            defaultValue={100}
            min={10}
            max={1000}
            step={10}
          />
        </div>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Enable WebGL</span>
            <span className="settings-label-description">
              Use hardware acceleration for graph rendering
            </span>
          </label>
          <input type="checkbox" className="settings-checkbox" defaultChecked />
        </div>
      </section>

      <section className="settings-section">
        <h4 className="settings-section-title">Developer</h4>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Debug mode</span>
            <span className="settings-label-description">
              Show additional logging and debug information
            </span>
          </label>
          <input type="checkbox" className="settings-checkbox" />
        </div>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">API response logging</span>
            <span className="settings-label-description">
              Log all API requests and responses to console
            </span>
          </label>
          <input type="checkbox" className="settings-checkbox" />
        </div>
      </section>

      <section className="settings-section">
        <h4 className="settings-section-title">Data Management</h4>

        <div className="settings-actions">
          <button className="settings-button settings-button-danger">
            Clear Local Storage
          </button>
          <button className="settings-button settings-button-secondary">
            Export Settings
          </button>
          <button className="settings-button settings-button-secondary">
            Import Settings
          </button>
        </div>
      </section>
    </div>
  );
};

export default AdvancedPane;
