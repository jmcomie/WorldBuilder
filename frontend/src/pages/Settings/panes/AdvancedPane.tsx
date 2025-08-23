import './PaneStyles.css';

const AdvancedPane = () => {
  return (
    <div className="settings-pane">
      <h3 className="settings-pane-title">Advanced Settings</h3>

      <section className="settings-section">
        <h4 className="settings-section-title">Performance</h4>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Cache duration</span>
            <span className="settings-label-description">
              How long to cache API responses (minutes)
            </span>
          </label>
          <input
            type="number"
            className="settings-input"
            defaultValue={5}
            min={1}
            max={60}
            step={1}
          />
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
