import './PaneStyles.css';

const GeneralPane = () => {
  return (
    <div className="settings-pane">
      <h3 className="settings-pane-title">General Settings</h3>
      
      <section className="settings-section">
        <h4 className="settings-section-title">Application</h4>
        
        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Auto-save episodes</span>
            <span className="settings-label-description">
              Automatically save your work as you type
            </span>
          </label>
          <input type="checkbox" className="settings-checkbox" defaultChecked />
        </div>
        
        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Auto-save interval</span>
            <span className="settings-label-description">
              How often to save (in seconds)
            </span>
          </label>
          <input 
            type="number" 
            className="settings-input" 
            defaultValue={30} 
            min={5} 
            max={300} 
          />
        </div>
      </section>
      
      <section className="settings-section">
        <h4 className="settings-section-title">Graph Visualization</h4>
        
        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Default layout</span>
            <span className="settings-label-description">
              Initial graph layout algorithm
            </span>
          </label>
          <select className="settings-select" defaultValue="force">
            <option value="force">Force-directed</option>
            <option value="circle">Circle</option>
            <option value="grid">Grid</option>
            <option value="concentric">Concentric</option>
          </select>
        </div>
        
        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Animation duration</span>
            <span className="settings-label-description">
              Graph animation speed (ms)
            </span>
          </label>
          <input 
            type="range" 
            className="settings-range" 
            defaultValue={500} 
            min={0} 
            max={2000} 
            step={100}
          />
        </div>
      </section>
    </div>
  );
};

export default GeneralPane;