import { useState } from 'react';
import './PaneStyles.css';

const AppearancePane = () => {
  const [accentColor, setAccentColor] = useState('#2563eb');

  return (
    <div className="settings-pane">
      <h3 className="settings-pane-title">Appearance</h3>

      <section className="settings-section">
        <h4 className="settings-section-title">Colors</h4>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Accent color</span>
            <span className="settings-label-description">
              Primary color for buttons and highlights
            </span>
          </label>
          <div className="settings-color-picker">
            <input
              type="color"
              className="settings-color-input"
              value={accentColor}
              onChange={(e) => setAccentColor(e.target.value)}
            />
            <span className="settings-color-value">{accentColor}</span>
          </div>
        </div>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Graph node colors</span>
            <span className="settings-label-description">
              Use custom colors for different node types
            </span>
          </label>
          <input type="checkbox" className="settings-checkbox" defaultChecked />
        </div>
      </section>

      <section className="settings-section">
        <h4 className="settings-section-title">Display</h4>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Animations</span>
            <span className="settings-label-description">
              Enable UI animations and transitions
            </span>
          </label>
          <input type="checkbox" className="settings-checkbox" defaultChecked />
        </div>

        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Compact mode</span>
            <span className="settings-label-description">
              Start in compact navigation mode
            </span>
          </label>
          <input type="checkbox" className="settings-checkbox" />
        </div>
      </section>
    </div>
  );
};

export default AppearancePane;
