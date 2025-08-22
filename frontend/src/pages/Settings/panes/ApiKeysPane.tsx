import { useState } from 'react';
import './PaneStyles.css';

const ApiKeysPane = () => {
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [openAIKey, setOpenAIKey] = useState('');

  const maskApiKey = (key: string) => {
    if (!key) return '';
    if (key.length <= 8) return '•'.repeat(key.length);
    return key.slice(0, 4) + '•'.repeat(key.length - 8) + key.slice(-4);
  };

  return (
    <div className="settings-pane">
      <h3 className="settings-pane-title">API Keys</h3>
      
      <section className="settings-section">
        <div className="settings-alert settings-alert-info">
          <span className="settings-alert-icon">ℹ️</span>
          <p>API keys are stored locally and never sent to our servers.</p>
        </div>
        
        <div className="settings-item settings-item-vertical">
          <label className="settings-label">
            <span className="settings-label-text">OpenAI API Key</span>
            <span className="settings-label-description">
              Required for AI-powered features like entity extraction
            </span>
          </label>
          <div className="settings-input-group">
            <input
              type={showOpenAIKey ? 'text' : 'password'}
              className="settings-input settings-input-monospace"
              placeholder="sk-..."
              value={openAIKey}
              onChange={(e) => setOpenAIKey(e.target.value)}
            />
            <button 
              className="settings-button settings-button-secondary"
              onClick={() => setShowOpenAIKey(!showOpenAIKey)}
            >
              {showOpenAIKey ? '🙈' : '👁️'}
            </button>
          </div>
        </div>
        
        <div className="settings-item">
          <label className="settings-label">
            <span className="settings-label-text">Model</span>
            <span className="settings-label-description">
              OpenAI model to use for processing
            </span>
          </label>
          <select className="settings-select" defaultValue="gpt-4">
            <option value="gpt-4">GPT-4</option>
            <option value="gpt-4-turbo">GPT-4 Turbo</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          </select>
        </div>
      </section>
      
      <section className="settings-section">
        <h4 className="settings-section-title">Other Services</h4>
        <p className="settings-description">
          Additional API integrations will be available here in future updates.
        </p>
      </section>
      
      <div className="settings-actions">
        <button className="settings-button settings-button-primary">
          Save API Keys
        </button>
      </div>
    </div>
  );
};

export default ApiKeysPane;