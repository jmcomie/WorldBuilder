import React, { useState } from 'react';
import GeneralPane from './panes/GeneralPane';
import ApiKeysPane from './panes/ApiKeysPane';
import AppearancePane from './panes/AppearancePane';
import AdvancedPane from './panes/AdvancedPane';
import './Settings.css';

type SettingsTab = 'general' | 'api-keys' | 'appearance' | 'advanced';

const Settings: React.FC = () => {
  const [activeTab, setActiveTab] = useState<SettingsTab>('general');

  const tabs = [
    { id: 'general' as SettingsTab, label: 'General' },
    { id: 'api-keys' as SettingsTab, label: 'API Keys' },
    { id: 'appearance' as SettingsTab, label: 'Appearance' },
    { id: 'advanced' as SettingsTab, label: 'Advanced' },
  ];

  const renderActivePane = () => {
    switch (activeTab) {
      case 'general':
        return <GeneralPane />;
      case 'api-keys':
        return <ApiKeysPane />;
      case 'appearance':
        return <AppearancePane />;
      case 'advanced':
        return <AdvancedPane />;
      default:
        return <GeneralPane />;
    }
  };

  return (
    <div className="settings-container">
      <div className="settings-sidebar">
        <div className="settings-tabs-vertical">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`settings-tab-vertical ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      <div className="settings-content">
        {renderActivePane()}
      </div>
    </div>
  );
};

export default Settings;