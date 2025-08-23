import { useState, useEffect, createContext, useContext } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import Navigation from './components/organisms/Navigation/Navigation';
import Overlay from './components/organisms/Overlay/Overlay';
import Help from './pages/Help';
import Settings from './pages/Settings';
import { setupApiClient, rootHealthGet } from './shared/api';
import './App.css';

interface AppContextType {
  openSettings: () => void;
  openHelp: () => void;
}

const AppContext = createContext<AppContextType | null>(null);

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within AppProvider');
  return context;
};

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('loading...');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const location = useLocation();
  const navigate = useNavigate();
  const isNavigationCompact = location.pathname !== '/';

  useEffect(() => {
    // Initialize API client
    setupApiClient();

    // Check backend health
    rootHealthGet()
      .then((response) => {
        if (response.data) {
          setBackendStatus('connected');
        } else if (response.error) {
          setBackendStatus('error');
        }
      })
      .catch(() => setBackendStatus('error'));
  }, []);

  const handleNavigate = (
    view: 'write' | 'graph' | 'play' | 'settings' | 'help'
  ) => {
    if (view === 'settings') {
      setIsSettingsOpen(true);
    } else if (view === 'help') {
      setIsHelpOpen(true);
    } else {
      navigate(`/${view}`);
    }
  };

  const StatusDisplay = () => (
    <div className="status-container">
      <p>Backend Status: {backendStatus}</p>
    </div>
  );

  return (
    <AppContext.Provider
      value={{
        openSettings: () => setIsSettingsOpen(true),
        openHelp: () => setIsHelpOpen(true),
      }}
    >
      <div className={`app ${isNavigationCompact ? 'app-compact' : ''}`}>
        <header className="app-header">
          {isNavigationCompact && (
            <h1
              className="animated-title title-compact"
              onClick={() => navigate('/')}
            >
              worldbuilder
            </h1>
          )}
          {isNavigationCompact && (
            <>
              <Navigation
                isCompact={isNavigationCompact}
                onNavigate={handleNavigate}
                variant="main"
              />
              <Navigation
                isCompact={isNavigationCompact}
                onNavigate={handleNavigate}
                variant="utility"
              />
            </>
          )}
        </header>

        <main className="app-main">
          <Outlet context={{ backendStatus }} />
        </main>

        {location.pathname === '/' && (
          <footer className="app-footer">
            <StatusDisplay />
          </footer>
        )}

        <Overlay
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          title="Settings"
          className="overlay-settings"
        >
          <Settings />
        </Overlay>

        <Overlay
          isOpen={isHelpOpen}
          onClose={() => setIsHelpOpen(false)}
          title="Help"
        >
          <Help />
        </Overlay>
      </div>
    </AppContext.Provider>
  );
}

export default App;
