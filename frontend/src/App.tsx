import { useState, useEffect } from 'react'
import { api } from './api'
import Navigation from './components/Navigation/Navigation'
import Overlay from './components/Overlay/Overlay'
import Write from './views/Write'
import Graph from './views/Graph'
import Play from './views/Play'
import Settings from './views/Settings'
import Help from './views/Help'
import './App.css'

type View = 'home' | 'write' | 'graph' | 'play' | 'settings' | 'help'

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('loading...')
  const [neo4jStatus, setNeo4jStatus] = useState<string>('loading...')
  const [currentView, setCurrentView] = useState<View>('home')
  const [isNavigationCompact, setIsNavigationCompact] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isHelpOpen, setIsHelpOpen] = useState(false)

  useEffect(() => {
    // Check backend status
    fetch('http://localhost:8000/')
      .then(res => res.json())
      .then(data => setBackendStatus(data.message))
      .catch(() => setBackendStatus('error'))

    // Check Neo4j connection
    api.testConnection()
      .then(data => setNeo4jStatus(data.message))
      .catch(() => setNeo4jStatus('error'))
  }, [])

  const handleNavigate = (view: 'write' | 'graph' | 'play' | 'settings' | 'help') => {
    if (view === 'settings') {
      setIsSettingsOpen(true)
    } else if (view === 'help') {
      setIsHelpOpen(true)
    } else {
      setCurrentView(view)
      if (view === 'write' || view === 'graph' || view === 'play') {
        setIsNavigationCompact(true)
      }
    }
  }

  const renderView = () => {
    switch (currentView) {
      case 'write':
        return <Write />
      case 'graph':
        return <Graph />
      case 'play':
        return <Play />
      default:
        return (
          <div className="status-container">
            <p>Backend Status: {backendStatus}</p>
            <p>Database Status: {neo4jStatus}</p>
          </div>
        )
    }
  }

  return (
    <div className={`app ${isNavigationCompact ? 'app-compact' : ''}`}>
      <header className="app-header">
        {isNavigationCompact && (
          <h1 
            className="animated-title title-compact"
            onClick={() => {
              setCurrentView('home')
              setIsNavigationCompact(false)
            }}
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
        {!isNavigationCompact && currentView === 'home' ? (
          <div className="home-view">
            <Navigation 
              isCompact={isNavigationCompact} 
              onNavigate={handleNavigate}
            />
            <h1 className="animated-title home-title">worldbuilder</h1>
          </div>
        ) : (
          renderView()
        )}
      </main>
      
      {currentView === 'home' && (
        <footer className="app-footer">
          <div className="status-container">
            <p>Backend Status: {backendStatus}</p>
            <p>Database Status: {neo4jStatus}</p>
          </div>
        </footer>
      )}
      
      <Overlay 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)}
        title="Settings"
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
  )
}

export default App