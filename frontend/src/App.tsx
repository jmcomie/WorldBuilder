import { useState, useEffect } from 'react'
import { api } from './api'
import './App.css'

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('loading...')
  const [neo4jStatus, setNeo4jStatus] = useState<string>('loading...')

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

  return (
    <>
      <h1>worldbuilder</h1>
      <div>
        <p>Backend Status: {backendStatus}</p>
        <p>Database Status: {neo4jStatus}</p>
      </div>
    </>
  )
}

export default App