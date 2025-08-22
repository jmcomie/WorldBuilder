import { useState } from 'react';
import './PaneStyles.css';

interface McpServer {
  id: string;
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
  enabled: boolean;
}

const McpServersPane = () => {
  const [servers, setServers] = useState<McpServer[]>([
    {
      id: '1',
      name: 'filesystem',
      command: 'npx',
      args: [
        '-y',
        '@modelcontextprotocol/server-filesystem',
        '/Users/username/Documents',
      ],
      env: {},
      enabled: true,
    },
  ]);
  const [editingServer, setEditingServer] = useState<McpServer | null>(null);
  const [isAddingNew, setIsAddingNew] = useState(false);

  const handleAddServer = () => {
    const newServer: McpServer = {
      id: Date.now().toString(),
      name: '',
      command: '',
      args: [],
      env: {},
      enabled: true,
    };
    setEditingServer(newServer);
    setIsAddingNew(true);
  };

  const handleSaveServer = () => {
    if (editingServer) {
      if (isAddingNew) {
        setServers([...servers, editingServer]);
      } else {
        setServers(
          servers.map((s) => (s.id === editingServer.id ? editingServer : s))
        );
      }
      setEditingServer(null);
      setIsAddingNew(false);
    }
  };

  const handleDeleteServer = (id: string) => {
    setServers(servers.filter((s) => s.id !== id));
  };

  const handleCancelEdit = () => {
    setEditingServer(null);
    setIsAddingNew(false);
  };

  const updateEditingServer = (updates: Partial<McpServer>) => {
    if (editingServer) {
      setEditingServer({ ...editingServer, ...updates });
    }
  };

  return (
    <div className="settings-pane">
      <h3 className="settings-pane-title">MCP Servers</h3>

      <div className="settings-alert settings-alert-info">
        <span className="settings-alert-icon">ℹ️</span>
        <p>
          Model Context Protocol (MCP) servers enable AI models to connect to
          external data sources and tools. Configure servers to extend AI
          capabilities with custom integrations.
        </p>
      </div>

      <div className="settings-section">
        <h4 className="settings-section-title">Configured Servers</h4>

        {servers.length === 0 && !editingServer && (
          <p className="settings-description">
            No MCP servers configured. Add a server to get started.
          </p>
        )}

        {servers.map((server) => (
          <div key={server.id} className="settings-item settings-item-vertical">
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div className="settings-label">
                <span className="settings-label-text">
                  {server.name || 'Unnamed Server'}
                </span>
                <span className="settings-label-description">
                  {server.command} {server.args.join(' ')}
                </span>
              </div>
              <div
                style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}
              >
                <input
                  type="checkbox"
                  className="settings-checkbox"
                  checked={server.enabled}
                  onChange={(e) => {
                    const updatedServers = servers.map((s) =>
                      s.id === server.id
                        ? { ...s, enabled: e.target.checked }
                        : s
                    );
                    setServers(updatedServers);
                  }}
                />
                <button
                  className="settings-button settings-button-secondary"
                  onClick={() => {
                    setEditingServer(server);
                    setIsAddingNew(false);
                  }}
                >
                  Edit
                </button>
                <button
                  className="settings-button settings-button-danger"
                  onClick={() => handleDeleteServer(server.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}

        {editingServer && (
          <div
            className="settings-section"
            style={{
              marginTop: '2rem',
              padding: '1.5rem',
              background: '#f9fafb',
              borderRadius: '8px',
            }}
          >
            <h4 className="settings-section-title">
              {isAddingNew ? 'Add New Server' : 'Edit Server'}
            </h4>

            <div className="settings-item settings-item-vertical">
              <label className="settings-label">
                <span className="settings-label-text">Server Name</span>
                <span className="settings-label-description">
                  A unique identifier for this server
                </span>
              </label>
              <input
                type="text"
                className="settings-input"
                value={editingServer.name}
                onChange={(e) => updateEditingServer({ name: e.target.value })}
                placeholder="e.g., filesystem, github, database"
              />
            </div>

            <div className="settings-item settings-item-vertical">
              <label className="settings-label">
                <span className="settings-label-text">Command</span>
                <span className="settings-label-description">
                  The executable to run (e.g., npx, python, node)
                </span>
              </label>
              <input
                type="text"
                className="settings-input settings-input-monospace"
                value={editingServer.command}
                onChange={(e) =>
                  updateEditingServer({ command: e.target.value })
                }
                placeholder="e.g., npx, python, /usr/local/bin/mcp-server"
              />
            </div>

            <div className="settings-item settings-item-vertical">
              <label className="settings-label">
                <span className="settings-label-text">Arguments</span>
                <span className="settings-label-description">
                  Command-line arguments (one per line)
                </span>
              </label>
              <textarea
                className="settings-input settings-input-monospace"
                rows={4}
                value={editingServer.args.join('\n')}
                onChange={(e) => {
                  const args = e.target.value
                    .split('\n')
                    .filter((arg) => arg.trim());
                  updateEditingServer({ args });
                }}
                placeholder="-y&#10;@modelcontextprotocol/server-filesystem&#10;/path/to/directory"
              />
            </div>

            <div className="settings-item settings-item-vertical">
              <label className="settings-label">
                <span className="settings-label-text">
                  Environment Variables
                </span>
                <span className="settings-label-description">
                  Key=value pairs (one per line)
                </span>
              </label>
              <textarea
                className="settings-input settings-input-monospace"
                rows={3}
                value={Object.entries(editingServer.env)
                  .map(([k, v]) => `${k}=${v}`)
                  .join('\n')}
                onChange={(e) => {
                  const env: Record<string, string> = {};
                  e.target.value.split('\n').forEach((line) => {
                    const [key, ...valueParts] = line.split('=');
                    if (key?.trim()) {
                      env[key.trim()] = valueParts.join('=').trim();
                    }
                  });
                  updateEditingServer({ env });
                }}
                placeholder="API_KEY=your-api-key&#10;BASE_URL=https://api.example.com"
              />
            </div>

            <div className="settings-actions">
              <button
                className="settings-button settings-button-primary"
                onClick={handleSaveServer}
                disabled={!editingServer.name || !editingServer.command}
              >
                {isAddingNew ? 'Add Server' : 'Save Changes'}
              </button>
              <button
                className="settings-button settings-button-secondary"
                onClick={handleCancelEdit}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {!editingServer && (
          <button
            className="settings-button settings-button-primary"
            onClick={handleAddServer}
            style={{ marginTop: '1rem' }}
          >
            Add MCP Server
          </button>
        )}
      </div>

      <div className="settings-section">
        <h4 className="settings-section-title">Common Server Examples</h4>
        <div className="settings-code">
          # Filesystem Server npx -y @modelcontextprotocol/server-filesystem
          /path/to/directory # GitHub Server npx -y
          @modelcontextprotocol/server-github # Requires:
          GITHUB_PERSONAL_ACCESS_TOKEN environment variable # Brave Search
          Server npx -y @modelcontextprotocol/server-brave-search # Requires:
          BRAVE_API_KEY environment variable # Python Server python
          /path/to/server.py --option value # Node.js Server node
          /path/to/server.js --config /path/to/config.json
        </div>
      </div>
    </div>
  );
};

export default McpServersPane;
