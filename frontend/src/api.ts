const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  async testConnection() {
    const response = await fetch(`${API_URL}/test-neo4j`);
    return response.json();
  }
};