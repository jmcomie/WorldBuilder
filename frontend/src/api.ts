import type { EpisodeRequest, EpisodeResponse } from './types/graphiti';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  async testConnection() {
    const response = await fetch(`${API_URL}/test-neo4j`);
    return response.json();
  }
};

export async function createEpisode(episode: EpisodeRequest): Promise<EpisodeResponse> {
  const response = await fetch(`${API_URL}/episodes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(episode),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create episode');
  }

  return response.json();
}