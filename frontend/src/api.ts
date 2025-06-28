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
    
    // Provide more specific error messages based on status code
    if (response.status === 429) {
      throw new Error('Rate limit exceeded. Please wait a moment and try again.');
    } else if (response.status === 401) {
      throw new Error('OpenAI API authentication failed. Please check the API key configuration.');
    }
    
    throw new Error(error.detail || 'Failed to create episode');
  }

  return response.json();
}