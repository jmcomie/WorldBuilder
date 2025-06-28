import type { EpisodeRequest, EpisodeResponse } from './types/graphiti';
import type { GraphResponse, GraphStatsResponse } from './components/GraphVisualization/types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  async testConnection() {
    const response = await fetch(`${API_URL}/test-neo4j`);
    return response.json();
  },

  async getGraph(params?: { limit?: number; offset?: number; nodeType?: string }): Promise<GraphResponse> {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());
    if (params?.nodeType) queryParams.append('node_type', params.nodeType);

    const response = await fetch(`${API_URL}/graph?${queryParams}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch graph data');
    }
    return response.json();
  },

  async getGraphWithFacts(params?: { limit?: number; offset?: number }): Promise<GraphResponse> {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());

    const response = await fetch(`${API_URL}/graph/with-facts?${queryParams}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch graph data with facts');
    }
    return response.json();
  },

  async getGraphStats(): Promise<GraphStatsResponse> {
    const response = await fetch(`${API_URL}/graph/stats`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch graph statistics');
    }
    return response.json();
  },

  async getGraphNodes(params?: {
    nodeType?: string;
    limit?: number;
    offset?: number;
    search?: string;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.nodeType) queryParams.append('node_type', params.nodeType);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());
    if (params?.search) queryParams.append('search', params.search);

    const response = await fetch(`${API_URL}/graph/nodes?${queryParams}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch nodes');
    }
    return response.json();
  },

  async getGraphEdges(params?: {
    sourceId?: string;
    targetId?: string;
    edgeType?: string;
    limit?: number;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.sourceId) queryParams.append('source_id', params.sourceId);
    if (params?.targetId) queryParams.append('target_id', params.targetId);
    if (params?.edgeType) queryParams.append('edge_type', params.edgeType);
    if (params?.limit) queryParams.append('limit', params.limit.toString());

    const response = await fetch(`${API_URL}/graph/edges?${queryParams}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch edges');
    }
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