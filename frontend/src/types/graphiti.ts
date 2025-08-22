export type EpisodeType = 'text' | 'json' | 'message';

export interface EpisodeRequest {
  name: string;
  content: string;
  source_description?: string;
}

export interface EpisodeResponse {
  success: boolean;
  message: string;
  episode_id?: string;
}

export interface ApiError {
  detail: string;
}
