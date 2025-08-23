/**
 * API Client Setup
 * Configures the auto-generated SDK client with application settings
 */

import { API_BASE_URL } from '../../config';
import { client } from './sdk/client.gen';

/**
 * Initialize the API client with backend URL and any required interceptors.
 * This should be called once during application initialization.
 */
export function setupApiClient(): void {
  // Configure the SDK client with the backend URL
  client.setConfig({
    baseUrl: API_BASE_URL,
  });

  // Future: Add authentication interceptors here if needed
  // client.interceptors.request.use((request, options) => { ... });
}

export default setupApiClient;
