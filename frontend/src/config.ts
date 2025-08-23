/**
 * Application configuration
 * Centralized configuration for the frontend application
 */

// Backend API configuration
export const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Export configuration object for SDK and other services
export const config = {
  api: {
    baseUrl: API_BASE_URL,
  },
} as const;

export default config;
