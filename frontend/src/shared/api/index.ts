/**
 * API Client Exports
 * Centralized exports for all API-related functionality
 */

// Export SDK types and functions
export * from './sdk';
export { client } from './sdk/client.gen';

// Export setup function
export { setupApiClient } from './setup';
