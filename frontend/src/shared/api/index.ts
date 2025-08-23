/**
 * API client exports for transition period
 * 
 * During migration:
 * - Use `api` for existing manual client (throws errors)
 * - Use `sdk` for new generated client ({data, error} pattern)
 * 
 * After migration complete, this file will only export the SDK
 */

// Export existing manual client
export * from './client';
export { api } from './client';

// Export generated SDK once available
// Note: These exports will be active after running `pnpm api:generate`
// export * as sdk from './sdk';
// export { client as sdkClient } from './sdk/client';