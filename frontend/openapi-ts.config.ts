import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: './openapi.json', // Will be provided later
  output: {
    path: 'src/shared/api/sdk',
    format: 'prettier', // Uses your existing .prettierrc
    lint: 'eslint', // Uses your existing eslint.config.js
  },
  client: '@hey-api/client-fetch', // Native fetch, default client
  plugins: [
    {
      name: '@hey-api/typescript',
      enums: 'javascript', // Better tree-shaking, React DevTools compatibility
    },
    {
      name: '@hey-api/sdk',
      operationId: true, // Use OpenAPI operationIds for method names
      asClass: false, // Functions over classes (React-friendly)
      throwOnError: false, // Returns {data, error} instead of throwing
    },
  ],
});