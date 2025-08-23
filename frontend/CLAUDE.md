# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Worldbuilder Frontend - A React-based web application for worldbuilding with knowledge graph visualization and AI-assisted content creation.

## Tech Stack

- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite 7
- **UI Libraries**: 
  - Material-UI (@mui/material) - Primary UI component library
  - @assistant-ui/react - AI assistant integration (prepared for future chat features)
- **Routing**: React Router v7 (hash router)
- **State Management**: React hooks and Context API

## Development Commands

```bash
# Install dependencies
pnpm install

# Run development server (port 5173, proxied to 3000 in Docker)
pnpm run dev

# Build for production
pnpm run build

# Lint code
pnpm run lint

# Preview production build
pnpm run preview
```

## Architecture

### Routing Structure

Routes defined in `src/router.tsx` using `createHashRouter`:
- `#/` - Home view with status display
- `#/write` - Writing interface with three modes
- `#/graph` - Knowledge graph visualization  
- `#/play` - Game interface (placeholder)
- Settings and Help are overlays, not routes

**Note**: The app uses hash-based routing (URLs with `#/`) for compatibility with static file hosting and to avoid server configuration requirements.

### View Components

**Main Views** (`src/views/`):
- `Home` - Landing page with navigation and status display
- `Write` - Container for writing modes
  - `Episode` mode - Create knowledge graph episodes with draft saving
  - `Ideation` mode - AI-powered brainstorming interface (mock implementation)
  - `Ontology` mode - Define world concepts and relationships
- `Graph` - Interactive knowledge graph visualization
- `Play` - Placeholder for future game features

**Overlay Views**:
- `Settings` - Multi-pane configuration (General, Appearance, API Keys, MCP Servers, Advanced)
- `Help` - Documentation and guides

### Backend API Integration

API client uses auto-generated SDK from OpenAPI spec.

**Configuration**:
- SDK setup in `src/shared/api/setup.ts`
- Backend URL from `src/config.ts`
- Initialized in App.tsx on mount

**Available Endpoints**:
- `GET /health` - Backend health check

**Temporarily Disabled Features**:
- Episode creation - UI preserved but submission disabled
- Graph visualization - Placeholder displayed

### Status Monitoring

The app performs status check on mount:
1. Backend health check via `/health` endpoint
2. Status displayed in App component
3. State tracked: `backendStatus`

### Key Component Patterns

**AppContext** (App.tsx:15-21):
- Provides global functions: `openSettings()`, `openHelp()`
- Used for triggering overlay modals from any component

**Draft Saving** (Episode mode):
- Auto-saves to localStorage on content change
- Restores draft on component mount
- Feature temporarily disabled with "Coming Soon" message

**MCP Server Configuration** (McpServersPane.tsx):
- UI for configuring Model Context Protocol servers
- Stores server configurations with command, args, and environment variables
- Example configurations provided for common servers

### Material-UI Usage

Primary usage in Write modes and Settings:
- Form components: TextField, Button, Alert
- Layout: Box, Container, Paper, Stack
- Icons: @mui/icons-material throughout
- Theme: Default MUI theme (no custom theme provider)

### State Management Patterns

- **Local component state**: useState for UI state
- **Global app state**: Context API (AppContext)
- **Persistent state**: localStorage for drafts and preferences
- **Server state**: Direct API calls with loading/error states

## Important Implementation Notes

### Navigation Component
- Dual variant system: "main" (Write, Graph, Play) and "utility" (Settings, Help)
- Compact mode triggered when not on home page
- Animated title acts as home navigation link

### Graph Visualization
- Currently displays placeholder message
- Feature temporarily unavailable pending reimplementation

### Write Mode Architecture
- Mode selector switches between Episode, Ideation, and Ontology
- Each mode is a separate component with independent state
- Episode mode integrates with backend Graphiti system
- Ideation mode currently uses mock responses (prepared for LLM integration)

## Development Considerations

- **CORS**: Backend must accept requests from `http://localhost:3000` (Docker) and `http://localhost:5173` (dev)
- **Environment Variables**: API URL configurable via `VITE_API_URL`
- **TypeScript**: Strict mode enabled, all components fully typed
- **Hot Reload**: Vite HMR enabled for rapid development
- **API Architecture**: Clean separation between generated SDK and manual setup