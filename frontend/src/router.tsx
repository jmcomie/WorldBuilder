import { createHashRouter, Navigate } from 'react-router-dom';
import App from './App';
import GraphView from './pages/Graph';
import HomeView from './pages/Home';
import PlayView from './pages/Play';
import WriteView from './pages/Write';

export const router = createHashRouter([
  {
    path: '/',
    element: <App />,
    children: [
      {
        index: true,
        element: <HomeView />,
      },
      {
        path: 'write',
        element: <WriteView />,
      },
      {
        path: 'graph',
        element: <GraphView />,
      },
      {
        path: 'play',
        element: <PlayView />,
      },
      {
        path: 'settings',
        element: <Navigate to="/" replace />,
      },
      {
        path: 'help',
        element: <Navigate to="/" replace />,
      },
      {
        path: '*',
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);
