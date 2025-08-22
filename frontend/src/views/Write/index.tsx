import { useState } from 'react';
import { Box, Container } from '@mui/material';
import ModeSelector from './components/ModeSelector';
import type { WriteMode } from './components/ModeSelector';
import EpisodeMode from './modes/Episode';
import OntologyMode from './modes/Ontology';
import IdeationMode from './modes/Ideation';
import './Write.css';

const Write = () => {
  const [mode, setMode] = useState<WriteMode>('episode');

  const renderMode = () => {
    switch (mode) {
      case 'episode':
        return <EpisodeMode />;
      case 'ontology':
        return <OntologyMode />;
      case 'ideation':
        return <IdeationMode />;
      default:
        return <EpisodeMode />;
    }
  };

  return (
    <Container maxWidth={false} sx={{ py: 3, height: '100%' }}>
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <ModeSelector mode={mode} onModeChange={setMode} />
        <Box sx={{ flexGrow: 1, minHeight: 0 }}>
          {renderMode()}
        </Box>
      </Box>
    </Container>
  );
};

export default Write;