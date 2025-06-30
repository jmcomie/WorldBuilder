import React from 'react';
import { ToggleButton, ToggleButtonGroup, Box, Typography } from '@mui/material';
import { Create, Schema, Psychology } from '@mui/icons-material';

export type WriteMode = 'episode' | 'ontology' | 'ideation';

interface ModeSelectorProps {
  mode: WriteMode;
  onModeChange: (mode: WriteMode) => void;
}

const ModeSelector: React.FC<ModeSelectorProps> = ({ mode, onModeChange }) => {
  const handleModeChange = (_: React.MouseEvent<HTMLElement>, newMode: WriteMode | null) => {
    if (newMode !== null) {
      onModeChange(newMode);
    }
  };

  return (
    <Box sx={{ mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Select Mode
      </Typography>
      <ToggleButtonGroup
        value={mode}
        exclusive
        onChange={handleModeChange}
        aria-label="write mode"
        sx={{ width: '100%' }}
      >
        <ToggleButton value="episode" aria-label="episode mode" sx={{ flex: 1 }}>
          <Create sx={{ mr: 1 }} />
          Episode
        </ToggleButton>
        <ToggleButton value="ontology" aria-label="ontology mode" sx={{ flex: 1 }}>
          <Schema sx={{ mr: 1 }} />
          Ontology
        </ToggleButton>
        <ToggleButton value="ideation" aria-label="ideation mode" sx={{ flex: 1 }}>
          <Psychology sx={{ mr: 1 }} />
          Ideation
        </ToggleButton>
      </ToggleButtonGroup>
    </Box>
  );
};

export default ModeSelector;