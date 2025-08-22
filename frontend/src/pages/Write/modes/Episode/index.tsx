import React, { useState, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Alert, 
  AlertTitle,
  Typography,
  Paper,
  Stack,
  IconButton,
  Collapse
} from '@mui/material';
import { Close } from '@mui/icons-material';
import { createEpisode } from '../../../../shared/api/client';
import type { EpisodeRequest } from '../../../../types/graphiti';

const EpisodeMode = () => {
  const [name, setName] = useState('');
  const [content, setContent] = useState('');
  const [sourceDescription, setSourceDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Load draft on mount
  useEffect(() => {
    const savedDraft = localStorage.getItem('episode-draft');
    if (savedDraft) {
      const draft = JSON.parse(savedDraft);
      setName(draft.name || '');
      setContent(draft.content || '');
      setSourceDescription(draft.sourceDescription || '');
    }
  }, []);

  // Save draft on content change
  useEffect(() => {
    const draft = { name, content, sourceDescription };
    localStorage.setItem('episode-draft', JSON.stringify(draft));
  }, [name, content, sourceDescription]);

  const validateForm = (): boolean => {
    if (!name.trim()) {
      setError('Episode name is required');
      return false;
    }
    if (!content.trim() || content.length < 10) {
      setError('Episode content must be at least 10 characters');
      return false;
    }
    return true;
  };

  const resetForm = () => {
    setName('');
    setContent('');
    setSourceDescription('');
    setError(null);
    localStorage.removeItem('episode-draft');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    
    if (!validateForm()) return;

    setIsSubmitting(true);
    
    try {
      const episodeData: EpisodeRequest = {
        name: name.trim(),
        content: content.trim(),
        source_description: sourceDescription.trim() || undefined,
      };
      
      await createEpisode(episodeData);
      setSuccess(true);
      resetForm();
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Paper elevation={0} sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        Create New Episode
      </Typography>
      
      <Collapse in={!!error}>
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          action={
            <IconButton
              aria-label="close"
              color="inherit"
              size="small"
              onClick={() => setError(null)}
            >
              <Close fontSize="inherit" />
            </IconButton>
          }
        >
          {error}
        </Alert>
      </Collapse>
      
      <Collapse in={success}>
        <Alert severity="success" sx={{ mb: 2 }}>
          <AlertTitle>Success</AlertTitle>
          Episode created successfully!
        </Alert>
      </Collapse>
      
      <Box component="form" onSubmit={handleSubmit}>
        <Stack spacing={3}>
          <TextField
            label="Episode Name"
            required
            fullWidth
            value={name}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
            disabled={isSubmitting}
            placeholder="Enter episode name"
          />

          <TextField
            label="Content"
            required
            fullWidth
            multiline
            rows={8}
            value={content}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setContent(e.target.value)}
            disabled={isSubmitting}
            placeholder="Enter episode content (minimum 10 characters)"
            helperText={`${content.length} characters`}
          />

          <TextField
            label="Source Description"
            fullWidth
            value={sourceDescription}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSourceDescription(e.target.value)}
            disabled={isSubmitting}
            placeholder="Where did this information come from?"
          />

          <Stack direction="row" spacing={2}>
            <Button 
              type="submit" 
              variant="contained"
              disabled={isSubmitting}
              size="large"
            >
              {isSubmitting ? 'Creating...' : 'Create Episode'}
            </Button>
            
            <Button 
              type="button" 
              variant="outlined"
              onClick={resetForm}
              disabled={isSubmitting}
              size="large"
            >
              Reset
            </Button>
          </Stack>
        </Stack>
      </Box>
    </Paper>
  );
};

export default EpisodeMode;