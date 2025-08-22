import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Chip,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Alert,
  Stack,
  Grid
} from '@mui/material';
import {
  Send,
  CheckCircle,
  Cancel,
  Edit,
  Schema
} from '@mui/icons-material';

interface Entity {
  id: string;
  name: string;
  type: string;
  confidence: number;
  status: 'pending' | 'approved' | 'rejected';
}

interface Relationship {
  id: string;
  source: string;
  target: string;
  type: string;
  confidence: number;
  status: 'pending' | 'approved' | 'rejected';
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const OntologyMode = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [entities, setEntities] = useState<Entity[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsProcessing(true);

    // Simulate AI processing
    setTimeout(() => {
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I've analyzed your description and identified the following entities and relationships:`,
        timestamp: new Date()
      };

      // Simulate extracted entities
      const newEntities: Entity[] = [
        { id: '1', name: 'Arthur', type: 'Person', confidence: 0.95, status: 'pending' },
        { id: '2', name: 'Excalibur', type: 'Artifact', confidence: 0.98, status: 'pending' },
        { id: '3', name: 'Camelot', type: 'Location', confidence: 0.92, status: 'pending' }
      ];

      // Simulate extracted relationships
      const newRelationships: Relationship[] = [
        { id: '1', source: 'Arthur', target: 'Excalibur', type: 'WIELDS', confidence: 0.96, status: 'pending' },
        { id: '2', source: 'Arthur', target: 'Camelot', type: 'RULES', confidence: 0.93, status: 'pending' }
      ];

      setMessages(prev => [...prev, assistantMessage]);
      setEntities(prev => [...prev, ...newEntities]);
      setRelationships(prev => [...prev, ...newRelationships]);
      setIsProcessing(false);
    }, 1500);
  };

  const handleEntityStatus = (id: string, status: 'approved' | 'rejected') => {
    setEntities(prev => prev.map(entity => 
      entity.id === id ? { ...entity, status } : entity
    ));
  };

  const handleRelationshipStatus = (id: string, status: 'approved' | 'rejected') => {
    setRelationships(prev => prev.map(rel => 
      rel.id === id ? { ...rel, status } : rel
    ));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'success';
      case 'rejected': return 'error';
      default: return 'default';
    }
  };

  return (
    <Grid container spacing={3} sx={{ height: '100%' }}>
      {/* Chat Panel */}
      <Grid item xs={12} md={6}>
        <Paper elevation={0} sx={{ height: '100%', p: 2, display: 'flex', flexDirection: 'column' }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Schema /> Ontology Chat
          </Typography>
          
          <Box sx={{ flexGrow: 1, overflowY: 'auto', mb: 2, minHeight: 400 }}>
            {messages.length === 0 ? (
              <Alert severity="info">
                Describe your world's concepts, and I'll help extract entities and relationships for your knowledge graph.
              </Alert>
            ) : (
              <Stack spacing={2}>
                {messages.map(message => (
                  <Box
                    key={message.id}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      bgcolor: message.role === 'user' ? 'primary.light' : 'grey.100',
                      color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
                      ml: message.role === 'user' ? 4 : 0,
                      mr: message.role === 'assistant' ? 4 : 0
                    }}
                  >
                    <Typography variant="body1">{message.content}</Typography>
                    <Typography variant="caption" sx={{ opacity: 0.7 }}>
                      {message.timestamp.toLocaleTimeString()}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            )}
          </Box>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Describe entities and their relationships..."
              value={inputMessage}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputMessage(e.target.value)}
              onKeyPress={(e: React.KeyboardEvent) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              disabled={isProcessing}
              multiline
              maxRows={3}
            />
            <Button
              variant="contained"
              onClick={handleSendMessage}
              disabled={isProcessing || !inputMessage.trim()}
              endIcon={<Send />}
            >
              Send
            </Button>
          </Box>
        </Paper>
      </Grid>

      {/* Artifacts Panel */}
      <Grid item xs={12} md={6}>
        <Paper elevation={0} sx={{ height: '100%', p: 2, overflowY: 'auto' }}>
          <Typography variant="h6" gutterBottom>
            Extracted Artifacts
          </Typography>

          {/* Entities Section */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                Entities
              </Typography>
              {entities.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No entities extracted yet
                </Typography>
              ) : (
                <List>
                  {entities.map((entity, index) => (
                    <React.Fragment key={entity.id}>
                      {index > 0 && <Divider />}
                      <ListItem>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body1">{entity.name}</Typography>
                              <Chip label={entity.type} size="small" color="primary" />
                              <Chip 
                                label={`${Math.round(entity.confidence * 100)}%`} 
                                size="small" 
                                variant="outlined" 
                              />
                              <Chip 
                                label={entity.status} 
                                size="small" 
                                color={getStatusColor(entity.status) as any}
                              />
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          {entity.status === 'pending' && (
                            <>
                              <IconButton 
                                edge="end" 
                                aria-label="approve"
                                onClick={() => handleEntityStatus(entity.id, 'approved')}
                                color="success"
                              >
                                <CheckCircle />
                              </IconButton>
                              <IconButton 
                                edge="end" 
                                aria-label="reject"
                                onClick={() => handleEntityStatus(entity.id, 'rejected')}
                                color="error"
                              >
                                <Cancel />
                              </IconButton>
                              <IconButton edge="end" aria-label="edit">
                                <Edit />
                              </IconButton>
                            </>
                          )}
                        </ListItemSecondaryAction>
                      </ListItem>
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>

          {/* Relationships Section */}
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                Relationships
              </Typography>
              {relationships.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No relationships extracted yet
                </Typography>
              ) : (
                <List>
                  {relationships.map((rel, index) => (
                    <React.Fragment key={rel.id}>
                      {index > 0 && <Divider />}
                      <ListItem>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body1">
                                {rel.source} → {rel.type} → {rel.target}
                              </Typography>
                              <Chip 
                                label={`${Math.round(rel.confidence * 100)}%`} 
                                size="small" 
                                variant="outlined" 
                              />
                              <Chip 
                                label={rel.status} 
                                size="small" 
                                color={getStatusColor(rel.status) as any}
                              />
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          {rel.status === 'pending' && (
                            <>
                              <IconButton 
                                edge="end" 
                                aria-label="approve"
                                onClick={() => handleRelationshipStatus(rel.id, 'approved')}
                                color="success"
                              >
                                <CheckCircle />
                              </IconButton>
                              <IconButton 
                                edge="end" 
                                aria-label="reject"
                                onClick={() => handleRelationshipStatus(rel.id, 'rejected')}
                                color="error"
                              >
                                <Cancel />
                              </IconButton>
                              <IconButton edge="end" aria-label="edit">
                                <Edit />
                              </IconButton>
                            </>
                          )}
                        </ListItemSecondaryAction>
                      </ListItem>
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>

          {/* Save Button */}
          {(entities.some(e => e.status === 'approved') || relationships.some(r => r.status === 'approved')) && (
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Button variant="contained" size="large" fullWidth>
                Save Approved Artifacts to Graph
              </Button>
            </Box>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
};

export default OntologyMode;