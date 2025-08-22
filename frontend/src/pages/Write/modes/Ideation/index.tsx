import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Stack,
  Chip,
  Alert,
  TextField,
  CircularProgress,
  IconButton,
  Divider
} from '@mui/material';
import { Psychology, Lightbulb, Book, Send as SendIcon, ContentCopy } from '@mui/icons-material';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const IdeationMode = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const suggestedPrompts = [
    { icon: <Psychology />, text: "Help me create a unique magic system" },
    { icon: <Lightbulb />, text: "I need ideas for a mysterious ancient civilization" },
    { icon: <Book />, text: "Create an interesting character with a complex backstory" }
  ];

  const generateResponse = async (userQuery: string): Promise<string> => {
    // Simulate AI thinking time
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Generate mock response based on context
    if (userQuery.toLowerCase().includes('character')) {
      return `Here's an idea for a character in your world:

**Name**: Lyra Shadowweaver

**Background**: Born in the mystical city of Aethermoor, Lyra discovered her unique ability to manipulate shadows at a young age. Her powers manifested during a solar eclipse, marking her as one of the rare Umbralists.

**Personality**: Curious and determined, yet haunted by the responsibility of her powers. She struggles between using her abilities for personal gain and protecting those who fear her kind.

**Abilities**:
- Shadow manipulation and travel
- Can create solid constructs from darkness
- Heightened senses in low light
- Weakness to bright, sudden light

**Story Hook**: Lyra discovers an ancient prophecy that speaks of an Umbralist who will either save the realm from eternal darkness or plunge it into shadow forever.`;
    } else if (userQuery.toLowerCase().includes('magic')) {
      return `Let me help you develop a magic system:

**The Resonance System**

Magic in your world could work through harmonic resonance with natural elements. Here's how:

**Core Principle**: Everything in nature vibrates at specific frequencies. Mages learn to attune their own spiritual frequency to match these natural vibrations.

**Types of Resonance**:
1. **Elemental Resonance**: Fire, Water, Earth, Air
2. **Life Resonance**: Healing, growth, decay
3. **Mind Resonance**: Telepathy, illusions, memory manipulation
4. **Time Resonance**: Rare and dangerous, affecting temporal flow

**Limitations**:
- Resonating with opposing frequencies causes physical pain
- Overuse leads to "frequency fatigue" - temporary loss of abilities
- Certain materials (like lead or obsidian) block resonance

**Training**: Mages must spend years learning to "hear" the frequencies of nature through meditation and practice.`;
    } else if (userQuery.toLowerCase().includes('location') || userQuery.toLowerCase().includes('place')) {
      return `Here's a unique location for your world:

**The Floating Gardens of Memoriam**

**Description**: A series of interconnected islands that float above a perpetual mist, held aloft by ancient crystalline anchors that glow with soft blue light.

**Features**:
- Each garden represents a different emotion or memory
- The Garden of Joy blooms with eternal spring flowers
- The Garden of Sorrow has perpetual autumn, with golden leaves that never fall
- The Garden of Rage features volcanic stone and fire flowers

**Inhabitants**: The Keepers - monk-like beings who tend to the gardens and help visitors process their memories and emotions

**Mystery**: No one knows who created the gardens or why they float. Some say they're fragments of a shattered divine realm.

**Story Potential**: Characters might need to visit to recover lost memories, seek emotional healing, or find a rare plant that only grows in one specific garden.`;
    } else {
      return `I'm here to help you brainstorm and develop ideas for your world! 

Try asking me about:
- Character concepts and backstories
- Magic systems and their rules
- Unique locations and landmarks
- Cultural traditions and societies
- Conflicts and plot hooks
- Creatures and monsters
- Historical events that shaped your world
- Technologies or innovations
- Religions and belief systems

What aspect of your world would you like to explore?`;
    }
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await generateResponse(inputValue);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error generating response:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePromptClick = (promptText: string) => {
    setInputValue(promptText);
  };

  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Paper elevation={0} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Psychology /> Ideation Assistant
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Brainstorm and develop creative ideas for your world with AI assistance
        </Typography>
      </Box>

      <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
        {messages.length === 0 ? (
          <Box sx={{ textAlign: 'center', mt: 4 }}>
            <Psychology sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Welcome to Ideation Mode
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              I'm here to help you brainstorm and develop creative ideas for your worldbuilding project.
            </Typography>
            
            <Stack direction="row" spacing={1} justifyContent="center" flexWrap="wrap" sx={{ mt: 3, mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Try one of these:
              </Typography>
            </Stack>
            
            <Stack spacing={1} sx={{ maxWidth: 600, mx: 'auto' }}>
              {suggestedPrompts.map((prompt, index) => (
                <Chip
                  key={index}
                  icon={prompt.icon}
                  label={prompt.text}
                  onClick={() => handlePromptClick(prompt.text)}
                  variant="outlined"
                  sx={{ 
                    justifyContent: 'flex-start',
                    height: 'auto',
                    padding: '8px 12px',
                    '& .MuiChip-label': {
                      whiteSpace: 'normal',
                      textAlign: 'left'
                    }
                  }}
                />
              ))}
            </Stack>
          </Box>
        ) : (
          <Stack spacing={2}>
            {messages.map((message) => (
              <Box
                key={message.id}
                sx={{
                  display: 'flex',
                  justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start'
                }}
              >
                <Paper
                  sx={{
                    p: 2,
                    maxWidth: '80%',
                    bgcolor: message.role === 'user' ? 'primary.light' : 'grey.100',
                    color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
                    position: 'relative'
                  }}
                >
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                    {message.content}
                  </Typography>
                  <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography variant="caption" sx={{ opacity: 0.7 }}>
                      {message.timestamp.toLocaleTimeString()}
                    </Typography>
                    {message.role === 'assistant' && (
                      <IconButton 
                        size="small" 
                        onClick={() => handleCopyMessage(message.content)}
                        sx={{ ml: 1 }}
                      >
                        <ContentCopy fontSize="small" />
                      </IconButton>
                    )}
                  </Box>
                </Paper>
              </Box>
            ))}
            {isLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'flex-start' }}>
                <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <CircularProgress size={20} />
                </Paper>
              </Box>
            )}
          </Stack>
        )}
      </Box>

      <Divider />

      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask for worldbuilding ideas, character concepts, plot hooks..."
            disabled={isLoading}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2
              }
            }}
          />
          <Button
            variant="contained"
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
            sx={{ px: 3 }}
            endIcon={<SendIcon />}
          >
            Send
          </Button>
        </Box>
        
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            <strong>Tip:</strong> You can export interesting ideas to the Episode mode to develop them further.
          </Typography>
        </Alert>
      </Box>
    </Paper>
  );
};

export default IdeationMode;