# Graphiti Episode Form Implementation Steps

## Phase 1: Type Definitions and API Setup

### Step 1.1: Create TypeScript Types
Create `frontend/src/types/graphiti.ts`:
```typescript
export enum EpisodeType {
  TEXT = 'text',
  JSON = 'json',
  MESSAGE = 'message'
}

export interface EpisodeRequest {
  name: string;
  content: string;
  source_description?: string;
}

export interface EpisodeResponse {
  success: boolean;
  message: string;
  episode_id?: string;
}

export interface ApiError {
  detail: string;
}
```

### Step 1.2: Update API Client
Add to `frontend/src/api.ts`:
```typescript
import { EpisodeRequest, EpisodeResponse } from './types/graphiti';

export async function createEpisode(episode: EpisodeRequest): Promise<EpisodeResponse> {
  const response = await fetch(`${API_URL}/episodes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(episode),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create episode');
  }

  return response.json();
}
```

## Phase 2: Create the Episode Form Component

### Step 2.1: Create Component File
Create `frontend/src/components/EpisodeForm.tsx`:
```typescript
import React, { useState } from 'react';
import { createEpisode } from '../api';
import { EpisodeType, EpisodeRequest } from '../types/graphiti';
```

### Step 2.2: Define Component State
```typescript
const [name, setName] = useState('');
const [content, setContent] = useState('');
const [sourceDescription, setSourceDescription] = useState('');
const [episodeType, setEpisodeType] = useState<EpisodeType>(EpisodeType.TEXT);
const [isSubmitting, setIsSubmitting] = useState(false);
const [error, setError] = useState<string | null>(null);
const [success, setSuccess] = useState(false);
```

### Step 2.3: Implement Form Validation
```typescript
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
```

### Step 2.4: Implement Submit Handler
```typescript
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
```

### Step 2.5: Implement Form Reset
```typescript
const resetForm = () => {
  setName('');
  setContent('');
  setSourceDescription('');
  setEpisodeType(EpisodeType.TEXT);
  setError(null);
};
```

## Phase 3: Build the Form UI

### Step 3.1: Create Form JSX Structure
```typescript
return (
  <div className="episode-form-container">
    <h2>Create New Episode</h2>
    
    {error && (
      <div className="error-message">
        {error}
        <button onClick={() => setError(null)}>×</button>
      </div>
    )}
    
    {success && (
      <div className="success-message">
        Episode created successfully!
      </div>
    )}
    
    <form onSubmit={handleSubmit}>
      {/* Form fields here */}
    </form>
  </div>
);
```

### Step 3.2: Add Form Fields
```typescript
<div className="form-field">
  <label htmlFor="episode-name">
    Episode Name <span className="required">*</span>
  </label>
  <input
    id="episode-name"
    type="text"
    value={name}
    onChange={(e) => setName(e.target.value)}
    disabled={isSubmitting}
    placeholder="Enter episode name"
  />
</div>

<div className="form-field">
  <label htmlFor="episode-content">
    Content <span className="required">*</span>
  </label>
  <textarea
    id="episode-content"
    value={content}
    onChange={(e) => setContent(e.target.value)}
    disabled={isSubmitting}
    placeholder="Enter episode content (minimum 10 characters)"
    rows={8}
  />
  <div className="char-count">
    {content.length} characters
  </div>
</div>

<div className="form-field">
  <label htmlFor="source-description">
    Source Description (optional)
  </label>
  <input
    id="source-description"
    type="text"
    value={sourceDescription}
    onChange={(e) => setSourceDescription(e.target.value)}
    disabled={isSubmitting}
    placeholder="Where did this information come from?"
  />
</div>

<div className="form-actions">
  <button 
    type="submit" 
    disabled={isSubmitting}
    className="submit-button"
  >
    {isSubmitting ? 'Creating...' : 'Create Episode'}
  </button>
  
  <button 
    type="button" 
    onClick={resetForm}
    disabled={isSubmitting}
    className="reset-button"
  >
    Reset
  </button>
</div>
```

## Phase 4: Add Styling

### Step 4.1: Create CSS Module
Create `frontend/src/components/EpisodeForm.module.css`:
```css
.episode-form-container {
  max-width: 600px;
  margin: 2rem auto;
  padding: 2rem;
}

.form-field {
  margin-bottom: 1.5rem;
}

.form-field label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
}

.form-field input,
.form-field textarea,
.form-field select {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
}

.form-field input:disabled,
.form-field textarea:disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

.required {
  color: #e74c3c;
}

.char-count {
  text-align: right;
  font-size: 0.875rem;
  color: #666;
  margin-top: 0.25rem;
}

.error-message {
  background-color: #fee;
  color: #c33;
  padding: 1rem;
  border-radius: 4px;
  margin-bottom: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.success-message {
  background-color: #efe;
  color: #3c763d;
  padding: 1rem;
  border-radius: 4px;
  margin-bottom: 1rem;
}

.form-actions {
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
}

.submit-button {
  background-color: #007bff;
  color: white;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.2s;
}

.submit-button:hover:not(:disabled) {
  background-color: #0056b3;
}

.submit-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.reset-button {
  background-color: #6c757d;
  color: white;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
}
```

## Phase 5: Integration with App

### Step 5.1: Update App Routes
In `frontend/src/App.tsx`, add:
```typescript
import EpisodeForm from './components/EpisodeForm';

// In the Routes section:
<Route path="/write" element={<EpisodeForm />} />
```

### Step 5.2: Update Navigation
Add link to the write view in your navigation component:
```typescript
<NavLink to="/write">Create Episode</NavLink>
```

### Step 5.3: Update View Navigation
Ensure the Write view is accessible from the main navigation tabs.

## Phase 6: Testing

### Step 6.1: Manual Testing Checklist
- [ ] Form renders correctly
- [ ] Required field validation works
- [ ] Character count updates
- [ ] Submit button disables during submission
- [ ] Success message appears and auto-dismisses
- [ ] Error messages display correctly
- [ ] Form resets after successful submission
- [ ] API integration works correctly

### Step 6.2: Test Different Scenarios
1. Submit with empty fields
2. Submit with content less than 10 characters
3. Submit valid episode
4. Test network error handling
5. Test form reset functionality

## Phase 7: Optional Enhancements

### Step 7.1: Add Local Storage Draft
```typescript
// Save draft on content change
useEffect(() => {
  const draft = { name, content, sourceDescription };
  localStorage.setItem('episode-draft', JSON.stringify(draft));
}, [name, content, sourceDescription]);

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

// Clear draft on successful submission
const clearDraft = () => {
  localStorage.removeItem('episode-draft');
};
```

### Step 7.2: Add Keyboard Shortcuts
```typescript
// Cmd/Ctrl + Enter to submit
useEffect(() => {
  const handleKeyPress = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      handleSubmit(e as any);
    }
  };
  
  document.addEventListener('keydown', handleKeyPress);
  return () => document.removeEventListener('keydown', handleKeyPress);
}, [name, content]);
```

## Implementation Order

1. **Day 1**: Complete Phase 1-2 (Types, API, Basic Component)
2. **Day 2**: Complete Phase 3-4 (UI and Styling)
3. **Day 3**: Complete Phase 5-6 (Integration and Testing)
4. **Day 4**: Add optional enhancements if time permits

## Success Criteria

- Users can create episodes through the UI
- Form provides clear feedback for all actions
- Validation prevents invalid submissions
- Integration with backend works seamlessly
- UI is responsive and accessible