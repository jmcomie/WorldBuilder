# Graphiti Episode Creation Interface Plan

## Overview
This document outlines the plan for creating a user interface to add new episodes to the Graphiti knowledge graph in the Worldbuilder application.

## 1. Component Structure
- Create a new component `EpisodeForm.tsx` in the frontend
- Use React functional component with hooks
- Implement form state management using `useState` 
- Handle async submission with proper loading/error states

## 2. Form Design
Based on Graphiti's episode structure, the form will include:
- **Episode Name** (required) - Text input for the episode title
- **Episode Content** (required) - Textarea for the main content/body
- **Source Description** (optional) - Text input to describe where this info came from
- **Episode Type** - Select dropdown with options:
  - `text` (default) - for user-written content
  - `json` - for structured data
  - `message` - for conversation/chat content

## 3. State Management
Using React's modern hooks approach:
- `useState` for form fields (name, content, sourceDescription, episodeType)
- `useState` for UI states (isSubmitting, error, success)
- Controlled components pattern for all inputs

## 4. Form Validation
- Client-side validation:
  - Required fields: name and content
  - Minimum length for content (e.g., 10 characters)
  - Clear error messages using conditional rendering

## 5. API Integration
- Create new API function in `frontend/src/api.ts`:
  - `createEpisode()` - POST to `/episodes` endpoint
  - Handle request/response with proper typing
  - Error handling with user-friendly messages

## 6. User Experience Features
- Loading state during submission (disable form, show spinner)
- Success feedback (toast notification or success message)
- Error handling with specific messages
- Form reset after successful submission
- Character count for content field
- Auto-save draft to localStorage (optional enhancement)

## 7. Styling Considerations
- Consistent with existing app design
- Responsive layout
- Clear visual hierarchy
- Accessible form labels and error messages

## 8. Integration Points
- Add route to access the form (e.g., `/write` or `/create`)
- Update navigation to include link to episode creation
- Consider where this fits in the overall app flow

## 9. Type Safety
- Define TypeScript interfaces for:
  - Episode request/response types
  - Form state
  - API error responses

## 10. Future Enhancements
- Rich text editor for content field
- File upload for JSON episodes
- Tags/categories for episodes
- Preview mode before submission
- Batch episode creation
- Template system for common episode types

## Technical Decisions

### Why useState over useReducer?
For this form, `useState` is sufficient because:
- The form state is relatively simple
- State updates are straightforward
- No complex state transitions or dependencies

### Form Submission Pattern
Using the async/await pattern with try/catch for clarity:
```typescript
async function handleSubmit(e: React.FormEvent) {
  e.preventDefault();
  setIsSubmitting(true);
  try {
    await createEpisode(formData);
    setSuccess(true);
    resetForm();
  } catch (err) {
    setError(err.message);
  } finally {
    setIsSubmitting(false);
  }
}
```

### Error Handling Strategy
- Display inline errors below relevant fields
- Show general errors in a dismissible alert
- Log detailed errors to console for debugging
- Provide actionable error messages to users

This plan follows React best practices from the official documentation, using functional components with hooks, proper state management, and clean separation of concerns.