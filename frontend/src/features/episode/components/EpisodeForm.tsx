import React, { useState, useEffect } from 'react';
// Episode creation functionality temporarily disabled
import './EpisodeForm.css';

const EpisodeForm = () => {
  const [name, setName] = useState('');
  const [content, setContent] = useState('');
  const [sourceDescription, setSourceDescription] = useState('');
  const [isSubmitting, _setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, _setSuccess] = useState(false);

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

  const _validateForm = (): boolean => {
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
    setError(
      'Episode creation is not yet implemented. This feature is coming soon.'
    );
  };

  return (
    <div className="episode-form-container">
      <h2>Create New Episode</h2>

      {error && (
        <div className="error-message">
          {error}
          <button onClick={() => setError(null)} className="close-button">
            ×
          </button>
        </div>
      )}

      {success && (
        <div className="success-message">Episode created successfully!</div>
      )}

      <form onSubmit={handleSubmit}>
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
          <div className="char-count">{content.length} characters</div>
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
            disabled={false}
            className="submit-button"
            title="Episode creation coming soon"
          >
            Create Episode (Coming Soon)
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
      </form>
    </div>
  );
};

export default EpisodeForm;
