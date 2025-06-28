import React from 'react';
import EpisodeForm from '../../components/EpisodeForm';
import './Write.css';

const Write: React.FC = () => {
  return (
    <div className="view-container write-view">
      <EpisodeForm />
    </div>
  );
};

export default Write;