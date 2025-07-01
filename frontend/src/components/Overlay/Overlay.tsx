import React from 'react';
import './Overlay.css';

interface OverlayProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  className?: string;
}

const Overlay: React.FC<OverlayProps> = ({ isOpen, onClose, title, children, className }) => {
  if (!isOpen) return null;

  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div className={`overlay-content ${className || ''}`} onClick={(e) => e.stopPropagation()}>
        <div className="overlay-header">
          <h2>{title}</h2>
          <button className="overlay-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="overlay-body">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Overlay;