/**
 * Placeholder component for graph visualization
 * Displayed while graph functionality is being reimplemented
 */

import React from 'react';

export const GraphPlaceholder: React.FC = () => {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: '400px',
        color: '#666',
        fontSize: '1.2rem',
        fontFamily: 'inherit',
      }}
    >
      <p>Graph visualization temporarily unavailable</p>
    </div>
  );
};

export default GraphPlaceholder;
