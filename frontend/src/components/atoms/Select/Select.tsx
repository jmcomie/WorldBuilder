import { forwardRef } from 'react';
import type { SelectHTMLAttributes } from 'react';
import './Select.css';

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  fullWidth?: boolean;
  error?: boolean;
  helperText?: string;
}

const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ fullWidth = false, error = false, helperText, className = '', children, ...props }, ref) => {
    const selectClasses = [
      'select',
      fullWidth && 'select--full-width',
      error && 'select--error',
      className
    ].filter(Boolean).join(' ');

    return (
      <div className="select-wrapper">
        <select
          ref={ref}
          className={selectClasses}
          aria-invalid={error}
          aria-describedby={helperText ? 'select-helper-text' : undefined}
          {...props}
        >
          {children}
        </select>
        <span className="select-arrow">▼</span>
        {helperText && (
          <span 
            id="select-helper-text" 
            className={`select-helper ${error ? 'select-helper--error' : ''}`}
          >
            {helperText}
          </span>
        )}
      </div>
    );
  }
);

Select.displayName = 'Select';

export default Select;