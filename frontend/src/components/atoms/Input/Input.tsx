import { forwardRef } from 'react';
import type { InputHTMLAttributes } from 'react';
import './Input.css';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  variant?: 'default' | 'monospace';
  fullWidth?: boolean;
  error?: boolean;
  helperText?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      variant = 'default',
      fullWidth = false,
      error = false,
      helperText,
      className = '',
      ...props
    },
    ref
  ) => {
    const inputClasses = [
      'input',
      `input--${variant}`,
      fullWidth && 'input--full-width',
      error && 'input--error',
      className,
    ]
      .filter(Boolean)
      .join(' ');

    return (
      <div className="input-wrapper">
        <input
          ref={ref}
          className={inputClasses}
          aria-invalid={error}
          aria-describedby={helperText ? 'input-helper-text' : undefined}
          {...props}
        />
        {helperText && (
          <span
            id="input-helper-text"
            className={`input-helper ${error ? 'input-helper--error' : ''}`}
          >
            {helperText}
          </span>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
