import { forwardRef } from 'react';
import type { InputHTMLAttributes } from 'react';
import './Checkbox.css';

export interface CheckboxProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  indeterminate?: boolean;
}

const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  (
    { label, indeterminate: _indeterminate = false, className = '', ...props },
    ref
  ) => {
    return (
      <label className="checkbox-wrapper">
        <input
          ref={ref}
          type="checkbox"
          className={`checkbox ${className}`}
          {...props}
        />
        <span className="checkbox-checkmark"></span>
        {label && <span className="checkbox-label">{label}</span>}
      </label>
    );
  }
);

Checkbox.displayName = 'Checkbox';

export default Checkbox;
