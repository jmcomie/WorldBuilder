import type { LabelHTMLAttributes } from 'react';
import './Label.css';

export interface LabelProps extends LabelHTMLAttributes<HTMLLabelElement> {
  text: string;
  description?: string;
  required?: boolean;
}

const Label = ({
  text,
  description,
  required = false,
  className = '',
  children,
  ...props
}: LabelProps) => {
  return (
    <label className={`label ${className}`} {...props}>
      <div className="label-content">
        <span className="label-text">
          {text}
          {required && <span className="label-required">*</span>}
        </span>
        {description && (
          <span className="label-description">{description}</span>
        )}
      </div>
      {children}
    </label>
  );
};

export default Label;
