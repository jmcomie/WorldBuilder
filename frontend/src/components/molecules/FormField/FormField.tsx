import type { ReactNode } from 'react';
import Label from '../../atoms/Label';
import './FormField.css';

export interface FormFieldProps {
  label: string;
  description?: string;
  required?: boolean;
  error?: boolean;
  errorMessage?: string;
  htmlFor?: string;
  children: ReactNode;
  orientation?: 'vertical' | 'horizontal';
}

const FormField = ({
  label,
  description,
  required = false,
  error = false,
  errorMessage,
  htmlFor,
  children,
  orientation = 'vertical'
}: FormFieldProps) => {
  const fieldClasses = [
    'form-field',
    `form-field--${orientation}`,
    error && 'form-field--error'
  ].filter(Boolean).join(' ');

  return (
    <div className={fieldClasses}>
      <Label
        text={label}
        description={description}
        required={required}
        htmlFor={htmlFor}
        className="form-field__label"
      />
      <div className="form-field__control">
        {children}
        {error && errorMessage && (
          <span className="form-field__error">{errorMessage}</span>
        )}
      </div>
    </div>
  );
};

export default FormField;