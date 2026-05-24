import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { FileUpload } from './FileUpload';

describe('FileUpload examples', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders the six example inputs', () => {
    render(<FileUpload onFileSelect={jest.fn()} />);

    expect(screen.getByText('Preloaded examples')).toBeTruthy();
    expect(screen.getByRole('button', { name: /definite integral/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /product rule/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /integration by parts/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /eigenvalues/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /linear system/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /quadratic complex roots/i })).toBeTruthy();
  });

  it('loads a clicked example through the cached example path', () => {
    const onFileSelect = jest.fn();
    const onExampleSelect = jest.fn();

    render(<FileUpload onFileSelect={onFileSelect} onExampleSelect={onExampleSelect} />);

    fireEvent.click(screen.getByRole('button', { name: /eigenvalues/i }));

    expect(onExampleSelect).toHaveBeenCalledTimes(1);
    expect(onExampleSelect).toHaveBeenCalledWith('eigenvalues');
    expect(onFileSelect).not.toHaveBeenCalled();
  });

  it('does not load examples when upload quota is exhausted', () => {
    const onFileSelect = jest.fn();
    const onExampleSelect = jest.fn();

    render(
      <FileUpload
        onFileSelect={onFileSelect}
        onExampleSelect={onExampleSelect}
        quota={{
          enabled: true,
          reset_seconds: 100,
          limits: {
            upload: { limit: 5, used: 5, remaining: 0 },
            pipeline: { limit: 3, used: 0, remaining: 3 },
            reason: { limit: 0, used: 0, remaining: 0 },
            explain: { limit: 10, used: 0, remaining: 10 },
            feedback: { limit: 10, used: 0, remaining: 10 },
          },
        }}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /linear system/i }));

    expect(onExampleSelect).not.toHaveBeenCalled();
    expect(onFileSelect).not.toHaveBeenCalled();
  });
});
