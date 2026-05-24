import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { FileUpload } from './FileUpload';

describe('FileUpload examples', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.restoreAllMocks();
  });

  it('renders the six example inputs', () => {
    render(<FileUpload onFileSelect={jest.fn()} />);

    expect(screen.getByText('Example inputs')).toBeTruthy();
    expect(screen.getByRole('button', { name: /definite integral/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /product rule/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /integration by parts/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /eigenvalues/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /linear system/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /quadratic complex roots/i })).toBeTruthy();
  });

  it('converts a clicked example into a File for the existing upload path', async () => {
    const onFileSelect = jest.fn();
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['fake image bytes'], { type: 'image/png' }),
    } as Response);

    render(<FileUpload onFileSelect={onFileSelect} />);

    fireEvent.click(screen.getByRole('button', { name: /eigenvalues/i }));

    await waitFor(() => expect(onFileSelect).toHaveBeenCalledTimes(1));
    const file = onFileSelect.mock.calls[0][0] as File;
    expect(file).toBeInstanceOf(File);
    expect(file.name).toBe('eigenvalues.png');
    expect(file.type).toBe('image/png');
  });

  it('shows a clean error when an example image cannot be loaded', async () => {
    const onFileSelect = jest.fn();
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 404,
    } as Response);

    render(<FileUpload onFileSelect={onFileSelect} />);

    fireEvent.click(screen.getByRole('button', { name: /linear system/i }));

    expect(await screen.findByText(/example image could not be loaded/i)).toBeTruthy();
    expect(onFileSelect).not.toHaveBeenCalled();
  });
});
