import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
  error?: string | null;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isLoading = false, error }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) onFileSelect(acceptedFiles[0]);
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'], 'application/pdf': ['.pdf'] },
    multiple: false,
    disabled: isLoading,
  });

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 24px', background: 'var(--cream)' }}>
      <div style={{ maxWidth: 520, width: '100%', textAlign: 'center' }}>

        {/* Wordmark */}
        <div style={{ fontSize: 32, fontWeight: 600, letterSpacing: '0.08em', color: 'var(--ink)', marginBottom: 6 }}>
          MIDAS
        </div>
        <div style={{ width: 40, height: 1.5, background: 'var(--rule)', margin: '10px auto 12px' }} />
        <div style={{ fontSize: 14.5, fontStyle: 'italic', color: 'var(--ink-3)', fontFamily: "'Crimson Pro', Georgia, serif", marginBottom: 44, lineHeight: 1.5 }}>
          Mathematical Intelligence with Deductive, Algebraic Synthesis<br />
          Upload a problem — MIDAS will solve, verify, and explain it step by step.
        </div>

        {/* Dropzone */}
        <div
          {...getRootProps()}
          style={{
            border: `1.5px dashed ${isDragActive && !isDragReject ? 'var(--accent)' : isDragReject ? 'var(--failed)' : 'var(--rule)'}`,
            borderRadius: 8,
            padding: '52px 32px 44px',
            background: isDragActive && !isDragReject ? 'var(--accent-lt)' : isDragReject ? 'var(--failed-bg)' : 'var(--parchment)',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            transition: 'border-color 0.2s, background 0.2s',
            marginBottom: 28,
            opacity: isLoading ? 0.6 : 1,
          }}
        >
          <input {...getInputProps()} />
          <div style={{ fontSize: 42, lineHeight: 1, color: isDragActive ? 'var(--accent)' : 'var(--rule)', marginBottom: 18, transition: 'color 0.2s' }}>
            {isLoading ? '◌' : '⟁'}
          </div>
          <div style={{ fontSize: 18, color: 'var(--ink)', marginBottom: 6, fontFamily: "'EB Garamond', Georgia, serif" }}>
            {isLoading ? 'Processing document…' : isDragActive && !isDragReject ? 'Drop here…' : isDragReject ? 'PDF or image files only' : 'Drop a problem set or photograph here'}
          </div>
          <div style={{ fontSize: 12.5, fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-3)' }}>
            PDF · PNG · JPG · up to 20 MB
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ marginTop: 16, padding: '12px 16px', background: 'var(--failed-bg)', border: '1px solid var(--failed-bd)', borderRadius: 6, fontSize: 13.5, color: 'var(--failed)', fontFamily: "'Crimson Pro', serif", fontStyle: 'italic' }}>
            {error}
          </div>
        )}

        {/* Caption */}
        <div style={{ marginTop: 20, fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-3)', letterSpacing: '0.04em' }}>
          phi4-mini-reasoning · qwen2.5-coder · SymPy verification
        </div>
      </div>
    </div>
  );
};
