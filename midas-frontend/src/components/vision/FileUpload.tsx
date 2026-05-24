import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { QuotaStatus } from '../../types/api';

const EXAMPLE_INPUTS = [
  {
    label: 'Definite integral',
    fileName: 'definite-integral.png',
    src: `${process.env.PUBLIC_URL}/examples/definite-integral.png`,
  },
  {
    label: 'Product rule',
    fileName: 'product-rule.png',
    src: `${process.env.PUBLIC_URL}/examples/product-rule.png`,
  },
  {
    label: 'Integration by parts',
    fileName: 'integration-by-parts.png',
    src: `${process.env.PUBLIC_URL}/examples/integration-by-parts.png`,
  },
  {
    label: 'Eigenvalues',
    fileName: 'eigenvalues.png',
    src: `${process.env.PUBLIC_URL}/examples/eigenvalues.png`,
  },
  {
    label: 'Linear system',
    fileName: 'system-linear-equations.png',
    src: `${process.env.PUBLIC_URL}/examples/system-linear-equations.png`,
  },
  {
    label: 'Quadratic complex roots',
    fileName: 'quadratic-with-discriminant.png',
    src: `${process.env.PUBLIC_URL}/examples/quadratic-with-discriminant.png`,
  },
];

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
  error?: string | null;
  quota?: QuotaStatus | null;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isLoading = false, error, quota }) => {
  const [selectedExample, setSelectedExample] = React.useState<string | null>(null);
  const [exampleError, setExampleError] = React.useState<string | null>(null);

  const uploadQuota = quota?.limits.upload;
  const uploadBlocked = uploadQuota?.remaining === 0;
  const interactionDisabled = isLoading || uploadBlocked;

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) onFileSelect(acceptedFiles[0]);
  }, [onFileSelect]);

  const handleExampleClick = useCallback(async (example: typeof EXAMPLE_INPUTS[number]) => {
    if (interactionDisabled || selectedExample) return;

    setSelectedExample(example.fileName);
    setExampleError(null);

    try {
      const response = await fetch(example.src);
      if (!response.ok) {
        throw new Error(`Example image could not be loaded (${response.status}).`);
      }

      const blob = await response.blob();
      if (!blob.size) {
        throw new Error('Example image was empty.');
      }

      const file = new File([blob], example.fileName, {
        type: blob.type || 'image/png',
      });
      onFileSelect(file);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Example image could not be loaded.';
      setExampleError(message);
      setSelectedExample(null);
    }
  }, [interactionDisabled, onFileSelect, selectedExample]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] },
    multiple: false,
    disabled: interactionDisabled,
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

        {/* Quota display */}
        {uploadQuota && (
          <div style={{ marginBottom: 14, fontSize: 12.5, fontFamily: "'JetBrains Mono', monospace", color: uploadBlocked ? 'var(--failed)' : 'var(--ink-3)' }}>
            Uploads remaining today: {uploadQuota.remaining} / {uploadQuota.limit}
          </div>
        )}

        {/* Dropzone */}
        <div
          {...getRootProps()}
          style={{
            border: `1.5px dashed ${isDragActive && !isDragReject ? 'var(--accent)' : isDragReject ? 'var(--failed)' : 'var(--rule)'}`,
            borderRadius: 8,
            padding: '52px 32px 44px',
            background: isDragActive && !isDragReject ? 'var(--accent-lt)' : isDragReject ? 'var(--failed-bg)' : 'var(--parchment)',
            cursor: interactionDisabled ? 'not-allowed' : 'pointer',
            transition: 'border-color 0.2s, background 0.2s',
            marginBottom: 28,
            opacity: interactionDisabled ? 0.6 : 1,
          }}
        >
          <input {...getInputProps()} />
          <div style={{ fontSize: 42, lineHeight: 1, color: isDragActive ? 'var(--accent)' : 'var(--rule)', marginBottom: 18, transition: 'color 0.2s' }}>
            {isLoading ? '◌' : '⟁'}
          </div>
          <div style={{ fontSize: 18, color: 'var(--ink)', marginBottom: 6, fontFamily: "'EB Garamond', Georgia, serif" }}>
            {isLoading ? 'Processing document…' : uploadBlocked ? 'Upload limit reached for today' : isDragActive && !isDragReject ? 'Drop here…' : isDragReject ? 'PNG or JPG files only' : 'Drop a problem set or photograph here'}
          </div>
          <div style={{ fontSize: 12.5, fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-3)' }}>
            PNG · JPG · up to 8 MB
          </div>
        </div>

        {/* Example inputs */}
        <div style={{ marginTop: 4, marginBottom: 14, textAlign: 'left' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, marginBottom: 9 }}>
            <div style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)' }}>
              Example inputs
            </div>
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
              gap: 8,
            }}
          >
            {EXAMPLE_INPUTS.map((example) => {
              const isSelecting = selectedExample === example.fileName;
              const disabled = interactionDisabled || Boolean(selectedExample);
              return (
                <button
                  key={example.fileName}
                  type="button"
                  className="example-input-card"
                  aria-label={`Use example input: ${example.label}`}
                  onClick={() => handleExampleClick(example)}
                  disabled={disabled}
                  style={{
                    border: '1px solid var(--rule)',
                    borderRadius: 6,
                    background: 'var(--cream)',
                    padding: 0,
                    cursor: disabled ? (uploadBlocked ? 'not-allowed' : 'wait') : 'pointer',
                    overflow: 'hidden',
                    opacity: disabled && !isSelecting ? 0.58 : 1,
                    textAlign: 'left',
                    color: 'var(--ink)',
                  }}
                >
                  <div style={{ aspectRatio: '5 / 3', background: 'var(--parchment)', overflow: 'hidden', borderBottom: '1px solid var(--rule-lt)' }}>
                    <img
                      src={example.src}
                      alt=""
                      loading="lazy"
                      draggable={false}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        display: 'block',
                        padding: 5,
                        boxSizing: 'border-box',
                      }}
                    />
                  </div>
                  <div style={{ padding: '6px 7px 7px' }}>
                    <div style={{ fontSize: 10.5, lineHeight: 1.25, fontFamily: "'JetBrains Mono', monospace", color: isSelecting ? 'var(--accent)' : 'var(--ink-3)' }}>
                      {isSelecting ? 'loading…' : example.label}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
          {exampleError && (
            <div style={{ marginTop: 10, fontSize: 12.5, color: 'var(--failed)', fontFamily: "'Crimson Pro', serif", fontStyle: 'italic' }}>
              {exampleError}
            </div>
          )}
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
