import React from 'react';
import { FileUpload } from './FileUpload';
import { DocumentSelectionScreen } from './DocumentSelectionScreen';
import { DocumentPreview } from './DocumentPreview';
import { useDocumentState } from '../../hooks/useDocumentState';
import { PipelineResults } from '../results/PipelineResults';
import { PipelineLoading } from '../ui/PipelineLoading';

export const FullVisionPipeline: React.FC = () => {
  const {
    document,
    originalImageBase64,
    uploadedFile,
    selectedProblemId,
    editedLatex,
    editedVisualContext,
    removeVisualContext,
    isLoading,
    error,
    processingStage,
    completePipelineResult,
    quota,
    actions,
  } = useDocumentState();

  // State 1: idle
  if (processingStage === 'idle') {
    return <FileUpload onFileSelect={actions.handleFileUpload} error={error} quota={quota} />;
  }

  // State 2: file dropped, not yet OCR'd — show brief preview
  if (processingStage === 'uploading' && uploadedFile && originalImageBase64) {
    return (
      <DocumentPreview
        imageBase64={originalImageBase64}
        fileName={uploadedFile.name}
        onProcess={actions.processUploadedFile}
        onCancel={actions.cancelUpload}
        isProcessing={isLoading}
      />
    );
  }

  // State 3: error
  if (processingStage === 'error') {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--cream)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
        <div style={{ maxWidth: 480, width: '100%', background: 'var(--failed-bg)', border: '1px solid var(--failed-bd)', borderRadius: 8, padding: 28 }}>
          <div style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--failed)', marginBottom: 12 }}>
            Error
          </div>
          <div style={{ fontSize: 15, fontFamily: "'Crimson Pro', serif", color: 'var(--ink-2)', lineHeight: 1.6, marginBottom: 20 }}>
            {error}
          </div>
          <button
            onClick={actions.startOver}
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: '8px 20px', borderRadius: 4, cursor: 'pointer', border: '1px solid var(--rule)', background: 'var(--cream)', color: 'var(--ink-2)', letterSpacing: '0.03em' }}
          >
            ← Start over
          </button>
        </div>
      </div>
    );
  }

  // State 4: loading
  if (isLoading && ['validating', 'processing', 'analyzing', 'thinking', 'solving', 'writing_code'].includes(processingStage)) {
    return <PipelineLoading stage={processingStage} />;
  }

  // State 5: results
  if (processingStage === 'complete' && completePipelineResult) {
    return <PipelineResults result={completePipelineResult} onStartOver={actions.startOver} />;
  }

  // State 6: document ready — selection screen
  if (processingStage === 'complete' && document) {
    return (
      <DocumentSelectionScreen
        document={document}
        originalImageBase64={originalImageBase64}
        selectedProblemId={selectedProblemId}
        editedLatex={editedLatex}
        editedVisualContext={editedVisualContext}
        removeVisualContext={removeVisualContext}
        isLoading={isLoading}
        error={error}
        quota={quota}
        onSelectProblem={actions.selectProblem}
        onUpdateLatex={actions.updateEditedLatex}
        onUpdateVisualContext={actions.updateEditedVisualContext}
        onRemoveVisualContext={actions.removeSelectedVisualContext}
        onRestoreVisualContext={actions.restoreSelectedVisualContext}
        onRunPipeline={actions.runCompletePipeline}
        onBack={actions.startOver}
      />
    );
  }

  return <FileUpload onFileSelect={actions.handleFileUpload} error="An unexpected state occurred. Please start over." quota={quota} />;
};
