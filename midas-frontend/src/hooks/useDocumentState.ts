// midas-frontend/src/hooks/useDocumentState.ts

import { useState, useCallback, useMemo, useEffect } from 'react';
import { DocumentState, QuotaStatus, CompletePipelineResponse } from '../types/api';
import { SimpleAPIService, handleAPIError } from '../services/SimpleAPIService';

const initialState: DocumentState = {
  document: null,
  documentId: null,
  originalImageBase64: null,
  selectedProblemId: null,
  editedLatex: '',
  editedVisualContext: '',
  removeVisualContext: false,
  isLoading: false,
  error: null,
  processingStage: 'idle',
  uploadedFile: null,
  processingMetadata: null,
  completePipelineResult: null,
  quota: null,
};

/**
 * Linus's Note: A single state hook is a pragmatic choice for this app size.
 * The original sin was putting business logic in the component. I've moved it here.
 * This hook now exposes `actions` that encapsulate state changes and API calls.
 * The component's job is to call an action, not to know how to perform it.
 */
export const useDocumentState = () => {
  const [state, setState] = useState<DocumentState>(initialState);

  const refreshQuota = useCallback(async (): Promise<QuotaStatus | null> => {
    try {
      const quota = await SimpleAPIService.getQuota();
      setState(prev => ({ ...prev, quota }));
      return quota;
    } catch (err) {
      console.warn('Failed to refresh quota:', err);
      return null;
    }
  }, []);

  useEffect(() => {
    refreshQuota();
  }, [refreshQuota]);

  // --- ACTIONS ---
  // These functions encapsulate the logic that was previously in FullVisionPipeline.
  const handleFileUpload = useCallback(async (file: File) => {
    setState(prev => ({ ...prev, uploadedFile: file, processingStage: 'uploading', error: null }));
    try {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = (reader.result as string).split(',')[1];
        setState(prev => ({ ...prev, originalImageBase64: base64 }));
        console.log("LOADED GOOD");
      };
      reader.onerror = () => {
        setState(prev => ({ ...prev, error: 'Failed to create file preview.', processingStage: 'error' }));
        console.log("NOT LOADED GOOD #1");
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setState(prev => ({ ...prev, error: 'Failed to create file preview.', processingStage: 'error' }));
        console.log("NOT LOADED GOOD #2");
    }
  }, []);

  const processUploadedFile = useCallback(async () => {
    if (!state.uploadedFile) return;

    setState(prev => ({ ...prev, processingStage: 'validating', isLoading: true, error: null }));

    try {
      const response = await SimpleAPIService.uploadDocument(state.uploadedFile);
      if (response.success && response.data) {
        await refreshQuota();
        setState(prev => ({
          ...prev,
          document: response.data.document,
          documentId: response.data.document_id,
          originalImageBase64: response.data.original_image_base64,
          processingMetadata: response.data.processing_metadata,
          processingStage: 'complete',
          isLoading: false,
        }));
      } else {
        console.log("SCREAM 1");
        await refreshQuota();
        setState(prev => ({ ...prev, error: response.message || 'Processing failed', processingStage: 'error', isLoading: false }));
      }
    } catch (err) {
      console.log("SCREAM 2");
      await refreshQuota();
      setState(prev => ({ ...prev, error: handleAPIError(err), processingStage: 'error', isLoading: false }));
    }
  }, [state.uploadedFile, refreshQuota]);

  const loadExampleDocument = useCallback(async (exampleId: string) => {
    setState(prev => ({
      ...prev,
      uploadedFile: null,
      originalImageBase64: null,
      processingMetadata: null,
      processingStage: 'processing',
      isLoading: true,
      error: null,
    }));

    try {
      const response = await SimpleAPIService.loadExampleDocument(exampleId);
      if (response.success && response.data) {
        setState(prev => ({
          ...prev,
          document: response.data.document,
          documentId: response.data.document_id,
          originalImageBase64: response.data.original_image_base64,
          processingMetadata: response.data.processing_metadata,
          processingStage: 'complete',
          isLoading: false,
        }));
      } else {
        setState(prev => ({ ...prev, error: response.message || 'Example could not be loaded', processingStage: 'error', isLoading: false }));
      }
    } catch (err) {
      setState(prev => ({ ...prev, error: handleAPIError(err), processingStage: 'error', isLoading: false }));
    }
  }, []);

  const runCompletePipeline = useCallback(async () => {
    if (!state.documentId || !state.selectedProblemId || !state.editedLatex.trim()) {
      setState(prev => ({ ...prev, error: 'A problem must be selected and not empty.' }));
      return;
    }

    setState(prev => ({ ...prev, processingStage: 'thinking', isLoading: true, error: null }));

    try {
      const response = await SimpleAPIService.runCompletePipeline({
        document_id: state.documentId,
        problem_id: state.selectedProblemId,
        edited_latex: state.editedLatex.trim(),
        visual_context_override: state.editedVisualContext.trim() || null,
        remove_visual_context: state.removeVisualContext,
      });

      if (response.success) {
        await refreshQuota();
        setState(prev => ({
          ...prev,
          completePipelineResult: response,
          processingStage: 'complete',
          isLoading: false,
        }));
      } else {
        await refreshQuota();
        setState(prev => ({ ...prev, completePipelineResult: response, error: response.message || 'Pipeline failed', processingStage: 'complete', isLoading: false }));
      }
    } catch (err) {
      await refreshQuota();
      const errorMsg = handleAPIError(err);
      const failedResponse: CompletePipelineResponse = { success: false, message: errorMsg, timestamp: new Date().toISOString(), data: null };
      setState(prev => ({ ...prev, completePipelineResult: failedResponse, error: errorMsg, processingStage: 'complete', isLoading: false }));
    }
  }, [state.documentId, state.selectedProblemId, state.editedLatex, state.editedVisualContext, state.removeVisualContext, refreshQuota]);

  const selectProblem = useCallback((problemId: string | null) => {
    setState(prev => {
      if (!prev.document) return prev;
      if (prev.selectedProblemId === problemId) {
        return { ...prev, selectedProblemId: null, editedLatex: '' }; // Deselect
      }
      const selectedProblem = prev.document.problems.find(p => p.problem_id === problemId);
      return {
        ...prev,
        selectedProblemId: problemId,
        editedLatex: selectedProblem ? selectedProblem.problem_text : '',
        editedVisualContext: selectedProblem?.visual_context_summary || '',
        removeVisualContext: false,
      };
    });
  }, []);

  const clearSelection = useCallback(() => {
    setState(prev => ({ ...prev, selectedProblemId: null, editedLatex: '', editedVisualContext: '', removeVisualContext: false }));
  }, []);

  const updateEditedLatex = useCallback((latex: string) => {
    setState(prev => ({ ...prev, editedLatex: latex }));
  }, []);

  const updateEditedVisualContext = useCallback((visualContext: string) => {
    setState(prev => ({ ...prev, editedVisualContext: visualContext, removeVisualContext: false }));
  }, []);

  const removeSelectedVisualContext = useCallback(() => {
    setState(prev => ({ ...prev, editedVisualContext: '', removeVisualContext: true }));
  }, []);

  const restoreSelectedVisualContext = useCallback(() => {
    setState(prev => {
      const selectedProblem = prev.document?.problems.find(p => p.problem_id === prev.selectedProblemId);
      return {
        ...prev,
        editedVisualContext: selectedProblem?.visual_context_summary || '',
        removeVisualContext: false,
      };
    });
  }, []);

  const startOver = useCallback(() => {
    setState(prev => ({
      ...initialState,
      quota: prev.quota,
    }));
  }, []);

  // --- MEMOIZED DERIVED STATE ---
  const selectedProblem = useMemo(() => {
    if (!state.document || !state.selectedProblemId) return null;
    return state.document.problems.find(p => p.problem_id === state.selectedProblemId) || null;
  }, [state.document, state.selectedProblemId]);

  const selectedBlockIds = useMemo(() => {
    return selectedProblem?.block_ids || [];
  }, [selectedProblem]);

  return {
    // Raw state values
    ...state,
    // Derived state
    selectedProblem,
    selectedBlockIds,
    hasSelection: state.selectedProblemId !== null,
    // Actions to manipulate state
    actions: {
      handleFileUpload,
      loadExampleDocument,
      processUploadedFile,
      runCompletePipeline,
      selectProblem,
      clearSelection,
      updateEditedLatex,
      updateEditedVisualContext,
      removeSelectedVisualContext,
      restoreSelectedVisualContext,
      startOver,
      cancelUpload: startOver, // Alias for clarity
      refreshQuota,
    },
  };
};

// You need to add `completePipelineResult` to your DocumentState type
// in midas-frontend/src/types/api.ts
/*
export interface DocumentState {
  //... all existing fields
  completePipelineResult: CompletePipelineResponse | null;
}
*/
