// midas-frontend/src/hooks/useDocumentState.ts

import { useState, useCallback, useMemo } from 'react';
import { DocumentState } from '../types/api';
import { SimpleAPIService, handleAPIError } from '../services/SimpleAPIService';
import { CompletePipelineResponse } from '../types/api';

const initialState: DocumentState = {
  document: null,
  documentId: null,
  originalImageBase64: null,
  selectedProblemId: null,
  editedLatex: '',
  isLoading: false,
  error: null,
  processingStage: 'idle',
  uploadedFile: null,
  completePipelineResult: null, // Added to state to hold the final result
};

/**
 * Linus's Note: A single state hook is a pragmatic choice for this app size.
 * The original sin was putting business logic in the component. I've moved it here.
 * This hook now exposes `actions` that encapsulate state changes and API calls.
 * The component's job is to call an action, not to know how to perform it.
 */
export const useDocumentState = () => {
  const [state, setState] = useState<DocumentState>(initialState);

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
        setState(prev => ({
          ...prev,
          document: response.data.document,
          documentId: response.data.document_id,
          // The base64 is already set, but we could re-sync from response if needed
          // originalImageBase64: response.data.original_image_base64,
          processingStage: 'complete',
          isLoading: false,
        }));
      } else {
        console.log("SCREAM 1");
        setState(prev => ({ ...prev, error: response.message || 'Processing failed', processingStage: 'error', isLoading: false }));
      }
    } catch (err) {
      console.log("SCREAM 2");
      setState(prev => ({ ...prev, error: handleAPIError(err), processingStage: 'error', isLoading: false }));
    }
  }, [state.uploadedFile]);

  const runCompletePipeline = useCallback(async () => {
    if (!state.documentId || !state.selectedProblemId || !state.editedLatex.trim()) {
      setState(prev => ({ ...prev, error: 'A problem must be selected and not empty.' }));
      return;
    }

    setState(prev => ({ ...prev, processingStage: 'thinking', isLoading: true, error: null })); // A more descriptive stage
    
    try {
      const response = await SimpleAPIService.runCompletePipeline({
        document_id: state.documentId,
        problem_id: state.selectedProblemId,
        edited_latex: state.editedLatex.trim(),
      });

      if (response.success) {
        setState(prev => ({
          ...prev,
          completePipelineResult: response,
          processingStage: 'complete',
          isLoading: false,
        }));
      } else {
         setState(prev => ({ ...prev, completePipelineResult: response, error: response.message || 'Pipeline failed', processingStage: 'complete', isLoading: false }));
      }
    } catch (err) {
      const errorMsg = handleAPIError(err);
      const failedResponse: CompletePipelineResponse = { success: false, message: errorMsg, timestamp: new Date().toISOString(), data: null };
      setState(prev => ({ ...prev, completePipelineResult: failedResponse, error: errorMsg, processingStage: 'complete', isLoading: false }));
    }
  }, [state.documentId, state.selectedProblemId, state.editedLatex]);

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
      };
    });
  }, []);

  const clearSelection = useCallback(() => {
    setState(prev => ({ ...prev, selectedProblemId: null, editedLatex: '' }));
  }, []);

  const updateEditedLatex = useCallback((latex: string) => {
    setState(prev => ({ ...prev, editedLatex: latex }));
  }, []);

  const startOver = useCallback(() => {
    setState(initialState);
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
      processUploadedFile,
      runCompletePipeline,
      selectProblem,
      clearSelection,
      updateEditedLatex,
      startOver,
      cancelUpload: startOver, // Alias for clarity
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