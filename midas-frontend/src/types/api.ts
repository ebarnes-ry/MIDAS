// midas-frontend/src/types/api.ts

export interface APIBlock {
  id: string;
  block_type: string;
  html: string;
  polygon: number[];
  bbox: number[];
  is_editable: boolean;
  latex_content: string | null;
  cropped_image: string | null; 
  image_description: string | null;
}

export type ProcessingStage =
  | 'idle' | 'uploading' |'validating'| 'processing' | 'analyzing'
  | 'thinking' | 'solving' | 'writing_code' | 'complete' | 'error';


export interface APIProblem {
  problem_id: string;
  problem_text: string; // <-- This is the new source of truth for the editor
  block_ids: string[];  // This is just for highlighting
  problem_type?: string | null;
  figure_references?: string[];
  visual_context_required?: boolean;
  visual_context_attached?: boolean;
  visual_context_summary?: string | null;
  visual_context_description_count?: number;
  problem_input_complete?: boolean;
  missing_problem_content?: boolean;
  missing_content_reason?: string | null;
  extraction_recovery_source?: string | null;
}

export interface APIDocument {
  blocks: APIBlock[];
  problems: APIProblem[];
  total_blocks: number;
  editable_blocks: number;
  images: Record<string, string>;
  dimensions: [number, number];
  metadata: Record<string, any>;
}

export interface DocumentUploadResponse {
  success: boolean;
  message: string;
  timestamp: string;
  data: {
    document_id: string;
    document: APIDocument;
    original_image_base64: string;
    processing_time: number;
    processing_metadata: Record<string, any>;
  };
}

export interface UserSelectionRequest {
  document_id: string;
  problem_id: string; 
  edited_latex: string;
  visual_context_override?: string | null;
  remove_visual_context?: boolean;
}

export interface APIVisualElement {
  description: string;
  visual_type: string;
  data: any;
}

export interface APIVisualContext {
  elements: APIVisualElement[];
  summary: string | null;
  contains_essential_info: boolean;
}

export interface VisionAnalysisResponse {
  success: boolean;
  message: string;
  timestamp: string;
  data: {
    problem_statement: string;
    visual_context: APIVisualContext | null;
    processing_time: number;
    analysis_metadata: Record<string, any>;
  };
}

export interface VerificationRepairAttempt {
  attempt: number;
  type: string;
  reason: string;
  success: boolean;
  processing_time: number;
  error_message: string | null;
}

export interface RepairAttempt {
  attempt_number: number;
  reasoning_steps: ReasoningStepResponse[];
  final_answer: string;
  verification_status: string;
  steps_verified: number;
  steps_failed: number;
}

export interface TrajectoryOutcome {
  final_status: string;
  attempt_count: number;
  difficulty_signal: number;
}

export interface StepVerificationResponse {
  step_number: number;
  description: string;
  verified: boolean;
  note: string;
}

export interface VerificationErrorResponse {
  error_type: string;
  message: string;
  line_number?: number;
  problematic_code?: string;
  suggested_fix?: string;
}

export interface CompletePipelineResponse {
  success: boolean;
  message: string;
  timestamp: string;
  data: {
    vision: {
      problem_statement: string;
      visual_context: APIVisualContext | null;
      processing_time: number;
      metadata: Record<string, any>;
    };
    reasoning: {
      original_problem: string;
      steps: ReasoningStepResponse[];
      worked_solution: string;
      final_answer: string;
      think_reasoning: string;
      processing_time: number;
      metadata: Record<string, any>;
    };
    verification: {
      original_problem: string;
      worked_solution: string;
      final_answer: string;
      generated_code: string;
      processing_time: number;
      status: string;
      confidence_score: number;
      answer_match: boolean | null;
      errors: Array<{
        error_type: string;
        message: string;
        line_number?: number;
        problematic_code?: string;
        suggested_fix?: string;
        traceback?: string;
      }>;
      repair_history: RepairAttempt[];
      step_verifications: StepVerificationResponse[];
      metadata: Record<string, any> & {
        reasoning_repair_attempts: number;
        codegen_repair_attempts: number;
      };
    };
    total_processing_time: number;
  } | null;
}

export interface DocumentState {
  document: APIDocument | null;
  documentId: string | null;
  originalImageBase64: string | null;
  // selectedBlockIds is GONE. We now track the selected problem.
  selectedProblemId: string | null;
  editedLatex: string;
  editedVisualContext: string;
  removeVisualContext: boolean;
  isLoading: boolean;
  error: string | null;
  processingStage: ProcessingStage;
  uploadedFile: File | null;
  completePipelineResult: CompletePipelineResponse | null;
}

export interface ReasoningStepResponse {
  step_number: number;
  claim: string;
  justification: string;
  latex_expression?: string;
  verification_status: boolean | null;
  verification_note?: string;
  feedback?: string;
}

export interface FeedbackRequest {
  problem_statement: string;
  step_number: number;
  claim: string;
  latex_expression?: string;
  justification: string;
  verification_note?: string;
}

export interface FeedbackResponse {
  success: boolean;
  data?: { feedback: string };
  error?: string;
}

export interface ReasoningExplainRequest {
  problem_statement: string;
  worked_solution: string;
  step_text: string;
}

export interface ReasoningExplainResponse {
  success: boolean;
  message: string;
  data?: {
    explanation: string;
    processing_time: number;
  };
}
export interface HealthStatus {
  status: string;
  version: string;
  uptime: number;
  dependencies: Record<string, string>;
}
