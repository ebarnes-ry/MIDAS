// midas-frontend/src/services/SimpleAPIService.ts

import {
  DocumentUploadResponse,
  CompletePipelineResponse,
  UserSelectionRequest,
  ReasoningExplainRequest,
  ReasoningExplainResponse,
  FeedbackRequest,
  FeedbackResponse,
  HealthStatus
} from '../types/api';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const DEFAULT_TIMEOUT = 300000; // 5 minutes

export const handleAPIError = (error: any): string => {
  if (error instanceof Error && error.name === 'AbortError') {
    return 'The request timed out. The server is busy or the document is taking a long time to process.';
  }
  if (error.response?.data?.detail) return error.response.data.detail;
  if (error.message) return error.message;
  return 'An unexpected error occurred. Check the backend server logs for details.';
};

export class SimpleAPIService {

  /**
   * Linus's Note: I've abstracted the fetch logic.
   * You had duplicated timeout and error handling code. That's a sign of bad design.
   * This one private method now handles it for all API calls. Don't Repeat Yourself.
   */
  private static async _fetchWithTimeout(url: string, options: RequestInit, timeout: number = DEFAULT_TIMEOUT): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      console.error('API Service Error:', error);
      throw error;
    }
  }

  static async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this._fetchWithTimeout(`${API_BASE}/api/v1/vision/upload`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }
  
  static async runCompletePipeline(request: UserSelectionRequest): Promise<CompletePipelineResponse> {
    const response = await this._fetchWithTimeout(`${API_BASE}/api/v1/vision/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return response.json();
  }

  static async explainStep(request: ReasoningExplainRequest): Promise<ReasoningExplainResponse> {
    const response = await this._fetchWithTimeout(`${API_BASE}/api/v1/reasoning/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    }, 60000); // Shorter timeout for this faster operation
    return response.json();
  }
  
  static async getFeedback(request: FeedbackRequest): Promise<FeedbackResponse> {
    const response = await this._fetchWithTimeout(`${API_BASE}/api/v1/reasoning/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    }, 60000);
    return response.json();
  }

  static async healthCheck(): Promise<HealthStatus> {
    const response = await this._fetchWithTimeout(`${API_BASE}/health/`, {}, 5000);
    return response.json();
  }
}