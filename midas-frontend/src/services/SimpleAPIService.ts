// midas-frontend/src/services/SimpleAPIService.ts

import {
  DocumentUploadResponse,
  CompletePipelineResponse,
  UserSelectionRequest,
  ReasoningExplainRequest,
  ReasoningExplainResponse,
  FeedbackRequest,
  FeedbackResponse,
  HealthStatus,
  QuotaStatus,
} from '../types/api';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const DEFAULT_TIMEOUT = 300000; // 5 minutes

const CLIENT_ID_KEY = 'midas_client_id';

function getClientId(): string {
  let id = localStorage.getItem(CLIENT_ID_KEY);

  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(CLIENT_ID_KEY, id);
  }

  return id;
}

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
  private static async _fetchWithTimeout(url: string, options: RequestInit, timeout: number = DEFAULT_TIMEOUT): 
    Promise<Response> {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const headers = new Headers(options.headers || {});
      headers.set('X-MIDAS-Client-ID', getClientId());

      try {
        const response = await fetch(url, {
          ...options,
          headers,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const detail = errorData.detail;

          if (response.status === 429 && detail?.error === 'quota_exceeded') {
            throw new Error(
              `Demo limit reached: ${detail.used}/${detail.limit} ${detail.kind} runs used today.`
            );
          }

          throw new Error(
            typeof detail === 'string'
              ? detail
              : detail?.error || detail?.message || `HTTP ${response.status}: ${response.statusText}`
          );
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

  static async getQuota(): Promise<QuotaStatus> {
    const response = await this._fetchWithTimeout(
      `${API_BASE}/api/v1/demo/quota`,
      { method: 'GET' },
      5000
    );
    return response.json();
  }

  static async healthCheck(): Promise<HealthStatus> {
    const response = await this._fetchWithTimeout(`${API_BASE}/health/`, {}, 5000);
    return response.json();
  }
}