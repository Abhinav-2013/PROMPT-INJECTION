import type {
  AnalyzeRequest,
  AnalyzeResponse,
  APIError,
  APIInfoResponse,
  DocumentScanResponse as DocumentScanResponseType,
  ExplainResponse,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
} from './types'

// Re-export the document response type so pages can import it from athsApi.ts
export type { DocumentScanResponse } from './types'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000'
).replace(/\/+$/, '')

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        Accept: 'application/json',

        ...(options?.body instanceof FormData
          ? {}
          : options?.body
            ? {
                'Content-Type': 'application/json',
              }
            : {}),

        ...options?.headers,
      },
    })

    if (!response.ok) {
      let message = 'The request could not be completed.'

      try {
        const errorBody = (await response.json()) as {
          detail?: string
          message?: string
        }

        message =
          errorBody.detail ||
          errorBody.message ||
          message
      } catch {
        // Keep the default message.
      }

      const error: APIError = {
        message,
        status: response.status,
      }

      throw error
    }

    return (await response.json()) as T
  } catch (error) {
    if (
      typeof error === 'object' &&
      error !== null &&
      'message' in error
    ) {
      throw error
    }

    throw {
      message: 'Detection service unavailable.',
    } satisfies APIError
  }
}

/**
 * Get basic API information.
 */
export async function getAPIInfo(): Promise<APIInfoResponse> {
  return request<APIInfoResponse>('/')
}

/**
 * Check whether the ATHS backend is healthy.
 */
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

/**
 * Analyze a prompt using the complete backend
 * detection pipeline:
 *
 * ML + Rules + Semantic + Threat Fusion + Hypothesis
 */
export async function analyzePrompt(
  payload: AnalyzeRequest,
): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>('/analyze', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/**
 * Submit human feedback for a prediction.
 */
export async function submitFeedback(
  payload: FeedbackRequest,
): Promise<FeedbackResponse> {
  return request<FeedbackResponse>('/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/**
 * Scan a PDF, DOCX, or TXT document.
 *
 * The backend endpoint is:
 * POST /documents/scan
 *
 * The file must be sent as multipart/form-data.
 */
export async function scanDocument(
  files: File[],
): Promise<DocumentScanResponseType> {
  const formData = new FormData()

  files.forEach((file) => {
    formData.append('files', file)
  })

  return request<DocumentScanResponseType>('/documents/scan', {
    method: 'POST',
    body: formData,
  })
}

/**
 * Request SHAP-based explainability from the backend.
 *
 * The backend endpoint is:
 * POST /explain
 */
export async function explainPrompt(
  text: string,
): Promise<ExplainResponse> {
  return request<ExplainResponse>('/explain', {
    method: 'POST',
    body: JSON.stringify({
      text,
    }),
  })
}

/**
 * Get the configured backend URL.
 */
export function getAPIBaseURL(): string {
  return API_BASE_URL
}