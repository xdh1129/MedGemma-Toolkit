export interface AnalyzeResponse {
  vlm_output: string;
  llm_report: string;
}

const normalizeBaseUrl = (value?: string) => {
  if (!value) {
    return '/api';
  }
  return value.endsWith('/') ? value.slice(0, -1) : value;
};

const API_BASE_URL = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL as string | undefined);
const ANALYZE_ENDPOINT = `${API_BASE_URL}/analyze/`;

export const analyzeCase = async (prompt: string, image?: File | null): Promise<AnalyzeResponse> => {
  const formData = new FormData();
  formData.append('prompt', prompt);
  if (image) {
    formData.append('image', image);
  }

  const response = await fetch(ANALYZE_ENDPOINT, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const detail = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(detail?.detail || 'Request failed');
  }

  return response.json() as Promise<AnalyzeResponse>;
};
