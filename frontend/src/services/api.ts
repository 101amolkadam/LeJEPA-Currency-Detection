import axios from "axios";
import type { AnalysisResult, HistoryResponse } from "@/types";

const api = axios.create({
  baseURL: "http://127.0.0.1:8001/api/v1",
  // baseURL: "https://validcash.duckdns.org/api/v1",
  headers: { "Content-Type": "application/json" },
  timeout: 60000,
});

export async function analyzeImage(
  base64Image: string,
  source: "upload" | "camera" = "upload"
): Promise<AnalysisResult> {
  const response = await api.post<AnalysisResult>("/analyze", {
    image: base64Image,
    source,
  });
  return response.data;
}

export async function getHistory(params: {
  page?: number;
  limit?: number;
  filter?: "all" | "real" | "fake";
} = {}): Promise<HistoryResponse> {
  const response = await api.get<HistoryResponse>("/analyze/history", { params });
  return response.data;
}

export async function getAnalysisById(id: number): Promise<AnalysisResult> {
  const response = await api.get<AnalysisResult>(`/analyze/history/${id}`);
  return response.data;
}

export async function deleteAnalysis(id: number): Promise<void> {
  await api.delete(`/analyze/history/${id}`);
}
