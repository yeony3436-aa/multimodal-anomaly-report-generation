// src/app/api/reportsApi.ts
import { apiRequest, apiUrl, QueryParams } from "./http";

/**
 * 모델명이 LLaVA -> 다른 것으로 바뀌어도 ("/"")
 * REPORTS_BASE만 바꾸면 프론트 전반은 그대로 유지됨
 */
const REPORTS_BASE = "/llava"; 

export type ReportDTO = {
  id: number;
  filename: string;
  image_path: string;
  dataset: string;
  category: string;
  ground_truth: string | null;
  decision: string;
  confidence: number | null;

  has_defect: number;
  defect_type: string;
  location: string;
  severity: string;

  defect_description: string;
  possible_cause: string;
  product_description: string;

  summary: string;
  impact: string;
  recommendation: string;

  inference_time: number | null;
  datetime: string;
};

export type ReportListDTO = {
  items: ReportDTO[];
  total: number;
};

export type ReportStatsDTO = {
  total: number;
  by_dataset: Record<string, number>;
  by_category: Record<string, number>;
  by_decision: Record<string, number>;
};

export type ReportListQuery = {
  limit?: number;
  offset?: number;
  dataset?: string;
  category?: string;
  decision?: string;
};

async function fetchReportsRaw(query: ReportListQuery, opts?: { signal?: AbortSignal }) {
  return apiRequest<unknown>(`${REPORTS_BASE}/reports`, {
    query: query as QueryParams,
    signal: opts?.signal,
  });
}

/**
 * 백엔드가 배열로 주든 {items,total}로 주든
 * 프론트는 항상 {items,total}로 받도록 정규화
 */
export async function fetchReports(
  query: ReportListQuery,
  opts?: { signal?: AbortSignal }
): Promise<ReportListDTO> {
  const data = await fetchReportsRaw(query, opts);

  const items = Array.isArray(data)
    ? (data as ReportDTO[])
    : ((data as any)?.items ?? []) as ReportDTO[];

  const total = Array.isArray(data)
    ? items.length
    : Number((data as any)?.total ?? items.length);

  return { items, total };
}

export async function fetchReportStats(opts?: { signal?: AbortSignal }) {
  return apiRequest<ReportStatsDTO>(`${REPORTS_BASE}/stats`, { signal: opts?.signal });
}

export async function fetchReportById(id: number, opts?: { signal?: AbortSignal }) {
  return apiRequest<ReportDTO>(`${REPORTS_BASE}/reports/${encodeURIComponent(String(id))}`, {
    signal: opts?.signal,
  });
}

export function getReportJsonUrl(id: number) {
  return apiUrl(`${REPORTS_BASE}/reports/${encodeURIComponent(String(id))}/json`);
}
