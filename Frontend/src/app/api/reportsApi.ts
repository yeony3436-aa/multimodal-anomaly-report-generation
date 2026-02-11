// src/app/api/reportsApi.ts
import { apiRequest, QueryParams } from "./http";

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

  inference_time: number | null; // seconds
  datetime: string;

  heatmap_path?: string | null;
  overlay_path?: string | null;

};

export type ReportListDTO = {
  items: ReportDTO[];
  total: number;
};

export type ReportListQuery = {
  limit?: number;
  offset?: number;
  dataset?: string;
  category?: string;
  decision?: string;

  // 프론트에서만 먼저 “붙일 준비” (백엔드 구현은 나중에)
  date_from?: string;
  date_to?: string;
};

async function fetchReportsRaw(
  query: ReportListQuery,
  opts?: { signal?: AbortSignal }
) {
  return apiRequest<unknown>(`${REPORTS_BASE}/reports`, {
    query: query as QueryParams,
    signal: opts?.signal,
  });
}

export async function fetchReports(
  query: ReportListQuery,
  opts?: { signal?: AbortSignal }
): Promise<ReportListDTO> {
  const data = await fetchReportsRaw(query, opts);

  const items = Array.isArray(data)
    ? (data as ReportDTO[])
    : (((data as any)?.items ?? []) as ReportDTO[]);

  const total = Array.isArray(data)
    ? items.length
    : Number((data as any)?.total ?? items.length);

  return { items, total };
}

export async function fetchReportsAll(
  baseQuery?: Omit<ReportListQuery, "limit" | "offset">,
  opts?: { signal?: AbortSignal; pageSize?: number; maxItems?: number }
): Promise<ReportDTO[]> {
  const pageSize = opts?.pageSize ?? 500;
  const maxItems = opts?.maxItems ?? 5000;

  let offset = 0;
  let out: ReportDTO[] = [];
  let total = Infinity;

  while (offset < total && out.length < maxItems) {
    const { items, total: t } = await fetchReports(
      { ...(baseQuery ?? {}), limit: pageSize, offset },
      { signal: opts?.signal }
    );

    total = Number.isFinite(t) ? t : Infinity;
    out = out.concat(items);

    if (items.length === 0) break;
    offset += items.length;
  }

  return out.slice(0, maxItems);
}
