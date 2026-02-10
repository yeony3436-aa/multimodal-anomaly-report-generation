// src/app/api/http.ts
export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type QueryParams = Record<
  string,
  string | number | boolean | null | undefined
>;

export class ApiError extends Error {
  status?: number;
  url?: string;
  body?: unknown;

  constructor(
    message: string,
    opts?: { status?: number; url?: string; body?: unknown }
  ) {
    super(message);
    this.name = "ApiError";
    this.status = opts?.status;
    this.url = opts?.url;
    this.body = opts?.body;
  }
}

function buildQueryString(query?: QueryParams) {
  if (!query) return "";
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === "") continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export function apiUrl(path: string, query?: QueryParams) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${p}${buildQueryString(query)}`;
}

function fastapiDetailMessage(payload: unknown): string | null {

  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as any).detail;
  if (!detail) return null;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    const first = detail[0];
    if (typeof first === "string") return first;
    if (first && typeof first === "object") {
      return first.msg ?? first.message ?? JSON.stringify(first);
    }
    return JSON.stringify(detail);
  }

  if (typeof detail === "object") return JSON.stringify(detail);
  return null;
}

export async function apiRequest<T>(
  path: string,
  opts?: {
    method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
    query?: QueryParams;
    body?: unknown;
    headers?: Record<string, string>;
    signal?: AbortSignal;
  }
): Promise<T> {
  const url = apiUrl(path, opts?.query);

  const hasBody = opts?.body !== undefined;
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(opts?.headers ?? {}),
  };

  // body가 있을 때만 Content-Type을 붙여 preflight 가능성을 줄임
  if (hasBody) headers["Content-Type"] = "application/json";

  const res = await fetch(url, {
    method: opts?.method ?? "GET",
    headers,
    body: hasBody ? JSON.stringify(opts!.body) : undefined,
    signal: opts?.signal,
  });

  const contentType = res.headers.get("content-type") ?? "";
  const isJson = contentType.includes("application/json");

  let payload: unknown = null;
  try {
    payload = isJson ? await res.json() : await res.text();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    const detail = fastapiDetailMessage(payload);
    const msg = detail ? `${res.status} ${detail}` : `HTTP ${res.status}`;
    throw new ApiError(msg, { status: res.status, url, body: payload });
  }

  return payload as T;
}
