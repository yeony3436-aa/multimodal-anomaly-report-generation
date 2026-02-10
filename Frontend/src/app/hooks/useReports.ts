// src/app/hooks/useReports.ts
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../api/http";
import {
  fetchReports,
  fetchReportStats,
  fetchReportById,
  ReportListDTO,
  ReportListQuery,
  ReportStatsDTO,
  ReportDTO,
} from "../api/reportsApi";

function stableKey(obj: unknown) {
  return JSON.stringify(obj ?? {});
}

function useDebounced<T>(value: T, delayMs: number) {
  const [debounced, setDebounced] = useState(value);
  const key = useMemo(() => stableKey(value), [value]);

  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(t);
  }, [key, delayMs]);

  return debounced;
}

export function useReports(query: ReportListQuery, options?: { enabled?: boolean; debounceMs?: number }) {
  const enabled = options?.enabled ?? true;
  const debounceMs = options?.debounceMs ?? 250;

  const debouncedQuery = useDebounced(query, debounceMs);
  const queryKey = useMemo(() => stableKey(debouncedQuery), [debouncedQuery]);

  const [data, setData] = useState<ReportListDTO | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const ac = new AbortController();
    setLoading(true);
    setError(null);

    fetchReports(debouncedQuery, { signal: ac.signal })
      .then(setData)
      .catch((e) => {
        if ((e as any)?.name === "AbortError") return;
        setError(e as ApiError);
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });

    return () => ac.abort();
  }, [enabled, queryKey]);

  return { data, loading, error };
}

export function useReportStats() {
  const [stats, setStats] = useState<ReportStatsDTO | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const load = async () => {
    inFlight.current?.abort();
    const ac = new AbortController();
    inFlight.current = ac;

    try {
      setLoading(true);
      setError(null);
      const res = await fetchReportStats({ signal: ac.signal });
      setStats(res);
    } catch (e) {
      if ((e as any)?.name === "AbortError") return;
      setError(e as ApiError);
    } finally {
      if (!ac.signal.aborted) setLoading(false);
    }
  };

  return { stats, loading, error, load };
}

export function useReportDetail(id: number | null) {
  const [data, setData] = useState<ReportDTO | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    if (id == null) return;

    const ac = new AbortController();
    setLoading(true);
    setError(null);

    fetchReportById(id, { signal: ac.signal })
      .then(setData)
      .catch((e) => {
        if ((e as any)?.name === "AbortError") return;
        setError(e as ApiError);
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });

    return () => ac.abort();
  }, [id]);

  return { data, loading, error };
}
