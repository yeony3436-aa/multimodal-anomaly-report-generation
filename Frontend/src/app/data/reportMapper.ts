// src/app/data/reportMapper.ts
import type { AnomalyCase, ActionLog } from "./mockData";
import type { ReportDTO } from "../api/reportsApi";

/**
 * 백엔드 ReportDTO -> 프론트 AnomalyCase 매핑
 * (모델명이 LLaVA든 아니든 상관없이 "보고서/케이스"라는 도메인으로만 유지)
 */

// 매핑용 상수
const PACKAGING_CLASS_LABEL: Record<string, string> = {
  cigarette_box: "cigarette box",
  drink_bottle: "drink bottle",
  drink_can: "drink can",
  food_bottle: "food_bottle",
  food_box: "food box",
  food_package: "food package",
  breakfast_box: "breakfast box",
  juice_bottle: "juice bottle",
  pushpins: "pushpins",
  screw_bag: "screw bag",
};

const LINES = ["LINE-A-01", "LINE-B-02", "LINE-C-03"];
const LOCS = ["top-left", "top-right", "bottom-left", "bottom-right", "center"] as const;

function hash01(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}

function parseDatetime(dt?: string): Date {
  if (!dt) return new Date();
  const cleaned = dt.replace(/(\.\d{3})\d+/, "$1");
  const d = new Date(cleaned);
  return isNaN(d.getTime()) ? new Date() : d;
}

function toProductGroup(category: string): string {
  return PACKAGING_CLASS_LABEL[category] ?? category.replace(/_/g, " ");
}

function toLineId(seed: string): string {
  const v = hash01(seed);
  return LINES[Math.floor(v * LINES.length) % LINES.length];
}

function toShift(d: Date): string {
  const hour = d.getHours();
  return hour >= 7 && hour < 19 ? "주간" : "야간";
}

function normalizeDecision(decisionRaw?: string, gt?: string | null): "OK" | "NG" | "REVIEW" {
  const d = (decisionRaw ?? "").trim().toLowerCase();

  if (d === "ok") return "OK";
  if (d === "ng") return "NG";
  if (d === "review") return "REVIEW";

  if (d === "normal") return "OK";
  if (d === "anomaly") return "NG";

  if (!gt) return "REVIEW";
  if (gt === "good") return "OK";
  if (gt === "opened") return "REVIEW";
  return "NG";
}

function normalizeSeverity(raw?: string, decision?: "OK" | "NG" | "REVIEW"): "low" | "med" | "high" {
  if (decision && decision !== "NG") return "low";
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "high") return "high";
  if (s === "med" || s === "medium") return "med";
  if (s === "low") return "low";
  return "low";
}

function toLocation(decision: "OK" | "NG" | "REVIEW", seed: string): string {
  if (decision === "OK") return "none";
  const v = hash01(seed);
  return LOCS[Math.floor(v * LOCS.length) % LOCS.length];
}

function toAffectedAreaPct(severity: "low" | "med" | "high", seed: string, decision: "OK" | "NG" | "REVIEW"): number {
  if (decision === "OK") return 0;
  const v = hash01(seed);
  if (severity === "high") return 15 + v * 12;
  if (severity === "med") return 6 + v * 8;
  return 1 + v * 4;
}

function toAnomalyScoreFallback(decision: "OK" | "NG" | "REVIEW", seed: string): number {
  const v = hash01(seed);
  if (decision === "OK") return Math.min(0.55, 0.25 + 0.30 * v);
  if (decision === "REVIEW") return Math.min(0.74, 0.66 + 0.08 * v);
  return Math.min(0.99, 0.80 + 0.19 * v);
}

function toDefectConfidence(decision: "OK" | "NG" | "REVIEW", seed: string): number {
  if (decision === "OK") return 0;
  const v = hash01(seed);
  return Math.min(0.99, 0.75 + 0.20 * v);
}

function toKoreanSummary(decision: "OK" | "NG" | "REVIEW", defectType: string, location: string): string {
  if (decision === "OK") return "정상 제품으로 판정되었습니다. 이상 징후가 발견되지 않았습니다.";
  if (decision === "REVIEW") return "경계 케이스입니다. 육안 재검토를 통해 판정을 확정해 주세요.";

  const defectKo =
    defectType === "seal_issue" ? "실링 불량" :
    defectType === "contamination" ? "오염" :
    defectType === "crack" ? "파손/균열" :
    defectType === "missing_component" ? "구성요소 누락" :
    defectType === "scratch" ? "스크래치" : defectType;

  const locKo =
    location === "top-left" ? "상단 좌측" :
    location === "top-right" ? "상단 우측" :
    location === "bottom-left" ? "하단 좌측" :
    location === "bottom-right" ? "하단 우측" :
    location === "center" ? "중앙" : "미상";

  return `${locKo} 영역에서 ${defectKo}가 감지되었습니다. 불량으로 분류됩니다.`;
}

function toActionLog(decision: "OK" | "NG" | "REVIEW", ts: Date): ActionLog[] {
  const base = ts.getTime();
  if (decision === "OK") return [{ who: "System", when: new Date(base + 1000), what: "자동 승인" }];
  if (decision === "REVIEW") return [{ who: "박철수", when: new Date(base + 60_000), what: "재검 요청" }];
  return [
    { who: "김민수", when: new Date(base + 60_000), what: "불량 확정" },
    { who: "이영희", when: new Date(base + 120_000), what: "메모 추가" },
  ];
}

export function mapReportsToAnomalyCases(raw: ReportDTO[]): AnomalyCase[] {
  return raw.map((r, idx) => {
    const ts = parseDatetime(r.datetime);

    const dataset = r.dataset ?? "UNKNOWN";
    const category = r.category ?? "unknown";
    const filename = r.filename ?? `unknown_${idx}.jpg`;

    const decision = normalizeDecision(r.decision, r.ground_truth);

    const defect_type = r.defect_type ?? "scratch";
    const line_id = toLineId(`${dataset}-${category}`);
    const shift = toShift(ts);

    const loc =
      decision === "OK"
        ? "none"
        : (r.location ?? toLocation(decision, `${filename}-${category}`));

    const severity = normalizeSeverity(r.severity, decision);

    // 백엔드가 anomaly_score를 직접 주지 않는 구조라 fallback
    const anomaly_score = toAnomalyScoreFallback(decision, `${filename}-${dataset}-${category}`);

    const affected_area_pct = toAffectedAreaPct(severity, `${filename}-${category}`, decision);
    const defect_confidence = toDefectConfidence(decision, `${filename}-${category}`);

    const id = `CASE-${dataset}-${category}-${filename}`;

    return {
      id,
      timestamp: ts,
      line_id,
      shift,

      product_group: toProductGroup(category),
      image_id: filename,

      decision,
      anomaly_score,
      threshold: 0.65,

      defect_type,
      defect_confidence,
      location: decision === "OK" ? "none" : loc,
      affected_area_pct,
      severity,

      model_name: "EfficientAD", // App.tsx에서 activeModel로 덮어쓰기
      model_version: "v1.0.0",
      inference_time_ms: typeof r.inference_time === "number" ? Math.round(r.inference_time * 1000) : 0,

      llm_summary: (r.summary && r.summary.trim())
        ? r.summary.trim()
        : toKoreanSummary(decision, defect_type, loc),

      llm_structured_json: { source: r },
      operator_note: (r.recommendation ?? "").trim(),
      action_log: toActionLog(decision, ts),
    };
  });
}
