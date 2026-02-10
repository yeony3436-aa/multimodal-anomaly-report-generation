// src/app/data/llavaMapper.ts
import type { AnomalyCase, ActionLog } from "./mockData";

/**
 * 백엔드 응답 스키마가 확정되지 않은 상태를 고려해서
 * 다양한 키/중첩 구조를 "안전하게" 흡수하는 형태로 작성했습니다.
 */
export type LlavaReport = {
  id?: number | string;
  filename?: string;
  image_path?: string;

  dataset?: string;
  category?: string;

  ground_truth?: string | null;

  // 백엔드가 decision을 주는 경우 우선 사용
  decision?: string;

  confidence?: number | null;
  inference_time?: number | null; // seconds
  datetime?: string;

  // 백엔드가 주는 결함 필드가 있을 수 있음
  defect_type?: string;
  location?: string;
  severity?: string;

  anomaly_score?: number;

  recommendation?: string;
  summary?: string;

  anomaly_info?: {
    has_defect?: boolean;
    defect_type?: string;
    location?: string;
    severity?: string;
    score?: number;
  };

  product_info?: any;
};

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

  // 이미 OK/NG/REVIEW라면 그대로
  if (d === "ok") return "OK";
  if (d === "ng") return "NG";
  if (d === "review") return "REVIEW";

  // 백엔드가 normal/anomaly/review 같은 값을 준다면 흡수
  if (d === "normal") return "OK";
  if (d === "anomaly") return "NG";

  // fallback: GT 기반 (기존 로직 유지)
  if (!gt) return "REVIEW";
  if (gt === "good") return "OK";
  if (gt === "opened") return "REVIEW";
  return "NG";
}

function toDefectTypeFromGT(gt?: string | null): string {
  switch (gt) {
    case "good":
      return "none";
    case "cap_open":
    case "opened":
      return "seal_issue";
    case "deformation":
      return "missing_component";
    case "structural_anomalies":
      return "crack";
    default:
      return "scratch";
  }
}

function normalizeSeverity(raw?: string, decision?: "OK" | "NG" | "REVIEW"): "low" | "med" | "high" {
  if (decision && decision !== "NG") return "low";
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "high") return "high";
  if (s === "med" || s === "medium") return "med";
  if (s === "low") return "low";
  return "low";
}

function toSeverity(defectType: string, decision: "OK" | "NG" | "REVIEW"): "low" | "med" | "high" {
  if (decision !== "NG") return "low";
  if (defectType === "crack") return "high";
  if (defectType === "missing_component") return "med";
  if (defectType === "seal_issue") return "med";
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

// ✅ 매핑 함수 (product_class 제거 완료)
export function mapLlavaReportsToAnomalyCases(raw: LlavaReport[]): AnomalyCase[] {
  return raw.map((r, idx) => {
    const ts = parseDatetime(r.datetime);

    const dataset = r.dataset ?? "UNKNOWN";
    const category = r.category ?? "unknown";
    const filename = r.filename ?? `unknown_${idx}.jpg`;

    const decision = normalizeDecision(r.decision, r.ground_truth);

    const defect_type =
      r.anomaly_info?.defect_type ??
      r.defect_type ??
      toDefectTypeFromGT(r.ground_truth);

    const line_id = toLineId(`${dataset}-${category}`);
    const shift = toShift(ts);

    const loc =
      (decision === "OK"
        ? "none"
        : r.anomaly_info?.location ?? r.location ?? toLocation(decision, `${filename}-${category}`)) as string;

    // severity: 백엔드가 주면 우선, 없으면 heuristic
    const severity =
      normalizeSeverity(r.anomaly_info?.severity ?? r.severity, decision) ??
      toSeverity(defect_type, decision);

    // anomaly_score: 백엔드가 주면 우선, 없으면 fallback
    const anomaly_score =
      typeof r.anomaly_info?.score === "number"
        ? r.anomaly_info!.score
        : typeof r.anomaly_score === "number"
          ? r.anomaly_score
          : toAnomalyScoreFallback(decision, `${filename}-${dataset}-${category}`);

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

      model_name: "EfficientAD", // App.tsx에서 activeModel로 덮어씌워짐
      model_version: "v1.0.0",
      inference_time_ms: typeof r.inference_time === "number" ? Math.round(r.inference_time * 1000) : 0,

      llm_summary: (r.summary && r.summary.trim()) ? r.summary.trim() : toKoreanSummary(decision, defect_type, loc),
      llm_structured_json: { source: r },

      operator_note: (r.recommendation ?? "").trim(),
      action_log: toActionLog(decision, ts),
    };
  });
}
