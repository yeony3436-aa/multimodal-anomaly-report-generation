// src/app/data/sampleMapper.ts
import { mapSampleReportsToAnomalyCases } from './data/sampleMapper';
import type { SampleReport } from './data/sampleMapper';


export type SampleReport = {
  filename: string;
  image_path: string;
  dataset: string;
  category: string;
  ground_truth: string;
  datetime: string; // e.g. "2026-02-04T08:28:52.272068"
  inference_time: number; // e.g. 5.97 (단위 불명확 → 아래에서 ms로 변환)
  product_info: any;
  anomaly_info: {
    has_defect: boolean; // 샘플은 false지만, 우리는 ground_truth로 재구성
    defect_type: string;
    location: string;
    severity: string;
  };
  decision: string; // "normal" (샘플 고정)
  confidence: number; // 샘플은 1.0
  summary: string;
  impact: string;
  recommendation: string;
};

const FOOD_CATS = new Set([
  "breakfast_box",
  "drink_bottle",
  "drink_can",
  "food_bottle",
  "food_box",
  "food_package",
  "juice_bottle",
]);

const CATEGORY_LABEL_KO: Record<string, string> = {
  cigarette_box: "담배갑",
  screw_bag: "나사봉투",
  pushpins: "압정",
  breakfast_box: "아침식품 박스",
  drink_bottle: "음료 병",
  drink_can: "음료 캔",
  food_bottle: "식품 병",
  food_box: "식품 박스",
  food_package: "식품 포장",
  juice_bottle: "주스 병",
};

const LINES = ["LINE-A-01", "LINE-B-02", "LINE-C-03"];
const LOCS = ["top-left", "top-right", "bottom-left", "bottom-right", "center"] as const;

function hash01(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  // 0~1
  return (h >>> 0) / 4294967295;
}

function parseDatetime(dt?: string): Date {
  if (!dt) return new Date();
  // JS Date는 마이크로초(6자리) 파싱이 불안정할 수 있어서 ms 3자리로 자릅니다.
  const cleaned = dt.replace(/(\.\d{3})\d+/, "$1");
  const d = new Date(cleaned);
  return isNaN(d.getTime()) ? new Date() : d;
}

function toProductGroup(category: string): "Food" | "Household" {
  return FOOD_CATS.has(category) ? "Food" : "Household";
}

function toProductClass(category: string): string {
  return CATEGORY_LABEL_KO[category] ?? category;
}

function toLineId(seed: string): string {
  const v = hash01(seed);
  return LINES[Math.floor(v * LINES.length) % LINES.length];
}

function toShift(d: Date): string {
  const hour = d.getHours();
  return hour >= 7 && hour < 19 ? "주간" : "야간";
}

/**
 * ✅ 샘플 anomaly_info가 전부 비어있어서,
 * ground_truth로 defect_type / decision을 재구성합니다.
 * FilterBar 옵션(seal_issue/contamination/crack/dent/scratch/none)과 맞춰줍니다.
 */
function toDefectTypeFromGT(gt: string): string {
  switch (gt) {
    case "good":
      return "none";
    case "cap_open":
    case "opened":
      return "seal_issue";
    case "deformation":
      return "dent";
    case "structural_anomalies":
      return "crack";
    default:
      return "scratch";
  }
}

function toDecisionFromGT(gt: string): "OK" | "NG" | "REVIEW" {
  if (gt === "good") return "OK";
  // opened는 경계 케이스로 REVIEW로 두고 싶다면 여기
  if (gt === "opened") return "REVIEW";
  return "NG";
}

function toSeverity(defectType: string, decision: "OK" | "NG" | "REVIEW"): "low" | "med" | "high" {
  if (decision === "OK") return "low";
  if (decision === "REVIEW") return "low";
  if (defectType === "crack") return "high";
  if (defectType === "dent") return "med";
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
  if (severity === "high") return 15 + v * 12; // 15~27
  if (severity === "med") return 6 + v * 8;    // 6~14
  return 1 + v * 4;                            // 1~5
}

function toAnomalyScore(decision: "OK" | "NG" | "REVIEW", seed: string): number {
  const v = hash01(seed);
  if (decision === "OK") return Math.min(0.55, 0.25 + 0.30 * v);         // 0.25~0.55
  if (decision === "REVIEW") return Math.min(0.74, 0.66 + 0.08 * v);     // 0.66~0.74
  return Math.min(0.99, 0.80 + 0.19 * v);                                 // 0.80~0.99
}

function toDefectConfidence(decision: "OK" | "NG" | "REVIEW", seed: string): number {
  if (decision === "OK") return 0;
  const v = hash01(seed);
  return Math.min(0.99, 0.75 + 0.20 * v); // 0.75~0.95
}

function toBatchId(dataset: string, category: string, ts: Date): string {
  const yyyy = ts.getFullYear();
  const mm = String(ts.getMonth() + 1).padStart(2, "0");
  const dd = String(ts.getDate()).padStart(2, "0");
  return `BATCH-${dataset}-${category}-${yyyy}${mm}${dd}`;
}

function toKoreanSummary(decision: "OK" | "NG" | "REVIEW", defectType: string, location: string): string {
  if (decision === "OK") {
    return "정상 제품으로 판정되었습니다. 이상 징후가 발견되지 않았습니다.";
  }
  if (decision === "REVIEW") {
    return "경계 케이스입니다. 육안 재검토를 통해 판정을 확정해 주세요.";
  }
  // NG
  const defectKo =
    defectType === "seal_issue" ? "실링 불량" :
    defectType === "contamination" ? "오염" :
    defectType === "crack" ? "파손/균열" :
    defectType === "dent" ? "찌그러짐" :
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
  // ✅ 요청대로 이름 사용(나중에 roles/OP-###로 바꿔도 됨)
  const base = ts.getTime();
  if (decision === "OK") {
    return [{ who: "System", when: new Date(base + 1000), what: "자동 승인" }];
  }
  if (decision === "REVIEW") {
    return [{ who: "박철수", when: new Date(base + 60_000), what: "재검 요청" }];
  }
  // NG
  return [
    { who: "김민수", when: new Date(base + 60_000), what: "불량 확정" },
    { who: "이영희", when: new Date(base + 120_000), what: "메모 추가" },
  ];
}

export function mapSampleReportsToAnomalyCases(raw: SampleReport[]): AnomalyCase[] {
  return raw.map((r, idx) => {
    const ts = parseDatetime(r.datetime);
    const dataset = r.dataset ?? "UNKNOWN";
    const category = r.category ?? "unknown";
    const filename = r.filename ?? `unknown_${idx}.jpg`;

    const defect_type = toDefectTypeFromGT(r.ground_truth);
    const decision = toDecisionFromGT(r.ground_truth);
    const line_id = toLineId(`${dataset}-${category}`);
    const shift = toShift(ts);

    const loc = toLocation(decision, `${filename}-${category}`);
    const severity = toSeverity(defect_type, decision);

    const anomaly_score = toAnomalyScore(decision, `${filename}-${dataset}-${category}`);
    const threshold = 0.65;

    const affected_area_pct = toAffectedAreaPct(severity, `${filename}-${category}`, decision);
    const defect_confidence = toDefectConfidence(decision, `${filename}-${category}`);

    const id = `CASE-${dataset}-${category}-${filename}`;

    return {
      id,
      timestamp: ts,
      line_id,
      shift,
      product_group: toProductGroup(category),
      product_class: toProductClass(category),
      batch_id: toBatchId(dataset, category, ts),
      image_id: filename,

      decision,
      anomaly_score,
      threshold,

      defect_type,
      defect_confidence,
      location: decision === "OK" ? "none" : loc,
      affected_area_pct,
      severity,

      model_name: "EfficientAD",
      model_version: "v1.0.0",
      // inference_time이 5~6대라서 일단 seconds로 보고 ms 변환
      inference_time_ms: typeof r.inference_time === "number" ? Math.round(r.inference_time * 1000) : 0,

      llm_summary: toKoreanSummary(decision, defect_type, loc),
      llm_structured_json: {
        source: r, // 원본 전체 보관(필요할 때 열람)
      },

      operator_note: (r.recommendation ?? "").trim(),
      action_log: toActionLog(decision, ts),
    };
  });
}
