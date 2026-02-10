// src/app/utils/labels.ts
export function decisionLabel(decision: string) {
  switch (decision) {
    case "OK":
      return "OK";
    case "NG":
      return "NG";
    case "REVIEW":
      return "재검토";
    default:
      return decision; // fallback
  }
}
