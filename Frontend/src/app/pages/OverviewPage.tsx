// src/app/pages/OverviewPage.tsx
import React, { useMemo } from "react";
import { AnomalyCase } from "../data/mockData";
import { Alert } from "../data/AlertData";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  AlertOctagon,
  FileText,
  Search,
  TrendingUp,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface OverviewPageProps {
  cases: AnomalyCase[];
  alerts: Alert[];
  activeModel: string;
}

function KPICard({
  title,
  value,
  subtext,
  icon: Icon,
}: {
  title: string;
  value: string;
  subtext: string;
  icon: React.ElementType;
}) {
  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm flex items-start justify-between">
      <div>
        <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
        <h3 className="text-3xl font-bold text-gray-900 mb-2">{value}</h3>
        <div className="flex items-center text-sm text-gray-400">{subtext}</div>
      </div>
      <div className="p-3 bg-gray-50 rounded-lg">
        <Icon className="w-6 h-6 text-gray-600" />
      </div>
    </div>
  );
}

function defectKo(defectType?: string) {
  if (!defectType) return "이상";
  return defectType === "seal_issue"
    ? "실링 불량"
    : defectType === "contamination"
      ? "오염"
      : defectType === "crack"
        ? "파손/균열"
        : defectType === "missing_component"
          ? "구성요소 누락"
          : defectType === "scratch"
            ? "스크래치"
            : defectType;
}

// ✅ 알림 설명을 "필드 기반"으로 생성
function alertDescription(a: Alert) {
  if (a.type === "review") {
    const pg = a.product_group ?? "제품";
    const s = typeof a.score === "number" ? ` (Score: ${a.score.toFixed(2)})` : "";
    return `${pg} 제품에 ${defectKo(a.defect_type)}가 의심됩니다.${s} 육안 검사가 필요합니다.`;
  }

  if (a.type === "critical") {
    const line = a.line_id ?? "라인";
    const conf = typeof a.confidence === "number" ? `${Math.round(a.confidence * 100)}%` : "";
    return `${line}에서 신뢰도 ${conf}의 ${defectKo(a.defect_type)}이 감지되었습니다. 즉시 확인 바랍니다.`;
  }

  if (a.type === "consecutive") {
    const line = a.line_id ?? "라인";
    const n = a.count ?? 3;
    const loc = a.location ?? "동일 위치";
    return `${line}에서 동일한 위치(${loc})의 결함이 ${n}건 연속 발생했습니다. 공정 점검을 권장합니다.`;
  }

  return a.description ?? "";
}

export function OverviewPage({ cases, alerts, activeModel }: OverviewPageProps) {
  const total = cases.length;
  const defects = cases.filter((c) => c.decision === "NG").length;
  const reviews = cases.filter((c) => c.decision === "REVIEW").length;

  const defectRate = total > 0 ? ((defects / total) * 100).toFixed(1) : "0.0";
  const reviewRate = total > 0 ? ((reviews / total) * 100).toFixed(1) : "0.0";
  const avgInference =
    total > 0
      ? Math.round(cases.reduce((acc, c) => acc + c.inference_time_ms, 0) / total)
      : 0;

  // ✅ (운영 리포트에 있던) 시간대별 검사 결과 차트를 개요로 이동
  const hourlyTrend = useMemo(() => {
    const hourlyData: {
      [key: number]: { total: number; ng: number; review: number; ok: number };
    } = {};

    cases.forEach((c) => {
      const hour = c.timestamp.getHours();
      if (!hourlyData[hour]) hourlyData[hour] = { total: 0, ng: 0, review: 0, ok: 0 };
      hourlyData[hour].total++;
      if (c.decision === "NG") hourlyData[hour].ng++;
      else if (c.decision === "REVIEW") hourlyData[hour].review++;
      else hourlyData[hour].ok++;
    });

    return Object.entries(hourlyData)
      .map(([hour, data]) => ({
        hour: `${hour}시`,
        불량: data.ng,
        재검토: data.review,
        정상: data.ok,
      }))
      .sort((a, b) => parseInt(a.hour) - parseInt(b.hour));
  }, [cases]);

  const getAlertIcon = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
        return AlertOctagon;
      case "review":
        return Search;
      case "report":
        return FileText;
      case "consecutive":
        return TrendingUp;
      default:
        return Activity;
    }
  };

  return (
    <div className="p-6 space-y-6 bg-gray-50 min-h-full">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">개요</h1>
        <p className="text-gray-500 mt-1">실시간 품질 검사 현황 및 이상 탐지 결과</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard title="검사 수" value={total.toLocaleString()} subtext="오늘 누적 검사량" icon={Activity} />
        <KPICard title="불량률" value={`${defectRate}%`} subtext="오늘 검사 대비 불량" icon={AlertTriangle} />
        <KPICard title="재검률" value={`${reviewRate}%`} subtext="AI 판정 보류 건" icon={CheckCircle} />
        <KPICard title="평균 추론 시간" value={`${avgInference}ms`} subtext={activeModel} icon={Clock} />
      </div>

      {/* Alerts */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-gray-700" />
            <h2 className="text-lg font-bold text-gray-900">실시간 탐지 알림</h2>
          </div>
        </div>

        <div className="space-y-4">
          {alerts.map((alert) => {
            const Icon = getAlertIcon(alert.type);
            const isCritical = alert.severity === "high";
            const isMedium = alert.severity === "med";

            return (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border flex gap-4 items-start ${
                  isCritical
                    ? "bg-red-50 border-red-100"
                    : isMedium
                      ? "bg-orange-50 border-orange-100"
                      : "bg-blue-50 border-blue-100"
                }`}
              >
                <div className="mt-1">
                  <Icon
                    className={`w-5 h-5 ${
                      isCritical ? "text-red-600" : isMedium ? "text-orange-600" : "text-blue-600"
                    }`}
                  />
                </div>

                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <h3
                      className={`font-semibold ${
                        isCritical ? "text-red-900" : isMedium ? "text-orange-900" : "text-blue-900"
                      }`}
                    >
                      {alert.title}
                    </h3>
                    <span className="text-xs text-gray-500">
                      {alert.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  </div>

                  <p
                    className={`text-sm mt-1 ${
                      isCritical ? "text-red-700" : isMedium ? "text-orange-700" : "text-blue-700"
                    }`}
                  >
                    {alertDescription(alert)}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ✅ 운영 리포트의 "시간대별 검사 결과"를 개요로 이동 (최근 이상 케이스 자리 대체) */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">시간대별 검사 결과</h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={hourlyTrend}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="hour" tick={{ fontSize: 12 }} stroke="#6b7280" interval={0} />
            <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
            <Tooltip />
            <Legend />
            <Bar dataKey="불량" fill="#ef4444" stackId="a" />
            <Bar dataKey="재검토" fill="#f59e0b" stackId="a" />
            <Bar dataKey="정상" fill="#10b981" stackId="a" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
