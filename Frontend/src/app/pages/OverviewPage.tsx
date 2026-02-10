// src/app/pages/OverviewPage.tsx
import React, { useMemo } from "react";
import { AnomalyCase } from "../data/mockData";
import { Alert } from "../data/AlertData";
import { FilterState } from "../components/FilterBar";
import { Badge } from "../components/Badge";
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
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface OverviewPageProps {
  cases: AnomalyCase[];
  alerts: Alert[];
  filters: FilterState;
  activeModel: string;
  onCaseClick: (id: string) => void;
  onFilterUpdate: (filters: Partial<FilterState>) => void;
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

// 알림 설명을 필드 기반으로 생성 (review 알림은 product_group 우선)
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

export function OverviewPage({
  cases,
  alerts,
  activeModel,
  onCaseClick,
  onFilterUpdate,
}: OverviewPageProps) {
  const total = cases.length;
  const defects = cases.filter((c) => c.decision === "NG").length;
  const reviews = cases.filter((c) => c.decision === "REVIEW").length;

  const defectRate = total > 0 ? ((defects / total) * 100).toFixed(1) : "0.0";
  const reviewRate = total > 0 ? ((reviews / total) * 100).toFixed(1) : "0.0";
  const avgInference =
    total > 0
      ? Math.round(cases.reduce((acc, c) => acc + c.inference_time_ms, 0) / total)
      : 0;

  const trendData = useMemo(() => {
    const hourlyData: { [key: string]: { total: number; ng: number } } = {};
    cases.forEach((c) => {
      const hour = c.timestamp.getHours();
      const key = `${hour}:00`;
      if (!hourlyData[key]) hourlyData[key] = { total: 0, ng: 0 };
      hourlyData[key].total++;
      if (c.decision === "NG") hourlyData[key].ng++;
    });
    return Object.entries(hourlyData)
      .map(([time, data]) => ({
        time,
        불량률: Number(((data.ng / data.total) * 100).toFixed(1)),
      }))
      .sort((a, b) => parseInt(a.time) - parseInt(b.time));
  }, [cases]);

  const productDefects = useMemo(() => {
    const defectsByClass: { [key: string]: number } = {};
    cases
      .filter((c) => c.decision === "NG")
      .forEach((c) => {
        defectsByClass[c.product_group] = (defectsByClass[c.product_group] || 0) + 1;
      });
    return Object.entries(defectsByClass)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [cases]);

  const defectTypeData = useMemo(() => {
    const types: { [key: string]: number } = {};
    cases
      .filter((c) => c.decision === "NG")
      .forEach((c) => {
        types[c.defect_type] = (types[c.defect_type] || 0) + 1;
      });

    return Object.entries(types).map(([type, value]) => ({
      name: defectKo(type),
      value,
      type,
    }));
  }, [cases]);

  const COLORS = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#06b6d4", "#8b5cf6"];

  const recentAnomalies = useMemo(() => {
    return cases
      .filter((c) => c.decision === "NG" || c.decision === "REVIEW")
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 8);
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
        <KPICard
          title="검사 수"
          value={total.toLocaleString()}
          subtext="오늘 누적 검사량"
          icon={Activity}
        />
        <KPICard
          title="불량률"
          value={`${defectRate}%`}
          subtext="오늘 검사 대비 불량"
          icon={AlertTriangle}
        />
        <KPICard
          title="재검률"
          value={`${reviewRate}%`}
          subtext="AI 판정 보류 건"
          icon={CheckCircle}
        />
        <KPICard
          title="평균 추론 시간"
          value={`${avgInference}ms`}
          subtext={activeModel}
          icon={Clock}
        />
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
                      isCritical
                        ? "text-red-600"
                        : isMedium
                          ? "text-orange-600"
                          : "text-blue-600"
                    }`}
                  />
                </div>

                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <h3
                      className={`font-semibold ${
                        isCritical
                          ? "text-red-900"
                          : isMedium
                            ? "text-orange-900"
                            : "text-blue-900"
                      }`}
                    >
                      {alert.title}
                    </h3>
                    <span className="text-xs text-gray-500">
                      {alert.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>

                  <p
                    className={`text-sm mt-1 ${
                      isCritical
                        ? "text-red-700"
                        : isMedium
                          ? "text-orange-700"
                          : "text-blue-700"
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

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">불량률 추이</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#6b7280" />
              <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="불량률"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ fill: "#ef4444", r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">결함 타입 분포</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie
                data={defectTypeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                dataKey="value"
              >
                {defectTypeData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                    className="cursor-pointer hover:opacity-80"
                    onClick={() => onFilterUpdate({ defectType: entry.type })}
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top Products */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">제품별 불량 Top 10</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={productDefects}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#6b7280" />
            <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Anomalies */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-gray-900">최근 이상 케이스</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">케이스 ID</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">시간</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">라인</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">제품</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">결함 타입</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Score</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">판정</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">심각도</th>
              </tr>
            </thead>
            <tbody>
              {recentAnomalies.map((anomaly) => (
                <tr
                  key={anomaly.id}
                  onClick={() => onCaseClick(anomaly.id)}
                  className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors"
                >
                  <td className="py-3 px-4 text-sm font-mono text-blue-600">{anomaly.id}</td>
                  <td className="py-3 px-4 text-sm text-gray-700">
                    {anomaly.timestamp.toLocaleTimeString("ko-KR", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-700">{anomaly.line_id}</td>
                  <td className="py-3 px-4 text-sm text-gray-700">{anomaly.product_group}</td>
                  <td className="py-3 px-4 text-sm text-gray-700">{defectKo(anomaly.defect_type)}</td>
                  <td className="py-3 px-4 text-sm">
                    <span
                      className={`font-medium ${
                        anomaly.anomaly_score >= 0.8
                          ? "text-red-600"
                          : anomaly.anomaly_score >= 0.65
                            ? "text-orange-600"
                            : "text-gray-600"
                      }`}
                    >
                      {anomaly.anomaly_score.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={anomaly.decision} size="sm">
                      {anomaly.decision}
                    </Badge>
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={anomaly.severity} size="sm">
                      {anomaly.severity === "high" ? "높음" : anomaly.severity === "med" ? "중간" : "낮음"}
                    </Badge>
                  </td>
                </tr>
              ))}
              {recentAnomalies.length === 0 && (
                <tr>
                  <td className="py-8 px-4 text-sm text-gray-500 text-center" colSpan={8}>
                    최근 이상 케이스가 없습니다.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}