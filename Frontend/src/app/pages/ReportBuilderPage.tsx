// src/app/pages/ReportBuilderPage.tsx
import React, { useMemo } from "react";
import { Badge } from "../components/Badge";
import { Download } from "lucide-react";
import { AnomalyCase } from "../data/mockData";
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

interface ReportBuilderPageProps {
  cases: AnomalyCase[]; 
}

export function ReportBuilderPage({ cases }: ReportBuilderPageProps) {
  const reportMetrics = useMemo(() => {
    const total = cases.length;
    const ngCount = cases.filter((c) => c.decision === "NG").length;
    const reviewCount = cases.filter((c) => c.decision === "REVIEW").length;
    const okCount = cases.filter((c) => c.decision === "OK").length;

    const ngRate = total > 0 ? ((ngCount / total) * 100).toFixed(2) : "0.00";
    const reviewRate = total > 0 ? ((reviewCount / total) * 100).toFixed(2) : "0.00";
    const passRate = total > 0 ? ((okCount / total) * 100).toFixed(2) : "0.00";

    const avgScore =
      total > 0
        ? (cases.reduce((sum, c) => sum + c.anomaly_score, 0) / total).toFixed(3)
        : "0.000";

    const avgInferenceTime =
      total > 0
        ? (cases.reduce((sum, c) => sum + c.inference_time_ms, 0) / total).toFixed(1)
        : "0.0";

    const defectBreakdown: { [key: string]: number } = {};
    cases
      .filter((c) => c.decision === "NG")
      .forEach((c) => {
        defectBreakdown[c.defect_type] = (defectBreakdown[c.defect_type] || 0) + 1;
      });

    const representativeCases = cases
      .filter((c) => c.decision === "NG")
      .sort((a, b) => b.anomaly_score - a.anomaly_score)
      .slice(0, 5);

    return {
      total,
      ngCount,
      reviewCount,
      okCount,
      ngRate,
      reviewRate,
      passRate,
      avgScore,
      avgInferenceTime,
      defectBreakdown,
      representativeCases,
    };
  }, [cases]);

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

  const defectChartData = useMemo(() => {
    const typeLabels: { [key: string]: string } = {
      seal_issue: "실링 불량",
      contamination: "오염",
      crack: "파손/균열",
      missing_component: "구성요소 누락",
      scratch: "스크래치",
    };

    return Object.entries(reportMetrics.defectBreakdown).map(([type, count]) => ({
      type: typeLabels[type] || type,
      count,
    }));
  }, [reportMetrics.defectBreakdown]);

  const handleExport = (format: string) => {
    alert(`이상 큐 페이지의 필터 조건 기준으로 리포트를 ${format} 형식으로 내보냅니다.`);
  };

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">운영 리포트</h1>
      </div>

      {/* Executive Summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">요약 (Executive Summary)</h2>
        <div className="grid grid-cols-5 gap-6 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-1">총 검사 수</p>
            <p className="text-3xl font-semibold text-gray-900">{reportMetrics.total}</p>
          </div>

          <div className="bg-red-50 rounded-lg p-4">
            <p className="text-sm text-red-700 mb-1">불량 (NG)</p>
            <p className="text-3xl font-semibold text-red-700">{reportMetrics.ngCount}</p>
            <p className="text-xs text-red-600 mt-1">{reportMetrics.ngRate}%</p>
          </div>

          <div className="bg-amber-50 rounded-lg p-4">
            <p className="text-sm text-amber-700 mb-1">재검토 (REVIEW)</p>
            <p className="text-3xl font-semibold text-amber-700">{reportMetrics.reviewCount}</p>
            <p className="text-xs text-amber-600 mt-1">{reportMetrics.reviewRate}%</p>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <p className="text-sm text-green-700 mb-1">정상 (OK)</p>
            <p className="text-3xl font-semibold text-green-700">{reportMetrics.okCount}</p>
            <p className="text-xs text-green-600 mt-1">{reportMetrics.passRate}%</p>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-sm text-blue-700 mb-1">평균 Score</p>
            <p className="text-3xl font-semibold text-blue-700">{reportMetrics.avgScore}</p>
            <p className="text-xs text-blue-600 mt-1">{reportMetrics.avgInferenceTime}ms</p>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">시간대별 검사 결과</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={hourlyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="hour" tick={{ fontSize: 12 }} stroke="#6b7280" />
              <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
              <Tooltip />
              <Legend />
              <Bar dataKey="불량" fill="#ef4444" stackId="a" />
              <Bar dataKey="재검토" fill="#f59e0b" stackId="a" />
              <Bar dataKey="정상" fill="#10b981" stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">결함 타입별 분포</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={defectChartData} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" tick={{ fontSize: 12 }} stroke="#6b7280" />
              <YAxis dataKey="type" type="category" tick={{ fontSize: 12 }} stroke="#6b7280" width={100} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Representative Cases */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">대표 사례 (Top 5 심각 케이스)</h2>
        {reportMetrics.representativeCases.length > 0 ? (
          <div className="space-y-4">
            {reportMetrics.representativeCases.map((caseData, idx) => (
              <div key={caseData.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                        {idx + 1}
                      </span>
                      <span className="font-mono text-sm text-blue-600">{caseData.id}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {caseData.timestamp.toLocaleString("ko-KR")} · {caseData.line_id}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={caseData.decision} size="sm">{caseData.decision}</Badge>
                    <Badge variant={caseData.severity} size="sm">
                      {caseData.severity === "high" ? "높음" : caseData.severity === "med" ? "중간" : "낮음"}
                    </Badge>
                  </div>
                </div>

                <div className="bg-gray-50 rounded p-3">
                  <p className="text-sm text-gray-700">{caseData.llm_summary}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500 text-center py-8">선택한 조건에서 불량 케이스가 없습니다.</p>
        )}
      </div>

      {/* Export Actions */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">리포트 내보내기</h2>
        <div className="flex items-center gap-3">
          <button
            onClick={() => handleExport("PDF")}
            className="flex items-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <Download className="w-5 h-5" />
            <span>PDF 다운로드</span>
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-4">리포트에는 위의 모든 차트, 통계, 대표 사례가 포함됩니다.</p>
      </div>
    </div>
  );
}
