// src/app/pages/LlavaReportsPage.tsx
import React, { useEffect, useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import {
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Download,
  Search,
  X,
  BarChart3,
} from "lucide-react";
import { useReports, useReportDetail, useReportStats } from "../hooks/useReports";
import { getReportJsonUrl, ReportDTO } from "../api/reportsApi";

const PAGE_SIZE = 20;

function DecisionBadge({ decision }: { decision: string }) {
  const d = (decision ?? "").trim().toLowerCase();

  if (d === "ok" || d === "normal") return <Badge variant="OK" size="sm">정상 (OK)</Badge>;
  if (d === "ng" || d === "anomaly") return <Badge variant="NG" size="sm">불량 (NG)</Badge>;
  if (d === "review") return <Badge variant="REVIEW" size="sm">재검토 (REVIEW)</Badge>;

  return <Badge variant="low" size="sm">{decision || "미분류"}</Badge>;
}

function decisionLabel(key: string) {
  const d = (key ?? "").trim().toLowerCase();
  if (d === "ok" || d === "normal") return "정상 (OK)";
  if (d === "ng" || d === "anomaly") return "불량 (NG)";
  if (d === "review") return "재검토 (REVIEW)";
  return key || "미분류";
}

function prettyCategoryLabel(key: string) {
  // snake_case면 보기 좋게
  return (key ?? "").replace(/_/g, " ");
}

export function ReportsPage() {
  // list state
  const [offset, setOffset] = useState(0);

  // filters
  const [filterDataset, setFilterDataset] = useState("");
  const [filterCategory, setFilterCategory] = useState("");   // ✅ dropdown value
  const [filterDecision, setFilterDecision] = useState("");   // ✅ dropdown value

  // stats (dropdown 옵션에도 사용)
  const [showStats, setShowStats] = useState(false);
  const { stats, loading: statsLoading, error: statsError, load: loadStats } = useReportStats();

  // detail
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [showJson, setShowJson] = useState(false);
  const { data: selectedReport, loading: detailLoading, error: detailError } = useReportDetail(selectedId);

  // sort
  type SortField = "id" | "dataset" | "category" | "decision" | "inference_time";
  const [sortField, setSortField] = useState<SortField>("id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // ✅ 드롭다운 옵션 확보를 위해 stats는 페이지 진입 시 1회 로드
  useEffect(() => {
    if (!stats && !statsLoading) loadStats();
  }, [stats, statsLoading, loadStats]);

  const categoryOptions = useMemo(() => {
    if (!stats?.by_category) return [];
    return Object.keys(stats.by_category).sort((a, b) => a.localeCompare(b));
  }, [stats]);

  const decisionOptions = useMemo(() => {
    if (!stats?.by_decision) return [];
    return Object.keys(stats.by_decision).sort((a, b) => a.localeCompare(b));
  }, [stats]);

  const query = useMemo(() => {
    return {
      limit: PAGE_SIZE,
      offset,
      dataset: filterDataset || undefined,
      category: filterCategory || undefined,
      decision: filterDecision || undefined,
    };
  }, [offset, filterDataset, filterCategory, filterDecision]);

  const { data, loading, error } = useReports(query, { debounceMs: 250 });

  const reports = data?.items ?? [];
  const total = data?.total ?? 0;

  // 통계 패널은 버튼 눌러서 보이게만 유지 (로드는 이미 위에서 해둠)
  useEffect(() => {
    if (showStats && !stats && !statsLoading) loadStats();
  }, [showStats, stats, statsLoading, loadStats]);

  const sorted = useMemo(() => {
    const arr = [...reports];
    arr.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case "id": cmp = a.id - b.id; break;
        case "dataset": cmp = a.dataset.localeCompare(b.dataset); break;
        case "category": cmp = a.category.localeCompare(b.category); break;
        case "decision": cmp = a.decision.localeCompare(b.decision); break;
        case "inference_time": cmp = (a.inference_time ?? 0) - (b.inference_time ?? 0); break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [reports, sortField, sortDir]);

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDir === "asc" ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />;
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  const handleDownloadJson = (reportId: number) => {
    window.open(getReportJsonUrl(reportId), "_blank");
  };

  // ---- detail view ----
  if (selectedId != null) {
    const r: ReportDTO | null = selectedReport;

    return (
      <div className="p-8">
        <div className="flex items-center gap-2 text-sm text-gray-600 mb-6">
          <button onClick={() => { setSelectedId(null); setShowJson(false); }} className="hover:text-gray-900">
            AI 생성 리포트
          </button>
          <ChevronRight className="w-4 h-4" />
          <span className="text-gray-900 font-medium">리포트 #{selectedId}</span>
        </div>

        {detailLoading && (
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-600">
            상세 불러오는 중...
          </div>
        )}

        {detailError && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            상세 API 연결 실패: {detailError.message}
          </div>
        )}

        {r && (
          <>
            <div className="flex items-start justify-between mb-6">
              <div>
                <h1 className="text-2xl font-semibold text-gray-900 mb-2">{r.filename}</h1>
                <p className="text-sm text-gray-600">
                  {r.datetime} · {r.dataset} · {r.category}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <DecisionBadge decision={r.decision} />
                <button
                  onClick={() => handleDownloadJson(r.id)}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                >
                  <Download className="w-4 h-4" />
                  JSON
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-6">
              <div className="col-span-2 space-y-6">
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">AI 분석 요약</h2>
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                    <p className="text-sm text-blue-900">{r.summary || "요약 없음"}</p>
                  </div>

                  {r.defect_description && (
                    <div className="mb-4">
                      <label className="text-sm font-medium text-gray-500 block mb-1">결함 설명</label>
                      <p className="text-sm text-gray-700">{r.defect_description}</p>
                    </div>
                  )}

                  {r.impact && (
                    <div className="mb-4">
                      <label className="text-sm font-medium text-gray-500 block mb-1">영향</label>
                      <p className="text-sm text-gray-700">{r.impact}</p>
                    </div>
                  )}

                  {r.recommendation && (
                    <div>
                      <label className="text-sm font-medium text-gray-500 block mb-1">권고사항</label>
                      <p className="text-sm text-gray-700">{r.recommendation}</p>
                    </div>
                  )}
                </div>

                {r.has_defect === 1 && (
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">결함 정보</h2>
                    <div className="grid grid-cols-3 gap-6">
                      <div>
                        <label className="text-sm font-medium text-gray-500 block mb-1">결함 유형</label>
                        <p className="text-sm text-gray-900">{r.defect_type || "-"}</p>
                      </div>
                      <div>
                        <label className="text-sm font-medium text-gray-500 block mb-1">위치</label>
                        <p className="text-sm text-gray-900">{r.location || "-"}</p>
                      </div>
                      <div>
                        <label className="text-sm font-medium text-gray-500 block mb-1">심각도</label>
                        <p className="text-sm text-gray-900">{r.severity || "-"}</p>
                      </div>
                    </div>
                    {r.possible_cause && (
                      <div className="mt-4">
                        <label className="text-sm font-medium text-gray-500 block mb-1">추정 원인</label>
                        <p className="text-sm text-gray-700">{r.possible_cause}</p>
                      </div>
                    )}
                  </div>
                )}

                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <button
                    onClick={() => setShowJson(!showJson)}
                    className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
                  >
                    {showJson ? "JSON 숨기기" : "JSON 전체 보기"}
                  </button>
                  {showJson && (
                    <pre className="mt-3 p-4 bg-gray-900 text-green-400 rounded-lg text-xs overflow-x-auto">
                      {JSON.stringify(r, null, 2)}
                    </pre>
                  )}
                </div>
              </div>

              <div className="space-y-6">
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">메타 정보</h2>
                  <div className="space-y-3 text-sm">
                    <div>
                      <span className="text-gray-500 block mb-1">파일명</span>
                      <span className="font-mono text-gray-900">{r.filename}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">데이터셋</span>
                      <span className="font-medium text-gray-900">{r.dataset}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">카테고리</span>
                      <span className="font-medium text-gray-900">{r.category}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">Ground Truth</span>
                      <span className="font-medium text-gray-900">{r.ground_truth || "-"}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">판정</span>
                      <DecisionBadge decision={r.decision} />
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">신뢰도</span>
                      <span className="font-medium text-gray-900">
                        {r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : "-"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">제품 정보</h2>
                  <p className="text-sm text-gray-700">{r.product_description || "-"}</p>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">추론 정보</h2>
                  <div className="space-y-3 text-sm">
                    <div>
                      <span className="text-gray-500 block mb-1">추론 시간</span>
                      <span className="font-medium text-gray-900">
                        {r.inference_time != null ? `${r.inference_time.toFixed(2)}s` : "-"}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">일시</span>
                      <span className="font-mono text-gray-900">{r.datetime}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    );
  }

  // ---- list view ----
  return (
    <div className="p-8">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">AI 생성 리포트</h1>
          <p className="text-sm text-gray-600">
            총 <span className="font-medium text-gray-900">{total}</span>건
          </p>
        </div>
        <button
          onClick={() => setShowStats((s) => !s)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
            showStats ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          통계
        </button>
      </div>

      {showStats && (
        <>
          {statsError && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              통계 API 연결 실패: {statsError.message}
            </div>
          )}
          {statsLoading && (
            <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-600">
              통계 불러오는 중...
            </div>
          )}
          {stats && (
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-500 mb-3">데이터셋별</h3>
                <div className="space-y-2">
                  {Object.entries(stats.by_dataset).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-gray-700">{k}</span>
                      <span className="font-medium text-gray-900">{v}건</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-500 mb-3">카테고리별</h3>
                <div className="space-y-2">
                  {Object.entries(stats.by_category).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-gray-700">{k}</span>
                      <span className="font-medium text-gray-900">{v}건</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-500 mb-3">판정별</h3>
                <div className="space-y-2">
                  {Object.entries(stats.by_decision).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-gray-700">{k}</span>
                      <span className="font-medium text-gray-900">{v}건</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* ✅ Filters: category/decision을 드롭다운으로 */}
      <div className="flex items-center gap-3 mb-6">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="데이터셋 필터"
            value={filterDataset}
            onChange={(e) => { setFilterDataset(e.target.value); setOffset(0); }}
            className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <select
          value={filterCategory}
          onChange={(e) => { setFilterCategory(e.target.value); setOffset(0); }}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 w-52"
          disabled={!stats && statsLoading}
        >
          <option value="">{statsLoading && !stats ? "카테고리 로딩..." : "전체 카테고리"}</option>
          {categoryOptions.map((cat) => (
            <option key={cat} value={cat}>
              {prettyCategoryLabel(cat)}{stats?.by_category?.[cat] != null ? ` (${stats.by_category[cat]}건)` : ""}
            </option>
          ))}
        </select>

        <select
          value={filterDecision}
          onChange={(e) => { setFilterDecision(e.target.value); setOffset(0); }}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 w-44"
          disabled={!stats && statsLoading}
        >
          <option value="">{statsLoading && !stats ? "판정 로딩..." : "전체 판정"}</option>
          {decisionOptions.map((d) => (
            <option key={d} value={d}>
              {decisionLabel(d)}{stats?.by_decision?.[d] != null ? ` (${stats.by_decision[d]}건)` : ""}
            </option>
          ))}
        </select>

        {(filterDataset || filterCategory || filterDecision) && (
          <button
            onClick={() => {
              setFilterDataset("");
              setFilterCategory("");
              setFilterDecision("");
              setOffset(0);
            }}
            className="p-2 hover:bg-gray-100 rounded"
            title="필터 초기화"
          >
            <X className="w-4 h-4 text-gray-500" />
          </button>
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          API 연결 실패: {error.message}. 서버가 실행 중인지 확인하세요.
        </div>
      )}

      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden mb-4">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th
                  onClick={() => handleSort("id")}
                  className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-1">ID <SortIcon field="id" /></div>
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">파일명</th>
                <th
                  onClick={() => handleSort("dataset")}
                  className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-1">데이터셋 <SortIcon field="dataset" /></div>
                </th>
                <th
                  onClick={() => handleSort("category")}
                  className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-1">카테고리 <SortIcon field="category" /></div>
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Ground Truth</th>
                <th
                  onClick={() => handleSort("decision")}
                  className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-1">판정 <SortIcon field="decision" /></div>
                </th>
                <th
                  onClick={() => handleSort("inference_time")}
                  className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-1">추론시간 <SortIcon field="inference_time" /></div>
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">일시</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600"></th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={9} className="py-12 text-center text-gray-500">불러오는 중...</td></tr>
              ) : sorted.length === 0 ? (
                <tr><td colSpan={9} className="py-12 text-center text-gray-500">데이터가 없습니다</td></tr>
              ) : (
                sorted.map((r) => (
                  <tr
                    key={r.id}
                    onClick={() => { setSelectedId(r.id); setShowJson(false); }}
                    className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors"
                  >
                    <td className="py-3 px-4 text-sm font-mono text-blue-600">{r.id}</td>
                    <td className="py-3 px-4 text-sm text-gray-700 max-w-[200px] truncate">{r.filename}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{r.dataset}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{r.category}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{r.ground_truth || "-"}</td>
                    <td className="py-3 px-4"><DecisionBadge decision={r.decision} /></td>
                    <td className="py-3 px-4 text-sm text-gray-700">
                      {r.inference_time != null ? `${r.inference_time.toFixed(2)}s` : "-"}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-500 whitespace-nowrap">
                      {r.datetime ? new Date(r.datetime).toLocaleString("ko-KR", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "-"}
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDownloadJson(r.id); }}
                        className="text-gray-400 hover:text-gray-600"
                        title="JSON 다운로드"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-600">
            {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} / {total}건
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
              disabled={offset === 0}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg disabled:opacity-50 hover:bg-gray-50"
            >
              이전
            </button>
            <span className="text-sm text-gray-600">{currentPage} / {totalPages}</span>
            <button
              onClick={() => setOffset((o) => o + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= total}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg disabled:opacity-50 hover:bg-gray-50"
            >
              다음
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
