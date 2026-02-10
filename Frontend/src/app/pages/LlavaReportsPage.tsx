// src/pages/LlavaReportsPage.tsx
import { Badge } from '../components/Badge';
import React, { useState, useEffect, useCallback } from 'react';
import { ChevronDown, ChevronUp, ChevronRight, Download, Search, X, BarChart3 } from 'lucide-react';

// --------------- types ---------------
interface LlavaReport {
  id: number;
  filename: string;
  image_path: string;
  dataset: string;
  category: string;
  ground_truth: string | null;
  decision: string;
  confidence: number | null;
  has_defect: number;
  defect_type: string;
  location: string;
  severity: string;
  defect_description: string;
  possible_cause: string;
  product_description: string;
  summary: string;
  impact: string;
  recommendation: string;
  inference_time: number | null;
  datetime: string;
}

interface LlavaStats {
  total: number;
  by_dataset: Record<string, number>;
  by_category: Record<string, number>;
  by_decision: Record<string, number>;
}

interface LlavaReportsPageProps {
  apiBase?: string;
}

// --------------- constants ---------------
const PAGE_SIZE = 20;

// --------------- helpers ---------------
function DecisionBadge({ decision }: { decision: string }) {
  const d = (decision ?? '').trim().toLowerCase();

  if (d === 'normal') return <Badge variant="OK" size="sm">정상 (OK)</Badge>;
  if (d === 'anomaly') return <Badge variant="NG" size="sm">불량 (NG)</Badge>;
  if (d === 'review') return <Badge variant="REVIEW" size="sm">재검토 (REVIEW)</Badge>;

  return <Badge variant="low" size="sm">{decision || '미분류'}</Badge>;
}

// --------------- component ---------------
export function LlavaReportsPage({ apiBase = 'http://localhost:8000' }: LlavaReportsPageProps) {
  // list state
  const [reports, setReports] = useState<LlavaReport[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // filters
  const [filterDataset, setFilterDataset] = useState('');
  const [filterCategory, setFilterCategory] = useState('');
  const [filterDecision, setFilterDecision] = useState('');

  // stats
  const [stats, setStats] = useState<LlavaStats | null>(null);
  const [showStats, setShowStats] = useState(false);

  // detail
  const [selectedReport, setSelectedReport] = useState<LlavaReport | null>(null);
  const [showJson, setShowJson] = useState(false);

  // sort
  type SortField = 'id' | 'dataset' | 'category' | 'decision' | 'inference_time';
  const [sortField, setSortField] = useState<SortField>('id');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  // --------------- fetch ---------------
  const fetchReports = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.set('limit', String(PAGE_SIZE));
      params.set('offset', String(offset));
      if (filterDataset) params.set('dataset', filterDataset);
      if (filterCategory) params.set('category', filterCategory);
      if (filterDecision) params.set('decision', filterDecision);

      const res = await fetch(`${apiBase}/llava/reports?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setReports(data.items);
      setTotal(data.total);
    } catch (e: any) {
      setError(e.message || 'Failed to fetch');
    } finally {
      setLoading(false);
    }
  }, [apiBase, offset, filterDataset, filterCategory, filterDecision]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/llava/stats`);
      if (!res.ok) return;
      setStats(await res.json());
    } catch {}
  }, [apiBase]);

  useEffect(() => { fetchReports(); }, [fetchReports]);
  useEffect(() => { fetchStats(); }, [fetchStats]);

  // --------------- sort ---------------
  const sorted = [...reports].sort((a, b) => {
    let cmp = 0;
    switch (sortField) {
      case 'id': cmp = a.id - b.id; break;
      case 'dataset': cmp = a.dataset.localeCompare(b.dataset); break;
      case 'category': cmp = a.category.localeCompare(b.category); break;
      case 'decision': cmp = a.decision.localeCompare(b.decision); break;
      case 'inference_time': cmp = (a.inference_time ?? 0) - (b.inference_time ?? 0); break;
    }
    return sortDir === 'asc' ? cmp : -cmp;
  });

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDir === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />;
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  const handleDownloadJson = (reportId: number) => {
    window.open(`${apiBase}/llava/reports/${reportId}/json`, '_blank');
  };

  // --------------- detail view ---------------
  if (selectedReport) {
    return (
      <div className="p-8">
        {/* Breadcrumbs */}
        <div className="flex items-center gap-2 text-sm text-gray-600 mb-6">
          <button onClick={() => setSelectedReport(null)} className="hover:text-gray-900">LLaVA 리포트</button>
          <ChevronRight className="w-4 h-4" />
          <span className="text-gray-900 font-medium">리포트 #{selectedReport.id}</span>
        </div>

        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">
              {selectedReport.filename}
            </h1>
            <p className="text-sm text-gray-600">
              {selectedReport.datetime} · {selectedReport.dataset} · {selectedReport.category}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <DecisionBadge decision={selectedReport.decision} />
            <button
              onClick={() => handleDownloadJson(selectedReport.id)}
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
            >
              <Download className="w-4 h-4" />
              JSON
            </button>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* Left - main info */}
          <div className="col-span-2 space-y-6">
            {/* Summary */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">AI 분석 요약</h2>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                <p className="text-sm text-blue-900">{selectedReport.summary || '요약 없음'}</p>
              </div>

              {selectedReport.defect_description && (
                <div className="mb-4">
                  <label className="text-sm font-medium text-gray-500 block mb-1">결함 설명</label>
                  <p className="text-sm text-gray-700">{selectedReport.defect_description}</p>
                </div>
              )}

              {selectedReport.impact && (
                <div className="mb-4">
                  <label className="text-sm font-medium text-gray-500 block mb-1">영향</label>
                  <p className="text-sm text-gray-700">{selectedReport.impact}</p>
                </div>
              )}

              {selectedReport.recommendation && (
                <div>
                  <label className="text-sm font-medium text-gray-500 block mb-1">권고사항</label>
                  <p className="text-sm text-gray-700">{selectedReport.recommendation}</p>
                </div>
              )}
            </div>

            {/* Defect info */}
            {selectedReport.has_defect === 1 && (
              <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">결함 정보</h2>
                <div className="grid grid-cols-3 gap-6">
                  <div>
                    <label className="text-sm font-medium text-gray-500 block mb-1">결함 유형</label>
                    <p className="text-sm text-gray-900">{selectedReport.defect_type || '-'}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 block mb-1">위치</label>
                    <p className="text-sm text-gray-900">{selectedReport.location || '-'}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 block mb-1">심각도</label>
                    <p className="text-sm text-gray-900">{selectedReport.severity || '-'}</p>
                  </div>
                </div>
                {selectedReport.possible_cause && (
                  <div className="mt-4">
                    <label className="text-sm font-medium text-gray-500 block mb-1">추정 원인</label>
                    <p className="text-sm text-gray-700">{selectedReport.possible_cause}</p>
                  </div>
                )}
              </div>
            )}

            {/* Raw JSON */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <button
                onClick={() => setShowJson(!showJson)}
                className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                {showJson ? 'JSON 숨기기' : 'JSON 전체 보기'}
              </button>
              {showJson && (
                <pre className="mt-3 p-4 bg-gray-900 text-green-400 rounded-lg text-xs overflow-x-auto">
                  {JSON.stringify(selectedReport, null, 2)}
                </pre>
              )}
            </div>
          </div>

          {/* Right - metadata */}
          <div className="space-y-6">
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">메타 정보</h2>
              <div className="space-y-3 text-sm">
                <div>
                  <span className="text-gray-500 block mb-1">파일명</span>
                  <span className="font-mono text-gray-900">{selectedReport.filename}</span>
                </div>
                <div>
                  <span className="text-gray-500 block mb-1">데이터셋</span>
                  <span className="font-medium text-gray-900">{selectedReport.dataset}</span>
                </div>
                <div>
                  <span className="text-gray-500 block mb-1">카테고리</span>
                  <span className="font-medium text-gray-900">{selectedReport.category}</span>
                </div>
                <div>
                  <span className="text-gray-500 block mb-1">Ground Truth</span>
                  <span className="font-medium text-gray-900">{selectedReport.ground_truth || '-'}</span>
                </div>
                <div>
                  <span className="text-gray-500 block mb-1">판정</span>
                  <DecisionBadge decision={selectedReport.decision} />
                </div>
                <div>
                  <span className="text-gray-500 block mb-1">신뢰도</span>
                  <span className="font-medium text-gray-900">
                    {selectedReport.confidence != null ? `${(selectedReport.confidence * 100).toFixed(1)}%` : '-'}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">제품 정보</h2>
              <p className="text-sm text-gray-700">
                {selectedReport.product_description || '-'}
              </p>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">추론 정보</h2>
              <div className="space-y-3 text-sm">
                <div>
                  <span className="text-gray-500 block mb-1">추론 시간</span>
                  <span className="font-medium text-gray-900">
                    {selectedReport.inference_time != null ? `${selectedReport.inference_time.toFixed(2)}s` : '-'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 block mb-1">일시</span>
                  <span className="font-mono text-gray-900">{selectedReport.datetime}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // --------------- list view ---------------
  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">LLaVA 리포트</h1>
          <p className="text-sm text-gray-600">
            총 <span className="font-medium text-gray-900">{total}</span>건
          </p>
        </div>
        <button
          onClick={() => setShowStats(s => !s)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
            showStats ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          통계
        </button>
      </div>

      {/* Stats Panel */}
      {showStats && stats && (
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

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="데이터셋 필터"
            value={filterDataset}
            onChange={e => { setFilterDataset(e.target.value); setOffset(0); }}
            className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <input
          type="text"
          placeholder="카테고리"
          value={filterCategory}
          onChange={e => { setFilterCategory(e.target.value); setOffset(0); }}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 w-40"
        />
        <input
          type="text"
          placeholder="판정"
          value={filterDecision}
          onChange={e => { setFilterDecision(e.target.value); setOffset(0); }}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 w-32"
        />
        {(filterDataset || filterCategory || filterDecision) && (
          <button
            onClick={() => { setFilterDataset(''); setFilterCategory(''); setFilterDecision(''); setOffset(0); }}
            className="p-2 hover:bg-gray-100 rounded"
          >
            <X className="w-4 h-4 text-gray-500" />
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          API 연결 실패: {error}. 서버가 실행 중인지 확인하세요.
        </div>
      )}

      {/* Table */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden mb-4">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th onClick={() => handleSort('id')} className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100">
                  <div className="flex items-center gap-1">ID <SortIcon field="id" /></div>
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">파일명</th>
                <th onClick={() => handleSort('dataset')} className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100">
                  <div className="flex items-center gap-1">데이터셋 <SortIcon field="dataset" /></div>
                </th>
                <th onClick={() => handleSort('category')} className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100">
                  <div className="flex items-center gap-1">카테고리 <SortIcon field="category" /></div>
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Ground Truth</th>
                <th onClick={() => handleSort('decision')} className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100">
                  <div className="flex items-center gap-1">판정 <SortIcon field="decision" /></div>
                </th>
                <th onClick={() => handleSort('inference_time')} className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100">
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
                sorted.map(r => (
                  <tr
                    key={r.id}
                    onClick={() => setSelectedReport(r)}
                    className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors"
                  >
                    <td className="py-3 px-4 text-sm font-mono text-blue-600">{r.id}</td>
                    <td className="py-3 px-4 text-sm text-gray-700 max-w-[200px] truncate">{r.filename}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{r.dataset}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{r.category}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{r.ground_truth || '-'}</td>
                    <td className="py-3 px-4"><DecisionBadge decision={r.decision} /></td>
                    <td className="py-3 px-4 text-sm text-gray-700">
                      {r.inference_time != null ? `${r.inference_time.toFixed(2)}s` : '-'}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-500 whitespace-nowrap">
                      {r.datetime ? new Date(r.datetime).toLocaleString('ko-KR', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }) : '-'}
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={e => { e.stopPropagation(); handleDownloadJson(r.id); }}
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

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-600">
            {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} / {total}건
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOffset(o => Math.max(0, o - PAGE_SIZE))}
              disabled={offset === 0}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg disabled:opacity-50 hover:bg-gray-50"
            >
              이전
            </button>
            <span className="text-sm text-gray-600">{currentPage} / {totalPages}</span>
            <button
              onClick={() => setOffset(o => o + PAGE_SIZE)}
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
