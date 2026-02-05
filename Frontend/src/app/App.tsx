// src/app/App.tsx
import React, { useState, useMemo, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { FilterBar, FilterState } from './components/FilterBar';
import { OverviewPage } from './pages/OverviewPage';
import { AnomalyQueuePage } from './pages/AnomalyQueuePage';
import { CaseDetailPage } from './pages/CaseDetailPage';
import { ReportBuilderPage } from './pages/ReportBuilderPage';
import { SettingsPage } from './pages/SettingsPage';

import { mockCases, mockAlerts, AnomalyCase } from './data/mockData';
import { mapSampleReportsToAnomalyCases, SampleReport } from './data/sampleMapper';

// ✅ dateRange 계산 유틸 (App.tsx 상단에 둡니다)
function startOfDay(d: Date) {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x;
}

function endOfDay(d: Date) {
  const x = new Date(d);
  x.setHours(23, 59, 59, 999);
  return x;
}

function getDateRangeWindow(range: string, now = new Date()): { from: Date; to: Date } | null {
  const todayStart = startOfDay(now);
  const todayEnd = endOfDay(now);

  if (range === 'today') return { from: todayStart, to: todayEnd };

  if (range === 'yesterday') {
    const y = new Date(todayStart);
    y.setDate(y.getDate() - 1);
    return { from: startOfDay(y), to: endOfDay(y) };
  }

  if (range === 'week') {
    // 최근 7일: 오늘 포함 7일
    const from = new Date(todayStart);
    from.setDate(from.getDate() - 6);
    return { from, to: todayEnd };
  }

  if (range === 'month') {
    // 최근 30일: 오늘 포함 30일
    const from = new Date(todayStart);
    from.setDate(from.getDate() - 29);
    return { from, to: todayEnd };
  }

  return null;
}

// 샘플 JSON 사용 여부 토글
const USE_SAMPLE_JSON = true;

export default function App() {
  const [currentPage, setCurrentPage] = useState<string>('overview');
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);

  // ✅ mockCases 대신, cases state를 기준으로 앱 전체가 동작하게 변경
  const [cases, setCases] = useState<AnomalyCase[]>(mockCases);

  const [filters, setFilters] = useState<FilterState>({
    dateRange: 'today',
    line: 'all',
    productGroup: 'all',
    defectType: 'all',
    decision: 'all',
    scoreRange: [0, 1]
  });

  // ✅ 앱 시작 시 sampleReports.json' 로드 → AnomalyCase[]로 매핑 → setCases
  useEffect(() => {
    if (!USE_SAMPLE_JSON) return;

    (async () => {
      const url = `${import.meta.env.BASE_URL}mock/sampleReports.json`;
      const res = await fetch(url);

      if (!res.ok) throw new Error(`Failed to fetch sample json: ${res.status}`);
      const raw = (await res.json()) as SampleReport[];
      const mapped = mapSampleReportsToAnomalyCases(raw);
      setCases(mapped);
    })().catch((err) => {
      console.warn('[App] sample json load failed. fallback to mockCases.', err);
      setCases(mockCases);
    });
  }, []);

  // Filter cases based on current filters
 const filteredCases = useMemo(() => {
  const window = getDateRangeWindow(filters.dateRange);

  return cases.filter(c => {
    // ✅ dateRange 필터
    if (window) {
      const t = c.timestamp.getTime();
      if (t < window.from.getTime() || t > window.to.getTime()) return false;
    }

    if (filters.line !== 'all' && c.line_id !== filters.line) return false;
    if (filters.productGroup !== 'all' && c.product_group !== filters.productGroup) return false;
    if (filters.defectType !== 'all' && c.defect_type !== filters.defectType) return false;
    if (filters.decision !== 'all' && c.decision !== filters.decision) return false;
    if (c.anomaly_score < filters.scoreRange[0] || c.anomaly_score > filters.scoreRange[1]) return false;

    return true;
  });
}, [cases, filters]);

  // Handle navigation
  const handleNavigate = (page: string) => {
    setCurrentPage(page);
    setSelectedCaseId(null);
  };

  // Handle case click - navigate to detail
  const handleCaseClick = (caseId: string) => {
    setSelectedCaseId(caseId);
    setCurrentPage('detail');
  };

  // Handle filter updates from charts or other components
  const handleFilterUpdate = (newFilters: Partial<FilterState>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setCurrentPage('queue');
  };

  // Get current case data (✅ cases에서 찾도록 변경)
  const currentCase = selectedCaseId
    ? cases.find(c => c.id === selectedCaseId)
    : null;

  // Render current page
  const renderPage = () => {
    if (currentPage === 'detail' && currentCase) {
      return (
        <CaseDetailPage
          caseData={currentCase}
          onBack={() => setCurrentPage('queue')}
        />
      );
    }

    switch (currentPage) {
      case 'overview':
        return (
          <OverviewPage
            cases={filteredCases}
            alerts={mockAlerts}
            filters={filters}
            onCaseClick={handleCaseClick}
            onFilterUpdate={handleFilterUpdate}
          />
        );
      case 'queue':
        return (
          <AnomalyQueuePage
            cases={filteredCases}
            onCaseClick={handleCaseClick}
          />
        );
      case 'report':
        return (
          <ReportBuilderPage
            cases={filteredCases}
          />
        );
      case 'settings':
        return <SettingsPage />;
      default:
        return (
          <OverviewPage
            cases={filteredCases}
            alerts={mockAlerts}
            filters={filters}
            onCaseClick={handleCaseClick}
            onFilterUpdate={handleFilterUpdate}
          />
        );
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar currentPage={currentPage} onNavigate={handleNavigate} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Filter Bar - only show on overview and queue pages */}
        {(currentPage === 'overview' || currentPage === 'queue') && (
          <FilterBar filters={filters} onFilterChange={setFilters} />
        )}

        {/* Page Content */}
        <div className="flex-1 overflow-y-auto">
          {renderPage()}
        </div>
      </div>
    </div>
  );
}
