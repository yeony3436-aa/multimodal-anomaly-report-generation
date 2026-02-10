// src/app/App.tsx
import React, { useState, useMemo, useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { FilterBar, FilterState } from "./components/FilterBar";

// pages
import { OverviewPage } from "./pages/OverviewPage";
import { AnomalyQueuePage } from "./pages/AnomalyQueuePage";
import { CaseDetailPage } from "./pages/CaseDetailPage";
import { ReportBuilderPage } from "./pages/ReportBuilderPage";
import { SettingsPage } from "./pages/SettingsPage";
import { LlavaReportsPage } from "./pages/LlavaReportsPage";

// data & utils
import { mapLlavaReportsToAnomalyCases } from "./data/llavaMapper";
import { mockCases, AnomalyCase } from "./data/mockData";
import { mockAlerts, Alert, NotificationSettings } from "./data/AlertData";
import { getDateRangeWindow } from "./utils/dateUtils";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const MODEL_VERSION: Record<string, string> = {
  PatchCore: "v2.3.1",
  EfficientAD: "v3.1.0",
};

const DEFAULT_NOTI: NotificationSettings = {
  highSeverity: true,
  reviewRequest: true,
  dailyReport: false,
  systemError: true,
  consecutiveDefects: true,
};

function clamp01(x: number) {
  if (Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<string>("overview");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);

  const [cases, setCases] = useState<AnomalyCase[]>([]);
  const [alerts] = useState<Alert[]>(mockAlerts);

  // ✅ 전역 모델/임계값/알림설정
  const [activeModel, setActiveModel] = useState<string>(() => {
    return localStorage.getItem("activeModel") ?? "PatchCore";
  });

  const [threshold, setThreshold] = useState<number>(() => {
    const v = Number(localStorage.getItem("threshold"));
    return Number.isFinite(v) ? clamp01(v) : 0.65;
  });

  const [notificationSettings, setNotificationSettings] =
    useState<NotificationSettings>(() => {
      try {
        const raw = localStorage.getItem("notificationSettings");
        if (!raw) return DEFAULT_NOTI;
        return { ...DEFAULT_NOTI, ...JSON.parse(raw) };
      } catch {
        return DEFAULT_NOTI;
      }
    });

  useEffect(() => {
    localStorage.setItem("activeModel", activeModel);
  }, [activeModel]);

  useEffect(() => {
    localStorage.setItem("threshold", String(threshold));
  }, [threshold]);

  useEffect(() => {
    localStorage.setItem(
      "notificationSettings",
      JSON.stringify(notificationSettings),
    );
  }, [notificationSettings]);

  // ✅ 백엔드에서 llava reports를 받아 케이스로 매핑
  useEffect(() => {
    const params = new URLSearchParams();
    params.set("limit", "5000");
    params.set("offset", "0");

    const endpoint = `${API_BASE}/llava/reports?${params.toString()}`;
    console.log(`📡 데이터 요청 시작: ${endpoint}`);

    fetch(endpoint)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP 에러! 상태코드: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        console.log("✅ 백엔드 데이터 수신 성공:", data);
        const rawItems = Array.isArray(data) ? data : data.items || [];
        const mappedCases = mapLlavaReportsToAnomalyCases(rawItems);
        setCases(mappedCases);
      })
      .catch((err) => {
        console.error("❌ 데이터 가져오기 실패 (Mock 데이터 사용):", err);
        setCases(mockCases);
      });
  }, []);

  // ✅ 필터 상태
  const [filters, setFilters] = useState<FilterState>({
    dateRange: "today",
    line: "all",
    productGroup: "all",
    defectType: "all",
    decision: "all",
    scoreRange: [0, 1],
  });

  // ✅ 모델/임계값 설정을 모든 케이스에 주입 (표시/리포트 기준 통일)
  const casesWithSettings = useMemo(() => {
    const version = MODEL_VERSION[activeModel] ?? "v1.0.0";
    return cases.map((c) => ({
      ...c,
      model_name: activeModel,
      model_version: version,
      threshold,
    }));
  }, [cases, activeModel, threshold]);

  // ✅ 필터 적용
  const filteredCases = useMemo(() => {
    const window = getDateRangeWindow(filters.dateRange);

    return casesWithSettings.filter((c) => {
      if (window) {
        const t = c.timestamp.getTime();
        if (t < window.from.getTime() || t > window.to.getTime()) return false;
      }
      if (filters.line !== "all" && c.line_id !== filters.line) return false;
      if (
        filters.productGroup !== "all" &&
        c.product_group !== filters.productGroup
      )
        return false;
      if (filters.defectType !== "all" && c.defect_type !== filters.defectType)
        return false;
      if (filters.decision !== "all" && c.decision !== filters.decision)
        return false;
      if (
        c.anomaly_score < filters.scoreRange[0] ||
        c.anomaly_score > filters.scoreRange[1]
      )
        return false;

      return true;
    });
  }, [casesWithSettings, filters]);

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
    setSelectedCaseId(null);
  };

  const handleCaseClick = (caseId: string) => {
    setSelectedCaseId(caseId);
    setCurrentPage("detail");
  };

  // ✅ 상세페이지에서도 모델/임계값 반영된 case를 사용
  const currentCase = selectedCaseId
    ? casesWithSettings.find((c) => c.id === selectedCaseId) ?? null
    : null;

  const renderPage = () => {
    if (currentPage === "detail" && currentCase) {
      return (
        <CaseDetailPage
          caseData={currentCase}
          onBack={() => setCurrentPage("queue")}
        />
      );
    }

    switch (currentPage) {
      case "overview":
        return (
          <OverviewPage
            cases={filteredCases}
            alerts={alerts}
            filters={filters}
            activeModel={activeModel}
            onCaseClick={handleCaseClick}
            onFilterUpdate={(newFilters) =>
              setFilters((prev) => ({ ...prev, ...newFilters }))
            }
          />
        );
      case "queue":
        return <AnomalyQueuePage cases={filteredCases} onCaseClick={handleCaseClick} />;
      case "report":
        return <ReportBuilderPage cases={filteredCases} />;
      case "llava":
        return <LlavaReportsPage apiBase={API_BASE} />;
      case "settings":
        return (
          <SettingsPage
            activeModel={activeModel}
            onModelChange={setActiveModel}
            threshold={threshold}
            onThresholdChange={setThreshold}
            notifications={notificationSettings}
            onNotificationsChange={setNotificationSettings}
          />
        );
      default:
        return (
          <OverviewPage
            cases={filteredCases}
            alerts={alerts}
            filters={filters}
            activeModel={activeModel}
            onCaseClick={handleCaseClick}
            onFilterUpdate={(newFilters) =>
              setFilters((prev) => ({ ...prev, ...newFilters }))
            }
          />
        );
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar currentPage={currentPage} onNavigate={handleNavigate} />
      <div className="flex-1 flex flex-col overflow-hidden">
        {(currentPage === "overview" || currentPage === "queue") && (
          <FilterBar filters={filters} onFilterChange={(next) => setFilters(next)} />
        )}
        <div className="flex-1 overflow-y-auto">{renderPage()}</div>
      </div>
    </div>
  );
}
