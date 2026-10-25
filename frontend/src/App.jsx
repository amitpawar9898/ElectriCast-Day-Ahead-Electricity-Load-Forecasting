import { useState } from "react";
import MetricsBar from "./components/MetricsBar";
import EDAPage from "./pages/EDAPage";
import PredictionsPage from "./pages/PredictionsPage";
import WeeklyPage from "./pages/WeeklyPage";
import MetricsPage from "./pages/MetricsPage";
import ExplainPage from "./pages/ExplainPage";
import DataQualityPage from "./pages/DataQualityPage";
import LimitationsPage from "./pages/LimitationsPage";
import UploadPage from "./pages/UploadPage";
import ChatDrawer from "./components/ChatDrawer";
import "./index.css";

const NAV_ITEMS = [
  { key: "eda",         label: "EDA",           icon: "◈" },
  { key: "predictions", label: "Predictions",    icon: "◉" },
  { key: "weekly",      label: "Weekly Pattern", icon: "◇" },
  { key: "metrics",     label: "Model Metrics",  icon: "◆" },
  { key: "explain",     label: "Explain",        icon: "◎" },
  { key: "quality",     label: "Data Quality",   icon: "▤" },
  { key: "limitations", label: "Limitations",    icon: "⚑" },
  { key: "upload",      label: "Upload & Validate", icon: "⬆" },
];

export default function App() {
  const [page, setPage] = useState("eda");

  return (
    <div style={{ minHeight: "100vh", background: "#f4f6f8", fontFamily: "'IBM Plex Sans', sans-serif" }}>
      {/* Header */}
      <header style={{
        position: "sticky", top: 0, zIndex: 100,
        background: "#ffffff",
        borderBottom: "1px solid #e5e7eb",
        boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
      }}>
        <div className="app-header" style={{ maxWidth: 1280, margin: "0 auto", padding: "0 28px",
          display: "flex", alignItems: "center", height: 56, gap: 24, minWidth: 0 }}>

          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
            <div style={{
              width: 30, height: 30, borderRadius: 8,
              background: "linear-gradient(135deg, #1a1a2e 0%, #2563eb 100%)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, color: "#fff",
            }}>⚡</div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#111827",
                fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.1 }}>ElectriCast</div>
              <div style={{ fontSize: 9, color: "#4b5563", fontFamily: "'IBM Plex Sans', sans-serif" }}>
                IIT Bombay Residential Buildings · Energy Forecasting
              </div>
            </div>
          </div>

          {/* Nav */}
          <nav className="app-nav" style={{ display: "flex", gap: 4, flex: "1 1 0", minWidth: 0,
            overflowX: "auto", overflowY: "hidden" }}>
            {NAV_ITEMS.map(({ key, label, icon }) => {
              const active = page === key;
              return (
                <button key={key} onClick={() => setPage(key)} style={{
                  display: "flex", alignItems: "center", gap: 6,
                  padding: "6px 14px", borderRadius: 7, border: "none", cursor: "pointer",
                  background: active ? "#eff6ff" : "transparent",
                  color: active ? "#2563eb" : "#4b5563",
                  fontFamily: "'IBM Plex Mono', monospace",
                  fontSize: 11, fontWeight: active ? 700 : 500,
                  transition: "all 0.15s", whiteSpace: "nowrap", flexShrink: 0,
                  borderBottom: active ? "2px solid #2563eb" : "2px solid transparent",
                }}>
                  <span style={{ fontSize: 12 }}>{icon}</span>
                  {label}
                </button>
              );
            })}
          </nav>

          {/* Badge */}
          <div className="app-badges" style={{ display: "flex", gap: 6, flexShrink: 0 }}>
            {[
              { label: "SARIMA", color: "#b91c1c" },
              { label: "XGBoost", color: "#15803d" },
            ].map(({ label, color }) => (
              <span key={label} style={{
                fontSize: 9, fontWeight: 700, padding: "2px 8px",
                borderRadius: 10, border: `1px solid ${color}40`,
                background: `${color}12`, color, fontFamily: "'IBM Plex Mono', monospace",
                letterSpacing: 0.5, textTransform: "uppercase",
              }}>{label}</span>
            ))}
          </div>
        </div>
      </header>

      {/* Metrics bar */}
      <div style={{ background: "#f0f4ff", borderBottom: "1px solid #dce4fb",
        padding: "12px 28px" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto" }}>
          <MetricsBar />
        </div>
      </div>

      {/* Page content */}
      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 28px 48px" }}>
        {page === "eda"         && <EDAPage />}
        {page === "predictions" && <PredictionsPage />}
        {page === "weekly"      && <WeeklyPage />}
        {page === "metrics"     && <MetricsPage />}
        {page === "explain"     && <ExplainPage />}
        {page === "quality"     && <DataQualityPage />}
        {page === "limitations" && <LimitationsPage />}
        {page === "upload"      && <UploadPage />}
      </main>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid #e5e7eb", background: "#fff",
        padding: "12px 28px", textAlign: "center" }}>
        <div style={{ fontSize: 10, color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace" }}>
          ElectriCast &nbsp;·&nbsp; Energy Demand Forecasting &nbsp;·&nbsp;
          SARIMA · XGBoost
        </div>
      </footer>

      <ChatDrawer />
    </div>
  );
}
