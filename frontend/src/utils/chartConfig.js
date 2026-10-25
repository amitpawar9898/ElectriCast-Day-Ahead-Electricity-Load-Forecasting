// Chart.js default config for all charts — clean white theme

export const COLORS = {
  actual:  { line: "#1a1a2e", fill: "rgba(26,26,46,0.08)",  label: "Actual" },
  sarima:  { line: "#dc2626", fill: "rgba(220,38,38,0.08)", label: "SARIMA" },
  xgboost: { line: "#16a34a", fill: "rgba(22,163,74,0.09)", label: "XGBoost" },
  std:     "rgba(100,116,139,0.18)",
};

export const MODEL_META = {
  sarima:  { label: "SARIMA",  color: "#dc2626", bg: "rgba(220,38,38,0.80)"  },
  xgboost: { label: "XGBoost", color: "#16a34a", bg: "rgba(22,163,74,0.80)"  },
  actual:  { label: "Actual",  color: "#1a1a2e", bg: "rgba(26,26,46,0.80)"   },
};

export const baseChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: "top",
      align: "end",
      labels: {
        usePointStyle: true,
        pointStyleWidth: 10,
        font: { family: "'IBM Plex Sans', sans-serif", size: 11 },
        color: "#374151",
        padding: 16,
      },
    },
    tooltip: {
      backgroundColor: "#ffffff",
      titleColor: "#111827",
      bodyColor: "#374151",
      borderColor: "#e5e7eb",
      borderWidth: 1,
      padding: 10,
      titleFont: { family: "'IBM Plex Mono', monospace", size: 11 },
      bodyFont:  { family: "'IBM Plex Sans', sans-serif", size: 11 },
    },
  },
  scales: {
    x: {
      grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
      ticks: { color: "#4b5563", font: { family: "'IBM Plex Mono', monospace", size: 10 }, maxTicksLimit: 12 },
      border: { display: false },
    },
    y: {
      grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
      ticks: { color: "#4b5563", font: { family: "'IBM Plex Mono', monospace", size: 10 } },
      border: { display: false },
    },
  },
  elements: {
    point: { radius: 0, hoverRadius: 4 },
    line: { tension: 0.35, borderWidth: 2 },
  },
};

export function lineDs(label, data, colorKey, opts = {}) {
  const c = COLORS[colorKey] || COLORS.actual;
  return {
    label,
    data,
    borderColor: c.line,
    backgroundColor: c.fill,
    fill: opts.fill ?? false,
    borderWidth: opts.borderWidth ?? 2,
    borderDash: opts.dash ?? [],
    tension: 0.35,
    pointRadius: 0,
    pointHoverRadius: 4,
    ...opts,
  };
}

export function barDs(label, data, color, opts = {}) {
  return {
    label,
    data,
    backgroundColor: color,
    borderColor: color.replace(/[\d.]+\)$/, "1)"),
    borderWidth: 1,
    borderRadius: 4,
    borderSkipped: false,
    ...opts,
  };
}

export const MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
export const DAY_LABELS   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
