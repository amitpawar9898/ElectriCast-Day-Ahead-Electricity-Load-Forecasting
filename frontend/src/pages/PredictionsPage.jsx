import { useState } from "react";
import {
  Chart, CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import { useData } from "../hooks/useData";
import { api } from "../utils/api";
import { baseChartOptions } from "../utils/chartConfig";
import DataError from "../components/DataError";

Chart.register(CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler);

const SERIES = {
  actual:         { label: "Actual",            color: "#111827", dash: [] },
  seasonal_naive: { label: "Seasonal naive 24h", color: "#6b7280", dash: [2, 3] },
  xgboost:        { label: "XGBoost",            color: "#16a34a", dash: [8, 3] },
  sarima:         { label: "SARIMA",             color: "#dc2626", dash: [3, 2] },
  p50:            { label: "Quantile P50",       color: "#d97706", dash: [] },
};

const MAE_KEYS = ["xgboost", "sarima", "p50", "seasonal_naive"];

const BAR_COLORS = {
  xgboost: "rgba(22,163,74,0.78)",
  sarima: "rgba(220,38,38,0.78)",
  p50: "rgba(217,119,6,0.78)",
  seasonal_naive: "rgba(107,114,128,0.55)",
};

const fmtWh = (value) => Number.isFinite(value) ? `${value.toFixed(1)} Wh` : "N/A";

function Card({ title, sub, height, children }) {
  return (
    <div style={{ background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 12, padding: "18px 22px" }}>
      {title && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#374151",
            fontFamily: "'IBM Plex Mono', monospace",
            textTransform: "uppercase", letterSpacing: 0.5 }}>{title}</div>
          {sub && <div style={{ fontSize: 11, color: "#4b5563", marginTop: 3 }}>{sub}</div>}
        </div>
      )}
      {height ? <div style={{ height }}>{children}</div> : children}
    </div>
  );
}

function ToggleBtn({ label, active, color, onClick }) {
  return (
    <button onClick={onClick} style={{
      display: "flex", alignItems: "center", gap: 6,
      padding: "5px 12px", borderRadius: 6, cursor: "pointer", border: "1.5px solid",
      borderColor: active ? color : "#e5e7eb",
      background: active ? `${color}14` : "#fff",
      transition: "all 0.15s", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace",
      color: active ? color : "#4b5563", fontWeight: 600,
    }}>
      <span style={{ width: 8, height: 8, borderRadius: 2, background: active ? color : "#d1d5db" }}/>
      {label}
    </button>
  );
}

function MaeCard({ label, color, stats }) {
  return (
    <div style={{ border: "1px solid #e5e7eb", borderLeft: `3px solid ${color}`,
      borderRadius: 8, padding: "8px 12px", background: "#fff" }}>
      <div style={{ fontSize: 9, color: "#4b5563", fontFamily: "'IBM Plex Sans', sans-serif",
        textTransform: "uppercase", letterSpacing: 0.4 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace",
        color: "#111827", marginTop: 2 }}>{fmtWh(stats?.mae_wh)}</div>
      <div style={{ fontSize: 9, color: "#6b7280", fontFamily: "'IBM Plex Mono', monospace" }}>
        n={stats?.n ?? 0}
      </div>
    </div>
  );
}

function errorDatasets(rows) {
  return MAE_KEYS.map(key => ({
    label: key === "p50" ? "Quantile P50" : SERIES[key].label,
    data: rows.map(row => row.models?.[key]?.mae_wh ?? null),
    backgroundColor: BAR_COLORS[key],
    borderRadius: 3,
    borderSkipped: false,
  }));
}

export default function PredictionsPage() {
  const [visible, setVisible] = useState({
    actual: true, seasonal_naive: false, xgboost: true, sarima: true, quantile: true,
  });
  const [foldId, setFoldId] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);

  const { data: daysData, loading: daysLoading, error: daysError } =
    useData(api.backtestDays);
  const { data: errors, loading: errorsLoading, error: errorsError } =
    useData(api.backtestErrors);

  const folds = daysData?.folds || [];
  const activeFold = folds.find(fold => fold.id === foldId) || folds[0] || null;
  const foldDays = activeFold?.days || [];
  const effectiveDate = foldDays.some(day => day.date === selectedDate)
    ? selectedDate
    : (foldDays[0]?.date ?? null);

  const { data: day, loading: dayLoading, error: dayError } = useData(
    (signal) => effectiveDate
      ? api.backtestDay(effectiveDate, signal)
      : Promise.resolve(null),
    [effectiveDate]
  );

  if (daysError) return <DataError error={daysError} label="backtest days" />;
  if (errorsError) return <DataError error={errorsError} label="backtest errors" />;

  if (daysLoading || errorsLoading) {
    return <div style={{ color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace" }}>
      Loading backtest predictions...
    </div>;
  }

  if (!folds.length) {
    return (
      <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12,
        padding: "18px 22px", fontSize: 12, color: "#4b5563",
        fontFamily: "'IBM Plex Mono', monospace" }}>
        No hourly backtest predictions available. Run evaluate.py to regenerate results.json.
      </div>
    );
  }

  const opts = baseChartOptions;
  const toggle = key => setVisible(prev => ({ ...prev, [key]: !prev[key] }));

  const yAxis = {
    ...opts.scales.y,
    title: {
      display: true, text: "Wh", color: "#6b7280",
      font: { family: "'IBM Plex Mono', monospace", size: 10 },
    },
  };

  const hourlyDatasets = day ? [
    {
      label: "P10", data: day.p10, borderColor: "rgba(217,119,6,0.5)",
      backgroundColor: "transparent", borderWidth: 1, borderDash: [3, 3],
      pointRadius: 0, tension: 0.3, fill: false, hidden: !visible.quantile,
    },
    {
      label: "P10–P90 band", data: day.p90, borderColor: "rgba(217,119,6,0.5)",
      backgroundColor: "rgba(217,119,6,0.15)", borderWidth: 1, borderDash: [3, 3],
      pointRadius: 0, tension: 0.3, fill: 0, hidden: !visible.quantile,
    },
    ...["actual", "seasonal_naive", "xgboost", "sarima"].map(key => ({
      label: SERIES[key].label,
      data: day[key],
      borderColor: SERIES[key].color,
      backgroundColor: "transparent",
      borderDash: SERIES[key].dash,
      borderWidth: key === "actual" ? 2.2 : 1.6,
      pointRadius: 0, pointHoverRadius: 4, tension: 0.3, fill: false,
      hidden: !visible[key],
    })),
    {
      label: SERIES.p50.label, data: day.p50, borderColor: SERIES.p50.color,
      backgroundColor: "transparent", borderWidth: 1.8, pointRadius: 0,
      pointHoverRadius: 4, tension: 0.3, fill: false, hidden: !visible.quantile,
    },
  ] : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      <Card title="Day-ahead backtest predictions"
        sub="Every day is forecast from 00:00 IST using only earlier observations. The band is the uncalibrated P10–P90 quantile XGBoost interval; error breakdowns use hours with both an actual value and a seasonal-naive reference. SARIMA can cover fewer hours after long gaps.">
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          {folds.map(fold => (
            <button key={fold.id} onClick={() => { setFoldId(fold.id); setSelectedDate(null); }}
              style={{
                padding: "5px 12px", borderRadius: 6, cursor: "pointer",
                border: "1.5px solid",
                borderColor: activeFold?.id === fold.id ? "#2563eb" : "#e5e7eb",
                background: activeFold?.id === fold.id ? "#eff6ff" : "#fff",
                color: activeFold?.id === fold.id ? "#2563eb" : "#4b5563",
                fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, fontWeight: 600,
              }}>{fold.label}</button>
          ))}
          <select value={effectiveDate || ""}
            onChange={event => setSelectedDate(event.target.value)}
            disabled={!foldDays.length}
            style={{ marginLeft: "auto", padding: "5px 10px", borderRadius: 6,
              border: "1.5px solid #e5e7eb", fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 11, color: "#374151", cursor: "pointer", background: "#fff",
              maxWidth: "100%" }}>
            {foldDays.map(foldDay => (
              <option key={foldDay.date} value={foldDay.date}>
                {foldDay.date} · {foldDay.weekday} · {foldDay.eval_hours} h
              </option>
            ))}
          </select>
        </div>
      </Card>

      <DataError error={dayError} label="selected day" />

      {dayLoading && !day && (
        <div style={{ color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>
          Loading selected day...
        </div>
      )}

      {day && (
        <>
          <Card>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#111827",
                fontFamily: "'IBM Plex Mono', monospace" }}>
                {day.weekday}, {day.date}
              </div>
              <div style={{ fontSize: 11, color: "#4b5563", fontFamily: "'IBM Plex Sans', sans-serif" }}>
                {day.fold_label} · {day.n_eval_hours} evaluable hours · actual peak {fmtWh(day.actual_peak_wh)}
                {" "}· actual mean {fmtWh(day.actual_mean_wh)}
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: 8, marginTop: 12 }}>
              {MAE_KEYS.map(key => (
                <MaeCard key={key}
                  label={key === "p50" ? "Quantile P50" : SERIES[key].label}
                  color={SERIES[key].color}
                  stats={day.model_mae?.[key]} />
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
              {["actual", "xgboost", "sarima", "seasonal_naive"].map(key => (
                <ToggleBtn key={key} label={SERIES[key].label}
                  active={visible[key]} color={SERIES[key].color}
                  onClick={() => toggle(key)} />
              ))}
              <ToggleBtn label="P10–P90 / P50" active={visible.quantile}
                color="#d97706" onClick={() => toggle("quantile")} />
            </div>
          </Card>

          <Card title={`Hourly load — ${day.date}`}
            sub="Actual versus day-ahead forecasts (Wh per reporting apartment)">
            <div style={{ height: 320 }}>
              <Line
                data={{ labels: day.labels, datasets: hourlyDatasets }}
                options={{
                  ...opts,
                  scales: { ...opts.scales, y: yAxis },
                  plugins: {
                    ...opts.plugins,
                    legend: {
                      ...opts.plugins.legend,
                      labels: {
                        ...opts.plugins.legend.labels,
                        filter: item => item.text !== "P10",
                      },
                    },
                    tooltip: {
                      ...opts.plugins.tooltip,
                      callbacks: {
                        label: ctx => ctx.parsed.y === null
                          ? `${ctx.dataset.label}: N/A`
                          : `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} Wh`,
                      },
                    },
                  },
                }}
              />
            </div>
          </Card>
        </>
      )}

      {errors && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 14 }}>
            <Card title="MAE by hour of day" sub="Pooled across all evaluated folds, including the 2018 stress test; each model uses its available hours" height={250}>
              <Bar
                data={{
                  labels: errors.by_hour.map(row => `${String(row.hour).padStart(2, "0")}:00`),
                  datasets: errorDatasets(errors.by_hour),
                }}
                options={{ ...opts, scales: { ...opts.scales, y: yAxis } }}
              />
            </Card>
            <Card title="MAE by calendar month" sub="Months present in all evaluated folds, including the 2018 stress test" height={250}>
              <Bar
                data={{
                  labels: errors.by_month.map(row => row.label),
                  datasets: errorDatasets(errors.by_month),
                }}
                options={{ ...opts, scales: { ...opts.scales, y: yAxis } }}
              />
            </Card>
          </div>

          <Card title="Worst days"
            sub={`Top 10 days ranked by ${errors.ranking}. Per-model n counts the hours actually scored.`}>
            <div style={{ overflowX: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "max-content",
                minWidth: "100%", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11 }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                    {["Date", "Fold", "Day", "Hours", "Actual peak", "XGBoost", "SARIMA", "Quantile P50", "Seasonal naive"].map(header => (
                      <th key={header} style={{
                        padding: "8px 10px", whiteSpace: "nowrap",
                        textAlign: header === "Date" ? "left" : "right",
                        fontSize: 9, color: "#4b5563", fontWeight: 700,
                        textTransform: "uppercase", letterSpacing: 0.5,
                      }}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {errors.worst_days.map(row => (
                    <tr key={`${row.fold_label}-${row.date}`}
                      style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "8px 10px", whiteSpace: "nowrap" }}>{row.date}</td>
                      <td style={{ padding: "8px 10px", textAlign: "right", whiteSpace: "nowrap",
                        color: "#4b5563" }}>{row.fold_label}</td>
                      <td style={{ padding: "8px 10px", textAlign: "right" }}>{row.weekday}</td>
                      <td style={{ padding: "8px 10px", textAlign: "right", color: "#4b5563" }}>
                        {row.n_eval_hours}
                      </td>
                      <td style={{ padding: "8px 10px", textAlign: "right" }}>{fmtWh(row.actual_peak_wh)}</td>
                      {MAE_KEYS.map(key => (
                        <td key={key} title={`n=${row.models?.[key]?.n ?? 0}`}
                          style={{ padding: "8px 10px", textAlign: "right", whiteSpace: "nowrap" }}>
                          {fmtWh(row.models?.[key]?.mae_wh)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 10, fontSize: 10, color: "#4b5563",
              fontFamily: "'IBM Plex Sans', sans-serif" }}>
              Rankings are computed from the committed backtest artifact; hovering a model cell shows its paired hour count.
              Days with fewer than 12 evaluable hours are excluded from the ranking.
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
