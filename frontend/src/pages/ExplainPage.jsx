import { useState } from "react";
import {
  Chart, CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend
} from "chart.js";
import { Bar, Scatter } from "react-chartjs-2";
import { useData } from "../hooks/useData";
import { api } from "../utils/api";
import { baseChartOptions } from "../utils/chartConfig";
import DataError from "../components/DataError";

Chart.register(CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend);

const RESIDUAL_MODELS = {
  xgboost: { label: "XGBoost", color: "#16a34a" },
  sarima:  { label: "SARIMA", color: "#dc2626" },
  p50:     { label: "Quantile P50", color: "#d97706" },
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

function ChipButton({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: "5px 12px", borderRadius: 6, cursor: "pointer",
      border: "1.5px solid",
      borderColor: active ? "#2563eb" : "#e5e7eb",
      background: active ? "#eff6ff" : "#fff",
      color: active ? "#2563eb" : "#4b5563",
      fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, fontWeight: 600,
    }}>{label}</button>
  );
}

function SummaryCard({ label, value, hint }) {
  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 8,
      padding: "10px 12px", background: "#f9fafb" }}>
      <div style={{ fontSize: 9, color: "#4b5563", fontFamily: "'IBM Plex Sans', sans-serif",
        textTransform: "uppercase", letterSpacing: 0.4 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace",
        color: "#111827", marginTop: 2 }}>{value}</div>
      {hint && <div style={{ fontSize: 9, color: "#6b7280",
        fontFamily: "'IBM Plex Sans', sans-serif" }}>{hint}</div>}
    </div>
  );
}

export default function ExplainPage() {
  const [foldId, setFoldId] = useState(null);
  const [model, setModel] = useState("xgboost");

  const { data: importance, loading: importanceLoading, error: importanceError } =
    useData(api.explainImportance);

  const folds = importance?.folds || [];
  const activeFold = folds.find(fold => fold.id === foldId) || folds[0] || null;

  const { data: residuals, loading: residualLoading, error: residualError } = useData(
    (signal) => activeFold
      ? api.explainResiduals(activeFold.id, model, signal)
      : Promise.resolve(null),
    [activeFold?.id, model]
  );

  if (importanceError) return <DataError error={importanceError} label="feature importance" />;

  if (importanceLoading) {
    return <div style={{ color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace" }}>
      Loading explanations...
    </div>;
  }

  if (!folds.length) {
    return (
      <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12,
        padding: "18px 22px", fontSize: 12, color: "#4b5563",
        fontFamily: "'IBM Plex Mono', monospace" }}>
        No gain importance available. Run evaluate.py to regenerate results.json.
      </div>
    );
  }

  const opts = baseChartOptions;
  const topFeatures = (activeFold?.features || []).slice(0, 14);
  const color = RESIDUAL_MODELS[model].color;

  const yAxisWh = {
    ...opts.scales.y,
    title: { display: true, text: "Wh", color: "#6b7280",
      font: { family: "'IBM Plex Mono', monospace", size: 10 } },
  };

  const predictedXs = residuals?.points?.map(point => point.x) || [];
  const zeroLine = predictedXs.length
    ? [{ x: Math.min(...predictedXs), y: 0 }, { x: Math.max(...predictedXs), y: 0 }]
    : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      <Card>
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#111827",
            fontFamily: "'IBM Plex Mono', monospace" }}>
            Explainability and residuals
          </div>
          <div style={{ fontSize: 11, color: "#4b5563",
            fontFamily: "'IBM Plex Sans', sans-serif" }}>
            Feature gain is read from the trained XGBoost booster; residual plots use the
            fold's hourly backtest rows. Nothing is retrained or tuned here.
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
          {folds.map(fold => (
            <ChipButton key={fold.id} label={fold.label}
              active={activeFold?.id === fold.id}
              onClick={() => setFoldId(fold.id)} />
          ))}
        </div>
      </Card>

      <Card
        title="XGBoost gain importance"
        sub={importance?.note}
        height={Math.max(280, topFeatures.length * 24)}
      >
        <Bar
          data={{
            labels: topFeatures.map(row => row.feature),
            datasets: [{
              label: "Gain share",
              data: topFeatures.map(row => +(row.share * 100).toFixed(2)),
              backgroundColor: "rgba(37,99,235,0.75)",
              borderColor: "rgba(37,99,235,1)",
              borderWidth: 1,
              borderRadius: 3,
            }],
          }}
          options={{
            ...opts,
            indexAxis: "y",
            plugins: {
              ...opts.plugins,
              legend: { ...opts.plugins.legend, display: false },
              tooltip: {
                ...opts.plugins.tooltip,
                callbacks: {
                  label: ctx => {
                    const row = topFeatures[ctx.dataIndex];
                    return `${ctx.parsed.x.toFixed(2)}% of gain (raw gain ${row.gain.toFixed(2)})`;
                  },
                },
              },
            },
            scales: {
              x: {
                ...opts.scales.x,
                title: { display: true, text: "Share of total gain (%)", color: "#6b7280",
                  font: { family: "'IBM Plex Mono', monospace", size: 10 } },
              },
              y: {
                ...opts.scales.y,
                ticks: { ...opts.scales.y.ticks, autoSkip: false },
              },
            },
          }}
        />
      </Card>

      <Card>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#374151",
            fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase",
            letterSpacing: 0.5 }}>
            Residual analysis — {activeFold?.label}
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginLeft: "auto" }}>
            {Object.entries(RESIDUAL_MODELS).map(([key, cfg]) => (
              <ChipButton key={key} label={cfg.label} active={model === key}
                onClick={() => setModel(key)} />
            ))}
          </div>
        </div>
        <div style={{ fontSize: 11, color: "#4b5563", marginTop: 6,
          fontFamily: "'IBM Plex Sans', sans-serif" }}>
          Residual = actual − forecast. Positive means the model underforecast.
          {model !== "xgboost" && " Gain importance above belongs to XGBoost only; residuals are shown for the selected model."}
        </div>

        {residualError && <div style={{ marginTop: 12 }}>
          <DataError error={residualError} label="residual data" />
        </div>}

        {residualLoading && !residuals && (
          <div style={{ marginTop: 12, fontSize: 12, color: "#4b5563",
            fontFamily: "'IBM Plex Mono', monospace" }}>
            Loading residuals...
          </div>
        )}

        {residuals && (
          <>
            <div style={{ display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              gap: 8, marginTop: 12 }}>
              <SummaryCard label="MAE" value={fmtWh(residuals.summary.mae_wh)}
                hint={`${residuals.summary.n} hours`} />
              <SummaryCard label="Mean residual (bias)"
                value={fmtWh(residuals.summary.mean_residual_wh)}
                hint="actual − forecast" />
              <SummaryCard label="RMSE" value={fmtWh(residuals.summary.rmse_wh)} />
              <SummaryCard label="Model" value={residuals.model_label}
                hint={`${residuals.fold_label} fold`} />
            </div>

            <div style={{ display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 14, marginTop: 16 }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#4b5563",
                  fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase",
                  letterSpacing: 0.5, marginBottom: 8 }}>
                  Residual vs predicted load
                </div>
                <div style={{ height: 300 }}>
                  <Scatter
                    data={{
                      datasets: [
                        {
                          label: residuals.model_label,
                          data: residuals.points,
                          backgroundColor: `${color}66`,
                          borderColor: color,
                          pointRadius: 2,
                          pointHoverRadius: 4,
                        },
                        {
                          type: "line",
                          label: "Zero residual",
                          data: zeroLine,
                          borderColor: "#9ca3af",
                          borderDash: [4, 4],
                          borderWidth: 1,
                          pointRadius: 0,
                          fill: false,
                        },
                      ],
                    }}
                    options={{
                      ...opts,
                      plugins: {
                        ...opts.plugins,
                        legend: {
                          ...opts.plugins.legend,
                          labels: {
                            ...opts.plugins.legend.labels,
                            filter: item => item.text !== "Zero residual",
                          },
                        },
                        tooltip: {
                          ...opts.plugins.tooltip,
                          callbacks: {
                            label: ctx => ctx.dataset.type === "line"
                              ? "zero"
                              : `predicted ${ctx.parsed.x.toFixed(1)} Wh · residual ${ctx.parsed.y.toFixed(1)} Wh`,
                          },
                        },
                      },
                      scales: {
                        x: {
                          ...opts.scales.x,
                          type: "linear",
                          title: { display: true, text: "Predicted load (Wh)",
                            color: "#6b7280",
                            font: { family: "'IBM Plex Mono', monospace", size: 10 } },
                        },
                        y: {
                          ...opts.scales.y,
                          title: { display: true, text: "Residual (actual − forecast, Wh)",
                            color: "#6b7280",
                            font: { family: "'IBM Plex Mono', monospace", size: 10 } },
                        },
                      },
                    }}
                  />
                </div>
              </div>

              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#4b5563",
                  fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase",
                  letterSpacing: 0.5, marginBottom: 8 }}>
                  Mean residual and MAE by hour
                </div>
                <div style={{ height: 300 }}>
                  <Bar
                    data={{
                      labels: residuals.by_hour.map(row => `${String(row.hour).padStart(2, "0")}:00`),
                      datasets: [
                        {
                          label: "Mean residual",
                          data: residuals.by_hour.map(row => row.mean_residual_wh),
                          backgroundColor: residuals.by_hour.map(row =>
                            (row.mean_residual_wh ?? 0) >= 0
                              ? "rgba(220,38,38,0.72)"
                              : "rgba(37,99,235,0.72)"),
                          borderRadius: 3,
                        },
                        {
                          label: "MAE",
                          data: residuals.by_hour.map(row => row.mae_wh),
                          backgroundColor: "rgba(107,114,128,0.35)",
                          borderRadius: 3,
                        },
                      ],
                    }}
                    options={{ ...opts, scales: { ...opts.scales, y: yAxisWh } }}
                  />
                </div>
              </div>
            </div>

            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#4b5563",
                fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase",
                letterSpacing: 0.5, marginBottom: 8 }}>
                Residual distribution
              </div>
              <div style={{ height: 220 }}>
                <Bar
                  data={{
                    labels: residuals.histogram.map(bin => bin.start.toFixed(1)),
                    datasets: [{
                      label: "Hours",
                      data: residuals.histogram.map(bin => bin.count),
                      backgroundColor: "rgba(37,99,235,0.6)",
                      borderRadius: 2,
                    }],
                  }}
                  options={{
                    ...opts,
                    plugins: {
                      ...opts.plugins,
                      legend: { ...opts.plugins.legend, display: false },
                      tooltip: {
                        ...opts.plugins.tooltip,
                        callbacks: {
                          title: items => {
                            const bin = residuals.histogram[items[0].dataIndex];
                            return `${bin.start.toFixed(1)} to ${bin.end.toFixed(1)} Wh`;
                          },
                          label: ctx => `${ctx.parsed.y} hours`,
                        },
                      },
                    },
                    scales: {
                      ...opts.scales,
                      x: {
                        ...opts.scales.x,
                        title: { display: true, text: "Residual bin start (Wh)",
                          color: "#6b7280",
                          font: { family: "'IBM Plex Mono', monospace", size: 10 } },
                      },
                      y: { ...opts.scales.y, beginAtZero: true },
                    },
                  }}
                />
              </div>
            </div>

            <div style={{ marginTop: 10, fontSize: 10, color: "#4b5563",
              fontFamily: "'IBM Plex Sans', sans-serif", lineHeight: 1.6 }}>
              Residuals use only hours where both the actual target and this model's forecast exist;
              SARIMA therefore shows fewer hours after long gaps. Gain importance is computed on the
              fold's training window and is descriptive, not causal; correlated lag features can share
              credit.
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
