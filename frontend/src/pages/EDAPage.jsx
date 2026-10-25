import { useRef, useEffect, useState } from "react";
import {
  Chart, CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import { useData } from "../hooks/useData";
import { api } from "../utils/api";
import { baseChartOptions, COLORS } from "../utils/chartConfig";
import DataError from "../components/DataError";

Chart.register(CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler);

function SectionHeader({ title, sub }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <h2 style={{ fontSize: 17, fontWeight: 700, color: "#111827",
        fontFamily: "'IBM Plex Mono', monospace", margin: 0, letterSpacing: -0.3 }}>
        {title}
      </h2>
      {sub && <p style={{ fontSize: 12, color: "#4b5563", marginTop: 4,
        fontFamily: "'IBM Plex Sans', sans-serif" }}>{sub}</p>}
    </div>
  );
}

function ChartCard({ title, height = 260, children }) {
  return (
    <div style={{
      background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 12, padding: "20px 22px",
    }}>
      {title && (
        <div style={{ fontSize: 11, fontWeight: 600, color: "#374151",
          fontFamily: "'IBM Plex Mono', monospace", letterSpacing: 0.5,
          textTransform: "uppercase", marginBottom: 14 }}>
          {title}
        </div>
      )}
      <div style={{ height }}>{children}</div>
    </div>
  );
}

function StatBox({ label, value, unit }) {
  return (
    <div style={{ background: "#f9fafb", borderRadius: 8, padding: "12px 14px",
      border: "1px solid #e5e7eb" }}>
      <div style={{ fontSize: 10, color: "#4b5563", fontFamily: "'IBM Plex Sans', sans-serif",
        textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace",
        color: "#111827", marginTop: 4 }}>
        {value}<span style={{ fontSize: 11, color: "#4b5563", marginLeft: 3 }}>{unit}</span>
      </div>
    </div>
  );
}

// Heatmap — pure canvas (no chart.js)
function Heatmap({ data }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const { months, data: grid } = data;
    const W = canvas.width, H = canvas.height;
    const padL = 38, padT = 28, padB = 24, padR = 8;
    const cellW = (W - padL - padR) / 24;
    const cellH = (H - padT - padB) / 12;

    const allVals = grid.flat();
    const minV = Math.min(...allVals), maxV = Math.max(...allVals);

    const lerp = (t) => {
      // Yellow → Orange → Red gradient
      if (t < 0.5) {
        const s = t * 2;
        return `rgb(${Math.round(255*0.95)},${Math.round(255*(0.95-0.5*s))},${Math.round(255*(0.4-0.4*s))})`;
      } else {
        const s = (t - 0.5) * 2;
        return `rgb(${Math.round(255*(0.95-0.2*s))},${Math.round(255*(0.45-0.35*s))},${Math.round(255*(0-0)*s)})`;
      }
    };

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);

    // Draw cells
    for (let m = 0; m < 12; m++) {
      for (let h = 0; h < 24; h++) {
        const v = grid[m][h];
        const t = (v - minV) / (maxV - minV);
        ctx.fillStyle = lerp(t);
        ctx.fillRect(padL + h * cellW, padT + m * cellH, cellW - 1, cellH - 1);
      }
    }

    // Month labels
    ctx.fillStyle = "#374151";
    ctx.font = "600 9px 'IBM Plex Mono'";
    months.forEach((mn, i) => {
      ctx.fillText(mn, 2, padT + i * cellH + cellH / 2 + 3);
    });

    // Hour labels
    ctx.fillStyle = "#374151";
    ctx.font = "9px 'IBM Plex Mono'";
    for (let h = 0; h < 24; h += 3) {
      ctx.fillText(`${h}`, padL + h * cellW + 2, padT - 6);
    }
  }, [data]);

  return <canvas ref={canvasRef} width={700} height={260}
    style={{ width: "100%", height: "100%", imageRendering: "pixelated" }} />;
}

export default function EDAPage() {
  const { data: hourly, error: hourlyError } = useData(api.edaHourly);
  const { data: monthly, error: monthlyError } = useData(api.edaMonthly);
  const { data: daily, error: dailyError }  = useData(api.edaDaily);
  const { data: yearly, error: yearlyError } = useData(api.edaYearly);
  const { data: heatmap, error: heatmapError } = useData(api.edaHeatmap);
  const { data: wdwe, error: wdweError }   = useData(api.edaWeekdayWeekend);
  const { data: stats, error: statsError }  = useData(api.statistics);
  const dataError = hourlyError || monthlyError || dailyError || yearlyError
    || heatmapError || wdweError || statsError;

  const opts = baseChartOptions;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>

      <DataError error={dataError} label="EDA data" />

      {/* Dataset Stats */}
      <div>
        <SectionHeader title="Dataset Overview"
          sub={`${stats?.dataset ?? 'IIT Bombay Residential Buildings'} · hourly resolution · timezone: IST (Asia/Kolkata)`} />
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8 }}>
          {stats && [
            { label: "Total Hours", value: stats.total_hours.toLocaleString(), unit: "" },
            { label: "Train Hours", value: stats.train_hours.toLocaleString(), unit: "" },
            { label: "Test Hours",  value: stats.test_hours.toLocaleString(), unit: "" },
            { label: "Mean Load",   value: stats.W_mean.toFixed(1), unit: "Wh" },
            { label: "Max Load",    value: stats.W_max.toFixed(1), unit: "Wh" },
            { label: "Min Load",    value: stats.W_min.toFixed(1), unit: "Wh" },
            { label: "Peak Hour",   value: `${stats.peak_hour}:00`, unit: "" },
            { label: "Peak Month",  value: stats.peak_month, unit: "" },
          ].map(s => <StatBox key={s.label} {...s} />)}
        </div>
      </div>

      {/* Yearly trend */}
      <div>
        <SectionHeader title="Yearly Load Trend"
          sub="Daily mean electricity consumption trend (Wh)" />
        <ChartCard height={220}>
          {yearly && (
            <Line
              data={{
                labels: yearly.labels,
                datasets: [{
                  label: "Daily Mean Load (Wh)",
                  data: yearly.actual,
                  borderColor: COLORS.actual.line,
                  backgroundColor: "rgba(26,26,46,0.06)",
                  borderWidth: 1.2,
                  fill: true,
                  tension: 0.3,
                  pointRadius: 0,
                }]
              }}
              options={{
                ...opts,
                scales: {
                  ...opts.scales,
                  x: { ...opts.scales.x, ticks: { ...opts.scales.x.ticks, maxTicksLimit: 14 } }
                }
              }}
            />
          )}
        </ChartCard>
      </div>

      {/* 2×2 grid: hourly, monthly, daily, weekday/weekend */}
      <div>
        <SectionHeader title="Load Variation — Hourly · Monthly · Weekly"
          sub="Average electricity consumption patterns across different time scales (Wh)" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>

          {/* Hourly profile */}
          <ChartCard title="Average Hourly Load Profile (24h)">
            {hourly && (
              <Line
                data={{
                  labels: hourly.labels,
                  datasets: [
                    {
                      label: "Mean Load (Wh)",
                      data: hourly.load,
                      borderColor: "#2563eb",
                      backgroundColor: "rgba(37,99,235,0.08)",
                      fill: true, tension: 0.4, pointRadius: 2, borderWidth: 2,
                    },
                    {
                      label: "+1σ",
                      data: hourly.load.map((v, i) => v + hourly.std[i]),
                      borderColor: "rgba(37,99,235,0.25)",
                      borderDash: [4, 4], borderWidth: 1, pointRadius: 0,
                      fill: false,
                    },
                    {
                      label: "−1σ",
                      data: hourly.load.map((v, i) => Math.max(0, v - hourly.std[i])),
                      borderColor: "rgba(37,99,235,0.25)",
                      borderDash: [4, 4], borderWidth: 1, pointRadius: 0,
                      backgroundColor: "rgba(37,99,235,0.06)",
                      fill: "-1",
                    },
                  ]
                }}
                options={opts}
              />
            )}
          </ChartCard>

          {/* Monthly profile */}
          <ChartCard title="Monthly Average Load (±1σ)">
            {monthly && (
              <Bar
                data={{
                  labels: monthly.months,
                  datasets: [
                    {
                      label: "Monthly Avg (Wh)",
                      data: monthly.load,
                      backgroundColor: monthly.load.map((v, i) => {
                        const t = i / 11;
                        const r = Math.round(37 + (220 - 37) * t);
                        const g = Math.round(99 + (38 - 99) * t);
                        const b = Math.round(235 + (38 - 235) * t);
                        return `rgba(${r},${g},${b},0.82)`;
                      }),
                      borderRadius: 5,
                      borderSkipped: false,
                    }
                  ]
                }}
                options={opts}
              />
            )}
          </ChartCard>

          {/* Daily avg by weekday */}
          <ChartCard title="Average Load by Day of Week">
            {daily && (
              <Bar
                data={{
                  labels: daily.days,
                  datasets: [{
                      label: "Daily Avg (Wh)",
                    data: daily.load,
                    backgroundColor: daily.days.map((d, i) =>
                      i >= 5 ? "rgba(220,38,38,0.75)" : "rgba(37,99,235,0.75)"
                    ),
                    borderRadius: 5,
                    borderSkipped: false,
                  }]
                }}
                options={{
                  ...opts,
                  plugins: {
                    ...opts.plugins,
                    legend: { display: false },
                  }
                }}
              />
            )}
          </ChartCard>

          {/* Weekday vs Weekend */}
          <ChartCard title="Weekday vs Weekend Hourly Profile (Wh)">
            {wdwe && (
              <Line
                data={{
                  labels: wdwe.labels,
                  datasets: [
                    {
                      label: "Weekday",
                      data: wdwe.weekday,
                      borderColor: "#2563eb",
                      backgroundColor: "rgba(37,99,235,0.07)",
                      fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2,
                    },
                    {
                      label: "Weekend",
                      data: wdwe.weekend,
                      borderColor: "#dc2626",
                      backgroundColor: "rgba(220,38,38,0.07)",
                      fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2,
                      borderDash: [5, 3],
                    },
                  ]
                }}
                options={opts}
              />
            )}
          </ChartCard>
        </div>
      </div>

      {/* Heatmap */}
      <div>
        <SectionHeader title="Hour × Month Heatmap"
          sub="Average load (Wh) for each hour of day × month combination" />
        <ChartCard height={280}>
          <Heatmap data={heatmap} />
        </ChartCard>
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8, justifyContent: "flex-end" }}>
          <span style={{ fontSize: 10, color: "#4b5563" }}>Low</span>
          <div style={{ width: 120, height: 10, borderRadius: 4,
            background: "linear-gradient(to right, #f5e642, #ff7f00, #cc2200)", border: "1px solid #e5e7eb" }}/>
          <span style={{ fontSize: 10, color: "#4b5563" }}>High</span>
        </div>
      </div>
    </div>
  );
}
