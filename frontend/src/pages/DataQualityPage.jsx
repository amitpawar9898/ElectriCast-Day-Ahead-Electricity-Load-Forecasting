import {
  Chart, CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { useData } from "../hooks/useData";
import { api } from "../utils/api";
import { baseChartOptions } from "../utils/chartConfig";
import DataError from "../components/DataError";

Chart.register(CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend);

const MONO = "'IBM Plex Mono', monospace";
const SANS = "'IBM Plex Sans', sans-serif";

const CARD = {
  background: "#fff", border: "1px solid #e5e7eb",
  borderRadius: 12, padding: "18px 22px",
};

const CARD_TITLE = {
  fontSize: 11, fontWeight: 700, color: "#374151",
  fontFamily: MONO, textTransform: "uppercase",
  letterSpacing: 0.5, marginBottom: 14,
};

const TH = {
  padding: "8px 10px", textAlign: "left", fontSize: 10,
  color: "#4b5563", fontWeight: 700, fontFamily: MONO,
  textTransform: "uppercase", letterSpacing: 0.5,
  borderBottom: "2px solid #e5e7eb", position: "sticky",
  top: 0, background: "#fff",
};

const TD = {
  padding: "7px 10px", fontSize: 11, fontFamily: MONO,
  borderBottom: "1px solid #f3f4f6", color: "#374151",
};

const num = (value) => (Number.isFinite(value) ? value.toLocaleString("en-US") : "N/A");

function Stat({ label, value, sub, tone = "neutral" }) {
  const color = tone === "warn" ? "#b91c1c" : tone === "good" ? "#15803d" : "#111827";
  return (
    <div style={{ background: "#f9fafb", borderRadius: 8, padding: "12px 14px",
      border: "1px solid #e5e7eb" }}>
      <div style={{ fontSize: 10, color: "#4b5563", fontFamily: SANS,
        textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, fontFamily: MONO, color, marginTop: 3 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 10, color: "#4b5563", fontFamily: SANS, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

export default function DataQualityPage() {
  const { data, loading, error } = useData(api.dataQuality);

  if (error) return <DataError error={error} label="data quality" />;
  if (loading || !data) {
    return <div style={{ color: "#4b5563", fontFamily: MONO }}>Loading data quality...</div>;
  }

  const { source, panel, stable_panel, target, monthly_coverage, inactive, notes } = data;
  const stableSet = new Set(stable_panel.apartments
    .filter((a) => a.coverage >= stable_panel.min_coverage)
    .map((a) => a.apartment));

  const monthlyOptions = {
    ...baseChartOptions,
    scales: {
      x: baseChartOptions.scales.x,
      y: {
        ...baseChartOptions.scales.y,
        title: { display: true, text: "Target hours", color: "#4b5563",
          font: { family: SANS, size: 10 } },
      },
      y1: {
        position: "right",
        grid: { drawOnChartArea: false },
        ticks: { color: "#4b5563", font: { family: MONO, size: 10 } },
        title: { display: true, text: "Apartments", color: "#4b5563",
          font: { family: SANS, size: 10 } },
      },
    },
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      <div>
        <h2 style={{ fontSize: 17, fontWeight: 700, color: "#111827",
          fontFamily: MONO, margin: 0, letterSpacing: -0.3 }}>Data Quality</h2>
        <p style={{ fontSize: 12, color: "#4b5563", marginTop: 4, fontFamily: SANS }}>
          Coverage, gaps and inactivity for the {panel.apartments}-apartment meter panel.
          Generated {data.generated_at?.slice(0, 10)}.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
        gap: 10 }}>
        <Stat label="Files read" value={`${source.files_read} / ${source.files_found}`}
          sub={`${source.files_failed} failed`} tone={source.files_failed ? "warn" : "good"} />
        <Stat label="Raw rows" value={num(source.rows_total)}
          sub={`${source.first_ist?.slice(0, 10)} → ${source.last_ist?.slice(0, 10)}`} />
        <Stat label="Hourly panel" value={`${num(panel.hours)} h`}
          sub={`${panel.apartments} apartments · ${num(panel.hours_zero_reporting)} h empty`} />
        <Stat label="Stable apartments" value={`${stable_panel.count} / ${panel.apartments}`}
          sub={`≥${Math.round(stable_panel.min_coverage * 100)}% coverage`} tone="good" />
        <Stat label="Target valid" value={`${num(target.valid_hours)} h`}
          sub={`${num(target.imputed_hours)} h imputed · ${num(target.excluded_hours)} h excluded`} />
        <Stat label="Flagged inactive" value={num(inactive.flagged.length)}
          sub={`${inactive.month} · <${inactive.threshold_wh} Wh/h`}
          tone={inactive.flagged.length ? "warn" : "good"} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
        gap: 14 }}>

        <div style={CARD}>
          <div style={CARD_TITLE}>Monthly availability</div>
          <div style={{ height: 280 }}>
            <Bar
              data={{
                labels: monthly_coverage.map((m) => m.month),
                datasets: [
                  {
                    label: "Target hours",
                    data: monthly_coverage.map((m) => m.target_hours),
                    backgroundColor: "rgba(37,99,235,0.55)",
                    borderColor: "#2563eb",
                    borderWidth: 1,
                    borderRadius: 3,
                  },
                  {
                    label: "Apartments reporting",
                    type: "line",
                    data: monthly_coverage.map((m) => m.apartments_reporting),
                    yAxisID: "y1",
                    borderColor: "#1a1a2e",
                    backgroundColor: "rgba(26,26,46,0.08)",
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.3,
                  },
                ],
              }}
              options={monthlyOptions}
            />
          </div>
        </div>

        <div style={CARD}>
          <div style={CARD_TITLE}>Gaps and exclusions</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8,
            marginBottom: 14 }}>
            <Stat label="Imputed" value={`${num(target.imputed_hours)} h`}
              sub={`runs ≤ ${target.max_gap_hours} h`} tone="good" />
            <Stat label="Excluded" value={`${num(target.excluded_hours)} h`}
              sub={`${target.excluded_runs} longer/edge runs`} tone="warn" />
            <Stat label="Valid target" value={`${num(target.valid_hours)} h`}
              sub={`mean ${target.mean_wh?.toFixed(0)} Wh/h`} />
            <Stat label="Longest gap" value={`${num(target.longest_gap_hours)} h`}
              sub={`${target.gap_runs} runs total`} />
          </div>
          <div style={{ maxHeight: 150, overflow: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Start (IST)", "End (IST)", "Hours", "Edge"].map((h) => (
                    <th key={h} style={{ ...TH, fontSize: 9 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {target.longest_runs.map((run) => (
                  <tr key={`${run.start}-${run.end}`}>
                    <td style={TD}>{run.start.slice(0, 16).replace("T", " ")}</td>
                    <td style={TD}>{run.end.slice(0, 16).replace("T", " ")}</td>
                    <td style={TD}>{num(run.hours)}</td>
                    <td style={TD}>{run.edge ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
        gap: 14 }}>

        <div style={CARD}>
          <div style={CARD_TITLE}>
            Stable apartment coverage · {stable_panel.window_start} → {stable_panel.window_end.slice(0, 10)}
          </div>
          <div style={{ maxHeight: 320, overflow: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Apartment", "Coverage", "Status"].map((h) => (
                    <th key={h} style={TH}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stable_panel.apartments.map(({ apartment, coverage }) => {
                  const included = stableSet.has(apartment);
                  return (
                    <tr key={apartment}>
                      <td style={TD}>{apartment}</td>
                      <td style={{ ...TD, minWidth: 120 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <div style={{ flex: 1, height: 5, background: "#f3f4f6", borderRadius: 3 }}>
                            <div style={{ width: `${coverage * 100}%`, height: "100%",
                              background: included ? "#15803d" : "#b91c1c", borderRadius: 3 }} />
                          </div>
                          <span>{(coverage * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td style={{ ...TD, color: included ? "#15803d" : "#b91c1c",
                        fontWeight: 700 }}>
                        {included ? "Included" : "Excluded"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div style={CARD}>
          <div style={CARD_TITLE}>Suspect inactivity · {inactive.month}</div>
          <p style={{ fontSize: 11, color: "#4b5563", fontFamily: SANS, marginTop: 0 }}>
            Mean below {inactive.threshold_wh} Wh/h with at least {inactive.min_hours} readings.
            {inactive.flagged_stable.length > 0 &&
              ` ${inactive.flagged_stable.length} of ${stable_panel.count} stable apartments are flagged.`}
          </p>
          <div style={{ maxHeight: 320, overflow: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Apartment", "Mean Wh/h", "Readings"].map((h) => (
                    <th key={h} style={TH}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {inactive.apartments.filter((a) => a.suspect_inactive).map((a) => (
                  <tr key={a.apartment}>
                    <td style={TD}>{a.apartment}</td>
                    <td style={{ ...TD, color: "#b91c1c" }}>{a.mean_wh?.toFixed(2)}</td>
                    <td style={TD}>{num(a.hours)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div style={CARD}>
        <div style={CARD_TITLE}>What this means</div>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, lineHeight: 1.75,
          color: "#374151", fontFamily: SANS }}>
          {notes.map((note) => <li key={note}>{note}</li>)}
        </ul>
      </div>
    </div>
  );
}
