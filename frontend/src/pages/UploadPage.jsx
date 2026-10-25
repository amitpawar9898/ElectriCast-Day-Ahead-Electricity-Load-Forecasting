import { useState, useRef } from "react";
import {
  Chart, CategoryScale, LinearScale, PointElement, LineElement,
  Filler, Tooltip, Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { useData } from "../hooks/useData";
import { api } from "../utils/api";
import { baseChartOptions } from "../utils/chartConfig";

Chart.register(CategoryScale, LinearScale, PointElement, LineElement,
  Filler, Tooltip, Legend);

const CARD = {
  background: "#fff", borderRadius: 12,
  border: "1px solid #e5e7eb", padding: 24, marginBottom: 20,
};
const MONO = { fontFamily: "'IBM Plex Mono', monospace" };
const SANS = { fontFamily: "'IBM Plex Sans', sans-serif" };

const fmtInt = (value) => Number.isFinite(value) ? value.toLocaleString("en-IN") : "N/A";
const fmtNum = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : "N/A";
const fmtPct = (value, digits = 1) => Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "N/A";
const fmtP = (value) => {
  if (!Number.isFinite(value)) return "—";
  return value < 0.001 ? value.toExponential(1) : value.toFixed(3);
};

function Card({ title, subtitle, children, style }) {
  return (
    <div style={{ ...CARD, ...style }}>
      {(title || subtitle) && (
        <div style={{ marginBottom: 14 }}>
          {title && <div style={{ fontSize: 13, fontWeight: 700, color: "#111827", ...MONO }}>{title}</div>}
          {subtitle && <div style={{ fontSize: 11, color: "#4b5563", marginTop: 3 }}>{subtitle}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

function Fact({ label, value, hint, tone = "#111827" }) {
  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: "10px 12px", background: "#f9fafb" }}>
      <div style={{ fontSize: 9, color: "#4b5563", textTransform: "uppercase", letterSpacing: 0.4 }}>
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 700, ...MONO, color: tone, marginTop: 2 }}>{value}</div>
      {hint && <div style={{ fontSize: 9, color: "#6b7280", marginTop: 2 }}>{hint}</div>}
    </div>
  );
}

function Alert({ tone = "red", title, children }) {
  const palette = tone === "red"
    ? { bg: "#fef2f2", border: "#fecaca", color: "#b91c1c" }
    : tone === "amber"
      ? { bg: "#fffbeb", border: "#fde68a", color: "#b45309" }
      : { bg: "#eff6ff", border: "#bfdbfe", color: "#1d4ed8" };
  return (
    <div style={{ background: palette.bg, border: `1px solid ${palette.border}`,
      color: palette.color, borderRadius: 8, padding: "12px 14px", fontSize: 12 }}>
      {title && <div style={{ fontWeight: 700, ...MONO, marginBottom: 4 }}>{title}</div>}
      <div style={{ lineHeight: 1.55 }}>{children}</div>
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 9, fontWeight: 700, color: "#374151",
        textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</span>
      {children}
      {hint && <span style={{ fontSize: 9, color: "#6b7280", ...SANS }}>{hint}</span>}
    </label>
  );
}

const selectStyle = {
  padding: "8px 10px", borderRadius: 7, border: "1px solid #d1d5db",
  background: "#fff", color: "#111827", fontSize: 12, ...MONO, width: "100%",
};

function makeLine(label, data, color, dash = []) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: `${color}10`,
    borderWidth: 2,
    borderDash: dash,
    tension: 0.3,
    pointRadius: 0,
    pointHoverRadius: 3,
    fill: false,
  };
}

export default function UploadPage() {
  const { data: optionData } = useData(api.uploadOptions);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mapping, setMapping] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [errorReport, setErrorReport] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);

  const options = preview?.options || optionData;
  const maxMb = options?.max_upload_mb || 50;

  function reset() {
    setFile(null);
    setPreview(null);
    setMapping(null);
    setResult(null);
    setError(null);
    setErrorReport(null);
    setPreviewing(false);
    setRunning(false);
    setProgress(0);
    if (fileRef.current) fileRef.current.value = "";
  }

  async function handleFile(nextFile) {
    if (!nextFile || previewing || running) return;
    if (!nextFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please upload a .csv file.");
      return;
    }
    if (nextFile.size > maxMb * 1024 * 1024) {
      setError(`File too large: ${(nextFile.size / (1024 * 1024)).toFixed(1)} MB (max ${maxMb} MB).`);
      return;
    }

    setFile(nextFile);
    setPreview(null);
    setMapping(null);
    setResult(null);
    setError(null);
    setErrorReport(null);
    setPreviewing(true);
    try {
      const slice = nextFile.slice(0, options?.preview_bytes || 262144);
      const data = await api.uploadPreview(slice, nextFile.name);
      const suggested = data.suggested || {};
      setPreview(data);
      setMapping({
        timestamp_col: suggested.timestamp_col || "",
        value_col: suggested.value_col || "",
        series_col: suggested.series_col || "",
        unit: suggested.unit || "Wh",
        timezone: suggested.timezone || "UTC",
        timestamp_mode: suggested.timestamp_mode || "auto",
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setPreviewing(false);
    }
  }

  async function runValidation() {
    if (!file || !mapping || running) return;
    setRunning(true);
    setProgress(0);
    setError(null);
    setErrorReport(null);
    try {
      const data = await api.uploadValidate(file, mapping, (pct) => setProgress(pct));
      if (!data?.validation || !data?.evaluation) {
        throw new Error("The server returned an incomplete validation result.");
      }
      setResult(data);
    } catch (e) {
      setError(e.message);
      setErrorReport(e.validation || null);
    } finally {
      setRunning(false);
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragOver(false);
    handleFile(event.dataTransfer?.files?.[0]);
  }

  function downloadJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `validation-${(result.validation?.filename || "upload").replace(/\.csv$/i, "")}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ── Results ─────────────────────────────────────────────────────
  if (result) {
    const validation = result.validation;
    const evaluation = result.evaluation;
    const hourly = validation.hourly;
    const split = evaluation.split || {};
    const best = evaluation.best_baseline || {};
    const bestModel = best.name ? evaluation.models?.[best.name] : null;
    const daily = evaluation.daily;
    const modelOrder = evaluation.model_order || Object.keys(evaluation.models || {});

    const chartData = daily ? {
      labels: daily.labels,
      datasets: [
        makeLine("Actual", daily.actual, "#1a1a2e"),
        makeLine("Seasonal naive", daily.models.seasonal_naive_24, "#dc2626", [4, 4]),
        makeLine("Zero-shot XGBoost", daily.models.zero_shot_xgboost, "#d97706", [2, 3]),
        makeLine("Fit-on-60% XGBoost", daily.models.fit_60_xgboost, "#16a34a"),
      ],
    } : null;

    return (
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start",
          gap: 16, flexWrap: "wrap", marginBottom: 20 }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#111827", ...MONO, margin: 0 }}>
              Upload validation report
            </h2>
            <p style={{ fontSize: 12, color: "#4b5563", marginTop: 4 }}>
              {validation.filename} · {validation.timestamp.first_ist?.slice(0, 10)} →{" "}
              {validation.timestamp.last_ist?.slice(0, 10)} IST · {fmtInt(validation.rows_raw)} rows
            </p>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={downloadJson} style={{
              padding: "8px 16px", borderRadius: 8, border: "1px solid #2563eb",
              background: "#eff6ff", color: "#2563eb", cursor: "pointer",
              fontSize: 11, fontWeight: 700, ...MONO,
            }}>Download JSON</button>
            <button onClick={reset} style={{
              padding: "8px 16px", borderRadius: 8, border: "1px solid #d1d5db",
              background: "#fff", color: "#374151", cursor: "pointer",
              fontSize: 11, fontWeight: 600, ...MONO,
            }}>Upload another</button>
          </div>
        </div>

        <div style={{ display: "grid", gap: 10, marginBottom: 20,
          gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))" }}>
          <Fact label="Rows used" value={fmtInt(validation.rows_used)} hint={`of ${fmtInt(validation.rows_raw)} raw`} />
          <Fact label="Series" value={fmtInt(validation.series_count)} hint="per-hour reporting counted" />
          <Fact label="Valid target hours" value={fmtInt(hourly.valid_hours)} hint={`${fmtPct(hourly.coverage)} of span`} />
          <Fact label="Interpolated" value={`${fmtInt(hourly.imputed_hours)} h`} hint={`gaps ≤ ${hourly.max_gap_hours ?? 6} h`} />
          <Fact label="Excluded gaps" value={`${fmtInt(hourly.excluded_hours)} h`} hint={`${fmtInt(hourly.gap_runs)} runs`} />
          <Fact label="Target mean" value={`${fmtInt(hourly.target_mean_wh)} Wh`} hint="per series-hour" />
          <Fact label="Evaluated hours" value={fmtInt(split.eval_hours || 0)} hint={`${fmtInt(split.test_days || 0)} test days`} />
        </div>

        <Card title="Parsing and validation" subtitle="Rows are converted to IST, aggregated to hourly Wh, and checked before evaluation.">
          <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", marginBottom: 14 }}>
            <Fact label="Timestamp" value={validation.mapping.timestamp_col}
              hint={`${validation.timestamp.mode} · ${validation.timestamp.source_note}`} />
            <Fact label="Value" value={validation.mapping.value_col}
              hint={`${validation.unit.label} · ${validation.unit.aggregation} within each hour`} />
            <Fact label="Series column" value={validation.mapping.series_col || "None"}
              hint={validation.mapping.series_col ? "target is the per-hour mean" : "single series"} />
            <Fact label="Dropped rows" value={fmtInt(validation.rows_raw - validation.rows_used)}
              hint={`${fmtInt(validation.rows_timestamp_invalid)} bad time · ${fmtInt(validation.rows_value_invalid)} bad value · ${fmtInt(validation.rows_negative)} negative`} />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {validation.warnings.map((warning, index) => (
              <div key={index} style={{ fontSize: 11, color: "#4b5563", lineHeight: 1.55,
                paddingLeft: 12, borderLeft: "2px solid #e5e7eb" }}>{warning}</div>
            ))}
          </div>
        </Card>

        <Card title="Day-ahead evaluation" subtitle="Same protocol as the campus backtest: IST origins, >=24 h features, per-model available hours.">
          {evaluation.status !== "ok" ? (
            <Alert tone="amber" title="Evaluation not available">
              {evaluation.reason || "The uploaded series is too short or too gappy for a day-ahead test."}
            </Alert>
          ) : (
            <>
              <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", marginBottom: 16 }}>
                <Fact label="Train window" value={`${fmtInt(split.train_hours)} h`}
                  hint={`${split.train_start?.slice(0, 10)} → ${split.train_end?.slice(0, 10)}`} />
                <Fact label="Test window" value={`${fmtInt(split.test_hours)} h`}
                  hint={`${split.test_start?.slice(0, 10)} → ${split.test_end?.slice(0, 10)}`} />
                <Fact label="Evaluable hours" value={fmtInt(split.eval_hours)}
                  hint="actual + seasonal naive available" />
                <Fact label="Best baseline" value={bestModel ? best.label : "N/A"}
                  hint={bestModel ? `${fmtNum(bestModel.MAE)} Wh MAE · ${fmtInt(best.selection_hours)} h` : "no baseline scored"} />
              </div>

              <div style={{ overflowX: "auto", marginBottom: 18 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, ...MONO }}>
                  <thead>
                    <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                      {["Model", "MAE (Wh)", "RMSE (Wh)", "MASE", "WAPE", "R²", "Bias (Wh)", "n", "vs best baseline"].map((head) => (
                        <th key={head} scope="col" style={{ textAlign: head === "Model" ? "left" : "right",
                          padding: "7px 9px", fontSize: 9, color: "#4b5563",
                          textTransform: "uppercase", letterSpacing: 0.3, whiteSpace: "nowrap" }}>{head}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {modelOrder.map((key) => {
                      const entry = evaluation.models[key];
                      if (!entry) return null;
                      const isFit = key === "fit_60_xgboost";
                      const isBestBaseline = key === best.name;
                      const sig = evaluation.significance?.[key];
                      const fit = entry.fit;
                      return (
                        <tr key={key} style={{
                          borderBottom: "1px solid #f1f5f9",
                          background: isFit ? "#f0fdf4" : isBestBaseline ? "#eff6ff" : "transparent",
                        }}>
                          <td style={{ padding: "7px 9px", whiteSpace: "nowrap" }}>
                            <div style={{ fontWeight: 700, color: isFit ? "#15803d" : "#111827" }}>{entry.label}</div>
                            <div style={{ fontSize: 8, color: "#6b7280", textTransform: "uppercase", letterSpacing: 0.3 }}>
                              {entry.kind}
                              {fit?.status === "ok" && key === "zero_shot_xgboost" && ` · scale ×${fit.scale_factor_to_campus}`}
                              {fit?.status === "ok" && isFit && ` · ${fmtInt(fit.n_train)} train h`}
                            </div>
                          </td>
                          {entry.status !== "ok" ? (
                            <td colSpan={8} style={{ padding: "7px 9px", textAlign: "left", color: "#b45309" }}>
                              {entry.fit?.reason || entry.status}
                            </td>
                          ) : (
                            <>
                              <td style={{ padding: "7px 9px", textAlign: "right" }}>{fmtNum(entry.MAE)}</td>
                              <td style={{ padding: "7px 9px", textAlign: "right" }}>{fmtNum(entry.RMSE)}</td>
                              <td style={{ padding: "7px 9px", textAlign: "right" }}>{fmtNum(entry.MASE, 3)}</td>
                              <td style={{ padding: "7px 9px", textAlign: "right" }}>{Number.isFinite(entry.WAPE) ? `${fmtNum(entry.WAPE)}%` : "N/A"}</td>
                              <td style={{ padding: "7px 9px", textAlign: "right",
                                color: Number.isFinite(entry.R2) && entry.R2 < 0 ? "#b91c1c" : "#111827" }}>
                                {fmtNum(entry.R2, 4)}
                              </td>
                              <td style={{ padding: "7px 9px", textAlign: "right" }}>{fmtNum(entry.bias)}</td>
                              <td style={{ padding: "7px 9px", textAlign: "right" }}>{fmtInt(entry.n)}</td>
                              <td style={{ padding: "7px 9px", textAlign: "right", whiteSpace: "nowrap" }}>
                                {sig ? (
                                  <span title={`Wilcoxon signed-rank on paired absolute errors; ${sig.n_pairs} pairs`}>
                                    {sig.model_better === true ? "better" : sig.model_better === false ? "worse" : "tie"}
                                    {" · p="}{fmtP(sig.p_value)}
                                  </span>
                                ) : "—"}
                              </td>
                            </>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {daily && (
                <div style={{ height: 280, marginBottom: 8 }}>
                  <Line data={chartData} options={{
                    ...baseChartOptions,
                    scales: {
                      ...baseChartOptions.scales,
                      x: { ...baseChartOptions.scales.x, ticks: { ...baseChartOptions.scales.x.ticks, maxTicksLimit: 8 } },
                      y: { ...baseChartOptions.scales.y, title: { display: true, text: "Wh (daily mean)", color: "#6b7280", font: { family: "'IBM Plex Mono', monospace", size: 10 } } },
                    },
                  }} />
                </div>
              )}
              <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 8 }}>
                {evaluation.notes.map((note, index) => (
                  <div key={index} style={{ fontSize: 10, color: "#4b5563", lineHeight: 1.5,
                    paddingLeft: 12, borderLeft: "2px solid #e5e7eb" }}>{note}</div>
                ))}
              </div>
            </>
          )}
        </Card>

        <Card style={{ background: "#fffbeb", border: "1px solid #fde68a" }}>
          <div style={{ fontSize: 11, color: "#92400e", lineHeight: 1.6 }}>
            These metrics describe this uploaded file only. They do not revalidate the campus backtest,
            and a zero-shot result depends on how closely the upload resembles the campus load and calendar.
          </div>
        </Card>
      </div>
    );
  }

  // ── Upload + mapping ────────────────────────────────────────────
  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: "#111827", ...MONO, margin: 0 }}>
          Upload &amp; validate
        </h2>
        <p style={{ fontSize: 12, color: "#4b5563", marginTop: 6, lineHeight: 1.6, maxWidth: 900 }}>
          Map the timestamp and load columns, declare the unit and timestamp timezone, and run the
          same day-ahead protocol used in the backtest. Timestamps are converted to IST; energy is
          aggregated to hourly Wh before any metric is computed.
        </p>
      </div>

      <div
        onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !previewing && !running && fileRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? "#2563eb" : "#d1d5db"}`,
          borderRadius: 14, padding: preview?.columns ? "24px 28px" : "48px 32px",
          textAlign: "center", cursor: previewing || running ? "default" : "pointer",
          background: dragOver ? "#eff6ff" : "#fafbfc", transition: "all 0.2s",
        }}
      >
        <div style={{ fontSize: preview?.columns ? 20 : 32, marginBottom: 8 }}>⬆</div>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#374151" }}>
          {file ? file.name : `Drop a .csv file here (max ${maxMb} MB)`}
        </div>
        <div style={{ fontSize: 11, color: "#4b5563", marginTop: 5 }}>
          {file
            ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${previewing ? "reading columns…" : "click to choose another file"}`
            : "or click to browse · .CSV is also accepted"}
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.CSV,text/csv"
          onChange={(event) => handleFile(event.target.files?.[0])}
          style={{ display: "none" }}
        />
      </div>

      {error && (
        <div style={{ marginTop: 14 }}>
          <Alert title="Upload could not be validated">{error}</Alert>
          {errorReport?.warnings?.length > 0 && (
            <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
              {errorReport.warnings.map((warning, index) => (
                <div key={index} style={{ fontSize: 10, color: "#6b7280" }}>{warning}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {preview && mapping && (
        <Card title="Column mapping" subtitle="Suggested from the first rows of your file. Check every field before running."
          style={{ marginTop: 20 }}>
          <div style={{ display: "grid", gap: 14, gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))" }}>
            <Field label="Timestamp column" hint="Required. Converted to IST.">
              <select style={selectStyle} value={mapping.timestamp_col} disabled={running}
                onChange={(event) => setMapping({ ...mapping, timestamp_col: event.target.value })}>
                <option value="">Select a column</option>
                {preview.columns.map((column) => <option key={column} value={column}>{column}</option>)}
              </select>
            </Field>
            <Field label="Value column" hint="Required. Must be numeric.">
              <select style={selectStyle} value={mapping.value_col} disabled={running}
                onChange={(event) => setMapping({ ...mapping, value_col: event.target.value })}>
                <option value="">Select a column</option>
                {preview.columns.map((column) => <option key={column} value={column}>{column}</option>)}
              </select>
            </Field>
            <Field label="Series column (optional)" hint="Leave none for a single series.">
              <select style={selectStyle} value={mapping.series_col} disabled={running}
                onChange={(event) => setMapping({ ...mapping, series_col: event.target.value })}>
                <option value="">None (single series)</option>
                {preview.columns.map((column) => <option key={column} value={column}>{column}</option>)}
              </select>
            </Field>
            <Field label="Unit" hint={options?.units?.find((unit) => unit.key === mapping.unit)?.detail || ""}>
              <select style={selectStyle} value={mapping.unit} disabled={running}
                onChange={(event) => setMapping({ ...mapping, unit: event.target.value })}>
                {(options?.units || []).map((unit) => (
                  <option key={unit.key} value={unit.key}>{unit.key} — {unit.label}</option>
                ))}
              </select>
            </Field>
            <Field label="Naive timestamp timezone" hint="Ignored when the timestamp is an epoch or has an offset.">
              <select style={selectStyle} value={mapping.timezone} disabled={running}
                onChange={(event) => setMapping({ ...mapping, timezone: event.target.value })}>
                {(options?.timezones || []).map((zone) => <option key={zone} value={zone}>{zone}</option>)}
              </select>
            </Field>
            <Field label="Timestamp format" hint="Auto-detection uses the sample values.">
              <select style={selectStyle} value={mapping.timestamp_mode} disabled={running}
                onChange={(event) => setMapping({ ...mapping, timestamp_mode: event.target.value })}>
                {(options?.timestamp_modes || []).map((mode) => (
                  <option key={mode.key} value={mode.key}>{mode.label}</option>
                ))}
              </select>
            </Field>
          </div>

          <div style={{ marginTop: 16, overflowX: "auto", border: "1px solid #f1f5f9", borderRadius: 8 }}>
            <table style={{ borderCollapse: "collapse", fontSize: 10, ...MONO, whiteSpace: "nowrap" }}>
              <thead>
                <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
                  {preview.columns.map((column) => (
                    <th key={column} scope="col" style={{ textAlign: "left", padding: "6px 8px", color: "#4b5563" }}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.sample_rows.map((row, index) => (
                  <tr key={index} style={{ borderBottom: "1px solid #f8fafc" }}>
                    {preview.columns.map((column) => (
                      <td key={column} style={{ padding: "5px 8px", color: "#374151" }}>{String(row[column] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
            <button
              onClick={runValidation}
              disabled={running || !mapping.timestamp_col || !mapping.value_col}
              style={{
                padding: "9px 20px", borderRadius: 8, border: "none", cursor: running ? "wait" : "pointer",
                background: running ? "#93c5fd" : "#2563eb", color: "#fff",
                fontSize: 12, fontWeight: 700, ...MONO,
              }}
            >{running ? (progress >= 100 ? "Validating and evaluating…" : `Uploading ${progress}%`) : "Validate and run evaluation"}</button>
            <span style={{ fontSize: 10, color: "#6b7280" }}>
              {preview.rows_previewed} rows previewed · delimiter {preview.delimiter} · full file parsed once
            </span>
          </div>

          {running && (
            <div style={{ marginTop: 12 }}>
              <div style={{ height: 6, borderRadius: 3, background: "#e5e7eb", overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${Math.max(progress, 3)}%`,
                  background: progress >= 100 ? "#16a34a" : "#2563eb",
                  transition: "width 0.2s",
                }} />
              </div>
              <div style={{ fontSize: 10, color: "#4b5563", marginTop: 5 }}>
                {progress >= 100
                  ? "Upload complete. Running column validation, baselines, zero-shot, and fit-on-60% evaluation."
                  : "Uploading the CSV. Evaluation starts after the upload completes."}
              </div>
            </div>
          )}
        </Card>
      )}

      {!preview && !previewing && (
        <Card title="What happens to the upload" style={{ marginTop: 20, background: "#f8fafc" }}>
          <ol style={{ fontSize: 12, color: "#4b5563", margin: 0, paddingLeft: 20, lineHeight: 2 }}>
            <li>The header and first rows are read for column suggestions; the full file is sent only when you validate.</li>
            <li>Naive timestamps are localized with your chosen timezone and every timestamp is converted to IST.</li>
            <li>Energy readings are summed per hour; power readings are averaged per hour and converted to Wh.</li>
            <li>The first 60% of valid target hours trains the local model; the last 40% is the day-ahead test window.</li>
            <li>Baselines, zero-shot, and fit-on-60% metrics use the same evaluation mask and paired significance test.</li>
          </ol>
        </Card>
      )}
    </div>
  );
}
