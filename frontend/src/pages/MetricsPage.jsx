import { useData } from "../hooks/useData";
import { api } from "../utils/api";
import DataError from "../components/DataError";

const MODEL_COLORS = {
  persistence: "#6b7280",
  seasonal_naive_24: "#2563eb",
  same_hour_last_week: "#9333ea",
  mean_last_4_same_weekdays: "#0891b2",
  xgboost: "#16a34a",
  sarima: "#dc2626",
  xgboost_quantile: "#d97706",
};

function meanStd(mean, std, digits) {
  if (!Number.isFinite(mean)) return "N/A";
  if (!Number.isFinite(std)) return mean.toFixed(digits);
  return `${mean.toFixed(digits)} ± ${std.toFixed(digits)}`;
}

function fmtP(value) {
  if (!Number.isFinite(value)) return "N/A";
  return value < 0.001 ? value.toExponential(1) : value.toFixed(3);
}

function fmtSigned(value, digits = 1) {
  if (!Number.isFinite(value)) return "N/A";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function BeatsCell({ model }) {
  if (model.is_reference) {
    return <span style={{ color: "#2563eb", fontSize: 11 }}>Reference</span>;
  }
  if (model.beats_seasonal_naive === null) {
    return <span style={{ color: "#9ca3af" }}>N/A</span>;
  }
  const won = model.beats_seasonal_naive;
  return (
    <span style={{ color: won ? "#15803d" : "#b91c1c", fontWeight: 700 }}>
      {won ? "Yes" : "No"}
      <span style={{ fontWeight: 400, color: "#4b5563", marginLeft: 6, fontSize: 10 }}>
        ({model.folds_beaten}/{model.comparable_folds} folds)
      </span>
    </span>
  );
}

function TableShell({ title, subtitle, children }) {
  return (
    <div style={{ background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 12, padding: "18px 22px" }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#374151",
        fontFamily: "'IBM Plex Mono', monospace",
        textTransform: "uppercase", letterSpacing: 0.5 }}>{title}</div>
      {subtitle && (
        <div style={{ fontSize: 10, color: "#4b5563", marginTop: 4,
          fontFamily: "'IBM Plex Sans', sans-serif" }}>{subtitle}</div>
      )}
      <div style={{ marginTop: 14, overflowX: "auto" }}>{children}</div>
    </div>
  );
}

export default function MetricsPage() {
  const { data, loading, error } = useData(api.backtestSummary);

  if (error) return <DataError error={error} label="backtest results" />;

  if (loading || !data) {
    return <div style={{ color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace" }}>Loading metrics...</div>;
  }

  const models = data.models || [];
  const folds = data.folds || [];
  const foldOrder = data.fold_order || folds.map(f => f.id);
  const foldById = Object.fromEntries(folds.map(f => [f.id, f]));
  const scoringFolds = folds.filter(f => !f.stress_test);
  const stressFolds = folds.filter(f => f.stress_test);
  const labelById = Object.fromEntries(models.map(model => [model.key, model.label]));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      <div style={{ background: "#fff", border: "1px solid #e5e7eb",
        borderRadius: 12, padding: "18px 22px" }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "#111827",
          fontFamily: "'IBM Plex Mono', monospace" }}>
          {scoringFolds.length}-fold day-ahead backtest
          {stressFolds.length > 0 ? " + stress test" : ""}
        </div>
        {data.protocol && (
          <div style={{ display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "4px 18px", marginTop: 8 }}>
            {Object.entries(data.protocol).map(([key, value]) => (
              <div key={key} style={{ fontSize: 11, color: "#4b5563",
                fontFamily: "'IBM Plex Sans', sans-serif", lineHeight: 1.5 }}>
                <span style={{ color: "#111827", fontWeight: 600 }}>
                  {key.replace(/_/g, " ").replace(/^./, c => c.toUpperCase())}:
                </span>{" "}
                {String(value)}
              </div>
            ))}
          </div>
        )}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
          {folds.map(fold => (
            <span key={fold.id} style={{
              fontSize: 10, fontFamily: "'IBM Plex Mono', monospace",
              color: fold.stress_test ? "#92400e" : "#374151",
              background: fold.stress_test ? "#fffbeb" : "#f9fafb",
              border: `1px solid ${fold.stress_test ? "#fcd34d" : "#e5e7eb"}`,
              borderRadius: 20, padding: "3px 10px",
            }}>
              {fold.label} · {fold.eval_hours.toLocaleString()} h · {fold.eval_days} d
              {fold.stress_test ? " · excluded from means" : ""}
            </span>
          ))}
        </div>
      </div>

      {models.length === 0 && (
        <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12,
          padding: "18px 22px", fontSize: 12, color: "#4b5563",
          fontFamily: "'IBM Plex Mono', monospace" }}>
          No backtest results available.
        </div>
      )}

      {models.length > 0 && (
        <TableShell
          title="Fold mean ± std by model"
          subtitle={`Mean and sample standard deviation across the ${scoringFolds.length} scoring folds with results. ${stressFolds.length > 0 ? "The stress test is excluded. " : ""}“Beats seasonal naive” means ${data.beats_rule}.`}
        >
          <table style={{ borderCollapse: "collapse",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                {["Model", "MAE (Wh)", "RMSE (Wh)", "MASE", "Beats seasonal naive?", "Folds"].map(h => (
                  <th key={h} style={{
                    padding: "8px 10px",
                    width: h === "Model" ? "1%" : undefined,
                    textAlign: h === "Model" ? "left" : "right",
                    fontSize: 10, color: "#4b5563", fontWeight: 700,
                    textTransform: "uppercase", letterSpacing: 0.5,
                    whiteSpace: "nowrap",
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map(model => (
                <tr key={model.key} style={{
                  borderBottom: "1px solid #f3f4f6",
                  background: model.kind === "learned" ? "#f9fafb" : "transparent",
                }}>
                  <td style={{ padding: "10px 12px" }}>
                    <div title={model.description}
                      style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2,
                        background: MODEL_COLORS[model.key] || "#6b7280" }}/>
                      <span style={{ whiteSpace: "nowrap" }}>{model.label}</span>
                      <span style={{ fontSize: 9, color: "#6b7280",
                        fontFamily: "'IBM Plex Sans', sans-serif",
                        border: "1px solid #e5e7eb", borderRadius: 4,
                        padding: "1px 4px", textTransform: "uppercase" }}>
                        {model.kind}
                      </span>
                    </div>
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right", whiteSpace: "nowrap" }}>
                    {meanStd(model.mean.MAE, model.std.MAE, 1)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right", whiteSpace: "nowrap" }}>
                    {meanStd(model.mean.RMSE, model.std.RMSE, 1)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right", whiteSpace: "nowrap" }}>
                    {meanStd(model.mean.MASE, model.std.MASE, 3)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right", whiteSpace: "nowrap" }}>
                    <BeatsCell model={model} />
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right", color: "#4b5563" }}>
                    {model.n_folds}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableShell>
      )}

      {models.length > 0 && (
        <TableShell
          title="MAE by fold"
          subtitle={`Each cell shows MAE in Wh and the number of evaluable hours (n) in that fold.${stressFolds.length > 0 ? " The stress test is shown for context only and is not part of any fold mean." : ""}`}
        >
          <table style={{ borderCollapse: "collapse",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                <th style={{ padding: "8px 10px", textAlign: "left", width: "1%",
                  fontSize: 10, color: "#4b5563", fontWeight: 700,
                  textTransform: "uppercase", letterSpacing: 0.5,
                  whiteSpace: "nowrap" }}>Model</th>
                {foldOrder.map(id => (
                  <th key={id} style={{ padding: "8px 12px", textAlign: "right",
                    fontSize: 10, color: "#4b5563", fontWeight: 700,
                    textTransform: "uppercase", letterSpacing: 0.5,
                    whiteSpace: "nowrap" }}>
                    {foldById[id]?.label || id}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map(model => (
                <tr key={model.key} style={{
                  borderBottom: "1px solid #f3f4f6",
                  background: model.is_reference ? "#eff6ff" : "transparent",
                }}>
                  <td style={{ padding: "10px 12px" }}>{model.label}</td>
                  {foldOrder.map(id => {
                    const row = (model.per_fold || []).find(f => f.id === id);
                    const ok = row?.status === "ok" && Number.isFinite(row.MAE);
                    return (
                      <td key={id} style={{ padding: "10px 12px", textAlign: "right" }}>
                        {ok ? (
                          <>
                            <div>{row.MAE.toFixed(1)}</div>
                            <div style={{ fontSize: 9, color: "#6b7280" }}>n={row.n.toLocaleString()}</div>
                          </>
                        ) : (
                          <span style={{ color: "#9ca3af" }}>N/A</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </TableShell>
      )}

      {models.length > 0 && (
        <TableShell
          title="MAE on common hours"
          subtitle={`Every row is scored on exactly the same hours: the evaluation mask intersected with all seven models' forecasts. ${stressFolds.length > 0 ? "A fold with no common hours, such as the stress test where a lagged baseline has no forecasts, is shown as N/A. " : ""}Compare this table with “MAE by fold” above, where each model uses its own available hours.`}
        >
          <table style={{ borderCollapse: "collapse",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                <th style={{ padding: "8px 10px", textAlign: "left", width: "1%",
                  fontSize: 10, color: "#4b5563", fontWeight: 700,
                  textTransform: "uppercase", letterSpacing: 0.5,
                  whiteSpace: "nowrap" }}>Model</th>
                {foldOrder.map(id => (
                  <th key={id} style={{ padding: "8px 12px", textAlign: "right",
                    fontSize: 10, color: "#4b5563", fontWeight: 700,
                    textTransform: "uppercase", letterSpacing: 0.5,
                    whiteSpace: "nowrap" }}>
                    {foldById[id]?.label || id}
                    <div style={{ fontSize: 9, color: "#9ca3af", fontWeight: 400,
                      textTransform: "none", letterSpacing: 0 }}>
                      {foldById[id]?.common_hours?.n != null
                        ? `n=${foldById[id].common_hours.n.toLocaleString()}`
                        : "n=0"}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map(model => (
                <tr key={model.key} style={{
                  borderBottom: "1px solid #f3f4f6",
                  background: model.is_reference ? "#eff6ff" : "transparent",
                }}>
                  <td style={{ padding: "10px 12px" }}>{model.label}</td>
                  {foldOrder.map(id => {
                    const row = (model.per_fold || []).find(f => f.id === id);
                    const common = row?.common;
                    const ok = common?.status === "ok"
                      && Number.isFinite(common.MAE) && common.n > 0;
                    return (
                      <td key={id} style={{ padding: "10px 12px", textAlign: "right" }}>
                        {ok ? (
                          <>
                            <div>{common.MAE.toFixed(1)}</div>
                            <div style={{ fontSize: 9, color: "#6b7280" }}>n={common.n.toLocaleString()}</div>
                          </>
                        ) : (
                          <span style={{ color: "#9ca3af" }}>N/A</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </TableShell>
      )}

      {models.length > 0 && (
        <TableShell
          title="Day-level test vs best baseline (common hours)"
          subtitle="Each cell shows Δ = model MAE − best-baseline MAE over the common hours (negative is better), its 95% block-bootstrap CI (2,000 resamples of contiguous 7-day blocks, seed 42), and the paired Wilcoxon signed-rank p-value across days. Δ and the CI are hour-weighted; the Wilcoxon test weights each day equally. p-values are unadjusted. N/A means no test was possible, including the stress test with no common hours."
        >
          <table style={{ borderCollapse: "collapse",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                <th style={{ padding: "8px 10px", textAlign: "left", width: "1%",
                  fontSize: 10, color: "#4b5563", fontWeight: 700,
                  textTransform: "uppercase", letterSpacing: 0.5,
                  whiteSpace: "nowrap" }}>Model</th>
                {foldOrder.map(id => {
                  const best = foldById[id]?.common_hours?.best_baseline;
                  return (
                    <th key={id} style={{ padding: "8px 12px", textAlign: "right",
                      fontSize: 10, color: "#4b5563", fontWeight: 700,
                      textTransform: "uppercase", letterSpacing: 0.5,
                      whiteSpace: "nowrap" }}>
                      {foldById[id]?.label || id}
                      <div style={{ fontSize: 9, color: "#9ca3af", fontWeight: 400,
                        textTransform: "none", letterSpacing: 0 }}>
                        {best ? `best: ${labelById[best] || best}` : "best: N/A"}
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {models.map(model => (
                <tr key={model.key} style={{
                  borderBottom: "1px solid #f3f4f6",
                  background: model.is_reference ? "#eff6ff" : "transparent",
                }}>
                  <td style={{ padding: "10px 12px" }}>{model.label}</td>
                  {foldOrder.map(id => {
                    const row = (model.per_fold || []).find(f => f.id === id);
                    const sig = row?.daily_vs_best;
                    if (sig?.status === "reference") {
                      return (
                        <td key={id} style={{ padding: "10px 12px", textAlign: "right",
                          fontSize: 10, color: "#2563eb" }}>
                          Reference
                        </td>
                      );
                    }
                    if (sig?.status !== "ok") {
                      return (
                        <td key={id} style={{ padding: "10px 12px", textAlign: "right",
                          color: "#9ca3af" }}>
                          N/A
                        </td>
                      );
                    }
                    return (
                      <td key={id} style={{ padding: "10px 12px", textAlign: "right",
                        lineHeight: 1.5 }}>
                        <div style={{
                          color: sig.mae_diff_wh < 0 ? "#15803d" : "#b91c1c",
                          fontWeight: 700,
                        }}>
                          Δ {fmtSigned(sig.mae_diff_wh)} Wh
                        </div>
                        <div style={{ fontSize: 9, color: "#4b5563" }}>
                          [{fmtSigned(sig.ci_low_wh)}, {fmtSigned(sig.ci_high_wh)}]
                        </div>
                        <div style={{ fontSize: 9, color: "#6b7280" }}>
                          p={fmtP(sig.p_value)} · {sig.n_days} d
                          {sig.small_sample ? " · small" : ""}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </TableShell>
      )}

      <div style={{ fontSize: 10, color: "#4b5563",
        fontFamily: "'IBM Plex Sans', sans-serif", lineHeight: 1.6 }}>
        Fold means weight each scoring fold equally, not each hour, and exclude the
        2018 stress test. Models with partial coverage have fewer evaluable hours; the
        per-fold n shows the difference. The “beats seasonal naive” column is a
        descriptive majority rule, not a significance test. The common-hours table
        removes the coverage difference entirely, but it uses fewer hours in folds
        where a model is missing. The day-level table reports the paired Wilcoxon
        p-value on daily MAE and the block-bootstrap CI of the common-hours MAE
        difference; a CI that spans zero or a p-value above 0.05 is not evidence
        that the model is better.
      </div>
    </div>
  );
}
