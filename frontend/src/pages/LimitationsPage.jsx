import { useData } from "../hooks/useData";
import { api } from "../utils/api";
import DataError from "../components/DataError";

const SEVERITY = {
  high:   { border: "#dc2626", bg: "#fef2f2", color: "#b91c1c", label: "High" },
  medium: { border: "#d97706", bg: "#fffbeb", color: "#b45309", label: "Medium" },
  info:   { border: "#64748b", bg: "#f8fafc", color: "#475569", label: "Context" },
};

function pct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "N/A";
}

function int(value) {
  return Number.isFinite(value) ? value.toLocaleString("en-IN") : "N/A";
}

function LimitationItem({ item }) {
  const tone = SEVERITY[item.severity] || SEVERITY.info;
  return (
    <div style={{
      border: "1px solid #e5e7eb", borderLeft: `3px solid ${tone.border}`,
      borderRadius: 8, padding: "12px 14px", background: tone.bg,
    }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 8 }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: tone.color,
          fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase",
          letterSpacing: 0.5 }}>{item.label}</div>
        <div style={{ fontSize: 8, color: tone.color, fontFamily: "'IBM Plex Mono', monospace",
          textTransform: "uppercase", letterSpacing: 0.4, opacity: 0.8 }}>{tone.label}</div>
      </div>
      <div style={{ fontSize: 17, fontWeight: 700, color: "#111827", margin: "3px 0 4px",
        fontFamily: "'IBM Plex Mono', monospace" }}>{item.value}</div>
      <div style={{ fontSize: 11, color: "#4b5563", lineHeight: 1.5 }}>{item.detail}</div>
    </div>
  );
}

function GroupCard({ group }) {
  return (
    <section style={{ background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 12, padding: "18px 22px" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: "#111827",
        fontFamily: "'IBM Plex Mono', monospace" }}>{group.title}</div>
      <div style={{ fontSize: 11, color: "#4b5563", margin: "3px 0 14px" }}>{group.intro}</div>
      <div style={{ display: "grid", gap: 10,
        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
        {group.items.map((item) => <LimitationItem key={item.label} item={item} />)}
      </div>
    </section>
  );
}

function FoldTable({ folds, intervalRule }) {
  return (
    <div style={{ background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 12, padding: "18px 22px" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: "#111827",
        fontFamily: "'IBM Plex Mono', monospace" }}>Coverage by fold</div>
      <div style={{ fontSize: 11, color: "#4b5563", margin: "3px 0 12px" }}>
        Evaluable hours, training-window labels, SARIMA coverage, and measured P10–P90 coverage
        before and after rolling conformal calibration. "Cal. days" excludes the 7-day warm-up.
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11,
          fontFamily: "'IBM Plex Mono', monospace" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
              {["Fold", "Test window", "Evaluable h", "Days", "Train labels", "SARIMA h", "Raw P10–P90", "After CQR", "Cal. days"].map((head) => (
                <th key={head} scope="col" style={{ textAlign: "left", padding: "7px 10px",
                  fontSize: 9, color: "#4b5563", textTransform: "uppercase",
                  letterSpacing: 0.4, whiteSpace: "nowrap" }}>{head}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {folds.map((fold) => (
              <tr key={fold.id} style={{ borderBottom: "1px solid #f1f5f9" }}>
                <td style={{ padding: "7px 10px", fontWeight: 700, whiteSpace: "nowrap" }}>{fold.label}</td>
                <td style={{ padding: "7px 10px", color: "#4b5563", whiteSpace: "nowrap" }}>
                  {fold.test_start.slice(0, 10)} → {fold.test_end.slice(0, 10)}
                </td>
                <td style={{ padding: "7px 10px" }}>{int(fold.eval_hours)}</td>
                <td style={{ padding: "7px 10px" }}>{int(fold.eval_days)}</td>
                <td style={{ padding: "7px 10px" }}>{int(fold.n_train_hours)}</td>
                <td style={{ padding: "7px 10px" }}>{int(fold.sarima_hours)}</td>
                <td style={{ padding: "7px 10px",
                  color: fold.interval_coverage < 0.6 ? "#b91c1c" : "#111827" }}>
                  {pct(fold.interval_coverage)}
                </td>
                <td style={{ padding: "7px 10px", fontWeight: 700,
                  color: fold.interval_after_coverage < 0.6 ? "#b91c1c" : "#111827" }}>
                  {pct(fold.interval_after_coverage)}
                </td>
                <td style={{ padding: "7px 10px" }}>{int(fold.conformal_calibrated_days)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {intervalRule ? (
        <div style={{ fontSize: 9, color: "#6b7280", marginTop: 10, lineHeight: 1.5 }}>
          {intervalRule}
        </div>
      ) : null}
    </div>
  );
}

function Caveats({ caveats }) {
  return (
    <section style={{ background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 12, padding: "18px 22px" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: "#111827",
        fontFamily: "'IBM Plex Mono', monospace" }}>What this dashboard does not establish</div>
      <div style={{ fontSize: 11, color: "#4b5563", margin: "3px 0 14px" }}>
        Scope statements that apply to every model on the site.
      </div>
      <div style={{ display: "grid", gap: 10,
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
        {caveats.map((caveat) => (
          <div key={caveat.title} style={{ border: "1px solid #fecaca", background: "#fff7f7",
            borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#b91c1c",
              fontFamily: "'IBM Plex Mono', monospace" }}>{caveat.title}</div>
            <div style={{ fontSize: 11, color: "#4b5563", lineHeight: 1.5, marginTop: 4 }}>
              {caveat.detail}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function LimitationsPage() {
  const { data, loading, error } = useData(api.limitations);

  if (error) return <DataError error={error} label="limitations" />;
  if (loading) {
    return <div style={{ color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace" }}>
      Loading limitations...
    </div>;
  }
  if (!data) return null;

  const generated = data.generated_at
    ? new Date(data.generated_at).toLocaleString("en-IN", {
        timeZone: "Asia/Kolkata", dateStyle: "medium", timeStyle: "short",
      })
    : "unknown";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12,
        padding: "20px 24px", borderTop: "3px solid #dc2626" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ width: 24, height: 24, borderRadius: 6, background: "#fef2f2",
            color: "#b91c1c", display: "inline-flex", alignItems: "center",
            justifyContent: "center", fontSize: 14, fontWeight: 700,
            fontFamily: "'IBM Plex Mono', monospace" }}>!</span>
          <h1 style={{ margin: 0, fontSize: 20, color: "#111827",
            fontFamily: "'IBM Plex Mono', monospace" }}>Limitations and scope</h1>
          <span style={{ fontSize: 9, fontWeight: 700, color: "#b91c1c",
            background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 10,
            padding: "2px 8px", fontFamily: "'IBM Plex Mono', monospace",
            letterSpacing: 0.4, textTransform: "uppercase" }}>Read before quoting</span>
        </div>
        <p style={{ margin: "10px 0 0", fontSize: 12, color: "#4b5563",
          lineHeight: 1.6, maxWidth: 900 }}>{data.headline}</p>
      </div>

      {data.groups.map((group) => <GroupCard key={group.id} group={group} />)}
      <FoldTable folds={data.folds} intervalRule={data.interval_rule} />
      <Caveats caveats={data.caveats} />

      <div style={{ fontSize: 9, color: "#6b7280", fontFamily: "'IBM Plex Mono', monospace",
        textAlign: "right" }}>
        Sources: {data.sources.map((source) => source.file).join(" · ")} · artifacts generated {generated} IST
      </div>
    </div>
  );
}
