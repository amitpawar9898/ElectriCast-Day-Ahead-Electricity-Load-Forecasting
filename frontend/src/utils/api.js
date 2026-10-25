const BASE = import.meta.env.VITE_API_URL || "";

async function get(path, signal) {
  const res = await fetch(`${BASE}${path}`, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || body?.error || `API error ${res.status}: ${path}`);
  }
  return res.json();
}

export const api = {
  metrics:            (signal) => get("/api/metrics", signal),
  backtestSummary:    (signal) => get("/api/backtest/summary", signal),
  backtestDays:       (signal) => get("/api/backtest/days", signal),
  backtestDay:        (date, signal) => get(`/api/backtest/day?date=${encodeURIComponent(date)}`, signal),
  backtestErrors:     (signal) => get("/api/backtest/errors", signal),
  explainImportance:  (signal) => get("/api/explain/importance", signal),
  explainResiduals:   (fold, model, signal) => get(`/api/explain/residuals?fold=${encodeURIComponent(fold)}&model=${encodeURIComponent(model)}`, signal),
  dataQuality:        (signal) => get("/api/data_quality", signal),
  limitations:        (signal) => get("/api/limitations", signal),
  statistics:         (signal) => get("/api/eda/statistics", signal),
  edaHourly:          (signal) => get("/api/eda/hourly", signal),
  edaDaily:           (signal) => get("/api/eda/daily", signal),
  edaMonthly:         (signal) => get("/api/eda/monthly", signal),
  edaYearly:          (signal) => get("/api/eda/yearly", signal),
  edaHeatmap:         (signal) => get("/api/eda/heatmap", signal),
  edaWeekdayWeekend:  (signal) => get("/api/eda/weekday_vs_weekend", signal),
  predMonthly:        (signal) => get("/api/predictions/monthly", signal),
  predDaily:          (month = "", signal) => get(`/api/predictions/daily${month ? `?month=${month}` : ""}`, signal),
  predHourly:         (m, d, signal) => get(`/api/predictions/hourly?month=${m}&day=${d}`, signal),
  predWeeklyPattern:  (signal) => get("/api/predictions/weekly_pattern", signal),
  predMonthlyErrors:  (signal) => get("/api/predictions/monthly_errors", signal),

  chatStatus:         (signal) => get("/api/chat/status", signal),

  chat: async (question) => {
    const res = await fetch(`${BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      const error = new Error(
        [body?.error, body?.detail].filter(Boolean).join(": ") ||
        `API error ${res.status}: /api/chat`
      );
      error.status = res.status;
      error.title = body?.error || "Assistant error";
      error.detail = body?.detail || null;
      throw error;
    }
    return res.json();
  },

  uploadOptions:     (signal) => get("/api/upload/options", signal),

  uploadPreview: async (blob, filename) => {
    const form = new FormData();
    form.append("file", blob, filename);
    const res = await fetch(`${BASE}/api/upload/preview`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(err.detail || err.error || `Preview failed: ${res.status}`);
    }
    return res.json();
  },

  uploadValidate: (file, mapping, onProgress) => new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    Object.entries(mapping).forEach(([key, value]) => form.append(key, value ?? ""));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/api/upload/validate`);
    xhr.timeout = 10 * 60 * 1000;
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(100);
        resolve(body);
      } else {
        const error = new Error(
          body?.detail || body?.error || `Upload failed: ${xhr.status}`
        );
        error.validation = body?.validation || null;
        reject(error);
      }
    };
    xhr.onerror = () => reject(new Error("Network error while uploading the CSV."));
    xhr.ontimeout = () => reject(new Error("Upload or evaluation timed out after 10 minutes."));
    xhr.send(form);
  }),
};
