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

const DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

const MODEL_COLORS = {
  actual:  "#111827",
  sarima:  "#dc2626",
  xgboost: "#16a34a",
};

export default function WeeklyPage() {
  const [selectedDay, setSelectedDay] = useState("Mon");
  const { data: weeklyData, error } = useData(api.predWeeklyPattern);
  const opts = baseChartOptions;

  const dayData = weeklyData?.[selectedDay];
  const hours = Array.from({length: 24}, (_, i) => `${i.toString().padStart(2,"0")}:00`);

  // Day-avg summary (one bar per day)
  const dayAvgs = weeklyData
    ? DAYS.map(d => ({
        day: d,
        actual: weeklyData[d] ? Math.round(weeklyData[d].actual.reduce((a,b)=>a+b,0)/24) : 0,
        xgboost: weeklyData[d] ? Math.round(weeklyData[d].xgboost.reduce((a,b)=>a+b,0)/24) : 0,
      }))
    : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <DataError error={error} label="weekly pattern data" />

      {/* Day selector */}
      <div style={{ background: "#fff", border: "1px solid #e5e7eb",
        borderRadius: 12, padding: "18px 22px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "#374151",
          fontFamily: "'IBM Plex Mono', monospace",
          textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 12 }}>
          Select Day of Week
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {DAYS.map(d => {
            const isWeekend = d === "Sat" || d === "Sun";
            const isActive = d === selectedDay;
            return (
              <button key={d} onClick={() => setSelectedDay(d)} style={{
                flex: 1, padding: "10px 4px", borderRadius: 8, cursor: "pointer",
                border: "2px solid",
                borderColor: isActive ? (isWeekend ? "#dc2626" : "#2563eb") : "#e5e7eb",
                background: isActive ? (isWeekend ? "#fef2f2" : "#eff6ff") : "#f9fafb",
                color: isActive ? (isWeekend ? "#b91c1c" : "#2563eb") : "#4b5563",
                fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, fontWeight: 700,
                textTransform: "uppercase", transition: "all 0.15s",
              }}>
                {d}
                {isWeekend && <div style={{ fontSize: 8, marginTop: 3, opacity: 0.7 }}>weekend</div>}
              </button>
            );
          })}
        </div>
      </div>

      {/* Hourly profile for selected day */}
      <div style={{ background: "#fff", border: "1px solid #e5e7eb",
        borderRadius: 12, padding: "18px 22px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "#374151",
          fontFamily: "'IBM Plex Mono', monospace",
          textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 14 }}>
          {selectedDay} — Average 24h Load Profile (Wh)
        </div>
        <div style={{ height: 280 }}>
          {dayData && (
            <Line
              data={{
                labels: hours,
                datasets: [
                  {
                    label: "Actual",
                    data: dayData.actual,
                    borderColor: MODEL_COLORS.actual,
                    backgroundColor: "rgba(17,24,39,0.07)",
                    fill: true, tension: 0.4, pointRadius: 2, borderWidth: 2.2,
                  },
                  {
                    label: "SARIMA",
                    data: dayData.sarima,
                    borderColor: MODEL_COLORS.sarima,
                    borderDash: [3, 2], tension: 0.4, pointRadius: 0, borderWidth: 1.8,
                  },
                  {
                    label: "XGBoost",
                    data: dayData.xgboost,
                    borderColor: MODEL_COLORS.xgboost,
                    borderDash: [8, 3], tension: 0.4, pointRadius: 0, borderWidth: 1.8,
                  },
                ]
              }}
              options={opts}
            />
          )}
        </div>
      </div>

      {/* All days overlay */}
      <div style={{ background: "#fff", border: "1px solid #e5e7eb",
        borderRadius: 12, padding: "18px 22px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "#374151",
          fontFamily: "'IBM Plex Mono', monospace",
          textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 14 }}>
          All Days Overlay — XGBoost Predictions (Wh)
        </div>
        <div style={{ height: 250 }}>
          {weeklyData && (
            <Line
              data={{
                labels: hours,
                datasets: DAYS.map((d, i) => {
                  const isWe = d === "Sat" || d === "Sun";
                  const isSelected = d === selectedDay;
                  const hue = [220, 200, 170, 140, 110, 0, 20][i];
                  return {
                    label: d,
                    data: weeklyData[d]?.xgboost || [],
                    borderColor: isSelected
                      ? (isWe ? "#dc2626" : "#2563eb")
                      : `hsla(${hue}, 60%, 50%, 0.45)`,
                    borderWidth: isSelected ? 2.5 : 1,
                    borderDash: isSelected ? [] : [3, 2],
                    tension: 0.4, pointRadius: 0,
                  };
                })
              }}
              options={opts}
            />
          )}
        </div>
      </div>

      {/* Daily average bar */}
      <div style={{ background: "#fff", border: "1px solid #e5e7eb",
        borderRadius: 12, padding: "18px 22px" }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "#374151",
          fontFamily: "'IBM Plex Mono', monospace",
          textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 14 }}>
          Average Daily Load by Weekday — Actual vs XGBoost (Wh)
        </div>
        <div style={{ height: 220 }}>
          {dayAvgs.length > 0 && (
            <Bar
              data={{
                labels: dayAvgs.map(d => d.day),
                datasets: [
                  {
                    label: "Actual",
                    data: dayAvgs.map(d => d.actual),
                    backgroundColor: dayAvgs.map(d =>
                      d.day === "Sat" || d.day === "Sun"
                        ? "rgba(220,38,38,0.78)"
                        : "rgba(17,24,39,0.75)"
                    ),
                    borderRadius: 5,
                  },
                  {
                    label: "XGBoost",
                    data: dayAvgs.map(d => d.xgboost),
                    backgroundColor: "rgba(22,163,74,0.65)",
                    borderRadius: 5,
                  }
                ]
              }}
              options={opts}
            />
          )}
        </div>
      </div>
    </div>
  );
}
