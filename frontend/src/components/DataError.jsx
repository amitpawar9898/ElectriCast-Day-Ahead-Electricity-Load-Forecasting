export default function DataError({ error, label = "data" }) {
  if (!error) return null;
  return (
    <div role="alert" style={{
      background: "#fef2f2", border: "1px solid #fecaca", color: "#b91c1c",
      borderRadius: 10, padding: "10px 14px", fontSize: 12, fontWeight: 500,
      fontFamily: "'IBM Plex Sans', sans-serif",
    }}>
      Could not load {label}: {error}
    </div>
  );
}
