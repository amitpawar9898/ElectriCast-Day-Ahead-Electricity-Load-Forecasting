import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../utils/api";

const SUGGESTIONS = [
  "Which model is best and by how much?",
  "Why is December different?",
  "Where does XGBoost fail?",
];

const MAX_QUESTION_CHARS = 500;

const mono = { fontFamily: "'IBM Plex Mono', monospace" };

const bubbleBase = {
  maxWidth: "88%",
  borderRadius: 12,
  padding: "9px 12px",
  fontSize: 12,
  lineHeight: 1.55,
  whiteSpace: "pre-wrap",
  overflowWrap: "anywhere",
};

export default function ChatDrawer() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState(null);
  const [statusNonce, setStatusNonce] = useState(0);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [lastQuestion, setLastQuestion] = useState("");
  const bodyRef = useRef(null);
  const inputRef = useRef(null);

  const available = status?.available === true;

  useEffect(() => {
    if (!open) return undefined;
    const controller = new AbortController();
    setStatusLoading(true);
    setStatusError(null);
    api.chatStatus(controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setStatus(data);
        setStatusLoading(false);
      })
      .catch((e) => {
        if (controller.signal.aborted || e.name === "AbortError") return;
        setStatusError(e.message);
        setStatusLoading(false);
      });
    return () => controller.abort();
  }, [open, statusNonce]);

  useEffect(() => {
    if (open && available) inputRef.current?.focus();
  }, [open, available]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending, error, open]);

  useEffect(() => {
    if (available) {
      setError((prev) => (prev && prev.status === 503 ? null : prev));
    }
  }, [available]);

  const ask = useCallback(
    async (raw) => {
      const question = raw.trim();
      if (!question || sending || !available) return;
      if (question.length > MAX_QUESTION_CHARS) {
        setError({ message: `Questions are limited to ${MAX_QUESTION_CHARS} characters.` });
        return;
      }
      setOpen(true);
      setInput("");
      setError(null);
      setLastQuestion(question);
      setMessages((prev) => [...prev, { role: "user", text: question }]);
      setSending(true);
      try {
        const data = await api.chat(question);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            text: data.answer,
            tools: (data.tools || []).map((t) => t.name).filter(Boolean),
            sources: data.sources || [],
            provider: data.provider,
            model: data.model,
          },
        ]);
      } catch (e) {
        setError({ message: e.message, status: e.status });
        if (e.status === 503) {
          setStatus((prev) => ({
            ...(prev || {}),
            available: false,
            provider: null,
            model: null,
            detail: e.detail || e.message,
          }));
        }
      } finally {
        setSending(false);
      }
    },
    [available, sending]
  );

  const recheck = () => setStatusNonce((n) => n + 1);

  if (!open) {
    return (
      <button
        type="button"
        className="chat-launcher"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded="false"
        aria-label="Open Ask ElectriCast"
      >
        <span aria-hidden="true" style={{ fontSize: 13 }}>⚡</span>
        Ask ElectriCast
      </button>
    );
  }

  return (
    <section
      className="chat-drawer"
      role="dialog"
      aria-modal="false"
      aria-label="Ask ElectriCast"
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          padding: "11px 14px",
          borderBottom: "1px solid #e5e7eb",
          background: "#ffffff",
          flexShrink: 0,
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div style={{ ...mono, fontSize: 12, fontWeight: 700, color: "#111827" }}>
            Ask ElectriCast
          </div>
          <div style={{ fontSize: 9, color: "#6b7280", marginTop: 1 }}>
            Answers only from results.json · data_quality.json
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          {available && (
            <span
              style={{
                ...mono,
                fontSize: 8,
                fontWeight: 600,
                color: "#15803d",
                background: "#f0fdf4",
                border: "1px solid #bbf7d0",
                borderRadius: 9,
                padding: "2px 7px",
                maxWidth: 150,
                display: "inline-block",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={`${status.provider} · ${status.model}`}
            >
              {status.provider} · {status.model}
            </span>
          )}
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close assistant"
            style={{
              border: "1px solid #e5e7eb",
              background: "#fff",
              color: "#4b5563",
              width: 26,
              height: 26,
              borderRadius: 7,
              cursor: "pointer",
              fontSize: 15,
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>
      </div>

      <div className="chat-drawer-body" ref={bodyRef}>
        <div
          style={{
            background: "#f8fafc",
            border: "1px dashed #d1d5db",
            borderRadius: 10,
            padding: "9px 11px",
            fontSize: 11,
            lineHeight: 1.5,
            color: "#4b5563",
          }}
        >
          Every number comes from the saved backtest artifacts. If the data does
          not contain an answer, the assistant says so.
        </div>

        {!statusLoading && (statusError || (status && !available)) && (
          <div
            role="alert"
            style={{
              background: "#fffbeb",
              border: "1px solid #fde68a",
              borderRadius: 10,
              padding: "10px 12px",
              color: "#92400e",
              fontSize: 11,
              lineHeight: 1.5,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 2 }}>Assistant unavailable</div>
            <div>{statusError || status?.detail || "No answer provider is reachable."}</div>
            <button
              type="button"
              onClick={recheck}
              disabled={statusLoading}
              style={{
                marginTop: 8,
                padding: "4px 10px",
                borderRadius: 6,
                border: "1px solid #f59e0b",
                background: "#fff",
                color: "#92400e",
                ...mono,
                fontSize: 10,
                fontWeight: 600,
                cursor: statusLoading ? "wait" : "pointer",
              }}
            >
              {statusLoading ? "Checking…" : "Recheck"}
            </button>
          </div>
        )}

        {statusLoading && (
          <div
            role="status"
            style={{ ...mono, fontSize: 10, color: "#6b7280", padding: "4px 2px" }}
          >
            Checking assistant…
          </div>
        )}

        {available && (
          <>
            {messages.length === 0 && (
              <div style={{ marginTop: 2 }}>
                <div
                  style={{
                    ...mono,
                    fontSize: 9,
                    fontWeight: 700,
                    color: "#6b7280",
                    textTransform: "uppercase",
                    letterSpacing: 0.6,
                    marginBottom: 6,
                  }}
                >
                  Suggested questions
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {SUGGESTIONS.map((question) => (
                    <button
                      key={question}
                      type="button"
                      onClick={() => ask(question)}
                      style={{
                        textAlign: "left",
                        padding: "8px 10px",
                        borderRadius: 9,
                        border: "1px solid #dbeafe",
                        background: "#eff6ff",
                        color: "#1d4ed8",
                        fontSize: 11,
                        lineHeight: 1.4,
                        cursor: "pointer",
                      }}
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                style={{
                  display: "flex",
                  justifyContent: message.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                <div
                  style={{
                    ...bubbleBase,
                    ...(message.role === "user"
                      ? { background: "#2563eb", color: "#ffffff" }
                      : {
                          background: "#ffffff",
                          color: "#111827",
                          border: "1px solid #e5e7eb",
                        }),
                  }}
                >
                  {message.text}
                  {message.role === "assistant" &&
                    (message.tools.length > 0 || message.sources.length > 0) && (
                      <div
                        style={{
                          ...mono,
                          marginTop: 6,
                          paddingTop: 5,
                          borderTop: "1px solid #f3f4f6",
                          fontSize: 8,
                          color: "#9ca3af",
                          whiteSpace: "normal",
                        }}
                      >
                        {message.tools.length > 0 && `tools: ${message.tools.join(", ")}`}
                        {message.tools.length > 0 && message.sources.length > 0 && " · "}
                        {message.sources.length > 0 &&
                          `sources: ${message.sources
                            .map((s) => s.split("/").pop())
                            .join(", ")}`}
                      </div>
                    )}
                </div>
              </div>
            ))}

            {sending && (
              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div
                  role="status"
                  aria-label="Loading answer"
                  style={{
                    ...bubbleBase,
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    background: "#ffffff",
                    border: "1px solid #e5e7eb",
                  }}
                >
                  <span className="chat-dot" />
                  <span className="chat-dot" />
                  <span className="chat-dot" />
                  <span style={{ fontSize: 10, color: "#6b7280", marginLeft: 3 }}>
                    Consulting the results…
                  </span>
                </div>
              </div>
            )}

            {error && (
              <div
                role="alert"
                style={{
                  background: "#fef2f2",
                  border: "1px solid #fecaca",
                  borderRadius: 10,
                  padding: "9px 11px",
                  color: "#b91c1c",
                  fontSize: 11,
                  lineHeight: 1.5,
                }}
              >
                <div style={{ fontWeight: 700, marginBottom: 2 }}>
                  {error.status === 503 ? "Assistant unavailable" : "Could not get an answer"}
                </div>
                <div>{error.message}</div>
                {error.status !== 503 && lastQuestion && (
                  <button
                    type="button"
                    onClick={() => ask(lastQuestion)}
                    disabled={sending}
                    style={{
                      marginTop: 7,
                      padding: "3px 9px",
                      borderRadius: 6,
                      border: "1px solid #fca5a5",
                      background: "#fff",
                      color: "#b91c1c",
                      ...mono,
                      fontSize: 10,
                      fontWeight: 600,
                      cursor: sending ? "wait" : "pointer",
                    }}
                  >
                    Retry
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {available && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
          style={{
            borderTop: "1px solid #e5e7eb",
            padding: "10px 12px 12px",
            background: "#ffffff",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
            <textarea
              ref={inputRef}
              rows={2}
              maxLength={MAX_QUESTION_CHARS}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              placeholder="Ask about metrics, December, or bad hours…"
              aria-label="Your question"
              disabled={sending}
              style={{
                flex: 1,
                resize: "none",
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                fontSize: 12,
                fontFamily: "'IBM Plex Sans', sans-serif",
                lineHeight: 1.45,
                color: "#111827",
                background: sending ? "#f9fafb" : "#ffffff",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={!input.trim() || sending}
              style={{
                padding: "8px 13px",
                borderRadius: 8,
                border: "none",
                background: !input.trim() || sending ? "#d1d5db" : "#2563eb",
                color: "#ffffff",
                ...mono,
                fontSize: 10,
                fontWeight: 700,
                cursor: !input.trim() || sending ? "default" : "pointer",
                height: 36,
              }}
            >
              {sending ? "…" : "Send"}
            </button>
          </div>
          <div
            style={{
              ...mono,
              display: "flex",
              justifyContent: "space-between",
              gap: 8,
              marginTop: 4,
              fontSize: 8,
              color: "#9ca3af",
            }}
          >
            <span>Enter sends · Shift+Enter new line</span>
            <span>
              {input.length}/{MAX_QUESTION_CHARS}
            </span>
          </div>
        </form>
      )}
    </section>
  );
}
