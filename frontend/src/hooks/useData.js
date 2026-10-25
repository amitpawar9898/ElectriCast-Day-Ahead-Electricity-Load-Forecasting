import { useState, useEffect } from "react";

export function useData(fetcher, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetcher(controller.signal)
      .then((d) => {
        if (controller.signal.aborted) return;
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        if (controller.signal.aborted || e.name === "AbortError") return;
        setError(e.message);
        setLoading(false);
      });
    return () => controller.abort();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error };
}
