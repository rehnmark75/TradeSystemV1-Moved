export type FetchResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: string; status?: number };

export async function fetchJson<T>(
  url: string,
  options: RequestInit & { timeoutMs?: number } = {}
): Promise<FetchResult<T>> {
  const { timeoutMs = 5000, ...init } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal, cache: "no-store" });
    clearTimeout(timer);
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return { ok: false, error: body || res.statusText, status: res.status };
    }
    const data = (await res.json()) as T;
    return { ok: true, data };
  } catch (err) {
    clearTimeout(timer);
    if ((err as Error).name === "AbortError") {
      return { ok: false, error: "upstream_timeout", status: 504 };
    }
    return { ok: false, error: (err as Error).message, status: 500 };
  }
}
