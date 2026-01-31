import { apiUrl } from "./api";

export async function logTelemetry(event: {
  event_type: string;
  user?: string;
  page?: string;
  details?: Record<string, unknown>;
}) {
  try {
    await fetch(apiUrl("/api/settings/telemetry"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event)
    });
  } catch {
    // no-op
  }
}
