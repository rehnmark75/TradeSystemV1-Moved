import { useState } from "react";
import { useSettings } from "./useSettings";

function parseDefaults(raw: Record<string, string | null>) {
  const parsed: Record<string, unknown> = {};
  Object.entries(raw).forEach(([key, value]) => {
    if (!value) {
      parsed[key] = null;
      return;
    }
    const trimmed = value.trim();
    if (trimmed === "true" || trimmed === "false") {
      parsed[key] = trimmed === "true";
      return;
    }
    if (trimmed.startsWith("'") && trimmed.includes("'::")) {
      parsed[key] = trimmed.split("'")[1];
      return;
    }
    if (!Number.isNaN(Number(trimmed))) {
      parsed[key] = Number(trimmed);
      return;
    }
    parsed[key] = trimmed;
  });
  return parsed;
}

export function useScannerConfig() {
  const [conflict, setConflict] = useState<Record<string, unknown> | null>(null);
  const settings = useSettings({
    endpoint: "/api/settings/scanner",
    defaultsEndpoint: "/api/settings/scanner/defaults",
    draftKey: "scanner-settings-draft",
    onConflict: (current) => setConflict(current),
    parseDefaults
  });

  return { ...settings, conflict, setConflict };
}
