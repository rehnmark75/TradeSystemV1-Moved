import { useState } from "react";
import { useSettings } from "./useSettings";

export function useSmcConfig(configSet?: string) {
  const [conflict, setConflict] = useState<Record<string, unknown> | null>(null);
  const settings = useSettings({
    endpoint: "/api/settings/strategy/smc",
    draftKey: "smc-global-settings-draft",
    configSet,
    onConflict: (current) => setConflict(current)
  });

  return { ...settings, conflict, setConflict };
}
