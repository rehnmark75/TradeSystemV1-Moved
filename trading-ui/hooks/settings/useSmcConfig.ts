import { useState } from "react";
import { useSettings } from "./useSettings";

export function useStrategyConfig(endpoint: string, draftKey: string, configSet?: string) {
  const [conflict, setConflict] = useState<Record<string, unknown> | null>(null);
  const settings = useSettings({
    endpoint,
    draftKey,
    configSet,
    onConflict: (current) => setConflict(current)
  });

  return { ...settings, conflict, setConflict };
}

export function useSmcConfig(configSet?: string) {
  return useStrategyConfig("/api/settings/strategy/smc", "smc-global-settings-draft", configSet);
}
