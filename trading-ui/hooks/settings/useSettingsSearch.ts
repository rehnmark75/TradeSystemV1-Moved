import { useMemo } from "react";

export function useSettingsSearch<T extends { [key: string]: unknown }>(
  keys: string[],
  data: T | null,
  defaults: Record<string, unknown>,
  query: string,
  showModifiedOnly: boolean
) {
  return useMemo(() => {
    if (!data) return [];
    const normalizedQuery = query.trim().toLowerCase();
    const hasDefaults = Object.keys(defaults).length > 0;
    return keys.filter((key) => {
      const value = data[key];
      const defaultValue = defaults[key];
      const isModified = hasDefaults
        ? defaultValue !== undefined &&
          (() => {
            if (typeof value === "object" && typeof defaultValue === "object") {
              return JSON.stringify(value) !== JSON.stringify(defaultValue);
            }
            return value !== defaultValue;
          })()
        : false;
      if (showModifiedOnly && hasDefaults && !isModified) return false;
      if (!normalizedQuery) return true;
      return key.toLowerCase().includes(normalizedQuery);
    });
  }, [keys, data, defaults, query, showModifiedOnly]);
}
