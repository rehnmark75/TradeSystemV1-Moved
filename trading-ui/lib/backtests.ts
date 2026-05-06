export const dynamic = "force-dynamic";

export const DEFAULT_BACKTEST_DAYS = 14;
export const DEFAULT_BACKTEST_LIMIT = 20;
export const BACKTEST_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"] as const;
export const BACKTEST_STRATEGIES = [
  "SMC_SIMPLE",
  "SMC_MOMENTUM",
  "MEAN_REVERSION",
  "IMPULSE_FADE",
  "RANGE_FADE",
  "XAU_GOLD",
] as const;

export type BacktestStrategy = (typeof BACKTEST_STRATEGIES)[number];

export const BACKTEST_STRATEGY_LABELS: Record<BacktestStrategy, string> = {
  SMC_SIMPLE: "SMC Simple (FX scalp)",
  SMC_MOMENTUM: "SMC Momentum",
  MEAN_REVERSION: "Mean Reversion",
  IMPULSE_FADE: "Impulse Fade",
  RANGE_FADE: "Range Fade",
  XAU_GOLD: "XAU Gold (4H/1H/15m)",
};

export const GOLD_EPIC = "CS.D.CFEGOLD.CEE.IP";

const EPIC_OPTIONS = [
  ["EURUSD", "CS.D.EURUSD.MINI.IP"],
  ["GBPUSD", "CS.D.GBPUSD.MINI.IP"],
  ["USDJPY", "CS.D.USDJPY.MINI.IP"],
  ["AUDUSD", "CS.D.AUDUSD.MINI.IP"],
  ["USDCHF", "CS.D.USDCHF.MINI.IP"],
  ["USDCAD", "CS.D.USDCAD.MINI.IP"],
  ["NZDUSD", "CS.D.NZDUSD.MINI.IP"],
  ["EURJPY", "CS.D.EURJPY.MINI.IP"],
  ["AUDJPY", "CS.D.AUDJPY.MINI.IP"],
  ["GBPJPY", "CS.D.GBPJPY.MINI.IP"],
  ["XAUUSD", GOLD_EPIC],
] as const;

export function parsePositiveInt(value: string | null, fallback: number) {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.floor(parsed);
}

export function epicToPair(epic: string | null | undefined) {
  if (!epic) return "N/A";
  const explicit = EPIC_OPTIONS.find(([, fullEpic]) => fullEpic === epic);
  if (explicit) return explicit[0];
  const parts = epic.split(".");
  if (parts.length >= 3) return parts[2].slice(0, 6);
  return epic;
}

export function formatDateTime(value: string | null | undefined) {
  if (!value) return "N/A";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDuration(seconds: number | null | undefined) {
  if (!seconds || seconds <= 0) return "Pending";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

export const STRATEGY_METADATA_ENDPOINT: Record<BacktestStrategy, string> = {
  SMC_SIMPLE: "/trading/api/settings/strategy/smc/metadata",
  SMC_MOMENTUM: "/trading/api/settings/strategy/smc-momentum/metadata",
  MEAN_REVERSION: "/trading/api/settings/strategy/mean-reversion/metadata",
  IMPULSE_FADE: "/trading/api/settings/strategy/impulse-fade/metadata",
  RANGE_FADE: "/trading/api/settings/strategy/range-fade/metadata",
  XAU_GOLD: "/trading/api/settings/strategy/xau-gold/metadata",
};

export function getStrategyLabel(strategyName: string | null | undefined): string {
  if (!strategyName) return "N/A";
  return BACKTEST_STRATEGY_LABELS[strategyName as BacktestStrategy] ?? strategyName;
}

export function getStrategyBadgeStyle(strategyName: string | null | undefined): { background: string; color: string; borderColor: string } {
  if (strategyName === "XAU_GOLD") return { background: "#fff7ed", color: "#9a3412", borderColor: "#fed7aa" };
  if (strategyName === "MEAN_REVERSION") return { background: "#ecfdf5", color: "#047857", borderColor: "#a7f3d0" };
  if (strategyName === "IMPULSE_FADE") return { background: "#eff6ff", color: "#1d4ed8", borderColor: "#bfdbfe" };
  if (strategyName === "RANGE_FADE") return { background: "#f5f3ff", color: "#6d28d9", borderColor: "#ddd6fe" };
  if (strategyName?.startsWith("SMC")) return { background: "#f8fafc", color: "#334155", borderColor: "#cbd5e1" };
  return { background: "#f3f4f6", color: "#4b5563", borderColor: "#d1d5db" };
}

export function getEpicOptions() {
  return EPIC_OPTIONS.map(([label, value]) => ({ label, value }));
}

export function coerceBoolean(value: unknown) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true") return true;
    if (normalized === "false") return false;
  }
  return false;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export type ValidVariationConfig = {
  enabled: true;
  param_grid: Record<string, number[]>;
  workers: number;
  rank_by: string;
  top_n: number;
};

const ALLOWED_RANK_BY = new Set([
  "composite_score",
  "win_rate",
  "total_pips",
  "profit_factor",
  "expectancy",
]);

export function validateVariationConfig(value: unknown): { value: ValidVariationConfig | null; error?: string } {
  if (value == null) return { value: null };
  if (!isRecord(value)) return { value: null, error: "Variation config must be an object" };
  if (!coerceBoolean(value.enabled)) return { value: null };

  const paramGrid = value.param_grid;
  if (!isRecord(paramGrid)) {
    return { value: null, error: "Variation param_grid must be an object" };
  }

  const normalizedGrid: Record<string, number[]> = {};
  const gridEntries = Object.entries(paramGrid);
  if (!gridEntries.length) {
    return { value: null, error: "Variation param_grid cannot be empty" };
  }
  if (gridEntries.length > 5) {
    return { value: null, error: "Variation testing is limited to 5 parameters per run" };
  }

  let totalCombos = 1;
  for (const [key, rawValues] of gridEntries) {
    if (!Array.isArray(rawValues) || !rawValues.length) {
      return { value: null, error: `Variation parameter "${key}" must be a non-empty array` };
    }
    if (rawValues.length > 10) {
      return { value: null, error: `Variation parameter "${key}" exceeds the 10-value limit` };
    }
    const values = rawValues.map((entry) => Number(entry));
    if (values.some((entry) => !Number.isFinite(entry))) {
      return { value: null, error: `Variation parameter "${key}" contains a non-numeric value` };
    }
    normalizedGrid[key] = values;
    totalCombos *= values.length;
  }

  if (totalCombos > 100) {
    return { value: null, error: `Variation run is too large (${totalCombos} combinations). Limit is 100.` };
  }

  const workers = Math.max(2, Math.min(8, Number(value.workers) || 4));
  const topN = Math.max(1, Math.min(50, Number(value.top_n) || 10));
  const rankBy =
    typeof value.rank_by === "string" && ALLOWED_RANK_BY.has(value.rank_by)
      ? value.rank_by
      : "composite_score";

  return {
    value: {
      enabled: true,
      param_grid: normalizedGrid,
      workers,
      rank_by: rankBy,
      top_n: topN,
    },
  };
}
