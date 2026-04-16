export type CategoryKey =
  | "structure"
  | "risk"
  | "confidence"
  | "volume"
  | "momentum"
  | "sr"
  | "time"
  | "market"
  | "scalp"
  | "filters";

export interface CategoryDef {
  label: string;
  description: string;
  color: string;
  stages: readonly string[];
}

export const CATEGORIES: Record<CategoryKey, CategoryDef> = {
  structure: {
    label: "Structure",
    description: "SMC structure checks: HTF EMA/candle alignment, swing break, pullback zone, proximity to swing S/R",
    color: "#60a5fa",
    stages: ["TIER1_EMA", "TIER1_HTF_CANDLE", "TIER2_SWING", "TIER3_PULLBACK", "TIER4_PROXIMITY"],
  },
  risk: {
    label: "Risk",
    description: "Position-sizing and R:R validation",
    color: "#f87171",
    stages: ["RISK_LIMIT", "RISK_RR", "RISK_TP"],
  },
  confidence: {
    label: "Confidence",
    description: "Signal confidence score outside allowed min/max range",
    color: "#fbbf24",
    stages: ["CONFIDENCE", "CONFIDENCE_CAP"],
  },
  volume: {
    label: "Volume",
    description: "Volume ratio too low or missing volume data",
    color: "#a78bfa",
    stages: ["VOLUME_LOW", "VOLUME_NO_DATA"],
  },
  momentum: {
    label: "Momentum",
    description: "MACD or EMA slope against trade direction",
    color: "#34d399",
    stages: ["MACD_MISALIGNED", "EMA_SLOPE"],
  },
  sr: {
    label: "S/R",
    description: "Support / resistance levels block path to target",
    color: "#fb923c",
    stages: ["SR_PATH_BLOCKED", "SR_LEVEL", "SR_CLUSTER"],
  },
  time: {
    label: "Time",
    description: "Outside trading session, market hours, or cooldown active",
    color: "#94a3b8",
    stages: ["SESSION", "COOLDOWN", "MARKET_HOURS"],
  },
  market: {
    label: "Market",
    description: "Market context filters: SMC conflict, directional bias, reversal regime, breakout regime",
    color: "#ec4899",
    stages: ["SMC_CONFLICT", "MARKET_BIAS_FILTER", "REVERSAL_FILTER", "REGIME_BREAKOUT"],
  },
  scalp: {
    label: "Scalp",
    description: "Scalp-mode entry gates (RSI / EMA distance / sweep / blocked hours) and per-pair scalp filters (EMA stack / efficiency ratio)",
    color: "#d946ef",
    stages: ["SCALP_ENTRY_FILTER", "PAIR_SCALP_FILTER"],
  },
  filters: {
    label: "Filters",
    description: "External / late-stage filters: 200 EMA, Claude AI, news blackouts",
    color: "#22d3ee",
    stages: ["EMA200_FILTER", "CLAUDE_FILTER", "NEWS_FILTER"],
  },
};

export const CATEGORY_ORDER: CategoryKey[] = [
  "structure",
  "risk",
  "confidence",
  "volume",
  "momentum",
  "sr",
  "time",
  "market",
  "scalp",
  "filters",
];

export const STAGE_DESCRIPTIONS: Record<string, string> = {
  TIER1_EMA: "4H EMA trend alignment failed",
  TIER1_HTF_CANDLE: "HTF candle direction conflicts with signal",
  TIER2_SWING: "No valid swing break / structure",
  TIER3_PULLBACK: "Pullback zone validation failed",
  TIER4_PROXIMITY: "Entry too close to nearest swing S/R",
  RISK_LIMIT: "Position size exceeds risk limit",
  RISK_RR: "Risk:reward below minimum",
  RISK_TP: "Take-profit distance too small",
  CONFIDENCE: "Confidence below minimum threshold",
  CONFIDENCE_CAP: "Confidence exceeds maximum cap",
  VOLUME_LOW: "Volume ratio below minimum",
  VOLUME_NO_DATA: "No volume data available",
  MACD_MISALIGNED: "Direction against MACD momentum",
  EMA_SLOPE: "Entry against EMA slope direction",
  SR_PATH_BLOCKED: "S/R levels block path to target",
  SR_LEVEL: "Blocked by individual S/R level",
  SR_CLUSTER: "Blocked by S/R cluster",
  SESSION: "Outside trading session / blocked hour",
  COOLDOWN: "Previous signal too recent",
  MARKET_HOURS: "Outside configured market hours",
  SMC_CONFLICT: "SMC structure conflicts with signal",
  MARKET_BIAS_FILTER: "Trade against dominant market bias",
  REVERSAL_FILTER: "Continues prior trend in reversal regime",
  REGIME_BREAKOUT: "Breakout regime blocked (low historical WR)",
  SCALP_ENTRY_FILTER: "Scalp entry gate: RSI zone, EMA distance, sweep, momentum, or blocked hour",
  PAIR_SCALP_FILTER: "Per-pair scalp filter: EMA stack / efficiency ratio",
  EMA200_FILTER: "Blocked by 200 EMA filter",
  CLAUDE_FILTER: "Claude AI flagged trade",
  NEWS_FILTER: "Blocked by news filter",
};

export function describeStage(stage: string): string {
  return STAGE_DESCRIPTIONS[stage] ?? stage;
}

export const ALL_STAGES: string[] = CATEGORY_ORDER.flatMap((k) => [...CATEGORIES[k].stages]);

export const stageToCategory: Record<string, CategoryKey> = (() => {
  const out: Record<string, CategoryKey> = {};
  for (const key of CATEGORY_ORDER) {
    for (const stage of CATEGORIES[key].stages) out[stage] = key;
  }
  return out;
})();

export const stageToColor: Record<string, string> = (() => {
  const out: Record<string, string> = {};
  for (const stage of ALL_STAGES) {
    out[stage] = CATEGORIES[stageToCategory[stage]].color;
  }
  return out;
})();

export const UNKNOWN_STAGE_COLOR = "#64748b";

export function colorForStage(stage: string): string {
  return stageToColor[stage] ?? UNKNOWN_STAGE_COLOR;
}
