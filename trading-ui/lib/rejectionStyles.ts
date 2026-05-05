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
  | "filters"
  | "lpf"
  | "claude"
  // Generic strategy categories (MEAN_REVERSION, IMPULSE_FADE, XAU_GOLD)
  | "regime"
  | "atr"
  | "data";

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
  lpf: {
    label: "LPF",
    description: "Loss Prevention Filter blocked the signal based on penalty rules",
    color: "#f97316",
    stages: ["LPF_BLOCKED"],
  },
  claude: {
    label: "Claude",
    description: "Claude AI rejected the signal after chart analysis",
    color: "#c084fc",
    stages: ["CLAUDE_REJECTED"],
  },
  regime: {
    label: "Regime",
    description: "Volatility / trend regime gate rejected the signal (ADX, low-vol, trending required)",
    color: "#f59e0b",
    stages: [
      "LOW_VOL_REGIME", "ADX_PRIMARY", "ADX_HTF", "REGIME_NOT_TRENDING", "REGIME_RANGING", "REGIME_EXPANSION",
      // SMC_MOMENTUM structure gates
      "HTF_MISALIGN", "HTF_DISTANCE", "ATR_EXPANSION",
      "NO_LIQUIDITY_POOLS", "NO_SWEEP", "LOW_CONFIDENCE",
    ],
  },
  atr: {
    label: "ATR / Body",
    description: "ATR spike guard or impulse body check failed",
    color: "#10b981",
    stages: ["ATR_SPIKE", "ATR_UNAVAILABLE", "BODY_TOO_SMALL"],
  },
  data: {
    label: "Data",
    description: "Insufficient candle data or missing indicators",
    color: "#6b7280",
    stages: [
      "INSUFFICIENT_DATA",
      "INSUFFICIENT_HTF_DATA",
      "NO_PATTERN",
      "NO_BOS",
      "NO_OB_OR_FVG",
      "TIER2_NO_SIGNAL",
      "TIER3_NO_ENTRY",
      // RANGE_FADE-specific
      "INDICATOR_NAN",
      "NO_PRIOR_RANGE",
      "NO_HTF_BIAS",
      "NO_TRIGGER",
      "PAIR_DISABLED",
      "BAND_WIDTH_OUT_OF_RANGE",
      "BAR_RANGE_TOO_WIDE",
    ],
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
  "lpf",
  "claude",
  "regime",
  "atr",
  "data",
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
  LPF_BLOCKED: "Loss Prevention Filter: signal blocked by penalty rules",
  CLAUDE_REJECTED: "Claude AI rejected signal after chart analysis",
  // Generic strategy stages
  LOW_VOL_REGIME: "Low volatility regime: ADX too low for mean-reversion entry",
  ADX_PRIMARY: "ADX on primary timeframe below minimum threshold",
  ADX_HTF: "ADX on higher timeframe below minimum threshold",
  REGIME_NOT_TRENDING: "Market regime is not trending (required for this strategy)",
  REGIME_RANGING: "Market regime is ranging (blocked for this strategy)",
  REGIME_EXPANSION: "Volatility expansion detected — news-spike guard active",
  ATR_SPIKE: "ATR spike: extreme post-news volatility blocks entry",
  ATR_UNAVAILABLE: "ATR could not be calculated (insufficient data)",
  BODY_TOO_SMALL: "Candle body below ATR multiplier threshold",
  INSUFFICIENT_DATA: "Not enough candle rows to evaluate strategy",
  INSUFFICIENT_HTF_DATA: "Not enough higher-timeframe candle rows",
  NO_PATTERN: "No qualifying BB/RSI pattern in current candle",
  NO_BOS: "No Break of Structure detected on trigger timeframe",
  NO_OB_OR_FVG: "No Order Block or Fair Value Gap found at pullback",
  TIER2_NO_SIGNAL: "Tier 2 (trigger) timeframe produced no valid signal",
  TIER3_NO_ENTRY: "Tier 3 (entry) timeframe: no pullback entry found",
  // RANGE_FADE stages
  INDICATOR_NAN: "Indicator returned NaN — insufficient candle history",
  NO_PRIOR_RANGE: "No prior range high/low found in lookback window",
  NO_HTF_BIAS: "Higher-timeframe EMA bias could not be determined",
  NO_TRIGGER: "Price not at BB band + RSI extremity + range boundary simultaneously",
  PAIR_DISABLED: "Pair is disabled in strategy configuration",
  BAND_WIDTH_OUT_OF_RANGE: "Bollinger Band width outside min/max pip threshold",
  BAR_RANGE_TOO_WIDE: "Current bar range exceeds max_current_range_pips limit",
  // SMC_MOMENTUM stages
  HTF_MISALIGN: "Trade direction conflicts with 4H EMA50 bias",
  HTF_DISTANCE: "Price too close to 4H EMA — minimum distance not met",
  ATR_EXPANSION: "Sweep bar true range did not expand (momentum filter)",
  NO_LIQUIDITY_POOLS: "No swing high/low liquidity pools found in lookback",
  NO_SWEEP: "No sweep-and-rejection wick pattern detected",
  LOW_CONFIDENCE: "Confidence score below minimum threshold",
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
