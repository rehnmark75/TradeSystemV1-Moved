import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export { strategyConfigPool };

export const DEFAULT_CONFIG_SET = "demo";
export const STRATEGY_LABEL = "KAMA_V2";
export const OVERRIDE_TABLE = "kama_v2_pair_overrides";

export const KAMA2_GLOBAL_DEFAULTS: Record<string, unknown> = {
  strategy_name: STRATEGY_LABEL,
  enabled_epic: "",
  monitor_only: true,
  kama_period: 10,
  cross_er_min: 0.35,
  slope_bars: 3,
  slope_min_pips: 0.5,
  ema_trend: true,
  macd_filter: true,
  rsi_extreme_filter: true,
  session_filter: false,
  blocked_hours_utc: "21,22,23,0,1,2,3",
  adx_min: 0.0,
  fixed_stop_loss_pips: 10.0,
  fixed_take_profit_pips: 15.0,
  signal_cooldown_minutes: 30.0,
  base_confidence: 0.60,
  min_confidence: 0.60,
  max_confidence: 0.80,
  updated_at: "1970-01-01T00:00:00.000Z",
};

export const KAMA2_METADATA = [
  ["enabled_epic", "Enabled Epic", "General", "string", "Optional single-epic override. Empty uses enabled pairs from the Kama2 table.", null],
  ["monitor_only", "Monitor Only", "General", "bool", "Log signals without trading by default.", null],
  ["kama_period", "KAMA Period", "KAMA / ER", "int", "KAMA period used by the scanner data feed.", null],
  ["cross_er_min", "Cross ER Min", "KAMA / ER", "float", "Minimum efficiency ratio for crossover signals.", null],
  ["slope_bars", "Slope Bars", "KAMA / ER", "int", "Lookback bars for KAMA slope validation.", null],
  ["slope_min_pips", "Slope Min", "KAMA / ER", "float", "Reject counter-slope beyond this pip threshold.", "pips"],
  ["ema_trend", "EMA200 Trend Filter", "Confirmation Filters", "bool", "Require price to align with EMA200 direction.", null],
  ["macd_filter", "MACD Filter", "Confirmation Filters", "bool", "Require MACD histogram sign alignment.", null],
  ["rsi_extreme_filter", "RSI Extreme Filter", "Confirmation Filters", "bool", "Reject BUY above RSI 70 and SELL below RSI 30.", null],
  ["session_filter", "Session Filter", "Session", "bool", "Block configured UTC hours when enabled.", null],
  ["blocked_hours_utc", "Blocked Hours UTC", "Session", "string", "Comma-separated UTC hours to block.", null],
  ["adx_min", "ADX Minimum", "Confirmation Filters", "float", "Optional ADX gate. Zero disables the gate.", null],
  ["fixed_stop_loss_pips", "Fixed Stop Loss", "Risk Management", "float", "Fixed stop loss used by Kama2.", "pips"],
  ["fixed_take_profit_pips", "Fixed Take Profit", "Risk Management", "float", "Fixed take profit used by Kama2.", "pips"],
  ["signal_cooldown_minutes", "Signal Cooldown", "Cooldown", "float", "Minimum minutes between signals per epic.", "min"],
  ["base_confidence", "Base Confidence", "Confidence", "float", "Base confidence assigned before bonuses.", null],
  ["min_confidence", "Min Confidence", "Confidence", "float", "Minimum confidence required to emit a signal.", null],
  ["max_confidence", "Max Confidence", "Confidence", "float", "Maximum confidence cap.", null],
] as const;

export function normalizeTimestamp(value: unknown): string {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

export async function getOverrideColumns(client?: { query: Function }): Promise<string[]> {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `SELECT column_name FROM information_schema.columns WHERE table_name = '${OVERRIDE_TABLE}'`,
  );
  return result.rows.map((row: { column_name: string }) => row.column_name);
}

export function applyOverride(
  globalConfig: Record<string, unknown>,
  override: Record<string, unknown> | null,
) {
  const effective: Record<string, unknown> = { ...globalConfig };
  if (!override) return effective;

  Object.assign(effective, override.parameter_overrides ?? {});
  Object.entries(override).forEach(([key, value]) => {
    if (["id", "config_set", "epic", "pair_name", "created_at", "updated_at", "parameter_overrides", "notes"].includes(key)) {
      return;
    }
    if (value !== null && value !== undefined) effective[key] = value;
  });
  return effective;
}
