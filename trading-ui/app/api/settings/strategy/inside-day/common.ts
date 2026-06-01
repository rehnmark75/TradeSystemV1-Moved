import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export { strategyConfigPool };

export const DEFAULT_CONFIG_SET = "demo";
export const STRATEGY_LABEL = "INSIDE_DAY";
export const OVERRIDE_TABLE = "inside_day_pair_overrides";

export const INSIDE_DAY_GLOBAL_DEFAULTS: Record<string, unknown> = {
  strategy_name: STRATEGY_LABEL,
  enabled_pairs: ["CS.D.EURUSD.CEEM.IP", "CS.D.USDJPY.MINI.IP"],
  monitor_only: true,
  weekly_bias_q: 0.30,
  inside_day_min_pips: 10.0,
  inside_day_max_pips: 100.0,
  atr_period: 14,
  atr_buffer_fraction: 0.05,
  reward_risk: 2.0,
  base_confidence: 0.65,
  updated_at: "1970-01-01T00:00:00.000Z",
};

export const INSIDE_DAY_METADATA = [
  ["enabled_pairs", "Enabled Pairs", "General", "json", "Pairs that cleared the inside-day OOS and spread-stress gate.", null],
  ["monitor_only", "Monitor Only", "General", "bool", "Log signals without live trading by default.", null],
  ["weekly_bias_q", "Weekly Bias Quantile", "Setup Detection", "float", "Top or bottom fraction of prior completed weekly range required for directional bias.", null],
  ["inside_day_min_pips", "Inside-Day Min Range", "Setup Detection", "float", "Minimum completed inside-day range.", "pips"],
  ["inside_day_max_pips", "Inside-Day Max Range", "Setup Detection", "float", "Maximum completed inside-day range.", "pips"],
  ["atr_period", "ATR Period", "Risk Management", "int", "Daily ATR period used for stop buffer sizing.", null],
  ["atr_buffer_fraction", "ATR Buffer Fraction", "Risk Management", "float", "Stop buffer beyond the opposite inside-day extreme as a fraction of daily ATR.", null],
  ["reward_risk", "Reward Risk", "Risk Management", "float", "Take-profit distance as a multiple of stop risk.", "R"],
  ["base_confidence", "Base Confidence", "Confidence", "float", "Fixed confidence emitted by the strategy.", null],
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
