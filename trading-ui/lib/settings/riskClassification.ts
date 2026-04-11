/**
 * Classify strategy parameters by their risk/impact level.
 * Used to show visual risk indicators in the settings UI.
 */

export type RiskLevel = "critical" | "high" | "normal";

const CRITICAL_PARAMS = new Set([
  "auto_trading_enabled",
  "enable_order_execution",
  "risk_per_trade_pct",
  "max_concurrent_signals",
  "enabled_pairs",
]);

const HIGH_RISK_PARAMS = new Set([
  "fixed_stop_loss_pips",
  "fixed_take_profit_pips",
  "min_rr_ratio",
  "min_confidence_threshold",
  "max_confidence_threshold",
  "signal_cooldown_hours",
  "max_consecutive_losses_before_block",
  "consecutive_loss_block_hours",
  "scalp_mode_enabled",
  "fixed_sl_tp_override_enabled",
  "session_filter_enabled",
  "max_sl_absolute_pips",
  "sl_buffer_pips",
]);

export function getParamRiskLevel(
  paramName: string,
  requiresRestart?: boolean
): RiskLevel {
  if (CRITICAL_PARAMS.has(paramName)) return "critical";
  if (HIGH_RISK_PARAMS.has(paramName)) return "high";
  if (requiresRestart) return "high";
  return "normal";
}

export function getRiskColor(level: RiskLevel): string {
  switch (level) {
    case "critical":
      return "var(--bad)";
    case "high":
      return "var(--warn)";
    default:
      return "transparent";
  }
}

export function getRiskLabel(level: RiskLevel): string {
  switch (level) {
    case "critical":
      return "Critical";
    case "high":
      return "High Impact";
    default:
      return "";
  }
}
