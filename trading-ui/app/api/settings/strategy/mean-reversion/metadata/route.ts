import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

/**
 * No dedicated mean_reversion_parameter_metadata table exists yet.
 * Derive metadata from information_schema so the UI has labels and data-types
 * for every column on mean_reversion_global_config, grouped by theme.
 *
 * If/when a proper metadata table is added, swap this for a
 * `SELECT * FROM mean_reversion_parameter_metadata` the way SMC does it.
 */

type InfoRow = { column_name: string; data_type: string };

const CATEGORY_MAP: Record<string, string> = {
  // Hard ADX gates
  hard_adx_gate_enabled: "Hard ADX Gates",
  adx_hard_ceiling_primary: "Hard ADX Gates",
  adx_hard_ceiling_htf: "Hard ADX Gates",
  adx_period: "Hard ADX Gates",

  // Bollinger Bands
  bb_period: "Bollinger Bands",
  bb_mult: "Bollinger Bands",
  bb_std_dev: "Bollinger Bands",
  bb_touch_required: "Bollinger Bands",

  // RSI
  rsi_period: "RSI",
  rsi_fast_period: "RSI",
  rsi_slow_period: "RSI",
  rsi_ema_period: "RSI",
  rsi_overbought: "RSI",
  rsi_oversold: "RSI",

  // Support / Resistance
  sr_lookback_bars: "Support / Resistance",
  sr_proximity_pips: "Support / Resistance",

  // Risk management
  fixed_stop_loss_pips: "Risk Management",
  fixed_take_profit_pips: "Risk Management",
  min_confidence: "Risk Management",
  max_confidence: "Risk Management",
  sl_buffer_pips: "Risk Management",

  // Timeframes & cooldown
  primary_timeframe: "Timeframes & Cooldown",
  confirmation_timeframe: "Timeframes & Cooldown",
  divergence_timeframe: "Timeframes & Cooldown",
  signal_cooldown_minutes: "Timeframes & Cooldown",

  // Regime / routing
  trust_regime_routing: "Regime & Routing",

  // Legacy mean-reversion helpers (carried over from the archived v0)
  luxalgo_length: "Legacy Oscillators (archived v0)",
  luxalgo_sensitivity: "Legacy Oscillators (archived v0)",
  divergence_lookback: "Legacy Oscillators (archived v0)",
  divergence_enabled: "Legacy Oscillators (archived v0)",
  min_divergence_score: "Legacy Oscillators (archived v0)",
  min_oscillator_agreement: "Legacy Oscillators (archived v0)",
};

const DISPLAY_ORDER: Record<string, number> = {
  "Hard ADX Gates": 10,
  "Bollinger Bands": 20,
  RSI: 30,
  "Support / Resistance": 40,
  "Risk Management": 50,
  "Timeframes & Cooldown": 60,
  "Regime & Routing": 70,
  "Legacy Oscillators (archived v0)": 90,
  Other: 100,
};

function toDisplayName(col: string) {
  return col.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function toDataType(sqlType: string) {
  const t = sqlType.toLowerCase();
  if (t.includes("int") && !t.includes("interval")) return "int";
  if (t.includes("numeric") || t.includes("double") || t.includes("real")) return "float";
  if (t.includes("bool")) return "bool";
  if (t.includes("json")) return "json";
  return "string";
}

const HIDDEN = new Set([
  "id",
  "config_set",
  "config_version",
  "is_active",
  "created_at",
  "updated_at",
  "updated_by",
  "change_reason",
]);

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'mean_reversion_global_config'
        ORDER BY ordinal_position
      `,
    );

    const rows = (result.rows as InfoRow[])
      .filter((r) => !HIDDEN.has(r.column_name))
      .map((r, index) => {
        const category = CATEGORY_MAP[r.column_name] ?? "Other";
        return {
          id: index + 1,
          parameter_name: r.column_name,
          display_name: toDisplayName(r.column_name),
          category,
          subcategory: null,
          data_type: toDataType(r.data_type),
          min_value: null,
          max_value: null,
          default_value: null,
          description: null,
          help_text: null,
          display_order: DISPLAY_ORDER[category] ?? 100,
          is_advanced: false,
          requires_restart: false,
          valid_options: null,
          unit: null,
        };
      });

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Failed to load MEAN_REVERSION metadata", error);
    return NextResponse.json({ error: "Failed to load MEAN_REVERSION metadata" }, { status: 500 });
  }
}
