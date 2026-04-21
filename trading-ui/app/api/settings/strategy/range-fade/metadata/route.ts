import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

type InfoRow = { column_name: string; data_type: string };

const CATEGORY_MAP: Record<string, string> = {
  strategy_name: "General",
  version: "General",
  monitor_only: "General",

  primary_timeframe: "Timeframes & Cooldown",
  confirmation_timeframe: "Timeframes & Cooldown",
  signal_cooldown_minutes: "Timeframes & Cooldown",

  bb_period: "Bollinger Bands",
  bb_mult: "Bollinger Bands",

  rsi_period: "RSI",
  rsi_oversold: "RSI",
  rsi_overbought: "RSI",

  range_lookback_bars: "Range Structure",
  range_proximity_pips: "Range Structure",

  min_band_width_pips: "Volatility Gates",
  max_band_width_pips: "Volatility Gates",
  max_current_range_pips: "Volatility Gates",

  htf_ema_period: "HTF Bias",
  htf_slope_bars: "HTF Bias",
  allow_neutral_htf: "HTF Bias",

  min_confidence: "Risk Management",
  max_confidence: "Risk Management",
  fixed_stop_loss_pips: "Risk Management",
  fixed_take_profit_pips: "Risk Management",

  london_start_hour_utc: "Session",
  new_york_end_hour_utc: "Session",
};

const DISPLAY_ORDER: Record<string, number> = {
  General: 10,
  "Timeframes & Cooldown": 20,
  "HTF Bias": 30,
  "Bollinger Bands": 40,
  RSI: 50,
  "Range Structure": 60,
  "Volatility Gates": 70,
  Session: 80,
  "Risk Management": 90,
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
  "profile_name",
  "config_set",
  "is_active",
  "created_at",
  "updated_at",
  "notes",
]);

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'eurusd_range_fade_global_config'
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
    console.error("Failed to load RANGE_FADE metadata", error);
    return NextResponse.json({ error: "Failed to load RANGE_FADE metadata" }, { status: 500 });
  }
}
