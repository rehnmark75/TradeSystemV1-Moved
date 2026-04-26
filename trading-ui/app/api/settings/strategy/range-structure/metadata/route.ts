import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

type InfoRow = { column_name: string; data_type: string };

const CATEGORY_MAP: Record<string, string> = {
  strategy_name: "General",
  strategy_version: "General",

  range_lookback_bars: "Range Build / Sweep",
  sweep_penetration_pips: "Range Build / Sweep",
  rejection_wick_ratio: "Range Build / Sweep",

  ob_fvg_confluence_required: "Confluence / Targets",
  equilibrium_target_enabled: "Confluence / Targets",

  adx_hard_ceiling_primary: "ADX Gates",
  adx_hard_ceiling_htf: "ADX Gates",
  adx_period: "ADX Gates",

  min_rr_ratio: "Risk Management",
  sl_pips_min: "Risk Management",
  sl_pips_max: "Risk Management",
  tp_pips_min: "Risk Management",
  tp_pips_max: "Risk Management",
  sl_buffer_pips: "Risk Management",

  htf_bias_neutral_band: "HTF Bias",

  signal_cooldown_minutes: "Confidence & Cooldown",
  min_confidence: "Confidence & Cooldown",
  max_confidence: "Confidence & Cooldown",

  trust_regime_routing: "Routing",

  primary_timeframe: "Timeframes",
  confirmation_timeframe: "Timeframes",
};

const DISPLAY_ORDER: Record<string, number> = {
  General: 10,
  "Range Build / Sweep": 20,
  "Confluence / Targets": 30,
  "ADX Gates": 40,
  "Risk Management": 50,
  "HTF Bias": 60,
  "Confidence & Cooldown": 70,
  Routing: 80,
  Timeframes: 90,
  Other: 100,
};

const UNIT_MAP: Record<string, string> = {
  sweep_penetration_pips: "pips",
  sl_pips_min: "pips",
  sl_pips_max: "pips",
  tp_pips_min: "pips",
  tp_pips_max: "pips",
  sl_buffer_pips: "pips",
  signal_cooldown_minutes: "minutes",
};

const HIDDEN = new Set([
  "id",
  "is_active",
  "created_at",
  "updated_at",
  "notes",
]);

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

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'range_structure_global_config'
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
          unit: UNIT_MAP[r.column_name] ?? null,
        };
      });

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Failed to load RANGE_STRUCTURE metadata", error);
    return NextResponse.json({ error: "Failed to load RANGE_STRUCTURE metadata" }, { status: 500 });
  }
}
