import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

type InfoRow = { column_name: string; data_type: string };

const CATEGORY_MAP: Record<string, string> = {
  session_start_hour: "Session",
  session_end_hour: "Session",
  atr_body_multiplier: "Impulse Detection",
  atr_period: "Impulse Detection",
  max_atr_pips: "Impulse Detection",
  fixed_stop_loss_pips: "Risk Management",
  fixed_take_profit_pips: "Risk Management",
  time_stop_candles: "Risk Management",
  min_confidence: "Confidence",
  max_confidence: "Confidence",
  signal_cooldown_minutes: "Cooldown",
};

const DISPLAY_ORDER: Record<string, number> = {
  Session: 10,
  "Impulse Detection": 20,
  "Risk Management": 30,
  Confidence: 40,
  Cooldown: 50,
  Other: 100,
};

const HIDDEN = new Set([
  "id",
  "strategy_name",
  "version",
  "is_active",
  "created_at",
  "updated_at",
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
        WHERE table_name = 'impulse_fade_global_config'
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
    console.error("Failed to load IMPULSE_FADE metadata", error);
    return NextResponse.json({ error: "Failed to load IMPULSE_FADE metadata" }, { status: 500 });
  }
}
