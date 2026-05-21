import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_CONFIG_SET = "demo";

const CATEGORY_DISPLAY: Record<string, string> = {
  cooldown: "Cooldown",
  entry: "Entry",
  filters: "Filters",
  general: "General",
  indicators: "Indicators",
  risk: "Risk Management",
  session: "Session",
  usd_jpy_filter: "USDJPY Filter",
};

function toDisplayName(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;
    const result = await strategyConfigPool.query(
      `
        SELECT parameter_name, value_type, category, description
        FROM fa_or_atr_trail_global_config
        WHERE config_set = $1 AND is_active = TRUE
        ORDER BY category, parameter_name
      `,
      [configSet],
    );

    const rows = result.rows.map(
      (row: { parameter_name: string; value_type: string; category: string; description: string | null }, index: number) => ({
        id: index + 1,
        parameter_name: row.parameter_name,
        display_name: toDisplayName(row.parameter_name),
        category: CATEGORY_DISPLAY[row.category] ?? row.category ?? "Other",
        subcategory: null,
        data_type: row.value_type,
        min_value: null,
        max_value: null,
        default_value: null,
        description: row.description ?? null,
        help_text: null,
        display_order: row.category === "general" ? 10 : 100,
        is_advanced: false,
        requires_restart: false,
        valid_options: null,
        unit: null,
      }),
    );

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Failed to load FA_OR_ATR_TRAIL metadata", error);
    return NextResponse.json({ error: "Failed to load FA_OR_ATR_TRAIL metadata" }, { status: 500 });
  }
}
