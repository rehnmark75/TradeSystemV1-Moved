import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_CONFIG_SET = "demo";

const CATEGORY_DISPLAY: Record<string, string> = {
  general: "General",
  confidence: "Confidence",
  cooldown: "Cooldown",
  filters: "Filters",
  risk: "Risk Management",
  sweep: "Sweep Detection",
  timeframes: "Timeframes",
};

const DISPLAY_ORDER: Record<string, number> = {
  General: 10,
  Timeframes: 20,
  "Sweep Detection": 30,
  "Risk Management": 40,
  Confidence: 50,
  Filters: 60,
  Cooldown: 70,
  Other: 100,
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
        FROM smc_momentum_global_config
        WHERE config_set = $1 AND is_active = TRUE
        ORDER BY category, parameter_name
      `,
      [configSet],
    );

    const rows = result.rows.map(
      (
        row: { parameter_name: string; value_type: string; category: string; description: string | null },
        index: number,
      ) => {
        const category = CATEGORY_DISPLAY[row.category] ?? row.category ?? "Other";
        return {
          id: index + 1,
          parameter_name: row.parameter_name,
          display_name: toDisplayName(row.parameter_name),
          category,
          subcategory: null,
          data_type: row.value_type,
          min_value: null,
          max_value: null,
          default_value: null,
          description: row.description ?? null,
          help_text: null,
          display_order: DISPLAY_ORDER[category] ?? 100,
          is_advanced: false,
          requires_restart: false,
          valid_options: null,
          unit: null,
        };
      },
    );

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Failed to load SMC_MOMENTUM metadata", error);
    return NextResponse.json({ error: "Failed to load SMC_MOMENTUM metadata" }, { status: 500 });
  }
}
