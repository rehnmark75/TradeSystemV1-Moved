import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function toDisplayName(parameterName: string) {
  return parameterName.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `
        SELECT DISTINCT ON (parameter_name)
          parameter_name,
          category,
          value_type,
          display_order,
          description
        FROM xau_gold_global_config
        WHERE is_active = TRUE
        ORDER BY parameter_name, config_set = 'live' DESC, updated_at DESC
      `
    );

    const rows = result.rows.map((row: any, index: number) => ({
      id: index + 1,
      parameter_name: row.parameter_name,
      display_name: toDisplayName(row.parameter_name),
      category: row.category || "Other",
      subcategory: null,
      data_type: row.value_type || "string",
      min_value: null,
      max_value: null,
      default_value: null,
      description: row.description,
      help_text: row.description,
      display_order: row.display_order ?? 0,
      is_advanced: false,
      requires_restart: false,
      valid_options: null,
      unit: null,
    }));

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Failed to load XAU_GOLD metadata", error);
    return NextResponse.json({ error: "Failed to load XAU_GOLD metadata" }, { status: 500 });
  }
}
