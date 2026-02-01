import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `
        SELECT
          id,
          parameter_name,
          display_name,
          category,
          subcategory,
          data_type,
          min_value,
          max_value,
          default_value,
          description,
          help_text,
          display_order,
          is_advanced,
          requires_restart,
          valid_options,
          unit
        FROM smc_simple_parameter_metadata
        ORDER BY category, display_order, display_name
      `
    );
    return NextResponse.json(result.rows ?? []);
  } catch (error) {
    console.error("Failed to load SMC metadata", error);
    return NextResponse.json(
      { error: "Failed to load SMC metadata" },
      { status: 500 }
    );
  }
}
