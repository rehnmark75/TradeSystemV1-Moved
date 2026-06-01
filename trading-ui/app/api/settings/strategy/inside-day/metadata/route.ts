import { NextResponse } from "next/server";
import { INSIDE_DAY_METADATA } from "../common";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(
    INSIDE_DAY_METADATA.map(([parameterName, displayName, category, dataType, description, unit], index) => ({
      id: index + 1,
      parameter_name: parameterName,
      display_name: displayName,
      category,
      subcategory: null,
      data_type: dataType,
      min_value: null,
      max_value: null,
      default_value: null,
      description,
      help_text: null,
      display_order: index + 1,
      is_advanced: ["weekly_bias_q", "atr_buffer_fraction", "base_confidence"].includes(parameterName),
      requires_restart: false,
      valid_options: null,
      unit,
    })),
  );
}
