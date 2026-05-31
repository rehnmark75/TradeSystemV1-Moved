import { NextResponse } from "next/server";
import { KAMA2_METADATA } from "../common";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(
    KAMA2_METADATA.map(([parameterName, displayName, category, dataType, description, unit], index) => ({
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
      is_advanced: ["enabled_epic", "kama_period", "base_confidence", "max_confidence"].includes(parameterName),
      requires_restart: false,
      valid_options: null,
      unit,
    })),
  );
}
