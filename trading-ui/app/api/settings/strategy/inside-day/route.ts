import { NextResponse } from "next/server";
import { INSIDE_DAY_GLOBAL_DEFAULTS, STRATEGY_LABEL } from "./common";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ ...INSIDE_DAY_GLOBAL_DEFAULTS, strategy_name: STRATEGY_LABEL });
}

export async function PATCH() {
  return NextResponse.json(
    { error: "INSIDE_DAY global defaults are code-defined. Use pair overrides for tradable settings." },
    { status: 400 },
  );
}
