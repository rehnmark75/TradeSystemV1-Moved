import { NextResponse } from "next/server";
import { KAMA2_GLOBAL_DEFAULTS, STRATEGY_LABEL } from "./common";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ ...KAMA2_GLOBAL_DEFAULTS, strategy_name: STRATEGY_LABEL });
}

export async function PATCH() {
  return NextResponse.json(
    { error: "KAMA_V2 global defaults are code-defined. Use pair overrides for tradable settings." },
    { status: 400 },
  );
}
