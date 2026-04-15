import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const LONG_ONLY_CLAUDE_CONTEXT = {
  trading_perspective: "long_only",
  allowed_trade_side: "buy",
  disallowed_trade_side: "sell",
  instruction:
    "Analyze this stock strictly from a long-only perspective. The user can only take buy trades. Do not recommend short or sell-side entries. If the setup is unattractive for a long entry, respond with HOLD, AVOID, or equivalent no-trade guidance."
};

export async function POST(request: Request) {
  const body = await request.json();
  const baseUrl = process.env.FASTAPI_GENERAL_URL || "http://fastapi-general:8008";
  const requestBody =
    body && typeof body === "object"
      ? { ...LONG_ONLY_CLAUDE_CONTEXT, ...body }
      : LONG_ONLY_CLAUDE_CONTEXT;

  try {
    const res = await fetch(`${baseUrl}/claude/analyze-watchlist`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody)
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    return NextResponse.json({ error: "Failed to reach Claude service" }, { status: 502 });
  }
}
