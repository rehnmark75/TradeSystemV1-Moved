import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const FASTAPI_BASE =
  process.env.FASTAPI_DEV_URL || "http://fastapi-dev:8000";
const FASTAPI_API_KEY =
  process.env.FASTAPI_API_KEY || "436abe054a074894a0517e5172f0e5b6";

async function fetchOutcome(path: string, days: number) {
  const response = await fetch(`${FASTAPI_BASE}${path}?days=${days}`, {
    headers: {
      "X-APIM-Gateway": "verified",
      "X-API-KEY": FASTAPI_API_KEY
    },
    cache: "no-store"
  });

  if (!response.ok) {
    const detail = await response.text();
    return { error: true, status: response.status, detail };
  }

  return response.json();
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = Number(searchParams.get("days") ?? 7);

  try {
    const [summary, winRate, suggestions] = await Promise.all([
      fetchOutcome("/api/rejection-outcomes/summary", days),
      fetchOutcome("/api/rejection-outcomes/win-rate-by-stage", days),
      fetchOutcome("/api/rejection-outcomes/parameter-suggestions", days)
    ]);

    if (summary.error || winRate.error || suggestions.error) {
      return NextResponse.json(
        { error: "Failed to load rejection outcomes" },
        { status: 502 }
      );
    }

    return NextResponse.json({
      summary,
      win_rate_by_stage: winRate,
      suggestions
    });
  } catch (error) {
    console.error("Failed to load rejection outcomes", error);
    return NextResponse.json(
      { error: "Failed to load rejection outcomes" },
      { status: 500 }
    );
  }
}
