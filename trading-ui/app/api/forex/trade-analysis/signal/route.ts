import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const FASTAPI_BASE =
  process.env.FASTAPI_DEV_URL || "http://fastapi-dev:8000";
const FASTAPI_API_KEY =
  process.env.FASTAPI_API_KEY || "436abe054a074894a0517e5172f0e5b6";

async function fetchTradeAnalysis(path: string) {
  const response = await fetch(`${FASTAPI_BASE}${path}`, {
    headers: {
      "X-APIM-Gateway": "verified",
      "X-API-KEY": FASTAPI_API_KEY
    },
    cache: "no-store"
  });

  const text = await response.text();
  if (!response.ok) {
    return { error: true, status: response.status, detail: text };
  }
  try {
    return JSON.parse(text);
  } catch {
    return { error: true, status: 500, detail: text };
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const tradeId = searchParams.get("tradeId");

  if (!tradeId) {
    return NextResponse.json({ error: "tradeId is required" }, { status: 400 });
  }

  const result = await fetchTradeAnalysis(`/api/trade-analysis/signal/${tradeId}`);
  if ((result as { error?: boolean }).error) {
    return NextResponse.json(
      { error: "Failed to load signal analysis", detail: result },
      { status: (result as { status?: number }).status ?? 500 }
    );
  }

  return NextResponse.json(result);
}
