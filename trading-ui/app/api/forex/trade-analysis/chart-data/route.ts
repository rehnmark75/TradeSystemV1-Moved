import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const FASTAPI_BASE = process.env.FASTAPI_DEV_URL || "http://fastapi-dev:8000";
const FASTAPI_API_KEY = process.env.FASTAPI_API_KEY || "436abe054a074894a0517e5172f0e5b6";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const tradeId = searchParams.get("tradeId");

  if (!tradeId) {
    return NextResponse.json({ error: "tradeId is required" }, { status: 400 });
  }

  try {
    const response = await fetch(
      `${FASTAPI_BASE}/api/trade-analysis/chart-data/${tradeId}`,
      {
        headers: {
          "X-APIM-Gateway": "verified",
          "X-API-KEY": FASTAPI_API_KEY,
        },
        cache: "no-store",
      }
    );

    const text = await response.text();
    if (!response.ok) {
      return NextResponse.json(
        { error: "Failed to load chart data", detail: text },
        { status: response.status }
      );
    }

    return NextResponse.json(JSON.parse(text));
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to load chart data", detail: String(error) },
      { status: 500 }
    );
  }
}
