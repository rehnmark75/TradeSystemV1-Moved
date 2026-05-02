import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const FASTAPI_DEV_URL =
  process.env.FASTAPI_DEV_URL || "http://fastapi-dev:8000";

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${FASTAPI_DEV_URL}/orders/place-order`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-apim-gateway": "verified",
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const data: unknown = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
