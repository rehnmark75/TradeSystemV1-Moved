import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = await request.json();
  const baseUrl = process.env.FASTAPI_GENERAL_URL || "http://fastapi-general:8008";

  try {
    const res = await fetch(`${baseUrl}/claude/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    return NextResponse.json({ error: "Failed to reach Claude service" }, { status: 502 });
  }
}
