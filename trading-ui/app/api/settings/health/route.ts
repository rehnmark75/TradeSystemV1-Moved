import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  const startedAt = Date.now();
  try {
    await strategyConfigPool.query("SELECT 1");
    return NextResponse.json({
      status: "connected",
      checked_at: new Date().toISOString(),
      latency_ms: Date.now() - startedAt
    });
  } catch (error) {
    console.error("Strategy config DB health check failed", error);
    return NextResponse.json(
      {
        status: "disconnected",
        checked_at: new Date().toISOString(),
        latency_ms: Date.now() - startedAt
      },
      { status: 500 }
    );
  }
}
