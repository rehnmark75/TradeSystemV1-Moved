import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_HOURS = 1;
const ALLOWED_HOURS = new Set([1, 3, 8, 24, 72, 168]);

function parseHours(value: string | null) {
  if (!value) return DEFAULT_HOURS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_HOURS;
  if (!ALLOWED_HOURS.has(parsed)) return DEFAULT_HOURS;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const hours = parseHours(searchParams.get("hours"));
  const env = searchParams.get("env") || "demo";

  try {
    const result = await forexPool.query(
      `
      SELECT *
      FROM alert_history
      WHERE alert_timestamp >= NOW() - ($1::int || ' hours')::interval
        AND environment = $2
      ORDER BY alert_timestamp DESC
      `,
      [hours, env]
    );

    return NextResponse.json({
      hours,
      total: result.rows?.length ?? 0,
      alerts: result.rows ?? []
    });
  } catch (error) {
    console.error("Failed to load alert data", error);
    return NextResponse.json({ error: "Failed to load alert data" }, { status: 500 });
  }
}
