import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { event_type, user, page, details } = body as {
    event_type?: string;
    user?: string;
    page?: string;
    details?: Record<string, unknown>;
  };

  if (!event_type) {
    return NextResponse.json(
      { error: "event_type is required" },
      { status: 400 }
    );
  }

  try {
    await strategyConfigPool.query(
      `
        CREATE TABLE IF NOT EXISTS settings_telemetry (
          id SERIAL PRIMARY KEY,
          event_type VARCHAR(50) NOT NULL,
          username VARCHAR(100),
          page TEXT,
          details JSONB,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
      `
    );

    await strategyConfigPool.query(
      `
        INSERT INTO settings_telemetry (event_type, username, page, details)
        VALUES ($1, $2, $3, $4)
      `,
      [event_type, user ?? null, page ?? null, details ?? null]
    );

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to log telemetry", error);
    return NextResponse.json(
      { error: "Failed to log telemetry" },
      { status: 500 }
    );
  }
}
