import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const epic = searchParams.get("epic");
  const from = searchParams.get("from");
  const to = searchParams.get("to");

  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }

  try {
    const params: (string | number)[] = [epic];
    let where = "WHERE tl.symbol = $1";

    if (from) {
      params.push(from);
      where += ` AND tl.timestamp >= $${params.length}`;
    }
    if (to) {
      params.push(to);
      where += ` AND tl.timestamp <= $${params.length}`;
    }

    const result = await forexPool.query(
      `
      SELECT
        tl.id,
        tl.timestamp,
        tl.entry_price,
        tl.direction,
        tl.profit_loss,
        tl.status,
        tl.pnl_currency,
        tl.environment,
        ah.strategy,
        ah.signal_type,
        ah.confidence_score
      FROM trade_log tl
      LEFT JOIN alert_history ah ON tl.alert_id = ah.id
      ${where}
      ORDER BY tl.timestamp ASC
      `,
      params
    );

    const trades = result.rows.map((row) => ({
      ...row,
      profit_loss: row.profit_loss == null ? null : Number(row.profit_loss),
      confidence_score:
        row.confidence_score == null ? null : Number(row.confidence_score),
    }));

    return NextResponse.json({ trades });
  } catch (error) {
    console.error("Failed to load chart trades", error);
    return NextResponse.json(
      { error: "Failed to load chart trades" },
      { status: 500 }
    );
  }
}
