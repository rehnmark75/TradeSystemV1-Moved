import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const env = searchParams.get("env") || "demo";
  const from = searchParams.get("from");
  const to = searchParams.get("to");
  const epic = searchParams.get("epic");
  const outcome = searchParams.get("outcome"); // "win" | "loss" | "be"
  const limit = Math.min(Number(searchParams.get("limit") || "200"), 500);

  const conditions: string[] = [
    "t.status IN ('closed', 'tracking')",
    "t.environment = $1",
  ];
  const params: unknown[] = [env];

  if (from) {
    params.push(from);
    conditions.push(`t.timestamp >= $${params.length}`);
  }
  if (to) {
    params.push(to);
    conditions.push(`t.timestamp < $${params.length}::date + INTERVAL '1 day'`);
  }
  if (epic) {
    params.push(epic);
    conditions.push(`t.symbol = $${params.length}`);
  }
  if (outcome === "win") {
    conditions.push("t.pips_gained > 0");
  } else if (outcome === "loss") {
    conditions.push("(t.pips_gained < 0 OR (t.pips_gained IS NULL AND t.profit_loss < 0))");
  } else if (outcome === "be") {
    conditions.push("(t.pips_gained = 0 OR (t.pips_gained IS NULL AND t.profit_loss = 0))");
  }

  params.push(limit);
  const limitParam = `$${params.length}`;

  try {
    const result = await forexPool.query(
      `SELECT
        t.id,
        t.symbol,
        t.direction,
        t.timestamp,
        t.closed_at,
        t.status,
        t.profit_loss,
        t.pnl_currency,
        t.entry_price,
        t.sl_price,
        t.initial_sl_price,
        t.tp_price,
        t.pips_gained,
        t.early_be_executed,
        t.moved_to_breakeven,
        t.moved_to_stage1,
        t.moved_to_stage2,
        t.stop_limit_changes_count,
        t.lifecycle_duration_minutes,
        t.is_scalp_trade,
        t.deal_id,
        a.strategy
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE ${conditions.join(" AND ")}
      ORDER BY t.timestamp DESC
      LIMIT ${limitParam}`,
      params
    );

    const trades = (result.rows ?? []).map((row) => ({
      ...row,
      profit_loss: row.profit_loss == null ? null : Number(row.profit_loss),
      pips_gained: row.pips_gained == null ? null : Number(row.pips_gained),
      pnl_display:
        row.profit_loss == null
          ? row.status === "tracking" ? "Open" : "-"
          : `${row.profit_loss >= 0 ? "+" : ""}${Number(row.profit_loss).toFixed(2)} ${
              row.pnl_currency ?? ""
            }`.trim(),
      stages_reached:
        (row.early_be_executed ? 1 : 0) +
        (row.moved_to_breakeven ? 1 : 0) +
        (row.moved_to_stage1 ? 1 : 0) +
        (row.moved_to_stage2 ? 1 : 0),
    }));

    return NextResponse.json({ trades });
  } catch (error) {
    console.error("Failed to load trade list", error);
    return NextResponse.json({ error: "Failed to load trade list" }, { status: 500 });
  }
}
