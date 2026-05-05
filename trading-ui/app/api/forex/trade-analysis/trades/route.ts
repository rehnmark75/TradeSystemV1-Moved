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
    "status IN ('closed', 'tracking')",
    "environment = $1",
  ];
  const params: unknown[] = [env];

  if (from) {
    params.push(from);
    conditions.push(`timestamp >= $${params.length}`);
  }
  if (to) {
    params.push(to);
    conditions.push(`timestamp < $${params.length}::date + INTERVAL '1 day'`);
  }
  if (epic) {
    params.push(epic);
    conditions.push(`symbol = $${params.length}`);
  }
  if (outcome === "win") {
    conditions.push("pips_gained > 0");
  } else if (outcome === "loss") {
    conditions.push("(pips_gained < 0 OR (pips_gained IS NULL AND profit_loss < 0))");
  } else if (outcome === "be") {
    conditions.push("(pips_gained = 0 OR (pips_gained IS NULL AND profit_loss = 0))");
  }

  params.push(limit);
  const limitParam = `$${params.length}`;

  try {
    const result = await forexPool.query(
      `SELECT
        id,
        symbol,
        direction,
        timestamp,
        closed_at,
        status,
        profit_loss,
        pnl_currency,
        entry_price,
        sl_price,
        initial_sl_price,
        tp_price,
        pips_gained,
        early_be_executed,
        moved_to_breakeven,
        moved_to_stage1,
        moved_to_stage2,
        stop_limit_changes_count,
        lifecycle_duration_minutes,
        is_scalp_trade,
        deal_id
      FROM trade_log
      WHERE ${conditions.join(" AND ")}
      ORDER BY timestamp DESC
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
