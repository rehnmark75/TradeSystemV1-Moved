import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 30;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_DAYS;
  if (parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

const PENDING_STATUS = new Set(["pending", "pending_limit"]);
const REJECTED_STATUS = new Set(["limit_rejected", "limit_cancelled"]);

function formatProfitLoss(value: number | null, currency: string | null, status: string) {
  if (value == null) {
    const map: Record<string, string> = {
      tracking: "Open",
      limit_not_filled: "Not Filled",
      limit_rejected: "Rejected",
      limit_cancelled: "Cancelled",
      pending: "Pending",
      pending_limit: "Pending"
    };
    return map[status] ?? "Pending";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)} ${currency ?? ""}`.trim();
}

function deriveTradeResult(value: number | null, status: string) {
  if (value != null) {
    if (value > 0) return "WIN";
    if (value < 0) return "LOSS";
    return "BREAKEVEN";
  }
  if (PENDING_STATUS.has(status)) return "PENDING";
  if (status === "tracking") return "OPEN";
  if (status === "limit_not_filled") return "EXPIRED";
  if (REJECTED_STATUS.has(status)) return "REJECTED";
  return "PENDING";
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    const result = await forexPool.query(
      `
      SELECT
        t.id,
        t.symbol,
        t.entry_price,
        t.direction,
        t.timestamp,
        t.status,
        t.profit_loss,
        t.pnl_currency,
        a.strategy
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      ORDER BY t.timestamp DESC
      `,
      [since]
    );

    const trades = (result.rows ?? []).map((row) => {
      const status = row.status ?? "pending";
      const profitLoss = row.profit_loss == null ? null : Number(row.profit_loss);
      return {
        ...row,
        profit_loss: profitLoss,
        trade_result: deriveTradeResult(profitLoss, status),
        profit_loss_formatted: formatProfitLoss(profitLoss, row.pnl_currency, status)
      };
    });

    return NextResponse.json({ trades });
  } catch (error) {
    console.error("Failed to load forex trades", error);
    return NextResponse.json(
      { error: "Failed to load forex trades" },
      { status: 500 }
    );
  }
}
