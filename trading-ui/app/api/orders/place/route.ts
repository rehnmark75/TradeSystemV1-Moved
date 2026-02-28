import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const API_KEY = process.env.ROBOMARKETS_API_KEY || "";
const ACCOUNT_ID = process.env.ROBOMARKETS_ACCOUNT_ID || "";
const API_URL = process.env.ROBOMARKETS_API_URL || "https://api.stockstrader.com/api/v1";

interface PlaceOrderBody {
  ticker: string;
  side: "buy" | "sell";
  order_type: "market" | "limit";
  quantity: number;
  price?: number;
  stop_loss: number;
  take_profit?: number;
  trade_ready_override?: boolean;
  signal_id?: number;
}

export async function POST(request: Request) {
  let body: PlaceOrderBody;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { ticker, side, order_type, quantity, price, stop_loss, take_profit, trade_ready_override, signal_id } = body;

  // Validation
  if (!ticker || typeof ticker !== "string") {
    return NextResponse.json({ error: "ticker is required" }, { status: 400 });
  }
  if (!["buy", "sell"].includes(side)) {
    return NextResponse.json({ error: "side must be 'buy' or 'sell'" }, { status: 400 });
  }
  if (!["market", "limit"].includes(order_type)) {
    return NextResponse.json({ error: "order_type must be 'market' or 'limit'" }, { status: 400 });
  }
  if (!quantity || quantity <= 0) {
    return NextResponse.json({ error: "quantity must be positive" }, { status: 400 });
  }
  if (!stop_loss || stop_loss <= 0) {
    return NextResponse.json({ error: "stop_loss is required and must be positive" }, { status: 400 });
  }
  if (order_type === "limit" && (!price || price <= 0)) {
    return NextResponse.json({ error: "price is required for limit orders" }, { status: 400 });
  }
  if (side === "buy" && price && stop_loss >= price) {
    return NextResponse.json({ error: "stop_loss must be below price for buy orders" }, { status: 400 });
  }

  if (!API_KEY || !ACCOUNT_ID) {
    return NextResponse.json({ error: "Broker credentials not configured" }, { status: 500 });
  }

  const client = await pool.connect();
  try {
    // Check trade_ready gate
    if (!trade_ready_override) {
      const trResult = await client.query(
        `SELECT trade_ready FROM stock_watchlist_results
         WHERE ticker = $1 AND status = 'active'
         ORDER BY scan_date DESC LIMIT 1`,
        [ticker]
      );
      if (trResult.rows.length > 0 && !trResult.rows[0].trade_ready) {
        return NextResponse.json(
          { error: `${ticker} is not trade-ready. Check the override box to place anyway.` },
          { status: 400 }
        );
      }
    }

    // POST to RoboMarkets
    const formData = new URLSearchParams();
    formData.append("ticker", ticker);
    formData.append("side", side);
    formData.append("type", order_type);
    formData.append("quantity", String(quantity));
    if (price && order_type === "limit") {
      formData.append("price", String(price));
    }
    formData.append("stopLoss", String(stop_loss));
    if (take_profit && take_profit > 0) {
      formData.append("takeProfit", String(take_profit));
    }

    let brokerOrderId: string | null = null;
    let orderStatus: "submitted" | "rejected" = "submitted";
    let errorMessage: string | null = null;

    try {
      const brokerRes = await fetch(`${API_URL}/accounts/${ACCOUNT_ID}/orders`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${API_KEY}`,
          "Content-Type": "application/x-www-form-urlencoded",
          Accept: "application/json",
        },
        body: formData.toString(),
      });

      const brokerData = await brokerRes.json().catch(() => ({}));

      if (!brokerRes.ok) {
        orderStatus = "rejected";
        errorMessage = brokerData?.message || brokerData?.error || `Broker returned ${brokerRes.status}`;
      } else {
        brokerOrderId = brokerData?.orderId ? String(brokerData.orderId) : null;
      }
    } catch (err: unknown) {
      orderStatus = "rejected";
      errorMessage = err instanceof Error ? err.message : "Network error contacting broker";
    }

    // Write audit trail to stock_orders
    const insertResult = await client.query(
      `INSERT INTO stock_orders
        (signal_id, robomarkets_order_id, ticker, order_type, side, quantity, price, stop_loss, take_profit, status, error_message)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
       RETURNING id`,
      [
        signal_id || null,
        brokerOrderId,
        ticker,
        order_type,
        side,
        quantity,
        price || null,
        stop_loss,
        take_profit || null,
        orderStatus,
        errorMessage,
      ]
    );

    const dbOrderId = insertResult.rows[0]?.id;

    if (orderStatus === "rejected") {
      return NextResponse.json(
        { error: errorMessage, db_order_id: dbOrderId, status: "rejected" },
        { status: 502 }
      );
    }

    return NextResponse.json({
      robomarkets_order_id: brokerOrderId,
      db_order_id: dbOrderId,
      status: "submitted",
      ticker,
      side,
      order_type,
      quantity,
      price: price || null,
      stop_loss,
      take_profit: take_profit || null,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    client.release();
  }
}
