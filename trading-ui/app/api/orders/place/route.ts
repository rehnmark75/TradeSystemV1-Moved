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

    // Resolve broker ticker (ARLO → ARLO.ny, AAPL → AAPL.nq)
    const exchangeResult = await client.query(
      `SELECT COALESCE(exchange, 'NASDAQ') as exchange FROM stock_instruments WHERE ticker = $1 LIMIT 1`,
      [ticker]
    );
    const exchange = exchangeResult.rows[0]?.exchange || "NASDAQ";
    const suffix = exchange.toUpperCase().includes("NYSE") ? ".ny" : ".nq";
    const brokerTicker = `${ticker}${suffix}`;

    // POST to RoboMarkets
    const formData = new URLSearchParams();
    formData.append("ticker", brokerTicker);
    formData.append("side", side);
    formData.append("type", order_type);
    formData.append("volume", String(Math.floor(quantity)));
    if (price && order_type === "limit") {
      formData.append("price", String(price));
    }
    formData.append("stop_loss", String(stop_loss));
    if (take_profit && take_profit > 0) {
      formData.append("take_profit", String(take_profit));
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

      const brokerText = await brokerRes.text().catch(() => "");
      let brokerData: Record<string, unknown> = {};
      try { brokerData = JSON.parse(brokerText); } catch { /* not JSON */ }

      if (!brokerRes.ok) {
        orderStatus = "rejected";
        errorMessage = String(
          brokerData?.msg || brokerData?.message || brokerData?.error || brokerText || `Broker returned ${brokerRes.status}`
        );
      } else {
        const dataObj = brokerData?.data as Record<string, unknown> | undefined;
        brokerOrderId = String(dataObj?.order_id || dataObj?.orderId || brokerData?.orderId || "");
      }
    } catch (err: unknown) {
      orderStatus = "rejected";
      errorMessage = err instanceof Error ? err.message : "Network error contacting broker";
    }

    // The broker ignores SL/TP sent during POST /orders creation.
    // We must follow up with a PUT to apply them:
    //   - Limit orders: PUT /orders/{order_id} (deal doesn't exist yet)
    //   - Market orders: PUT /deals/{deal_id} (order fills instantly, creates a deal)
    let slTpApplied = false;
    if (orderStatus === "submitted" && brokerOrderId) {
      const slTpForm = new URLSearchParams();
      slTpForm.append("stop_loss", String(stop_loss));
      if (take_profit && take_profit > 0) {
        slTpForm.append("take_profit", String(take_profit));
      }
      const putHeaders = {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json",
      };

      try {
        if (order_type === "limit") {
          // Set SL/TP on the pending order — they'll carry over when it fills
          const modifyRes = await fetch(
            `${API_URL}/accounts/${ACCOUNT_ID}/orders/${brokerOrderId}`,
            { method: "PUT", headers: putHeaders, body: slTpForm.toString() }
          );
          slTpApplied = modifyRes.ok;
          if (!modifyRes.ok) {
            const t = await modifyRes.text().catch(() => "");
            console.error(`Failed to set SL/TP on order ${brokerOrderId}: ${t}`);
          }
        } else {
          // Market order: wait for fill, then set SL/TP on the deal
          await new Promise((r) => setTimeout(r, 1500));

          const dealsRes = await fetch(`${API_URL}/accounts/${ACCOUNT_ID}/deals`, {
            headers: { Authorization: `Bearer ${API_KEY}`, Accept: "application/json" },
          });

          if (dealsRes.ok) {
            const dealsJson = await dealsRes.json() as Record<string, unknown>;
            const deals = ((dealsJson?.data || []) as Array<Record<string, unknown>>);

            // Find the most recent deal for our ticker (highest id)
            const matchingDeals = deals
              .filter((d) => d.ticker === brokerTicker)
              .sort((a, b) => String(b.id).localeCompare(String(a.id)));

            if (matchingDeals.length > 0) {
              const dealId = String(matchingDeals[0].id);
              const modifyRes = await fetch(
                `${API_URL}/accounts/${ACCOUNT_ID}/deals/${dealId}`,
                { method: "PUT", headers: putHeaders, body: slTpForm.toString() }
              );
              slTpApplied = modifyRes.ok;
              if (!modifyRes.ok) {
                const t = await modifyRes.text().catch(() => "");
                console.error(`Failed to set SL/TP on deal ${dealId}: ${t}`);
              }
            } else {
              console.warn(`No matching deal found for ${brokerTicker} to set SL/TP`);
            }
          }
        }
      } catch (slTpErr) {
        console.error("Failed to apply SL/TP:", slTpErr);
      }
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
      sl_tp_applied: slTpApplied,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    client.release();
  }
}
