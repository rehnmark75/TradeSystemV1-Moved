import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const API_KEY = process.env.ROBOMARKETS_API_KEY || "";
const ACCOUNT_ID = process.env.ROBOMARKETS_ACCOUNT_ID || "";
const API_URL = process.env.ROBOMARKETS_API_URL || "https://api.stockstrader.com/api/v1";
const PRICE_STEP_ATTEMPTS = [0.01, 0.05, 0.1];
const MAX_LEVEL_ATTEMPTS = 3; // Initial broker-rounded level + max two retries.

interface PlaceOrderBody {
  ticker: string;
  watchlist_name?: string;
  side: "buy" | "sell";
  order_type: "market" | "limit";
  quantity: number;
  price?: number;
  stop_loss: number;
  take_profit?: number;
  trade_ready_override?: boolean;
  signal_id?: number;
  breakeven_enabled?: boolean;
  breakeven_trigger_usd?: number;
}

type BrokerLevelAttempt = {
  price?: number;
  stopLoss: number;
  takeProfit?: number;
  step: number;
  widenSteps: number;
  label: string;
};

type BrokerSubmitResult = {
  ok: boolean;
  orderId: string | null;
  error: string | null;
  raw: Record<string, unknown>;
  status: number;
};

type SlTpApplyResult = {
  ok: boolean;
  attempt: BrokerLevelAttempt;
  error: string | null;
};

type ProtectionAction = {
  attempted: boolean;
  action: "none" | "cancel_order" | "close_deal";
  ok: boolean;
  error: string | null;
};

const roundToStep = (value: number, step: number, direction: "nearest" | "up" | "down") => {
  const factor = value / step;
  const rounded =
    direction === "up" ? Math.ceil(factor) * step :
    direction === "down" ? Math.floor(factor) * step :
    Math.round(factor) * step;
  return Number(rounded.toFixed(4));
};

const formatLevel = (value: number | undefined) => {
  if (value === undefined || value === null) return undefined;
  return Number(value.toFixed(4));
};

const levelChanged = (a: number | undefined, b: number | undefined) => {
  if (a === undefined && b === undefined) return false;
  if (a === undefined || b === undefined) return true;
  return Math.abs(a - b) > 0.00001;
};

const isLevelError = (message: string | null) => {
  if (!message) return false;
  return /stop|take|profit|loss|level|price|incorrect|invalid|distance|step/i.test(message);
};

const buildLevelAttempts = (
  side: "buy" | "sell",
  orderType: "market" | "limit",
  price: number | undefined,
  stopLoss: number,
  takeProfit: number | undefined,
): BrokerLevelAttempt[] => {
  const attempts: BrokerLevelAttempt[] = [];
  const seen = new Set<string>();
  const slDirection = side === "buy" ? "down" : "up";
  const tpDirection = side === "buy" ? "up" : "down";

  for (const step of PRICE_STEP_ATTEMPTS) {
    const attempt: BrokerLevelAttempt = {
      price: orderType === "limit" && price ? formatLevel(price) : undefined,
      stopLoss: roundToStep(stopLoss, step, slDirection),
      takeProfit: takeProfit && takeProfit > 0 ? roundToStep(takeProfit, step, tpDirection) : undefined,
      step,
      widenSteps: 0,
      label: `${step.toFixed(2)} rounded`,
    };

    if (attempt.stopLoss <= 0 || (attempt.takeProfit !== undefined && attempt.takeProfit <= 0)) {
      continue;
    }
    if (side === "buy" && attempt.price && attempt.stopLoss >= attempt.price) {
      continue;
    }
    if (side === "sell" && attempt.price && attempt.stopLoss <= attempt.price) {
      continue;
    }

    const key = `${attempt.price ?? ""}:${attempt.stopLoss}:${attempt.takeProfit ?? ""}`;
    if (!seen.has(key)) {
      seen.add(key);
      attempts.push(attempt);
    }
  }

  return attempts.slice(0, MAX_LEVEL_ATTEMPTS);
};

const buildSlTpForm = (attempt: BrokerLevelAttempt) => {
  const form = new URLSearchParams();
  form.append("stop_loss", String(attempt.stopLoss));
  if (attempt.takeProfit && attempt.takeProfit > 0) {
    form.append("take_profit", String(attempt.takeProfit));
  }
  return form;
};

const readBrokerResponse = async (response: Response): Promise<{ data: Record<string, unknown>; text: string }> => {
  const text = await response.text().catch(() => "");
  let data: Record<string, unknown> = {};
  try { data = JSON.parse(text); } catch { /* not JSON */ }
  return { data, text };
};

const brokerErrorMessage = (data: Record<string, unknown>, text: string, status: number) => {
  return String(data?.msg || data?.message || data?.error || text || `Broker returned ${status}`);
};

const submitBrokerOrder = async (
  brokerTicker: string,
  side: "buy" | "sell",
  orderType: "market" | "limit",
  quantity: number,
  attempt: BrokerLevelAttempt,
): Promise<BrokerSubmitResult> => {
  const formData = new URLSearchParams();
  formData.append("ticker", brokerTicker);
  formData.append("side", side);
  formData.append("type", orderType);
  formData.append("volume", String(Math.floor(quantity)));
  if (attempt.price && orderType === "limit") {
    formData.append("price", String(attempt.price));
  }
  formData.append("stop_loss", String(attempt.stopLoss));
  if (attempt.takeProfit && attempt.takeProfit > 0) {
    formData.append("take_profit", String(attempt.takeProfit));
  }

  const brokerRes = await fetch(`${API_URL}/accounts/${ACCOUNT_ID}/orders`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "application/json",
    },
    body: formData.toString(),
  });

  const { data, text } = await readBrokerResponse(brokerRes);
  if (!brokerRes.ok || data?.code === "error") {
    return {
      ok: false,
      orderId: null,
      error: brokerErrorMessage(data, text, brokerRes.status),
      raw: data,
      status: brokerRes.status,
    };
  }

  const dataObj = data?.data as Record<string, unknown> | undefined;
  return {
    ok: true,
    orderId: String(dataObj?.order_id || dataObj?.orderId || data?.orderId || ""),
    error: null,
    raw: data,
    status: brokerRes.status,
  };
};

const deleteBrokerResource = async (target: "orders" | "deals", id: string): Promise<{ ok: boolean; error: string | null }> => {
  const res = await fetch(`${API_URL}/accounts/${ACCOUNT_ID}/${target}/${id}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      Accept: "application/json",
    },
  });

  if (res.ok) {
    return { ok: true, error: null };
  }

  const { data, text } = await readBrokerResponse(res);
  return { ok: false, error: brokerErrorMessage(data, text, res.status) };
};

export async function POST(request: Request) {
  let body: PlaceOrderBody;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const {
    ticker,
    watchlist_name,
    side,
    order_type,
    quantity,
    price,
    stop_loss,
    take_profit,
    trade_ready_override,
    signal_id,
    breakeven_enabled = true,
    breakeven_trigger_usd = 10,
  } = body;

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
  if (breakeven_enabled && (!breakeven_trigger_usd || breakeven_trigger_usd <= 0)) {
    return NextResponse.json({ error: "breakeven_trigger_usd must be positive" }, { status: 400 });
  }

  if (!API_KEY || !ACCOUNT_ID) {
    return NextResponse.json({ error: "Broker credentials not configured" }, { status: 500 });
  }

  const client = await pool.connect();
  try {
    // Check trade_ready gate
    if (!trade_ready_override) {
      if (watchlist_name) {
        const trResult = await client.query(
          `SELECT trade_ready FROM stock_watchlist_results
           WHERE ticker = $1 AND watchlist_name = $2 AND status = 'active'
           ORDER BY scan_date DESC LIMIT 1`,
          [ticker, watchlist_name]
        );
        if (trResult.rows.length > 0 && !trResult.rows[0].trade_ready) {
          return NextResponse.json(
            { error: `${ticker} is not trade-ready. Check the override box to place anyway.` },
            { status: 400 }
          );
        }
      } else {
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
    }

    // Resolve broker ticker (ARLO → ARLO.ny, AAPL → AAPL.nq)
    const exchangeResult = await client.query(
      `SELECT COALESCE(exchange, 'NASDAQ') as exchange FROM stock_instruments WHERE ticker = $1 LIMIT 1`,
      [ticker]
    );
    const exchange = exchangeResult.rows[0]?.exchange || "NASDAQ";
    const suffix = exchange.toUpperCase().includes("NYSE") ? ".ny" : ".nq";
    const brokerTicker = `${ticker}${suffix}`;
    const levelAttempts = buildLevelAttempts(side, order_type, price, stop_loss, take_profit);

    if (!levelAttempts.length) {
      return NextResponse.json({ error: "No valid broker level attempts could be built" }, { status: 400 });
    }

    // POST to RoboMarkets
    let brokerOrderId: string | null = null;
    let brokerDealId: string | null = null;
    let orderStatus: "submitted" | "rejected" = "submitted";
    let errorMessage: string | null = null;
    let acceptedAttempt = levelAttempts[0];
    const levelAdjustmentAttempts: Array<{
      stage: "submit" | "apply";
      label: string;
      stop_loss: number;
      take_profit: number | null;
      error?: string | null;
    }> = [];

    try {
      for (let i = 0; i < levelAttempts.length; i += 1) {
        const attempt = levelAttempts[i];
        const submit = await submitBrokerOrder(brokerTicker, side, order_type, quantity, attempt);

        if (submit.ok) {
          acceptedAttempt = attempt;
          brokerOrderId = submit.orderId;
          break;
        }

        levelAdjustmentAttempts.push({
          stage: "submit",
          label: attempt.label,
          stop_loss: attempt.stopLoss,
          take_profit: attempt.takeProfit ?? null,
          error: submit.error,
        });
        errorMessage = submit.error;

        if (!isLevelError(submit.error)) {
          break;
        }
      }

      if (!brokerOrderId) {
        orderStatus = "rejected";
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
    let slTpErrorMessage: string | null = null;
    let protectionAction: ProtectionAction = {
      attempted: false,
      action: "none",
      ok: false,
      error: null,
    };
    if (orderStatus === "submitted" && brokerOrderId) {
      const putHeaders = {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json",
      };
      const orderedApplyAttempts = [
        acceptedAttempt,
        ...levelAttempts.filter((a) =>
          levelChanged(a.stopLoss, acceptedAttempt.stopLoss) ||
          levelChanged(a.takeProfit, acceptedAttempt.takeProfit)
        ),
      ];

      const applySlTp = async (target: "order" | "deal", id: string): Promise<SlTpApplyResult> => {
        for (const attempt of orderedApplyAttempts) {
          const endpoint = target === "order"
            ? `${API_URL}/accounts/${ACCOUNT_ID}/orders/${id}`
            : `${API_URL}/accounts/${ACCOUNT_ID}/deals/${id}`;
          const modifyRes = await fetch(endpoint, {
            method: "PUT",
            headers: putHeaders,
            body: buildSlTpForm(attempt).toString(),
          });

          if (modifyRes.ok) {
            return { ok: true, attempt, error: null };
          }

          const { data, text } = await readBrokerResponse(modifyRes);
          const message = brokerErrorMessage(data, text, modifyRes.status);
          levelAdjustmentAttempts.push({
            stage: "apply",
            label: attempt.label,
            stop_loss: attempt.stopLoss,
            take_profit: attempt.takeProfit ?? null,
            error: message,
          });

          if (!isLevelError(message)) {
            return { ok: false, attempt, error: message };
          }
        }

        const last = orderedApplyAttempts[orderedApplyAttempts.length - 1] || acceptedAttempt;
        return { ok: false, attempt: last, error: "Broker rejected all SL/TP level attempts" };
      };

      try {
        if (order_type === "limit") {
          // Set SL/TP on the pending order — they'll carry over when it fills
          const applied = await applySlTp("order", brokerOrderId);
          slTpApplied = applied.ok;
          acceptedAttempt = applied.attempt;
          slTpErrorMessage = applied.error;
          if (!applied.ok) {
            console.error(`Failed to set SL/TP on order ${brokerOrderId}: ${applied.error}`);
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
              brokerDealId = dealId;
              const applied = await applySlTp("deal", dealId);
              slTpApplied = applied.ok;
              acceptedAttempt = applied.attempt;
              slTpErrorMessage = applied.error;
              if (!applied.ok) {
                console.error(`Failed to set SL/TP on deal ${dealId}: ${applied.error}`);
              }
            } else {
              slTpErrorMessage = `No matching deal found for ${brokerTicker} to set SL/TP`;
              console.warn(slTpErrorMessage);
            }
          } else {
            const { data, text } = await readBrokerResponse(dealsRes);
            slTpErrorMessage = brokerErrorMessage(data, text, dealsRes.status);
          }
        }
      } catch (slTpErr) {
        slTpErrorMessage = slTpErr instanceof Error ? slTpErr.message : "Failed to apply SL/TP";
        console.error("Failed to apply SL/TP:", slTpErr);
      }

      if (!slTpApplied) {
        if (order_type === "limit") {
          const cleanup = await deleteBrokerResource("orders", brokerOrderId);
          protectionAction = {
            attempted: true,
            action: "cancel_order",
            ok: cleanup.ok,
            error: cleanup.error,
          };
          orderStatus = "rejected";
          errorMessage = cleanup.ok
            ? `SL/TP could not be applied, so broker order ${brokerOrderId} was cancelled. ${slTpErrorMessage || ""}`.trim()
            : `SL/TP could not be applied and broker order ${brokerOrderId} could not be cancelled. Manual broker check required. ${cleanup.error || slTpErrorMessage || ""}`.trim();
        } else if (brokerDealId) {
          const cleanup = await deleteBrokerResource("deals", brokerDealId);
          protectionAction = {
            attempted: true,
            action: "close_deal",
            ok: cleanup.ok,
            error: cleanup.error,
          };
          orderStatus = "rejected";
          errorMessage = cleanup.ok
            ? `SL/TP could not be applied, so broker deal ${brokerDealId} was closed. ${slTpErrorMessage || ""}`.trim()
            : `SL/TP could not be applied and broker deal ${brokerDealId} could not be closed. Manual broker check required. ${cleanup.error || slTpErrorMessage || ""}`.trim();
        } else {
          orderStatus = "rejected";
          errorMessage = `SL/TP could not be confirmed and no broker deal id was available to close. Manual broker check required. ${slTpErrorMessage || ""}`.trim();
        }
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
        acceptedAttempt.price || price || null,
        acceptedAttempt.stopLoss,
        acceptedAttempt.takeProfit || null,
        orderStatus,
        errorMessage || slTpErrorMessage,
      ]
    );

    const dbOrderId = insertResult.rows[0]?.id;

    let breakevenMonitorId: number | null = null;
    if (orderStatus === "submitted" && breakeven_enabled && dbOrderId) {
      await client.query(`
        CREATE TABLE IF NOT EXISTS stock_breakeven_monitors (
          id BIGSERIAL PRIMARY KEY,
          stock_order_id BIGINT REFERENCES stock_orders(id),
          robomarkets_order_id VARCHAR(100),
          robomarkets_deal_id VARCHAR(100),
          ticker VARCHAR(20) NOT NULL,
          broker_ticker VARCHAR(50),
          side VARCHAR(10) NOT NULL,
          quantity DECIMAL(12,4),
          entry_price DECIMAL(12,4),
          initial_stop_loss DECIMAL(12,4),
          breakeven_stop_price DECIMAL(12,4),
          take_profit DECIMAL(12,4),
          trigger_profit_usd DECIMAL(12,4) DEFAULT 10.00,
          poll_interval_seconds INTEGER DEFAULT 300,
          status VARCHAR(30) DEFAULT 'pending_fill',
          last_profit_usd DECIMAL(12,4),
          last_checked_at TIMESTAMP,
          moved_at TIMESTAMP,
          error_message TEXT,
          created_at TIMESTAMP DEFAULT NOW(),
          updated_at TIMESTAMP DEFAULT NOW()
        )
      `);
      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_stock_breakeven_monitors_active
        ON stock_breakeven_monitors(status, last_checked_at)
        WHERE status IN ('pending_fill', 'monitoring')
      `);

      const entryPrice = order_type === "limit" ? (acceptedAttempt.price || price) : null;
      const monitorResult = await client.query(
        `INSERT INTO stock_breakeven_monitors
          (stock_order_id, robomarkets_order_id, robomarkets_deal_id, ticker, broker_ticker,
           side, quantity, entry_price, initial_stop_loss, breakeven_stop_price, take_profit,
           trigger_profit_usd, poll_interval_seconds, status)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $8, $10, $11, 300, $12)
         RETURNING id`,
        [
          dbOrderId,
          brokerOrderId,
          brokerDealId,
          ticker,
          brokerTicker,
          side,
          quantity,
          entryPrice,
          acceptedAttempt.stopLoss,
          acceptedAttempt.takeProfit || null,
          breakeven_trigger_usd,
          brokerDealId ? "monitoring" : "pending_fill",
        ]
      );
      breakevenMonitorId = monitorResult.rows[0]?.id ?? null;
    }

    if (orderStatus === "rejected") {
      return NextResponse.json(
        {
          error: errorMessage,
          db_order_id: dbOrderId,
          status: "rejected",
          requested_levels: {
            price: price || null,
            stop_loss,
            take_profit: take_profit || null,
          },
          level_adjustment_attempts: levelAdjustmentAttempts,
          protection_action: protectionAction,
        },
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
      price: acceptedAttempt.price || price || null,
      stop_loss: acceptedAttempt.stopLoss,
      take_profit: acceptedAttempt.takeProfit || null,
      sl_tp_applied: slTpApplied,
      sl_tp_error: slTpErrorMessage,
      protection_action: protectionAction,
      level_adjusted:
        levelChanged(stop_loss, acceptedAttempt.stopLoss) ||
        levelChanged(take_profit, acceptedAttempt.takeProfit) ||
        levelChanged(price, acceptedAttempt.price),
      requested_levels: {
        price: price || null,
        stop_loss,
        take_profit: take_profit || null,
      },
      broker_levels: {
        price: acceptedAttempt.price || price || null,
        stop_loss: acceptedAttempt.stopLoss,
        take_profit: acceptedAttempt.takeProfit || null,
        step: acceptedAttempt.step,
        label: acceptedAttempt.label,
      },
      level_adjustment_attempts: levelAdjustmentAttempts,
      breakeven_monitor_id: breakevenMonitorId,
      breakeven_trigger_usd: breakevenMonitorId ? breakeven_trigger_usd : null,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    client.release();
  }
}
