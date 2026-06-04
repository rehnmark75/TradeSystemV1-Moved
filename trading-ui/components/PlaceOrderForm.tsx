"use client";

import { useState } from "react";

const BUDGET = 500;

const num = (v: unknown) => {
  if (v === null || v === undefined || v === "") return 0;
  const n = Number(v);
  return Number.isNaN(n) ? 0 : n;
};

type Props = {
  ticker: string;
  entryPrice: number | null;
  signalId?: number;
};

// Self-contained order form. Mirrors the inline order panel on the signals
// page, but with its own local state so it can be dropped onto any page
// (deep-dive, etc.). Posts to the absolute /trading/api/orders/place so it
// works regardless of the host route's depth.
export default function PlaceOrderForm({ ticker, entryPrice, signalId }: Props) {
  const entry = num(entryPrice);
  const [open, setOpen] = useState(false);
  const [orderType, setOrderType] = useState<"limit" | "market">("limit");
  const [price, setPrice] = useState<number>(Number(entry.toFixed(2)));
  const [quantity, setQuantity] = useState<number>(entry > 0 ? Math.floor(BUDGET / entry) : 0);
  const [stopLoss, setStopLoss] = useState<number>(entry > 0 ? Number((entry * 0.97).toFixed(2)) : 0);
  const [takeProfit, setTakeProfit] = useState<number>(entry > 0 ? Number((entry * 1.05).toFixed(2)) : 0);
  const [override, setOverride] = useState(false);
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const refPrice = orderType === "limit" ? price : entry;
  const posValue = quantity * refPrice;
  const riskPerShare = refPrice - stopLoss;
  const rrRatio =
    riskPerShare > 0 && takeProfit > 0 && refPrice > 0
      ? ((takeProfit - refPrice) / riskPerShare).toFixed(2)
      : "-";
  const canReview = quantity > 0 && stopLoss > 0 && (orderType === "market" || price > 0);

  const updatePrice = (p: number) => {
    setPrice(p);
    if (p > 0) setQuantity(Math.floor(BUDGET / p));
  };

  const placeOrder = async () => {
    setLoading(true);
    setMessage(null);
    try {
      const res = await fetch(`/trading/api/orders/place`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker,
          side: "buy",
          order_type: orderType,
          quantity,
          price: orderType === "limit" ? price : undefined,
          stop_loss: stopLoss,
          take_profit: takeProfit > 0 ? takeProfit : undefined,
          trade_ready_override: override,
          signal_id: signalId,
          breakeven_enabled: true,
          breakeven_trigger_usd: 10,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setMessage({ type: "error", text: data?.error || "Order failed" });
      } else {
        const adjustedText = data.level_adjusted && data.broker_levels
          ? `, adjusted SL ${Number(data.broker_levels.stop_loss).toFixed(2)}${data.broker_levels.take_profit ? ` / TP ${Number(data.broker_levels.take_profit).toFixed(2)}` : ""}`
          : "";
        setMessage({
          type: "success",
          text: `Order ${data.status}! Broker ID: ${data.robomarkets_order_id || "pending"}, DB #${data.db_order_id}${data.breakeven_monitor_id ? `, BE monitor #${data.breakeven_monitor_id}` : ""}${adjustedText}`,
        });
        setConfirming(false);
      }
    } catch {
      setMessage({ type: "error", text: "Network error placing order" });
    }
    setLoading(false);
  };

  return (
    <div className="order-panel">
      <h4>
        Place Order
        <button className="order-toggle-btn" onClick={() => setOpen((o) => !o)}>
          {open ? "Close" : "Open"}
        </button>
      </h4>
      {open && (
        <div>
          <div className="order-summary-bar">
            <span>Budget: ${BUDGET}</span>
            <span>Position: ${posValue.toFixed(2)}</span>
            <span>Risk/share: ${riskPerShare > 0 ? riskPerShare.toFixed(2) : "-"}</span>
            <span>R:R: {rrRatio}</span>
          </div>
          <div className="order-grid">
            <div>
              <label>Order Type</label>
              <select value={orderType} onChange={(e) => setOrderType(e.target.value as "limit" | "market")}>
                <option value="limit">Limit</option>
                <option value="market">Market</option>
              </select>
            </div>
            {orderType === "limit" && (
              <div>
                <label>Limit Price</label>
                <input type="number" step="0.01" value={price} onChange={(e) => updatePrice(Number(e.target.value))} />
              </div>
            )}
            <div>
              <label>Shares</label>
              <input type="number" step="1" min="1" value={quantity} onChange={(e) => setQuantity(Number(e.target.value))} />
            </div>
            <div>
              <label>Stop Loss</label>
              <input type="number" step="0.01" value={stopLoss} onChange={(e) => setStopLoss(Number(e.target.value))} />
            </div>
            <div>
              <label>Take Profit</label>
              <input type="number" step="0.01" value={takeProfit} onChange={(e) => setTakeProfit(Number(e.target.value))} />
            </div>
          </div>
          <div className="order-override-row">
            <input type="checkbox" checked={override} onChange={(e) => setOverride(e.target.checked)} />
            <span>Override trade-ready gate.</span>
          </div>
          {!confirming ? (
            <div className="order-actions">
              <button
                className="order-review-btn"
                disabled={!canReview || loading}
                onClick={() => setConfirming(true)}
              >
                Review Order
              </button>
            </div>
          ) : (
            <div className="order-confirm-box">
              <h5>Confirm Order</h5>
              <div className="order-confirm-summary">
                <div>Ticker: <strong>{ticker}</strong></div>
                <div>Side: <strong>BUY</strong></div>
                <div>Type: <strong>{orderType.toUpperCase()}</strong></div>
                <div>Shares: <strong>{quantity}</strong></div>
                {orderType === "limit" && <div>Limit: <strong>${price.toFixed(2)}</strong></div>}
                <div>Stop Loss: <strong>${stopLoss.toFixed(2)}</strong></div>
                <div>Take Profit: <strong>{takeProfit > 0 ? `$${takeProfit.toFixed(2)}` : "None"}</strong></div>
                <div>Total: <strong>${posValue.toFixed(2)}</strong></div>
              </div>
              <div className="order-confirm-actions">
                <button className="order-confirm-btn" disabled={loading} onClick={placeOrder}>
                  {loading ? "Placing..." : "Confirm & Place"}
                </button>
                <button className="order-back-btn" onClick={() => setConfirming(false)}>
                  Back
                </button>
              </div>
            </div>
          )}
          {message && (
            <div className={message.type === "success" ? "order-message-success" : "order-message-error"}>
              {message.text}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
