"use client";

import { useMemo, useState } from "react";
import { useEnvironment } from "../../lib/environment";
import { apiUrl } from "../../lib/settings/api";
import { fetchJson } from "../../lib/http";

// ─── Types ────────────────────────────────────────────────────────────────────

type OverrideRow = {
  key: string;
  type: "number" | "boolean" | "string";
  value: string;
};

type EvaluateResponse =
  | {
      fired: true;
      signal: Record<string, unknown>;
      trade_request: TradeRequest;
      forced?: boolean;
      original_rejection_reason?: string;
    }
  | { fired: false; rejection_reason: string; debug?: string };

type TradeRequest = {
  epic: string;
  direction: string;
  stop_distance: number | null;
  limit_distance: number | null;
  use_provided_sl_tp: boolean;
  alert_id: null;
  trigger_source: "manual_trigger";
};

type PlaceResult = { success: boolean; message: string };

// ─── Props ────────────────────────────────────────────────────────────────────

interface ManualTriggerPanelProps {
  epic: string;
  strategy: string;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ManualTriggerPanel({ epic, strategy }: ManualTriggerPanelProps) {
  const { environment } = useEnvironment();
  const isDemoMode = environment === "demo";

  const [rows, setRows] = useState<OverrideRow[]>([]);
  const [forceTrade, setForceTrade] = useState(false);
  const [manualDirection, setManualDirection] = useState<"BUY" | "SELL">("BUY");
  const [evaluating, setEvaluating] = useState(false);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  const [placing, setPlacing] = useState(false);
  const [placeResult, setPlaceResult] = useState<PlaceResult | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);

  // Build config_override dict from rows
  const configOverride = useMemo(() => {
    const out: Record<string, unknown> = {};
    rows.forEach((row) => {
      if (!row.key.trim()) return;
      if (row.type === "number") {
        const n = Number(row.value);
        out[row.key] = Number.isNaN(n) ? row.value : n;
      } else if (row.type === "boolean") {
        out[row.key] = row.value === "true";
      } else {
        out[row.key] = row.value;
      }
    });
    return out;
  }, [rows]);

  // ── Row helpers ──────────────────────────────────────────────────────────────

  const addRow = () =>
    setRows((prev) => [...prev, { key: "", type: "number", value: "" }]);

  const removeRow = (i: number) =>
    setRows((prev) => prev.filter((_, idx) => idx !== i));

  const updateRow = (
    i: number,
    field: keyof OverrideRow,
    value: string
  ) =>
    setRows((prev) =>
      prev.map((row, idx) => (idx === i ? { ...row, [field]: value } : row))
    );

  // ── Evaluate ─────────────────────────────────────────────────────────────────

  const handleEvaluate = async () => {
    setEvaluating(true);
    setResult(null);
    setEvalError(null);
    setPlaceResult(null);
    setShowConfirm(false);

    const res = await fetchJson<EvaluateResponse>(
      apiUrl("/api/strategy/manual-evaluate"),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          epic,
          strategy,
          config_override: configOverride,
          force_trade: forceTrade,
          manual_direction: manualDirection,
        }),
        timeoutMs: 100_000,
      }
    );

    setEvaluating(false);

    if (!res.ok) {
      setEvalError(res.error || "Unknown error from evaluator.");
    } else {
      setResult(res.data);
    }
  };

  // ── Place demo trade ──────────────────────────────────────────────────────────

  const handlePlaceConfirmed = async () => {
    if (!result || !result.fired) return;
    setShowConfirm(false);
    setPlacing(true);
    setPlaceResult(null);

    const tradeReq = result.trade_request;
    const res = await fetchJson<Record<string, unknown>>(
      apiUrl("/api/strategy/ig-place-demo"),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(tradeReq),
        timeoutMs: 30_000,
      }
    );

    setPlacing(false);

    if (!res.ok) {
      setPlaceResult({ success: false, message: res.error || "Order failed." });
    } else {
      const d = res.data;
      const ref =
        (d.deal_reference as string) ||
        (d.dealReference as string) ||
        (d.deal_id as string) ||
        "";
      setPlaceResult({
        success: true,
        message: ref ? `Order placed — ref: ${ref}` : "Order placed successfully.",
      });
    }
  };

  // ── Render helpers ────────────────────────────────────────────────────────────

  const renderSignalRow = (label: string, value: unknown) => {
    if (value === null || value === undefined) return null;
    return (
      <div key={label} className="trigger-signal-row">
        <span className="trigger-signal-label">{label}</span>
        <span className="trigger-signal-value">{String(value)}</span>
      </div>
    );
  };

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="manual-trigger-panel">
      <div className="manual-trigger-header">
        <span className="manual-trigger-title">Manual Trigger</span>
        <span className="manual-trigger-meta">
          {strategy} · {epic}
        </span>
        {!isDemoMode && (
          <span className="manual-trigger-env-warning">
            Switch to demo environment to enable this feature.
          </span>
        )}
      </div>

      {/* ── Force Trade ──────────────────────────────────────────────── */}
      <div className="manual-trigger-force">
        <label className="trigger-checkbox-label">
          <input
            type="checkbox"
            checked={forceTrade}
            disabled={!isDemoMode}
            onChange={(e) => setForceTrade(e.target.checked)}
          />
          <span>Create trade even when strategy has no signal</span>
        </label>
        <select
          value={manualDirection}
          disabled={!isDemoMode || !forceTrade}
          onChange={(e) => setManualDirection(e.target.value as "BUY" | "SELL")}
        >
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
        </select>
        <span className="manual-trigger-hint">
          Uses 15/30 pips unless overridden below.
        </span>
      </div>

      {/* ── Parameter Overrides ───────────────────────────────────────── */}
      <div className="manual-trigger-section">
        <div className="manual-trigger-section-title">
          One-Shot Parameter Overrides
          <span className="manual-trigger-hint">
            Applied for this evaluation only — DB config unchanged.
          </span>
        </div>

        <div className="override-row override-header">
          <span>Field</span>
          <span>Type</span>
          <span>Value</span>
          <span />
        </div>

        {rows.map((row, i) => (
          <div key={i} className="override-row">
            <input
              placeholder="e.g. fixed_stop_loss_pips"
              value={row.key}
              disabled={!isDemoMode}
              onChange={(e) => updateRow(i, "key", e.target.value)}
            />
            <select
              value={row.type}
              disabled={!isDemoMode}
              onChange={(e) =>
                updateRow(i, "type", e.target.value as OverrideRow["type"])
              }
            >
              <option value="number">Number</option>
              <option value="boolean">Boolean</option>
              <option value="string">String</option>
            </select>
            {row.type === "boolean" ? (
              <select
                value={row.value || "false"}
                disabled={!isDemoMode}
                onChange={(e) => updateRow(i, "value", e.target.value)}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <input
                value={row.value}
                disabled={!isDemoMode}
                onChange={(e) => updateRow(i, "value", e.target.value)}
              />
            )}
            <button
              onClick={() => removeRow(i)}
              disabled={!isDemoMode}
              className="trigger-remove-btn"
            >
              ✕
            </button>
          </div>
        ))}

        <button
          className="primary trigger-add-btn"
          onClick={addRow}
          disabled={!isDemoMode}
        >
          Add Override
        </button>
      </div>

      {/* ── Evaluate Button ───────────────────────────────────────────── */}
      <div className="manual-trigger-actions">
        <button
          className="primary trigger-evaluate-btn"
          onClick={handleEvaluate}
          disabled={!isDemoMode || evaluating}
        >
          {evaluating ? "Evaluating…" : "⚡ Evaluate Now"}
        </button>
        {evaluating && (
          <span className="manual-trigger-hint">
            Running strategy against live candles — up to ~30 s…
          </span>
        )}
      </div>

      {/* ── Error ─────────────────────────────────────────────────────── */}
      {evalError && (
        <div className="manual-trigger-error">
          <strong>Error:</strong> {evalError}
        </div>
      )}

      {/* ── Result: No Signal ─────────────────────────────────────────── */}
      {result && !result.fired && (
        <div className="manual-trigger-result manual-trigger-result-no-signal">
          <div className="trigger-result-icon">✗</div>
          <div>
            <div className="trigger-result-title">No Signal</div>
            <div className="trigger-result-reason">{result.rejection_reason}</div>
          </div>
        </div>
      )}

      {/* ── Result: Signal ────────────────────────────────────────────── */}
      {result && result.fired && (
        <div className="manual-trigger-result manual-trigger-result-signal">
          <div className="trigger-result-header">
            <div className="trigger-result-icon">✓</div>
            <div className="trigger-result-title">
              {result.forced ? "Manual Override" : "Signal Detected"} —{" "}
              <strong>
                {String(
                  result.signal.signal_type ||
                    result.signal.direction ||
                    result.trade_request.direction
                )}
              </strong>
            </div>
          </div>

          {result.forced && result.original_rejection_reason && (
            <div className="trigger-result-reason">
              Strategy rejection: {result.original_rejection_reason}
            </div>
          )}

          <div className="trigger-signal-grid">
            {renderSignalRow(
              "Direction",
              result.signal.signal_type ||
                result.signal.direction ||
                result.trade_request.direction
            )}
            {renderSignalRow(
              "Entry",
              result.signal.entry_price != null
                ? Number(result.signal.entry_price).toFixed(5)
                : null
            )}
            {renderSignalRow(
              "Stop (pips)",
              result.trade_request.stop_distance
            )}
            {renderSignalRow(
              "Target (pips)",
              result.trade_request.limit_distance
            )}
            {renderSignalRow(
              "Confidence",
              result.signal.confidence_score != null
                ? `${(Number(result.signal.confidence_score) * 100).toFixed(1)}%`
                : result.signal.confidence != null
                ? `${(Number(result.signal.confidence) * 100).toFixed(1)}%`
                : null
            )}
            {renderSignalRow("Strategy", result.signal.strategy)}
            {renderSignalRow("Regime", result.signal.market_regime)}
            {renderSignalRow(
              "SL/TP source",
              result.signal.sl_tp_source || result.signal.manual_override_reason
            )}
          </div>

          {/* Place trade */}
          {!placeResult && !showConfirm && (
            <button
              className="primary trigger-place-btn"
              onClick={() => setShowConfirm(true)}
              disabled={placing}
            >
              Place Demo Trade
            </button>
          )}

          {/* Confirm modal */}
          {showConfirm && (
            <div className="trigger-confirm">
              <span>
                Place a <strong>{result.trade_request.direction}</strong> market
                order on <strong>{epic}</strong> on demo?
              </span>
              <div className="trigger-confirm-actions">
                <button
                  className="primary"
                  onClick={handlePlaceConfirmed}
                  disabled={placing}
                >
                  {placing ? "Placing…" : "Confirm"}
                </button>
                <button onClick={() => setShowConfirm(false)}>Cancel</button>
              </div>
            </div>
          )}

          {/* Place result */}
          {placeResult && (
            <div
              className={`trigger-place-result ${
                placeResult.success
                  ? "trigger-place-result-ok"
                  : "trigger-place-result-err"
              }`}
            >
              {placeResult.success ? "✓" : "✗"} {placeResult.message}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
