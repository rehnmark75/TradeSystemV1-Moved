"use client";

import { useState, useEffect } from "react";
import { useEnvironment } from "../../../lib/environment";
import { useDailyPnlGate, type DailyPnlGateConfig } from "../../../hooks/settings/useDailyPnlGate";

function formatSek(v: number) {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return "—";
  return (n >= 0 ? "+" : "") + n.toFixed(2) + " SEK";
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString("sv-SE");
}

export default function DailyPnlGatePage() {
  const { environment: configSet } = useEnvironment();
  const { config, blocks, loading, saving, error, save } = useDailyPnlGate(configSet);

  const [isEnabled, setIsEnabled] = useState(true);
  const [profitLimit, setProfitLimit] = useState("200");
  const [lossLimit, setLossLimit] = useState("-300");
  const [dirty, setDirty] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  useEffect(() => {
    if (config) {
      setIsEnabled(config.is_enabled);
      setProfitLimit(String(config.profit_limit_sek));
      setLossLimit(String(config.loss_limit_sek));
      setDirty(false);
    }
  }, [config]);

  function markDirty() {
    setDirty(true);
    setSaveMsg(null);
  }

  async function handleSave() {
    const profit = parseFloat(profitLimit);
    const loss = parseFloat(lossLimit);
    if (isNaN(profit) || profit <= 0) { setSaveMsg("Profit limit must be a positive number."); return; }
    if (isNaN(loss) || loss >= 0) { setSaveMsg("Loss limit must be a negative number."); return; }

    const ok = await save({
      environment: configSet,
      is_enabled: isEnabled,
      profit_limit_sek: profit,
      loss_limit_sek: loss,
    });
    setSaveMsg(ok ? "Saved." : null);
    if (ok) setDirty(false);
  }

  const todayBlocks = blocks.filter(
    (b) => new Date(b.blocked_at).toDateString() === new Date().toDateString()
  );

  return (
    <div className="settings-page">
      <div className="settings-page-header">
        <div className="mission-kicker">Risk Protection</div>
        <h1>Daily PnL Gate</h1>
        <p>
          Block new orders automatically once the day&#39;s realized PnL (SEK) crosses a profit lock-in
          or a loss stop. Open positions are unaffected. Config takes effect immediately — no restart needed.
        </p>
      </div>

      {loading && <div className="settings-loading">Loading…</div>}
      {!loading && error && <div className="settings-error">{error}</div>}

      {!loading && config && (
        <>
          <section className="settings-section">
            <div className="settings-section-header">
              <h2>Configuration — {configSet}</h2>
              {config.updated_at && (
                <span className="settings-meta">Last saved {formatDate(config.updated_at)}</span>
              )}
            </div>

            <div className="settings-field-group">
              <div className="settings-field">
                <label>
                  <span className="settings-field-label">Gate enabled</span>
                  <span className="settings-field-desc">
                    When disabled, no orders are blocked regardless of PnL.
                  </span>
                </label>
                <button
                  type="button"
                  className={`toggle-btn${isEnabled ? " active" : ""}`}
                  onClick={() => { setIsEnabled((v) => !v); markDirty(); }}
                >
                  {isEnabled ? "ON" : "OFF"}
                </button>
              </div>

              <div className="settings-field">
                <label htmlFor="profit-limit">
                  <span className="settings-field-label">Profit lock-in (SEK)</span>
                  <span className="settings-field-desc">
                    Block new orders when today&#39;s realized PnL reaches this amount. Must be positive.
                  </span>
                </label>
                <input
                  id="profit-limit"
                  type="number"
                  min="1"
                  step="10"
                  value={profitLimit}
                  onChange={(e) => { setProfitLimit(e.target.value); markDirty(); }}
                  className="settings-input"
                />
              </div>

              <div className="settings-field">
                <label htmlFor="loss-limit">
                  <span className="settings-field-label">Loss stop (SEK)</span>
                  <span className="settings-field-desc">
                    Block new orders when today&#39;s realized PnL falls to this amount. Must be negative.
                  </span>
                </label>
                <input
                  id="loss-limit"
                  type="number"
                  max="-1"
                  step="10"
                  value={lossLimit}
                  onChange={(e) => { setLossLimit(e.target.value); markDirty(); }}
                  className="settings-input"
                />
              </div>
            </div>

            {saveMsg && (
              <div className={`settings-save-msg${saveMsg === "Saved." ? " ok" : " err"}`}>
                {saveMsg}
              </div>
            )}

            <div className="settings-actions">
              <button
                className="btn-primary"
                disabled={!dirty || saving}
                onClick={handleSave}
              >
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-header">
              <h2>Recent Blocks — last 30 days</h2>
              <span className="settings-meta">
                {todayBlocks.length} block{todayBlocks.length !== 1 ? "s" : ""} today
              </span>
            </div>

            {blocks.length === 0 ? (
              <div className="settings-empty">No blocks recorded in the last 30 days.</div>
            ) : (
              <div className="settings-table-wrap">
                <table className="settings-table">
                  <thead>
                    <tr>
                      <th>Time (UTC)</th>
                      <th>Limit hit</th>
                      <th>Daily PnL</th>
                      <th>Limit</th>
                      <th>Epic</th>
                      <th>Dir</th>
                    </tr>
                  </thead>
                  <tbody>
                    {blocks.map((b) => (
                      <tr key={b.id}>
                        <td>{formatDate(b.blocked_at)}</td>
                        <td>
                          <span className={`badge badge-${b.limit_hit}`}>
                            {b.limit_hit === "profit" ? "Profit" : "Loss"}
                          </span>
                        </td>
                        <td>{formatSek(b.daily_pnl_sek)}</td>
                        <td>
                          {b.limit_hit === "profit"
                            ? formatSek(b.profit_limit_sek)
                            : formatSek(b.loss_limit_sek)}
                        </td>
                        <td>{b.epic ?? "—"}</td>
                        <td>{b.direction ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
