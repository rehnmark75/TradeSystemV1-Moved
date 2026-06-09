/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type StrategySummary = {
  strategy: string;
  n: number;
  avg_mfe: number | null;
  avg_mae: number | null;
  avg_early_mae: number | null;
  avg_pnl_1440m: number | null;
  tp: number;
  sl: number;
  ref_wr: number | null;
  avg_ref_pnl: number | null;
};

type OutcomeRow = {
  alert_id: number;
  strategy: string;
  epic: string;
  pair: string;
  direction: string;
  signal_timestamp: string;
  status: string;
  entry_price: number | null;
  mfe_pips: number | null;
  mae_pips: number | null;
  early_mae_pips: number | null;
  pnl_60m_pips: number | null;
  pnl_240m_pips: number | null;
  pnl_1440m_pips: number | null;
  ref_sl_pips: number | null;
  ref_tp_pips: number | null;
  ref_outcome: string | null;
  ref_pnl_pips: number | null;
  time_to_mfe_minutes: number | null;
  candles_evaluated: number | null;
};

type Payload = { summary: StrategySummary[]; rows: OutcomeRow[] };

const DAY_OPTIONS = [7, 14, 30, 60, 90];

const fmt = (v: number | null, digits = 1) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);

const signed = (v: number | null) => {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(1)}`;
};

const fmtTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(0, 16).replace("T", " ");
};

export default function ForexMonitorOutcomesPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(30);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [strategyFilter, setStrategyFilter] = useState<string>("ALL");

  const load = () => {
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/monitor-outcomes/?days=${days}&env=${environment}`)
      .then((res) => res.json())
      .then((data) => {
        if (data?.error) throw new Error(data.error);
        setPayload(data);
      })
      .catch(() => setError("Failed to load monitor outcomes."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, [days, environment]);

  const summary = payload?.summary ?? [];
  const rows = payload?.rows ?? [];

  const strategies = useMemo(
    () => Array.from(new Set(rows.map((r) => r.strategy))).filter(Boolean),
    [rows]
  );

  const filteredRows = useMemo(
    () => (strategyFilter === "ALL" ? rows : rows.filter((r) => r.strategy === strategyFilter)),
    [rows, strategyFilter]
  );

  const totalSignals = summary.reduce((acc, s) => acc + s.n, 0);

  const outcomeClass = (o: string | null) =>
    o === "HIT_TP" ? "positive" : o === "HIT_SL" ? "negative" : "";

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          K.L.I.R.R
        </Link>
        <EnvironmentToggle />
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Monitor Outcomes</h1>
          <p>
            Counterfactual forward outcomes (MFE/MAE) for monitor-only signals — what every
            logged-but-not-executed signal would have done over 24h. Reference SL/TP grid is a
            comparable win-rate anchor, not each strategy&apos;s native stop.
          </p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/monitor-outcomes" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Time Period</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {DAY_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}d
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Strategy</label>
            <select value={strategyFilter} onChange={(event) => setStrategyFilter(event.target.value)}>
              <option value="ALL">All strategies</option>
              {strategies.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <button className="section-tab active" onClick={load}>
            Refresh
          </button>
          <div className="forex-badge">{totalSignals} resolved signals</div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading monitor outcomes...</div>
        ) : (
          <>
            <h3 style={{ marginTop: 8 }}>Per-strategy summary (resolved only)</h3>
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Signals</th>
                  <th>Avg MFE</th>
                  <th>Avg MAE</th>
                  <th>Early MAE (15m)</th>
                  <th>Avg P&amp;L @24h</th>
                  <th>Ref TP / SL</th>
                  <th>Ref WR</th>
                  <th>Avg Ref P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {summary.length === 0 ? (
                  <tr>
                    <td colSpan={9}>No resolved monitor-only outcomes in this window.</td>
                  </tr>
                ) : (
                  summary.map((s) => (
                    <tr key={s.strategy}>
                      <td>{s.strategy}</td>
                      <td>{s.n}</td>
                      <td>{fmt(s.avg_mfe)}</td>
                      <td>{fmt(s.avg_mae)}</td>
                      <td>{fmt(s.avg_early_mae)}</td>
                      <td className={s.avg_pnl_1440m != null && s.avg_pnl_1440m < 0 ? "negative" : "positive"}>
                        {signed(s.avg_pnl_1440m)}
                      </td>
                      <td>
                        {s.tp} / {s.sl}
                      </td>
                      <td className={s.ref_wr != null && s.ref_wr < 50 ? "negative" : "positive"}>
                        {s.ref_wr == null ? "—" : `${s.ref_wr.toFixed(0)}%`}
                      </td>
                      <td className={s.avg_ref_pnl != null && s.avg_ref_pnl < 0 ? "negative" : "positive"}>
                        {signed(s.avg_ref_pnl)}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>

            <h3 style={{ marginTop: 24 }}>
              Signals ({filteredRows.length}
              {strategyFilter !== "ALL" ? ` · ${strategyFilter}` : ""})
            </h3>
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Strategy</th>
                  <th>Pair</th>
                  <th>Dir</th>
                  <th>MFE</th>
                  <th>MAE</th>
                  <th>Early MAE</th>
                  <th>P&amp;L 60m</th>
                  <th>P&amp;L 240m</th>
                  <th>P&amp;L 24h</th>
                  <th>Ref</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.length === 0 ? (
                  <tr>
                    <td colSpan={12}>No signals in this window.</td>
                  </tr>
                ) : (
                  filteredRows.map((r) => (
                    <tr key={r.alert_id}>
                      <td>{fmtTime(r.signal_timestamp)}</td>
                      <td>{r.strategy}</td>
                      <td>{r.pair}</td>
                      <td>{r.direction}</td>
                      <td>{fmt(r.mfe_pips)}</td>
                      <td>{fmt(r.mae_pips)}</td>
                      <td>{fmt(r.early_mae_pips)}</td>
                      <td>{signed(r.pnl_60m_pips)}</td>
                      <td>{signed(r.pnl_240m_pips)}</td>
                      <td>{signed(r.pnl_1440m_pips)}</td>
                      <td className={outcomeClass(r.ref_outcome)}>{r.ref_outcome ?? "—"}</td>
                      <td>{r.status}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}
