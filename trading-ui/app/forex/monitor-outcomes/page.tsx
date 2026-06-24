/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { Fragment, useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type SweepCell = { sl: number; tp: number; pf: number | null; exp: number | null; wr: number | null };
type Sweep = { sl_values: number[]; tp_values: number[]; grid: SweepCell[][]; best: SweepCell };

type Candidate = {
  strategy: string;
  epic: string;
  pair: string;
  n: number;
  tp_hits: number;
  sl_hits: number;
  timeouts: number;
  wr: number | null;
  pf: number | null;
  expectancy: number | null;
  expectancy_r: number | null;
  edge_ratio: number | null;
  pct_mfe_favorable: number | null;
  avg_mfe: number | null;
  avg_mae: number | null;
  avg_early_mae: number | null;
  dead_on_arrival_pct: number | null;
  median_mfe_losers: number | null;
  ref_sl: number | null;
  ref_tp: number | null;
  per_month: number | null;
  thin: boolean;
  sweep: Sweep;
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

type Payload = { candidates: Candidate[]; rows: OutcomeRow[] };

const DAY_OPTIONS = [7, 14, 30, 60, 90];

const fmt = (v: number | null, digits = 1) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);

const signed = (v: number | null) => {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(1)}`;
};

const pf = (v: number | null) => (v == null ? "∞" : v.toFixed(2));

const fmtTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(0, 16).replace("T", " ");
};

// Green (positive) / red (negative) tint, normalized per heatmap so both tight FX
// cells and large-pip gold cells stay readable (scale relative to the cell's own
// max |expectancy| rather than a fixed pip divisor).
const expBg = (v: number | null, maxAbs: number) => {
  if (v == null || !Number.isFinite(v)) return "transparent";
  const a = maxAbs > 0 ? Math.min(0.85, (Math.abs(v) / maxAbs) * 0.85) : 0;
  return v >= 0 ? `rgba(34,197,94,${a})` : `rgba(239,68,68,${a})`;
};

type SortKey =
  | "edge_ratio"
  | "pct_mfe_favorable"
  | "expectancy_r"
  | "expectancy"
  | "pf"
  | "wr"
  | "n"
  | "per_month";
const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "edge_ratio", label: "Edge (MFE ÷ MAE) — primary" },
  { key: "pct_mfe_favorable", label: "% Favourable (MFE > MAE)" },
  { key: "expectancy_r", label: "R-multiple (exp ÷ SL)" },
  { key: "expectancy", label: "Expectancy (pips)" },
  { key: "pf", label: "Profit factor (diagnostic)" },
  { key: "wr", label: "Win rate (diagnostic)" },
  { key: "n", label: "Sample size (n)" },
  { key: "per_month", label: "Frequency (~/mo)" },
];
// null PF means no losses (∞) — sort it to the top.
const sortVal = (c: Candidate, key: SortKey) =>
  key === "pf" ? (c.pf == null ? Infinity : c.pf) : (c[key] ?? -Infinity);

// Edge colour: clearly favourable (>1.2) green, leaking (<1.0) red, marginal amber.
const edgeClass = (v: number | null) =>
  v == null ? "" : v >= 1.2 ? "positive" : v < 1.0 ? "negative" : "";

export default function ForexMonitorOutcomesPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(30);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [strategyFilter, setStrategyFilter] = useState<string>("ALL");
  const [sortKey, setSortKey] = useState<SortKey>("edge_ratio");
  const [expanded, setExpanded] = useState<string | null>(null);

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

  const candidates = payload?.candidates ?? [];
  const rows = payload?.rows ?? [];

  const strategies = useMemo(
    () => Array.from(new Set(rows.map((r) => r.strategy))).filter(Boolean),
    [rows]
  );

  const visibleCandidates = useMemo(() => {
    const base =
      strategyFilter === "ALL" ? candidates : candidates.filter((c) => c.strategy === strategyFilter);
    // Sort by chosen metric; thin cells (n<20) always sink below non-thin ones.
    return [...base].sort((a, b) => {
      if (a.thin !== b.thin) return a.thin ? 1 : -1;
      return sortVal(b, sortKey) - sortVal(a, sortKey);
    });
  }, [candidates, strategyFilter, sortKey]);

  const filteredRows = useMemo(
    () => (strategyFilter === "ALL" ? rows : rows.filter((r) => r.strategy === strategyFilter)),
    [rows, strategyFilter]
  );

  const totalDecided = candidates.reduce((acc, c) => acc + c.n, 0);

  const outcomeClass = (o: string | null) =>
    o === "HIT_TP" ? "positive" : o === "HIT_SL" ? "negative" : "";

  const renderSweep = (c: Candidate) => {
    const { sl_values, tp_values, grid, best } = c.sweep;
    const maxAbs = Math.max(
      0,
      ...grid.flat().map((cell) => (cell.exp == null || !Number.isFinite(cell.exp) ? 0 : Math.abs(cell.exp)))
    );
    return (
      <tr>
        <td colSpan={14} style={{ background: "rgba(255,255,255,0.02)", padding: "12px 16px" }}>
          <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 8 }}>
            <strong>SL/TP sweep</strong> — expectancy (pips/trade) re-derived from this cell&apos;s MFE/MAE
            distribution. Approximate (max-excursion timing, pessimistic ties); a screen, not a backtest.
            Current bracket outlined, best-expectancy bracket ★.
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="forex-table" style={{ fontSize: 11, minWidth: 480 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "right" }}>SL ↓ / TP →</th>
                  {tp_values.map((tp) => (
                    <th key={tp} style={{ textAlign: "center" }}>
                      {tp}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {grid.map((rowg, i) => (
                  <tr key={sl_values[i]}>
                    <td style={{ textAlign: "right", fontWeight: 600 }}>{sl_values[i]}</td>
                    {rowg.map((cell) => {
                      const isBest = cell.sl === best.sl && cell.tp === best.tp;
                      const isRef = cell.sl === c.ref_sl && cell.tp === c.ref_tp;
                      return (
                        <td
                          key={cell.tp}
                          title={`SL ${cell.sl} / TP ${cell.tp} · PF ${pf(cell.pf)} · WR ${fmt(cell.wr, 0)}% · exp ${signed(cell.exp)}`}
                          style={{
                            textAlign: "center",
                            background: expBg(cell.exp, maxAbs),
                            outline: isRef ? "2px solid rgba(255,255,255,0.6)" : undefined,
                            fontWeight: isBest ? 700 : 400,
                          }}
                        >
                          {isBest ? "★ " : ""}
                          {signed(cell.exp)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 12, marginTop: 8 }}>
            Best bracket: <strong>SL {best.sl} / TP {best.tp}</strong> → exp {signed(best.exp)} pips · PF{" "}
            {pf(best.pf)} · WR {fmt(best.wr, 0)}% &nbsp;|&nbsp; Current: SL {c.ref_sl} / TP {c.ref_tp} → exp{" "}
            {signed(c.expectancy)} · PF {pf(c.pf)}
          </div>
        </td>
      </tr>
    );
  };

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
          <h1>Monitor Outcomes — Candidate Edge</h1>
          <p>
            Per-(strategy · pair) directional edge for monitor-only signals. Ranked by{" "}
            <strong>Edge = avg&nbsp;MFE ÷ avg&nbsp;MAE</strong> — how much further price runs in our favour than
            against us over 24h. This is <strong>bracket-independent</strong>, so it measures the entry itself
            rather than the SL/TP geometry (a coin-flip on a 1.5:1 bracket prints PF&nbsp;~1.5 with no real edge —
            which is why PF/WR here are <strong>diagnostic only</strong>, not the ranking metric). Read it as a{" "}
            <strong>screen to pick demo forward-test candidates — not a promotion verdict</strong>: live execution
            uses progressive trailing/break-even stops that materially change P&amp;L. A strong cell means
            &ldquo;worth forward-testing&rdquo;, not &ldquo;ready to trade&rdquo;.
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
          <div>
            <label>Sort by</label>
            <select value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)}>
              {SORT_OPTIONS.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <button className="section-tab active" onClick={load}>
            Refresh
          </button>
          <div className="forex-badge">{totalDecided} decided signals</div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading monitor outcomes...</div>
        ) : (
          <>
            <h3 style={{ marginTop: 8 }}>
              Candidate scorecard — ranked by edge ({visibleCandidates.length} cells)
            </h3>
            <p style={{ fontSize: 12, opacity: 0.7, marginTop: -4 }}>
              Click a row for its SL/TP sweep. ⚠ = thin sample (n&lt;20), treat with caution. <strong>Edge</strong> =
              avg MFE ÷ avg MAE (<span className="positive">≥1.2 good</span>, <span className="negative">&lt;1.0 leaks</span>);{" "}
              <strong>%Fav</strong> = share of signals whose MFE beat their MAE (outlier-proof cross-check of Edge).{" "}
              <strong>%DoA</strong> = share whose max favourable move stayed under 2 pips (&ldquo;dead on arrival&rdquo;,
              an entry leak no stop can fix). The greyed <strong>PF / WR / Exp</strong> block is on the fixed reference
              bracket — <strong>diagnostic only</strong> (inflated by TP:SL geometry); do not rank on it.
            </p>
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Strategy · Pair</th>
                  <th>n</th>
                  <th title="avg MFE ÷ avg MAE — primary edge metric">Edge</th>
                  <th title="% of signals whose MFE beat their MAE">%Fav</th>
                  <th>Avg MFE</th>
                  <th>Avg MAE</th>
                  <th title="avg adverse move in the first 15 min — entry-timing quality">Early MAE</th>
                  <th>%DoA</th>
                  <th>~/mo</th>
                  <th style={{ opacity: 0.55 }} title="diagnostic only — fixed reference bracket">PF</th>
                  <th style={{ opacity: 0.55 }} title="diagnostic only — fixed reference bracket">WR</th>
                  <th style={{ opacity: 0.55 }} title="diagnostic only — fixed reference bracket">Exp</th>
                  <th>Best SL/TP</th>
                  <th>TP/SL/TO</th>
                </tr>
              </thead>
              <tbody>
                {visibleCandidates.length === 0 ? (
                  <tr>
                    <td colSpan={14}>No decided monitor-only outcomes in this window.</td>
                  </tr>
                ) : (
                  visibleCandidates.map((c) => {
                    const key = `${c.strategy}__${c.epic}`;
                    const isOpen = expanded === key;
                    const pfPos = c.pf == null || c.pf >= 1;
                    return (
                      <Fragment key={key}>
                        <tr
                          onClick={() => setExpanded(isOpen ? null : key)}
                          style={{ cursor: "pointer", opacity: c.thin ? 0.6 : 1 }}
                        >
                          <td>
                            {isOpen ? "▾ " : "▸ "}
                            <strong>{c.strategy}</strong> · {c.pair}
                          </td>
                          <td>
                            {c.thin ? "⚠ " : ""}
                            {c.n}
                          </td>
                          <td className={edgeClass(c.edge_ratio)} style={{ fontWeight: 700 }}>
                            {c.edge_ratio == null ? "—" : `${c.edge_ratio.toFixed(2)}×`}
                          </td>
                          <td className={c.pct_mfe_favorable != null && c.pct_mfe_favorable < 50 ? "negative" : ""}>
                            {c.pct_mfe_favorable == null ? "—" : `${c.pct_mfe_favorable.toFixed(0)}%`}
                          </td>
                          <td>{fmt(c.avg_mfe)}</td>
                          <td>{fmt(c.avg_mae)}</td>
                          <td>{fmt(c.avg_early_mae)}</td>
                          <td className={c.dead_on_arrival_pct != null && c.dead_on_arrival_pct > 40 ? "negative" : ""}>
                            {c.dead_on_arrival_pct == null ? "—" : `${c.dead_on_arrival_pct.toFixed(0)}%`}
                          </td>
                          <td>{c.per_month == null ? "—" : c.per_month.toFixed(0)}</td>
                          <td className={pfPos ? "positive" : "negative"} style={{ opacity: 0.55 }}>
                            {pf(c.pf)}
                          </td>
                          <td style={{ opacity: 0.55 }}>{fmt(c.wr, 0)}%</td>
                          <td
                            className={c.expectancy != null && c.expectancy < 0 ? "negative" : "positive"}
                            style={{ opacity: 0.55 }}
                          >
                            {signed(c.expectancy)}
                          </td>
                          <td>
                            {c.sweep.best.sl}/{c.sweep.best.tp}
                          </td>
                          <td>
                            {c.tp_hits}/{c.sl_hits}/{c.timeouts}
                          </td>
                        </tr>
                        {isOpen ? renderSweep(c) : null}
                      </Fragment>
                    );
                  })
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
