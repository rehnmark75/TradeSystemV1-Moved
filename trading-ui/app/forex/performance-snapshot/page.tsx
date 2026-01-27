/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type SnapshotPayload = {
  range: { start: string; end: string };
  summary: Record<string, number | string | null>;
  timeline: Array<{ hour: string; total_scans: number; signals: number }>;
  regimes: Array<{ market_regime: string; count: number; signals: number }>;
  sessions: Array<{ session: string; session_volatility: string; count: number }>;
  rejections: Array<{ rejection_reason: string; count: number }>;
  epics: Array<{
    epic: string;
    pair_name: string;
    total_scans: number;
    signals: number;
    avg_raw_confidence: number;
    avg_final_confidence: number;
    dominant_regime: string;
  }>;
  indicators: {
    signals: Record<string, number> | null;
    non_signals: Record<string, number> | null;
  };
};

const toDateInput = (value: Date) => value.toISOString().slice(0, 10);

export default function ForexPerformanceSnapshotPage() {
  const [start, setStart] = useState(toDateInput(new Date(Date.now() - 24 * 3600 * 1000)));
  const [end, setEnd] = useState(toDateInput(new Date()));
  const [payload, setPayload] = useState<SnapshotPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSnapshot = () => {
    setLoading(true);
    setError(null);
    fetch(
      `/stock-scanner/api/forex/performance-snapshot/?start=${start}&end=${end}`
    )
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load performance snapshot."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadSnapshot();
  }, []);

  const summary = payload?.summary ?? {};
  const timeline = payload?.timeline ?? [];

  const maxScans = useMemo(() => {
    if (!timeline.length) return 1;
    return Math.max(1, ...timeline.map((row) => Number(row.total_scans ?? 0)));
  }, [timeline]);

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          Trading Hub
        </Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Performance Snapshot</h1>
          <p>Scanner throughput, signals, rejections, and per-epic metrics.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <div className="forex-nav">
        <Link href="/forex" className="forex-pill">
          Overview
        </Link>
        <Link href="/forex/strategy" className="forex-pill">
          Strategy Performance
        </Link>
        <Link href="/forex/trade-performance" className="forex-pill">
          Trade Performance
        </Link>
        <Link href="/forex/entry-timing" className="forex-pill">
          Entry Timing
        </Link>
        <Link href="/forex/mae-analysis" className="forex-pill">
          MAE Analysis
        </Link>
        <Link href="/forex/alert-history" className="forex-pill">
          Alert History
        </Link>
        <Link href="/forex/trade-analysis" className="forex-pill">
          Trade Analysis
        </Link>
        <Link href="/forex/market-intelligence" className="forex-pill">
          Market Intelligence
        </Link>
        <Link href="/forex/smc-rejections" className="forex-pill">
          SMC Rejections
        </Link>
      </div>

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Start</label>
            <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
          </div>
          <div>
            <label>End</label>
            <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
          </div>
          <button className="section-tab active" onClick={loadSnapshot}>
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}
        {loading ? (
          <div className="chart-placeholder">Loading snapshot...</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Total Scans
                <strong>{summary.total_scans ?? 0}</strong>
              </div>
              <div className="summary-card">
                Scan Cycles
                <strong>{summary.scan_cycles ?? 0}</strong>
              </div>
              <div className="summary-card">
                Unique Epics
                <strong>{summary.unique_epics ?? 0}</strong>
              </div>
              <div className="summary-card">
                Signals Generated
                <strong>{summary.signals_generated ?? 0}</strong>
              </div>
              <div className="summary-card">
                Signal Rate
                <strong>
                  {summary.signal_rate
                    ? `${(Number(summary.signal_rate) * 100).toFixed(2)}%`
                    : "0%"}
                </strong>
              </div>
            </div>

            <div className="panel chart-panel">
              <div className="chart-title">Scan Activity Timeline</div>
              {timeline.length ? (
                <div className="bar-stack">
                  {timeline.map((row) => (
                    <div key={row.hour} className="bar-row">
                      <span>{new Date(row.hour).toLocaleString("en-GB", { hour: "2-digit", day: "2-digit" })}</span>
                      <div className="bar-track">
                        <div
                          className="bar-fill good"
                          style={{ width: `${(Number(row.total_scans ?? 0) / maxScans) * 100}%` }}
                        />
                      </div>
                      <strong>{row.total_scans}</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="chart-placeholder">No scan timeline data.</div>
              )}
            </div>

            <div className="forex-grid">
              <div className="panel chart-panel">
                <div className="chart-title">Market Regime Distribution</div>
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Regime</th>
                      <th>Scans</th>
                      <th>Signals</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(payload?.regimes ?? []).map((row) => (
                      <tr key={row.market_regime}>
                        <td>{row.market_regime}</td>
                        <td>{row.count}</td>
                        <td>{row.signals}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="panel chart-panel">
                <div className="chart-title">Session Distribution</div>
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Session</th>
                      <th>Volatility</th>
                      <th>Scans</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(payload?.sessions ?? []).map((row) => (
                      <tr key={`${row.session}-${row.session_volatility}`}>
                        <td>{row.session}</td>
                        <td>{row.session_volatility}</td>
                        <td>{row.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Rejection Analysis</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Reason</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.rejections ?? []).map((row) => (
                    <tr key={row.rejection_reason}>
                      <td>{row.rejection_reason}</td>
                      <td>{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Epic Performance</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Epic</th>
                    <th>Pair</th>
                    <th>Scans</th>
                    <th>Signals</th>
                    <th>Avg Raw Conf</th>
                    <th>Avg Final Conf</th>
                    <th>Dominant Regime</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.epics ?? []).slice(0, 20).map((row) => (
                    <tr key={row.epic}>
                      <td>{row.epic}</td>
                      <td>{row.pair_name ?? "-"}</td>
                      <td>{row.total_scans}</td>
                      <td>{row.signals}</td>
                      <td>{Number(row.avg_raw_confidence ?? 0).toFixed(2)}</td>
                      <td>{Number(row.avg_final_confidence ?? 0).toFixed(2)}</td>
                      <td>{row.dominant_regime ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Indicator Comparison</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Group</th>
                    <th>RSI</th>
                    <th>ADX</th>
                    <th>ER</th>
                    <th>ATR</th>
                    <th>BB%</th>
                    <th>SMC</th>
                    <th>MTF</th>
                    <th>Entry Quality</th>
                  </tr>
                </thead>
                <tbody>
                  {(["signals", "non_signals"] as const).map((key) => (
                    <tr key={key}>
                      <td>{key.replace("_", " ")}</td>
                      <td>{payload?.indicators?.[key]?.avg_rsi?.toFixed(2) ?? "-"}</td>
                      <td>{payload?.indicators?.[key]?.avg_adx?.toFixed(2) ?? "-"}</td>
                      <td>{payload?.indicators?.[key]?.avg_er?.toFixed(2) ?? "-"}</td>
                      <td>{payload?.indicators?.[key]?.avg_atr?.toFixed(2) ?? "-"}</td>
                      <td>{payload?.indicators?.[key]?.avg_bb_percentile?.toFixed(2) ?? "-"}</td>
                      <td>{payload?.indicators?.[key]?.avg_smc_score?.toFixed(2) ?? "-"}</td>
                      <td>{payload?.indicators?.[key]?.avg_mtf_score?.toFixed(2) ?? "-"}</td>
                      <td>{payload?.indicators?.[key]?.avg_entry_quality?.toFixed(2) ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
