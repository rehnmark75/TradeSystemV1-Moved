/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type IntelligencePayload = {
  summary: {
    total: number;
    avg_epics: number;
    unique_regimes: number;
    avg_confidence: number;
  };
  regimes: Record<string, number>;
  sessions: Record<string, number>;
  volatility: Record<string, number>;
  intelligence_sources: Record<string, number>;
  comprehensive: Array<Record<string, any>>;
  signals: Array<Record<string, any>>;
};

const toDateInput = (value: Date) => value.toISOString().slice(0, 10);

const mapToRows = (data: Record<string, number>) =>
  Object.entries(data || {})
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);

const formatPercent = (value: number) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";

const formatConfidence = (value: number) =>
  Number.isFinite(value) ? value.toFixed(2) : "-";

const REGIME_COLORS: Record<string, string> = {
  trending: "#2f9e44",
  breakout: "#f08c00",
  ranging: "#228be6",
  reversal: "#e03131",
  high_volatility: "#d9480f",
  low_volatility: "#2b8a3e",
  unknown: "#adb5bd"
};

const VOLATILITY_COLORS: Record<string, string> = {
  high: "#e03131",
  medium: "#f08c00",
  low: "#2f9e44"
};

const buildDonutGradient = (rows: { label: string; value: number }[]) => {
  const total = rows.reduce((sum, row) => sum + row.value, 0) || 1;
  let cursor = 0;
  const segments = rows.map((row) => {
    const start = cursor;
    const share = (row.value / total) * 360;
    cursor += share;
    const color = REGIME_COLORS[row.label] ?? "#adb5bd";
    return `${color} ${start.toFixed(2)}deg ${cursor.toFixed(2)}deg`;
  });
  return `conic-gradient(${segments.join(", ")})`;
};

export default function ForexMarketIntelligencePage() {
  const [start, setStart] = useState(toDateInput(new Date(Date.now() - 7 * 24 * 3600 * 1000)));
  const [end, setEnd] = useState(toDateInput(new Date()));
  const [source, setSource] = useState("comprehensive");
  const [payload, setPayload] = useState<IntelligencePayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = () => {
    setLoading(true);
    setError(null);
    fetch(
      `/stock-scanner/api/forex/market-intelligence/?start=${start}&end=${end}&source=${source}`
    )
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load market intelligence."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, []);

  const regimeRows = useMemo(() => mapToRows(payload?.regimes ?? {}), [payload]);
  const sessionRows = useMemo(() => mapToRows(payload?.sessions ?? {}), [payload]);
  const volatilityRows = useMemo(() => mapToRows(payload?.volatility ?? {}), [payload]);
  const sourceRows = useMemo(() => mapToRows(payload?.intelligence_sources ?? {}), [payload]);
  const recentItems = (source === "signal" ? payload?.signals : payload?.comprehensive) ?? [];
  const maxRegime = regimeRows.length ? Math.max(...regimeRows.map((row) => row.value)) : 1;
  const maxSession = sessionRows.length ? Math.max(...sessionRows.map((row) => row.value)) : 1;
  const maxVolatility = volatilityRows.length ? Math.max(...volatilityRows.map((row) => row.value)) : 1;
  const maxSource = sourceRows.length ? Math.max(...sourceRows.map((row) => row.value)) : 1;

  const regimeConfidenceRows = useMemo(() => {
    const data = (source === "signal" ? payload?.signals : payload?.comprehensive) ?? [];
    const buckets: Record<string, { total: number; count: number }> = {};
    data.forEach((row) => {
      if (!row.regime) return;
      const key = String(row.regime);
      const value = Number(row.regime_confidence ?? 0);
      if (!buckets[key]) buckets[key] = { total: 0, count: 0 };
      if (Number.isFinite(value)) {
        buckets[key].total += value;
        buckets[key].count += 1;
      }
    });
    return Object.entries(buckets)
      .map(([label, { total, count }]) => ({
        label,
        value: count ? total / count : 0
      }))
      .sort((a, b) => b.value - a.value);
  }, [payload, source]);
  const maxRegimeConfidence = regimeConfidenceRows.length
    ? Math.max(...regimeConfidenceRows.map((row) => row.value))
    : 1;

  const sessionRegimeConfidence = useMemo(() => {
    const data = (source === "signal" ? payload?.signals : payload?.comprehensive) ?? [];
    const buckets: Record<
      string,
      Record<string, { total: number; count: number }>
    > = {};

    data.forEach((row) => {
      if (!row.session || row.regime_confidence == null || !row.regime) return;
      const sessionKey = String(row.session);
      const regimeKey = String(row.regime);
      const value = Number(row.regime_confidence ?? 0);
      if (!Number.isFinite(value)) return;
      if (!buckets[sessionKey]) buckets[sessionKey] = {};
      if (!buckets[sessionKey][regimeKey]) {
        buckets[sessionKey][regimeKey] = { total: 0, count: 0 };
      }
      buckets[sessionKey][regimeKey].total += value;
      buckets[sessionKey][regimeKey].count += 1;
    });

    return Object.entries(buckets).map(([sessionKey, regimes]) => {
      const segments = Object.entries(regimes).map(([regimeKey, value]) => ({
        regime: regimeKey,
        avg: value.count ? value.total / value.count : 0
      }));
      const total = segments.reduce((sum, seg) => sum + seg.avg, 0) || 1;
      return { session: sessionKey, segments, total };
    });
  }, [payload, source]);

  const recommendedStrategyRows = useMemo(() => {
    const data = payload?.comprehensive ?? [];
    const buckets: Record<
      string,
      Record<string, { total: number; count: number }>
    > = {};

    data.forEach((row) => {
      if (!row.recommended_strategy || !row.regime) return;
      const strat = String(row.recommended_strategy);
      const regime = String(row.regime);
      const value = Number(row.regime_confidence ?? 0);
      if (!buckets[strat]) buckets[strat] = {};
      if (!buckets[strat][regime]) buckets[strat][regime] = { total: 0, count: 0 };
      if (Number.isFinite(value)) {
        buckets[strat][regime].total += value;
      }
      buckets[strat][regime].count += 1;
    });

    return Object.entries(buckets).flatMap(([strategy, regimes]) =>
      Object.entries(regimes).map(([regime, value]) => ({
        strategy,
        regime,
        avg_confidence: value.count ? value.total / value.count : 0,
        count: value.count
      }))
    );
  }, [payload]);

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
          <h1>Market Intelligence</h1>
          <p>Market regime, session, and volatility insights from scanner intelligence.</p>
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
        <Link href="/forex/performance-snapshot" className="forex-pill">
          Performance Snapshot
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
          <div>
            <label>Source</label>
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="comprehensive">Comprehensive Scans</option>
              <option value="signal">Signal-Based</option>
              <option value="both">Both</option>
            </select>
          </div>
          <button className="section-tab active" onClick={loadData}>
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}
        {loading ? (
          <div className="chart-placeholder">Loading intelligence...</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Total Records
                <strong>{payload?.summary?.total ?? 0}</strong>
              </div>
              <div className="summary-card">
                Avg Epics / Scan
                <strong>{payload?.summary?.avg_epics?.toFixed(1) ?? "-"}</strong>
              </div>
              <div className="summary-card">
                Regimes Detected
                <strong>{payload?.summary?.unique_regimes ?? 0}</strong>
              </div>
              <div className="summary-card">
                Avg Confidence
                <strong>{formatConfidence(payload?.summary?.avg_confidence ?? 0)}</strong>
              </div>
              <div className="summary-card">
                Data Source
                <strong>{source === "signal" ? "Signals" : source === "both" ? "Mixed" : "Scans"}</strong>
              </div>
            </div>

            <div className="forex-grid">
              <div className="panel chart-panel">
                <div className="chart-title">Market Regime Distribution</div>
                <p className="chart-caption">
                  Share of dominant regimes across scans in the selected period.
                </p>
                <div className="donut-wrap">
                  <div
                    className="donut-chart"
                    style={{ background: buildDonutGradient(regimeRows) }}
                  />
                  <div className="donut-legend">
                    {regimeRows.map((row) => (
                      <div key={row.label} className="legend-item">
                        <span
                          className="legend-dot"
                          style={{ background: REGIME_COLORS[row.label] ?? "#adb5bd" }}
                        />
                        <span>{row.label}</span>
                        <strong>{row.value}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="panel chart-panel">
                <div className="chart-title">Average Regime Confidence by Session</div>
                <p className="chart-caption">
                  Confidence split by regime within each trading session.
                </p>
                <div className="stacked-rows">
                  {sessionRegimeConfidence.map((row) => (
                    <div key={row.session} className="stacked-row">
                      <span className="stacked-label">{row.session}</span>
                      <div className="stacked-bar">
                        {row.segments.map((segment) => (
                          <span
                            key={`${row.session}-${segment.regime}`}
                            className="stacked-segment"
                            style={{
                              width: `${(segment.avg / row.total) * 100}%`,
                              background:
                                REGIME_COLORS[segment.regime] ?? "#adb5bd"
                            }}
                            title={`${segment.regime}: ${formatConfidence(segment.avg)}`}
                          />
                        ))}
                      </div>
                      <strong>{formatConfidence(row.total / row.segments.length)}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Regime Confidence Snapshot</div>
              <p className="chart-caption">
                Average confidence per regime (higher means stronger agreement).
              </p>
              <div className="bar-stack">
                {regimeConfidenceRows.map((row) => (
                  <div key={row.label} className="bar-row">
                    <span>{row.label}</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill good"
                        style={{ width: `${(row.value / maxRegimeConfidence) * 100}%` }}
                      />
                    </div>
                    <strong>{formatConfidence(row.value)}</strong>
                  </div>
                ))}
              </div>
            </div>

            <div className="forex-grid">
              <div className="panel chart-panel">
                <div className="chart-title">Trading Session Analysis</div>
                <p className="chart-caption">Signal/scan volume by trading session.</p>
                <div className="color-legend">
                  <span className="legend-dot" style={{ background: "#0f4c5c" }} />
                  <span>Session volume (darker = higher count)</span>
                </div>
                <div className="vertical-bars">
                  {sessionRows.map((row) => (
                    <div key={row.label} className="vertical-bar-item">
                      <div className="vertical-bar-track">
                        <div
                          className="vertical-bar-fill"
                          style={{ height: `${(row.value / maxSession) * 100}%` }}
                        />
                      </div>
                      <span className="vertical-bar-label">{row.label}</span>
                      <strong className="vertical-bar-value">{row.value}</strong>
                    </div>
                  ))}
                </div>
              </div>
              <div className="panel chart-panel">
                <div className="chart-title">Volatility Level Distribution</div>
                <p className="chart-caption">How often each volatility regime appears.</p>
                <div className="color-legend">
                  <span className="legend-dot" style={{ background: VOLATILITY_COLORS.high }} />
                  <span>High volatility</span>
                  <span className="legend-dot" style={{ background: VOLATILITY_COLORS.medium }} />
                  <span>Medium volatility</span>
                  <span className="legend-dot" style={{ background: VOLATILITY_COLORS.low }} />
                  <span>Low volatility</span>
                </div>
                <div className="vertical-bars">
                  {volatilityRows.map((row) => (
                    <div key={row.label} className="vertical-bar-item">
                      <div className="vertical-bar-track">
                        <div
                          className="vertical-bar-fill"
                          style={{
                            height: `${(row.value / maxVolatility) * 100}%`,
                            background: VOLATILITY_COLORS[row.label] ?? "#adb5bd"
                          }}
                        />
                      </div>
                      <span className="vertical-bar-label">{row.label}</span>
                      <strong className="vertical-bar-value">{row.value}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="panel chart-panel">
              <div className="chart-title">Intelligence Sources</div>
              <p className="chart-caption">Which engine produced the insight record.</p>
              <div className="color-legend">
                <span className="legend-dot" style={{ background: "#5bc0be" }} />
                <span>Relative share by source</span>
              </div>
              <div className="bar-stack">
                {sourceRows.map((row) => (
                  <div key={row.label} className="bar-row">
                    <span>{row.label}</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${(row.value / maxSource) * 100}%` }}
                      />
                    </div>
                    <strong>{row.value}</strong>
                  </div>
                ))}
              </div>
            </div>

            {source !== "signal" && recommendedStrategyRows.length ? (
              <div className="panel table-panel">
                <div className="chart-title">Recommended Strategy by Market Conditions</div>
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Regime</th>
                      <th>Avg Regime Confidence</th>
                      <th>Scan Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recommendedStrategyRows.map((row) => (
                      <tr key={`${row.strategy}-${row.regime}`}>
                        <td>{row.strategy}</td>
                        <td>{row.regime}</td>
                        <td>{formatConfidence(row.avg_confidence)}</td>
                        <td>{row.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}

            <div className="panel table-panel">
              <div className="chart-title">Recent Intelligence Records</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Regime</th>
                    <th>Session</th>
                    <th>Volatility</th>
                    <th>Confidence</th>
                    <th>Strategy</th>
                  </tr>
                </thead>
                <tbody>
                  {recentItems.slice(0, 20).map((row, idx) => (
                    <tr key={row.id ?? idx}>
                      <td>
                        {new Date(
                          row.scan_timestamp || row.alert_timestamp || Date.now()
                        ).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit" })}
                      </td>
                      <td>{row.regime ?? "-"}</td>
                      <td>{row.session ?? "-"}</td>
                      <td>{row.volatility_level ?? "-"}</td>
                      <td>
                        {row.regime_confidence != null
                          ? Number(row.regime_confidence).toFixed(2)
                          : "-"}
                      </td>
                      <td>{row.strategy ?? row.recommended_strategy ?? "-"}</td>
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
