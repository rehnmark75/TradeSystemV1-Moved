"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type AlignmentRow = {
  alignment: string;
  total_signals: number;
  total_trades: number;
  wins: number;
  losses: number;
  breakeven: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  trade_rate: number;
  sample_confidence: "HIGH" | "MEDIUM" | "LOW";
  recommendation: string;
};

type PatternRow = {
  pattern: string;
  signal_type: string;
  total_signals: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  trade_rate: number;
  sample_confidence: "HIGH" | "MEDIUM" | "LOW";
  recommendation: string;
};

type PairRow = {
  pair: string | null;
  alignment: string;
  total_signals: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  trade_rate: number;
  sample_confidence: "HIGH" | "MEDIUM" | "LOW";
  recommendation: string;
};

type RegimeRow = AlignmentRow & {
  regime: string;
};

type ClaudeRow = AlignmentRow & {
  claude_bucket: string;
};

type DistributionRow = {
  direction: string;
  signal_type: string;
  count: number;
};

type HtfDecision = {
  verdict: string;
  action: string;
  aligned_expectancy: number;
  counter_expectancy: number;
  expectancy_delta: number;
  win_rate_delta: number;
  pnl_impact_if_counter_blocked: number;
  counter_trades: number;
  missed_winners: number;
  avoided_losers: number;
  sample_confidence: "HIGH" | "MEDIUM" | "LOW";
};

type HtfPayload = {
  days: number;
  decision: HtfDecision;
  alignment: AlignmentRow[];
  patterns: PatternRow[];
  pairs: PairRow[];
  regimes: RegimeRow[];
  claude: ClaudeRow[];
  distribution: DistributionRow[];
};

const DAY_OPTIONS = [7, 14, 30, 60, 90];

const formatNumber = (value: number, digits = 1) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "0.0";

const formatPercent = (value: number) => (Number.isFinite(value) ? `${value.toFixed(1)}%` : "0%");

const tagClass = (value: string) => `htf-tag htf-tag-${value.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

const pnlClass = (value: number) =>
  value > 0 ? "htf-positive" : value < 0 ? "htf-negative" : "";

export default function ForexHtfAnalysisPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(30);
  const [payload, setPayload] = useState<HtfPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/htf-analysis/?days=${days}&env=${environment}`, {
      signal: controller.signal,
    })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Failed to load HTF analysis.");
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [days, environment]);

  const summary = useMemo(() => {
    const aligned = (payload?.alignment ?? []).find((row) => row.alignment === "ALIGNED");
    const counter = (payload?.alignment ?? []).find((row) => row.alignment === "COUNTER");
    return {
      alignedSignals: aligned?.total_signals ?? 0,
      counterSignals: counter?.total_signals ?? 0,
      alignedWinRate: aligned?.win_rate ?? 0,
      counterWinRate: counter?.win_rate ?? 0,
      alignedExpectancy: aligned?.avg_pnl ?? 0,
      counterExpectancy: counter?.avg_pnl ?? 0,
      pnlDiff: (aligned?.total_pnl ?? 0) - (counter?.total_pnl ?? 0),
    };
  }, [payload?.alignment]);

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
          <h1>HTF Analysis</h1>
          <p>Measure whether 4H candle alignment at signal time changes trade quality, pair behavior, and pattern edge.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/htf-analysis" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Analysis Window</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {DAY_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}d
                </option>
              ))}
            </select>
          </div>
          <div className="forex-badge">
            Environment
            <strong>{environment.toUpperCase()}</strong>
          </div>
          <div className="forex-badge">
            Pattern Rows
            <strong>{payload?.patterns?.length ?? 0}</strong>
          </div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        <div className="metrics-grid">
          <div className="summary-card">
            Aligned Signals
            <strong>{summary.alignedSignals}</strong>
          </div>
          <div className="summary-card">
            Counter Signals
            <strong>{summary.counterSignals}</strong>
          </div>
          <div className="summary-card">
            Aligned Win Rate
            <strong>{formatPercent(summary.alignedWinRate)}</strong>
          </div>
          <div className="summary-card">
            Counter Win Rate
            <strong>{formatPercent(summary.counterWinRate)}</strong>
          </div>
          <div className="summary-card">
            P&L Difference
            <strong>{formatNumber(summary.pnlDiff, 2)}</strong>
          </div>
        </div>

        <div className="htf-decision-grid">
          <div className="htf-decision-card htf-decision-primary">
            <span>Current HTF Verdict</span>
            <strong>{payload?.decision?.verdict ?? "Loading"}</strong>
            <p>{payload?.decision?.action ?? "Waiting for enough HTF outcome data."}</p>
          </div>
          <div className="htf-decision-card">
            <span>Expectancy Delta</span>
            <strong className={pnlClass(payload?.decision?.expectancy_delta ?? 0)}>
              {formatNumber(payload?.decision?.expectancy_delta ?? 0, 2)}
            </strong>
            <p>Aligned minus counter average P&amp;L per executed trade.</p>
          </div>
          <div className="htf-decision-card">
            <span>Block Counter Impact</span>
            <strong className={pnlClass(payload?.decision?.pnl_impact_if_counter_blocked ?? 0)}>
              {formatNumber(payload?.decision?.pnl_impact_if_counter_blocked ?? 0, 2)}
            </strong>
            <p>
              Removed trades: {payload?.decision?.counter_trades ?? 0}; avoided losers:{" "}
              {payload?.decision?.avoided_losers ?? 0}; missed winners:{" "}
              {payload?.decision?.missed_winners ?? 0}.
            </p>
          </div>
          <div className="htf-decision-card">
            <span>Sample Confidence</span>
            <strong>{payload?.decision?.sample_confidence ?? "LOW"}</strong>
            <p>Based on the smaller executed-trade sample between aligned and counter.</p>
          </div>
        </div>

        <div className="forex-grid">
          <div className="panel table-panel">
            <div className="chart-title">Alignment Summary</div>
            {loading ? (
              <div className="chart-placeholder">Loading HTF alignment...</div>
            ) : (
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Alignment</th>
                    <th>Signals</th>
                    <th>Trades</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>BE</th>
                    <th>Win Rate</th>
                    <th>Trade Rate</th>
                    <th>Expectancy</th>
                    <th>Profit Factor</th>
                    <th>Total P&amp;L</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.alignment ?? []).map((row) => (
                    <tr key={row.alignment}>
                      <td>{row.alignment}</td>
                      <td>{row.total_signals}</td>
                      <td>{row.total_trades}</td>
                      <td>{row.wins}</td>
                      <td>{row.losses}</td>
                      <td>{row.breakeven}</td>
                      <td>{formatPercent(row.win_rate)}</td>
                      <td>{formatPercent(row.trade_rate)}</td>
                      <td className={pnlClass(row.avg_pnl)}>{formatNumber(row.avg_pnl, 2)}</td>
                      <td>{formatNumber(row.profit_factor, 2)}</td>
                      <td className={pnlClass(row.total_pnl)}>{formatNumber(row.total_pnl, 2)}</td>
                      <td>
                        <span className={tagClass(row.recommendation)}>{row.recommendation}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="panel table-panel">
            <div className="chart-title">Direction Distribution</div>
            {loading ? (
              <div className="chart-placeholder">Loading distribution...</div>
            ) : (
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>4H Direction</th>
                    <th>Signal Type</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.distribution ?? []).map((row, index) => (
                    <tr key={`${row.direction}-${row.signal_type}-${index}`}>
                      <td>{row.direction}</td>
                      <td>{row.signal_type}</td>
                      <td>{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        <div className="forex-grid">
          <div className="panel table-panel">
            <div className="chart-title">Claude Approval vs HTF</div>
            {loading ? (
              <div className="chart-placeholder">Loading Claude comparison...</div>
            ) : (
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Claude Bucket</th>
                    <th>Alignment</th>
                    <th>Signals</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Expectancy</th>
                    <th>Total P&amp;L</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.claude ?? []).map((row) => (
                    <tr key={`${row.claude_bucket}-${row.alignment}`}>
                      <td>{row.claude_bucket}</td>
                      <td>{row.alignment}</td>
                      <td>{row.total_signals}</td>
                      <td>{row.total_trades}</td>
                      <td>{formatPercent(row.win_rate)}</td>
                      <td className={pnlClass(row.avg_pnl)}>{formatNumber(row.avg_pnl, 2)}</td>
                      <td className={pnlClass(row.total_pnl)}>{formatNumber(row.total_pnl, 2)}</td>
                      <td>
                        <span className={tagClass(row.recommendation)}>{row.recommendation}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="panel table-panel">
            <div className="chart-title">Regime Split</div>
            {loading ? (
              <div className="chart-placeholder">Loading regime split...</div>
            ) : (
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Regime</th>
                    <th>Alignment</th>
                    <th>Signals</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Expectancy</th>
                    <th>Total P&amp;L</th>
                    <th>Sample</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.regimes ?? []).map((row) => (
                    <tr key={`${row.regime}-${row.alignment}`}>
                      <td>{row.regime}</td>
                      <td>{row.alignment}</td>
                      <td>{row.total_signals}</td>
                      <td>{row.total_trades}</td>
                      <td>{formatPercent(row.win_rate)}</td>
                      <td className={pnlClass(row.avg_pnl)}>{formatNumber(row.avg_pnl, 2)}</td>
                      <td className={pnlClass(row.total_pnl)}>{formatNumber(row.total_pnl, 2)}</td>
                      <td>{row.sample_confidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Two-Candle Patterns</div>
          {loading ? (
            <div className="chart-placeholder">Loading pattern analysis...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Pattern</th>
                  <th>Signal</th>
                  <th>Signals</th>
                  <th>Trades</th>
                  <th>Wins</th>
                  <th>Losses</th>
                  <th>Win Rate</th>
                  <th>Expectancy</th>
                  <th>PF</th>
                  <th>Sample</th>
                  <th>Total P&amp;L</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                {(payload?.patterns ?? []).map((row) => (
                  <tr key={`${row.pattern}-${row.signal_type}`}>
                    <td>{row.pattern}</td>
                    <td>{row.signal_type}</td>
                    <td>{row.total_signals}</td>
                    <td>{row.total_trades}</td>
                    <td>{row.wins}</td>
                    <td>{row.losses}</td>
                    <td>{formatPercent(row.win_rate)}</td>
                    <td className={pnlClass(row.avg_pnl)}>{formatNumber(row.avg_pnl, 2)}</td>
                    <td>{formatNumber(row.profit_factor, 2)}</td>
                    <td>{row.sample_confidence}</td>
                    <td className={pnlClass(row.total_pnl)}>{formatNumber(row.total_pnl, 2)}</td>
                    <td>
                      <span className={tagClass(row.recommendation)}>{row.recommendation}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="panel table-panel">
          <div className="chart-title">By Pair</div>
          {loading ? (
            <div className="chart-placeholder">Loading pair breakdown...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Pair</th>
                  <th>Alignment</th>
                  <th>Signals</th>
                  <th>Trades</th>
                  <th>Wins</th>
                  <th>Losses</th>
                  <th>Win Rate</th>
                  <th>Expectancy</th>
                  <th>PF</th>
                  <th>Total P&amp;L</th>
                  <th>Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {(payload?.pairs ?? []).map((row, index) => (
                  <tr key={`${row.pair}-${row.alignment}-${index}`}>
                    <td>{row.pair ?? "N/A"}</td>
                    <td>{row.alignment}</td>
                    <td>{row.total_signals}</td>
                    <td>{row.total_trades}</td>
                    <td>{row.wins}</td>
                    <td>{row.losses}</td>
                    <td>{formatPercent(row.win_rate)}</td>
                    <td className={pnlClass(row.avg_pnl)}>{formatNumber(row.avg_pnl, 2)}</td>
                    <td>{formatNumber(row.profit_factor, 2)}</td>
                    <td className={pnlClass(row.total_pnl)}>{formatNumber(row.total_pnl, 2)}</td>
                    <td>
                      <span className={tagClass(row.recommendation)}>{row.recommendation}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
