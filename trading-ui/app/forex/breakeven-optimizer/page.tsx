"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import EnvironmentToggle from "../../../components/EnvironmentToggle";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";

type PolicyCandidate = {
  trigger: number;
  reachRate: number;
  saveRate: number;
  cutWinnerRisk: number;
  score: number;
};

type PolicyRow = {
  strategy: string;
  epic: string;
  epic_display: string;
  direction: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_mfe: number;
  median_winner_mfe: number;
  p75_winner_mfe: number;
  avg_mae: number;
  p75_mae: number;
  current_be_rate: number;
  recommended_trigger: number | null;
  reach_rate: number;
  save_rate: number;
  cut_winner_risk: number;
  policy_score: number;
  recommendation: string;
  priority: string;
  rationale: string;
  candidates: PolicyCandidate[];
};

type SummaryPayload = {
  groups: number;
  trades: number;
  win_rate: number;
  actionable: number;
  high_priority: number;
  avg_current_be_rate: number;
};

type Payload = {
  source: string;
  days: number;
  env: string;
  strategy_options: string[];
  summary: SummaryPayload | null;
  rows: PolicyRow[];
};

const formatNumber = (value: number | null | undefined, digits = 1) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "N/A";

const formatWhole = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(0) : "N/A";

const formatPercent = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}%` : "N/A";

const priorityTone = (value: string | null | undefined) => {
  if (value === "high") return "warn";
  if (value === "medium") return "off";
  return "on";
};

export default function ForexBreakevenOptimizerPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(60);
  const [strategyFilter, setStrategyFilter] = useState("ALL");
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetch(`/trading/api/forex/breakeven-optimizer/?days=${days}&env=${environment}&strategy=${encodeURIComponent(strategyFilter)}`, { signal: controller.signal })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Failed to load breakeven policy evaluator.");
        }
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [days, environment, strategyFilter]);

  const topPolicyRows = useMemo(
    () => [...(payload?.rows ?? [])].sort((a, b) => b.policy_score - a.policy_score).slice(0, 6),
    [payload?.rows]
  );

  const highRiskRows = useMemo(
    () => [...(payload?.rows ?? [])].sort((a, b) => b.cut_winner_risk - a.cut_winner_risk).slice(0, 6),
    [payload?.rows]
  );

  const summary = payload?.summary;

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
          <h1>Breakeven Optimizer</h1>
          <p>Evaluate live break-even policy candidates by strategy, pair, and direction using MFE, MAE, loser protection, and winner interruption risk.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/breakeven-optimizer" />

      {error ? <div className="error">{error}</div> : null}

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Window</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {[14, 30, 60, 90, 180].map((option) => (
                <option key={option} value={option}>{option}d</option>
              ))}
            </select>
          </div>
          <div>
            <label>Strategy</label>
            <select value={strategyFilter} onChange={(event) => setStrategyFilter(event.target.value)}>
              <option value="ALL">All strategies</option>
              {(payload?.strategy_options ?? []).map((strategy) => (
                <option key={strategy} value={strategy}>{strategy}</option>
              ))}
            </select>
          </div>
          <div className="forex-badge">Source: live trade_log</div>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="summary-card">
          Policy Groups
          <strong>{summary?.groups ?? 0}</strong>
        </div>
        <div className="summary-card">
          Closed Trades
          <strong>{summary?.trades ?? 0}</strong>
        </div>
        <div className="summary-card">
          Actionable Policies
          <strong>{summary?.actionable ?? 0}</strong>
        </div>
        <div className="summary-card">
          High Priority
          <strong>{summary?.high_priority ?? 0}</strong>
        </div>
        <div className="summary-card">
          Win Rate
          <strong>{formatPercent(summary?.win_rate)}</strong>
        </div>
      </div>

      <div className="panel">
        <div className="chart-title">Policy Model</div>
        <div className="stack-list">
          <div className="analysis-card">
            <strong>What this evaluates</strong>
            <p>For each strategy, pair, and direction, candidate BE triggers are scored by loser protection minus winner interruption risk. This avoids blindly raising BE to levels that trades rarely reach.</p>
          </div>
          <div className="analysis-card">
            <strong>Current BE usage</strong>
            <p>Average current BE execution rate is {formatPercent(summary?.avg_current_be_rate)} across visible groups.</p>
          </div>
        </div>
      </div>

      <div className="forex-grid">
        <div className="panel table-panel">
          <div className="chart-title">Best BE Test Candidates</div>
          {loading ? (
            <div className="chart-placeholder">Loading BE policy candidates...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Epic</th>
                  <th>Dir</th>
                  <th>Trigger</th>
                  <th>Save%</th>
                  <th>Cut Risk</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {topPolicyRows.map((row) => (
                  <tr key={`${row.strategy}-${row.epic}-${row.direction}-be`}>
                    <td>{row.strategy}</td>
                    <td>{row.epic_display}</td>
                    <td>{row.direction}</td>
                    <td>{formatWhole(row.recommended_trigger)}</td>
                    <td>{formatPercent(row.save_rate)}</td>
                    <td>{formatPercent(row.cut_winner_risk)}</td>
                    <td>{row.recommendation}</td>
                  </tr>
                ))}
                {!topPolicyRows.length ? (
                  <tr>
                    <td colSpan={7}>No BE policy candidates available.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          )}
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Highest Winner Interruption Risk</div>
          {loading ? (
            <div className="chart-placeholder">Loading risk summary...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Epic</th>
                  <th>Dir</th>
                  <th>Trigger</th>
                  <th>Reach%</th>
                  <th>Cut Risk</th>
                  <th>Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {highRiskRows.map((row) => (
                  <tr key={`${row.strategy}-${row.epic}-${row.direction}-risk`}>
                    <td>{row.strategy}</td>
                    <td>{row.epic_display}</td>
                    <td>{row.direction}</td>
                    <td>{formatWhole(row.recommended_trigger)}</td>
                    <td>{formatPercent(row.reach_rate)}</td>
                    <td>{formatPercent(row.cut_winner_risk)}</td>
                    <td>{row.recommendation}</td>
                  </tr>
                ))}
                {!highRiskRows.length ? (
                  <tr>
                    <td colSpan={7}>No winner interruption risk rows available.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="panel table-panel">
        <div className="chart-title">Live BE Policy Evaluation</div>
        {loading ? (
          <div className="chart-placeholder">Loading policy table...</div>
        ) : (
          <table className="forex-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Epic</th>
                <th>Dir</th>
                <th>Trades</th>
                <th>Win%</th>
                <th>Avg MFE</th>
                <th>Median Winner MFE</th>
                <th>Avg MAE</th>
                <th>Candidate BE</th>
                <th>Reach%</th>
                <th>Save%</th>
                <th>Cut Risk</th>
                <th>Action</th>
                <th>Priority</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.rows ?? []).map((row) => (
                <tr key={`${row.strategy}-${row.epic}-${row.direction}-be-table`}>
                  <td>{row.strategy}</td>
                  <td>{row.epic_display}</td>
                  <td>{row.direction}</td>
                  <td>{row.trades}</td>
                  <td>{formatPercent(row.win_rate)}</td>
                  <td>{formatWhole(row.avg_mfe)}</td>
                  <td>{formatWhole(row.median_winner_mfe)}</td>
                  <td>{formatWhole(row.avg_mae)}</td>
                  <td>{formatWhole(row.recommended_trigger)}</td>
                  <td>{formatPercent(row.reach_rate)}</td>
                  <td>{formatPercent(row.save_rate)}</td>
                  <td>{formatPercent(row.cut_winner_risk)}</td>
                  <td>{row.recommendation}</td>
                  <td>
                    <span className={`status-pill ${priorityTone(row.priority)}`}>{row.priority.toUpperCase()}</span>
                  </td>
                </tr>
              ))}
              {!payload?.rows?.length ? (
                <tr>
                  <td colSpan={14}>No live BE policy rows found.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel table-panel">
        <div className="chart-title">Candidate Trigger Breakdown</div>
        {loading ? (
          <div className="chart-placeholder">Loading candidate table...</div>
        ) : (
          <table className="forex-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Epic</th>
                <th>Dir</th>
                <th>Trigger</th>
                <th>Reach%</th>
                <th>Save%</th>
                <th>Cut Risk</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.rows ?? []).flatMap((row) =>
                row.candidates.slice(0, 4).map((candidate) => (
                  <tr key={`${row.strategy}-${row.epic}-${row.direction}-${candidate.trigger}`}>
                    <td>{row.strategy}</td>
                    <td>{row.epic_display}</td>
                    <td>{row.direction}</td>
                    <td>{formatWhole(candidate.trigger)}</td>
                    <td>{formatPercent(candidate.reachRate)}</td>
                    <td>{formatPercent(candidate.saveRate)}</td>
                    <td>{formatPercent(candidate.cutWinnerRisk)}</td>
                    <td>{formatNumber(candidate.score, 0)}</td>
                  </tr>
                ))
              )}
              {!payload?.rows?.length ? (
                <tr>
                  <td colSpan={8}>No candidate rows found.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <div className="chart-title">Policy Notes</div>
        <div className="stack-list">
          {(payload?.rows ?? []).map((row) => (
            <div key={`${row.strategy}-${row.epic}-${row.direction}-detail`} className="analysis-card">
              <strong>
                {row.strategy} {row.epic_display} {row.direction} ({row.trades} trades)
              </strong>
              <p>
                {row.rationale}
              </p>
              <p>
                Winner profile: median winner MFE {formatNumber(row.median_winner_mfe, 1)}p, p75 winner MFE {formatNumber(row.p75_winner_mfe, 1)}p.
                Heat profile: avg MAE {formatNumber(row.avg_mae, 1)}p, p75 MAE {formatNumber(row.p75_mae, 1)}p.
              </p>
              <p>
                Current BE execution rate is {formatPercent(row.current_be_rate)}. Priority{" "}
                <span className={`status-pill ${priorityTone(row.priority)}`}>{row.priority.toUpperCase()}</span>.
              </p>
            </div>
          ))}
          {!loading && !payload?.rows?.length ? (
            <div className="analysis-card">
              <strong>No live policy rows available</strong>
              <p>No closed trades matched the current environment, window, and strategy filters.</p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
