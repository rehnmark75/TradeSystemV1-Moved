"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import EnvironmentToggle from "../../../components/EnvironmentToggle";
import ForexNav from "../_components/ForexNav";

type SummaryRow = {
  total_unfilled: number;
  would_fill_4h: number;
  would_fill_24h: number;
  good_signals: number;
  bad_signals: number;
  inconclusive_signals: number;
  win_rate_pct: number | null;
};

type DetailRow = {
  id: number;
  symbol: string;
  direction: string;
  order_time: string;
  expiry_time: string | null;
  entry_level: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  price_at_expiry: number | null;
  gap_to_entry_pips: number | null;
  would_fill_4h: boolean | null;
  outcome_4h: string | null;
  would_fill_24h: boolean | null;
  outcome_24h: string | null;
  signal_quality: string | null;
  max_favorable_pips: number | null;
  max_adverse_pips: number | null;
  alert_id: number | null;
};

type EpicBreakdownRow = {
  symbol: string;
  total_unfilled: number;
  good: number;
  bad: number;
  inconclusive: number;
  avg_gap_pips: number | null;
  avg_favorable: number | null;
  avg_adverse: number | null;
};

type RecommendationRow = EpicBreakdownRow & {
  issues: string[];
  recommendations: string[];
};

type UnfilledOrdersPayload = {
  viewExists: boolean;
  summary: SummaryRow | null;
  detail: DetailRow[];
  epicBreakdown: EpicBreakdownRow[];
  recommendations: RecommendationRow[];
};

const formatNumber = (value: number | null | undefined, digits = 1) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "N/A";

const formatPercent = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(0)}%` : "N/A";

const formatBool = (value: boolean | null | undefined) => {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return "N/A";
};

const formatDate = (value: string | null | undefined) => {
  if (!value) return "N/A";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("sv-SE", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const qualityTone = (quality: string | null | undefined) => {
  if (quality === "GOOD_SIGNAL") return "on";
  if (quality === "BAD_SIGNAL") return "warn";
  return "off";
};

export default function ForexUnfilledOrdersPage() {
  const [payload, setPayload] = useState<UnfilledOrdersPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetch("/trading/api/forex/unfilled-orders/", { signal: controller.signal })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Failed to load unfilled order analysis.");
        }
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, []);

  const summary = payload?.summary;

  const insightSummary = useMemo(() => {
    const epicRows = payload?.epicBreakdown ?? [];
    return {
      epicsTracked: epicRows.length,
      highGapEpics: epicRows.filter((row) => (row.avg_gap_pips ?? 0) > 5).length,
      directionRiskEpics: epicRows.filter((row) => row.bad > row.good && row.good + row.bad > 0).length,
      expiryCandidates: epicRows.filter((row) => row.good > 0 && row.bad === 0).length,
    };
  }, [payload?.epicBreakdown]);

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
          <h1>Unfilled Orders</h1>
          <p>Review stop-entry orders that expired without filling and separate missed winners from setups that were better left untouched.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/unfilled-orders" />

      {error ? <div className="error">{error}</div> : null}

      {!loading && payload && !payload.viewExists ? (
        <div className="panel">
          <div className="chart-title">Unavailable</div>
          <div className="chart-placeholder">
            The `v_unfilled_order_analysis` view is not present in the forex database yet.
          </div>
        </div>
      ) : null}

      <div className="metrics-grid">
        <div className="summary-card">
          Total Unfilled
          <strong>{summary?.total_unfilled ?? 0}</strong>
        </div>
        <div className="summary-card">
          Would Fill 4H
          <strong>{summary?.would_fill_4h ?? 0}</strong>
        </div>
        <div className="summary-card">
          Good Signals
          <strong>{summary?.good_signals ?? 0}</strong>
        </div>
        <div className="summary-card">
          Bad Signals
          <strong>{summary?.bad_signals ?? 0}</strong>
        </div>
        <div className="summary-card">
          Win Rate If Filled
          <strong>{formatPercent(summary?.win_rate_pct)}</strong>
        </div>
      </div>

      <div className="forex-grid">
        <div className="panel table-panel">
          <div className="chart-title">Summary</div>
          {loading ? (
            <div className="chart-placeholder">Loading unfilled order summary...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Total</th>
                  <th>Would Fill 4H</th>
                  <th>Would Fill 24H</th>
                  <th>Good</th>
                  <th>Bad</th>
                  <th>Inconclusive</th>
                  <th>Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {summary ? (
                  <tr>
                    <td>{summary.total_unfilled}</td>
                    <td>{summary.would_fill_4h}</td>
                    <td>{summary.would_fill_24h}</td>
                    <td>{summary.good_signals}</td>
                    <td>{summary.bad_signals}</td>
                    <td>{summary.inconclusive_signals}</td>
                    <td>{formatPercent(summary.win_rate_pct)}</td>
                  </tr>
                ) : (
                  <tr>
                    <td colSpan={7}>No summary data available.</td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Insight Snapshot</div>
          {loading ? (
            <div className="chart-placeholder">Loading recommendations...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Tracked epics</td>
                  <td>{insightSummary.epicsTracked}</td>
                </tr>
                <tr>
                  <td>High-gap epics</td>
                  <td>{insightSummary.highGapEpics}</td>
                </tr>
                <tr>
                  <td>Direction-risk epics</td>
                  <td>{insightSummary.directionRiskEpics}</td>
                </tr>
                <tr>
                  <td>Expiry-extension candidates</td>
                  <td>{insightSummary.expiryCandidates}</td>
                </tr>
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="panel table-panel">
        <div className="chart-title">Detailed Analysis</div>
        {loading ? (
          <div className="chart-placeholder">Loading order detail...</div>
        ) : (
          <table className="forex-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Epic</th>
                <th>Dir</th>
                <th>Order Time</th>
                <th>Gap</th>
                <th>Fill 4H</th>
                <th>4H Outcome</th>
                <th>Fill 24H</th>
                <th>24H Outcome</th>
                <th>Quality</th>
                <th>Fav Move</th>
                <th>Adv Move</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.detail ?? []).map((row) => (
                <tr key={row.id}>
                  <td>{row.id}</td>
                  <td>{row.symbol}</td>
                  <td>{row.direction}</td>
                  <td>{formatDate(row.order_time)}</td>
                  <td>{formatNumber(row.gap_to_entry_pips)}</td>
                  <td>{formatBool(row.would_fill_4h)}</td>
                  <td>{row.outcome_4h ?? "N/A"}</td>
                  <td>{formatBool(row.would_fill_24h)}</td>
                  <td>{row.outcome_24h ?? "N/A"}</td>
                  <td>
                    <span className={`status-pill ${qualityTone(row.signal_quality)}`}>
                      {row.signal_quality ?? "UNKNOWN"}
                    </span>
                  </td>
                  <td>{formatNumber(row.max_favorable_pips)}</td>
                  <td>{formatNumber(row.max_adverse_pips)}</td>
                </tr>
              ))}
              {!payload?.detail?.length ? (
                <tr>
                  <td colSpan={12}>No unfilled orders found.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel table-panel">
        <div className="chart-title">Per-Epic Breakdown</div>
        {loading ? (
          <div className="chart-placeholder">Loading epic breakdown...</div>
        ) : (
          <table className="forex-table">
            <thead>
              <tr>
                <th>Epic</th>
                <th>Total</th>
                <th>Good</th>
                <th>Bad</th>
                <th>Inconclusive</th>
                <th>Avg Gap</th>
                <th>Avg Favorable</th>
                <th>Avg Adverse</th>
                <th>Win Rate</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.epicBreakdown ?? []).map((row) => {
                const decisive = row.good + row.bad;
                const winRate = decisive > 0 ? `${((row.good / decisive) * 100).toFixed(0)}%` : "N/A";

                return (
                  <tr key={row.symbol}>
                    <td>{row.symbol}</td>
                    <td>{row.total_unfilled}</td>
                    <td>{row.good}</td>
                    <td>{row.bad}</td>
                    <td>{row.inconclusive}</td>
                    <td>{formatNumber(row.avg_gap_pips)}</td>
                    <td>{formatNumber(row.avg_favorable)}</td>
                    <td>{formatNumber(row.avg_adverse)}</td>
                    <td>{winRate}</td>
                  </tr>
                );
              })}
              {!payload?.epicBreakdown?.length ? (
                <tr>
                  <td colSpan={9}>No epic breakdown data available.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <div className="chart-title">Recommendations</div>
        <div className="stack-list">
          <div className="analysis-card">
            <strong>Key questions</strong>
            <p>Are entries too far from market, are signals directionally wrong, or are otherwise-valid setups expiring before the retracement happens?</p>
          </div>
          {(payload?.recommendations ?? []).map((row) => (
            <div key={row.symbol} className="analysis-card">
              <strong>
                {row.symbol} ({row.total_unfilled} unfilled)
              </strong>
              <p>
                Avg gap {formatNumber(row.avg_gap_pips)} pips, decisive record {row.good} good / {row.bad} bad.
              </p>
              <p>{row.issues.length ? row.issues.join(" ") : "No significant issues detected. Continue monitoring."}</p>
              {row.recommendations.length ? <p>{row.recommendations.join(" ")}</p> : null}
            </div>
          ))}
          {!loading && !payload?.recommendations?.length ? (
            <div className="analysis-card">
              <strong>No recommendations yet</strong>
              <p>There is not enough repeated unfilled-order data per epic to make a stable recommendation.</p>
            </div>
          ) : null}
        </div>
      </div>

      <div className="panel">
        <div className="chart-title">Order Drilldown</div>
        <div className="stack-list">
          {(payload?.detail ?? []).slice(0, 12).map((row) => (
            <div key={`detail-${row.id}`} className="analysis-card">
              <strong>
                {row.symbol} {row.direction} at {formatDate(row.order_time)}
              </strong>
              <p>
                Entry {formatNumber(row.entry_level, 4)}, stop {formatNumber(row.stop_loss, 4)}, target {formatNumber(row.take_profit, 4)}.
              </p>
              <p>
                Expired at {formatDate(row.expiry_time)} with market price {formatNumber(row.price_at_expiry, 4)} and a gap of{" "}
                {formatNumber(row.gap_to_entry_pips)} pips.
              </p>
              <p>
                24H fill: {formatBool(row.would_fill_24h)}. Outcome: {row.outcome_24h ?? "N/A"}. Favorable move{" "}
                {formatNumber(row.max_favorable_pips)} pips, adverse move {formatNumber(row.max_adverse_pips)} pips.
              </p>
            </div>
          ))}
          {!loading && !payload?.detail?.length ? (
            <div className="analysis-card">
              <strong>No unfilled orders found</strong>
              <p>This usually means stop-entry orders are filling as expected, which is the preferred state.</p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
