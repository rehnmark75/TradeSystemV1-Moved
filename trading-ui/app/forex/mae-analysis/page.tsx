/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type MaeTrade = {
  id: number;
  symbol: string;
  symbol_short: string;
  direction: string;
  entry_price: number | null;
  sl_price: number | null;
  timestamp: string;
  status: string;
  profit_loss: number | null;
  mfe_pips: number | null;
  mae_pips: number | null;
  mae_price: number | null;
  mae_time: string | null;
  stop_distance_pips: number | null;
  vsl_stage: string | null;
  result: string;
  mae_pct_of_stop: number | null;
};

type MaeSummary = {
  symbol: string;
  symbol_short: string;
  total_trades: number;
  win_rate: number;
  avg_mae_pips: number;
  median_mae_pips: number;
  p75_mae_pips: number;
  p90_mae_pips: number;
  max_mae_pips: number;
  avg_mfe_pips: number;
  avg_stop_setting: number;
};

type MaePayload = {
  trades: MaeTrade[];
  summary: MaeSummary[];
};

type PairRecommendation = {
  pair: string;
  completedTrades: number;
  winners: number;
  losers: number;
  currentStop: number | null;
  medianWinnerMae: number | null;
  p75WinnerMae: number | null;
  medianLoserMae: number | null;
  recommendation: string;
  rationale: string;
  confidence: "LOW" | "MEDIUM" | "HIGH";
  stopScenarios: {
    stop: number;
    winnersStoppedPct: number | null;
    losersStoppedPct: number | null;
  }[];
};

const DAY_OPTIONS = [1, 3, 7, 14, 30];

const STOP_SCENARIOS = [5, 10, 15, 20, 25, 30, 40];

const formatNumber = (value: number, digits = 1) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      })
    : "0.0";

const formatDateTime = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit"
  });
};

const percentile = (values: number[], pct: number) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * pct;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
};

const average = (values: number[]) => {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

const unique = <T,>(values: T[]) => Array.from(new Set(values));

export default function ForexMaeAnalysisPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(7);
  const [payload, setPayload] = useState<MaePayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [pairFilter, setPairFilter] = useState<string[]>([]);
  const [resultFilter, setResultFilter] = useState<string[]>([]);
  const [focusPair, setFocusPair] = useState<string>("ALL");

  const loadMae = () => {
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/mae-analysis/?days=${days}&env=${environment}`)
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load MAE analysis."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadMae();
  }, [days, environment]);

  const trades = payload?.trades ?? [];
  const summary = payload?.summary ?? [];

  const availablePairs = useMemo(
    () => Array.from(new Set(trades.map((trade) => trade.symbol_short))).filter(Boolean),
    [trades]
  );
  const availableResults = useMemo(
    () => Array.from(new Set(trades.map((trade) => trade.result))).filter(Boolean),
    [trades]
  );

  useEffect(() => {
    if (!pairFilter.length && availablePairs.length) setPairFilter(availablePairs);
  }, [availablePairs, pairFilter.length]);
  useEffect(() => {
    if (!resultFilter.length && availableResults.length) setResultFilter(availableResults);
  }, [availableResults, resultFilter.length]);
  useEffect(() => {
    if (focusPair !== "ALL" && availablePairs.length && !availablePairs.includes(focusPair)) {
      setFocusPair("ALL");
    }
  }, [availablePairs, focusPair]);

  const filteredTrades = trades.filter((trade) => {
    if (pairFilter.length && !pairFilter.includes(trade.symbol_short)) return false;
    if (resultFilter.length && !resultFilter.includes(trade.result)) return false;
    return true;
  });

  const pairRecommendations = useMemo<PairRecommendation[]>(() => {
    const pairs = unique(filteredTrades.map((trade) => trade.symbol_short)).filter(Boolean);

    const confidenceFor = (completedTrades: number, winners: number): "LOW" | "MEDIUM" | "HIGH" => {
      if (completedTrades >= 20 && winners >= 8) return "HIGH";
      if (completedTrades >= 8 && winners >= 4) return "MEDIUM";
      return "LOW";
    };

    return pairs
      .map((pair) => {
        const pairTrades = filteredTrades.filter(
          (trade) => trade.symbol_short === pair && trade.result !== "OPEN" && trade.result !== "PENDING"
        );
        const winners = pairTrades.filter((trade) => trade.result === "WIN");
        const losers = pairTrades.filter((trade) => trade.result === "LOSS");

        const winnerMae = winners
          .map((trade) => trade.mae_pips)
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
        const loserMae = losers
          .map((trade) => trade.mae_pips)
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
        const pairStops = pairTrades
          .map((trade) => trade.stop_distance_pips)
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value));

        const currentStop = average(pairStops);
        const medianWinnerMae = percentile(winnerMae, 0.5);
        const p75WinnerMae = percentile(winnerMae, 0.75);
        const medianLoserMae = percentile(loserMae, 0.5);
        const confidence = confidenceFor(pairTrades.length, winners.length);

        let recommendation = "WAIT_FOR_MORE_DATA";
        let rationale = "Not enough completed trades and winners to trust a pair-level stop recommendation yet.";

        if (pairTrades.length >= 8 && winners.length >= 4 && currentStop != null && p75WinnerMae != null) {
          if (p75WinnerMae > currentStop * 1.05) {
            recommendation = "TEST_WIDER_STOP";
            rationale = `75% of winners need about ${formatNumber(p75WinnerMae)} pips of room versus a current average stop of ${formatNumber(currentStop)}.`;
          } else if (
            medianWinnerMae != null &&
            medianLoserMae != null &&
            medianWinnerMae < currentStop * 0.55 &&
            medianLoserMae >= currentStop * 0.8
          ) {
            recommendation = "TEST_TIGHTER_STOP";
            rationale = `Winners usually stay clean while losers still use most of the stop. This pair may tolerate a tighter stop.`;
          } else if (
            medianWinnerMae != null &&
            medianLoserMae != null &&
            Math.abs(medianWinnerMae - medianLoserMae) <= Math.max(2, currentStop * 0.15)
          ) {
            recommendation = "ENTRY_QUALITY_ISSUE";
            rationale = "Winners and losers take similar heat, so changing stop width alone is less likely to help.";
          } else {
            recommendation = "KEEP_CURRENT_STOP";
            rationale = "Winner heat stays inside the current stop envelope without a strong case for widening or tightening.";
          }
        } else if (pairTrades.length === 0) {
          rationale = "No completed trades for this pair in the current slice.";
        } else if (winners.length < 4) {
          rationale = `Only ${winners.length} winners available. The page needs at least 4 winners for a meaningful pair-level read.`;
        } else if (pairTrades.length < 8) {
          rationale = `Only ${pairTrades.length} completed trades available. More sample size is needed before changing stops.`;
        } else if (currentStop == null) {
          rationale = "Stored stop distance is missing for this pair, so stop-versus-MAE comparison is blocked.";
        }

        return {
          pair,
          completedTrades: pairTrades.length,
          winners: winners.length,
          losers: losers.length,
          currentStop,
          medianWinnerMae,
          p75WinnerMae,
          medianLoserMae,
          recommendation,
          rationale,
          confidence,
          stopScenarios: STOP_SCENARIOS.map((stop) => ({
            stop,
            winnersStoppedPct: winnerMae.length
              ? (winnerMae.filter((value) => value >= stop).length / winnerMae.length) * 100
              : null,
            losersStoppedPct: loserMae.length
              ? (loserMae.filter((value) => value >= stop).length / loserMae.length) * 100
              : null
          }))
        };
      })
      .sort((a, b) => {
        const confidenceOrder = { HIGH: 0, MEDIUM: 1, LOW: 2 };
        return (
          confidenceOrder[a.confidence] - confidenceOrder[b.confidence] ||
          b.completedTrades - a.completedTrades ||
          a.pair.localeCompare(b.pair)
        );
      });
  }, [filteredTrades]);

  const focusedRecommendation = useMemo(() => {
    if (focusPair === "ALL") return pairRecommendations[0] ?? null;
    return pairRecommendations.find((item) => item.pair === focusPair) ?? null;
  }, [focusPair, pairRecommendations]);

  const interpreted = useMemo(() => {
    const completedTrades = filteredTrades.filter((trade) => trade.result !== "OPEN" && trade.result !== "PENDING");
    const winnerTrades = completedTrades.filter((trade) => trade.result === "WIN");
    const loserTrades = completedTrades.filter((trade) => trade.result === "LOSS");

    const winnerMae = winnerTrades
      .map((trade) => trade.mae_pips)
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const loserMae = loserTrades
      .map((trade) => trade.mae_pips)
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const allMae = completedTrades
      .map((trade) => trade.mae_pips)
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const allStops = completedTrades
      .map((trade) => trade.stop_distance_pips)
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const winnerMfe = winnerTrades
      .map((trade) => trade.mfe_pips)
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));

    const currentStop = average(allStops);
    const medianWinnerMae = percentile(winnerMae, 0.5);
    const p75WinnerMae = percentile(winnerMae, 0.75);
    const p95WinnerMae = percentile(winnerMae, 0.95);
    const medianLoserMae = percentile(loserMae, 0.5);
    const p75LoserMae = percentile(loserMae, 0.75);
    const avgLoserMae = average(loserMae);
    const avgWinnerMfe = average(winnerMfe);
    const avgMae = average(allMae);

    const winnersAboveCurrentStop =
      currentStop && winnerMae.length
        ? (winnerMae.filter((value) => value >= currentStop).length / winnerMae.length) * 100
        : null;

    const stopScenarios = STOP_SCENARIOS.map((stop) => ({
      stop,
      winnersStoppedPct: winnerMae.length
        ? (winnerMae.filter((value) => value >= stop).length / winnerMae.length) * 100
        : null,
      losersStoppedPct: loserMae.length
        ? (loserMae.filter((value) => value >= stop).length / loserMae.length) * 100
        : null
    }));

    let interpretationTitle = "Need more data";
    let interpretationBody =
      "There are not enough completed trades in the current filter slice to say whether stop placement or entry quality is the main issue.";
    let interpretationMeta = `Window: ${days}d, environment: ${environment.toUpperCase()}, completed trades: ${completedTrades.length}, winners: ${winnerTrades.length}, losers: ${loserTrades.length}.`;

    if (completedTrades.length === 0) {
      interpretationTitle = "No completed scalp trades in this slice";
      interpretationBody = `No completed scalp trades were found in ${environment.toUpperCase()} for the last ${days} days with the current filters. Try a wider window, switch environment, or clear filters.`;
    } else if (completedTrades.length < 8) {
      interpretationTitle = "Too few completed trades";
      interpretationBody = `Only ${completedTrades.length} completed trades are available. The page needs at least 8 completed trades before stop-versus-entry conclusions are worth trusting.`;
    } else if (winnerMae.length < 4) {
      interpretationTitle = "Too few winners";
      interpretationBody = `There are ${completedTrades.length} completed trades, but only ${winnerMae.length} winners. The page needs at least 4 winners to estimate how much heat valid setups usually need before they work.`;
    } else if (currentStop == null) {
      interpretationTitle = "No stop data available";
      interpretationBody =
        "Completed trades exist, but the current slice does not have enough stored stop values to compare winner MAE against the stop that was actually used.";
    } else if (p75WinnerMae != null) {
      if (p75WinnerMae > currentStop * 1.05) {
        interpretationTitle = "Stop likely too tight";
        interpretationBody = `${formatNumber(
          p75WinnerMae
        )} pips covers 75% of winner heat, while the current average stop is only ${formatNumber(
          currentStop
        )}. Many valid trades may be getting cut before they recover.`;
      } else if (medianWinnerMae < currentStop * 0.55 && (medianLoserMae ?? 0) >= currentStop * 0.8) {
        interpretationTitle = "Stop may be wider than needed";
        interpretationBody = `Winning trades usually stay relatively clean, with median winner MAE around ${formatNumber(
          medianWinnerMae
        )} pips against a ${formatNumber(
          currentStop
        )} pip stop. This suggests room to test tighter risk without obviously choking winners.`;
      } else if (
        medianWinnerMae != null &&
        medianLoserMae != null &&
        Math.abs(medianWinnerMae - medianLoserMae) <= Math.max(2, currentStop * 0.15)
      ) {
        interpretationTitle = "Entry quality may matter more than stop placement";
        interpretationBody =
          "Winners and losers take a similar amount of heat before resolving. Changing the stop alone is less likely to fix the strategy than improving entry timing or filtering.";
      } else {
        interpretationTitle = "Current stop looks broadly reasonable";
        interpretationBody = `Winner MAE sits inside the current stop envelope, and the distribution does not show an obvious case for aggressively widening or tightening beyond targeted backtests.`;
      }
    }

    return {
      completedTrades: completedTrades.length,
      winnerTrades: winnerTrades.length,
      loserTrades: loserTrades.length,
      uniquePairs: unique(completedTrades.map((trade) => trade.symbol_short)).filter(Boolean).length,
      currentStop,
      avgMae,
      avgWinnerMfe,
      medianWinnerMae,
      p75WinnerMae,
      p95WinnerMae,
      medianLoserMae,
      p75LoserMae,
      avgLoserMae,
      winnersAboveCurrentStop,
      stopScenarios,
      interpretationTitle,
      interpretationBody,
      interpretationMeta
    };
  }, [days, environment, filteredTrades]);

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
          <h1>MAE Analysis (Scalp Trades)</h1>
          <p>Track maximum adverse excursion to refine scalp stop placement.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/mae-analysis" />

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
          <button className="section-tab active" onClick={loadMae}>
            Refresh
          </button>
          <div className="forex-badge">{trades.length} trades</div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading MAE analysis...</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Current Avg Stop
                <strong>{interpreted.currentStop != null ? formatNumber(interpreted.currentStop) : "N/A"}</strong>
              </div>
              <div className="summary-card">
                Median Winner MAE
                <strong>
                  {interpreted.medianWinnerMae != null ? formatNumber(interpreted.medianWinnerMae) : "N/A"}
                </strong>
              </div>
              <div className="summary-card">
                75% Winner Heat
                <strong>{interpreted.p75WinnerMae != null ? formatNumber(interpreted.p75WinnerMae) : "N/A"}</strong>
              </div>
              <div className="summary-card">
                95% Winner Heat
                <strong>{interpreted.p95WinnerMae != null ? formatNumber(interpreted.p95WinnerMae) : "N/A"}</strong>
              </div>
              <div className="summary-card">
                Winners Beyond Current Stop
                <strong>
                  {interpreted.winnersAboveCurrentStop != null
                    ? `${interpreted.winnersAboveCurrentStop.toFixed(0)}%`
                    : "N/A"}
                </strong>
              </div>
            </div>

            <div className="panel">
              <div className="chart-title">Interpretation</div>
              <div className="stack-list">
                <div className="analysis-card">
                  <strong>{interpreted.interpretationTitle}</strong>
                  <p>{interpreted.interpretationBody}</p>
                  <p>{interpreted.interpretationMeta}</p>
                </div>
                <div className="analysis-card">
                  <strong>How to use this view</strong>
                  <p>
                    Treat MAE as a stop-placement tool. Compare how much heat winners usually need versus your current
                    stop. If winners regularly exceed the current stop before working, test wider stops. If winners stay
                    well inside it, tighter stops may be worth testing.
                  </p>
                </div>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">MAE Summary by Pair</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th>Trades</th>
                    <th>Win %</th>
                    <th>Avg MAE</th>
                    <th>Median MAE</th>
                    <th>75th %</th>
                    <th>90th %</th>
                    <th>Max MAE</th>
                    <th>Avg MFE</th>
                    <th>Avg Stop</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row) => (
                    <tr key={row.symbol}>
                      <td>{row.symbol_short}</td>
                      <td>{row.total_trades}</td>
                      <td>{row.win_rate.toFixed(1)}%</td>
                      <td>{formatNumber(row.avg_mae_pips)}</td>
                      <td>{formatNumber(row.median_mae_pips)}</td>
                      <td>{formatNumber(row.p75_mae_pips)}</td>
                      <td>{formatNumber(row.p90_mae_pips)}</td>
                      <td>{formatNumber(row.max_mae_pips)}</td>
                      <td>{formatNumber(row.avg_mfe_pips)}</td>
                      <td>{formatNumber(row.avg_stop_setting)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Per-Pair Recommendations</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th>Completed</th>
                    <th>Winners</th>
                    <th>Current Stop</th>
                    <th>Median Winner MAE</th>
                    <th>75% Winner Heat</th>
                    <th>Loser Median MAE</th>
                    <th>Recommendation</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {pairRecommendations.map((row) => (
                    <tr key={row.pair}>
                      <td>{row.pair}</td>
                      <td>{row.completedTrades}</td>
                      <td>{row.winners}</td>
                      <td>{row.currentStop != null ? formatNumber(row.currentStop) : "N/A"}</td>
                      <td>{row.medianWinnerMae != null ? formatNumber(row.medianWinnerMae) : "N/A"}</td>
                      <td>{row.p75WinnerMae != null ? formatNumber(row.p75WinnerMae) : "N/A"}</td>
                      <td>{row.medianLoserMae != null ? formatNumber(row.medianLoserMae) : "N/A"}</td>
                      <td>{row.recommendation}</td>
                      <td>{row.confidence}</td>
                    </tr>
                  ))}
                  {!pairRecommendations.length ? (
                    <tr>
                      <td colSpan={9}>No pair-level recommendations available for the current slice.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>

            <div className="forex-grid">
              <div className="panel table-panel">
                <div className="chart-title">Winner vs Loser Heat</div>
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Slice</th>
                      <th>Trades</th>
                      <th>Median MAE</th>
                      <th>75th %ile MAE</th>
                      <th>Avg MAE</th>
                      <th>Avg MFE</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Winners</td>
                      <td>{interpreted.winnerTrades}</td>
                      <td>{interpreted.medianWinnerMae != null ? formatNumber(interpreted.medianWinnerMae) : "N/A"}</td>
                      <td>{interpreted.p75WinnerMae != null ? formatNumber(interpreted.p75WinnerMae) : "N/A"}</td>
                      <td>{interpreted.avgMae != null ? formatNumber(interpreted.avgMae) : "N/A"}</td>
                      <td>{interpreted.avgWinnerMfe != null ? formatNumber(interpreted.avgWinnerMfe) : "N/A"}</td>
                    </tr>
                    <tr>
                      <td>Losers</td>
                      <td>{interpreted.loserTrades}</td>
                      <td>{interpreted.medianLoserMae != null ? formatNumber(interpreted.medianLoserMae) : "N/A"}</td>
                      <td>{interpreted.p75LoserMae != null ? formatNumber(interpreted.p75LoserMae) : "N/A"}</td>
                      <td>{interpreted.avgLoserMae != null ? formatNumber(interpreted.avgLoserMae) : "N/A"}</td>
                      <td>-</td>
                    </tr>
                    <tr>
                      <td>Current stop context</td>
                      <td>{interpreted.completedTrades}</td>
                      <td>{interpreted.currentStop != null ? formatNumber(interpreted.currentStop) : "N/A"}</td>
                      <td>{interpreted.avgMae != null ? formatNumber(interpreted.avgMae) : "N/A"}</td>
                      <td>{interpreted.uniquePairs}</td>
                      <td>pairs</td>
                      <td>{interpreted.completedTrades} trades</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="panel table-panel">
                <div className="chart-title">Stop Scenario Table</div>
                <div className="forex-controls" style={{ marginBottom: 12 }}>
                  <div>
                    <label>Scenario Pair</label>
                    <select value={focusPair} onChange={(event) => setFocusPair(event.target.value)}>
                      <option value="ALL">Best Available</option>
                      {pairRecommendations.map((row) => (
                        <option key={row.pair} value={row.pair}>
                          {row.pair}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="forex-badge">
                    Recommendation
                    <strong>{focusedRecommendation?.recommendation ?? "N/A"}</strong>
                  </div>
                  <div className="forex-badge">
                    Confidence
                    <strong>{focusedRecommendation?.confidence ?? "N/A"}</strong>
                  </div>
                </div>
                {focusedRecommendation ? (
                  <div className="analysis-card" style={{ marginBottom: 12 }}>
                    <strong>{focusedRecommendation.pair}</strong>
                    <p>{focusedRecommendation.rationale}</p>
                  </div>
                ) : null}
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Stop</th>
                      <th>Winners preserved</th>
                      <th>Losers contained</th>
                      <th>Winners stopped out</th>
                      <th>Losers stopped out</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(focusedRecommendation?.stopScenarios ?? interpreted.stopScenarios).map((scenario) => (
                      <tr key={scenario.stop}>
                        <td>{scenario.stop} pips</td>
                        <td>
                          {scenario.winnersStoppedPct != null
                            ? `${(100 - scenario.winnersStoppedPct).toFixed(0)}%`
                            : "N/A"}
                        </td>
                        <td>
                          {scenario.losersStoppedPct != null ? `${scenario.losersStoppedPct.toFixed(0)}%` : "N/A"}
                        </td>
                        <td>
                          {scenario.winnersStoppedPct != null
                            ? `${scenario.winnersStoppedPct.toFixed(0)}%`
                            : "N/A"}
                        </td>
                        <td>
                          {scenario.losersStoppedPct != null
                            ? `${scenario.losersStoppedPct.toFixed(0)}%`
                            : "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel">
              <div className="chart-title">Trade Filters</div>
              <div className="forex-filters">
                <div>
                  <label>Pair</label>
                  <select
                    multiple
                    value={pairFilter}
                    onChange={(event) =>
                      setPairFilter(Array.from(event.target.selectedOptions).map((opt) => opt.value))
                    }
                  >
                    {availablePairs.map((pair) => (
                      <option key={pair} value={pair}>
                        {pair}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label>Result</label>
                  <select
                    multiple
                    value={resultFilter}
                    onChange={(event) =>
                      setResultFilter(Array.from(event.target.selectedOptions).map((opt) => opt.value))
                    }
                  >
                    {availableResults.map((result) => (
                      <option key={result} value={result}>
                        {result}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Scalp Trade Details</div>
              {filteredTrades.length ? (
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Pair</th>
                      <th>Direction</th>
                      <th>Result</th>
                      <th>MAE</th>
                      <th>MFE</th>
                      <th>MAE % Stop</th>
                      <th>Stored Stop</th>
                      <th>P&amp;L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTrades.slice(0, 50).map((row) => (
                      <tr key={row.id}>
                        <td>{formatDateTime(row.timestamp)}</td>
                        <td>{row.symbol_short}</td>
                        <td>{row.direction}</td>
                        <td>{row.result}</td>
                        <td>{formatNumber(row.mae_pips ?? 0)}</td>
                        <td>{formatNumber(row.mfe_pips ?? 0)}</td>
                        <td>{formatNumber(row.mae_pct_of_stop ?? 0)}</td>
                        <td>{row.stop_distance_pips != null ? formatNumber(row.stop_distance_pips) : "-"}</td>
                        <td className={row.profit_loss != null && row.profit_loss < 0 ? "bad" : "good"}>
                          {row.profit_loss != null ? formatNumber(row.profit_loss, 2) : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="chart-placeholder">No trades match the selected filters.</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
