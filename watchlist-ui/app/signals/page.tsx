/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Virtuoso } from "react-virtuoso";

type SignalRow = {
  id: number;
  signal_timestamp: string;
  scanner_name: string;
  ticker: string;
  signal_type: string | null;
  entry_price: number | null;
  composite_score: number | null;
  quality_tier: string | null;
  status: string | null;
  trend_score: number | null;
  momentum_score: number | null;
  volume_score: number | null;
  pattern_score: number | null;
  risk_percent: number | null;
  risk_reward_ratio: number | null;
  setup_description: string | null;
  confluence_factors: string | null;
  timeframe: string | null;
  market_regime: string | null;
  claude_grade: string | null;
  claude_score: number | null;
  claude_action: string | null;
  claude_thesis: string | null;
  claude_key_strengths: string | null;
  claude_key_risks: string | null;
  claude_analyzed_at: string | null;
  news_sentiment_score: number | null;
  news_sentiment_level: string | null;
  news_headlines_count: number | null;
  company_name: string | null;
  sector: string | null;
  exchange: string | null;
  rs_percentile: number | null;
  rs_trend: string | null;
  atr_14: number | null;
  atr_percent: number | null;
  swing_high: number | null;
  swing_low: number | null;
  swing_high_date: string | null;
  swing_low_date: string | null;
  relative_volume: number | null;
  tv_osc_buy: number | null;
  tv_osc_sell: number | null;
  tv_osc_neutral: number | null;
  tv_ma_buy: number | null;
  tv_ma_sell: number | null;
  tv_ma_neutral: number | null;
  tv_overall_signal: string | null;
  tv_overall_score: number | null;
  rsi_14: number | null;
  stoch_k: number | null;
  stoch_d: number | null;
  cci_20: number | null;
  adx_14: number | null;
  plus_di: number | null;
  minus_di: number | null;
  ao_value: number | null;
  momentum_10: number | null;
  macd: number | null;
  macd_signal: number | null;
  stoch_rsi_k: number | null;
  stoch_rsi_d: number | null;
  williams_r: number | null;
  bull_power: number | null;
  bear_power: number | null;
  ultimate_osc: number | null;
  ema_10: number | null;
  ema_20: number | null;
  ema_30: number | null;
  ema_50: number | null;
  ema_100: number | null;
  ema_200: number | null;
  sma_10: number | null;
  sma_20: number | null;
  sma_30: number | null;
  sma_50: number | null;
  sma_100: number | null;
  sma_200: number | null;
  ichimoku_base: number | null;
  vwma_20: number | null;
  daq_score: number | null;
  daq_grade: string | null;
  mtf_score: number | null;
  daq_volume_score: number | null;
  daq_smc_score: number | null;
  daq_quality_score: number | null;
  daq_catalyst_score: number | null;
  daq_news_score: number | null;
  daq_regime_score: number | null;
  daq_sector_score: number | null;
  earnings_within_7d: boolean | null;
  high_short_interest: boolean | null;
  sector_underperforming: boolean | null;
  earnings_date: string | null;
  days_to_earnings: number | null;
};

type SignalStats = {
  total_signals: number;
  active_signals: number;
  high_quality: number;
  today_signals: number;
  claude_analyzed: number;
  claude_high_grade: number;
  claude_strong_buys: number;
  claude_buys: number;
  awaiting_analysis: number;
  by_scanner: Array<{ scanner_name: string; signal_count: number; avg_score: number; active_count: number }>;
};

const numberOrNull = (value: unknown) => {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
};

const formatValue = (value: unknown, digits = 2) => {
  const num = numberOrNull(value);
  return num === null ? "N/A" : num.toFixed(digits);
};

const classifyRsi = (rsi: unknown) => {
  const val = numberOrNull(rsi);
  if (val === null) return "NEUTRAL";
  if (val < 30) return "BUY";
  if (val > 70) return "SELL";
  return "NEUTRAL";
};

const classifyStoch = (val: unknown) => {
  const num = numberOrNull(val);
  if (num === null) return "NEUTRAL";
  if (num < 20) return "BUY";
  if (num > 80) return "SELL";
  return "NEUTRAL";
};

const classifyCci = (val: unknown) => {
  const num = numberOrNull(val);
  if (num === null) return "NEUTRAL";
  if (num < -100) return "BUY";
  if (num > 100) return "SELL";
  return "NEUTRAL";
};

const classifyAdx = (adx: unknown, plus: unknown, minus: unknown) => {
  const adxVal = numberOrNull(adx);
  const plusVal = numberOrNull(plus);
  const minusVal = numberOrNull(minus);
  if (adxVal === null || plusVal === null || minusVal === null) return "NEUTRAL";
  if (adxVal < 25) return "NEUTRAL";
  if (plusVal > minusVal) return "BUY";
  if (minusVal > plusVal) return "SELL";
  return "NEUTRAL";
};

const classifyAo = (val: unknown) => {
  const num = numberOrNull(val);
  if (num === null) return "NEUTRAL";
  if (num > 0) return "BUY";
  if (num < 0) return "SELL";
  return "NEUTRAL";
};

const classifyMomentum = (val: unknown) => {
  const num = numberOrNull(val);
  if (num === null) return "NEUTRAL";
  if (num > 0) return "BUY";
  if (num < 0) return "SELL";
  return "NEUTRAL";
};

const classifyMacd = (macd: unknown, signal: unknown) => {
  const macdVal = numberOrNull(macd);
  const signalVal = numberOrNull(signal);
  if (macdVal === null || signalVal === null) return "NEUTRAL";
  if (macdVal > signalVal) return "BUY";
  if (macdVal < signalVal) return "SELL";
  return "NEUTRAL";
};

const classifyWilliams = (val: unknown) => {
  const num = numberOrNull(val);
  if (num === null) return "NEUTRAL";
  if (num < -80) return "BUY";
  if (num > -20) return "SELL";
  return "NEUTRAL";
};

const classifyBbp = (bull: unknown, bear: unknown) => {
  const bullVal = numberOrNull(bull);
  const bearVal = numberOrNull(bear);
  if (bullVal === null || bearVal === null) return "NEUTRAL";
  if (bullVal > Math.abs(bearVal)) return "BUY";
  if (bearVal < -Math.abs(bullVal)) return "SELL";
  return "NEUTRAL";
};

const classifyUo = (val: unknown) => {
  const num = numberOrNull(val);
  if (num === null) return "NEUTRAL";
  if (num < 30) return "BUY";
  if (num > 70) return "SELL";
  return "NEUTRAL";
};

const classifyMa = (price: unknown, ma: unknown) => {
  const priceVal = numberOrNull(price);
  const maVal = numberOrNull(ma);
  if (priceVal === null || maVal === null) return "NEUTRAL";
  if (priceVal > maVal) return "BUY";
  if (priceVal < maVal) return "SELL";
  return "NEUTRAL";
};

const gaugeScoreFromCounts = (buy: unknown, sell: unknown, neutral: unknown) => {
  const buyVal = numberOrNull(buy) || 0;
  const sellVal = numberOrNull(sell) || 0;
  const neutralVal = numberOrNull(neutral) || 0;
  const total = buyVal + sellVal + neutralVal;
  if (!total) return 0;
  return ((buyVal - sellVal) / total) * 100;
};

const clampScore = (score: unknown) => {
  const val = numberOrNull(score) ?? 0;
  return Math.max(-100, Math.min(100, val));
};

const tvSymbol = (exchange: string | null | undefined, ticker: string) => {
  const raw = exchange ? exchange.toUpperCase().trim() : "";
  let prefix = "NASDAQ";
  if (raw.includes("NYSE")) {
    prefix = "NYSE";
  } else if (raw.includes("AMEX") || raw.includes("ARCA")) {
    prefix = "AMEX";
  } else if (raw.includes("NASDAQ")) {
    prefix = "NASDAQ";
  }
  return `${prefix}:${ticker}`;
};

const rsTrendText = (trend: string | null | undefined) => {
  if (!trend) return "";
  if (trend === "improving") return "gaining strength";
  if (trend === "deteriorating") return "weakening";
  return "holding steady";
};

const daqWeighted = (score: unknown, maxPoints: number) => {
  const val = numberOrNull(score);
  if (val === null) return 0;
  return Math.round((val / 100) * maxPoints);
};

export default function SignalsPage() {
  const apiPath = (path: string) => `../api/${path}`;
  const [stats, setStats] = useState<SignalStats | null>(null);
  const [signals, setSignals] = useState<SignalRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const [oscOpen, setOscOpen] = useState<Record<number, boolean>>({});
  const [maOpen, setMaOpen] = useState<Record<number, boolean>>({});
  const [claudeLoading, setClaudeLoading] = useState<Record<number, boolean>>({});
  const [claudeMessage, setClaudeMessage] = useState<Record<number, string>>({});

  const [scannerFilter, setScannerFilter] = useState("All Scanners");
  const [tierFilter, setTierFilter] = useState("All Tiers");
  const [statusFilter, setStatusFilter] = useState("active");
  const [claudeFilter, setClaudeFilter] = useState("All Signals");
  const [dateFrom, setDateFrom] = useState<string>(() => {
    const d = new Date();
    d.setDate(d.getDate() - 2);
    return d.toISOString().slice(0, 10);
  });
  const [dateTo, setDateTo] = useState<string>(() => new Date().toISOString().slice(0, 10));
  const [rsFilter, setRsFilter] = useState("All RS");
  const [rsTrendFilter, setRsTrendFilter] = useState("All Trends");
  const [orderBy, setOrderBy] = useState("score");

  useEffect(() => {
    const loadStats = async () => {
      const res = await fetch(`${apiPath("signals/stats")}`);
      const data = await res.json();
      setStats(data);
    };
    loadStats();
  }, []);

  const scannerOptions = useMemo(() => {
    if (!stats?.by_scanner) {
      return ["All Scanners"];
    }
    return ["All Scanners", ...stats.by_scanner.map((s) => s.scanner_name)];
  }, [stats]);

  useEffect(() => {
    const loadSignals = async () => {
      setLoading(true);
      const params = new URLSearchParams();
      if (scannerFilter !== "All Scanners") params.set("scanner", scannerFilter);
      if (statusFilter !== "All") params.set("status", statusFilter);

      const minScore = { "A+": 85, "A": 70, "B": 60, "C": 50 }[tierFilter as "A+" | "A" | "B" | "C"];
      if (minScore) params.set("minScore", String(minScore));

      const claudeAnalyzedOnly = ["Claude Analyzed Only", "A+ Grade", "A Grade", "B Grade", "STRONG BUY", "BUY"].includes(claudeFilter);
      if (claudeAnalyzedOnly) params.set("claudeOnly", "true");
      if (claudeFilter === "A+ Grade") params.set("minClaudeGrade", "A+");
      if (claudeFilter === "A Grade") params.set("minClaudeGrade", "A");
      if (claudeFilter === "B Grade") params.set("minClaudeGrade", "B");
      if (claudeFilter === "STRONG BUY") params.set("claudeAction", "STRONG BUY");
      if (claudeFilter === "BUY") params.set("claudeAction", "BUY");

      if (dateFrom) params.set("dateFrom", dateFrom);
      if (dateTo) params.set("dateTo", dateTo);

      if (rsFilter === "Elite (90+)") params.set("minRs", "90");
      if (rsFilter === "Strong (70+)") params.set("minRs", "70");
      if (rsFilter === "Average (40+)") params.set("minRs", "40");
      if (rsFilter === "Weak (<40)") params.set("maxRs", "39");

      if (rsTrendFilter !== "All Trends") params.set("rsTrend", rsTrendFilter.toLowerCase());
      params.set("limit", "100");
      params.set("orderBy", orderBy);

      const res = await fetch(`${apiPath("signals/results")}?${params.toString()}`);
      const data = await res.json();
      setSignals(data.rows || []);
      setExpanded({});
      setLoading(false);
    };
    loadSignals();
  }, [scannerFilter, tierFilter, statusFilter, claudeFilter, dateFrom, dateTo, rsFilter, rsTrendFilter, orderBy]);

  const toggleExpand = (id: number) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const riskBadges = (signal: SignalRow) => {
    const badges: string[] = [];
    if (signal.earnings_within_7d) badges.push("EARNINGS");
    if (signal.high_short_interest) badges.push("HIGH SI");
    if (signal.sector_underperforming) badges.push("SECTOR WEAK");
    return badges;
  };

  const runClaudeAnalysis = async (signal: SignalRow) => {
    setClaudeLoading((prev) => ({ ...prev, [signal.id]: true }));
    setClaudeMessage((prev) => ({ ...prev, [signal.id]: "" }));
    const res = await fetch(`${apiPath("claude/analyze")}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signal_id: signal.id, ticker: signal.ticker })
    });
    const data = await res.json();
    if (!res.ok) {
      const message = data?.detail || data?.error || "Claude analysis failed.";
      setClaudeMessage((prev) => ({ ...prev, [signal.id]: message }));
    } else {
      setClaudeMessage((prev) => ({ ...prev, [signal.id]: "Claude analysis updated." }));
      if (data?.analysis) {
        setSignals((prev) =>
          prev.map((row) => (row.id === signal.id ? { ...row, ...data.analysis } : row))
        );
      }
    }
    setClaudeLoading((prev) => ({ ...prev, [signal.id]: false }));
  };

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          Stocks Hub
        </Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Signals</h1>
          <p>Unified view of scanner signals with Claude analysis and DAQ context.</p>
        </div>
      </div>

      <div className="panel">
        <div className="metrics-grid">
          <div className="summary-card">Active Signals<strong>{stats?.active_signals ?? 0}</strong></div>
          <div className="summary-card">High Quality (A/A+)<strong>{stats?.high_quality ?? 0}</strong></div>
          <div className="summary-card">Today’s Signals<strong>{stats?.today_signals ?? 0}</strong></div>
          <div className="summary-card">Total Signals<strong>{stats?.total_signals ?? 0}</strong></div>
          <div className="summary-card">Claude Analyzed<strong>{stats?.claude_analyzed ?? 0}</strong></div>
          <div className="summary-card">Claude A/A+<strong>{stats?.claude_high_grade ?? 0}</strong></div>
          <div className="summary-card">Strong Buys<strong>{stats?.claude_strong_buys ?? 0}</strong></div>
          <div className="summary-card">Awaiting Analysis<strong>{stats?.awaiting_analysis ?? 0}</strong></div>
        </div>

        <div className="signals-filters">
          <div>
            <label>Scanner</label>
            <select value={scannerFilter} onChange={(e) => setScannerFilter(e.target.value)}>
              {scannerOptions.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Quality Tier</label>
            <select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)}>
              {["All Tiers", "A+", "A", "B", "C"].map((tier) => (
                <option key={tier} value={tier}>{tier}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Status</label>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              {["All", "active", "triggered", "closed", "expired"].map((status) => (
                <option key={status} value={status}>{status}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Claude Analysis</label>
            <select value={claudeFilter} onChange={(e) => setClaudeFilter(e.target.value)}>
              {["All Signals", "Claude Analyzed Only", "A+ Grade", "A Grade", "B Grade", "STRONG BUY", "BUY"].map((val) => (
                <option key={val} value={val}>{val}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Signal Date From</label>
            <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
          </div>
          <div>
            <label>Signal Date To</label>
            <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
          </div>
          <div>
            <label>Relative Strength</label>
            <select value={rsFilter} onChange={(e) => setRsFilter(e.target.value)}>
              {["All RS", "Elite (90+)", "Strong (70+)", "Average (40+)", "Weak (<40)"].map((val) => (
                <option key={val} value={val}>{val}</option>
              ))}
            </select>
          </div>
          <div>
            <label>RS Trend</label>
            <select value={rsTrendFilter} onChange={(e) => setRsTrendFilter(e.target.value)}>
              {["All Trends", "Improving", "Stable", "Deteriorating"].map((val) => (
                <option key={val} value={val}>{val}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Order By</label>
            <select value={orderBy} onChange={(e) => setOrderBy(e.target.value)}>
              <option value="score">Score</option>
              <option value="timestamp">Most Recent</option>
            </select>
          </div>
        </div>

        <div className="signals-header">
          <span></span>
          <span>Ticker</span>
          <span>Scanner</span>
          <span>Score</span>
          <span>Tier</span>
          <span>RS</span>
          <span>DAQ</span>
          <span>Claude</span>
          <span>Date</span>
        </div>

        {loading ? (
          <div className="footer-note">Loading signals...</div>
        ) : (
          <Virtuoso
            style={{ height: 620 }}
            data={signals}
            itemContent={(index, signal) => {
              const expandedRow = expanded[signal.id] || false;
              const badges = riskBadges(signal);
              const entry = numberOrNull(signal.entry_price) || 0;
              const entryLow = entry * 0.995;
              const entryHigh = entry * 1.01;
              const stop = entry * 0.97;
              const target1 = entry * 1.05;
              const target2 = entry * 1.1;

              return (
                <div>
                  <div className={`signals-row ${expandedRow ? "row-expanded" : ""}`}>
                    <button className="expand-btn" onClick={() => toggleExpand(signal.id)} aria-label="Toggle details">
                      {expandedRow ? "▾" : "▸"}
                    </button>
                    <span className="ticker-btn">{signal.ticker}</span>
                    <span className="scanner-cell">{signal.scanner_name?.replace(/_/g, " ") || "-"}</span>
                    <span>{formatValue(signal.composite_score, 1)}</span>
                    <span className="pill">{signal.quality_tier ?? "-"}</span>
                    <span className="pill">{signal.rs_percentile ?? "-"}</span>
                    <span className="pill">{signal.daq_score ?? "-"}</span>
                    <span className="pill">{signal.claude_action ?? "-"}</span>
                    <span className="scan-cell">{signal.signal_timestamp ? new Date(signal.signal_timestamp).toLocaleDateString("en-US", { month: "2-digit", day: "2-digit" }) : "-"}</span>
                  </div>

                  {expandedRow ? (
                    <div>
                      <div className="expander">
                        <div>
                          <h4>Trade Plan (3% SL / 5% TP)</h4>
                          <div className="detail-grid">
                            <div className="detail-item">Entry Zone: {formatValue(entryLow)} - {formatValue(entryHigh)}</div>
                            <div className="detail-item">Stop Loss: {formatValue(stop)}</div>
                            <div className="detail-item">Target 1: {formatValue(target1)}</div>
                            <div className="detail-item">Target 2: {formatValue(target2)}</div>
                            <div className="detail-item">R:R: 1.67</div>
                            <div className="detail-item">ATR%: {formatValue(signal.atr_percent, 2)}</div>
                            <div className="detail-item">Relative Vol: {formatValue(signal.relative_volume, 2)}x</div>
                            <div className="detail-item">RS: {signal.rs_percentile ?? "-"} {signal.rs_trend ? `(${rsTrendText(signal.rs_trend)})` : ""}</div>
                          </div>
                          {badges.length ? (
                            <div className="daq-badges">
                              {badges.map((badge) => (
                                <span className={`risk-badge ${badge.toLowerCase().replace(" ", "-")}`} key={badge}>{badge}</span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                        <div>
                          <h4>Technical Summary</h4>
                          <div className="gauge-row">
                            <div className="gauge-card">
                              <div className="gauge-title">Oscillators</div>
                              <div className="gauge">
                                <div className="gauge-needle" style={{ transform: `rotate(${(clampScore(gaugeScoreFromCounts(signal.tv_osc_buy, signal.tv_osc_sell, signal.tv_osc_neutral)) / 100) * 90}deg)` }} />
                                <div className="gauge-dot" />
                              </div>
                              <div className="gauge-meta">
                                Sell: {numberOrNull(signal.tv_osc_sell) ?? 0} | Neutral: {numberOrNull(signal.tv_osc_neutral) ?? 0} | Buy: {numberOrNull(signal.tv_osc_buy) ?? 0}
                              </div>
                            </div>
                            <div className="gauge-card">
                              <div className="gauge-title">Moving Averages</div>
                              <div className="gauge">
                                <div className="gauge-needle" style={{ transform: `rotate(${(clampScore(gaugeScoreFromCounts(signal.tv_ma_buy, signal.tv_ma_sell, signal.tv_ma_neutral)) / 100) * 90}deg)` }} />
                                <div className="gauge-dot" />
                              </div>
                              <div className="gauge-meta">
                                Sell: {numberOrNull(signal.tv_ma_sell) ?? 0} | Neutral: {numberOrNull(signal.tv_ma_neutral) ?? 0} | Buy: {numberOrNull(signal.tv_ma_buy) ?? 0}
                              </div>
                            </div>
                            <div className="gauge-card">
                              <div className="gauge-title">Overall</div>
                              <div className="gauge">
                                <div className="gauge-needle" style={{ transform: `rotate(${(clampScore(signal.tv_overall_score) / 100) * 90}deg)` }} />
                                <div className="gauge-dot" />
                              </div>
                              <div className="gauge-meta">
                                {signal.tv_overall_signal || "NEUTRAL"} | Score: {formatValue(signal.tv_overall_score, 1)}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="daq-panel">
                        <div className="daq-header">
                          <div className="daq-grade">{signal.daq_score ?? "-"} {signal.daq_grade ?? ""}</div>
                        </div>
                        <div className="daq-grid">
                          <div className="daq-block">
                            <h5>Technical ({daqWeighted(signal.mtf_score, 20) + daqWeighted(signal.daq_volume_score, 10) + daqWeighted(signal.daq_smc_score, 15)}/45)</h5>
                            <div className="daq-row">
                              <span>MTF</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.mtf_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.mtf_score, 20)}/20</span>
                            </div>
                            <div className="daq-row">
                              <span>Volume</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.daq_volume_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.daq_volume_score, 10)}/10</span>
                            </div>
                            <div className="daq-row">
                              <span>SMC</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.daq_smc_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.daq_smc_score, 15)}/15</span>
                            </div>
                          </div>
                          <div className="daq-block">
                            <h5>Fundamental ({daqWeighted(signal.daq_quality_score, 15) + daqWeighted(signal.daq_catalyst_score, 10)}/25)</h5>
                            <div className="daq-row">
                              <span>Quality</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.daq_quality_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.daq_quality_score, 15)}/15</span>
                            </div>
                            <div className="daq-row">
                              <span>Catalyst</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.daq_catalyst_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.daq_catalyst_score, 10)}/10</span>
                            </div>
                          </div>
                          <div className="daq-block">
                            <h5>Contextual ({daqWeighted(signal.daq_news_score, 10) + daqWeighted(signal.daq_regime_score, 10) + daqWeighted(signal.daq_sector_score, 10)}/30)</h5>
                            <div className="daq-row">
                              <span>News</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.daq_news_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.daq_news_score, 10)}/10</span>
                            </div>
                            <div className="daq-row">
                              <span>Regime</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.daq_regime_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.daq_regime_score, 10)}/10</span>
                            </div>
                            <div className="daq-row">
                              <span>Sector</span>
                              <div className="bar"><span style={{ width: `${numberOrNull(signal.daq_sector_score) ?? 0}%` }} /></div>
                              <span>{daqWeighted(signal.daq_sector_score, 10)}/10</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="detail-section">
                        <button className="detail-toggle" onClick={() => setOscOpen((prev) => ({ ...prev, [signal.id]: !prev[signal.id] }))}>
                          {oscOpen[signal.id] ? "▾" : "▸"} Oscillator Details
                        </button>
                        {oscOpen[signal.id] ? (
                          <div className="detail-table">
                            {[
                              ["RSI (14)", signal.rsi_14, classifyRsi(signal.rsi_14)],
                              ["Stochastic %K (14,3,3)", signal.stoch_k, classifyStoch(signal.stoch_k)],
                              ["CCI (20)", signal.cci_20, classifyCci(signal.cci_20)],
                              ["ADX (14)", signal.adx_14, classifyAdx(signal.adx_14, signal.plus_di, signal.minus_di)],
                              ["Awesome Oscillator", signal.ao_value, classifyAo(signal.ao_value)],
                              ["Momentum (10)", signal.momentum_10, classifyMomentum(signal.momentum_10)],
                              ["MACD Level (12,26)", signal.macd, classifyMacd(signal.macd, signal.macd_signal)],
                              ["Stochastic RSI (3,3,14,14)", signal.stoch_rsi_k, classifyStoch(signal.stoch_rsi_k)],
                              ["Williams %R (14)", signal.williams_r, classifyWilliams(signal.williams_r)],
                              ["Bull Bear Power", signal.bull_power, classifyBbp(signal.bull_power, signal.bear_power)],
                              ["Ultimate Oscillator (7,14,28)", signal.ultimate_osc, classifyUo(signal.ultimate_osc)]
                            ].map(([name, value, result]) => (
                              <div className="detail-row" key={String(name)}>
                                <div>{name}</div>
                                <div>{formatValue(value)}</div>
                                <div className={`signal ${String(result).toLowerCase()}`}>{result}</div>
                              </div>
                            ))}
                          </div>
                        ) : null}
                      </div>

                      <div className="detail-section">
                        <button className="detail-toggle" onClick={() => setMaOpen((prev) => ({ ...prev, [signal.id]: !prev[signal.id] }))}>
                          {maOpen[signal.id] ? "▾" : "▸"} Moving Averages
                        </button>
                        {maOpen[signal.id] ? (
                          <div className="detail-table">
                            {[
                              ["EMA (10)", signal.ema_10],
                              ["SMA (10)", signal.sma_10],
                              ["EMA (20)", signal.ema_20],
                              ["SMA (20)", signal.sma_20],
                              ["EMA (30)", signal.ema_30],
                              ["SMA (30)", signal.sma_30],
                              ["EMA (50)", signal.ema_50],
                              ["SMA (50)", signal.sma_50],
                              ["EMA (100)", signal.ema_100],
                              ["SMA (100)", signal.sma_100],
                              ["EMA (200)", signal.ema_200],
                              ["SMA (200)", signal.sma_200],
                              ["Ichimoku Base (26)", signal.ichimoku_base],
                              ["VWMA (20)", signal.vwma_20]
                            ].map(([name, value]) => {
                              const result = classifyMa(signal.entry_price, value);
                              return (
                                <div className="detail-row" key={String(name)}>
                                  <div>{name}</div>
                                  <div>{formatValue(value)}</div>
                                  <div className={`signal ${result.toLowerCase()}`}>{result}</div>
                                </div>
                              );
                            })}
                          </div>
                        ) : null}
                      </div>

                      <div className="detail-section">
                        <div className="chart-grid">
                          <div className="chart-card">
                            <div className="chart-title">Daily Chart</div>
                            <iframe
                              src={`https://www.tradingview.com/widgetembed/?symbol=${encodeURIComponent(
                                tvSymbol(signal.exchange, signal.ticker)
                              )}&interval=D&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1`}
                              className="tv-frame"
                              loading="lazy"
                              title={`${signal.ticker} daily chart`}
                              allowFullScreen
                              referrerPolicy="no-referrer-when-downgrade"
                            />
                          </div>
                          <div className="chart-card">
                            <div className="chart-title">Weekly Chart</div>
                            <iframe
                              src={`https://www.tradingview.com/widgetembed/?symbol=${encodeURIComponent(
                                tvSymbol(signal.exchange, signal.ticker)
                              )}&interval=W&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1`}
                              className="tv-frame"
                              loading="lazy"
                              title={`${signal.ticker} weekly chart`}
                              allowFullScreen
                              referrerPolicy="no-referrer-when-downgrade"
                            />
                          </div>
                        </div>
                      </div>

                      <div className="detail-section">
                        <div className="signal-text">
                          <div className="claude-header">
                            <h4>Claude Analysis</h4>
                            <button
                              className="claude-btn"
                              onClick={() => runClaudeAnalysis(signal)}
                              disabled={claudeLoading[signal.id]}
                            >
                              {claudeLoading[signal.id] ? "Analyzing..." : "Analyze with Claude"}
                            </button>
                          </div>
                          <p><strong>{signal.claude_grade ?? "-"} {signal.claude_action ?? ""}</strong></p>
                          <p>{signal.claude_thesis || "No Claude thesis yet."}</p>
                          {signal.claude_key_strengths ? <p><strong>Strengths:</strong> {signal.claude_key_strengths}</p> : null}
                          {signal.claude_key_risks ? <p><strong>Risks:</strong> {signal.claude_key_risks}</p> : null}
                          {claudeMessage[signal.id] ? <div className="footer-note">{claudeMessage[signal.id]}</div> : null}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            }}
          />
        )}
        <div className="footer-note">Signals: {signals.length}</div>
      </div>
    </div>
  );
}
