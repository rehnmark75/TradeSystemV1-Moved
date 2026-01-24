/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable @next/next/no-img-element */
"use client";

import { useEffect, useMemo, useState } from "react";
import { Virtuoso } from "react-virtuoso";
import { WATCHLIST_DEFINITIONS } from "../lib/watchlists";

type WatchlistRow = {
  ticker: string;
  price: number | null;
  volume: number | null;
  avg_volume: number | null;
  rsi_14: number | null;
  macd: number | null;
  gap_pct: number | null;
  price_change_1d: number | null;
  scan_date: string | null;
  crossover_date: string | null;
  days_on_list: number | null;
  avg_daily_change_5d: number | null;
  daq_score: number | null;
  daq_grade: string | null;
  daq_earnings_risk: boolean | null;
  daq_high_short_interest: boolean | null;
  daq_sector_underperforming: boolean | null;
  rs_percentile: number | null;
  rs_trend: string | null;
  tv_overall_score: number | null;
  tv_overall_signal: string | null;
  exchange: string | null;
};

type WatchlistDetail = WatchlistRow & {
  name: string | null;
  macd_signal: number | null;
  daq_mtf_score: number | null;
  daq_volume_score: number | null;
  daq_smc_score: number | null;
  daq_quality_score: number | null;
  daq_catalyst_score: number | null;
  daq_news_score: number | null;
  daq_regime_score: number | null;
  daq_sector_score: number | null;
  atr_14: number | null;
  atr_percent: number | null;
  swing_high: number | null;
  swing_low: number | null;
  swing_high_date: string | null;
  swing_low_date: string | null;
  suggested_entry_low: number | null;
  suggested_entry_high: number | null;
  suggested_stop_loss: number | null;
  suggested_target_1: number | null;
  suggested_target_2: number | null;
  risk_reward_ratio: number | null;
  risk_percent: number | null;
  volume_trend: string | null;
  relative_volume: number | null;
  earnings_date: string | null;
  days_to_earnings: number | null;
  tv_osc_buy: number | null;
  tv_osc_sell: number | null;
  tv_osc_neutral: number | null;
  tv_ma_buy: number | null;
  tv_ma_sell: number | null;
  tv_ma_neutral: number | null;
  tv_overall_signal: string | null;
  tv_overall_score: number | null;
  stoch_k: number | null;
  stoch_d: number | null;
  cci_20: number | null;
  adx_14: number | null;
  plus_di: number | null;
  minus_di: number | null;
  ao_value: number | null;
  momentum_10: number | null;
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
};

const CROSSOVER = new Set(["ema_50_crossover", "ema_20_crossover", "macd_bullish_cross"]);

export default function Page() {
  const watchlistKeys = Object.keys(WATCHLIST_DEFINITIONS);
  const [watchlist, setWatchlist] = useState(watchlistKeys[0]);
  const [scanDate, setScanDate] = useState<string | null>(null);
  const [limit, setLimit] = useState(200);
  const [filter, setFilter] = useState("");
  const [rows, setRows] = useState<WatchlistRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<{ counts: Record<string, number>; last_scan: string | null; total_stocks_scanned: number; event_date: string | null }>({
    counts: {},
    last_scan: null,
    total_stocks_scanned: 0,
    event_date: null
  });
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [details, setDetails] = useState<Record<string, WatchlistDetail | null>>({});
  const [oscOpen, setOscOpen] = useState<Record<string, boolean>>({});
  const [maOpen, setMaOpen] = useState<Record<string, boolean>>({});

  const apiPath = (path: string) => `api/${path}`;

  useEffect(() => {
    const loadStats = async () => {
      const params = scanDate ? `?date=${scanDate}` : "";
      const res = await fetch(`${apiPath("watchlist/stats")}${params}`);
      const data = await res.json();
      setStats(data);
      if (!scanDate && data.event_date && !CROSSOVER.has(watchlist)) {
        setScanDate(data.event_date);
      }
    };
    loadStats();
  }, []);

  useEffect(() => {
    const loadRows = async () => {
      setLoading(true);
      const params = new URLSearchParams({
        watchlist,
        limit: String(limit)
      });
      if (!CROSSOVER.has(watchlist) && scanDate) {
        params.set("date", scanDate);
      }
      const res = await fetch(`${apiPath("watchlist/results")}?${params.toString()}`);
      const data = await res.json();
      setRows(data.rows || []);
      setExpanded({});
      setDetails({});
      setLoading(false);
    };
    loadRows();
  }, [watchlist, scanDate, limit]);

  const watchlistInfo = WATCHLIST_DEFINITIONS[watchlist];

  const filteredRows = useMemo(() => {
    if (!filter) {
      return rows;
    }
    const term = filter.toUpperCase();
    return rows.filter((row) => row.ticker?.toUpperCase().includes(term));
  }, [rows, filter]);

  const toggleExpand = async (ticker: string) => {
    const next = !expanded[ticker];
    setExpanded((prev) => ({ ...prev, [ticker]: next }));
    if (next && !details[ticker]) {
      const params = new URLSearchParams({ watchlist, ticker });
      if (!CROSSOVER.has(watchlist) && scanDate) {
        params.set("date", scanDate);
      }
      const res = await fetch(`${apiPath("watchlist/detail")}?${params.toString()}`);
      const data = await res.json();
      setDetails((prev) => ({ ...prev, [ticker]: data.row || null }));
    }
  };

  const toggleOsc = (ticker: string) => {
    setOscOpen((prev) => ({ ...prev, [ticker]: !prev[ticker] }));
  };

  const toggleMa = (ticker: string) => {
    setMaOpen((prev) => ({ ...prev, [ticker]: !prev[ticker] }));
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

  const riskBadges = (detail: WatchlistDetail | null | undefined) => {
    if (!detail) return null;
    const badges: string[] = [];
    if (detail.daq_earnings_risk) badges.push("EARNINGS");
    if (detail.daq_high_short_interest) badges.push("HIGH SI");
    if (detail.daq_sector_underperforming) badges.push("SECTOR WEAK");
    return badges.length ? badges : null;
  };

  return (
    <div className="page">
      <div className="header">
        <div>
          <h1>Watchlist Fast</h1>
          <p>Virtualized list with on-demand detail expansion.</p>
        </div>
        <div className="header-chip">
          {watchlistInfo.icon} {watchlistInfo.name}
        </div>
      </div>

      <div className="panel">
        <div className="controls">
          <div>
            <label>Watchlist</label>
            <select value={watchlist} onChange={(e) => setWatchlist(e.target.value)}>
              {watchlistKeys.map((key) => (
                <option key={key} value={key}>
                  {WATCHLIST_DEFINITIONS[key].icon} {WATCHLIST_DEFINITIONS[key].name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Event Date</label>
            <input
              type="date"
              value={scanDate || ""}
              disabled={CROSSOVER.has(watchlist)}
              onChange={(e) => setScanDate(e.target.value || null)}
            />
          </div>
          <div>
            <label>Limit</label>
            <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
              {[100, 200, 500, 1000].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Filter Ticker</label>
            <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="AAPL" />
          </div>
        </div>

        <div className="summary">
          <div className="summary-card">
            {watchlistInfo.icon} {watchlistInfo.name}
            <strong>{stats.counts[watchlist] || 0} active</strong>
          </div>
          <div className="summary-card">
            Last scan
            <strong>{stats.last_scan || "N/A"}</strong>
          </div>
          <div className="summary-card">
            Total stocks
            <strong>{stats.total_stocks_scanned.toLocaleString()}</strong>
          </div>
          <div className="summary-card">
            Mode
            <strong>{watchlistInfo.type}</strong>
          </div>
        </div>

        <div className="list-header">
          <span></span>
          <span>Ticker</span>
          <span>Signal</span>
          <span>RS</span>
          <span>DAQ</span>
          <span>TV</span>
          <span>Days</span>
          <span>Price</span>
          <span>Volume</span>
          <span>RSI</span>
          <span>1D</span>
        </div>

        {loading ? (
          <div className="footer-note">Loading watchlist...</div>
        ) : (
          <Virtuoso
            style={{ height: 540 }}
            data={filteredRows}
            itemContent={(index, row) => {
              const rsVal = row.rs_percentile ?? null;
              const daqVal = row.daq_score ?? null;
              const days = row.days_on_list ?? 0;
              const expandedRow = expanded[row.ticker] || false;

              return (
                <div>
                  <div className={`row ${expandedRow ? "row-expanded" : ""}`}>
                    <button className="expand-btn" onClick={() => toggleExpand(row.ticker)} aria-label="Toggle details">
                      {expandedRow ? "▾" : "▸"}
                    </button>
                    <button className="ticker-btn" onClick={() => toggleExpand(row.ticker)}>
                      {row.ticker}
                    </button>
                    <span className="scan-cell">
                      {CROSSOVER.has(watchlist)
                        ? row.crossover_date
                          ? new Date(row.crossover_date).toLocaleDateString("en-US", { month: "2-digit", day: "2-digit" })
                          : "-"
                        : row.scan_date
                          ? new Date(row.scan_date).toLocaleDateString("en-US", { month: "2-digit", day: "2-digit" })
                          : "-"}
                    </span>
                    <span className={rsVal !== null ? "pill " + (rsVal >= 70 ? "good" : rsVal >= 40 ? "warn" : "bad") : ""}>
                      {rsVal !== null ? `${rsVal}` : "-"}
                    </span>
                    <span className={daqVal !== null ? "pill " + (daqVal >= 70 ? "good" : daqVal >= 50 ? "warn" : "bad") : ""}>
                      {daqVal !== null ? `${daqVal}` : "-"}
                    </span>
                    <span
                      className={
                        row.tv_overall_score !== null && row.tv_overall_score !== undefined
                          ? Number(row.tv_overall_score) >= 20
                            ? "pill good"
                            : Number(row.tv_overall_score) >= -20
                              ? "pill warn"
                              : "pill bad"
                          : ""
                      }
                    >
                      {row.tv_overall_score !== null && row.tv_overall_score !== undefined
                        ? Number(row.tv_overall_score).toFixed(1)
                        : "-"}
                    </span>
                    <span>{CROSSOVER.has(watchlist) ? `${days}d` : "-"}</span>
                    <span>
                      {row.price !== null && row.price !== undefined
                        ? `$${Number(row.price).toFixed(2)}`
                        : "-"}
                    </span>
                    <span>
                      {row.volume !== null && row.volume !== undefined
                        ? `${(Number(row.volume) / 1e6).toFixed(1)}M`
                        : "-"}
                    </span>
                    <span>
                      {row.rsi_14 !== null && row.rsi_14 !== undefined
                        ? Math.round(Number(row.rsi_14))
                        : "-"}
                    </span>
                    <span
                      className={
                        row.price_change_1d !== null && row.price_change_1d !== undefined
                          ? Number(row.price_change_1d) >= 0
                            ? "pill good"
                            : "pill bad"
                          : ""
                      }
                    >
                      {row.price_change_1d !== null && row.price_change_1d !== undefined
                        ? `${Number(row.price_change_1d).toFixed(1)}%`
                        : "-"}
                    </span>
                  </div>

                  {expandedRow ? (
                    <div>
                      <div className="expander">
                        <div>
                          <h4>Trade Plan</h4>
                          {details[row.ticker] ? (
                            <div className="detail-grid">
                              <div className="detail-item">Entry Low: {formatValue(details[row.ticker]?.suggested_entry_low)}</div>
                              <div className="detail-item">Entry High: {formatValue(details[row.ticker]?.suggested_entry_high)}</div>
                              <div className="detail-item">Stop Loss: {formatValue(details[row.ticker]?.suggested_stop_loss)}</div>
                              <div className="detail-item">Target 1: {formatValue(details[row.ticker]?.suggested_target_1)}</div>
                              <div className="detail-item">Target 2: {formatValue(details[row.ticker]?.suggested_target_2)}</div>
                              <div className="detail-item">R:R: {formatValue(details[row.ticker]?.risk_reward_ratio, 2)}</div>
                              <div className="detail-item">ATR%: {formatValue(details[row.ticker]?.atr_percent, 2)}</div>
                              <div className="detail-item">Volume Trend: {details[row.ticker]?.volume_trend ?? "-"}</div>
                              <div className="detail-item">
                                Relative Strength: {details[row.ticker]?.rs_percentile ?? "-"}{" "}
                                {details[row.ticker]?.rs_trend ? `(${rsTrendText(details[row.ticker]?.rs_trend)})` : ""}
                              </div>
                              <div className="detail-item">
                                Risk: {formatValue(details[row.ticker]?.risk_percent, 1)}%
                              </div>
                            </div>
                          ) : (
                            <div className="footer-note">Loading details...</div>
                          )}
                        </div>
                        <div>
                          <h4>Technical Summary</h4>
                          {details[row.ticker] ? (
                            <div className="gauge-row">
                              <div className="gauge-card">
                                <div className="gauge-title">Oscillators</div>
                                <div className="gauge">
                                  <div
                                    className="gauge-needle"
                                    style={{
                                      transform: `rotate(${(clampScore(gaugeScoreFromCounts(details[row.ticker]?.tv_osc_buy, details[row.ticker]?.tv_osc_sell, details[row.ticker]?.tv_osc_neutral)) / 100) * 90}deg)`
                                    }}
                                  />
                                  <div className="gauge-dot" />
                                </div>
                                <div className="gauge-meta">
                                  Sell: {numberOrNull(details[row.ticker]?.tv_osc_sell) ?? 0} | Neutral: {numberOrNull(details[row.ticker]?.tv_osc_neutral) ?? 0} | Buy: {numberOrNull(details[row.ticker]?.tv_osc_buy) ?? 0}
                                </div>
                              </div>
                              <div className="gauge-card">
                                <div className="gauge-title">Moving Averages</div>
                                <div className="gauge">
                                  <div
                                    className="gauge-needle"
                                    style={{
                                      transform: `rotate(${(clampScore(gaugeScoreFromCounts(details[row.ticker]?.tv_ma_buy, details[row.ticker]?.tv_ma_sell, details[row.ticker]?.tv_ma_neutral)) / 100) * 90}deg)`
                                    }}
                                  />
                                  <div className="gauge-dot" />
                                </div>
                                <div className="gauge-meta">
                                  Sell: {numberOrNull(details[row.ticker]?.tv_ma_sell) ?? 0} | Neutral: {numberOrNull(details[row.ticker]?.tv_ma_neutral) ?? 0} | Buy: {numberOrNull(details[row.ticker]?.tv_ma_buy) ?? 0}
                                </div>
                              </div>
                              <div className="gauge-card">
                                <div className="gauge-title">Overall</div>
                                <div className="gauge">
                                  <div
                                    className="gauge-needle"
                                    style={{
                                      transform: `rotate(${(clampScore(details[row.ticker]?.tv_overall_score) / 100) * 90}deg)`
                                    }}
                                  />
                                  <div className="gauge-dot" />
                                </div>
                                <div className="gauge-meta">
                                  {details[row.ticker]?.tv_overall_signal || "NEUTRAL"} | Score: {formatValue(details[row.ticker]?.tv_overall_score, 1)}
                                </div>
                              </div>
                            </div>
                          ) : (
                            <div className="footer-note">Loading summary...</div>
                          )}
                        </div>
                      </div>

                      {details[row.ticker] ? (
                        <div className="daq-panel">
                          <div className="daq-header">
                            <div className="daq-grade">
                              {details[row.ticker]?.daq_score ?? "-"} {details[row.ticker]?.daq_grade ?? ""}
                            </div>
                            <div className="daq-badges">
                              {(riskBadges(details[row.ticker]) || []).map((badge) => (
                                <span className={`risk-badge ${badge.toLowerCase().replace(" ", "-")}`} key={badge}>
                                  {badge}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div className="daq-grid">
                            <div className="daq-block">
                              <h5>Technical ({daqWeighted(details[row.ticker]?.daq_mtf_score, 20) + daqWeighted(details[row.ticker]?.daq_volume_score, 10) + daqWeighted(details[row.ticker]?.daq_smc_score, 15)}/45)</h5>
                              <div className="daq-row">
                                <span>MTF</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_mtf_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_mtf_score, 20)}/20</span>
                              </div>
                              <div className="daq-row">
                                <span>Volume</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_volume_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_volume_score, 10)}/10</span>
                              </div>
                              <div className="daq-row">
                                <span>SMC</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_smc_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_smc_score, 15)}/15</span>
                              </div>
                            </div>
                            <div className="daq-block">
                              <h5>Fundamental ({daqWeighted(details[row.ticker]?.daq_quality_score, 15) + daqWeighted(details[row.ticker]?.daq_catalyst_score, 10)}/25)</h5>
                              <div className="daq-row">
                                <span>Quality</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_quality_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_quality_score, 15)}/15</span>
                              </div>
                              <div className="daq-row">
                                <span>Catalyst</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_catalyst_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_catalyst_score, 10)}/10</span>
                              </div>
                            </div>
                            <div className="daq-block">
                              <h5>Contextual ({daqWeighted(details[row.ticker]?.daq_news_score, 10) + daqWeighted(details[row.ticker]?.daq_regime_score, 10) + daqWeighted(details[row.ticker]?.daq_sector_score, 10)}/30)</h5>
                              <div className="daq-row">
                                <span>News</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_news_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_news_score, 10)}/10</span>
                              </div>
                              <div className="daq-row">
                                <span>Regime</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_regime_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_regime_score, 10)}/10</span>
                              </div>
                              <div className="daq-row">
                                <span>Sector</span>
                                <div className="bar"><span style={{ width: `${numberOrNull(details[row.ticker]?.daq_sector_score) ?? 0}%` }} /></div>
                                <span>{daqWeighted(details[row.ticker]?.daq_sector_score, 10)}/10</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : null}

                      <div className="detail-section">
                        <button className="detail-toggle" onClick={() => toggleOsc(row.ticker)}>
                          {oscOpen[row.ticker] ? "▾" : "▸"} Oscillator Details
                        </button>
                        {oscOpen[row.ticker] && details[row.ticker] ? (
                          <div className="detail-table">
                            {[
                              ["RSI (14)", details[row.ticker]?.rsi_14, classifyRsi(details[row.ticker]?.rsi_14)],
                              ["Stochastic %K (14,3,3)", details[row.ticker]?.stoch_k, classifyStoch(details[row.ticker]?.stoch_k)],
                              ["CCI (20)", details[row.ticker]?.cci_20, classifyCci(details[row.ticker]?.cci_20)],
                              ["ADX (14)", details[row.ticker]?.adx_14, classifyAdx(details[row.ticker]?.adx_14, details[row.ticker]?.plus_di, details[row.ticker]?.minus_di)],
                              ["Awesome Oscillator", details[row.ticker]?.ao_value, classifyAo(details[row.ticker]?.ao_value)],
                              ["Momentum (10)", details[row.ticker]?.momentum_10, classifyMomentum(details[row.ticker]?.momentum_10)],
                              ["MACD Level (12,26)", details[row.ticker]?.macd, classifyMacd(details[row.ticker]?.macd, details[row.ticker]?.macd_signal)],
                              ["Stochastic RSI (3,3,14,14)", details[row.ticker]?.stoch_rsi_k, classifyStoch(details[row.ticker]?.stoch_rsi_k)],
                              ["Williams %R (14)", details[row.ticker]?.williams_r, classifyWilliams(details[row.ticker]?.williams_r)],
                              ["Bull Bear Power", details[row.ticker]?.bull_power, classifyBbp(details[row.ticker]?.bull_power, details[row.ticker]?.bear_power)],
                              ["Ultimate Oscillator (7,14,28)", details[row.ticker]?.ultimate_osc, classifyUo(details[row.ticker]?.ultimate_osc)]
                            ].map(([name, value, signal]) => (
                              <div className="detail-row" key={String(name)}>
                                <div>{name}</div>
                                <div>{formatValue(value)}</div>
                                <div className={`signal ${String(signal).toLowerCase()}`}>{signal}</div>
                              </div>
                            ))}
                          </div>
                        ) : null}
                      </div>

                      <div className="detail-section">
                        <button className="detail-toggle" onClick={() => toggleMa(row.ticker)}>
                          {maOpen[row.ticker] ? "▾" : "▸"} Moving Averages
                        </button>
                        {maOpen[row.ticker] && details[row.ticker] ? (
                          <div className="detail-table">
                            {[
                              ["EMA (10)", details[row.ticker]?.ema_10],
                              ["SMA (10)", details[row.ticker]?.sma_10],
                              ["EMA (20)", details[row.ticker]?.ema_20],
                              ["SMA (20)", details[row.ticker]?.sma_20],
                              ["EMA (30)", details[row.ticker]?.ema_30],
                              ["SMA (30)", details[row.ticker]?.sma_30],
                              ["EMA (50)", details[row.ticker]?.ema_50],
                              ["SMA (50)", details[row.ticker]?.sma_50],
                              ["EMA (100)", details[row.ticker]?.ema_100],
                              ["SMA (100)", details[row.ticker]?.sma_100],
                              ["EMA (200)", details[row.ticker]?.ema_200],
                              ["SMA (200)", details[row.ticker]?.sma_200],
                              ["Ichimoku Base (26)", details[row.ticker]?.ichimoku_base],
                              ["VWMA (20)", details[row.ticker]?.vwma_20]
                            ].map(([name, value]) => {
                              const signal = classifyMa(details[row.ticker]?.price, value);
                              return (
                                <div className="detail-row" key={String(name)}>
                                  <div>{name}</div>
                                  <div>{formatValue(value)}</div>
                                  <div className={`signal ${signal.toLowerCase()}`}>{signal}</div>
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
                            {details[row.ticker] ? (
                              <iframe
                                src={`https://www.tradingview.com/widgetembed/?symbol=${encodeURIComponent(
                                  tvSymbol(details[row.ticker]?.exchange, row.ticker)
                                )}&interval=D&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1`}
                                className="tv-frame"
                                loading="lazy"
                                title={`${row.ticker} daily chart`}
                                allowFullScreen
                                referrerPolicy="no-referrer-when-downgrade"
                              />
                            ) : (
                              <div className="footer-note">Loading chart...</div>
                            )}
                          </div>
                          <div className="chart-card">
                            <div className="chart-title">Weekly Chart</div>
                            {details[row.ticker] ? (
                              <iframe
                                src={`https://www.tradingview.com/widgetembed/?symbol=${encodeURIComponent(
                                  tvSymbol(details[row.ticker]?.exchange, row.ticker)
                                )}&interval=W&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1`}
                                className="tv-frame"
                                loading="lazy"
                                title={`${row.ticker} weekly chart`}
                                allowFullScreen
                                referrerPolicy="no-referrer-when-downgrade"
                              />
                            ) : (
                              <div className="footer-note">Loading chart...</div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            }}
          />
        )}
        <div className="footer-note">Rows: {filteredRows.length} | Data source: stocks database</div>
      </div>
    </div>
  );
}
