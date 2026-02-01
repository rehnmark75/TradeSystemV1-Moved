/* eslint-disable @next/next/no-img-element */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

type StockInstrument = {
  ticker: string;
  name: string | null;
  exchange: string | null;
  sector: string | null;
  industry: string | null;
  market_cap: string | null;
  avg_volume: number | null;
  currency: string | null;
  earnings_date: string | null;
  dividend_yield: number | null;
  trailing_pe: number | null;
  forward_pe: number | null;
  profit_margin: number | null;
  revenue_growth: number | null;
  earnings_growth: number | null;
  debt_to_equity: number | null;
  current_ratio: number | null;
  quick_ratio: number | null;
  analyst_rating: string | null;
  target_price: number | null;
  target_high: number | null;
  target_low: number | null;
  number_of_analysts: number | null;
  fifty_two_week_high: number | null;
  fifty_two_week_low: number | null;
  fifty_two_week_change: number | null;
  fifty_day_average: number | null;
  two_hundred_day_average: number | null;
};

type StockMetrics = {
  current_price: number | null;
  rs_percentile: number | null;
  relative_volume: number | null;
  rsi_14: number | null;
  macd: number | null;
  atr_percent: number | null;
  trend_strength: string | null;
  price_change_20d: number | null;
} | null;

type WatchlistRow = {
  price: number | null;
  rs_percentile: number | null;
  daq_score: number | null;
  suggested_entry_low: number | null;
  suggested_entry_high: number | null;
  suggested_stop_loss: number | null;
  suggested_target_1: number | null;
  suggested_target_2: number | null;
  risk_reward_ratio: number | null;
  atr_percent: number | null;
  relative_volume: number | null;
} | null;

type AnalystReco = {
  period: string | null;
  strong_buy: number | null;
  buy: number | null;
  hold: number | null;
  sell: number | null;
  strong_sell: number | null;
};

type SignalRow = {
  setup_description: string | null;
  claude_thesis: string | null;
  trend_score: number | null;
  momentum_score: number | null;
  volume_score: number | null;
  daq_score: number | null;
  entry_price: number | null;
} | null;

type SignalHistory = {
  id: number;
  signal_timestamp: string;
  scanner_name: string;
  signal_type: string | null;
  composite_score: number | null;
  quality_tier: string | null;
  status: string | null;
  claude_action: string | null;
  claude_grade: string | null;
  news_sentiment_level: string | null;
  entry_price: number | null;
  risk_reward_ratio: number | null;
};

type NewsHeadline = {
  headline: string;
  summary: string | null;
  source: string | null;
  url: string | null;
  published_at: string | null;
  sentiment_score: number | null;
};

type SectorContext = {
  sector: string;
  sector_return_1d: number | null;
  sector_return_5d: number | null;
  sector_return_20d: number | null;
  rs_vs_spy: number | null;
  rs_percentile: number | null;
  rs_trend: string | null;
  stocks_in_sector: number | null;
  pct_above_sma50: number | null;
  pct_bullish_trend: number | null;
  sector_stage: string | null;
};

type MarketRegime = {
  market_regime: string | null;
  spy_price: number | null;
  spy_vs_sma50_pct: number | null;
  spy_vs_sma200_pct: number | null;
  market_health: number | null;
  intermediate_trend: number | null;
  volatility_regime: string | null;
  recommended_strategies: string[] | string | null;
  strategy_guidance: string | null;
};

type NoteEntry = {
  id: number;
  ticker: string;
  note_text: string;
  context: string | null;
  created_at: string;
  updated_at: string;
};

type StockDetail = {
  instrument: StockInstrument | null;
  metrics: StockMetrics;
  watchlist: WatchlistRow;
  signal: SignalRow;
  signal_history: SignalHistory[];
  analyst: AnalystReco | null;
  news: NewsHeadline[];
  sector_context: SectorContext | null;
  market_regime: MarketRegime | null;
};

const numberOrNull = (value: unknown) => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
};

const formatValue = (value: unknown, digits = 2) => {
  const num = numberOrNull(value);
  return num === null ? "-" : num.toFixed(digits);
};

const formatDate = (value: string | null | undefined) => {
  if (!value) return "-";
  return new Date(value).toLocaleDateString("en-US", { month: "short", day: "2-digit" });
};

const trendScoreFromStrength = (strength: string | null | undefined) => {
  if (!strength) return null;
  switch (strength) {
    case "strong_up":
      return 80;
    case "up":
      return 65;
    case "neutral":
      return 50;
    case "down":
      return 35;
    case "strong_down":
      return 20;
    default:
      return null;
  }
};

const apiPath = (path: string) => `/trading/api/${path}`;

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

const ForecastChart = ({
  current,
  avg,
  max,
  min,
  analysts
}: {
  current: number;
  avg: number | null;
  max: number | null;
  min: number | null;
  analysts: number | null;
}) => {
  const width = 520;
  const height = 180;
  const padding = 24;
  const xStart = padding;
  const xEnd = width - padding;
  const yMin = Math.min(current, min ?? current, avg ?? current, max ?? current);
  const yMax = Math.max(current, min ?? current, avg ?? current, max ?? current);
  const scale = (value: number) => {
    if (yMax === yMin) return height / 2;
    return height - padding - ((value - yMin) / (yMax - yMin)) * (height - padding * 2);
  };
  const currentY = scale(current);
  const avgY = avg !== null ? scale(avg) : null;
  const maxY = max !== null ? scale(max) : null;
  const minY = min !== null ? scale(min) : null;

  const formatPct = (target: number | null) => {
    if (target === null || current === 0) return "-";
    const pct = ((target - current) / current) * 100;
    return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
  };

  return (
    <div className="forecast-card">
      <div className="forecast-header">
        <h4>Analyst Forecast</h4>
        <span className="forecast-meta">
          {analysts ? `${analysts} analysts` : "Analyst targets"}
        </span>
      </div>
      <div className="forecast-stats">
        <div>
          <span>Current</span>
          <strong>${current.toFixed(2)}</strong>
        </div>
        <div>
          <span>Avg Target</span>
          <strong>{avg !== null ? `$${avg.toFixed(2)} (${formatPct(avg)})` : "-"}</strong>
        </div>
        <div>
          <span>Max Target</span>
          <strong>{max !== null ? `$${max.toFixed(2)} (${formatPct(max)})` : "-"}</strong>
        </div>
        <div>
          <span>Min Target</span>
          <strong>{min !== null ? `$${min.toFixed(2)} (${formatPct(min)})` : "-"}</strong>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="forecast-chart" role="img">
        <defs>
          <linearGradient id="forecast-band" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#e8f4f4" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#d7f0e7" stopOpacity="0.9" />
          </linearGradient>
        </defs>
        {maxY !== null && minY !== null ? (
          <polygon
            points={`${xStart},${currentY} ${xEnd},${maxY} ${xEnd},${minY}`}
            fill="url(#forecast-band)"
          />
        ) : null}
        <line x1={xStart} y1={currentY} x2={xEnd} y2={currentY} stroke="#6e6254" strokeDasharray="4 4" />
        {avgY !== null ? <line x1={xStart} y1={currentY} x2={xEnd} y2={avgY} stroke="#0f4c5c" strokeWidth="2" /> : null}
        {maxY !== null ? <line x1={xStart} y1={currentY} x2={xEnd} y2={maxY} stroke="#2f9e44" strokeWidth="2" /> : null}
        {minY !== null ? <line x1={xStart} y1={currentY} x2={xEnd} y2={minY} stroke="#c92a2a" strokeWidth="2" /> : null}
        <circle cx={xStart} cy={currentY} r="4" fill="#0f4c5c" />
        <text x={xEnd + 4} y={currentY + 4} className="forecast-label">Current</text>
        {avgY !== null ? <text x={xEnd + 4} y={avgY + 4} className="forecast-label">Avg</text> : null}
        {maxY !== null ? <text x={xEnd + 4} y={maxY + 4} className="forecast-label">Max</text> : null}
        {minY !== null ? <text x={xEnd + 4} y={minY + 4} className="forecast-label">Min</text> : null}
      </svg>
    </div>
  );
};

export default function StockDeepDivePage() {
  const params = useParams();
  const router = useRouter();
  const initialTicker = typeof params.ticker === "string" ? params.ticker.toUpperCase() : "";
  const [tickerInput, setTickerInput] = useState(initialTicker);
  const [data, setData] = useState<StockDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState<NoteEntry[]>([]);
  const [noteDraft, setNoteDraft] = useState("");
  const [noteEditing, setNoteEditing] = useState<number | null>(null);
  const [noteEditDraft, setNoteEditDraft] = useState("");
  const [noteMessage, setNoteMessage] = useState("");
  const [noteLoading, setNoteLoading] = useState(false);

  const ticker = initialTicker;

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    fetch(`${apiPath("stocks/detail")}?ticker=${encodeURIComponent(ticker)}`)
      .then((res) => res.json())
      .then((payload) => setData(payload))
      .finally(() => setLoading(false));
  }, [ticker]);

  useEffect(() => {
    if (!ticker) return;
    setNoteLoading(true);
    fetch(`${apiPath("notes")}?ticker=${encodeURIComponent(ticker)}&context=deep_dive`)
      .then((res) => res.json())
      .then((payload) => setNotes(payload.rows || []))
      .finally(() => setNoteLoading(false));
  }, [ticker]);

  const saveNote = async () => {
    const text = noteDraft.trim();
    if (!text) return;
    setNoteLoading(true);
    setNoteMessage("");
    const res = await fetch(`${apiPath("notes")}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, note: text, context: "deep_dive" })
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      setNoteMessage(payload?.error || "Failed to save note.");
    } else {
      const payload = await res.json();
      setNotes((prev) => [payload.row, ...prev]);
      setNoteDraft("");
    }
    setNoteLoading(false);
  };

  const saveNoteEdit = async () => {
    if (!noteEditing) return;
    const text = noteEditDraft.trim();
    if (!text) return;
    setNoteLoading(true);
    setNoteMessage("");
    const res = await fetch(`${apiPath("notes")}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: noteEditing, note: text })
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      setNoteMessage(payload?.error || "Failed to update note.");
    } else {
      setNotes((prev) => prev.map((note) => (note.id === noteEditing ? { ...note, note_text: text } : note)));
      setNoteEditing(null);
      setNoteEditDraft("");
    }
    setNoteLoading(false);
  };

  const deleteNote = async (id: number) => {
    setNoteLoading(true);
    setNoteMessage("");
    const res = await fetch(`${apiPath("notes")}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id })
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      setNoteMessage(payload?.error || "Failed to delete note.");
    } else {
      setNotes((prev) => prev.filter((note) => note.id !== id));
    }
    setNoteLoading(false);
  };

  const checklist = useMemo(() => {
    const rs = numberOrNull(data?.metrics?.rs_percentile ?? data?.watchlist?.rs_percentile) ?? null;
    const daq = numberOrNull(data?.signal?.daq_score ?? data?.watchlist?.daq_score) ?? null;
    const relVol = numberOrNull(data?.metrics?.relative_volume) ?? null;
    const earnings = data?.instrument?.earnings_date ? new Date(data.instrument.earnings_date) : null;
    const daysToEarnings = earnings ? Math.ceil((earnings.getTime() - Date.now()) / 86400000) : null;
    const marketRegime = data?.market_regime?.market_regime || "";

    return [
      { label: "RS percentile", value: rs, pass: rs !== null && rs >= 70, warn: rs !== null && rs >= 40 && rs < 70 },
      { label: "DAQ score", value: daq, pass: daq !== null && daq >= 70, warn: daq !== null && daq >= 50 && daq < 70 },
      { label: "Relative volume", value: relVol, pass: relVol !== null && relVol >= 1, warn: relVol !== null && relVol >= 0.7 && relVol < 1 },
      { label: "Earnings risk", value: daysToEarnings !== null ? `${daysToEarnings}d` : "N/A", pass: daysToEarnings === null || daysToEarnings > 7, warn: daysToEarnings !== null && daysToEarnings <= 14 },
      { label: "Market regime", value: marketRegime || "N/A", pass: marketRegime.includes("bull"), warn: marketRegime.includes("weak") }
    ];
  }, [data]);

  const technicalFallbacks = useMemo(() => {
    const trendScore = numberOrNull(data?.signal?.trend_score) ?? trendScoreFromStrength(data?.metrics?.trend_strength);
    const momentumScore = numberOrNull(data?.signal?.momentum_score);
    const volumeScore = numberOrNull(data?.signal?.volume_score);
    const momentumPct = numberOrNull(data?.metrics?.price_change_20d);
    const relVol = numberOrNull(data?.metrics?.relative_volume);
    return {
      trendScore,
      momentumScore,
      volumeScore,
      momentumPct,
      relVol
    };
  }, [data]);

  const analystSummary = useMemo(() => {
    const a = data?.analyst;
    const strongBuy = numberOrNull(a?.strong_buy) ?? 0;
    const buy = numberOrNull(a?.buy) ?? 0;
    const hold = numberOrNull(a?.hold) ?? 0;
    const sell = numberOrNull(a?.sell) ?? 0;
    const strongSell = numberOrNull(a?.strong_sell) ?? 0;
    const total = strongBuy + buy + hold + sell + strongSell;
    if (!total) {
      const analyst = data?.instrument?.analyst_rating;
      if (analyst) {
        const normalized = analyst.toLowerCase();
        if (normalized.includes("buy")) return { label: analyst, tone: "good" };
        if (normalized.includes("sell")) return { label: analyst, tone: "bad" };
        if (normalized.includes("hold")) return { label: analyst, tone: "warn" };
        return { label: analyst, tone: "warn" };
      }
      return { label: "No data", tone: "" };
    }
    const score = strongBuy * 2 + buy - sell - strongSell * 2;
    const strongThreshold = total * 0.5;
    const mildThreshold = total * 0.2;
    if (score >= strongThreshold) return { label: "Strong Buy", tone: "good" };
    if (score >= mildThreshold) return { label: "Buy", tone: "good" };
    if (score <= -strongThreshold) return { label: "Strong Sell", tone: "bad" };
    if (score <= -mildThreshold) return { label: "Sell", tone: "bad" };
    return { label: "Hold", tone: "warn" };
  }, [data]);

  const forecast = useMemo(() => {
    const current = numberOrNull(data?.metrics?.current_price);
    const avg = numberOrNull(data?.instrument?.target_price);
    const max = numberOrNull(data?.instrument?.target_high);
    const min = numberOrNull(data?.instrument?.target_low);
    if (!current) return null;
    if (avg === null && max === null && min === null) return null;
    return {
      current,
      avg,
      max,
      min,
      analysts: data?.instrument?.number_of_analysts ?? null
    };
  }, [data]);

  const onSearch = () => {
    const next = tickerInput.trim().toUpperCase();
    if (next) {
      router.push(`/stocks/${next}`);
    }
  };

  return (
    <div className="page deepdive-page">
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
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="deepdive-header">
        <div>
          <h1>{ticker} Deep Dive</h1>
          <p>{data?.instrument?.name || "Stock intelligence and execution plan"}</p>
        </div>
        <div className="ticker-search">
          <input
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            placeholder="Search ticker (TSLA)"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                onSearch();
              }
            }}
          />
          <button onClick={onSearch}>Analyze</button>
        </div>
      </div>

      {loading ? (
        <div className="footer-note">Loading deep dive data...</div>
      ) : (
        <>
          <div className="summary-grid">
            <div className="summary-card">
              Price
              <strong>{formatValue(data?.metrics?.current_price ?? data?.watchlist?.price ?? data?.signal?.entry_price, 2)}</strong>
            </div>
            <div className="summary-card">
              RS
              <strong>{formatValue(data?.metrics?.rs_percentile ?? data?.watchlist?.rs_percentile, 0)}</strong>
            </div>
            <div className="summary-card">
              DAQ
              <strong>{formatValue(data?.signal?.daq_score ?? data?.watchlist?.daq_score, 0)}</strong>
            </div>
            <div className="summary-card">
              Analyst
              <strong className={analystSummary.tone ? `pill ${analystSummary.tone}` : ""}>
                {analystSummary.label}
                {data?.instrument?.number_of_analysts ? ` (${data.instrument.number_of_analysts})` : ""}
              </strong>
            </div>
            <div className="summary-card">
              Earnings
              <strong>{formatDate(data?.instrument?.earnings_date)}</strong>
            </div>
          </div>

          <div className="deepdive-grid">
            <div className="deepdive-col">
              <div className="panel-card">
                <h3>Thesis & Checklist</h3>
                <p className="muted">
                  {data?.signal?.claude_thesis ||
                    data?.signal?.setup_description ||
                    "No thesis available yet."}
                </p>
                <div className="checklist">
                  {checklist.map((item) => (
                    <div className="check-row" key={item.label}>
                      <span>{item.label}</span>
                      <span className={`check-pill ${item.pass ? "good" : item.warn ? "warn" : "bad"}`}>
                        {item.value ?? "-"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel-card">
                <h3>Risk & Execution</h3>
                <div className="detail-grid">
                  <div className="detail-item">Entry Low: {formatValue(data?.watchlist?.suggested_entry_low)}</div>
                  <div className="detail-item">Entry High: {formatValue(data?.watchlist?.suggested_entry_high)}</div>
                  <div className="detail-item">Stop Loss: {formatValue(data?.watchlist?.suggested_stop_loss)}</div>
                  <div className="detail-item">Target 1: {formatValue(data?.watchlist?.suggested_target_1)}</div>
                  <div className="detail-item">Target 2: {formatValue(data?.watchlist?.suggested_target_2)}</div>
                <div className="detail-item">R:R: {formatValue(data?.watchlist?.risk_reward_ratio, 2)}</div>
                <div className="detail-item">ATR%: {formatValue(data?.watchlist?.atr_percent ?? data?.metrics?.atr_percent, 2)}</div>
                <div className="detail-item">
                  Relative Vol: {formatValue(data?.watchlist?.relative_volume ?? data?.metrics?.relative_volume, 2)}x
                </div>
              </div>
            </div>

              <div className="panel-card">
                <h3>Notes & Journal</h3>
                <textarea
                  value={noteDraft}
                  onChange={(e) => setNoteDraft(e.target.value)}
                  placeholder="Log thesis, risk plan, or post-trade notes..."
                />
                <div className="notes-actions">
                  <button className="notes-btn" onClick={saveNote} disabled={noteLoading || !noteDraft.trim()}>
                    {noteLoading ? "Saving..." : "Save note"}
                  </button>
                  {noteMessage ? <span className="notes-status">{noteMessage}</span> : null}
                </div>
                <div className="notes-list">
                  {noteLoading && !notes.length ? (
                    <div className="footer-note">Loading notes...</div>
                  ) : notes.length ? (
                    notes.map((note) => (
                      <div className="note-item" key={note.id}>
                        <div className="note-meta">
                          {new Date(note.created_at).toLocaleString("en-US", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })}
                          <span className="note-actions">
                            <button
                              className="note-link"
                              onClick={() => {
                                setNoteEditing(note.id);
                                setNoteEditDraft(note.note_text);
                              }}
                            >
                              Edit
                            </button>
                            <button className="note-link danger" onClick={() => deleteNote(note.id)}>
                              Delete
                            </button>
                          </span>
                        </div>
                        {noteEditing === note.id ? (
                          <div>
                            <textarea
                              value={noteEditDraft}
                              onChange={(e) => setNoteEditDraft(e.target.value)}
                            />
                            <div className="notes-actions">
                              <button className="notes-btn" onClick={saveNoteEdit} disabled={noteLoading || !noteEditDraft.trim()}>
                                {noteLoading ? "Saving..." : "Save edit"}
                              </button>
                              <button
                                className="notes-btn secondary"
                                onClick={() => {
                                  setNoteEditing(null);
                                  setNoteEditDraft("");
                                }}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="note-text">{note.note_text}</div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="footer-note">No notes yet.</div>
                  )}
                </div>
              </div>
            </div>

            <div className="deepdive-col">
              <div className="chart-grid">
                <div className="chart-card">
                  <div className="chart-title">Daily Chart</div>
                  <iframe
                    src={`https://www.tradingview.com/widgetembed/?symbol=${encodeURIComponent(
                      tvSymbol(data?.instrument?.exchange, ticker)
                    )}&interval=D&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1`}
                    className="tv-frame"
                    loading="lazy"
                    title={`${ticker} daily chart`}
                    allowFullScreen
                    referrerPolicy="no-referrer-when-downgrade"
                  />
                </div>
                <div className="chart-card">
                  <div className="chart-title">Weekly Chart</div>
                  <iframe
                    src={`https://www.tradingview.com/widgetembed/?symbol=${encodeURIComponent(
                      tvSymbol(data?.instrument?.exchange, ticker)
                    )}&interval=W&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1`}
                    className="tv-frame"
                    loading="lazy"
                    title={`${ticker} weekly chart`}
                    allowFullScreen
                    referrerPolicy="no-referrer-when-downgrade"
                  />
                </div>
              </div>

              <div className="panel-card">
                <h3>Technical Summary</h3>
                <div className="detail-grid">
                  <div className="detail-item">RSI: {formatValue(data?.metrics?.rsi_14, 0)}</div>
                  <div className="detail-item">MACD: {formatValue(data?.metrics?.macd, 2)}</div>
                  <div className="detail-item">ATR%: {formatValue(data?.metrics?.atr_percent, 2)}</div>
                  <div className="detail-item">Trend Score: {technicalFallbacks.trendScore ?? "-"}</div>
                  <div className="detail-item">
                    Momentum: {technicalFallbacks.momentumScore !== null
                      ? technicalFallbacks.momentumScore.toFixed(1)
                      : technicalFallbacks.momentumPct !== null
                        ? `${technicalFallbacks.momentumPct.toFixed(1)}%`
                        : "-"}
                  </div>
                  <div className="detail-item">
                    Volume Score: {technicalFallbacks.volumeScore !== null
                      ? technicalFallbacks.volumeScore.toFixed(1)
                      : technicalFallbacks.relVol !== null
                        ? `${technicalFallbacks.relVol.toFixed(2)}x`
                        : "-"}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="deepdive-bottom">
            <div className="panel-card">
              <h3>Catalysts & Sentiment</h3>
              <div className="reco-grid">
                <div><span>Strong Buy</span><strong>{data?.analyst?.strong_buy ?? 0}</strong></div>
                <div><span>Buy</span><strong>{data?.analyst?.buy ?? 0}</strong></div>
                <div><span>Hold</span><strong>{data?.analyst?.hold ?? 0}</strong></div>
                <div><span>Sell</span><strong>{data?.analyst?.sell ?? 0}</strong></div>
                <div><span>Strong Sell</span><strong>{data?.analyst?.strong_sell ?? 0}</strong></div>
              </div>
              <div className="detail-grid">
                <div className="detail-item">Consensus: {data?.instrument?.analyst_rating || "-"}</div>
                <div className="detail-item">Target: {data?.instrument?.target_price ? `$${Number(data.instrument.target_price).toFixed(2)}` : "-"}</div>
                <div className="detail-item">Target High: {data?.instrument?.target_high ? `$${Number(data.instrument.target_high).toFixed(2)}` : "-"}</div>
                <div className="detail-item">Target Low: {data?.instrument?.target_low ? `$${Number(data.instrument.target_low).toFixed(2)}` : "-"}</div>
              </div>
              {forecast ? (
                <ForecastChart
                  current={forecast.current}
                  avg={forecast.avg}
                  max={forecast.max}
                  min={forecast.min}
                  analysts={forecast.analysts}
                />
              ) : (
                <div className="footer-note">No forecast targets available.</div>
              )}
              <div className="news-list">
                {(data?.news || []).map((headline) => (
                  <a className="news-item" key={headline.url || headline.headline} href={headline.url || "#"} target="_blank" rel="noreferrer">
                    <div>
                      <strong>{headline.headline}</strong>
                      <span>{headline.source || "News"}</span>
                    </div>
                    <div className="news-meta">{formatDate(headline.published_at)}</div>
                  </a>
                ))}
                {!data?.news?.length ? <div className="footer-note">No recent news cached.</div> : null}
              </div>
            </div>

            <div className="panel-card">
              <h3>Signal History</h3>
              <div className="history-list">
                {(data?.signal_history || []).map((signal) => (
                  <div className="history-row" key={signal.id}>
                    <span>{new Date(signal.signal_timestamp).toLocaleDateString("en-US", { month: "short", day: "2-digit" })}</span>
                    <span>{signal.scanner_name.replace(/_/g, " ")}</span>
                    <span>{signal.signal_type || "-"}</span>
                    <span>{formatValue(signal.composite_score, 1)}</span>
                    <span>{signal.quality_tier || "-"}</span>
                    <span>{signal.status || "-"}</span>
                  </div>
                ))}
                {!data?.signal_history?.length ? <div className="footer-note">No signal history found.</div> : null}
              </div>
            </div>

            <div className="panel-card">
              <h3>Market & Sector Context</h3>
              <div className="detail-grid">
                <div className="detail-item">Market Regime: {data?.market_regime?.market_regime || "-"}</div>
                <div className="detail-item">Volatility: {data?.market_regime?.volatility_regime || "-"}</div>
                <div className="detail-item">SPY vs SMA50: {formatValue(data?.market_regime?.spy_vs_sma50_pct, 1)}%</div>
                <div className="detail-item">SPY vs SMA200: {formatValue(data?.market_regime?.spy_vs_sma200_pct, 1)}%</div>
                <div className="detail-item">Sector RS: {formatValue(data?.sector_context?.rs_vs_spy, 2)}</div>
                <div className="detail-item">Sector Stage: {data?.sector_context?.sector_stage || "-"}</div>
              </div>
              {data?.market_regime?.strategy_guidance ? (
                <p className="muted">{data.market_regime.strategy_guidance}</p>
              ) : null}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
