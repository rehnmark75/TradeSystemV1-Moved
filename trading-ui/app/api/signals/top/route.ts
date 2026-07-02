import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const NO_STORE_HEADERS = {
  "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
  Pragma: "no-cache",
  Expires: "0",
};

// --- Scanner edge floor -------------------------------------------------
// scanner_pf is a closed-trade profit factor. Winners and losers must use
// the same status='closed' population, otherwise open/expired/partial rows can
// inflate the edge metric. A scanner with closed wins and zero closed losses is
// treated as capped-positive PF; a scanner with no closed outcome history stays
// null and is not floor-filtered until it reaches EDGE_MIN_CLOSED.
const EDGE_WINDOW_DAYS = 60; // trailing window for scanner edge (larger = more stable)
const EDGE_PF_FLOOR = 1.0; // drop a scanner below this PF...
const EDGE_MIN_CLOSED = 10; // ...only once it has >= this many closed trades
const EDGE_NO_LOSS_PF_CAP = 3.0; // score cap for closed wins with zero closed losses
const DAYTRADE_SIGNAL_DATE_COUNT = 2; // include today's partial scan plus latest complete batch
// Monitor-only scanners: still emit/log signals to stock_scanner_signals (so the
// nightly monitor_only_outcomes forward-test layer tracks them), but are excluded
// fail-closed from the tradable day-trade candidate pool the auto-trader pulls.
// Use this to forward-test an unvalidated scanner on the LIVE account without
// allocating capital. There is no per-scanner DB flag in the stock path, so this
// list is the single chokepoint that keeps a scanner out of live execution.
// high_retest (re-enabled Jun 2026): edge survives day-trade re-grade (PF ~3.0)
// but only 8 trading sessions of one bull regime, never forward-tested.
// zlma_trend + regime_adaptive_composite (demoted Jul 1 2026): live per-scanner
// review found these two = 72% of the candidate pool but ~100%+ of the realized
// net loss (zlma_trend PF 0.25 / -$128 on 13 closed; regime_adaptive_composite
// 0% WR / -$39 on 4). candidate_score had no discriminating power even within
// zlma_trend (corr -0.36), so a higher score bar can't rescue them. Kept
// monitor-only (still logged for re-validation) rather than deleted. The other
// 6 scanners were net +$64/12 trades over the same window.
const MONITOR_ONLY_SCANNERS = ["high_retest", "zlma_trend", "regime_adaptive_composite"];
// Edge-map router (c) — SHADOW MODE ONLY. When true, log the intended per-cell
// routing decision for every returned candidate WITHOUT changing what trades or
// what this endpoint returns. Maps each candidate's stored character cell
// (trend_state x vol_regime, off migration 043) to its learned verdict in
// scanner_cell_edge: block->would-drop, monitor->would-hold, trade->allowed,
// missing/insufficient->fail-open (allowed). Never drops for real. Flip to false
// to silence. There is deliberately NO enforce mode here.
const CELL_ROUTER_SHADOW = true;
// Day-trade EMA50 freshness gate (default; overridable via ?maxEmaCrossAgeSessions).
// A candidate is kept only if it is currently above its daily EMA50 AND the upward
// cross happened within this many trading sessions (i.e. a fresh reclaim, not an
// established multi-week trend). Computed from stock_screening_metrics history.
const DEFAULT_MAX_EMA_CROSS_AGE_SESSIONS = 10;
const ROBOMARKETS_API_URL = process.env.ROBOMARKETS_API_URL || "https://api.stockstrader.com/api/v1";
const ROBOMARKETS_API_KEY = process.env.ROBOMARKETS_API_KEY || "";
const ROBOMARKETS_ACCOUNT_ID = process.env.ROBOMARKETS_ACCOUNT_ID || "";

// Candidate score (validated against the live candidate pool):
//   0.55 * rs_percentile
// + 0.45 * rescaled tv_overall_score (-100..100 -> 0..100)
// + rs_trend: improving +5 / stable 0 / deteriorating -15   (user prefers RISING RS)
// - risk penalties: earnings<=7d (-15), high short interest (-10), RSI>80 overbought (-10)
//
// Day-trade score (0-100 core, bounded sub-scores so each weight == its max
// influence). Leads with "in play" (volume + range), not swing leadership:
//   0.30 RVOL (prior-session daily) + 0.14 daily range + 0.16 RS (demoted from
//   0.28) + 0.12 TV consensus + 0.12 premarket conviction + 0.08 live news
//   + 0.08 entry-not-extended (vs EMA20, ATR units).
// Why RS was demoted: measured on the live tradable pool, RS-led ordering
// floats quiet names to the top (7 of the top-10 had RVOL <= 1.1 while the
// day's actual movers ranked #10-#17). A day trade needs activity, not a
// multi-week leadership badge. ± rising-RS, premarket-gap, overbought,
// days-to-earnings, plus day-trade adjustments: a SOFT (not hard) low-liquidity
// penalty, a low-float-momentum bonus, a short-interest reframe (crowded+quiet
// = risk, crowded+active = squeeze fuel), and a 52w-breakout / 5d-extension tilt.
// Dead/missing DAQ terms stay removed (sector/regime stuck on fallbacks,
// catalyst pinned ~100, financial quality is a swing factor).
//
// Live enrichment (additive-only): the stock-scanner intraday-vwap worker pulls
// batched 5m bars from yfinance for this candidate pool during the open window
// and writes derived scalars (session VWAP, cumulative volume, intraday RVOL
// pace) to stock_intraday_state. For covered names we fold those in as a bounded
// BONUS (never the core term -- mixing live mid-session RVOL with others'
// prior-session RVOL would be apples-to-oranges). VWAP position feeds that bonus
// (holding above VWAP = full bonus, below = halved); the EMA20 extension term is
// unchanged. The broker quote API carries no volume, so live RVOL comes from the
// 5m worker, not the quote.
//
// Tradability gate (ranks ABOVE score; non-tradable rows sink and are dimmed):
// a current-session premarket BUY + acceptable spread/room. SECOND path: a name
// confirmed in play by live volume (intraday RVOL pace >= 2, holding >= VWAP) is
// tradable even WITHOUT a premarket row -- this breaks a feedback loop, because
// the 9:00 AM ET premarket-pricing job enriches exactly the tickers THIS route
// returns as the Top 50, so a mover buried for lack of PM data would never get
// enriched. Surfacing it lets the next cycle enrich it.
//
// The row projection MIRRORS /api/signals/results so the signals-page detail
// panel can render an expanded Top-10 row with full field parity.

type QuoteSnapshot = {
  broker_bid: number | null;
  broker_ask: number | null;
  broker_last: number | null;
  broker_volume: number | null;
  broker_spread: number | null;
  broker_spread_pct: number | null;
  broker_quote_time: string | null;
};

const firstFiniteNumber = (...values: unknown[]) => {
  for (const value of values) {
    if (value == null) continue;
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const fetchBrokerQuotes = async (tickers: string[]): Promise<Record<string, QuoteSnapshot>> => {
  if (!ROBOMARKETS_API_KEY || !ROBOMARKETS_ACCOUNT_ID || tickers.length === 0) return {};

  const entries = await Promise.all(
    tickers.map(async (ticker) => {
      try {
        const res = await fetch(
          `${ROBOMARKETS_API_URL}/accounts/${ROBOMARKETS_ACCOUNT_ID}/instruments/${encodeURIComponent(ticker)}/quote`,
          {
            headers: {
              Authorization: `Bearer ${ROBOMARKETS_API_KEY}`,
              Accept: "application/json",
            },
            cache: "no-store",
          }
        );
        if (!res.ok) return [ticker, null] as const;
        const payload = await res.json();
        const data = payload?.data ?? {};
        const bid = data.bid_price == null ? null : Number(data.bid_price);
        const ask = data.ask_price == null ? null : Number(data.ask_price);
        const last = data.last_price == null ? null : Number(data.last_price);
        const volume = firstFiniteNumber(
          data.volume,
          data.day_volume,
          data.daily_volume,
          data.traded_volume,
          data.turnover_volume
        );
        const spread = bid != null && ask != null ? ask - bid : null;
        const mid = bid != null && ask != null ? (bid + ask) / 2 : last;
        const quoteTime =
          data.ask_bid_price_time != null
            ? new Date(Number(data.ask_bid_price_time) * 1000).toISOString()
            : data.last_price_time != null
              ? new Date(Number(data.last_price_time) * 1000).toISOString()
              : null;
        return [
          ticker,
          {
            broker_bid: bid,
            broker_ask: ask,
            broker_last: last,
            broker_volume: volume,
            broker_spread: spread,
            broker_spread_pct: spread != null && mid ? Number(((spread / mid) * 100).toFixed(3)) : null,
            broker_quote_time: quoteTime,
          },
        ] as const;
      } catch {
        return [ticker, null] as const;
      }
    })
  );

  return Object.fromEntries(entries.filter((entry): entry is readonly [string, QuoteSnapshot] => entry[1] !== null));
};

type IntradayCandleStat = {
  intraday_rvol_pace: number | null; // today's session vol / 20d avg over the SAME hours-of-day
  session_vwap: number | null;
  last_close: number | null;
  bars_today: number | null;
  candle_session: string | null;
  live_as_of: string | null; // ISO UTC; when the intraday-vwap worker last wrote this row
};

// Live intraday enrichment, sourced from stock_intraday_state -- a small
// derived-scalar table written by the stock-scanner intraday-vwap worker, which
// pulls batched 5m bars from yfinance for the day-trade candidate pool only,
// during the open window (~9:25-10:30 ET). session_vwap is a real sub-hourly
// VWAP (not a single 1h bar) and intraday_rvol_pace is today's cumulative volume
// vs a time-of-day-adjusted daily baseline. Scoped to TODAY's ET session: before
// the worker runs (or off-session) there are no rows, so callers cleanly fall
// back to prior-session daily RVOL with no live bonus. (yfinance intraday is
// ~15 min delayed, so treat this as a surge/in-play read, not tick-precise.)
const fetchIntradayCandleStats = async (
  client: { query: (text: string, params: unknown[]) => Promise<{ rows: Record<string, unknown>[] }> },
  tickers: string[]
): Promise<Record<string, IntradayCandleStat>> => {
  if (tickers.length === 0) return {};
  try {
    const { rows } = await client.query(
      `
      SELECT ticker,
             intraday_rvol_pace,
             session_vwap,
             last_price AS last_close,
             bars_today,
             trade_date::text AS candle_session,
             as_of
      FROM stock_intraday_state
      WHERE ticker = ANY($1)
        AND trade_date = (NOW() AT TIME ZONE 'America/New_York')::date
        -- Freshness guard: if the worker dies mid-session its rows freeze. The
        -- worker refreshes every ~150s, so anything older than 20 min means the
        -- live feed is down -> drop it and fall back to prior-session daily
        -- rather than serve a stale VWAP as "live" (matters once this feeds a
        -- hard gate, not just the bonus).
        AND as_of > NOW() - INTERVAL '20 minutes'
      `,
      [tickers]
    );
    const out: Record<string, IntradayCandleStat> = {};
    for (const r of rows) {
      out[String(r.ticker)] = {
        intraday_rvol_pace: r.intraday_rvol_pace == null ? null : Number(r.intraday_rvol_pace),
        session_vwap: r.session_vwap == null ? null : Number(r.session_vwap),
        last_close: r.last_close == null ? null : Number(r.last_close),
        bars_today: r.bars_today == null ? null : Number(r.bars_today),
        candle_session: r.candle_session == null ? null : String(r.candle_session),
        live_as_of: r.as_of == null ? null : new Date(r.as_of as string).toISOString(),
      };
    }
    return out;
  } catch (err) {
    console.error("intraday candle stats query failed", err);
    return {};
  }
};

// Edge-map router (c) — 2-axis cell verdict lookup (SHADOW MODE).
// Fetches the learned (scanner_name x trend_state x vol_regime) edge rows in ONE
// batched query and returns a Map keyed "scanner|trend|vol" for O(1) join in JS.
// Restricted to the 2-axis grid (liquidity_tier IS NULL AND market_regime IS
// NULL) per INTEGRATION_NOTES step (c). pf is cast ::float8 (pg numeric -> JS
// string gotcha); verdict is text so it is safe as-is.
type CellEdge = { verdict: string | null; pf: number | null; n: number | null };
const fetchCellEdgeMap = async (
  client: { query: (text: string, params: unknown[]) => Promise<{ rows: Record<string, unknown>[] }> },
  scannerNames: string[]
): Promise<Map<string, CellEdge>> => {
  const map = new Map<string, CellEdge>();
  if (scannerNames.length === 0) return map;
  try {
    const { rows } = await client.query(
      `
      SELECT scanner_name, trend_state, vol_regime, verdict, pf::float8 AS pf, n
      FROM scanner_cell_edge
      WHERE liquidity_tier IS NULL
        AND market_regime IS NULL
        AND scanner_name = ANY($1)
      `,
      [scannerNames]
    );
    for (const r of rows) {
      const key = `${String(r.scanner_name)}|${String(r.trend_state)}|${String(r.vol_regime)}`;
      map.set(key, {
        verdict: r.verdict == null ? null : String(r.verdict),
        pf: r.pf == null ? null : Number(r.pf),
        n: r.n == null ? null : Number(r.n),
      });
    }
  } catch (err) {
    console.warn("[CELL-ROUTER-SHADOW] edge-map fetch failed (non-fatal):", err);
  }
  return map;
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Math.min(Math.max(1, Number(searchParams.get("limit") || 10)), 50);
  const mode = searchParams.get("mode") === "daytrades" ? "daytrades" : "candidates";
  const maxPerScanner = Math.min(
    limit,
    Math.max(1, Number(searchParams.get("maxPerScanner") || limit))
  );
  const queryLimit = mode === "daytrades" ? Math.min(50, Math.max(limit, limit * 3)) : limit;
  // EMA50 freshness threshold (trading sessions). Daytrades-only gate; clamped 0-60.
  const rawCrossAge = searchParams.get("maxEmaCrossAgeSessions");
  const parsedCrossAge = rawCrossAge === null ? DEFAULT_MAX_EMA_CROSS_AGE_SESSIONS : Number(rawCrossAge);
  const maxEmaCrossAgeSessions = Number.isFinite(parsedCrossAge)
    ? Math.min(60, Math.max(0, Math.floor(parsedCrossAge)))
    : DEFAULT_MAX_EMA_CROSS_AGE_SESSIONS;

  const client = await pool.connect();
  try {
    const scoreExpression =
      mode === "daytrades"
        ? `
              -- Bounded 0-100 sub-scores; core weights sum to 1.0 so each weight
              -- equals that factor's maximum influence. Absolute (not batch-
              -- relative) scales on purpose: a day trade needs activity to be
              -- high outright, not merely high relative to a weak batch. Leads
              -- with volume + range; RS is demoted to a context term (see header).
              -- All core terms use 100%-populated daily metrics so every name is
              -- scored on the same basis (live intraday RVOL/VWAP is added in JS
              -- as a bonus for the sparse streaming subset).
              -- RVOL ceiling lowered 3.0 -> 2.0 for the swing (1-3 day) horizon:
              -- a name at 2x average volume is sufficiently active to enter and
              -- exit over multiple days; rewarding 3x+ over-credits explosive
              -- intraday movers that suit day trading, not a multi-day hold.
              0.30 * LEAST(COALESCE(relative_volume, 0) / 2.0 * 100, 100)
            -- Range earns credit only in PROPORTION to how active the name is:
            -- a wide range on light volume is not a tradable mover, just a
            -- volatile sleeper. Scaled by RVOL capped at 1.0 (at/above average
            -- volume earns full range credit; below average is damped toward 0).
            -- Both range and RVOL here are the prior completed session's
            -- realized values (metrics batch is one day lagged); live intraday
            -- pace, when streaming, is layered on as the JS bonus.
            + 0.14 * LEAST(COALESCE(daily_range_percent, 0) / 5.0 * 100, 100)
                   * LEAST(COALESCE(relative_volume, 0), 1.0)
            + 0.16 * COALESCE(rs_percentile, 0)
            + 0.12 * ((COALESCE(tv_overall_score, 0) + 100) / 2.0)
            + 0.12 * (
                CASE
                  WHEN pm_is_current_session AND pm_direction = 'BUY'
                    THEN LEAST(COALESCE(pm_confidence, 0) * 100, 100)
                  ELSE 50
                END
              )
            + 0.08 * COALESCE(daq_news_score, 50)
            + 0.08 * (
                -- entry-not-extended, SWING benchmark: 100 at/below EMA50 (the
                -- institutional swing-pullback reference), decaying ~15pts per ATR
                -- above it. Was EMA20 (an intraday-reversion proxy) -- too tight
                -- for a 1-3 day hold, whose targets are larger so a wider ATR
                -- tolerance is appropriate. Neutral 60 when ATR/EMA unavailable
                -- (atr_14 is 'NaN' during warmup, which would otherwise sort to TOP).
                CASE
                  WHEN atr_14 IS NULL OR atr_14 <= 0 OR atr_14 = 'NaN'
                    OR entry_price IS NULL OR entry_price = 'NaN'
                    OR ema_50 IS NULL OR ema_50 = 'NaN'
                  THEN 60
                  ELSE GREATEST(0, 100 - 15 * GREATEST((entry_price - ema_50) / atr_14, 0))
                END
              )
            -- rs_trend 'deteriorating' penalty softened -12 -> -5: ~92% of the
            -- universe carries the 'deteriorating' label, so the old -12 was a
            -- near-constant floor reduction (no discrimination), not a signal.
            + (CASE rs_trend WHEN 'improving' THEN 5 WHEN 'deteriorating' THEN -5 ELSE 0 END)
            + (CASE
                WHEN pm_is_current_session AND pm_direction = 'BUY' AND pm_gap_percent BETWEEN 1 AND 8 THEN 6
                WHEN pm_is_current_session AND pm_direction = 'BUY' AND pm_gap_percent > 8 THEN 2
                WHEN pm_is_current_session AND pm_direction = 'SELL' THEN -10
                ELSE 0
              END)
            - (CASE WHEN rsi_14 > 80 THEN 12 WHEN rsi_14 > 75 THEN 6 ELSE 0 END)
            -- Liquidity: SOFT size-warning penalty, NOT a hard floor. Low-float
            -- movers are often low-ADV; you trade them smaller, not never.
            -- avg_dollar_volume is in dollars. (Thresholds unvalidated.)
            - (CASE
                WHEN avg_dollar_volume IS NULL THEN 0
                WHEN avg_dollar_volume < 3000000 THEN 8
                WHEN avg_dollar_volume < 8000000 THEN 4
                ELSE 0
              END)
            -- Short interest reframed (was a flat -8): crowded + quiet = trapped
            -- risk; crowded + active = squeeze fuel on a long.
            + (CASE
                WHEN high_short_interest AND COALESCE(relative_volume, 0) >= 2 THEN 5
                WHEN high_short_interest AND COALESCE(relative_volume, 0) < 1 THEN -6
                WHEN high_short_interest THEN -2
                ELSE 0
              END)
            -- Low-float momentum: a tight float that is actually trading today is
            -- the explosive intraday profile. shares_float is in shares.
            + (CASE
                WHEN shares_float IS NULL OR shares_float <= 0 THEN 0
                WHEN shares_float < 20000000 AND COALESCE(relative_volume, 0) >= 1.5 THEN 6
                WHEN shares_float < 50000000 AND COALESCE(relative_volume, 0) >= 1.5 THEN 3
                ELSE 0
              END)
            -- Breakout fuel (near 52w high) vs over-extended multi-day chase.
            + (CASE WHEN COALESCE(pct_from_52w_high, -100) >= -3 THEN 3 ELSE 0 END)
            - (CASE WHEN COALESCE(price_change_5d, 0) > 25 THEN 4 ELSE 0 END)
            - (CASE
                -- Real earnings guard on the 98%-populated days_to_earnings, not
                -- the unreachable catalyst<50 gate of the old formula.
                WHEN days_to_earnings <= 1 THEN 18
                WHEN days_to_earnings <= 3 THEN 12
                WHEN days_to_earnings <= 7 THEN 6
                ELSE 0
              END)
`
        : `
              0.55 * COALESCE(rs_percentile, 0)
            + 0.45 * ((COALESCE(tv_overall_score, 0) + 100) / 2.0)
            + (CASE rs_trend WHEN 'improving' THEN 5 WHEN 'deteriorating' THEN -15 ELSE 0 END)
            - (CASE
                WHEN days_to_earnings <= 1 THEN 18
                WHEN days_to_earnings <= 3 THEN 15
                WHEN days_to_earnings <= 7 THEN 8
                ELSE 0
              END)
            - (CASE WHEN high_short_interest THEN 10 ELSE 0 END)
            - (CASE WHEN rsi_14 > 80 THEN 10 ELSE 0 END)
        `;

    const query = `
      WITH scanner_pctl AS (
        -- Per-scanner 5th/95th percentile of realized_pnl_pct over the window.
        -- Used to winsorize before computing PF, so a single penny-stock survivor
        -- (e.g. realized +350%) or one blow-up loss can't dominate a scanner's
        -- edge metric. The native-SL/TP realized series is heavily fat-tailed on
        -- low-priced names; clamping the tails makes the floor track the
        -- systematic edge rather than outlier luck.
        SELECT
          scanner_name,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY realized_pnl_pct) AS p95,
          percentile_cont(0.05) WITHIN GROUP (ORDER BY realized_pnl_pct) AS p05
        FROM stock_scanner_signals
        WHERE signal_timestamp >= NOW() - ($1 * INTERVAL '1 day')
          AND status = 'closed'
          AND realized_pnl_pct IS NOT NULL
        GROUP BY scanner_name
      ),
      scanner_edge AS (
        SELECT
          scanner_name,
          closed_n,
          CASE
            WHEN gross_loss > 0 THEN gross_profit / gross_loss
            WHEN gross_profit > 0 THEN $6::numeric
            ELSE NULL
          END AS pf
        FROM (
          SELECT
            ss.scanner_name,
            count(*) AS closed_n,
            COALESCE(sum(w) FILTER (WHERE w > 0), 0)::numeric AS gross_profit,
            ABS(COALESCE(sum(w) FILTER (WHERE w <= 0), 0))::numeric AS gross_loss
          FROM (
            SELECT
              s.scanner_name,
              LEAST(GREATEST(s.realized_pnl_pct, p.p05), p.p95) AS w
            FROM stock_scanner_signals s
            JOIN scanner_pctl p ON s.scanner_name = p.scanner_name
            WHERE s.signal_timestamp >= NOW() - ($1 * INTERVAL '1 day')
              AND s.status = 'closed'
              AND s.realized_pnl_pct IS NOT NULL
          ) ss
          GROUP BY ss.scanner_name
        ) closed_edge
      ),
      latest_batch AS (
        SELECT DISTINCT ON (ticker, scanner_name) *
        FROM stock_scanner_signals
        WHERE ${
          mode === "daytrades"
            ? `
          signal_type = 'BUY'
          ${
            MONITOR_ONLY_SCANNERS.length
              ? `AND scanner_name NOT IN (${MONITOR_ONLY_SCANNERS.map((s) => `'${s}'`).join(", ")})`
              : ""
          }
          AND signal_date IN (
            SELECT signal_date
            FROM (
              SELECT DISTINCT signal_date
              FROM stock_scanner_signals
              ORDER BY signal_date DESC
              LIMIT ${DAYTRADE_SIGNAL_DATE_COUNT}
            ) recent_signal_dates
          )
        `
            : "signal_date = (SELECT max(signal_date) FROM stock_scanner_signals)"
        }
        ORDER BY ticker, scanner_name, signal_timestamp DESC
      ),
      -- EMA50 freshness: from the stored daily metrics history, compute whether
      -- each pool ticker is currently above its EMA50 and how many trading
      -- sessions ago the current above-EMA50 run began (the upward cross). Uses
      -- the SAME stored ema_50 the scoring uses, so the snapshot and the cross
      -- age can never disagree at the boundary. Restricted to pool tickers for
      -- speed. A name continuously above EMA50 for its entire recorded history
      -- has no in-history cross -> age NULL -> excluded as an established trend.
      ema_hist AS (
        SELECT
          sm.ticker,
          (sm.current_price > sm.ema_50) AS above,
          ROW_NUMBER() OVER (PARTITION BY sm.ticker ORDER BY sm.calculation_date) AS rn
        FROM stock_screening_metrics sm
        WHERE sm.ema_50 IS NOT NULL
          AND sm.ema_50::text <> 'NaN'
          AND sm.current_price IS NOT NULL
          AND sm.current_price::text <> 'NaN'
          AND sm.ticker IN (SELECT DISTINCT ticker FROM latest_batch)
      ),
      ema_cross AS (
        SELECT
          ticker,
          -- the latest row (max rn) is above its EMA50
          (MAX(rn) FILTER (WHERE above) = MAX(rn)) AS above_ema50,
          CASE
            WHEN (MAX(rn) FILTER (WHERE above) = MAX(rn))
                 AND MAX(rn) FILTER (WHERE NOT above) IS NOT NULL
              -- sessions between the last below-EMA50 day and now = cross age
              THEN MAX(rn) - MAX(rn) FILTER (WHERE NOT above) - 1
            ELSE NULL
          END AS ema50_cross_age_sessions
        FROM ema_hist
        GROUP BY ticker
      ),
      enriched AS (
        SELECT
          s.id,
          s.signal_timestamp,
          s.scanner_name,
          s.ticker,
          s.signal_type,
          s.entry_price,
          s.composite_score,
          s.quality_tier,
          s.status,
          s.trend_score,
          s.momentum_score,
          s.volume_score,
          s.pattern_score,
          s.risk_percent,
          s.risk_reward_ratio,
          s.setup_description,
          s.confluence_factors,
          s.timeframe,
          s.market_regime,
          -- Edge-map router (c): stored as-of character cell (migration 043).
          -- text columns, so no ::float8 cast needed. Flow through SELECT * below.
          s.trend_state,
          s.vol_regime,
          s.claude_grade,
          s.claude_score,
          s.claude_action,
          s.claude_thesis,
          s.claude_key_strengths,
          s.claude_key_risks,
          s.claude_analyzed_at,
          s.news_sentiment_score,
          s.news_sentiment_level,
          s.news_headlines_count,
          i.name AS company_name,
          i.sector,
          COALESCE(i.exchange, 'NASDAQ') AS exchange,
          i.analyst_rating,
          i.target_price,
          i.number_of_analysts,
          i.shares_float,
          i.short_percent_float,
          m.rs_percentile,
          m.rs_trend,
          m.atr_14,
          m.atr_percent,
          m.swing_high,
          m.swing_low,
          m.swing_high_date,
          m.swing_low_date,
          m.avg_volume_20,
          m.relative_volume,
          m.avg_dollar_volume,
          m.daily_range_percent,
          m.current_volume,
          m.percentile_volume,
          m.price_change_5d,
          m.pct_from_52w_high,
          m.tv_osc_buy,
          m.tv_osc_sell,
          m.tv_osc_neutral,
          m.tv_ma_buy,
          m.tv_ma_sell,
          m.tv_ma_neutral,
          m.tv_overall_signal,
          m.tv_overall_score,
          m.rsi_14,
          m.stoch_k,
          m.stoch_d,
          m.cci_20,
          m.adx_14,
          m.plus_di,
          m.minus_di,
          m.ao_value,
          m.momentum_10,
          m.macd,
          m.macd_signal,
          m.stoch_rsi_k,
          m.stoch_rsi_d,
          m.williams_r,
          m.bull_power,
          m.bear_power,
          m.ultimate_osc,
          m.ema_10,
          m.ema_20,
          m.ema_30,
          m.ema_50,
          COALESCE(ec.above_ema50, FALSE) AS above_ema50,
          ec.ema50_cross_age_sessions,
          m.ema_100,
          m.ema_200,
          m.sma_10,
          m.sma_20,
          m.sma_30,
          m.sma_50,
          m.sma_100,
          m.sma_200,
          m.ichimoku_base,
          m.vwma_20,
          d.daq_score,
          d.daq_grade,
          d.mtf_score,
          d.volume_score AS daq_volume_score,
          d.smc_score AS daq_smc_score,
          d.quality_score AS daq_quality_score,
          d.catalyst_score AS daq_catalyst_score,
          d.news_score AS daq_news_score,
          d.regime_score AS daq_regime_score,
          d.sector_score AS daq_sector_score,
          d.earnings_within_7d,
          d.high_short_interest,
          d.sector_underperforming,
          i.earnings_date,
          CASE
            WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
            THEN (i.earnings_date - CURRENT_DATE)
            ELSE NULL
          END AS days_to_earnings,
          bt_summary.trade_count,
          bt_summary.open_trade_count,
          bt_summary.latest_open_time,
          bt_last.last_trade_status,
          bt_last.last_trade_open_time,
          bt_last.last_trade_close_time,
          bt_last.last_trade_profit,
          bt_last.last_trade_profit_pct,
          bt_last.last_trade_side,
          bt_closed.last_closed_time,
          bt_closed.last_closed_profit,
          bt_closed.last_closed_profit_pct,
          bt_closed.last_closed_side,
          ROUND(e.pf, 2) AS scanner_pf,
          e.closed_n AS scanner_closed_n,
          m.current_price AS prior_session_close,
          pm.signal_type AS pm_signal_type,
          pm.direction AS pm_direction,
          pm.strength AS pm_strength,
          pm.confidence AS pm_confidence,
          pm.gap_percent AS pm_gap_percent,
          pm.gap_type AS pm_gap_type,
          pm.current_price AS pm_current_price,
          pm.previous_close AS pm_previous_close,
          pm.news_count AS pm_news_count,
          pm.news_sentiment_score AS pm_news_sentiment_score,
          pm.news_sentiment_level AS pm_news_sentiment_level,
          pm.suggested_entry AS pm_suggested_entry,
          pm.suggested_stop AS pm_suggested_stop,
          pm.suggested_target AS pm_suggested_target,
          pm.risk_reward AS pm_risk_reward,
          pm.generated_at AS pm_generated_at,
          EXTRACT(EPOCH FROM (NOW() - pm.generated_at)) / 3600.0 AS pm_age_hours,
          (
            (pm.generated_at AT TIME ZONE 'America/New_York')::date =
            (NOW() AT TIME ZONE 'America/New_York')::date
          ) AS pm_is_current_session
        FROM latest_batch s
        LEFT JOIN stock_instruments i ON s.ticker = i.ticker
        LEFT JOIN stock_screening_metrics m ON s.ticker = m.ticker
          AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        LEFT JOIN ema_cross ec ON s.ticker = ec.ticker
        LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
          -- Age guard: the pool spans 2 signal dates, and daq_news_score feeds
          -- the candidate score (0.08 weight). Without this, stale DAQ rows
          -- (incl. the pre-Jun-12-2026 scorer's garbage rows) leak into ranking.
          AND d.analysis_timestamp >= CURRENT_DATE - INTERVAL '1 day'
        LEFT JOIN scanner_edge e ON s.scanner_name = e.scanner_name
        LEFT JOIN LATERAL (
          SELECT
            COUNT(*)::int AS trade_count,
            COUNT(*) FILTER (WHERE status = 'open')::int AS open_trade_count,
            MAX(open_time) FILTER (WHERE status = 'open') AS latest_open_time
          FROM broker_trades bt
          WHERE bt.ticker = s.ticker
             OR split_part(bt.ticker, '.', 1) = s.ticker
        ) bt_summary ON TRUE
        LEFT JOIN LATERAL (
          SELECT
            bt.status AS last_trade_status,
            bt.open_time AS last_trade_open_time,
            bt.close_time AS last_trade_close_time,
            bt.profit AS last_trade_profit,
            bt.profit_pct AS last_trade_profit_pct,
            bt.side AS last_trade_side
          FROM broker_trades bt
          WHERE bt.ticker = s.ticker
             OR split_part(bt.ticker, '.', 1) = s.ticker
          ORDER BY bt.open_time DESC NULLS LAST
          LIMIT 1
        ) bt_last ON TRUE
        LEFT JOIN LATERAL (
          SELECT
            bt.close_time AS last_closed_time,
            bt.profit AS last_closed_profit,
            bt.profit_pct AS last_closed_profit_pct,
            bt.side AS last_closed_side
          FROM broker_trades bt
          WHERE (bt.ticker = s.ticker OR split_part(bt.ticker, '.', 1) = s.ticker)
            AND bt.status = 'closed'
          ORDER BY bt.close_time DESC NULLS LAST
          LIMIT 1
        ) bt_closed ON TRUE
        LEFT JOIN LATERAL (
          SELECT
            p.signal_type,
            p.direction,
            p.strength,
            p.confidence,
            p.gap_percent,
            p.gap_type,
            p.current_price,
            p.previous_close,
            p.news_count,
            p.news_sentiment_score,
            p.news_sentiment_level,
            p.suggested_entry,
            p.suggested_stop,
            p.suggested_target,
            p.risk_reward,
            p.generated_at
          FROM stock_premarket_signals p
          WHERE p.symbol = s.ticker
          ORDER BY p.generated_at DESC
          LIMIT 1
        ) pm ON TRUE
      ),
      scored AS (
        SELECT *,
          round(${scoreExpression}, 1) AS candidate_score,
          (
            COALESCE(scanner_pf, 999) < $2
            AND COALESCE(scanner_closed_n, 0) >= $3
          ) AS edge_floor_blocked,
          CASE
            WHEN pm_generated_at IS NULL THEN 'No PM data'
            WHEN NOT pm_is_current_session THEN 'Stale PM'
            WHEN pm_direction <> 'BUY' THEN 'PM against'
            WHEN pm_gap_percent >= 1 AND COALESCE(pm_confidence, 0) >= 0.65 THEN 'PM confirmed'
            WHEN pm_gap_percent > 0 THEN 'PM watch'
            WHEN pm_gap_percent < -1 THEN 'PM fading'
            ELSE 'PM neutral'
          END AS pm_status,
          CASE
            WHEN pm_generated_at IS NULL THEN 'Wait'
            WHEN NOT pm_is_current_session THEN 'Refresh'
            WHEN pm_direction <> 'BUY' THEN 'Avoid'
            WHEN high_short_interest OR sector_underperforming THEN 'Small only'
            WHEN pm_suggested_entry IS NOT NULL AND COALESCE(pm_confidence, 0) >= 0.65 THEN 'Use PM levels'
            WHEN pm_gap_percent BETWEEN 1 AND 8 AND COALESCE(relative_volume, 0) >= 1.2 THEN 'Pullback'
            WHEN pm_gap_percent > 8 OR rsi_14 > 80 THEN 'Wait pullback'
            ELSE 'Watch'
          END AS order_bias
        FROM enriched
      ),
      eligible AS (
        -- Closed-only edge floor: enforce when possible, but never blank the
        -- whole current list if today's batch only contains below-floor scanners.
        SELECT * FROM scored
        WHERE (
          NOT edge_floor_blocked
          OR NOT EXISTS (SELECT 1 FROM scored WHERE NOT edge_floor_blocked)
        )${
          mode === "daytrades"
            ? `
          -- EMA50 freshness gate (hard filter, fail-closed): keep only names that
          -- are above EMA50 AND crossed up within $7 trading sessions. NULL age
          -- (no in-history cross / no metrics / below EMA50) is excluded.
          AND above_ema50
          AND ema50_cross_age_sessions BETWEEN 0 AND $7`
            : ""
        }
      ),
      capped AS (
        SELECT *,
          row_number() OVER (
            PARTITION BY scanner_name
            ORDER BY candidate_score DESC, rs_percentile DESC NULLS LAST, ticker
          ) AS scanner_rank
        FROM eligible
      ),
      included AS (
        SELECT *,
          row_number() OVER (ORDER BY candidate_score DESC, rs_percentile DESC NULLS LAST, ticker) AS rank
        FROM capped
        WHERE scanner_rank <= $4
        ORDER BY candidate_score DESC, rs_percentile DESC NULLS LAST, ticker
        LIMIT $5
      )
      -- Capped-out audit rows: names dropped by the per-scanner cap that would
      -- otherwise have made the list (score at/above the included cutoff, or the
      -- list isn't full). Returned flagged so callers can record the suppression;
      -- they are NOT part of the tradable pool.
      SELECT *, FALSE AS capped_by_scanner FROM included
      UNION ALL
      SELECT *, NULL::bigint AS rank, TRUE AS capped_by_scanner
      FROM capped
      WHERE scanner_rank > $4
        AND (
          (SELECT count(*) FROM included) < $5
          OR candidate_score >= (SELECT min(candidate_score) FROM included)
        )
      ORDER BY capped_by_scanner, candidate_score DESC, rs_percentile DESC NULLS LAST, ticker
    `;
    const queryParams: Array<number> = [
      EDGE_WINDOW_DAYS,
      EDGE_PF_FLOOR,
      EDGE_MIN_CLOSED,
      maxPerScanner,
      queryLimit,
      EDGE_NO_LOSS_PF_CAP,
    ];
    // $7 — only referenced by the daytrades EMA50 freshness gate.
    if (mode === "daytrades") queryParams.push(maxEmaCrossAgeSessions);
    const result = await client.query(query, queryParams);
    const includedRows = result.rows.filter((row) => !row.capped_by_scanner);
    const cappedRows = result.rows.filter((row) => row.capped_by_scanner);
    let daytradeTradableCount: number | null = null;
    let daytradeLiveCandleCount: number | null = null;
    const rows =
      mode === "daytrades"
        ? await (async () => {
            const tickers = includedRows.map((row) => row.ticker);
            const [quotes, candleStats] = await Promise.all([
              fetchBrokerQuotes(tickers),
              fetchIntradayCandleStats(client, tickers),
            ]);
            const enrichedRows = includedRows.map((row) => {
              const quote = quotes[row.ticker] ?? {};
              const spreadPct = quote.broker_spread_pct ?? null;
              const quoteTime = quote.broker_quote_time ? new Date(quote.broker_quote_time) : null;
              const quoteAgeMinutes =
                quoteTime && !Number.isNaN(quoteTime.getTime())
                  ? Math.round((Date.now() - quoteTime.getTime()) / 60000)
                  : null;
              // Live intraday RVOL pace + session VWAP for the sparse subset that
              // streams 1h candles today (the broker quote carries no volume).
              const candle = candleStats[String(row.ticker)] ?? null;
              const liveIntradayRvol = candle?.intraday_rvol_pace ?? null;
              const sessionVwap = candle?.session_vwap ?? null;
              const candleLastClose = candle?.last_close ?? null;
              const vwapPosition =
                sessionVwap != null && candleLastClose != null
                  ? candleLastClose >= sessionVwap ? "above" : "below"
                  : null;
              // intraday_relative_volume now carries the live candle pace (time-
              // of-day-matched), not a broker-volume ratio.
              const intradayRelativeVolume = liveIntradayRvol;
              const quotePrice = Number(quote.broker_last ?? quote.broker_ask ?? 0);
              const setupEntry = Number(row.entry_price ?? 0);
              const openConfirmed =
                row.pm_status === "No PM data" &&
                quoteAgeMinutes != null &&
                quoteAgeMinutes <= 3 &&
                quotePrice > 0 &&
                setupEntry > 0 &&
                quotePrice >= setupEntry * 0.995;
              // Live-confirmation bonus (0..8, ADDITIVE only so it never becomes
              // an apples-to-oranges core term): a name we can SEE moving on
              // volume and holding >= VWAP earns a boost; a covered-but-quiet name
              // earns 0 (no penalty -> symmetric with uncovered names). Thresholds
              // unvalidated.
              const liveConfirmationBonus =
                liveIntradayRvol == null
                  ? 0
                  : (() => {
                      const base =
                        liveIntradayRvol >= 3 ? 8 :
                        liveIntradayRvol >= 2 ? 6 :
                        liveIntradayRvol >= 1.5 ? 4 :
                        liveIntradayRvol >= 1 ? 2 :
                        0;
                      return vwapPosition === "below" ? Math.floor(base / 2) : base;
                    })();
              const spreadScore =
                spreadPct == null ? 0 :
                spreadPct <= 0.25 ? 4 :
                spreadPct <= 0.4 ? 2 :
                0;
              const quoteFreshnessScore =
                quoteAgeMinutes == null ? 0 :
                quoteAgeMinutes <= 1 ? 2 :
                quoteAgeMinutes <= 3 ? 1 :
                0;
              const pmSuggestedEntry = Number(row.pm_suggested_entry ?? 0);
              const ask = Number(quote.broker_ask ?? 0);
              const entryDisciplineScore =
                pmSuggestedEntry > 0 && ask > 0 && ask <= pmSuggestedEntry ? 2 :
                pmSuggestedEntry > 0 && ask > 0 && ask <= pmSuggestedEntry * 1.01 ? 1 :
                0;
              const quantityUnderCap = ask > 0 ? Math.floor(500 / ask) : 0;
              const sizeFitScore = quantityUnderCap >= 2 ? 2 : quantityUnderCap >= 1 ? 1 : 0;
              // #6 room-to-target vs spread: a wide spread relative to the
              // expected move makes a name un-day-tradable regardless of setup.
              // Room = premarket target distance when available, else daily ATR%.
              const pmTarget = Number(row.pm_suggested_target ?? 0);
              const refEntry = pmSuggestedEntry > 0 ? pmSuggestedEntry : ask > 0 ? ask : setupEntry;
              const atrPct = Number(row.atr_percent ?? 0);
              const roomPct =
                pmTarget > 0 && refEntry > 0 && pmTarget > refEntry
                  ? ((pmTarget - refEntry) / refEntry) * 100
                  : atrPct > 0
                    ? atrPct
                    : null;
              const spreadToRoom =
                spreadPct != null && roomPct != null && roomPct > 0 ? spreadPct / roomPct : null;
              const roomScore =
                spreadToRoom == null ? 0 : spreadToRoom <= 0.05 ? 2 : spreadToRoom <= 0.1 ? 1 : 0;
              const executionQualityScore =
                spreadScore + quoteFreshnessScore + entryDisciplineScore + sizeFitScore + roomScore;
              let orderBias =
                spreadToRoom != null && spreadToRoom > 0.25
                  ? "Spread eats range"
                  : spreadPct != null && spreadPct > 1
                    ? "Avoid spread"
                    : spreadPct != null && spreadPct > 0.4 && row.order_bias !== "Avoid"
                      ? "Small only"
                      : row.order_bias;
              if (openConfirmed && orderBias === "Wait") {
                orderBias = row.high_short_interest || row.sector_underperforming ? "Small only" : "Watch";
              }
              if (orderBias === "Pullback" && intradayRelativeVolume != null && intradayRelativeVolume < 1.2) {
                orderBias = "Watch";
              }
              // Tradability gate (the de-facto top-of-list ordering): a row is
              // tradable only if premarket confirms a current-session BUY and the
              // order bias is actionable. Surfaced per-row so the UI can show why.
              const finalPmStatus = openConfirmed ? "Open confirmed" : row.pm_status;
              // Swing-horizon gate: only EXPLICIT premarket bearishness hard-blocks.
              // "No PM data" / "Stale PM" no longer reject -- only ~14% of executed
              // trades ever had a current-session premarket BUY, so requiring a 4am
              // premarket catalog hit blocked the majority of legitimate multi-day
              // entries. Their order_bias ("Wait"/"Refresh") is likewise dropped
              // from the bias hard-reject below so the softening is coherent.
              const hardPmReject = ["PM against", "PM fading"].includes(
                String(finalPmStatus ?? "")
              );
              const hardBiasReject = [
                "Avoid",
                "Avoid spread",
                "Spread eats range",
                "Wait pullback",
              ].includes(orderBias);
              const hardSignalReject = String(row.signal_type ?? "") !== "BUY";
              const tradable = !(hardSignalReject || hardPmReject || hardBiasReject);
              // #1 SECOND tradable path: a name confirmed in play by LIVE volume
              // (intraday RVOL pace >= 2 and holding >= VWAP) is tradable even
              // without a premarket row -- never overriding an explicit bearish PM
              // read, and only when the spread is not a hard reject. This breaks
              // the feedback loop where the 9:00 AM premarket-pricing job enriches
              // only the tickers this route already ranks in the Top 50.
              const pmExplicitlyBearish = ["PM against", "PM fading"].includes(String(finalPmStatus ?? ""));
              const hardSpreadReject = orderBias === "Avoid spread" || orderBias === "Spread eats range";
              const liveInPlay =
                liveIntradayRvol != null && liveIntradayRvol >= 2 && vwapPosition !== "below" && !hardSpreadReject;
              const rescued = !tradable && liveInPlay && !pmExplicitlyBearish && !hardSignalReject;
              const finalTradable = tradable || rescued;
              return {
                ...row,
                ...quote,
                pm_status: finalPmStatus,
                broker_quote_age_minutes: quoteAgeMinutes,
                intraday_relative_volume: intradayRelativeVolume,
                live_intraday_rvol: liveIntradayRvol,
                live_as_of: candle?.live_as_of ?? null,
                session_vwap: sessionVwap,
                vwap_position: vwapPosition,
                live_confirmation_bonus: liveConfirmationBonus,
                relative_volume_source: "prior_session_daily",
                intraday_relative_volume_source:
                  intradayRelativeVolume == null ? null : "intraday_5m_cum_vol_vs_tod_adjusted_daily_baseline",
                execution_quality_score: executionQualityScore,
                execution_quality_parts: {
                  spread: spreadScore,
                  quote_age: quoteFreshnessScore,
                  entry: entryDisciplineScore,
                  size: sizeFitScore,
                  room: roomScore,
                },
                spread_to_room: spreadToRoom == null ? null : Number(spreadToRoom.toFixed(3)),
                // candidate_score is MERIT only: the SQL core (leadership + setup).
                // The live session-VWAP / intraday-RVOL bonus is NO LONGER added
                // here -- "holding above VWAP right now" has no meaning for a
                // position held 1-3 days overnight, so it must not move the swing
                // rank. It stays computed as a displayed metric (live_confirmation_bonus)
                // and still feeds the live-in-play tradability rescue path below.
                // Execution quality (spread/fill mechanics) likewise stays out of
                // the rank -- it answers "can I get a good fill," not "is this a
                // good swing candidate."
                // NOTE: the inner Number() is load-bearing — pg returns numeric
                // columns as strings, and .toFixed() on a string throws.
                candidate_score: Number(Number(row.candidate_score ?? 0).toFixed(1)),
                order_bias: orderBias,
                tradable: finalTradable,
                gate_reason: finalTradable
                  ? rescued ? "In play (live)" : "Tradable"
                  : hardSignalReject ? "Long-only auto trader"
                    : hardPmReject ? String(finalPmStatus)
                    : orderBias,
              };
            });
            daytradeTradableCount = enrichedRows.filter((row) => row.tradable).length;
            daytradeLiveCandleCount = enrichedRows.filter((row) => row.live_intraday_rvol != null).length;
            return enrichedRows
              .sort((a, b) => {
                const tradableDelta = (b.tradable ? 1 : 0) - (a.tradable ? 1 : 0);
                if (tradableDelta !== 0) return tradableDelta;
                const scoreDelta = Number(b.candidate_score ?? 0) - Number(a.candidate_score ?? 0);
                if (scoreDelta !== 0) return scoreDelta;
                return Number(b.rs_percentile ?? 0) - Number(a.rs_percentile ?? 0);
              })
              .slice(0, limit)
              .map((row, index) => ({ ...row, rank: index + 1 }));
          })()
        : includedRows;

    // Edge-map router (c) — SHADOW MODE: log the intended routing decision for
    // every returned candidate. Changes NOTHING about what trades or what this
    // endpoint returns; `actual_action` is always "unchanged". Fail-open on any
    // missing/insufficient cell or lookup error (never drop for real).
    if (CELL_ROUTER_SHADOW) {
      try {
        const scannerNames = [...new Set(rows.map((r) => String(r.scanner_name)))];
        const edgeMap = await fetchCellEdgeMap(client, scannerNames);
        for (const row of rows) {
          const trend = row.trend_state == null ? null : String(row.trend_state);
          const vol = row.vol_regime == null ? null : String(row.vol_regime);
          const cellKey = `${String(row.scanner_name)}|${trend ?? "?"}|${vol ?? "?"}`;
          const edge = trend && vol ? edgeMap.get(cellKey) : undefined;
          const verdict = edge?.verdict ?? null;
          // block->would-drop, monitor->would-hold, trade/missing/insufficient->allowed (fail-open)
          const intendedAction =
            verdict === "block"
              ? "would-drop"
              : verdict === "monitor"
                ? "would-hold"
                : "allowed";
          console.log(
            "[CELL-ROUTER-SHADOW]",
            JSON.stringify({
              ticker: row.ticker,
              scanner_name: row.scanner_name,
              cell_key: cellKey,
              verdict: edge ? (verdict ?? "insufficient") : "missing",
              pf: edge?.pf ?? null,
              n: edge?.n ?? null,
              intended_action: intendedAction,
              actual_action: "unchanged",
            })
          );
        }
      } catch (err) {
        console.warn("[CELL-ROUTER-SHADOW] shadow logging failed (non-fatal):", err);
      }
    }

    return NextResponse.json(
      {
        rows,
        // Names suppressed by the per-scanner cap that would otherwise have made
        // the list. Raw SQL-scored rows (no broker-quote enrichment) — for audit
        // logging by the auto-trader, not for trading.
        capped_rows: cappedRows,
        meta: {
          scoring_mode: mode,
          batch_date: includedRows[0]?.signal_timestamp ?? null,
          edge_window_days: EDGE_WINDOW_DAYS,
          edge_pf_floor: EDGE_PF_FLOOR,
          edge_min_closed: EDGE_MIN_CLOSED,
          edge_no_loss_pf_cap: EDGE_NO_LOSS_PF_CAP,
          max_per_scanner: maxPerScanner,
          max_ema_cross_age_sessions: mode === "daytrades" ? maxEmaCrossAgeSessions : null,
          daytrade_signal_date_count: mode === "daytrades" ? DAYTRADE_SIGNAL_DATE_COUNT : null,
          daytrade_pool_size: mode === "daytrades" ? includedRows.length : null,
          daytrade_capped_count: mode === "daytrades" ? cappedRows.length : null,
          daytrade_tradable_count: daytradeTradableCount,
          daytrade_live_candle_count: daytradeLiveCandleCount,
          daytrade_rvol_source:
            mode === "daytrades"
              ? "intraday_5m_state_when_present_else_prior_session_daily"
              : null,
          daytrade_execution_quality:
            mode === "daytrades"
              ? "0-12 live score: spread 4, quote age 2, entry discipline 2, size fit 2, room-vs-spread 2"
              : null,
          daytrade_live_confirmation_bonus:
            mode === "daytrades"
              ? "0-8 display metric (NOT added to candidate_score for swing horizon): live RVOL pace + above/below session VWAP (covered subset only)"
              : null,
        },
      },
      { headers: NO_STORE_HEADERS }
    );
  } catch (error) {
    console.error("top-candidates query failed", error);
    return NextResponse.json(
      { error: "Failed to load top candidates" },
      { status: 500, headers: NO_STORE_HEADERS }
    );
  } finally {
    client.release();
  }
}
