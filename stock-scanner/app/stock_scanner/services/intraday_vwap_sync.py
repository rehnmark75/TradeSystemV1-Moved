"""
Intraday VWAP / RVOL sync worker.

The day-trade picker (/api/signals/top?mode=daytrades) and the auto trader fire
in the 9:45-10:15 ET window, but the only intraday candle data we store is 1h
bars synced nightly -- so during that window the picker is blind to the live
tape. This worker closes that gap WITHOUT syncing sub-hourly data for the whole
universe.

Smart-scoping (see CLAUDE design discussion):
  * Narrow on tickers : only the ~50-name day-trade candidate pool (fetched from
    the trading-ui route, exactly the names the auto trader can trade).
  * Narrow on time    : only runs ~9:25-10:30 ET (open-window + margin).
  * Batch the pull    : ONE yf.download() for the whole pool, not 50 calls.
  * Store derived only: one tiny row per ticker per day in stock_intraday_state
    (session VWAP, cumulative volume, intraday RVOL pace), never the raw 5m bars.

RVOL baseline is computed WITHOUT storing any intraday history: today's
cumulative volume / (daily avg_volume_20 * typical cumulative-volume fraction by
time-of-day). yfinance intraday data is ~15 min delayed, so the pace is measured
against the timestamp of the LAST bar actually received (delay-robust), not wall
clock.
"""

import asyncio
import logging
import os
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import pytz
import yfinance as yf

from stock_scanner.core.detection.market_hours import is_trading_day

logger = logging.getLogger("intraday_vwap_sync")

# Approximate cumulative fraction of a regular session's volume completed by N
# minutes after the 9:30 ET open. US equity intraday volume is U-shaped and
# front-loaded; these anchors are interpolated linearly. Tunable -- the goal is a
# delay-robust "is this name trading heavier than a normal morning" read, not a
# precise volume forecast. (key = minutes after open, value = cumulative fraction)
_VOLUME_FRACTION_CURVE = {
    0: 0.000,
    5: 0.030,
    10: 0.050,
    15: 0.070,
    20: 0.088,
    30: 0.118,
    45: 0.155,
    60: 0.190,
    90: 0.250,
    120: 0.300,
    180: 0.400,
    240: 0.500,
    300: 0.620,
    330: 0.700,
    360: 0.820,
    390: 1.000,
}


def cumulative_volume_fraction(minutes_after_open: float) -> float:
    """Interpolate the typical cumulative-volume fraction at N minutes post-open."""
    if minutes_after_open <= 0:
        return _VOLUME_FRACTION_CURVE[0]
    anchors = sorted(_VOLUME_FRACTION_CURVE.items())
    if minutes_after_open >= anchors[-1][0]:
        return anchors[-1][1]
    prev_m, prev_f = anchors[0]
    for m, f in anchors[1:]:
        if minutes_after_open <= m:
            span = m - prev_m
            if span <= 0:
                return f
            ratio = (minutes_after_open - prev_m) / span
            return prev_f + ratio * (f - prev_f)
        prev_m, prev_f = m, f
    return anchors[-1][1]


class IntradayVwapSync:
    ET = pytz.timezone("America/New_York")
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    SETTING_DEFS = {
        "INTRADAY_VWAP_ENABLED": ("enabled", "bool", "true"),
        "INTRADAY_VWAP_POOL_LIMIT": ("pool_limit", "int", "50"),
        "INTRADAY_VWAP_START_BEFORE_OPEN_MIN": ("start_before_open_min", "int", "5"),
        "INTRADAY_VWAP_STOP_AFTER_OPEN_MIN": ("stop_after_open_min", "int", "60"),
        "INTRADAY_VWAP_INTERVAL_SECONDS": ("interval_seconds", "int", "150"),
    }

    def __init__(self, db_manager):
        self.db = db_manager
        self.trading_ui_url = os.getenv("TRADING_UI_URL", "http://trading-ui:3000/trading").rstrip("/")
        for env_key, (attr, value_type, default) in self.SETTING_DEFS.items():
            setattr(self, attr, self._cast_setting(os.getenv(env_key, default), value_type))

    @staticmethod
    def _cast_setting(raw: Any, value_type: str) -> Any:
        if value_type == "bool":
            return str(raw).lower() in ("1", "true", "yes", "on")
        if value_type == "int":
            return int(float(raw))
        if value_type == "float":
            return float(raw)
        return str(raw)

    async def ensure_schema(self):
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS stock_intraday_state (
                ticker VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                as_of TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session_vwap DOUBLE PRECISION,
                last_price DOUBLE PRECISION,
                cum_volume BIGINT,
                bars_today INTEGER,
                intraday_rvol_pace DOUBLE PRECISION,
                avg_volume_20 BIGINT,
                source VARCHAR(20) DEFAULT 'yfinance_5m',
                PRIMARY KEY (ticker, trade_date)
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_intraday_state_trade_date
            ON stock_intraday_state(trade_date)
        """)
        # Raw 5m bars the worker already downloads each cycle. We otherwise discard
        # them after deriving the scalars above; persisting them is near-zero cost
        # and the substrate for any future intraday-derived entry logic (intraday
        # ATR, VWAP-reclaim, pullback-to-structure). The PK makes re-fetching the
        # same bars every ~2.5 min idempotent -- the still-forming last bar is
        # overwritten, not duplicated.
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS stock_intraday_candles (
                ticker     VARCHAR(20)      NOT NULL,
                timeframe  VARCHAR(10)      NOT NULL DEFAULT '5m',
                bar_time   TIMESTAMPTZ      NOT NULL,
                open       DOUBLE PRECISION,
                high       DOUBLE PRECISION,
                low        DOUBLE PRECISION,
                close      DOUBLE PRECISION,
                volume     BIGINT,
                source     VARCHAR(20)      DEFAULT 'yfinance_5m',
                as_of      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
                PRIMARY KEY (ticker, timeframe, bar_time)
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_intraday_candles_ticker_time
            ON stock_intraday_candles(ticker, bar_time DESC)
        """)

    # ------------------------------------------------------------------ window
    def _minutes_after_open(self, now_et: datetime) -> Optional[int]:
        if not is_trading_day(now_et):
            return None
        market_open = self.ET.localize(datetime.combine(now_et.date(), self.MARKET_OPEN))
        return int((now_et - market_open).total_seconds() // 60)

    # ------------------------------------------------------------------- runner
    async def run_once(self) -> Dict[str, Any]:
        await self.ensure_schema()
        now_et = datetime.now(self.ET)
        result = {"enabled": self.enabled, "stage": "idle", "fetched": 0, "written": 0, "missing": 0, "bars_written": 0}

        if not self.enabled:
            result["stage"] = "disabled"
            return result
        if not is_trading_day(now_et):
            result["stage"] = "market_closed_holiday"
            return result

        mao = self._minutes_after_open(now_et)
        if mao is None or mao < -self.start_before_open_min:
            result["stage"] = "before_window"
            return result
        if mao > self.stop_after_open_min:
            result["stage"] = "after_window"
            return result

        tickers = await self._fetch_candidate_pool()
        if not tickers:
            result["stage"] = "no_candidates"
            return result

        avg_vol = await self._avg_volume_map(tickers)
        frame = await self._download_5m_batch(tickers)
        if frame is None:
            result["stage"] = "download_failed"
            return result

        trade_date = now_et.date()
        written = 0
        missing = 0
        bars_written = 0
        for ticker in tickers:
            sess = self._session_frame(ticker, frame, now_et)
            if sess is None:
                missing += 1
                continue
            state = self._compute_state(sess, now_et, avg_vol.get(ticker))
            if state is None:
                missing += 1
                continue
            await self._upsert_state(ticker, trade_date, state)
            written += 1
            # Persisting raw bars is a best-effort side effect: a failure here must
            # not break the state upsert the picker/auto-trader actually read.
            try:
                bars_written += await self._persist_bars(ticker, sess)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[IntradayVWAP] persist bars failed for %s: %s", ticker, exc)

        result.update(
            stage="updated", fetched=len(tickers), written=written,
            missing=missing, bars_written=bars_written,
        )
        return result

    async def run_loop(self):
        await self.ensure_schema()
        logger.info(
            "Intraday VWAP sync loop started enabled=%s pool=%s window=[-%sm,+%sm] interval=%ss url=%s",
            self.enabled, self.pool_limit, self.start_before_open_min,
            self.stop_after_open_min, self.interval_seconds, self.trading_ui_url,
        )
        while True:
            try:
                result = await self.run_once()
                logger.info("[IntradayVWAP] %s", result)
            except Exception as exc:
                logger.exception("[IntradayVWAP] loop error: %s", exc)
                result = {"stage": "error"}
            # Cheap idle outside the window; tight cadence while active.
            idle = result.get("stage") in ("before_window", "after_window", "market_closed_holiday", "disabled")
            await asyncio.sleep(max(self.interval_seconds, 300) if idle else max(30, self.interval_seconds))

    # --------------------------------------------------------------- data steps
    async def _fetch_candidate_pool(self) -> List[str]:
        url = f"{self.trading_ui_url}/api/signals/top?limit={self.pool_limit}&mode=daytrades"
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    payload = await response.json()
                    if response.status >= 400:
                        logger.warning("Top day trades API %s: %s", response.status, payload)
                        return []
                    seen, out = set(), []
                    for row in (payload.get("rows") or []):
                        ticker = str(row.get("ticker") or "").strip()
                        if ticker and ticker not in seen:
                            seen.add(ticker)
                            out.append(ticker)
                    return out[: self.pool_limit]
        except Exception as exc:
            logger.error("Failed to fetch candidate pool: %s", exc)
            return []

    async def _avg_volume_map(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        rows = await self.db.fetch("""
            SELECT ticker, avg_volume_20
            FROM stock_screening_metrics
            WHERE calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
              AND ticker = ANY($1)
        """, tickers)
        return {r["ticker"]: (float(r["avg_volume_20"]) if r["avg_volume_20"] is not None else None) for r in rows}

    async def _download_5m_batch(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """One batched yfinance pull for the whole pool (blocking -> thread)."""
        def _download() -> Optional[pd.DataFrame]:
            try:
                # period="2d" (not 1d): just after the open, intraday period="1d"
                # can still return only the PRIOR session until enough of today's
                # bars accrue -- which would silently no-op the worker in exactly
                # the window it exists for. 2d always spans today; _compute_state's
                # today-date filter isolates the right session.
                df = yf.download(
                    tickers=tickers,
                    interval="5m",
                    period="2d",
                    group_by="ticker",
                    threads=True,
                    prepost=False,
                    progress=False,
                    auto_adjust=False,
                )
                if df is None or df.empty:
                    return None
                return df
            except Exception as exc:  # noqa: BLE001
                logger.error("yf.download failed: %s", exc)
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)

    def _ticker_frame(self, ticker: str, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract a single ticker's OHLCV from the (possibly multi-index) batch."""
        try:
            if isinstance(frame.columns, pd.MultiIndex):
                if ticker not in frame.columns.get_level_values(0):
                    return None
                tdf = frame[ticker]
            else:
                tdf = frame  # single-ticker download -> flat columns
            tdf = tdf.dropna(how="all")
            return tdf if not tdf.empty else None
        except Exception:
            return None

    def _session_frame(
        self, ticker: str, frame: pd.DataFrame, now_et: datetime
    ) -> Optional[pd.DataFrame]:
        """Today's regular-session 5m bars for one ticker, ET-indexed (or None)."""
        tdf = self._ticker_frame(ticker, frame)
        if tdf is None:
            return None

        # Normalise index to ET and keep only TODAY's regular session.
        idx = tdf.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        et_idx = idx.tz_convert(self.ET)
        tdf = tdf.copy()
        tdf.index = et_idx

        today = now_et.date()
        session_mask = (
            (tdf.index.date == today)
            & (tdf.index.time >= self.MARKET_OPEN)
            & (tdf.index.time < self.MARKET_CLOSE)
        )
        sess = tdf[session_mask]
        return sess if not sess.empty else None

    def _compute_state(
        self, sess: pd.DataFrame, now_et: datetime, avg_volume_20: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        today = now_et.date()
        high = sess["High"].astype(float)
        low = sess["Low"].astype(float)
        close = sess["Close"].astype(float)
        volume = sess["Volume"].astype(float).fillna(0)

        cum_volume = float(volume.sum())
        typical = (high + low + close) / 3.0
        vwap = float((typical * volume).sum() / cum_volume) if cum_volume > 0 else None
        last_price = float(close.iloc[-1])
        bars_today = int(len(sess))

        # Delay-robust RVOL: measure against the LAST bar actually received, not
        # wall clock (yfinance intraday lags ~15 min). Each 5m bar's start is its
        # index; volume through it accrues to bar_end = start + 5 min.
        open_dt = self.ET.localize(datetime.combine(today, self.MARKET_OPEN))
        last_bar_end = sess.index[-1].to_pydatetime() + timedelta(minutes=5)
        minutes_elapsed = max(0.0, (last_bar_end - open_dt).total_seconds() / 60.0)
        frac = cumulative_volume_fraction(minutes_elapsed)

        intraday_rvol_pace = None
        if avg_volume_20 and avg_volume_20 > 0 and frac > 0:
            intraday_rvol_pace = round(cum_volume / (avg_volume_20 * frac), 3)

        return {
            "session_vwap": round(vwap, 4) if vwap is not None else None,
            "last_price": round(last_price, 4),
            "cum_volume": int(cum_volume),
            "bars_today": bars_today,
            "intraday_rvol_pace": intraday_rvol_pace,
            "avg_volume_20": int(avg_volume_20) if avg_volume_20 else None,
        }

    async def _upsert_state(self, ticker: str, trade_date, state: Dict[str, Any]):
        await self.db.execute("""
            INSERT INTO stock_intraday_state (
                ticker, trade_date, as_of, session_vwap, last_price,
                cum_volume, bars_today, intraday_rvol_pace, avg_volume_20, source
            ) VALUES ($1, $2, NOW(), $3, $4, $5, $6, $7, $8, 'yfinance_5m')
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
                as_of = NOW(),
                session_vwap = EXCLUDED.session_vwap,
                last_price = EXCLUDED.last_price,
                cum_volume = EXCLUDED.cum_volume,
                bars_today = EXCLUDED.bars_today,
                intraday_rvol_pace = EXCLUDED.intraday_rvol_pace,
                avg_volume_20 = EXCLUDED.avg_volume_20,
                source = EXCLUDED.source
        """,
            ticker, trade_date, state["session_vwap"], state["last_price"],
            state["cum_volume"], state["bars_today"], state["intraday_rvol_pace"],
            state["avg_volume_20"],
        )

    @staticmethod
    def _bar_float(value: Any) -> Optional[float]:
        return float(value) if value is not None and not pd.isna(value) else None

    async def _persist_bars(self, ticker: str, sess: pd.DataFrame) -> int:
        """Bulk-upsert today's raw 5m bars; idempotent on (ticker, '5m', bar_time).

        Re-run every cycle, so the still-forming last bar is overwritten in place
        rather than duplicated. Returns the number of bars written.
        """
        rows: List[tuple] = []
        for idx, bar in sess.iterrows():
            close = bar.get("Close")
            if close is None or pd.isna(close):
                continue  # skip a not-yet-printed / all-NaN bar
            volume = bar.get("Volume")
            volume = int(volume) if volume is not None and not pd.isna(volume) else 0
            rows.append((
                ticker,
                idx.to_pydatetime(),  # tz-aware (ET) bar start -> timestamptz
                self._bar_float(bar.get("Open")),
                self._bar_float(bar.get("High")),
                self._bar_float(bar.get("Low")),
                float(close),
                volume,
            ))
        if not rows:
            return 0
        await self.db.executemany("""
            INSERT INTO stock_intraday_candles (
                ticker, timeframe, bar_time, open, high, low, close, volume, source, as_of
            ) VALUES ($1, '5m', $2, $3, $4, $5, $6, $7, 'yfinance_5m', NOW())
            ON CONFLICT (ticker, timeframe, bar_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                as_of = NOW()
        """, rows)
        return len(rows)
