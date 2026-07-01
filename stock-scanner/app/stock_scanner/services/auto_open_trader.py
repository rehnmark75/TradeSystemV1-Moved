"""
Auto Open Trader - places capped day-trade limit orders after the open.

This worker intentionally delegates order execution to trading-ui's
/api/orders/place route so it reuses the same RoboMarkets SL/TP protection,
broker level retry, and breakeven monitor registration logic.
"""

import asyncio
import json
import logging
import os
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

import aiohttp
import pytz
from stock_scanner.core.detection.market_hours import is_trading_day
from stock_scanner.core.trading.robomarkets_client import RoboMarketsClient

logger = logging.getLogger("auto_open_trader")


class AutoOpenTrader:
    ET = pytz.timezone("America/New_York")

    # FOMC rate-decision (announcement) days, ET — the second day of each
    # scheduled two-day FOMC meeting. The auto-trader only opens positions in the
    # morning window (~9:45-10:15 ET), but the rate decision lands at 14:00 ET and
    # positions are held multi-day, so a morning entry on a decision day is held
    # straight into the announcement's volatility/spread blowout. We skip the whole
    # day rather than a window around 14:00 (which the morning opener never reaches).
    # Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    # ⚠️ UPDATE ANNUALLY — the stale-schedule warning in _is_fomc_blackout() fires
    # once per process when today is past the last known date.
    FOMC_DECISION_DATES = frozenset({
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    })

    SETTING_DEFS = {
        "AUTO_TRADING_ENABLED": ("enabled", "bool", "true"),
        "AUTO_TRADING_DRY_RUN": ("dry_run", "bool", "false"),
        # FOMC blackout: skip ALL new entries on a rate-decision day (capital
        # preservation — avoids holding fresh day-trades into the 14:00 ET
        # announcement). On by default; off switch: AUTO_TRADE_FOMC_BLACKOUT_ENABLED=false.
        "AUTO_TRADE_FOMC_BLACKOUT_ENABLED": ("fomc_blackout_enabled", "bool", "true"),
        # Market-breadth regime gate (validated Jul 2026, ema_cross_lab.py): skip
        # ALL new long entries on days when fewer than MIN_PCT of the liquid
        # universe is above its 50-day MA (weak-tape seatbelt — the long
        # momentum-pullback edge inverts in broad downtrends). OFF by default;
        # fail-open on a data gap (a seatbelt, not a hard risk gate). Breadth uses
        # SMA50 (stock_screening_metrics.price_vs_sma50) as the live proxy for the
        # backtest's EMA50 breadth.
        "AUTO_TRADE_BREADTH_GATE_ENABLED": ("breadth_gate_enabled", "bool", "false"),
        "AUTO_TRADE_BREADTH_MIN_PCT": ("breadth_min_pct", "float", "50"),
        "AUTO_TRADE_BREADTH_MIN_DOLLAR_VOL": ("breadth_min_dollar_vol", "float", "5000000"),
        "AUTO_TRADE_BREADTH_MIN_UNIVERSE": ("breadth_min_universe", "int", "100"),
        "MAX_ORDER_NOTIONAL_USD": ("max_notional", "float", "500"),
        "MAX_ACTIVE_STOCK_ORDERS": ("max_active_orders", "int", "5"),
        "MAX_ORDERS_PER_RUN": ("max_orders_per_run", "int", "5"),
        # Candidate pool size (NOT an execution cap -- the 5/5 caps above still
        # bound how many orders fire). Widening this lets the rank-ordered
        # placement loop fall back to lower-ranked backups when the top names are
        # gate-rejected. MUST be <= the intraday-vwap worker's pool
        # (INTRADAY_VWAP_POOL_LIMIT) or wider names lack session_vwap and get
        # rejected by the fail-closed VWAP gate. Clamped to the route's 50 cap;
        # >50 needs the route-cap change (deferred 100/tier-3 phase).
        "AUTO_TRADE_CANDIDATE_POOL_LIMIT": ("candidate_pool_limit", "int", "50"),
        # Per-scanner slot cap in the candidate pool. Without this the route's
        # maxPerScanner defaults to `limit` (inert) and one prolific scanner can
        # monopolize all 50 slots. MUST match INTRADAY_VWAP_MAX_PER_SCANNER or
        # the pools diverge and wider names lack session_vwap (fail-closed VWAP
        # gate rejects them).
        "AUTO_TRADE_MAX_PER_SCANNER": ("max_per_scanner", "int", "10"),
        # EMA50 freshness gate (sessions). The route keeps a candidate only if it
        # is above its daily EMA50 AND crossed up within this many trading
        # sessions (a fresh reclaim, not an established trend). MUST match
        # INTRADAY_VWAP_EMA50_MAX_CROSS_AGE_SESSIONS so the VWAP worker computes
        # session_vwap for the same gated pool (else fail-closed VWAP gate rejects
        # names lacking it). 0 keeps only same-session crosses; raise to loosen.
        "AUTO_TRADE_EMA50_MAX_CROSS_AGE_SESSIONS": ("ema50_max_cross_age_sessions", "int", "10"),
        "AUTO_TRADE_MAX_SPREAD_PCT": ("max_spread_pct", "float", "0.4"),
        "AUTO_TRADE_MIN_SCORE": ("min_score", "float", "65"),
        # RVOL floor + VWAP veto: live-intraday confirmation gates. The values
        # below are conservative code fallbacks; the live runtime values come from
        # stock_auto_trade_settings (authoritative) / compose env. Enabled Jun 7
        # 2026 after the intraday-vwap worker was verified populating on cadence.
        "AUTO_TRADE_MIN_RELATIVE_VOLUME": ("min_relative_volume", "float", "0"),
        "AUTO_TRADE_REQUIRE_INTRADAY_RVOL": ("require_intraday_rvol", "bool", "false"),
        "AUTO_TRADE_REQUIRE_ABOVE_VWAP": ("require_above_vwap", "bool", "true"),
        # Day-change gate: at execution time the ticker must be trading above
        # the previous session's close (real-time broker quote vs prior close).
        # Fail-closed: no usable prev close -> no confirmation -> skip.
        "AUTO_TRADE_REQUIRE_POSITIVE_DAY": ("require_positive_day", "bool", "true"),
        # Minimum positive margin (percent) the day-change must clear, not just
        # > 0. A name +0.1% on the day is effectively flat and clears a bare
        # > 0 check (e.g. POET Jun 16: opened ~flat vs true prior close, passed,
        # then faded to -4%). Require some cushion to avoid buying flat names.
        # 0 keeps the old behaviour (any positive). Only applies when
        # require_positive_day is on.
        "AUTO_TRADE_MIN_DAY_CHANGE_PCT": ("min_day_change_pct", "float", "0"),
        "AUTO_TRADE_MAX_QUOTE_AGE_MINUTES": ("max_quote_age_minutes", "int", "3"),
        "AUTO_TRADE_PULLBACK_LIMIT_OFFSET_PCT": ("pullback_limit_offset_pct", "float", "0.3"),
        "AUTO_TRADE_START_DELAY_MINUTES": ("start_delay_minutes", "int", "15"),
        "AUTO_TRADE_VALIDATE_DELAY_MINUTES": ("validate_delay_minutes", "int", "5"),
        "AUTO_TRADE_STOP_AFTER_MINUTES": ("stop_after_minutes", "int", "45"),
        # SL/TP from Jun 11 2026 bracket sweep (sl_tp_sweep.py): SL2/TP7 plateau
        # (PF 1.09 vs 1.00 for old 3/5 at 2-3d horizon, H1/H2 robust).
        "AUTO_TRADE_STOP_LOSS_PCT": ("stop_loss_pct", "float", "2.0"),
        "AUTO_TRADE_TAKE_PROFIT_PCT": ("take_profit_pct", "float", "7.0"),
        "AUTO_TRADE_MAX_STOP_DISTANCE_PCT": ("max_stop_distance_pct", "float", "3.0"),
        "AUTO_TRADE_MAX_RISK_PCT": ("max_risk_pct", "float", "3.0"),
        "AUTO_TRADE_MAX_RISK_USD": ("max_risk_usd", "float", "15.0"),
        # Conditional ATR stop: DISABLED Jun 11 2026 — full-population bracket
        # sweep (sl_tp_sweep.py) showed the ATR-dynamic grid <= PF 1.0 at the
        # 2-3d horizon everywhere at portfolio level; fixed brackets dominate.
        # (Supersedes the Jun 5 high-ATR-quartile finding.)
        "AUTO_TRADE_ATR_STOP_ENABLED": ("atr_stop_enabled", "bool", "false"),
        "AUTO_TRADE_ATR_THRESHOLD_PCT": ("atr_threshold_pct", "float", "7.0"),
        "AUTO_TRADE_ATR_STOP_MULT": ("atr_stop_mult", "float", "1.0"),   # k: stop = k*ATR%
        "AUTO_TRADE_ATR_RR": ("atr_rr", "float", "1.6667"),              # TP = rr*stop (keeps 5/3)
    }

    def __init__(self, db_manager):
        self.db = db_manager
        self.trading_ui_url = os.getenv("TRADING_UI_URL", "http://trading-ui:3000/trading").rstrip("/")
        self.robomarkets_api_key = os.getenv("ROBOMARKETS_API_KEY", "")
        self.robomarkets_account_id = os.getenv("ROBOMARKETS_ACCOUNT_ID", "")
        self._fomc_last_known_date = max(self.FOMC_DECISION_DATES)
        self._fomc_stale_warned = False
        self._apply_env_settings()

    def _apply_env_settings(self):
        for env_key, (attr, value_type, default) in self.SETTING_DEFS.items():
            raw = os.getenv(env_key, default)
            setattr(self, attr, self._cast_setting(raw, value_type))

    async def ensure_schema(self):
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS stock_auto_trade_runs (
                id BIGSERIAL PRIMARY KEY,
                trade_date DATE NOT NULL UNIQUE,
                status VARCHAR(30) DEFAULT 'active',
                enabled BOOLEAN DEFAULT TRUE,
                dry_run BOOLEAN DEFAULT FALSE,
                started_at TIMESTAMP DEFAULT NOW(),
                validated_at TIMESTAMP,
                traded_at TIMESTAMP,
                completed_at TIMESTAMP,
                config JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS stock_auto_trade_candidates (
                id BIGSERIAL PRIMARY KEY,
                run_id BIGINT REFERENCES stock_auto_trade_runs(id),
                trade_date DATE NOT NULL,
                rank INTEGER,
                ticker VARCHAR(20) NOT NULL,
                status VARCHAR(30) DEFAULT 'queued',
                candidate_score DECIMAL(10,4),
                scanner_name VARCHAR(100),
                order_bias VARCHAR(50),
                pm_status VARCHAR(50),
                pm_direction VARCHAR(10),
                broker_bid DECIMAL(12,4),
                broker_ask DECIMAL(12,4),
                broker_last DECIMAL(12,4),
                broker_spread_pct DECIMAL(10,4),
                broker_quote_age_minutes INTEGER,
                relative_volume DECIMAL(12,4),
                intraday_relative_volume DECIMAL(12,4),
                planned_entry DECIMAL(12,4),
                planned_stop_loss DECIMAL(12,4),
                planned_take_profit DECIMAL(12,4),
                planned_quantity INTEGER,
                order_response JSONB,
                robomarkets_order_id VARCHAR(100),
                stock_order_id BIGINT,
                reason TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(run_id, ticker)
            )
        """)
        await self.db.execute("""
            ALTER TABLE stock_auto_trade_candidates
            ADD COLUMN IF NOT EXISTS broker_quote_age_minutes INTEGER
        """)
        await self.db.execute("""
            ALTER TABLE stock_auto_trade_candidates
            ADD COLUMN IF NOT EXISTS intraday_relative_volume DECIMAL(12,4)
        """)
        await self.db.execute("""
            ALTER TABLE stock_auto_trade_candidates
            ADD COLUMN IF NOT EXISTS atr_percent DECIMAL(10,4)
        """)
        await self.db.execute("""
            ALTER TABLE stock_auto_trade_candidates
            ADD COLUMN IF NOT EXISTS session_vwap DECIMAL(12,4)
        """)
        await self.db.execute("""
            ALTER TABLE stock_auto_trade_candidates
            ADD COLUMN IF NOT EXISTS prev_close DECIMAL(12,4)
        """)
        await self.db.execute("""
            ALTER TABLE stock_auto_trade_candidates
            ADD COLUMN IF NOT EXISTS day_change_pct DECIMAL(10,4)
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_auto_trade_candidates_run_status
            ON stock_auto_trade_candidates(run_id, status, rank)
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS stock_auto_trade_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                value_type VARCHAR(20) NOT NULL DEFAULT 'string',
                label TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        for env_key, (_, value_type, default) in self.SETTING_DEFS.items():
            await self.db.execute("""
                INSERT INTO stock_auto_trade_settings (key, value, value_type)
                VALUES ($1, $2, $3)
                ON CONFLICT (key) DO NOTHING
            """, env_key, os.getenv(env_key, default), value_type)

    async def run_once(self) -> Dict[str, Any]:
        await self.ensure_schema()
        await self._load_runtime_settings()
        now_et = datetime.now(self.ET)
        trade_date = now_et.date()
        result = {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "stage": "idle",
            "created": 0,
            "capped": 0,
            "validated": 0,
            "placed": 0,
            "rejected": 0,
            "skipped": 0,
            "errors": 0,
        }

        run_id = await self._get_or_create_run(trade_date)
        if not self.enabled:
            result["stage"] = "disabled"
            await self._mark_run_status(run_id, "disabled")
            return result

        if not is_trading_day(now_et):
            result["stage"] = "market_closed_holiday"
            return result

        if self.fomc_blackout_enabled and self._is_fomc_blackout(trade_date):
            result["stage"] = "fomc_blackout"
            logger.info(
                "[AutoTrader] FOMC decision day %s — blackout active, no new entries placed",
                trade_date,
            )
            await self._mark_run_status(run_id, "fomc_blackout")
            return result

        if self.breadth_gate_enabled:
            breadth = await self._compute_breadth_pct()
            if breadth is not None:
                result["breadth_pct"] = round(breadth, 1)
                if breadth < self.breadth_min_pct:
                    result["stage"] = "breadth_blocked"
                    logger.info(
                        "[AutoTrader] Market breadth %.1f%% < %.1f%% — regime gate active, no new entries %s",
                        breadth, self.breadth_min_pct, trade_date,
                    )
                    await self._mark_run_status(run_id, "breadth_blocked")
                    return result
                logger.info("[AutoTrader] Market breadth %.1f%% >= %.1f%% — regime OK",
                            breadth, self.breadth_min_pct)
            else:
                logger.warning("[AutoTrader] Breadth gate ON but breadth unavailable — failing OPEN (allowing trades)")

        minutes_after_open = self._minutes_after_open(now_et)
        if minutes_after_open is None or minutes_after_open < self.validate_delay_minutes:
            result["stage"] = "waiting_for_open_window"
            return result
        if minutes_after_open > self.stop_after_minutes:
            result["stage"] = "window_closed"
            await self._mark_run_status(run_id, "completed")
            return result

        candidates, capped = await self._fetch_daytrade_candidates()
        if not candidates:
            result["stage"] = "no_candidates"
            return result

        created = await self._upsert_candidates(run_id, trade_date, candidates)
        result["created"] = created
        result["capped"] = await self._record_capped_candidates(run_id, trade_date, capped)

        validated = await self._validate_candidates(run_id, candidates)
        result["validated"] = validated["watching"]
        result["rejected"] = validated["rejected"]

        if minutes_after_open < self.start_delay_minutes:
            result["stage"] = "validated_waiting_to_trade"
            await self.db.execute(
                "UPDATE stock_auto_trade_runs SET validated_at = COALESCE(validated_at, NOW()), updated_at = NOW() WHERE id = $1",
                run_id,
            )
            return result

        placed = await self._place_orders(run_id)
        result.update(placed)
        result["stage"] = "trade_window"
        await self.db.execute(
            "UPDATE stock_auto_trade_runs SET traded_at = COALESCE(traded_at, NOW()), updated_at = NOW() WHERE id = $1",
            run_id,
        )
        return result

    async def run_loop(self):
        await self.ensure_schema()
        interval = int(os.getenv("AUTO_TRADER_INTERVAL_SECONDS", "60"))
        logger.info(
            "Auto trader loop started enabled=%s dry_run=%s interval=%ss max_notional=%.2f max_active=%s",
            self.enabled, self.dry_run, interval, self.max_notional, self.max_active_orders
        )
        while True:
            try:
                result = await self.run_once()
                logger.info("[AutoTrader] %s", result)
            except Exception as exc:
                logger.exception("[AutoTrader] loop error: %s", exc)
            await asyncio.sleep(max(15, interval))

    async def _compute_breadth_pct(self) -> Optional[float]:
        """Percent of the liquid universe above its 50-day MA on the latest
        nightly metrics date. Proxy (SMA50) for the backtest's EMA50 breadth.
        Returns None if the reading can't be trusted (universe too small)."""
        try:
            row = await self.db.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE price_vs_sma50 > 0)::float
                        / NULLIF(COUNT(*), 0) AS breadth,
                    COUNT(*) AS n
                FROM stock_screening_metrics
                WHERE calculation_date = (
                        SELECT MAX(calculation_date) FROM stock_screening_metrics
                      )
                  AND price_vs_sma50 IS NOT NULL
                  AND avg_dollar_volume >= $1
                """,
                float(self.breadth_min_dollar_vol),
            )
            if not row or row["breadth"] is None:
                return None
            if int(row["n"] or 0) < int(self.breadth_min_universe):
                logger.warning("[AutoTrader] Breadth universe too small (%s) — untrusted", row["n"])
                return None
            return float(row["breadth"]) * 100.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AutoTrader] Breadth compute failed: %s", exc)
            return None

    async def _load_runtime_settings(self):
        rows = await self.db.fetch("""
            SELECT key, value, value_type
            FROM stock_auto_trade_settings
            WHERE key = ANY($1::text[])
        """, list(self.SETTING_DEFS.keys()))
        for row in rows:
            env_key = row["key"]
            if env_key not in self.SETTING_DEFS:
                continue
            attr, expected_type, _ = self.SETTING_DEFS[env_key]
            try:
                setattr(self, attr, self._cast_setting(row["value"], row["value_type"] or expected_type))
            except Exception:
                logger.warning("Ignoring invalid auto trader setting %s=%r", env_key, row["value"])

    def _cast_setting(self, raw: Any, value_type: str) -> Any:
        if value_type == "bool":
            return str(raw).lower() in ("1", "true", "yes", "on")
        if value_type == "int":
            return int(float(raw))
        if value_type == "float":
            return float(raw)
        return str(raw)

    async def _get_or_create_run(self, trade_date) -> int:
        config = {
            "trading_ui_url": self.trading_ui_url,
            "fomc_blackout_enabled": self.fomc_blackout_enabled,
            "max_notional": self.max_notional,
            "max_active_orders": self.max_active_orders,
            "max_orders_per_run": self.max_orders_per_run,
            "max_spread_pct": self.max_spread_pct,
            "min_score": self.min_score,
            "min_relative_volume": self.min_relative_volume,
            "require_intraday_rvol": self.require_intraday_rvol,
            "require_above_vwap": self.require_above_vwap,
            "max_quote_age_minutes": self.max_quote_age_minutes,
            "pullback_limit_offset_pct": self.pullback_limit_offset_pct,
            "start_delay_minutes": self.start_delay_minutes,
            "validate_delay_minutes": self.validate_delay_minutes,
            "stop_after_minutes": self.stop_after_minutes,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_stop_distance_pct": self.max_stop_distance_pct,
            "max_risk_pct": self.max_risk_pct,
            "max_risk_usd": self.max_risk_usd,
        }
        row = await self.db.fetchrow("""
            INSERT INTO stock_auto_trade_runs (trade_date, enabled, dry_run, config)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (trade_date) DO UPDATE SET
                enabled = EXCLUDED.enabled,
                dry_run = EXCLUDED.dry_run,
                config = EXCLUDED.config,
                updated_at = NOW()
            RETURNING id
        """, trade_date, self.enabled, self.dry_run, json.dumps(config))
        return int(row["id"])

    async def _fetch_daytrade_candidates(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Clamp to the route's hard cap (daytrades returns at most 50). A larger
        # configured value is honored only once the route cap is raised (deferred).
        pool = min(int(self.candidate_pool_limit), 50)
        if int(self.candidate_pool_limit) > 50:
            logger.warning(
                "AUTO_TRADE_CANDIDATE_POOL_LIMIT=%s clamped to 50 (route cap); "
                "raise the /api/signals/top cap to go higher",
                self.candidate_pool_limit,
            )
        max_per_scanner = max(1, min(int(self.max_per_scanner), pool))
        url = (
            f"{self.trading_ui_url}/api/signals/top?limit={pool}&mode=daytrades"
            f"&maxPerScanner={max_per_scanner}"
            f"&maxEmaCrossAgeSessions={int(self.ema50_max_cross_age_sessions)}"
        )
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                payload = await response.json()
                if response.status >= 400:
                    raise RuntimeError(f"Top day trades API returned {response.status}: {payload}")
                return list(payload.get("rows") or []), list(payload.get("capped_rows") or [])

    async def _upsert_candidates(self, run_id: int, trade_date, candidates: List[Dict[str, Any]]) -> int:
        count = 0
        for rank, row in enumerate(candidates, start=1):
            await self.db.execute("""
                INSERT INTO stock_auto_trade_candidates (
                    run_id, trade_date, rank, ticker, candidate_score, scanner_name,
                    order_bias, pm_status, pm_direction, broker_bid, broker_ask,
                    broker_last, broker_spread_pct, broker_quote_age_minutes,
                    relative_volume, intraday_relative_volume, atr_percent,
                    planned_entry, planned_stop_loss, planned_take_profit, session_vwap,
                    prev_close, day_change_pct
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23)
                ON CONFLICT (run_id, ticker) DO UPDATE SET
                    status = CASE
                        WHEN stock_auto_trade_candidates.status = 'capped' THEN 'queued'
                        ELSE stock_auto_trade_candidates.status
                    END,
                    reason = CASE
                        WHEN stock_auto_trade_candidates.status = 'capped' THEN NULL
                        ELSE stock_auto_trade_candidates.reason
                    END,
                    rank = EXCLUDED.rank,
                    candidate_score = EXCLUDED.candidate_score,
                    scanner_name = EXCLUDED.scanner_name,
                    order_bias = EXCLUDED.order_bias,
                    pm_status = EXCLUDED.pm_status,
                    pm_direction = EXCLUDED.pm_direction,
                    -- Market snapshot (broker quote + derived gate inputs) freezes
                    -- once an order is committed, so the row stays an immutable record
                    -- of what the gates actually evaluated at decision time. Without
                    -- this, later polls overwrite these with post-entry values and a
                    -- clean entry can look like it entered red (e.g. POET Jun 16:
                    -- entered +0.43% on day, faded to -4%, stored value showed -4%).
                    broker_bid = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.broker_bid ELSE EXCLUDED.broker_bid END,
                    broker_ask = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.broker_ask ELSE EXCLUDED.broker_ask END,
                    broker_last = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.broker_last ELSE EXCLUDED.broker_last END,
                    broker_spread_pct = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.broker_spread_pct ELSE EXCLUDED.broker_spread_pct END,
                    broker_quote_age_minutes = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.broker_quote_age_minutes ELSE EXCLUDED.broker_quote_age_minutes END,
                    relative_volume = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.relative_volume ELSE EXCLUDED.relative_volume END,
                    intraday_relative_volume = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.intraday_relative_volume ELSE EXCLUDED.intraday_relative_volume END,
                    atr_percent = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.atr_percent ELSE EXCLUDED.atr_percent END,
                    session_vwap = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.session_vwap ELSE EXCLUDED.session_vwap END,
                    prev_close = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.prev_close ELSE EXCLUDED.prev_close END,
                    day_change_pct = CASE WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run') THEN stock_auto_trade_candidates.day_change_pct ELSE EXCLUDED.day_change_pct END,
                    planned_entry = CASE
                        WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run')
                        THEN stock_auto_trade_candidates.planned_entry
                        ELSE EXCLUDED.planned_entry
                    END,
                    planned_stop_loss = CASE
                        WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run')
                        THEN stock_auto_trade_candidates.planned_stop_loss
                        ELSE EXCLUDED.planned_stop_loss
                    END,
                    planned_take_profit = CASE
                        WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run')
                        THEN stock_auto_trade_candidates.planned_take_profit
                        ELSE EXCLUDED.planned_take_profit
                    END,
                    updated_at = NOW()
            """,
                run_id, trade_date, rank, row.get("ticker"), self._num(row.get("candidate_score")),
                row.get("scanner_name"), row.get("order_bias"), row.get("pm_status"), row.get("pm_direction"),
                self._num(row.get("broker_bid")), self._num(row.get("broker_ask")),
                self._num(row.get("broker_last")), self._num(row.get("broker_spread_pct")),
                self._int(row.get("broker_quote_age_minutes")),
                self._num(row.get("relative_volume")),
                self._num(row.get("intraday_relative_volume")),
                self._num(row.get("atr_percent")),
                self._num(row.get("pm_suggested_entry")),
                self._num(row.get("pm_suggested_stop")), self._num(row.get("pm_suggested_target")),
                self._num(row.get("session_vwap")),
                *self._day_change(row),
            )
            count += 1
        return count

    async def _record_capped_candidates(self, run_id: int, trade_date, capped: List[Dict[str, Any]]) -> int:
        """Audit-log names the route's per-scanner cap dropped from the pool.

        These rows never enter validation or trading (status='capped' is outside
        every selection filter); they exist so "did the cap suppress winners from
        scanner X?" is answerable later. Raw SQL-scored rows, no broker quote.
        """
        count = 0
        for row in capped:
            ticker = row.get("ticker")
            if not ticker:
                continue
            reason = (
                f"Per-scanner cap: {row.get('scanner_name') or 'unknown'} "
                f"rank {row.get('scanner_rank')} > max {self.max_per_scanner}"
            )
            await self.db.execute("""
                INSERT INTO stock_auto_trade_candidates (
                    run_id, trade_date, rank, ticker, status, reason, candidate_score,
                    scanner_name, order_bias, pm_status, pm_direction,
                    relative_volume, atr_percent
                ) VALUES ($1,$2,NULL,$3,'capped',$4,$5,$6,$7,$8,$9,$10,$11)
                ON CONFLICT (run_id, ticker) DO UPDATE SET
                    status = CASE
                        WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run', 'failed')
                        THEN stock_auto_trade_candidates.status
                        ELSE 'capped'
                    END,
                    reason = CASE
                        WHEN stock_auto_trade_candidates.status IN ('order_submitted', 'dry_run', 'failed')
                        THEN stock_auto_trade_candidates.reason
                        ELSE EXCLUDED.reason
                    END,
                    candidate_score = EXCLUDED.candidate_score,
                    scanner_name = EXCLUDED.scanner_name,
                    updated_at = NOW()
            """,
                run_id, trade_date, ticker, reason, self._num(row.get("candidate_score")),
                row.get("scanner_name"), row.get("order_bias"), row.get("pm_status"),
                row.get("pm_direction"), self._num(row.get("relative_volume")),
                self._num(row.get("atr_percent")),
            )
            count += 1
        return count

    async def _validate_candidates(self, run_id: int, candidates: List[Dict[str, Any]]) -> Dict[str, int]:
        watching = 0
        rejected = 0
        by_ticker = {str(row.get("ticker")): row for row in candidates}
        rows = await self.db.fetch("""
            SELECT id, ticker
            FROM stock_auto_trade_candidates
            WHERE run_id = $1 AND status IN ('queued', 'watching', 'rejected')
            ORDER BY rank
        """, run_id)
        for candidate in rows:
            ticker = candidate["ticker"]
            row = by_ticker.get(ticker)
            reason = self._reject_reason(row)
            if reason:
                await self._update_candidate_status(candidate["id"], "rejected", reason)
                rejected += 1
            else:
                await self._update_candidate_status(candidate["id"], "watching", "Passed open validation")
                watching += 1
        return {"watching": watching, "rejected": rejected}

    async def _place_orders(self, run_id: int) -> Dict[str, int]:
        result = {"placed": 0, "skipped": 0, "errors": 0}
        rows = await self.db.fetch("""
            SELECT *
            FROM stock_auto_trade_candidates
            WHERE run_id = $1 AND status = 'watching'
            ORDER BY rank
        """, run_id)

        active_stock_order_tickers = await self._active_stock_order_tickers()
        live_broker_tickers = await self._live_broker_position_tickers()

        for row in rows:
            if result["placed"] >= self.max_orders_per_run:
                result["skipped"] += 1
                continue

            existing_exposure = self._active_ticker_exposure(
                row["ticker"],
                active_stock_order_tickers,
                live_broker_tickers,
            )
            if existing_exposure:
                await self._update_candidate_status(
                    row["id"],
                    "skipped",
                    f"Existing active exposure for {row['ticker']}: {existing_exposure}",
                )
                result["skipped"] += 1
                continue

            active_count = len(active_stock_order_tickers | live_broker_tickers)
            if active_count >= self.max_active_orders:
                await self._update_candidate_status(row["id"], "skipped", f"Active ticker cap reached: {active_count}/{self.max_active_orders}")
                result["skipped"] += 1
                continue

            entry = self._entry_for_bias(row)
            if not entry or entry <= 0:
                await self._update_candidate_status(row["id"], "failed", f"No usable broker entry price for bias {row['order_bias'] or 'Watch'}")
                result["errors"] += 1
                continue

            plan = self._build_order_plan(entry, self._num(row.get("atr_percent")))
            if plan.get("reject_reason"):
                await self._update_candidate_status(row["id"], "rejected", plan["reject_reason"])
                result["skipped"] += 1
                continue

            quantity = plan["quantity"]
            stop = plan["stop_loss"]
            target = plan["take_profit"]
            risk_amount = plan["risk_amount"]
            risk_pct = plan["risk_pct"]
            stop_distance_pct = plan["stop_distance_pct"]
            mode = plan.get("mode", "fixed")

            await self.db.execute("""
                UPDATE stock_auto_trade_candidates
                SET planned_entry = $2, planned_stop_loss = $3, planned_take_profit = $4,
                    planned_quantity = $5, updated_at = NOW()
                WHERE id = $1
            """, row["id"], entry, stop, target, quantity)

            if self.dry_run:
                await self._update_candidate_status(row["id"], "dry_run", f"[{mode}] Would place buy limit qty={quantity} entry={entry:.2f} SL={stop:.2f} TP={target:.2f} risk=${risk_amount:.2f} ({risk_pct:.2f}%)")
                result["placed"] += 1
                continue

            try:
                response = await self._place_order_api(
                    ticker=row["ticker"],
                    quantity=quantity,
                    price=entry,
                    stop_loss=stop,
                    take_profit=target,
                )
                if response.get("status") != "submitted" and self._is_price_too_close_response(response):
                    original_response = response
                    fallback_entry = self._price_too_close_fallback_entry(row, entry)
                    if fallback_entry and fallback_entry < entry:
                        fallback_plan = self._build_order_plan(fallback_entry, self._num(row.get("atr_percent")))
                        if not fallback_plan.get("reject_reason"):
                            entry = fallback_entry
                            plan = fallback_plan
                            quantity = plan["quantity"]
                            stop = plan["stop_loss"]
                            target = plan["take_profit"]
                            risk_amount = plan["risk_amount"]
                            risk_pct = plan["risk_pct"]
                            stop_distance_pct = plan["stop_distance_pct"]
                            mode = f"{plan.get('mode', 'fixed')}+quote_pullback_retry"

                            await self.db.execute("""
                                UPDATE stock_auto_trade_candidates
                                SET planned_entry = $2, planned_stop_loss = $3, planned_take_profit = $4,
                                    planned_quantity = $5, updated_at = NOW()
                                WHERE id = $1
                            """, row["id"], entry, stop, target, quantity)

                            response = await self._place_order_api(
                                ticker=row["ticker"],
                                quantity=quantity,
                                price=entry,
                                stop_loss=stop,
                                take_profit=target,
                            )
                            response["fallback_reason"] = "price_too_close_to_quote"
                            response["initial_order_response"] = original_response

                status = "order_submitted" if response.get("status") == "submitted" else "failed"
                await self.db.execute("""
                    UPDATE stock_auto_trade_candidates
                    SET status = $2,
                        order_response = $3::jsonb,
                        robomarkets_order_id = $4,
                        stock_order_id = $5,
                        reason = $6,
                        updated_at = NOW()
                    WHERE id = $1
                """,
                    row["id"], status, json.dumps(response), response.get("robomarkets_order_id"),
                    self._int(response.get("db_order_id")),
                    f"Order submitted by auto trader [{mode}]; risk=${risk_amount:.2f} ({risk_pct:.2f}%), stop distance={stop_distance_pct:.2f}%" if status == "order_submitted" else response.get("error"),
                )
                if status == "order_submitted":
                    active_stock_order_tickers.add(str(row["ticker"]).split(".")[0].upper())
                    result["placed"] += 1
                else:
                    result["errors"] += 1
            except Exception as exc:
                await self._update_candidate_status(row["id"], "failed", str(exc))
                result["errors"] += 1

        return result

    def _build_order_plan(self, entry: float, atr_pct: Optional[float] = None) -> Dict[str, Any]:
        epsilon = 1e-9
        # Conditional ATR stop: for high-ATR names a flat 3% sits inside the noise
        # band (validated Jun 5 2026 — only the ATR>=threshold quartile showed an
        # edge). Use an ATR-width stop + ATR-scaled size so DOLLAR risk stays
        # max_risk_usd. Below threshold (or if ATR missing), keep the fixed stop.
        use_atr = (
            self.atr_stop_enabled
            and atr_pct is not None
            and atr_pct >= self.atr_threshold_pct
            and self.atr_stop_mult > 0
        )
        if use_atr:
            stop_loss_pct = max(0.01, self.atr_stop_mult * float(atr_pct))
            take_profit_pct = max(stop_loss_pct + 0.01, self.atr_rr * stop_loss_pct)
            mode = f"atr({float(atr_pct):.1f}%)"
        else:
            stop_loss_pct = max(0.01, float(self.stop_loss_pct))
            take_profit_pct = max(0.01, float(self.take_profit_pct))
            mode = "fixed"

        stop = round(entry * (1 - stop_loss_pct / 100), 2)
        target = round(entry * (1 + take_profit_pct / 100), 2)
        if stop >= entry:
            return {"reject_reason": f"Computed stop {stop:.2f} is not below entry {entry:.2f}"}
        if target <= entry:
            return {"reject_reason": f"Computed target {target:.2f} is not above entry {entry:.2f}"}

        per_share_risk = entry - stop
        # ATR branch: size to hold dollar risk at the cap regardless of stop width.
        # Fixed branch: size to notional (legacy behaviour, then verified vs caps).
        quantity = int(self.max_risk_usd // per_share_risk) if use_atr else int(self.max_notional // entry)
        if quantity <= 0:
            return {"reject_reason": (
                f"Stop width {stop_loss_pct:.1f}% too wide for max risk ${self.max_risk_usd:.2f} at entry {entry:.2f}"
                if use_atr else
                f"Price {entry:.2f} too high for max notional {self.max_notional:.2f}")}

        notional = entry * quantity
        risk_amount = per_share_risk * quantity
        risk_pct = (risk_amount / notional) * 100 if notional > 0 else 0
        stop_distance_pct = (per_share_risk / entry) * 100

        if not use_atr:
            # Fixed branch keeps the original %-based guard rails.
            if stop_distance_pct > self.max_stop_distance_pct + epsilon:
                return {"reject_reason": f"Stop distance {stop_distance_pct:.2f}% > max {self.max_stop_distance_pct:.2f}% at entry {entry:.2f}"}
            if risk_pct > self.max_risk_pct + epsilon:
                return {"reject_reason": f"Risk {risk_pct:.2f}% > max {self.max_risk_pct:.2f}% (${risk_amount:.2f} on ${notional:.2f})"}
        # Both branches: hard DOLLAR-risk ceiling (ATR sizing satisfies it by construction).
        if risk_amount > self.max_risk_usd + epsilon:
            return {"reject_reason": f"Risk ${risk_amount:.2f} > max ${self.max_risk_usd:.2f} ({risk_pct:.2f}% on ${notional:.2f})"}

        return {
            "stop_loss": stop,
            "take_profit": target,
            "quantity": quantity,
            "notional": notional,
            "risk_amount": risk_amount,
            "risk_pct": risk_pct,
            "stop_distance_pct": stop_distance_pct,
            "mode": mode,
        }

    async def _place_order_api(self, ticker: str, quantity: int, price: float, stop_loss: float, take_profit: float) -> Dict[str, Any]:
        url = f"{self.trading_ui_url}/api/orders/place"
        payload = {
            "ticker": ticker,
            "side": "buy",
            "order_type": "limit",
            "quantity": quantity,
            "price": round(price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "trade_ready_override": True,
            # BE disabled Jun 11 2026: trigger sweep (be_trigger_sweep.py) showed
            # BE kills ~1.7x more winners than the losers it rescues at every
            # trigger level/horizon; BE-off dominates (PF 0.99 vs 0.44 at $10).
            "breakeven_enabled": False,
        }
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if response.status >= 400:
                    data.setdefault("error", f"Order API returned {response.status}")
                return data

    async def _active_stock_order_tickers(self) -> set[str]:
        rows = await self.db.fetch("""
            SELECT UPPER(split_part(ticker, '.', 1)) AS normalized_ticker
            FROM stock_orders
            WHERE status IN ('pending', 'submitted', 'partially_filled')
        """)
        return {
            str(row["normalized_ticker"]).strip().upper()
            for row in rows
            if row.get("normalized_ticker")
        }

    async def _live_broker_position_tickers(self) -> set[str]:
        if not self.robomarkets_api_key or not self.robomarkets_account_id:
            logger.warning("RoboMarkets credentials missing; live broker exposure check returned no positions")
            return set()

        try:
            async with RoboMarketsClient(
                api_key=self.robomarkets_api_key,
                account_id=self.robomarkets_account_id,
            ) as client:
                positions = await client.get_positions()
        except Exception as e:
            logger.error("Failed to fetch live RoboMarkets positions for active cap: %s", e)
            raise

        return {
            str(position.ticker).split(".")[0].strip().upper()
            for position in positions
            if getattr(position, "ticker", None)
        }

    def _active_ticker_exposure(
        self,
        ticker: str,
        active_stock_order_tickers: set[str],
        live_broker_tickers: set[str],
    ) -> Optional[str]:
        normalized = str(ticker or "").split(".")[0].upper()
        if not normalized:
            return "missing ticker"
        if normalized in active_stock_order_tickers:
            return "stock_order:active"
        if normalized in live_broker_tickers:
            return "broker_live_position:open"
        return None

    def _day_change(self, row: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        """(prev_close, day_change_pct) from real-time broker price vs prior close.

        Prefer the screening-metrics prior-session close (nightly daily close for
        MAX(calculation_date)); the PM service's previous_close can be days stale
        (Jun 12: DAKT PM said 19.13 when yesterday closed 20.93) so it is only a
        fallback, and only when the PM row is from today's session.
        """
        prev_close = self._num(row.get("prior_session_close"))
        if (not prev_close or prev_close <= 0) and row.get("pm_is_current_session"):
            prev_close = self._num(row.get("pm_previous_close"))
        if not prev_close or prev_close <= 0:
            return None, None
        price = self._num(row.get("broker_last")) or self._num(row.get("broker_ask"))
        if not price or price <= 0:
            return prev_close, None
        return prev_close, (price - prev_close) / prev_close * 100

    def _reject_reason(self, row: Optional[Dict[str, Any]]) -> Optional[str]:
        if not row:
            return f"Missing from refreshed Day Trades pool (top {self.candidate_pool_limit})"
        # The route's daytrades tradability gate is authoritative for "not
        # tradable" -- honor it as a veto so we never place an order the picker
        # already rejected (e.g. "Spread eats range", or any future order_bias /
        # gate the route adds without this list being updated).
        if row.get("tradable") is False:
            return f"Route gate not tradable: {row.get('gate_reason') or 'rejected'}"
        if row.get("signal_type") and row.get("signal_type") != "BUY":
            return f"Signal is not BUY: {row.get('signal_type')}"
        if row.get("pm_direction") and row.get("pm_direction") != "BUY":
            return f"Premarket direction is not BUY: {row.get('pm_direction')}"
        # PM hard block covers only ACTIVE bearish reads (gap against the long /
        # fading). Stale/missing PM is a data-availability problem, not a signal:
        # since Jun 12 2026 those names are allowed through IF the route's
        # live-in-play rescue confirmed them (live RVOL >= 2, holding >= VWAP,
        # spread OK) -- non-rescued stale/no-PM names still arrive tradable=False
        # and are vetoed above. PM levels are not load-bearing for execution
        # (entry falls back to ask/last; brackets are fixed SL/TP pct).
        if row.get("pm_status") in ("PM against", "PM fading"):
            return f"Bad PM status: {row.get('pm_status')}"
        if row.get("order_bias") in ("Avoid", "Avoid spread", "Refresh", "Wait", "Wait pullback"):
            return f"Bad order bias: {row.get('order_bias')}"
        quote_age = self._num(row.get("broker_quote_age_minutes"))
        if quote_age is None:
            return "Missing broker quote age"
        if quote_age > self.max_quote_age_minutes:
            return f"Broker quote age {quote_age:.0f}m > max {self.max_quote_age_minutes}m"
        spread = self._num(row.get("broker_spread_pct"))
        if spread is None:
            return "Missing broker spread"
        if spread > self.max_spread_pct:
            return f"Spread {spread:.3f}% > max {self.max_spread_pct:.3f}%"
        score = self._num(row.get("candidate_score")) or 0
        if score < self.min_score:
            return f"Score {score:.1f} < min {self.min_score:.1f}"
        intraday_rvol = self._num(row.get("intraday_relative_volume"))
        if intraday_rvol is None:
            if self.require_intraday_rvol and self.min_relative_volume > 0:
                return "Missing live intraday RVOL"
        elif intraday_rvol < self.min_relative_volume:
            return f"Intraday RVOL {intraday_rvol:.2f} < min {self.min_relative_volume:.2f}"
        # Live VWAP veto: a day-trade long below the session VWAP is buying into
        # intraday weakness (exactly what BIDU/CADL did on Jun 5 -- both below
        # VWAP at the snapshot). session_vwap is the intraday-vwap worker's real
        # sub-hourly VWAP; compare it to the CURRENT broker price (last, fallback
        # ask) -- current price vs session average is the "holding above VWAP"
        # read. Fail-closed when required: no live VWAP -> no confirmation -> skip
        # rather than trade blind. Off switch: AUTO_TRADE_REQUIRE_ABOVE_VWAP=false.
        if self.require_above_vwap:
            session_vwap = self._num(row.get("session_vwap"))
            price_for_vwap = self._num(row.get("broker_last")) or self._num(row.get("broker_ask"))
            if session_vwap is None or session_vwap <= 0:
                return "Missing live session VWAP (require_above_vwap)"
            if price_for_vwap is None or price_for_vwap <= 0:
                return "Missing broker price for VWAP check"
            if price_for_vwap < session_vwap:
                return f"Price {price_for_vwap:.2f} below session VWAP {session_vwap:.2f}"
        # Day-change veto: a BUY on a ticker trading at/below yesterday's close
        # is buying a red name. Uses the real-time broker quote (age-gated
        # above), not the scanner's delayed data. Fail-closed like the VWAP
        # gate. Off switch: AUTO_TRADE_REQUIRE_POSITIVE_DAY=false.
        if self.require_positive_day:
            prev_close, day_change_pct = self._day_change(row)
            if prev_close is None:
                return "Missing previous close (require_positive_day)"
            if day_change_pct is None:
                return "Missing broker price for day-change check"
            min_margin = self.min_day_change_pct
            if day_change_pct <= 0:
                return f"Not positive on day: {day_change_pct:+.2f}% (prev close {prev_close:.2f})"
            if min_margin > 0 and day_change_pct < min_margin:
                return (
                    f"Day change {day_change_pct:+.2f}% below min margin "
                    f"{min_margin:.2f}% (prev close {prev_close:.2f})"
                )
        ask = self._num(row.get("broker_ask"))
        if ask is None or ask <= 0:
            return "Missing broker ask"
        return None

    def _entry_for_bias(self, row) -> Optional[float]:
        ask = self._num(row["broker_ask"])
        last = self._num(row["broker_last"])
        planned = self._num(row["planned_entry"])
        bias = row["order_bias"] or "Watch"

        if bias == "Pullback" and ask:
            return round(ask * (1 - max(0.0, self.pullback_limit_offset_pct) / 100), 2)
        if bias == "Use PM levels" and planned and ask:
            return round(min(planned, ask), 2)
        return ask or last or planned

    def _price_too_close_fallback_entry(self, row, current_entry: float) -> Optional[float]:
        ask = self._num(row.get("broker_ask"))
        base = ask or current_entry
        offset_pct = max(0.0, self.pullback_limit_offset_pct)
        if not base or base <= 0 or offset_pct <= 0:
            return None
        fallback = round(base * (1 - offset_pct / 100), 2)
        if current_entry and fallback >= current_entry:
            fallback = round(current_entry * (1 - offset_pct / 100), 2)
        return fallback if fallback > 0 else None

    def _is_price_too_close_response(self, response: Dict[str, Any]) -> bool:
        messages = [str(response.get("error") or "")]
        attempts = response.get("level_adjustment_attempts") or []
        if isinstance(attempts, list):
            messages.extend(str(attempt.get("error") or "") for attempt in attempts if isinstance(attempt, dict))
        return any("price too close to quote" in message.lower() for message in messages)

    async def _update_candidate_status(self, candidate_id: int, status: str, reason: str):
        await self.db.execute("""
            UPDATE stock_auto_trade_candidates
            SET status = $2, reason = $3, updated_at = NOW()
            WHERE id = $1
        """, candidate_id, status, reason)

    async def _mark_run_status(self, run_id: int, status: str):
        await self.db.execute("""
            UPDATE stock_auto_trade_runs
            SET status = $2::varchar,
                completed_at = CASE WHEN $2::varchar IN ('completed', 'disabled') THEN NOW() ELSE completed_at END,
                updated_at = NOW()
            WHERE id = $1
        """, run_id, status)

    def _is_fomc_blackout(self, trade_date: date) -> bool:
        """True if trade_date is a scheduled FOMC rate-decision day.

        Schedule is hardcoded (FOMC_DECISION_DATES) — deterministic and dependency
        free, the right trade-off for a live capital-preservation account. Warns
        once per process if the schedule has gone stale so it never silently
        expires (an empty/old list would leave the gate permanently open).
        """
        if trade_date in self.FOMC_DECISION_DATES:
            return True
        if not self._fomc_stale_warned and trade_date > self._fomc_last_known_date:
            logger.warning(
                "[AutoTrader] FOMC blackout schedule is stale (last known %s, today %s); "
                "update FOMC_DECISION_DATES from the Fed calendar",
                self._fomc_last_known_date, trade_date,
            )
            self._fomc_stale_warned = True
        return False

    def _minutes_after_open(self, now_et: datetime) -> Optional[int]:
        if not is_trading_day(now_et):
            return None
        market_open = self.ET.localize(datetime.combine(now_et.date(), time(9, 30)))
        delta = now_et - market_open
        return int(delta.total_seconds() // 60)

    def _num(self, value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int(self, value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
