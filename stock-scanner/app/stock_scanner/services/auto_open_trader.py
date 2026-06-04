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
from datetime import datetime, time
from typing import Any, Dict, List, Optional

import aiohttp
import pytz

logger = logging.getLogger("auto_open_trader")


class AutoOpenTrader:
    ET = pytz.timezone("America/New_York")
    SETTING_DEFS = {
        "AUTO_TRADING_ENABLED": ("enabled", "bool", "true"),
        "AUTO_TRADING_DRY_RUN": ("dry_run", "bool", "false"),
        "MAX_ORDER_NOTIONAL_USD": ("max_notional", "float", "500"),
        "MAX_ACTIVE_STOCK_ORDERS": ("max_active_orders", "int", "5"),
        "MAX_ORDERS_PER_RUN": ("max_orders_per_run", "int", "5"),
        "AUTO_TRADE_MAX_SPREAD_PCT": ("max_spread_pct", "float", "0.4"),
        "AUTO_TRADE_MIN_SCORE": ("min_score", "float", "65"),
        "AUTO_TRADE_MIN_RELATIVE_VOLUME": ("min_relative_volume", "float", "1.0"),
        "AUTO_TRADE_START_DELAY_MINUTES": ("start_delay_minutes", "int", "15"),
        "AUTO_TRADE_VALIDATE_DELAY_MINUTES": ("validate_delay_minutes", "int", "5"),
        "AUTO_TRADE_STOP_AFTER_MINUTES": ("stop_after_minutes", "int", "45"),
    }

    def __init__(self, db_manager):
        self.db = db_manager
        self.trading_ui_url = os.getenv("TRADING_UI_URL", "http://trading-ui:3000/trading").rstrip("/")
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
                relative_volume DECIMAL(12,4),
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

        minutes_after_open = self._minutes_after_open(now_et)
        if minutes_after_open is None or minutes_after_open < self.validate_delay_minutes:
            result["stage"] = "waiting_for_open_window"
            return result
        if minutes_after_open > self.stop_after_minutes:
            result["stage"] = "window_closed"
            await self._mark_run_status(run_id, "completed")
            return result

        candidates = await self._fetch_daytrade_candidates()
        if not candidates:
            result["stage"] = "no_candidates"
            return result

        created = await self._upsert_candidates(run_id, trade_date, candidates)
        result["created"] = created

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
            "max_notional": self.max_notional,
            "max_active_orders": self.max_active_orders,
            "max_orders_per_run": self.max_orders_per_run,
            "max_spread_pct": self.max_spread_pct,
            "min_score": self.min_score,
            "min_relative_volume": self.min_relative_volume,
            "start_delay_minutes": self.start_delay_minutes,
            "validate_delay_minutes": self.validate_delay_minutes,
            "stop_after_minutes": self.stop_after_minutes,
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

    async def _fetch_daytrade_candidates(self) -> List[Dict[str, Any]]:
        url = f"{self.trading_ui_url}/api/signals/top?limit=20&mode=daytrades"
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                payload = await response.json()
                if response.status >= 400:
                    raise RuntimeError(f"Top day trades API returned {response.status}: {payload}")
                return list(payload.get("rows") or [])

    async def _upsert_candidates(self, run_id: int, trade_date, candidates: List[Dict[str, Any]]) -> int:
        count = 0
        for rank, row in enumerate(candidates, start=1):
            await self.db.execute("""
                INSERT INTO stock_auto_trade_candidates (
                    run_id, trade_date, rank, ticker, candidate_score, scanner_name,
                    order_bias, pm_status, pm_direction, broker_bid, broker_ask,
                    broker_last, broker_spread_pct, relative_volume,
                    planned_entry, planned_stop_loss, planned_take_profit
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
                ON CONFLICT (run_id, ticker) DO UPDATE SET
                    rank = EXCLUDED.rank,
                    candidate_score = EXCLUDED.candidate_score,
                    scanner_name = EXCLUDED.scanner_name,
                    order_bias = EXCLUDED.order_bias,
                    pm_status = EXCLUDED.pm_status,
                    pm_direction = EXCLUDED.pm_direction,
                    broker_bid = EXCLUDED.broker_bid,
                    broker_ask = EXCLUDED.broker_ask,
                    broker_last = EXCLUDED.broker_last,
                    broker_spread_pct = EXCLUDED.broker_spread_pct,
                    relative_volume = EXCLUDED.relative_volume,
                    planned_entry = EXCLUDED.planned_entry,
                    planned_stop_loss = EXCLUDED.planned_stop_loss,
                    planned_take_profit = EXCLUDED.planned_take_profit,
                    updated_at = NOW()
            """,
                run_id, trade_date, rank, row.get("ticker"), self._num(row.get("candidate_score")),
                row.get("scanner_name"), row.get("order_bias"), row.get("pm_status"), row.get("pm_direction"),
                self._num(row.get("broker_bid")), self._num(row.get("broker_ask")),
                self._num(row.get("broker_last")), self._num(row.get("broker_spread_pct")),
                self._num(row.get("relative_volume")), self._num(row.get("pm_suggested_entry")),
                self._num(row.get("pm_suggested_stop")), self._num(row.get("pm_suggested_target")),
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

        for row in rows:
            if result["placed"] >= self.max_orders_per_run:
                result["skipped"] += 1
                continue

            active_count = await self._active_order_count()
            if active_count >= self.max_active_orders:
                await self._update_candidate_status(row["id"], "skipped", f"Active order cap reached: {active_count}/{self.max_active_orders}")
                result["skipped"] += 1
                continue

            entry = self._num(row["broker_ask"]) or self._num(row["broker_last"]) or self._num(row["planned_entry"])
            if not entry or entry <= 0:
                await self._update_candidate_status(row["id"], "failed", "No usable broker entry price")
                result["errors"] += 1
                continue

            quantity = int(self.max_notional // entry)
            if quantity <= 0:
                await self._update_candidate_status(row["id"], "skipped", f"Price {entry:.2f} too high for max notional {self.max_notional:.2f}")
                result["skipped"] += 1
                continue

            stop = self._num(row["planned_stop_loss"])
            target = self._num(row["planned_take_profit"])
            if not stop:
                stop = round(entry * 0.97, 2)
            if not target:
                target = round(entry * 1.05, 2)

            await self.db.execute("""
                UPDATE stock_auto_trade_candidates
                SET planned_entry = $2, planned_stop_loss = $3, planned_take_profit = $4,
                    planned_quantity = $5, updated_at = NOW()
                WHERE id = $1
            """, row["id"], entry, stop, target, quantity)

            if self.dry_run:
                await self._update_candidate_status(row["id"], "dry_run", f"Would place buy limit qty={quantity} entry={entry:.2f} SL={stop:.2f} TP={target:.2f}")
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
                    response.get("db_order_id"), "Order submitted by auto trader" if status == "order_submitted" else response.get("error"),
                )
                if status == "order_submitted":
                    result["placed"] += 1
                else:
                    result["errors"] += 1
            except Exception as exc:
                await self._update_candidate_status(row["id"], "failed", str(exc))
                result["errors"] += 1

        return result

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
            "breakeven_enabled": True,
            "breakeven_trigger_usd": 10,
        }
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if response.status >= 400:
                    data.setdefault("error", f"Order API returned {response.status}")
                return data

    async def _active_order_count(self) -> int:
        value = await self.db.fetchval("""
            SELECT
              (
                SELECT COUNT(*)
                FROM stock_orders
                WHERE status = 'submitted'
                  AND created_at::date = CURRENT_DATE
              )
              +
              (
                SELECT COUNT(*)
                FROM broker_trades
                WHERE status = 'open'
              )
        """)
        return int(value or 0)

    def _reject_reason(self, row: Optional[Dict[str, Any]]) -> Optional[str]:
        if not row:
            return "Missing from refreshed Top 20 Day Trades"
        if row.get("signal_type") and row.get("signal_type") != "BUY":
            return f"Signal is not BUY: {row.get('signal_type')}"
        if row.get("pm_direction") and row.get("pm_direction") != "BUY":
            return f"Premarket direction is not BUY: {row.get('pm_direction')}"
        if row.get("pm_status") in ("Stale PM", "No PM data", "PM against", "PM fading"):
            return f"Bad PM status: {row.get('pm_status')}"
        if row.get("order_bias") in ("Avoid", "Avoid spread", "Refresh", "Wait"):
            return f"Bad order bias: {row.get('order_bias')}"
        spread = self._num(row.get("broker_spread_pct"))
        if spread is None:
            return "Missing broker spread"
        if spread > self.max_spread_pct:
            return f"Spread {spread:.3f}% > max {self.max_spread_pct:.3f}%"
        score = self._num(row.get("candidate_score")) or 0
        if score < self.min_score:
            return f"Score {score:.1f} < min {self.min_score:.1f}"
        rvol = self._num(row.get("relative_volume")) or 0
        if rvol < self.min_relative_volume:
            return f"RVOL {rvol:.2f} < min {self.min_relative_volume:.2f}"
        ask = self._num(row.get("broker_ask"))
        if ask is None or ask <= 0:
            return "Missing broker ask"
        return None

    async def _update_candidate_status(self, candidate_id: int, status: str, reason: str):
        await self.db.execute("""
            UPDATE stock_auto_trade_candidates
            SET status = $2, reason = $3, updated_at = NOW()
            WHERE id = $1
        """, candidate_id, status, reason)

    async def _mark_run_status(self, run_id: int, status: str):
        await self.db.execute("""
            UPDATE stock_auto_trade_runs
            SET status = $2,
                completed_at = CASE WHEN $2 IN ('completed', 'disabled') THEN NOW() ELSE completed_at END,
                updated_at = NOW()
            WHERE id = $1
        """, run_id, status)

    def _minutes_after_open(self, now_et: datetime) -> Optional[int]:
        if now_et.weekday() >= 5:
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
