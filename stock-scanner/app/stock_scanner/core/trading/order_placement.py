"""
Direct RoboMarkets protected-order placement.

Python port of trading-ui's /api/orders/place route (trading-ui/app/api/
orders/place/route.ts). The auto-trader previously POSTed to that route,
which executes with trading-ui's OWN broker credentials — after the Jul 1
2026 live→demo split that silently kept auto-trader order flow on the live
account. This module reproduces the route's placement semantics with
caller-supplied credentials:

- broker level rounding with escalating price steps (0.01 → 0.05 → 0.1)
  and limit-price offsets, retrying only on level-shaped broker errors
- SL/TP applied via PUT after creation (the broker ignores SL/TP sent on
  POST /orders) — on the pending order for limits, on the deal for fills
- fail-closed protection: if SL/TP cannot be attached, the order is
  cancelled (or the filled deal closed) rather than left naked
- audit row in stock_orders

Not ported: the trade_ready gate (auto-trader always overrides) and the
stock_breakeven_monitors registration (auto-trader sends
breakeven_enabled=False since Jun 11 2026 — BE-off dominates).

The returned dict matches the route's JSON response shape so existing
callers (_is_price_too_close_response, order_response JSONB logging) work
unchanged: status is "submitted" or "rejected", with robomarkets_order_id,
db_order_id, error, level_adjustment_attempts, etc.
"""

import asyncio
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger("order_placement")


def resolve_trading_credentials() -> Tuple[str, str]:
    """
    Credentials for the TRADING account — order flow, broker_trades sync,
    stale-order guardian, breakeven management.

    ROBOMARKETS_TRADE_API_KEY / ROBOMARKETS_TRADE_ACCOUNT_ID take precedence
    over the container-wide ROBOMARKETS_* vars. This split exists because
    stock-scanner keeps ROBOMARKETS_* on the live account for market data,
    while all trade-side operations must target the demo forward-test account
    (Jul 2026). Fail-closed: if either TRADE var is present in the
    environment, BOTH are read from the TRADE vars only — an empty value then
    fails the caller's credential check rather than falling back to the
    (live) container-wide creds.
    """
    if "ROBOMARKETS_TRADE_API_KEY" in os.environ or "ROBOMARKETS_TRADE_ACCOUNT_ID" in os.environ:
        return (
            os.environ.get("ROBOMARKETS_TRADE_API_KEY", ""),
            os.environ.get("ROBOMARKETS_TRADE_ACCOUNT_ID", ""),
        )
    return (
        os.environ.get("ROBOMARKETS_API_KEY", ""),
        os.environ.get("ROBOMARKETS_ACCOUNT_ID", ""),
    )

DEFAULT_API_URL = "https://api.stockstrader.com/api/v1"
PRICE_STEP_ATTEMPTS = [0.01, 0.05, 0.1]
LIMIT_PRICE_OFFSET_ATTEMPTS_PCT = [0, 0.1, 0.25]
MAX_LEVEL_ATTEMPTS = 3  # Initial broker-rounded level + max two retries.
REQUEST_TIMEOUT_SECONDS = 30

_LEVEL_ERROR_TOKENS = ("stop", "take", "profit", "loss", "level", "price",
                       "incorrect", "invalid", "distance", "step")


def _round_to_step(value: float, step: float, direction: str) -> float:
    factor = value / step
    if direction == "up":
        rounded = math.ceil(factor) * step
    elif direction == "down":
        rounded = math.floor(factor) * step
    else:
        rounded = math.floor(factor + 0.5) * step
    return round(rounded, 4)


def _level_changed(a: Optional[float], b: Optional[float]) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(a - b) > 0.00001


def _is_level_error(message: Optional[str]) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(token in lowered for token in _LEVEL_ERROR_TOKENS)


def _build_level_attempts(
    side: str,
    order_type: str,
    price: Optional[float],
    stop_loss: float,
    take_profit: Optional[float],
) -> List[Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    seen = set()
    sl_direction = "down" if side == "buy" else "up"
    tp_direction = "up" if side == "buy" else "down"

    for i, step in enumerate(PRICE_STEP_ATTEMPTS):
        price_offset_pct = (
            LIMIT_PRICE_OFFSET_ATTEMPTS_PCT[i]
            if order_type == "limit" and i < len(LIMIT_PRICE_OFFSET_ATTEMPTS_PCT)
            else 0
        )
        rounded_price = None
        if order_type == "limit" and price:
            adjusted = price * (
                1 - price_offset_pct / 100 if side == "buy" else 1 + price_offset_pct / 100
            )
            rounded_price = round(
                _round_to_step(adjusted, step, "down" if side == "buy" else "up"), 4
            )

        attempt = {
            "price": rounded_price,
            "stop_loss": _round_to_step(stop_loss, step, sl_direction),
            "take_profit": (
                _round_to_step(take_profit, step, tp_direction)
                if take_profit and take_profit > 0 else None
            ),
            "step": step,
            "price_offset_pct": price_offset_pct,
            "label": (
                f"{step:.2f} rounded, limit {price_offset_pct:.2f}% "
                f"{'below' if side == 'buy' else 'above'} request"
                if price_offset_pct > 0 else f"{step:.2f} rounded"
            ),
        }

        if attempt["stop_loss"] <= 0 or (
            attempt["take_profit"] is not None and attempt["take_profit"] <= 0
        ):
            continue
        if side == "buy" and attempt["price"] and attempt["stop_loss"] >= attempt["price"]:
            continue
        if side == "sell" and attempt["price"] and attempt["stop_loss"] <= attempt["price"]:
            continue

        key = (attempt["price"], attempt["stop_loss"], attempt["take_profit"])
        if key not in seen:
            seen.add(key)
            attempts.append(attempt)

    return attempts[:MAX_LEVEL_ATTEMPTS]


class _Broker:
    """Thin form-encoded HTTP wrapper matching the route's broker calls."""

    def __init__(self, session: aiohttp.ClientSession, api_url: str, api_key: str, account_id: str):
        self._session = session
        self._base = f"{api_url}/accounts/{account_id}"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

    async def _read(self, response: aiohttp.ClientResponse) -> tuple:
        text = await response.text()
        try:
            import json as _json
            data = _json.loads(text)
        except ValueError:
            data = {}
        return data, text

    def _error_message(self, data: Dict, text: str, status: int) -> str:
        return str(
            data.get("msg") or data.get("message") or data.get("error")
            or text or f"Broker returned {status}"
        )

    async def submit_order(
        self, broker_ticker: str, side: str, order_type: str, quantity: float,
        attempt: Dict[str, Any],
    ) -> Dict[str, Any]:
        form = {
            "ticker": broker_ticker,
            "side": side,
            "type": order_type,
            "volume": str(int(quantity)),
            "stop_loss": str(attempt["stop_loss"]),
        }
        if attempt["price"] is not None and order_type == "limit":
            form["price"] = str(attempt["price"])
        if attempt["take_profit"]:
            form["take_profit"] = str(attempt["take_profit"])

        async with self._session.post(
            f"{self._base}/orders", headers=self._headers, data=form
        ) as response:
            data, text = await self._read(response)
            if response.status >= 400 or data.get("code") == "error":
                return {
                    "ok": False,
                    "order_id": None,
                    "error": self._error_message(data, text, response.status),
                }
            inner = data.get("data") or {}
            return {
                "ok": True,
                "order_id": str(
                    inner.get("order_id") or inner.get("orderId") or data.get("orderId") or ""
                ),
                "error": None,
            }

    async def put_sl_tp(self, target: str, resource_id: str, attempt: Dict[str, Any]) -> Optional[str]:
        """PUT SL/TP on 'orders' or 'deals'. Returns None on success, error message on failure."""
        form = {"stop_loss": str(attempt["stop_loss"])}
        if attempt["take_profit"]:
            form["take_profit"] = str(attempt["take_profit"])
        async with self._session.put(
            f"{self._base}/{target}/{resource_id}", headers=self._headers, data=form
        ) as response:
            if response.status < 400:
                return None
            data, text = await self._read(response)
            return self._error_message(data, text, response.status)

    async def delete(self, target: str, resource_id: str) -> Optional[str]:
        """DELETE an order or deal. Returns None on success, error message on failure."""
        async with self._session.delete(
            f"{self._base}/{target}/{resource_id}", headers=self._headers
        ) as response:
            if response.status < 400:
                return None
            data, text = await self._read(response)
            return self._error_message(data, text, response.status)

    async def find_latest_open_deal(self, broker_ticker: str) -> tuple:
        """Returns (deal_id or None, error message or None)."""
        async with self._session.get(f"{self._base}/deals", headers=self._headers) as response:
            data, text = await self._read(response)
            if response.status >= 400:
                return None, self._error_message(data, text, response.status)
            raw = data if isinstance(data, list) else (data.get("data") or [])
            matching = [
                d for d in raw
                if d.get("ticker") == broker_ticker
                and (not d.get("status") or d.get("status") == "open")
            ]
            matching.sort(key=lambda d: str(d.get("id")), reverse=True)
            return (str(matching[0]["id"]) if matching else None), None


async def _resolve_broker_ticker(db, ticker: str) -> str:
    exchange = await db.fetchval(
        "SELECT COALESCE(exchange, 'NASDAQ') FROM stock_instruments WHERE ticker = $1 LIMIT 1",
        ticker,
    )
    suffix = ".ny" if exchange and "NYSE" in str(exchange).upper() else ".nq"
    return f"{ticker}{suffix}"


async def place_protected_order(
    db,
    *,
    api_key: str,
    account_id: str,
    ticker: str,
    side: str,
    order_type: str,
    quantity: int,
    price: Optional[float],
    stop_loss: float,
    take_profit: Optional[float] = None,
    signal_id: Optional[int] = None,
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    if side not in ("buy", "sell"):
        return {"status": "rejected", "error": "side must be 'buy' or 'sell'"}
    if order_type not in ("market", "limit"):
        return {"status": "rejected", "error": "order_type must be 'market' or 'limit'"}
    if not quantity or quantity <= 0:
        return {"status": "rejected", "error": "quantity must be positive"}
    if not stop_loss or stop_loss <= 0:
        return {"status": "rejected", "error": "stop_loss is required and must be positive"}
    if order_type == "limit" and (not price or price <= 0):
        return {"status": "rejected", "error": "price is required for limit orders"}
    if side == "buy" and price and stop_loss >= price:
        return {"status": "rejected", "error": "stop_loss must be below price for buy orders"}
    if not api_key or not account_id:
        return {"status": "rejected", "error": "Broker credentials not configured"}

    broker_ticker = await _resolve_broker_ticker(db, ticker)
    level_attempts = _build_level_attempts(side, order_type, price, stop_loss, take_profit)
    if not level_attempts:
        return {"status": "rejected", "error": "No valid broker level attempts could be built"}

    broker_order_id: Optional[str] = None
    broker_deal_id: Optional[str] = None
    order_status = "submitted"
    error_message: Optional[str] = None
    accepted = level_attempts[0]
    adjustment_attempts: List[Dict[str, Any]] = []
    sl_tp_applied = False
    sl_tp_error: Optional[str] = None
    protection_action = {"attempted": False, "action": "none", "ok": False, "error": None}

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        broker = _Broker(session, api_url, api_key, account_id)

        try:
            for attempt in level_attempts:
                submit = await broker.submit_order(broker_ticker, side, order_type, quantity, attempt)
                if submit["ok"]:
                    accepted = attempt
                    broker_order_id = submit["order_id"]
                    break
                adjustment_attempts.append({
                    "stage": "submit",
                    "label": attempt["label"],
                    "stop_loss": attempt["stop_loss"],
                    "take_profit": attempt["take_profit"],
                    "error": submit["error"],
                })
                error_message = submit["error"]
                if not _is_level_error(submit["error"]):
                    break
            if not broker_order_id:
                order_status = "rejected"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            order_status = "rejected"
            error_message = f"Network error contacting broker: {exc}"

        # The broker ignores SL/TP sent during POST /orders creation; apply via
        # PUT — on the pending order for limits (carries over on fill), on the
        # deal for fills.
        if order_status == "submitted" and broker_order_id:
            ordered_apply = [accepted] + [
                a for a in level_attempts
                if _level_changed(a["stop_loss"], accepted["stop_loss"])
                or _level_changed(a["take_profit"], accepted["take_profit"])
            ]

            async def apply_sl_tp(target: str, resource_id: str) -> tuple:
                """Returns (ok, accepted_attempt, error)."""
                for attempt in ordered_apply:
                    message = await broker.put_sl_tp(target, resource_id, attempt)
                    if message is None:
                        return True, attempt, None
                    adjustment_attempts.append({
                        "stage": "apply",
                        "label": attempt["label"],
                        "stop_loss": attempt["stop_loss"],
                        "take_profit": attempt["take_profit"],
                        "error": message,
                    })
                    if not _is_level_error(message):
                        return False, attempt, message
                last = ordered_apply[-1] if ordered_apply else accepted
                return False, last, "Broker rejected all SL/TP level attempts"

            try:
                if order_type == "limit":
                    sl_tp_applied, accepted, sl_tp_error = await apply_sl_tp("orders", broker_order_id)
                    if not sl_tp_applied:
                        logger.error("Failed to set SL/TP on order %s: %s", broker_order_id, sl_tp_error)

                    # A marketable limit can fill before order-level SL/TP is
                    # attached — if a deal already exists, protect it directly.
                    await asyncio.sleep(1.2)
                    deal_id, find_error = await broker.find_latest_open_deal(broker_ticker)
                    if find_error:
                        sl_tp_error = sl_tp_error or find_error
                    if deal_id:
                        broker_deal_id = deal_id
                        sl_tp_applied, accepted, sl_tp_error = await apply_sl_tp("deals", deal_id)
                        if not sl_tp_applied:
                            logger.error("Failed to set SL/TP on immediate-fill deal %s: %s", deal_id, sl_tp_error)
                else:
                    await asyncio.sleep(1.5)
                    deal_id, find_error = await broker.find_latest_open_deal(broker_ticker)
                    if find_error:
                        sl_tp_error = sl_tp_error or find_error
                    if deal_id:
                        broker_deal_id = deal_id
                        sl_tp_applied, accepted, sl_tp_error = await apply_sl_tp("deals", deal_id)
                        if not sl_tp_applied:
                            logger.error("Failed to set SL/TP on deal %s: %s", deal_id, sl_tp_error)
                    else:
                        sl_tp_error = sl_tp_error or f"No matching deal found for {broker_ticker} to set SL/TP"
                        logger.warning(sl_tp_error)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                sl_tp_error = f"Failed to apply SL/TP: {exc}"
                logger.error(sl_tp_error)

            if not sl_tp_applied:
                # Fail closed: never leave an unprotected order/position behind.
                if order_type == "limit":
                    cleanup_error = await broker.delete("orders", broker_order_id)
                    protection_action = {
                        "attempted": True, "action": "cancel_order",
                        "ok": cleanup_error is None, "error": cleanup_error,
                    }
                    order_status = "rejected"
                    error_message = (
                        f"SL/TP could not be applied, so broker order {broker_order_id} was cancelled. {sl_tp_error or ''}".strip()
                        if cleanup_error is None else
                        f"SL/TP could not be applied and broker order {broker_order_id} could not be cancelled. Manual broker check required. {cleanup_error or sl_tp_error or ''}".strip()
                    )
                elif broker_deal_id:
                    cleanup_error = await broker.delete("deals", broker_deal_id)
                    protection_action = {
                        "attempted": True, "action": "close_deal",
                        "ok": cleanup_error is None, "error": cleanup_error,
                    }
                    order_status = "rejected"
                    error_message = (
                        f"SL/TP could not be applied, so broker deal {broker_deal_id} was closed. {sl_tp_error or ''}".strip()
                        if cleanup_error is None else
                        f"SL/TP could not be applied and broker deal {broker_deal_id} could not be closed. Manual broker check required. {cleanup_error or sl_tp_error or ''}".strip()
                    )
                else:
                    order_status = "rejected"
                    error_message = (
                        f"SL/TP could not be confirmed and no broker deal id was available to close. "
                        f"Manual broker check required. {sl_tp_error or ''}".strip()
                    )

    db_order_id = await db.fetchval("""
        INSERT INTO stock_orders
            (signal_id, robomarkets_order_id, ticker, order_type, side, quantity,
             price, stop_loss, take_profit, status, error_message)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
    """,
        signal_id, broker_order_id, ticker, order_type, side, float(quantity),
        accepted["price"] or price, accepted["stop_loss"], accepted["take_profit"],
        order_status, error_message or sl_tp_error,
    )

    requested_levels = {
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }

    if order_status == "rejected":
        return {
            "error": error_message,
            "db_order_id": db_order_id,
            "status": "rejected",
            "requested_levels": requested_levels,
            "level_adjustment_attempts": adjustment_attempts,
            "protection_action": protection_action,
        }

    return {
        "robomarkets_order_id": broker_order_id,
        "db_order_id": db_order_id,
        "status": "submitted",
        "ticker": ticker,
        "side": side,
        "order_type": order_type,
        "quantity": quantity,
        "price": accepted["price"] or price,
        "stop_loss": accepted["stop_loss"],
        "take_profit": accepted["take_profit"],
        "sl_tp_applied": sl_tp_applied,
        "sl_tp_error": sl_tp_error,
        "protection_action": protection_action,
        "level_adjusted": (
            _level_changed(stop_loss, accepted["stop_loss"])
            or _level_changed(take_profit, accepted["take_profit"])
            or _level_changed(price, accepted["price"])
        ),
        "requested_levels": requested_levels,
        "broker_levels": {
            "price": accepted["price"] or price,
            "stop_loss": accepted["stop_loss"],
            "take_profit": accepted["take_profit"],
            "step": accepted["step"],
            "label": accepted["label"],
        },
        "level_adjustment_attempts": adjustment_attempts,
    }
