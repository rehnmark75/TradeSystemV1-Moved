#!/usr/bin/env python3
"""
One-shot strategy evaluator for Trading UI manual triggers.

Runs inside task-worker and prints a single JSON object on the final line.
"""

import argparse
import json
import logging
import os
import sys
from datetime import timezone
from decimal import Decimal
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, "/app")

from forex_scanner.configdata import config
from forex_scanner.config import EPIC_MAP as ORDER_EPIC_MAP
from forex_scanner.core.data_fetcher import DataFetcher
from forex_scanner.core.database import DatabaseManager
from forex_scanner.core.strategies.impulse_fade_strategy import ImpulseFadeStrategy
from forex_scanner.core.strategies.mean_reversion_strategy import MeanReversionStrategy
from forex_scanner.core.strategies.range_fade_strategy import create_range_fade_strategy
from forex_scanner.core.strategies.smc_momentum_strategy import SMCMomentumStrategy
from forex_scanner.core.strategies.smc_simple_strategy import create_smc_simple_strategy
from forex_scanner.core.strategies.xau_gold_strategy import create_xau_gold_strategy


LOOKBACK_HOURS = {
    "1m": 24,
    "5m": 48,
    "15m": 168,
    "1h": 240,
    "4h": 900,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _pair_name(epic: str) -> str:
    parts = epic.split(".")
    if len(parts) >= 3:
        return parts[2].replace("_", "")
    return epic


def _latest_timestamp(*frames: Optional[pd.DataFrame]):
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if "start_time" in frame.columns:
            ts = pd.to_datetime(frame["start_time"].iloc[-1], utc=True)
            return ts.to_pydatetime()
        return pd.to_datetime(frame.index[-1], utc=True).to_pydatetime()
    return None


def _frame(frames: Dict[str, Optional[pd.DataFrame]], preferred: str, fallback: str):
    selected = frames.get(preferred)
    if selected is not None and not selected.empty:
        return selected
    return frames.get(fallback)


def _normalize_direction(signal: Dict[str, Any]) -> str:
    raw = (
        signal.get("direction")
        or signal.get("signal_type")
        or signal.get("side")
        or ""
    )
    text = str(raw).upper()
    if "SELL" in text or "SHORT" in text or "BEAR" in text:
        return "SELL"
    return "BUY"


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _latest_price(frames: Dict[str, Optional[pd.DataFrame]]) -> Optional[float]:
    frame = _frame(frames, "5m", "15m")
    if frame is None or frame.empty:
        return None
    latest = frame.iloc[-1]
    return _first_number(
        latest.get("close"),
        latest.get("current_price"),
        latest.get("bid_close"),
        latest.get("ask_close"),
    )


def _manual_distances(config_override: Dict[str, Any]) -> tuple[float, float, str]:
    stop_distance = _first_number(
        config_override.get("manual_stop_distance"),
        config_override.get("manual_stop_distance_pips"),
        config_override.get("stop_distance"),
        config_override.get("stop_loss_pips"),
        config_override.get("fixed_stop_loss_pips"),
        config_override.get("risk_pips"),
    )
    limit_distance = _first_number(
        config_override.get("manual_limit_distance"),
        config_override.get("manual_limit_distance_pips"),
        config_override.get("limit_distance"),
        config_override.get("take_profit_pips"),
        config_override.get("fixed_take_profit_pips"),
        config_override.get("reward_pips"),
    )

    source = "manual override fields"
    if stop_distance is None:
        stop_distance = 15.0
        source = "manual fallback defaults"
    if limit_distance is None:
        limit_distance = stop_distance * 2
        source = "manual fallback defaults"
    return stop_distance, limit_distance, source


def _make_forced_signal(
    args: argparse.Namespace,
    frames: Dict[str, Optional[pd.DataFrame]],
    strategy_name: str,
    rejection_reason: str,
    config_override: Dict[str, Any],
) -> Dict[str, Any]:
    direction = str(args.manual_direction or "BUY").upper()
    if direction not in {"BUY", "SELL"}:
        direction = "BUY"

    stop_distance, limit_distance, source = _manual_distances(config_override)
    return {
        "strategy": strategy_name,
        "epic": args.epic,
        "direction": direction,
        "signal_type": direction,
        "entry_price": _latest_price(frames),
        "current_price": _latest_price(frames),
        "risk_pips": stop_distance,
        "reward_pips": limit_distance,
        "confidence": 0,
        "market_regime": "manual_override",
        "manual_override": True,
        "manual_override_reason": rejection_reason,
        "sl_tp_source": source,
    }


def _make_trade_request(epic: str, signal: Dict[str, Any]) -> Dict[str, Any]:
    order_epic = ORDER_EPIC_MAP.get(epic, epic)
    return {
        "epic": order_epic,
        "direction": _normalize_direction(signal),
        "stop_distance": signal.get("risk_pips") or signal.get("stop_distance"),
        "limit_distance": signal.get("reward_pips") or signal.get("limit_distance"),
        "use_provided_sl_tp": True,
        "alert_id": None,
        "trigger_source": "manual_trigger",
        "signal_price": signal.get("entry_price") or signal.get("current_price"),
    }


def _create_strategy(
    strategy_name: str,
    db_manager: DatabaseManager,
    logger: logging.Logger,
    config_override: Dict[str, Any],
):
    name = strategy_name.upper()
    if name == "SMC_SIMPLE":
        return create_smc_simple_strategy(
            config=config,
            logger=logger,
            db_manager=db_manager,
            config_override=config_override,
        )
    if name == "XAU_GOLD":
        return create_xau_gold_strategy(
            config=config,
            logger=logger,
            db_manager=db_manager,
            config_override=config_override,
        )
    if name == "MEAN_REVERSION":
        return MeanReversionStrategy(
            config=config,
            logger=logger,
            db_manager=db_manager,
            config_override=config_override,
        )
    if name == "RANGE_FADE":
        return create_range_fade_strategy(
            config=config,
            logger=logger,
            db_manager=db_manager,
            config_override=config_override,
        )
    if name == "SMC_MOMENTUM":
        return SMCMomentumStrategy(
            config=config,
            logger=logger,
            db_manager=db_manager,
            config_override=config_override,
        )
    if name == "IMPULSE_FADE":
        return ImpulseFadeStrategy(
            config=None,
            logger=logger,
            db_manager=db_manager,
            config_override=config_override,
        )
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("manual_evaluate")

    config_override = json.loads(args.config_override or "{}")
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
    db_manager = DatabaseManager(db_url)
    data_fetcher = DataFetcher(db_manager)
    strategy = _create_strategy(args.strategy, db_manager, logger, config_override)

    epic = args.epic
    pair = _pair_name(epic)

    frames = {
        timeframe: data_fetcher.get_enhanced_data(
            epic=epic,
            pair=pair,
            timeframe=timeframe,
            lookback_hours=lookback,
        )
        for timeframe, lookback in LOOKBACK_HOURS.items()
    }

    if not any(frame is not None and not frame.empty for frame in frames.values()):
        return {
            "fired": False,
            "rejection_reason": "No candle data available for selected pair.",
        }

    now = _latest_timestamp(
        frames.get("5m"),
        frames.get("15m"),
        frames.get("1h"),
        frames.get("4h"),
    )
    if now and now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    name = args.strategy.upper()
    if name == "SMC_SIMPLE":
        df_trigger = _frame(frames, getattr(strategy, "trigger_tf", "15m"), "15m")
        df_entry = _frame(frames, getattr(strategy, "entry_tf", "5m"), "5m")
        signal = strategy.detect_signal(
            df_trigger=df_trigger,
            df_4h=frames.get("4h"),
            df_entry=df_entry,
            epic=epic,
            pair=pair,
        )
    elif name == "SMC_MOMENTUM":
        signal = strategy.detect_signal(
            df_trigger=frames.get("15m"),
            df_4h=frames.get("4h"),
            df_1h=frames.get("1h"),
            epic=epic,
            pair=pair,
            current_time=now,
        )
    elif name == "IMPULSE_FADE":
        signal = strategy.detect_signal(
            df_trigger=frames.get("5m"),
            df_4h=frames.get("4h"),
            epic=epic,
            pair=pair,
            spread_pips=args.spread_pips,
            current_timestamp=now,
        )
    else:
        signal = strategy.detect_signal(
            df_trigger=_frame(frames, "15m", "5m"),
            df_4h=frames.get("4h"),
            df_entry=frames.get("5m"),
            epic=epic,
            pair=pair,
            current_timestamp=now,
            current_time=now,
        )

    if not signal:
        rejection_reason = "Strategy did not produce a signal on the latest candles."
        if args.force_trade:
            signal = _make_forced_signal(
                args=args,
                frames=frames,
                strategy_name=name,
                rejection_reason=rejection_reason,
                config_override=config_override,
            )
            return {
                "fired": True,
                "forced": True,
                "original_rejection_reason": rejection_reason,
                "signal": signal,
                "trade_request": _make_trade_request(epic, signal),
            }
        return {
            "fired": False,
            "rejection_reason": rejection_reason,
        }

    signal = dict(signal)
    signal.setdefault("strategy", name)
    signal.setdefault("epic", epic)
    return {
        "fired": True,
        "signal": signal,
        "trade_request": _make_trade_request(epic, signal),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epic", required=True)
    parser.add_argument("--strategy", default="SMC_SIMPLE")
    parser.add_argument("--config-override", default="{}")
    parser.add_argument("--spread-pips", type=float, default=1.5)
    parser.add_argument("--force-trade", action="store_true")
    parser.add_argument("--manual-direction", default="BUY")
    args = parser.parse_args()

    try:
        result = evaluate(args)
    except Exception as exc:
        result = {
            "fired": False,
            "rejection_reason": f"Manual evaluation failed: {exc}",
        }
    print(json.dumps(result, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
