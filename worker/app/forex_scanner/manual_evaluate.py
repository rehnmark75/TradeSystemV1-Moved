#!/usr/bin/env python3
"""
Manual signal evaluation CLI.
Called by dev-app/routers/manual_trigger_router.py via docker exec.

Usage:
    python /app/forex_scanner/manual_evaluate.py \
        --epic CS.D.EURUSD.CEEM.IP \
        --strategy SMC_SIMPLE \
        --config-override '{"fixed_stop_loss_pips": 12}' \
        --spread-pips 1.5

Output: single JSON line to stdout.
  {"fired": true,  "signal": {...}, "trade_request": {...}}
  {"fired": false, "rejection_reason": "..."}
"""

import sys
import json
import argparse
import logging
import traceback

# Route all Python logging to stderr so stdout stays clean for JSON output.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

STRATEGY_METHOD = {
    "SMC_SIMPLE":      "detect_smc_simple_signals",
    "XAU_GOLD":        "detect_xau_gold_signals",
    "MEAN_REVERSION":  "detect_mean_reversion_signals",
    "RANGE_FADE":      "detect_range_fade_signals",
    "RANGE_STRUCTURE": "detect_range_structure_signals",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epic", required=True)
    parser.add_argument("--strategy", default="SMC_SIMPLE")
    parser.add_argument("--config-override", default="{}")
    parser.add_argument("--spread-pips", type=float, default=1.5)
    args = parser.parse_args()

    strategy_name = args.strategy.upper()

    try:
        config_override = json.loads(args.config_override)
    except json.JSONDecodeError as exc:
        _out({"fired": False, "rejection_reason": f"Invalid config_override JSON: {exc}"})
        return

    if strategy_name not in STRATEGY_METHOD:
        _out({"fired": False, "rejection_reason": f"Unknown strategy: {strategy_name}. Available: {', '.join(STRATEGY_METHOD)}"})
        return

    try:
        # Config module
        try:
            import config
        except ImportError:
            from forex_scanner import config  # type: ignore

        # Database
        try:
            from core.database import DatabaseManager
        except ImportError:
            from forex_scanner.core.database import DatabaseManager  # type: ignore

        # SignalDetector — accepts config_override which is forwarded to each
        # strategy's lazy-init, giving us the one-shot parameter override.
        try:
            from core.signal_detector import SignalDetector
        except ImportError:
            from forex_scanner.core.signal_detector import SignalDetector  # type: ignore

        db_manager = DatabaseManager(config.DATABASE_URL)

        detector = SignalDetector(
            db_manager=db_manager,
            user_timezone=getattr(config, "USER_TIMEZONE", "Europe/Stockholm"),
            config_override=config_override or None,
        )

        # Pair short name needed by detect_* methods.
        pair_info = getattr(config, "PAIR_INFO", {}).get(args.epic, {})
        parts = args.epic.split(".")
        pair = pair_info.get("pair", parts[2] if len(parts) > 2 else args.epic)

        spread_pips = getattr(config, "SPREAD_PIPS", args.spread_pips)

        method_name = STRATEGY_METHOD[strategy_name]
        detect_fn = getattr(detector, method_name)

        # All detect_* methods share (epic, pair, spread_pips, timeframe) positional args.
        signal = detect_fn(args.epic, pair, spread_pips, "5m")

        if signal is None:
            _out({"fired": False, "rejection_reason": "No signal at current market state"})
        else:
            trade_request = _to_trade_request(signal, args.epic)
            _out({"fired": True, "signal": _serialize(signal), "trade_request": trade_request})

    except Exception as exc:
        _out({
            "fired": False,
            "rejection_reason": f"Evaluation error: {exc}",
            "debug": traceback.format_exc(),
        })


# ────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────

def _out(payload: dict):
    print(json.dumps(payload), flush=True)


def _to_trade_request(signal: dict, epic: str) -> dict:
    direction = signal.get("signal_type") or signal.get("direction") or "BUY"
    if direction in ("BULL", "BULLISH"):
        direction = "BUY"
    elif direction in ("BEAR", "BEARISH"):
        direction = "SELL"

    stop_distance = (
        signal.get("stop_distance")
        or signal.get("risk_pips")
        or signal.get("stop_loss")
    )
    limit_distance = (
        signal.get("limit_distance")
        or signal.get("reward_pips")
        or signal.get("take_profit")
    )

    return {
        "epic": epic,
        "direction": direction,
        "stop_distance": float(stop_distance) if stop_distance is not None else None,
        "limit_distance": float(limit_distance) if limit_distance is not None else None,
        "use_provided_sl_tp": True,
        "alert_id": None,
        "trigger_source": "manual_trigger",
    }


def _serialize(obj):
    """Recursively coerce the signal dict to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    # Pandas / numpy types
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if hasattr(obj, "item"):          # numpy scalar
            return obj.item()
    except ImportError:
        pass
    try:
        return float(obj)
    except (TypeError, ValueError):
        pass
    return str(obj)


if __name__ == "__main__":
    main()
