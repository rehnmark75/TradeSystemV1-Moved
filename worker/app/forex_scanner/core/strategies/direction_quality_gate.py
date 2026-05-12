"""Directional quality gate for SMC_SIMPLE.

This gate captures pair-specific asymmetric behavior found in stored backtests:

* EURUSD BULL loses heavily during late New York and when it chases MACD.
* EURUSD BEAR behaves better with MACD momentum, with an evening exception.
* EURJPY BULL is sensitive to a weak small-negative MACD histogram band.
* EURJPY BEAR performs best when MACD histogram is mildly positive.

The gate is config-driven and supports MONITORING or ACTIVE mode. Unlike the
adaptive bucket gate, this one also runs in backtests because it does not depend
on future outcomes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _cfg(config: Any, key: str, default: Any, epic: str = "") -> Any:
    pair_overrides = getattr(config, "_pair_overrides", {}) if config else {}
    override = pair_overrides.get(epic, {}) if epic else {}
    if key in override and override[key] is not None:
        return override[key]
    params = override.get("parameter_overrides", {})
    if key in params:
        return params[key]
    return getattr(config, key, default) if config else default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "active", "block"}
    return bool(value)


def _float(signal: Dict[str, Any], key: str) -> Optional[float]:
    value = signal.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _signal_hour(signal: Dict[str, Any], signal_timestamp: Optional[datetime]) -> int:
    if signal_timestamp is not None:
        return int(signal_timestamp.hour)
    ts = signal.get("timestamp") or signal.get("market_timestamp")
    if isinstance(ts, datetime):
        return int(ts.hour)
    return datetime.utcnow().hour


def _targeted(epic: str, target_epics: Any) -> bool:
    if not target_epics:
        return epic == "CS.D.EURUSD.CEEM.IP"
    if isinstance(target_epics, str):
        targets = [item.strip() for item in target_epics.split(",") if item.strip()]
    else:
        targets = list(target_epics)
    return epic in targets


def _instrument_label(epic: str) -> str:
    if "EURUSD" in epic:
        return "EURUSD"
    if "EURJPY" in epic:
        return "EURJPY"
    return epic or "instrument"


def evaluate_direction_quality(
    signal: Dict[str, Any],
    config: Any,
    *,
    signal_timestamp: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (passed, reason, details) for a signal."""
    epic = str(signal.get("epic", ""))
    direction = str(signal.get("signal_type") or signal.get("signal") or signal.get("direction") or "").upper()
    hour = _signal_hour(signal, signal_timestamp)

    target_epics = _cfg(config, "direction_quality_gate_target_epics", "CS.D.EURUSD.CEEM.IP", epic)
    if not _targeted(epic, target_epics):
        return True, "not_targeted", {"hour": hour, "direction": direction}

    rsi = _float(signal, "rsi")
    ema21 = _float(signal, "ema_21")
    ema50 = _float(signal, "ema_50")
    macd_hist = _float(signal, "macd_histogram")

    details = {
        "hour": hour,
        "direction": direction,
        "rsi": rsi,
        "ema_21": ema21,
        "ema_50": ema50,
        "macd_histogram": macd_hist,
    }

    if direction == "BULL":
        label = _instrument_label(epic)
        block_start = int(_cfg(config, "direction_quality_bull_block_start_hour", 15, epic))
        block_end = int(_cfg(config, "direction_quality_bull_block_end_hour", 18, epic))
        rsi_min = float(_cfg(config, "direction_quality_bull_rsi_min", 50.0, epic))
        rsi_max = float(_cfg(config, "direction_quality_bull_rsi_max", 70.0, epic))
        min_conf_raw = _cfg(config, "direction_quality_bull_min_confidence", 0.60, epic)
        min_conf = float(min_conf_raw) if min_conf_raw not in (None, "") else None
        require_ema = _bool(_cfg(config, "direction_quality_bull_require_ema21_gt_ema50", True, epic))
        macd_mode = str(_cfg(config, "direction_quality_bull_macd_mode", "pullback", epic)).lower()
        macd_band = float(_cfg(config, "direction_quality_macd_small_band", 0.01, epic))

        if block_start <= hour <= block_end:
            return False, f"{label} BULL blocked during weak {block_start}-{block_end} UTC bucket", details
        if min_conf is not None and float(signal.get("confidence_score", signal.get("confidence", 0.0)) or 0.0) < min_conf:
            return False, f"{label} BULL confidence below {min_conf:.0%} quality floor", details
        if rsi is None or not (rsi_min <= rsi < rsi_max):
            return False, f"{label} BULL RSI {rsi} outside [{rsi_min:.0f},{rsi_max:.0f}) quality band", details
        if require_ema and (ema21 is None or ema50 is None or ema21 <= ema50):
            return False, f"{label} BULL requires EMA21 > EMA50", details
        if macd_mode == "pullback" and (macd_hist is None or macd_hist >= 0):
            return False, f"{label} BULL requires MACD pullback histogram < 0", details
        if macd_mode == "aligned" and (macd_hist is None or macd_hist <= 0):
            return False, f"{label} BULL requires MACD histogram > 0", details
        if macd_mode == "not_small_negative" and (
            macd_hist is None or -macd_band <= macd_hist < 0
        ):
            return False, f"{label} BULL blocks weak small-negative MACD histogram band", details

    elif direction == "BEAR":
        label = _instrument_label(epic)
        block_start_raw = _cfg(config, "direction_quality_bear_block_start_hour", 14, epic)
        block_end_raw = _cfg(config, "direction_quality_bear_block_end_hour", 17, epic)
        if block_start_raw not in (None, "") and block_end_raw not in (None, ""):
            block_start = int(block_start_raw)
            block_end = int(block_end_raw)
            if block_start <= hour <= block_end:
                return False, f"{label} BEAR blocked during weak {block_start}-{block_end} UTC bucket", details

        macd_mode = str(_cfg(config, "direction_quality_bear_macd_mode", "aligned_or_evening", epic)).lower()
        evening_start = int(_cfg(config, "direction_quality_bear_evening_start_hour", 19, epic))
        macd_band = float(_cfg(config, "direction_quality_macd_small_band", 0.01, epic))
        if macd_mode == "aligned" and (macd_hist is None or macd_hist >= 0):
            return False, f"{label} BEAR requires MACD histogram < 0", details
        if macd_mode == "aligned_or_evening" and hour < evening_start and (macd_hist is None or macd_hist >= 0):
            return False, f"{label} BEAR requires MACD histogram < 0 before {evening_start} UTC", details
        if macd_mode == "small_positive" and (
            macd_hist is None or not (0 <= macd_hist < macd_band)
        ):
            return False, f"{label} BEAR requires mildly positive MACD histogram", details
        if macd_mode == "non_negative" and (macd_hist is None or macd_hist < 0):
            return False, f"{label} BEAR requires non-negative MACD histogram", details

    return True, "passed", details


def apply_direction_quality_gate(
    signal: Optional[Dict[str, Any]],
    config: Any,
    strategy_logger: Optional[logging.Logger] = None,
    *,
    signal_timestamp: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    if signal is None:
        return None

    log = strategy_logger or logger
    epic = str(signal.get("epic", ""))
    enabled = _bool(_cfg(config, "direction_quality_gate_enabled", False, epic))
    if not enabled:
        signal.setdefault("direction_quality_gate_state", "disabled")
        return signal

    mode = str(_cfg(config, "direction_quality_gate_mode", "MONITORING", epic)).upper()
    passed, reason, details = evaluate_direction_quality(signal, config, signal_timestamp=signal_timestamp)

    signal["direction_quality_gate_state"] = "passed" if passed else "blocked"
    signal["direction_quality_gate_mode"] = mode
    signal["direction_quality_gate_reason"] = reason
    signal["direction_quality_gate_details"] = details
    signal["direction_quality_gate_would_block"] = not passed

    if not passed:
        log.warning("[DirectionQualityGate] %s (%s mode)", reason, mode)
        if mode == "ACTIVE":
            return None

    return signal
