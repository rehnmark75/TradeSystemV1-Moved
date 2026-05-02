"""
LPF Gate — opt-in strategy-side Loss Prevention Filter helper.

Usage in any strategy's detect_signal, just before returning:

    from .lpf_gate import apply_lpf_gate          # relative import
    # or
    from forex_scanner.core.strategies.lpf_gate import apply_lpf_gate

    signal = apply_lpf_gate(signal, self.logger)
    return signal  # returns None when blocked in 'block' mode

The gate is a no-op when:
  - LossPreventionFilter is unavailable (import error)
  - LPF is disabled in the database config
  - Per-pair LPF is disabled

Signal annotation (always written, never omitted):
  signal['lpf_penalty']        – float
  signal['lpf_would_block']    – bool
  signal['lpf_triggered_rules'] – list[str]

These keys are read by TradeValidator._save_alert_history to populate
the alert_history columns lpf_penalty / lpf_would_block.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton — one LPF instance shared across all strategies in-process.
# Instantiation is deferred to first call so that import errors at module
# load time don't break strategy loading.
# ---------------------------------------------------------------------------
_lpf_instance = None
_lpf_available: Optional[bool] = None  # None = not yet attempted


def _get_lpf():
    """Return a cached LossPreventionFilter or None if unavailable."""
    global _lpf_instance, _lpf_available
    if _lpf_available is False:
        return None
    if _lpf_instance is not None:
        return _lpf_instance

    try:
        try:
            from core.trading.loss_prevention_filter import LossPreventionFilter
        except (ImportError, ValueError):
            from forex_scanner.core.trading.loss_prevention_filter import LossPreventionFilter

        _lpf_instance = LossPreventionFilter(backtest_mode=False)
        _lpf_available = True
        logger.debug("LPF gate: LossPreventionFilter loaded (singleton)")
        return _lpf_instance
    except Exception as exc:
        _lpf_available = False
        logger.warning("LPF gate: could not load LossPreventionFilter: %s", exc)
        return None


def apply_lpf_gate(
    signal: Optional[Dict],
    strategy_logger: Optional[logging.Logger] = None,
    backtest_timestamp: Optional[datetime] = None,
) -> Optional[Dict]:
    """Apply the Loss Prevention Filter to *signal*.

    Parameters
    ----------
    signal:
        Raw signal dict produced by the strategy.  If None, returned unchanged.
    strategy_logger:
        The calling strategy's logger.  Falls back to module-level logger.
    backtest_timestamp:
        When called from backtest replay, pass the candle timestamp so the LPF
        evaluates time-based rules (session, hour_utc) against the historical
        time rather than wall-clock now.

    Returns
    -------
    The same *signal* dict (with lpf_* keys annotated) if allowed, or None
    if the LPF blocked the signal in 'block' mode.
    """
    if signal is None:
        return None

    log = strategy_logger or logger
    lpf = _get_lpf()

    if lpf is None or not lpf.is_enabled:
        # Ensure keys are always present so alert_history INSERT never fails.
        signal.setdefault('lpf_penalty', None)
        signal.setdefault('lpf_would_block', False)
        signal.setdefault('lpf_triggered_rules', [])
        return signal

    try:
        lpf_result = lpf.evaluate(signal, signal_timestamp=backtest_timestamp)
    except Exception as exc:
        log.warning("LPF gate: evaluate() raised %s — letting signal through", exc)
        signal.setdefault('lpf_penalty', None)
        signal.setdefault('lpf_would_block', False)
        signal.setdefault('lpf_triggered_rules', [])
        return signal

    # Annotate regardless of outcome
    signal['lpf_penalty'] = lpf_result.get('total_penalty', 0.0)
    signal['lpf_would_block'] = lpf_result.get('decision') in ('would_block', 'blocked')
    signal['lpf_triggered_rules'] = [
        r['rule_name'] for r in lpf_result.get('triggered_rules', [])
    ]

    if not lpf_result.get('allowed', True):
        rules_str = ', '.join(signal['lpf_triggered_rules'])
        epic = signal.get('epic', '?')
        log.warning(
            "[LPF] Blocked %s %s — penalty=%.2f rules=[%s]",
            epic, signal.get('signal_type', ''), signal['lpf_penalty'], rules_str,
        )
        return None

    if signal['lpf_would_block']:
        log.info(
            "[LPF] Would-block %s %s (monitor mode) — penalty=%.2f",
            signal.get('epic', '?'), signal.get('signal_type', ''), signal['lpf_penalty'],
        )

    return signal
