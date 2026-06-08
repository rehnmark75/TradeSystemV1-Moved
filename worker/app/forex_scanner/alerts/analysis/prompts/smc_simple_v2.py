"""Vision prompt for SMC_SIMPLE_V2 forward-test signals."""
import logging
from typing import Dict

from ._helpers import format_price, extract_pair, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """Build the chart-analysis prompt for the stripped-down V2 entry model."""
    try:
        epic = signal.get("epic", "Unknown")
        pair = extract_pair(epic)
        direction = signal.get("signal_type", signal.get("signal", "Unknown"))
        confidence = signal.get("confidence_score", 0)
        entry_price = signal.get("entry_price", signal.get("price", 0))
        stop_loss = signal.get("stop_loss", 0)
        take_profit = signal.get("take_profit", 0)
        risk_pips = signal.get("risk_pips", signal.get("stop_distance", 0))
        reward_pips = signal.get("reward_pips", signal.get("limit_distance", 0))
        timestamp = signal.get("timestamp") or signal.get("signal_timestamp") or "Unknown"
        indicators = signal.get("strategy_indicators", {}) or {}
        v2 = indicators.get("smc_simple_v2", {}) or indicators

        return f"""FOREX SIGNAL ANALYSIS - SMC_SIMPLE_V2 FORWARD TEST

Strategy: SMC_SIMPLE_V2
Purpose: forward-test the new EURUSD entry model before adding more filters.

Signal:
- Pair: {pair}
- Epic: {epic}
- Direction: {direction}
- Timestamp: {timestamp}
- Confidence: {confidence:.2f}
- Entry: {format_price(entry_price)}
- Stop loss: {format_price(stop_loss)} ({risk_pips} pips)
- Take profit: {format_price(take_profit)} ({reward_pips} pips)

Validated launch constraints:
- Only EURUSD should be approved.
- Only BUY/BULL signals should be approved.
- Only UTC hours 07-12 should be approved.
- Fixed SL/TP should be approximately 5/6 pips.
- This is a market-order rejection-break entry model, not the old SMC_SIMPLE scalp/VSL model.

V2 entry evidence:
- Entry model: {v2.get('entry_model', signal.get('entry_model', 'REJECTION_BREAK'))}
- Structure direction: {v2.get('structure_direction', signal.get('v2_structure_direction', 'Unknown'))}
- Trigger level: {format_price(v2.get('trigger_level', signal.get('trigger_level', 0)))}
- Rejection candle: {v2.get('rejection_candle', signal.get('rejection_candle', 'Unknown'))}
- Minimum signal gap: {v2.get('min_signal_gap_minutes', signal.get('min_signal_gap_minutes', 60))} minutes

Decision rules:
APPROVE only when the chart supports a clean EURUSD bullish rejection-break setup near the current price. Do not reject just because this lacks the older SMC_SIMPLE 1H/5m/1m confluence stack; that stack is intentionally removed in V2.

REJECT if any of these are true:
- Pair is not EURUSD.
- Direction is not BUY/BULL.
- The entry appears very late after the rejection break.
- Price is immediately below obvious resistance with too little room for the 6 pip target.
- The chart shows a clear bearish reversal or invalidated rejection structure.

Respond with exactly:
DECISION: APPROVE or REJECT
CONFIDENCE: 0-100
REASON: one concise sentence
REASON_CODE: V2_OK, V2_LATE_ENTRY, V2_RESISTANCE_BLOCK, V2_WRONG_PAIR, V2_WRONG_DIRECTION, V2_STRUCTURE_INVALID, or V2_OTHER
"""
    except Exception as e:
        logger.error(f"Error building SMC_SIMPLE_V2 prompt: {e}")
        return build_fallback_prompt(signal)
