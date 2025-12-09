"""
Stop Loss and Take Profit Calculator Service

This module provides clean, reusable functions for calculating stop loss and take profit
levels based on ATR and risk-reward ratios, replacing hardcoded values.
"""
from .ig_risk_utils import get_ema_atr, calculate_dynamic_sl_tp
from config import DEFAULT_RISK_REWARD_RATIO, EPIC_RISK_REWARD_RATIOS
import logging

logger = logging.getLogger(__name__)

async def calculate_trade_levels(epic: str, trading_headers: dict,
                               risk_reward_ratio: float = None) -> dict:
    """
    Calculate optimal stop loss and take profit levels using ATR-based analysis.

    Args:
        epic: IG market epic (e.g., "CS.D.EURUSD.CEEM.IP")
        trading_headers: IG API authentication headers
        risk_reward_ratio: Desired reward-to-risk ratio (uses epic-specific or default if None)

    Returns:
        dict: {
            "stopDistance": int,     # Stop distance in points
            "limitDistance": int,    # Take profit distance in points
            "atr": float,           # Current ATR value
            "calculationMethod": str # Method used for calculation
        }
    """
    # Use epic-specific risk-reward ratio if not provided
    if risk_reward_ratio is None:
        risk_reward_ratio = EPIC_RISK_REWARD_RATIOS.get(epic, DEFAULT_RISK_REWARD_RATIO)

    try:
        # Get ATR for the instrument
        atr = await get_ema_atr(epic, trading_headers)
        logger.info(f"📊 ATR for {epic}: {atr}")

        # Calculate dynamic SL/TP based on ATR and market constraints
        sl_tp_data = await calculate_dynamic_sl_tp(epic, trading_headers, atr, risk_reward_ratio)

        stop_distance = int(float(sl_tp_data["stopDistance"]))
        limit_distance = int(float(sl_tp_data["limitDistance"]))

        logger.info(f"📏 {epic} ATR-based levels: SL={stop_distance}, TP={limit_distance} (RR={risk_reward_ratio})")

        return {
            "stopDistance": stop_distance,
            "limitDistance": limit_distance,
            "atr": atr,
            "calculationMethod": "ATR-based with market constraints"
        }

    except Exception as e:
        # Fallback to conservative fixed levels if ATR calculation fails
        logger.warning(f"⚠️ ATR calculation failed for {epic}: {e}")
        logger.info(f"🔄 Using fallback calculation for {epic}")

        # Conservative fallback based on instrument type
        if "JPY" in epic:
            fallback_stop = 30  # ~30 points for JPY pairs
            fallback_limit = int(fallback_stop * risk_reward_ratio)
        else:
            fallback_stop = 25  # ~25 points for other pairs
            fallback_limit = int(fallback_stop * risk_reward_ratio)

        logger.info(f"📏 {epic} Fallback levels: SL={fallback_stop}, TP={fallback_limit}")

        return {
            "stopDistance": fallback_stop,
            "limitDistance": fallback_limit,
            "atr": None,
            "calculationMethod": "Fallback (ATR unavailable)"
        }

def validate_sl_tp_levels(stop_distance: int, limit_distance: int,
                         min_distance: int = None) -> dict:
    """
    Validate and adjust SL/TP levels to meet broker requirements.

    Args:
        stop_distance: Proposed stop distance in points
        limit_distance: Proposed limit distance in points
        min_distance: Minimum distance required by broker

    Returns:
        dict: {
            "stopDistance": int,      # Adjusted stop distance
            "limitDistance": int,     # Adjusted limit distance
            "adjustments": list       # List of adjustments made
        }
    """
    adjustments = []

    # Ensure minimum distance requirements (respect broker minimum exactly)
    if min_distance and stop_distance < min_distance:
        old_stop = stop_distance
        stop_distance = int(min_distance)  # Use broker minimum exactly, no buffer
        adjustments.append(f"Stop distance increased from {old_stop} to {stop_distance} (broker min: {min_distance})")

    # Ensure minimum distance requirements for limit
    if min_distance and limit_distance < min_distance:
        old_limit = limit_distance
        limit_distance = int(min_distance)  # Use broker minimum exactly, no buffer
        adjustments.append(f"Limit distance increased from {old_limit} to {limit_distance} (broker min: {min_distance})")

    return {
        "stopDistance": stop_distance,
        "limitDistance": limit_distance,
        "adjustments": adjustments
    }