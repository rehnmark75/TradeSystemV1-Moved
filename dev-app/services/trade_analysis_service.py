"""
Trade Analysis Service - Comprehensive trade outcome analysis

Provides functions for:
- MFE/MAE (Maximum Favorable/Adverse Excursion) calculation
- Exit type classification
- Entry quality assessment
- Exit quality assessment
- Learning insights generation

This service analyzes WHY trades won or lost to enable systematic learning.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
from decimal import Decimal


def get_pip_multiplier(symbol: str) -> int:
    """Get pip multiplier based on currency pair/epic type"""
    if not symbol:
        return 10000
    symbol_upper = symbol.upper()
    # CEEM epics already have prices scaled (e.g., 11739.9 for EURUSD 1.17399)
    if 'CEEM' in symbol_upper:
        return 1
    if 'JPY' in symbol_upper:
        return 100
    return 10000


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return default


def calculate_mfe_mae(
    trade,
    db: Session,
    candle_model,
    timeframe: int = 15
) -> Dict[str, Any]:
    """
    Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    by analyzing candles between entry and exit.

    MFE: Maximum profit the trade reached at any point
    MAE: Maximum drawdown the trade experienced at any point

    Args:
        trade: TradeLog object with entry_price, direction, timestamp, closed_at
        db: Database session
        candle_model: Candle model class (to avoid circular imports)
        timeframe: Candle timeframe in minutes (default 15)

    Returns:
        Dictionary with MFE/MAE data including timestamps and percentages
    """
    entry_price = safe_float(trade.entry_price)
    direction = trade.direction
    symbol = trade.symbol
    pip_multiplier = get_pip_multiplier(symbol)

    # Define time window
    start_time = trade.timestamp
    end_time = trade.closed_at or datetime.utcnow()

    # Fetch candles during the trade
    candles = db.query(candle_model).filter(
        and_(
            candle_model.epic == symbol,
            candle_model.timeframe == timeframe,
            candle_model.start_time >= start_time,
            candle_model.start_time <= end_time
        )
    ).order_by(candle_model.start_time).all()

    if not candles:
        return {
            "mfe_pips": 0.0,
            "mfe_price": entry_price,
            "mfe_timestamp": None,
            "mfe_time_to_peak_minutes": 0,
            "mae_pips": 0.0,
            "mae_price": entry_price,
            "mae_timestamp": None,
            "mae_time_to_trough_minutes": 0,
            "mfe_mae_ratio": 0.0,
            "initial_move": "UNKNOWN",
            "immediate_reversal": False,
            "percentage_of_tp": 0.0,
            "percentage_of_sl": 0.0,
            "candle_count": 0
        }

    mfe_pips = 0.0
    mae_pips = 0.0
    mfe_price = entry_price
    mae_price = entry_price
    mfe_timestamp = None
    mae_timestamp = None

    # Track first 3 candles for immediate reversal detection
    first_candle_moves = []

    for i, candle in enumerate(candles):
        candle_high = safe_float(candle.high)
        candle_low = safe_float(candle.low)

        if direction == "BUY":
            # Favorable: highs above entry
            favorable_pips = (candle_high - entry_price) * pip_multiplier
            favorable_price = candle_high
            # Adverse: lows below entry
            adverse_pips = (entry_price - candle_low) * pip_multiplier
            adverse_price = candle_low
        else:  # SELL
            # Favorable: lows below entry
            favorable_pips = (entry_price - candle_low) * pip_multiplier
            favorable_price = candle_low
            # Adverse: highs above entry
            adverse_pips = (candle_high - entry_price) * pip_multiplier
            adverse_price = candle_high

        # Track first candle moves for initial direction
        if i < 3:
            first_candle_moves.append({
                "favorable": favorable_pips,
                "adverse": adverse_pips
            })

        # Update MFE
        if favorable_pips > mfe_pips:
            mfe_pips = favorable_pips
            mfe_price = favorable_price
            mfe_timestamp = candle.start_time

        # Update MAE
        if adverse_pips > mae_pips:
            mae_pips = adverse_pips
            mae_price = adverse_price
            mae_timestamp = candle.start_time

    # Calculate time to peak/trough
    mfe_time_to_peak = 0
    mae_time_to_trough = 0
    if mfe_timestamp and start_time:
        mfe_time_to_peak = int((mfe_timestamp - start_time).total_seconds() / 60)
    if mae_timestamp and start_time:
        mae_time_to_trough = int((mae_timestamp - start_time).total_seconds() / 60)

    # Determine initial move direction
    if first_candle_moves:
        total_favorable = sum(m["favorable"] for m in first_candle_moves)
        total_adverse = sum(m["adverse"] for m in first_candle_moves)
        if total_favorable > total_adverse + 2:  # 2 pip threshold
            initial_move = "FAVORABLE"
        elif total_adverse > total_favorable + 2:
            initial_move = "ADVERSE"
        else:
            initial_move = "CONSOLIDATION"
    else:
        initial_move = "UNKNOWN"

    # Check for immediate reversal (significant adverse move in first 3 candles)
    immediate_reversal = False
    if first_candle_moves:
        max_early_adverse = max(m["adverse"] for m in first_candle_moves)
        if max_early_adverse > 10:  # More than 10 pips adverse in first 3 candles
            immediate_reversal = True

    # Calculate percentages of TP/SL reached
    tp_distance = 0.0
    sl_distance = 0.0
    if trade.tp_price and trade.sl_price:
        if direction == "BUY":
            tp_distance = (safe_float(trade.tp_price) - entry_price) * pip_multiplier
            sl_distance = (entry_price - safe_float(trade.sl_price)) * pip_multiplier
        else:
            tp_distance = (entry_price - safe_float(trade.tp_price)) * pip_multiplier
            sl_distance = (safe_float(trade.sl_price) - entry_price) * pip_multiplier

    percentage_of_tp = (mfe_pips / tp_distance * 100) if tp_distance > 0 else 0.0
    percentage_of_sl = (mae_pips / sl_distance * 100) if sl_distance > 0 else 0.0

    # Calculate MFE/MAE ratio
    mfe_mae_ratio = mfe_pips / mae_pips if mae_pips > 0 else float('inf') if mfe_pips > 0 else 0.0

    return {
        "mfe_pips": round(mfe_pips, 1),
        "mfe_price": mfe_price,
        "mfe_timestamp": str(mfe_timestamp) if mfe_timestamp else None,
        "mfe_time_to_peak_minutes": mfe_time_to_peak,
        "mae_pips": round(mae_pips, 1),
        "mae_price": mae_price,
        "mae_timestamp": str(mae_timestamp) if mae_timestamp else None,
        "mae_time_to_trough_minutes": mae_time_to_trough,
        "mfe_mae_ratio": round(mfe_mae_ratio, 2) if mfe_mae_ratio != float('inf') else 999.99,
        "initial_move": initial_move,
        "immediate_reversal": immediate_reversal,
        "percentage_of_tp": round(percentage_of_tp, 1),
        "percentage_of_sl": round(percentage_of_sl, 1),
        "candle_count": len(candles),
        "tp_distance_pips": round(tp_distance, 1),
        "sl_distance_pips": round(sl_distance, 1)
    }


def classify_exit_type(trade) -> str:
    """
    Determine how the trade was exited.

    Exit types:
    - TP_HIT: Take profit was reached
    - SL_HIT: Stop loss was hit
    - BREAKEVEN: Exited at entry price (usually after move to BE)
    - TRAILING_STOP: Trailing stop was triggered
    - MANUAL: Manual exit or unknown

    Args:
        trade: TradeLog object

    Returns:
        Exit type string
    """
    exit_price = safe_float(trade.exit_price_calculated) or safe_float(trade.sl_price)
    entry_price = safe_float(trade.entry_price)
    sl_price = safe_float(trade.sl_price)
    tp_price = safe_float(trade.tp_price)
    direction = trade.direction

    if not exit_price or not entry_price:
        return "UNKNOWN"

    # Calculate distances for tolerance
    pip_multiplier = get_pip_multiplier(trade.symbol)
    sl_distance = abs(sl_price - entry_price) * pip_multiplier if sl_price else 0
    tp_distance = abs(tp_price - entry_price) * pip_multiplier if tp_price else 0

    # Tolerance in pips (allow 2 pips variance)
    tolerance_pips = 2

    if direction == "BUY":
        exit_vs_tp_pips = (exit_price - tp_price) * pip_multiplier if tp_price else 999
        exit_vs_sl_pips = (sl_price - exit_price) * pip_multiplier if sl_price else 999
        exit_vs_entry_pips = (exit_price - entry_price) * pip_multiplier

        # Check TP hit (within tolerance of TP)
        if tp_price and exit_vs_tp_pips >= -tolerance_pips:
            return "TP_HIT"
        # Check SL hit (within tolerance of SL)
        elif sl_price and exit_vs_sl_pips >= -tolerance_pips:
            return "SL_HIT"
        # Check breakeven (within 2 pips of entry after move to BE)
        elif trade.moved_to_breakeven and abs(exit_vs_entry_pips) < tolerance_pips:
            return "BREAKEVEN"
        # Trailing stop (moved to BE and exited in profit)
        elif trade.moved_to_breakeven and exit_vs_entry_pips > tolerance_pips:
            return "TRAILING_STOP"
        # Trailing stop (exited above SL but below entry after BE)
        elif trade.moved_to_breakeven:
            return "TRAILING_STOP"
    else:  # SELL
        exit_vs_tp_pips = (tp_price - exit_price) * pip_multiplier if tp_price else 999
        exit_vs_sl_pips = (exit_price - sl_price) * pip_multiplier if sl_price else 999
        exit_vs_entry_pips = (entry_price - exit_price) * pip_multiplier

        # Check TP hit
        if tp_price and exit_vs_tp_pips >= -tolerance_pips:
            return "TP_HIT"
        # Check SL hit
        elif sl_price and exit_vs_sl_pips >= -tolerance_pips:
            return "SL_HIT"
        # Check breakeven
        elif trade.moved_to_breakeven and abs(exit_vs_entry_pips) < tolerance_pips:
            return "BREAKEVEN"
        # Trailing stop
        elif trade.moved_to_breakeven and exit_vs_entry_pips > tolerance_pips:
            return "TRAILING_STOP"
        elif trade.moved_to_breakeven:
            return "TRAILING_STOP"

    return "MANUAL"


def assess_entry_quality(
    alert,
    strategy_indicators: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Score the quality of the entry signal based on multiple factors.

    Scoring (out of 100):
    - Confidence score: 0-30 points
    - HTF alignment: 0-20 points
    - Pattern detection: 0-15 points
    - S/R level: 0-15 points
    - R:R ratio: 0-20 points

    Args:
        alert: AlertHistory object (can be None)
        strategy_indicators: Parsed strategy_indicators JSON

    Returns:
        Dictionary with score, factors breakdown, and verdict
    """
    score = 0.0
    factors = []

    if not alert and not strategy_indicators:
        return {
            "score": 0,
            "max_score": 100,
            "percentage": 0,
            "factors": ["No signal data available"],
            "verdict": "UNKNOWN"
        }

    # 1. Confidence score contribution (0-30 points)
    conf = safe_float(alert.confidence_score if alert else strategy_indicators.get('confidence', 0))
    conf_points = conf * 30
    score += conf_points
    factors.append({
        "name": "Confidence Score",
        "value": f"{conf*100:.0f}%",
        "points": round(conf_points, 1),
        "max_points": 30
    })

    # Detect strategy type - SMC_SIMPLE uses tier1_ema, SMC_STRUCTURE uses htf_data/bos_choch
    is_smc_simple = 'tier1_ema' in strategy_indicators
    tier1_ema = strategy_indicators.get('tier1_ema', {}) or {}
    tier2_swing = strategy_indicators.get('tier2_swing', {}) or {}
    tier3_entry = strategy_indicators.get('tier3_entry', {}) or {}
    risk_management = strategy_indicators.get('risk_management', {}) or {}
    dataframe_analysis = strategy_indicators.get('dataframe_analysis', {}) or {}

    # 2. HTF alignment (0-20 points)
    # SMC_SIMPLE: tier1_ema.direction, SMC_STRUCTURE: htf_data.trend or bos_choch.htf_direction
    htf_data = strategy_indicators.get('htf_data', {}) or {}
    bos_choch = strategy_indicators.get('bos_choch', {}) or {}

    if is_smc_simple:
        htf_trend = tier1_ema.get('direction')  # "BULL" or "BEAR"
        # Distance from EMA as strength indicator (closer = stronger alignment)
        ema_distance = safe_float(tier1_ema.get('distance_pips', 0))
        # Consider aligned if within reasonable distance and has direction
        htf_strength = 0.8 if htf_trend and ema_distance < 100 else 0.5 if htf_trend else 0
    else:
        htf_trend = htf_data.get('trend') or bos_choch.get('htf_direction')
        htf_strength = safe_float(htf_data.get('strength', 0) or bos_choch.get('htf_strength', 0))

    if htf_trend:
        htf_points = htf_strength * 20
        score += htf_points
        factors.append({
            "name": "HTF Alignment",
            "value": f"{htf_trend} ({htf_strength*100:.0f}%)",
            "points": round(htf_points, 1),
            "max_points": 20
        })
    else:
        factors.append({
            "name": "HTF Alignment",
            "value": "Not aligned",
            "points": 0,
            "max_points": 20
        })

    # 3. Pattern/Entry detection (0-15 points)
    # SMC_SIMPLE: tier3_entry (pullback entry), SMC_STRUCTURE: pattern_data
    pattern_data = strategy_indicators.get('pattern_data', {}) or {}

    if is_smc_simple:
        entry_type = tier3_entry.get('entry_type')  # "PULLBACK"
        pullback_depth = safe_float(tier3_entry.get('pullback_depth', 0))
        in_optimal_zone = tier3_entry.get('in_optimal_zone', False)
        fib_zone = tier3_entry.get('fib_zone', '')

        if entry_type:
            # Score based on pullback quality and zone
            if in_optimal_zone:
                pattern_strength = 0.9
            elif pullback_depth > 0.2 and pullback_depth < 0.7:
                pattern_strength = 0.7
            elif pullback_depth > 0:
                pattern_strength = 0.5
            else:
                pattern_strength = 0.3

            pattern_points = pattern_strength * 15
            score += pattern_points
            factors.append({
                "name": "Entry Pattern",
                "value": f"{entry_type} {fib_zone} ({pattern_strength*100:.0f}%)",
                "points": round(pattern_points, 1),
                "max_points": 15
            })
        else:
            factors.append({
                "name": "Entry Pattern",
                "value": "None detected",
                "points": 0,
                "max_points": 15
            })
    else:
        pattern_type = pattern_data.get('pattern_type')
        pattern_strength = safe_float(pattern_data.get('pattern_strength', 0))

        if pattern_type:
            pattern_points = pattern_strength * 15
            score += pattern_points
            factors.append({
                "name": "Price Pattern",
                "value": f"{pattern_type} ({pattern_strength*100:.0f}%)",
                "points": round(pattern_points, 1),
                "max_points": 15
            })
        else:
            factors.append({
                "name": "Price Pattern",
                "value": "None detected",
                "points": 0,
                "max_points": 15
            })

    # 4. S/R level (0-15 points)
    # SMC_SIMPLE: dataframe_analysis.sr_data, SMC_STRUCTURE: sr_data at root
    sr_data = strategy_indicators.get('sr_data', {}) or {}
    df_sr_data = dataframe_analysis.get('sr_data', {}) or {}

    # Use dataframe_analysis sr_data if root sr_data is empty
    if not sr_data.get('level_price') and df_sr_data:
        sr_data = df_sr_data

    sr_level = sr_data.get('level_price') or sr_data.get('nearest_support') or sr_data.get('nearest_resistance')
    sr_strength = safe_float(sr_data.get('level_strength', 0))

    # For SMC_SIMPLE, calculate strength based on distance to S/R
    if is_smc_simple and (df_sr_data.get('nearest_support') or df_sr_data.get('nearest_resistance')):
        dist_support = safe_float(df_sr_data.get('distance_to_support_pips', 999))
        dist_resistance = safe_float(df_sr_data.get('distance_to_resistance_pips', 999))
        min_dist = min(dist_support, dist_resistance)

        if min_dist < 10:
            sr_strength = 0.9
            sr_level_type = "Very close"
        elif min_dist < 20:
            sr_strength = 0.7
            sr_level_type = "Near"
        elif min_dist < 40:
            sr_strength = 0.5
            sr_level_type = "Moderate"
        else:
            sr_strength = 0.3
            sr_level_type = "Far"

        sr_points = sr_strength * 15
        score += sr_points
        factors.append({
            "name": "S/R Level",
            "value": f"{sr_level_type} ({min_dist:.1f} pips)",
            "points": round(sr_points, 1),
            "max_points": 15
        })
    elif sr_level:
        sr_points = sr_strength * 15
        score += sr_points
        factors.append({
            "name": "S/R Level",
            "value": f"{sr_data.get('level_type', 'level')} ({sr_strength*100:.0f}%)",
            "points": round(sr_points, 1),
            "max_points": 15
        })
    else:
        factors.append({
            "name": "S/R Level",
            "value": "No key level",
            "points": 0,
            "max_points": 15
        })

    # 5. R:R ratio (0-20 points)
    # SMC_SIMPLE: risk_management.rr_ratio, SMC_STRUCTURE: rr_data.rr_ratio
    rr_data = strategy_indicators.get('rr_data', {}) or {}
    rr_ratio = safe_float(rr_data.get('rr_ratio', 0))

    # Fallback to risk_management for SMC_SIMPLE
    if rr_ratio == 0 and risk_management:
        rr_ratio = safe_float(risk_management.get('rr_ratio', 0))

    if rr_ratio >= 2.5:
        rr_points = 20
    elif rr_ratio >= 2.0:
        rr_points = 18
    elif rr_ratio >= 1.5:
        rr_points = 14
    elif rr_ratio >= 1.0:
        rr_points = 10
    elif rr_ratio >= 0.5:
        rr_points = 5
    else:
        rr_points = 0

    score += rr_points
    factors.append({
        "name": "R:R Ratio",
        "value": f"{rr_ratio:.2f}",
        "points": round(rr_points, 1),
        "max_points": 20
    })

    # Determine verdict
    percentage = (score / 100) * 100
    if score >= 70:
        verdict = "GOOD_ENTRY"
    elif score >= 50:
        verdict = "AVERAGE_ENTRY"
    elif score >= 30:
        verdict = "BELOW_AVERAGE_ENTRY"
    else:
        verdict = "POOR_ENTRY"

    return {
        "score": round(score, 1),
        "max_score": 100,
        "percentage": round(percentage, 1),
        "factors": factors,
        "verdict": verdict,
        "htf_aligned": bool(htf_trend),
        "confluence_count": sum(1 for f in factors if f.get('points', 0) > 0)
    }


def assess_exit_quality(
    trade,
    mfe_data: Dict[str, Any],
    exit_type: str
) -> Dict[str, Any]:
    """
    Score how well we exited the trade based on MFE captured.

    Exit efficiency = actual_pips / MFE
    Higher efficiency means we captured more of the available profit.

    Args:
        trade: TradeLog object
        mfe_data: Dictionary from calculate_mfe_mae
        exit_type: Exit type string from classify_exit_type

    Returns:
        Dictionary with exit quality assessment
    """
    mfe_pips = safe_float(mfe_data.get('mfe_pips', 0))
    actual_pips = safe_float(trade.pips_gained)
    pip_multiplier = get_pip_multiplier(trade.symbol)

    # If pips_gained is not recorded, calculate from entry/exit prices
    if actual_pips == 0:
        entry_price = safe_float(trade.entry_price)
        exit_price = safe_float(trade.exit_price_calculated)
        # Fallback: if exit_price not recorded, estimate from SL/TP based on P&L
        if exit_price == 0:
            profit_loss = safe_float(trade.profit_loss)
            if profit_loss < 0:
                # Loss - likely hit SL
                exit_price = safe_float(trade.sl_price)
            elif profit_loss > 0:
                # Win - likely hit TP
                exit_price = safe_float(trade.tp_price)
        if entry_price > 0 and exit_price > 0:
            if trade.direction == "BUY":
                actual_pips = (exit_price - entry_price) * pip_multiplier
            else:
                actual_pips = (entry_price - exit_price) * pip_multiplier

    # Calculate missed profit
    if actual_pips >= 0:
        missed_profit = max(0, mfe_pips - actual_pips)
    else:
        # For losses, missed profit is MFE + the loss
        missed_profit = mfe_pips + abs(actual_pips)

    # Calculate exit efficiency
    if mfe_pips > 0:
        if actual_pips > 0:
            exit_efficiency = min(1.0, actual_pips / mfe_pips)
        else:
            exit_efficiency = 0.0
    else:
        exit_efficiency = 0.0 if actual_pips <= 0 else 1.0

    # Calculate optimal exit price (at MFE)
    optimal_exit_price = mfe_data.get('mfe_price', 0)
    actual_exit_price = safe_float(trade.exit_price_calculated) or safe_float(trade.sl_price)

    # Determine verdict based on exit type and efficiency
    if exit_type == "TP_HIT":
        verdict = "OPTIMAL_EXIT"
        verdict_details = "Take profit reached as planned"
    elif exit_efficiency >= 0.8:
        verdict = "EXCELLENT_EXIT"
        verdict_details = f"Captured {exit_efficiency*100:.0f}% of maximum profit"
    elif exit_efficiency >= 0.6:
        verdict = "GOOD_EXIT"
        verdict_details = f"Captured {exit_efficiency*100:.0f}% of maximum profit"
    elif exit_efficiency >= 0.4:
        verdict = "ACCEPTABLE_EXIT"
        verdict_details = f"Captured {exit_efficiency*100:.0f}% of maximum profit"
    elif actual_pips > 0:
        verdict = "PREMATURE_EXIT"
        verdict_details = f"Exited too early, left {missed_profit:.1f} pips on table"
    elif mfe_pips > 10:  # Had significant profit but ended in loss
        verdict = "REVERSAL_EXIT"
        verdict_details = f"Was +{mfe_pips:.1f} pips but reversed to loss"
    else:
        verdict = "ADVERSE_EXIT"
        verdict_details = "Trade never developed favorably"

    # Count stages activated for trailing stop assessment
    stages_activated = 0
    if trade.moved_to_breakeven:
        stages_activated = 1
        # Would need to check stage data for accurate count

    return {
        "exit_type": exit_type,
        "mfe_pips": round(mfe_pips, 1),
        "actual_pips": round(actual_pips, 1),
        "missed_profit_pips": round(missed_profit, 1),
        "exit_efficiency": round(exit_efficiency, 3),
        "exit_efficiency_pct": round(exit_efficiency * 100, 1),
        "optimal_exit_price": optimal_exit_price,
        "actual_exit_price": actual_exit_price,
        "stages_activated": stages_activated,
        "verdict": verdict,
        "verdict_details": verdict_details,
        "trade_was_profitable": actual_pips > 0
    }


def generate_learning_insights(
    trade,
    entry_quality: Dict[str, Any],
    exit_quality: Dict[str, Any],
    mfe_mae: Dict[str, Any],
    strategy_indicators: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate actionable learning insights from trade analysis.

    Analyzes the trade outcome and provides:
    - Primary factor for win/loss
    - Contributing factors
    - Specific improvement suggestions

    Args:
        trade: TradeLog object
        entry_quality: From assess_entry_quality
        exit_quality: From assess_exit_quality
        mfe_mae: From calculate_mfe_mae
        strategy_indicators: Parsed strategy indicators

    Returns:
        Dictionary with insights and recommendations
    """
    profit_loss = safe_float(trade.profit_loss)
    actual_pips = safe_float(trade.pips_gained)
    mfe_pips = mfe_mae.get('mfe_pips', 0)
    mae_pips = mfe_mae.get('mae_pips', 0)
    mfe_mae_ratio = mfe_mae.get('mfe_mae_ratio', 0)
    pip_multiplier = get_pip_multiplier(trade.symbol)

    # If pips_gained is not recorded, calculate from entry/exit prices
    if actual_pips == 0:
        entry_price = safe_float(trade.entry_price)
        exit_price = safe_float(trade.exit_price_calculated)
        # Fallback: if exit_price not recorded, estimate from SL/TP based on P&L
        if exit_price == 0:
            if profit_loss < 0:
                exit_price = safe_float(trade.sl_price)
            elif profit_loss > 0:
                exit_price = safe_float(trade.tp_price)
        if entry_price > 0 and exit_price > 0:
            if trade.direction == "BUY":
                actual_pips = (exit_price - entry_price) * pip_multiplier
            else:
                actual_pips = (entry_price - exit_price) * pip_multiplier

    # Determine if winner using pips or profit_loss as fallback
    if actual_pips != 0:
        is_winner = actual_pips > 2
        is_loser = actual_pips < -2
    else:
        is_winner = profit_loss > 5
        is_loser = profit_loss < -5

    # Determine trade result
    if is_winner:
        trade_result = "WIN"
    elif is_loser:
        trade_result = "LOSS"
    else:
        trade_result = "BREAKEVEN"

    insights = {
        "trade_result": trade_result,
        "primary_factor": None,
        "contributing_factors": [],
        "what_went_right": [],
        "what_went_wrong": [],
        "improvement_suggestions": [],
        "pattern_identified": None,
        "key_takeaway": None
    }

    entry_verdict = entry_quality.get('verdict', 'UNKNOWN')
    exit_verdict = exit_quality.get('verdict', 'UNKNOWN')
    initial_move = mfe_mae.get('initial_move', 'UNKNOWN')
    immediate_reversal = mfe_mae.get('immediate_reversal', False)

    # ===== ANALYZE WINNERS =====
    if is_winner:
        # Determine primary factor for win
        if entry_verdict == "GOOD_ENTRY" and exit_verdict in ["OPTIMAL_EXIT", "EXCELLENT_EXIT"]:
            insights["primary_factor"] = "Well-executed trade with strong entry and good exit"
            insights["pattern_identified"] = "TEXTBOOK_TRADE"
        elif entry_verdict == "GOOD_ENTRY":
            insights["primary_factor"] = "Strong entry signal with good confluence"
            insights["pattern_identified"] = "STRONG_ENTRY"
        elif exit_verdict in ["OPTIMAL_EXIT", "EXCELLENT_EXIT"]:
            insights["primary_factor"] = "Good exit timing maximized profit"
            insights["pattern_identified"] = "GOOD_EXIT_MANAGEMENT"
        elif mfe_mae_ratio > 3:
            insights["primary_factor"] = "Clean price action with favorable MFE/MAE"
            insights["pattern_identified"] = "CLEAN_MOMENTUM"
        else:
            insights["primary_factor"] = "Trade worked despite mixed signals"
            insights["pattern_identified"] = "FORTUNATE_WIN"

        # What went right
        if entry_quality.get('htf_aligned'):
            insights["what_went_right"].append("HTF alignment provided directional support")
        if mfe_mae_ratio > 2:
            insights["what_went_right"].append(f"Excellent MFE/MAE ratio of {mfe_mae_ratio:.1f} - clean trade")
        if initial_move == "FAVORABLE":
            insights["what_went_right"].append("Price moved favorably immediately after entry")
        if exit_quality.get('exit_efficiency', 0) > 0.7:
            insights["what_went_right"].append(f"Captured {exit_quality['exit_efficiency_pct']:.0f}% of available profit")

        # Check for improvements even on winners (if left >10 pips on table)
        if exit_quality.get('exit_efficiency', 0) < 0.7 and exit_quality.get('missed_profit_pips', 0) > 10:
            missed = exit_quality.get('missed_profit_pips', 0)
            # Use pips if available, otherwise use profit_loss for display
            if actual_pips != 0:
                actual_display = f"{actual_pips:.1f} pips"
            else:
                actual_display = f"{profit_loss:.2f} (P&L)"
            insights["improvement_suggestions"].append(
                f"Could have captured more profit - MFE was {mfe_pips:.1f} pips vs actual {actual_display}"
            )
            if missed > 20:
                insights["improvement_suggestions"].append(
                    "Consider adjusting trailing stop to lock in more profit during strong moves"
                )

        # Key takeaway
        if entry_verdict == "GOOD_ENTRY":
            insights["key_takeaway"] = "High-quality entries continue to show positive results"
        else:
            insights["key_takeaway"] = "This win pattern could be studied for replication"

    # ===== ANALYZE LOSERS =====
    elif is_loser:
        # Was trade ever profitable?
        if mfe_pips >= 10:  # Had at least 10 pips profit at some point
            insights["primary_factor"] = "Trade was profitable but reversed before exit"
            insights["pattern_identified"] = "PROFITABLE_REVERSAL"
            insights["what_went_wrong"].append(
                f"Price reached +{mfe_pips:.1f} pips profit before reversing"
            )
            insights["improvement_suggestions"].append(
                "Consider tighter trailing stop or partial take profit"
            )
            if mfe_pips >= 20:
                insights["improvement_suggestions"].append(
                    f"With MFE of {mfe_pips:.1f} pips, a partial TP at 50% would have secured profit"
                )
        elif mfe_pips >= 5:
            insights["primary_factor"] = "Trade showed promise but failed to develop"
            insights["pattern_identified"] = "WEAK_MOMENTUM"
            insights["what_went_wrong"].append(
                f"Only reached +{mfe_pips:.1f} pips before reversing"
            )
        else:
            insights["primary_factor"] = "Trade never developed - immediate adverse move"
            insights["pattern_identified"] = "FAILED_ENTRY"
            insights["what_went_wrong"].append("Price moved against position almost immediately")

        # Analyze entry quality impact
        if entry_verdict == "POOR_ENTRY":
            insights["contributing_factors"].append("Entry signal quality was below threshold")
            insights["improvement_suggestions"].append(
                f"Entry score was {entry_quality.get('score', 0):.0f}/100 - consider requiring higher confluence"
            )
        elif entry_verdict == "BELOW_AVERAGE_ENTRY":
            insights["contributing_factors"].append("Entry signal had weak confluence")

        # Check HTF alignment
        if not entry_quality.get('htf_aligned'):
            insights["contributing_factors"].append("HTF was not aligned with entry direction")
            insights["improvement_suggestions"].append(
                "Strict HTF alignment filter could have prevented this loss"
            )

        # Check immediate reversal
        if immediate_reversal:
            insights["contributing_factors"].append("Immediate adverse move within first 3 candles")
            insights["improvement_suggestions"].append(
                "Look for entries after pullback confirmation, not at resistance"
            )

        # Check if close to TP before losing
        pct_of_tp = mfe_mae.get('percentage_of_tp', 0)
        if pct_of_tp >= 70:
            insights["contributing_factors"].append(
                f"Price reached {pct_of_tp:.0f}% of TP distance before reversing"
            )
            insights["improvement_suggestions"].append(
                "Consider partial TP at 70% of target distance"
            )

        # Check MAE severity
        pct_of_sl = mfe_mae.get('percentage_of_sl', 0)
        if pct_of_sl >= 100:
            insights["what_went_wrong"].append("Stop loss was hit directly")
        elif pct_of_sl >= 80:
            insights["what_went_wrong"].append(f"Deep drawdown reached {pct_of_sl:.0f}% of SL distance")

        # Key takeaway based on pattern
        if insights["pattern_identified"] == "PROFITABLE_REVERSAL":
            insights["key_takeaway"] = "Consider tighter profit protection for trades that reach significant MFE"
        elif insights["pattern_identified"] == "FAILED_ENTRY":
            insights["key_takeaway"] = "Review entry criteria - this pattern shows immediate failure"
        else:
            insights["key_takeaway"] = "Analyze similar setups to identify common loss patterns"

    # ===== ANALYZE BREAKEVEN =====
    else:
        insights["primary_factor"] = "Trade closed near entry price"
        insights["pattern_identified"] = "BREAKEVEN_EXIT"
        if mfe_pips > 5:
            insights["what_went_right"].append(f"Protected capital despite reaching +{mfe_pips:.1f} pips")
            insights["improvement_suggestions"].append("Consider partial take profit to secure gains")
        else:
            insights["what_went_right"].append("Limited losses by exiting near entry")
        insights["key_takeaway"] = "Breakeven trades preserve capital for better opportunities"

    # Add general contributing factors
    if mfe_mae_ratio < 1:
        insights["contributing_factors"].append(
            f"Unfavorable MFE/MAE ratio of {mfe_mae_ratio:.2f} indicates adverse price action"
        )

    # Add pattern-based suggestions
    htf_data = strategy_indicators.get('htf_data', {}) or {}
    if htf_data.get('in_pullback'):
        if is_winner:
            insights["what_went_right"].append("Entered during pullback in established trend")
        elif is_loser:
            insights["what_went_wrong"].append("Pullback continued against position")

    return insights


def get_outcome_summary(
    trade,
    mfe_mae: Dict[str, Any],
    exit_type: str
) -> Dict[str, Any]:
    """
    Generate high-level outcome summary for display.

    Args:
        trade: TradeLog object
        mfe_mae: MFE/MAE calculation results
        exit_type: Exit type classification

    Returns:
        Summary dictionary with key metrics
    """
    profit_loss = safe_float(trade.profit_loss)
    pips = safe_float(trade.pips_gained)
    entry_price = safe_float(trade.entry_price)
    exit_price = safe_float(trade.exit_price_calculated)
    sl_price = safe_float(trade.sl_price)
    tp_price = safe_float(trade.tp_price)
    pip_multiplier = get_pip_multiplier(trade.symbol)

    # If pips_gained is not recorded, calculate from entry/exit prices
    if pips == 0 and entry_price > 0:
        # Fallback: if exit_price not recorded, estimate from SL/TP based on P&L and trade state
        if exit_price == 0:
            if profit_loss < 0:
                # Loss - likely hit SL
                exit_price = sl_price
            elif profit_loss > 0:
                # Winner - check if exited via trailing stop or TP
                # If SL is above entry (BUY) or below entry (SELL), trade was trailed
                is_trailing_exit = False
                if trade.direction == "BUY" and sl_price > entry_price:
                    is_trailing_exit = True
                    exit_price = sl_price  # Exited at trailing stop
                elif trade.direction == "SELL" and sl_price < entry_price:
                    is_trailing_exit = True
                    exit_price = sl_price  # Exited at trailing stop

                # If not trailing exit, assume TP was hit
                if not is_trailing_exit:
                    exit_price = tp_price

        if exit_price > 0:
            if trade.direction == "BUY":
                pips = (exit_price - entry_price) * pip_multiplier
            else:
                pips = (entry_price - exit_price) * pip_multiplier

    # Calculate R-multiple
    if trade.direction == "BUY":
        risk_pips = (entry_price - sl_price) * pip_multiplier if sl_price else 0
    else:
        risk_pips = (sl_price - entry_price) * pip_multiplier if sl_price else 0

    r_multiple = pips / risk_pips if risk_pips > 0 else 0

    # Determine result - use pips if available, otherwise use profit_loss
    # Many trades have pips_gained = None but have valid profit_loss
    if pips != 0:
        # Use pips if available
        if pips > 2:
            result = "WIN"
        elif pips < -2:
            result = "LOSS"
        else:
            result = "BREAKEVEN"
    else:
        # Fall back to profit_loss
        if profit_loss > 5:  # Small buffer for fees/spread
            result = "WIN"
        elif profit_loss < -5:
            result = "LOSS"
        else:
            result = "BREAKEVEN"

    # Calculate duration
    duration_minutes = 0
    if trade.timestamp and trade.closed_at:
        duration = trade.closed_at - trade.timestamp
        duration_minutes = int(duration.total_seconds() / 60)
    elif trade.lifecycle_duration_minutes:
        duration_minutes = trade.lifecycle_duration_minutes

    return {
        "result": result,
        "profit_loss": round(profit_loss, 2),
        "pips_gained": round(pips, 1),
        "r_multiple": round(r_multiple, 2),
        "duration_minutes": duration_minutes,
        "duration_display": _format_duration(duration_minutes),
        "exit_type": exit_type,
        "mfe_pips": mfe_mae.get('mfe_pips', 0),
        "mae_pips": mfe_mae.get('mae_pips', 0),
        "mfe_mae_ratio": mfe_mae.get('mfe_mae_ratio', 0)
    }


def _format_duration(minutes: int) -> str:
    """Format duration in minutes to human-readable string"""
    if minutes < 60:
        return f"{minutes}m"
    elif minutes < 1440:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m"
    else:
        days = minutes // 1440
        hours = (minutes % 1440) // 60
        return f"{days}d {hours}h"


def fetch_trade_candles(
    trade,
    db: Session,
    candle_model,
    context_candles: int = 5,
    timeframe: int = 15
) -> Dict[str, List[Dict]]:
    """
    Fetch candles around the trade for visualization.

    Args:
        trade: TradeLog object
        db: Database session
        candle_model: Candle model class
        context_candles: Number of candles before entry and after exit
        timeframe: Candle timeframe

    Returns:
        Dictionary with entry_context, trade_candles, and exit_context
    """
    symbol = trade.symbol
    start_time = trade.timestamp
    end_time = trade.closed_at or datetime.utcnow()

    # Calculate context time windows
    candle_duration = timedelta(minutes=timeframe)
    entry_context_start = start_time - (candle_duration * context_candles)
    exit_context_end = end_time + (candle_duration * context_candles)

    # Fetch all relevant candles
    candles = db.query(candle_model).filter(
        and_(
            candle_model.epic == symbol,
            candle_model.timeframe == timeframe,
            candle_model.start_time >= entry_context_start,
            candle_model.start_time <= exit_context_end
        )
    ).order_by(candle_model.start_time).all()

    # Categorize candles
    entry_context = []
    trade_candles = []
    exit_context = []

    for candle in candles:
        candle_data = {
            "timestamp": str(candle.start_time),
            "open": safe_float(candle.open),
            "high": safe_float(candle.high),
            "low": safe_float(candle.low),
            "close": safe_float(candle.close),
            "volume": candle.volume
        }

        if candle.start_time < start_time:
            entry_context.append(candle_data)
        elif candle.start_time <= end_time:
            trade_candles.append(candle_data)
        else:
            exit_context.append(candle_data)

    # Limit context candles to requested amount
    entry_context = entry_context[-context_candles:]
    exit_context = exit_context[:context_candles]

    return {
        "entry_context": entry_context,
        "trade_candles": trade_candles,
        "exit_context": exit_context,
        "total_candles": len(candles)
    }
