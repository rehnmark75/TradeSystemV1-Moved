"""
Trade Analysis Router - Detailed trailing stop analysis for individual trades
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from services.db import get_db
from services.models import TradeLog
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
from pathlib import Path

router = APIRouter(prefix="/api/trade-analysis", tags=["trade-analysis"])


def parse_trade_logs(trade_id: int, log_file: str = "/app/logs/trade_monitor.log") -> Dict[str, Any]:
    """
    Parse trade monitor logs to extract trailing stop events for a specific trade

    Args:
        trade_id: Trade ID to search for
        log_file: Path to log file

    Returns:
        Dictionary with parsed events
    """
    events = {
        "break_even_triggers": [],
        "stage2_triggers": [],
        "stage3_triggers": [],
        "profit_updates": [],
        "stop_adjustments": [],
        "status_changes": [],
        "errors": []
    }

    try:
        log_path = Path(log_file)
        if not log_path.exists():
            return events

        with open(log_file, 'r') as f:
            for line in f:
                # Only process lines for this trade
                if f"trade {trade_id}" not in line.lower() and f"trade.id.*{trade_id}" not in line.lower():
                    continue

                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                timestamp = timestamp_match.group(1) if timestamp_match else None

                # Break-even triggers
                if "BREAK-EVEN TRIGGER" in line:
                    profit_match = re.search(r'Profit (\d+)pts >= trigger (\d+)pts', line)
                    if profit_match:
                        events["break_even_triggers"].append({
                            "timestamp": timestamp,
                            "profit_pts": int(profit_match.group(1)),
                            "trigger_pts": int(profit_match.group(2)),
                            "line": line.strip()
                        })

                # Profit updates
                elif "PROFIT]" in line and "entry=" in line:
                    profit_match = re.search(r'entry=([\d.]+), current=([\d.]+), profit=(\d+)pts', line)
                    if profit_match:
                        events["profit_updates"].append({
                            "timestamp": timestamp,
                            "entry": float(profit_match.group(1)),
                            "current": float(profit_match.group(2)),
                            "profit_pts": int(profit_match.group(3))
                        })

                # Stop adjustments
                elif "IMMEDIATE TRAIL SUCCESS" in line or "Stop moved" in line:
                    stop_match = re.search(r'Stop moved to ([\d.]+)', line)
                    if stop_match:
                        events["stop_adjustments"].append({
                            "timestamp": timestamp,
                            "new_stop": float(stop_match.group(1)),
                            "line": line.strip()
                        })

                # Status changes
                elif "Processing trade" in line and "status=" in line:
                    status_match = re.search(r'status=(\w+)', line)
                    if status_match:
                        events["status_changes"].append({
                            "timestamp": timestamp,
                            "status": status_match.group(1)
                        })

                # Errors
                elif "ERROR" in line or "FAILED" in line:
                    events["errors"].append({
                        "timestamp": timestamp,
                        "message": line.strip()
                    })

    except Exception as e:
        events["errors"].append({
            "timestamp": None,
            "message": f"Failed to parse logs: {str(e)}"
        })

    return events


def get_pair_config(symbol: str) -> Dict[str, Any]:
    """Get pair-specific trailing configuration"""
    try:
        from config import get_trailing_config_for_epic
        return get_trailing_config_for_epic(symbol)
    except Exception:
        return {
            "stage1_trigger_points": 12,
            "stage1_lock_points": 2,
            "stage2_trigger_points": 16,
            "stage2_lock_points": 10,
            "stage3_trigger_points": 17,
            "stage3_atr_multiplier": 0.8,
            "stage3_min_distance": 2,
            "min_trail_distance": 15,
            "break_even_trigger_points": 6,
        }


def analyze_stage_activation(trade: TradeLog, log_events: Dict[str, Any], pair_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze which trailing stop stages were activated

    Structure:
    - Break-even: Moves stop to entry (0 profit)
    - Stage 1: First profit lock
    - Stage 2: Second profit lock
    - Stage 3: ATR-based trailing

    Returns:
        Dictionary with stage analysis
    """
    analysis = {
        "breakeven": {
            "activated": False,
            "trigger_threshold": pair_config.get('break_even_trigger_points', 12),
            "lock_amount": 0,  # Break-even locks at 0 profit (entry price)
            "activation_time": None,
            "max_profit_reached": 0,
            "final_lock": 0
        },
        "stage1": {
            "activated": False,
            "trigger_threshold": pair_config.get('stage1_trigger_points', 16),
            "lock_amount": pair_config.get('stage1_lock_points', 8),
            "activation_time": None
        },
        "stage2": {
            "activated": False,
            "trigger_threshold": pair_config.get('stage2_trigger_points', 22),
            "lock_amount": pair_config.get('stage2_lock_points', 12),
            "activation_time": None
        },
        "stage3": {
            "activated": False,
            "trigger_threshold": pair_config.get('stage3_trigger_points', 23),
            "atr_multiplier": pair_config.get('stage3_atr_multiplier', 0.8),
            "activation_time": None
        }
    }

    # Calculate max profit from updates FIRST
    if log_events["profit_updates"]:
        max_profit = max(event["profit_pts"] for event in log_events["profit_updates"])
        analysis["breakeven"]["max_profit_reached"] = max_profit
    else:
        max_profit = 0

    # Helper function to calculate lock distance in points
    def calculate_lock_points(stop_price: float, entry_price: float, symbol: str, direction: str) -> float:
        """Calculate how many points profit the stop is locking in"""
        if direction == "BUY":
            lock_distance = stop_price - entry_price
        else:
            lock_distance = entry_price - stop_price

        # Convert to points
        if "JPY" in symbol:
            return round(lock_distance * 100, 1)
        else:
            return round(lock_distance * 10000, 1)

    # Calculate final lock from current trade data (for overall summary)
    final_lock_pts = 0
    if trade.entry_price and trade.sl_price:
        final_lock_pts = calculate_lock_points(trade.sl_price, trade.entry_price, trade.symbol, trade.direction)

    # Check Break-even activation and find what it locked
    if max_profit >= analysis["breakeven"]["trigger_threshold"]:
        analysis["breakeven"]["activated"] = True
        # Find when it was triggered and what stop was set
        for event in log_events["profit_updates"]:
            if event["profit_pts"] >= analysis["breakeven"]["trigger_threshold"]:
                analysis["breakeven"]["activation_time"] = event["timestamp"]
                # Find the next stop adjustment after this time
                for stop_adj in log_events["stop_adjustments"]:
                    if stop_adj["timestamp"] >= event["timestamp"]:
                        lock_pts = calculate_lock_points(stop_adj["new_stop"], trade.entry_price, trade.symbol, trade.direction)
                        analysis["breakeven"]["final_lock"] = lock_pts
                        break
                break

    # Check Stage 1
    if max_profit >= analysis["stage1"]["trigger_threshold"]:
        analysis["stage1"]["activated"] = True
        # Find when it was triggered and what stop was set
        for event in log_events["profit_updates"]:
            if event["profit_pts"] >= analysis["stage1"]["trigger_threshold"]:
                analysis["stage1"]["activation_time"] = event["timestamp"]
                # Find the next stop adjustment after this time
                for stop_adj in log_events["stop_adjustments"]:
                    if stop_adj["timestamp"] >= event["timestamp"]:
                        lock_pts = calculate_lock_points(stop_adj["new_stop"], trade.entry_price, trade.symbol, trade.direction)
                        analysis["stage1"]["actual_lock"] = lock_pts
                        break
                break

    # Check Stage 2
    if max_profit >= analysis["stage2"]["trigger_threshold"]:
        analysis["stage2"]["activated"] = True
        # Find when it was triggered and what stop was set
        for event in log_events["profit_updates"]:
            if event["profit_pts"] >= analysis["stage2"]["trigger_threshold"]:
                analysis["stage2"]["activation_time"] = event["timestamp"]
                # Find the next stop adjustment after this time
                for stop_adj in log_events["stop_adjustments"]:
                    if stop_adj["timestamp"] >= event["timestamp"]:
                        lock_pts = calculate_lock_points(stop_adj["new_stop"], trade.entry_price, trade.symbol, trade.direction)
                        analysis["stage2"]["actual_lock"] = lock_pts
                        break
                break

    # Check Stage 3
    if max_profit >= analysis["stage3"]["trigger_threshold"]:
        analysis["stage3"]["activated"] = True
        for event in log_events["profit_updates"]:
            if event["profit_pts"] >= analysis["stage3"]["trigger_threshold"]:
                analysis["stage3"]["activation_time"] = event["timestamp"]
                break

    # Set overall final lock for summary
    analysis["breakeven"]["final_lock"] = analysis["breakeven"].get("final_lock", final_lock_pts)

    return analysis


@router.get("/trade/{trade_id}")
async def get_trade_analysis(trade_id: int, db: Session = Depends(get_db)):
    """
    Get comprehensive trailing stop analysis for a specific trade

    Args:
        trade_id: Trade ID to analyze

    Returns:
        Detailed analysis including:
        - Trade details
        - Pair-specific configuration
        - Stage activation timeline
        - Log events
        - Performance metrics
    """
    # Get trade from database
    trade = db.query(TradeLog).filter(TradeLog.id == trade_id).first()

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    # Get pair configuration
    pair_config = get_pair_config(trade.symbol)

    # Parse logs
    log_events = parse_trade_logs(trade_id)

    # Analyze stage activation
    stage_analysis = analyze_stage_activation(trade, log_events, pair_config)

    # Calculate distances and metrics
    entry_price = trade.entry_price or 0
    sl_price = trade.sl_price or 0
    tp_price = trade.tp_price or 0

    if "JPY" in trade.symbol:
        multiplier = 100
    else:
        multiplier = 10000

    # Calculate distances
    if trade.direction == "BUY":
        sl_distance = (sl_price - entry_price) * multiplier
        tp_distance = (tp_price - entry_price) * multiplier if tp_price else 0
    else:
        sl_distance = (entry_price - sl_price) * multiplier
        tp_distance = (entry_price - tp_price) * multiplier if tp_price else 0

    # Build response
    response = {
        "trade_id": trade_id,
        "trade_details": {
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "status": trade.status,
            "deal_id": trade.deal_id,
            "ig_min_stop_distance": trade.min_stop_distance_points,
            "moved_to_breakeven": trade.moved_to_breakeven,
            "trigger_time": str(trade.trigger_time) if trade.trigger_time else None,
        },
        "calculated_metrics": {
            "sl_distance_pts": round(abs(sl_distance), 1),
            "tp_distance_pts": round(abs(tp_distance), 1),
            "sl_above_entry": sl_distance > 0,
            "risk_reward_ratio": round(tp_distance / abs(sl_distance), 2) if sl_distance != 0 else 0
        },
        "pair_configuration": pair_config,
        "stage_analysis": stage_analysis,
        "timeline": {
            "profit_progression": log_events["profit_updates"][-50:],  # Last 50 updates
            "break_even_events": log_events["break_even_triggers"],
            "stop_adjustments": log_events["stop_adjustments"],
            "status_changes": log_events["status_changes"][-20:],  # Last 20 status changes
        },
        "summary": {
            "breakeven_activated": stage_analysis["breakeven"]["activated"],
            "stages_activated": sum([
                stage_analysis["stage1"]["activated"],
                stage_analysis["stage2"]["activated"],
                stage_analysis["stage3"]["activated"]
            ]),
            "max_profit_reached": stage_analysis["breakeven"]["max_profit_reached"],
            "final_protection": stage_analysis["breakeven"]["final_lock"],
            "fully_trailed": stage_analysis["stage3"]["activated"]
        }
    }

    return response


@router.get("/trade/{trade_id}/timeline")
async def get_trade_timeline(trade_id: int, db: Session = Depends(get_db)):
    """
    Get detailed timeline of events for a trade

    Returns a chronological list of all events
    """
    trade = db.query(TradeLog).filter(TradeLog.id == trade_id).first()

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    log_events = parse_trade_logs(trade_id)

    # Combine all events into single timeline
    timeline = []

    for event in log_events["profit_updates"]:
        timeline.append({
            "timestamp": event["timestamp"],
            "type": "profit_update",
            "data": event
        })

    for event in log_events["break_even_triggers"]:
        timeline.append({
            "timestamp": event["timestamp"],
            "type": "break_even_trigger",
            "data": event
        })

    for event in log_events["stop_adjustments"]:
        timeline.append({
            "timestamp": event["timestamp"],
            "type": "stop_adjustment",
            "data": event
        })

    for event in log_events["status_changes"]:
        timeline.append({
            "timestamp": event["timestamp"],
            "type": "status_change",
            "data": event
        })

    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"] or "")

    return {
        "trade_id": trade_id,
        "symbol": trade.symbol,
        "timeline": timeline,
        "total_events": len(timeline)
    }
