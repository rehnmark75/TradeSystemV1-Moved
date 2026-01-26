"""
Trade Analysis Router - Detailed trailing stop analysis for individual trades
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from services.db import get_db
from services.models import TradeLog, AlertHistory, IGCandle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
import json
from pathlib import Path

# Import trade analysis service functions
from services.trade_analysis_service import (
    calculate_mfe_mae,
    classify_exit_type,
    assess_entry_quality,
    assess_exit_quality,
    generate_learning_insights,
    get_outcome_summary,
    fetch_trade_candles,
    safe_float as service_safe_float
)

router = APIRouter(prefix="/api/trade-analysis", tags=["trade-analysis"])


def get_rotated_log_files(base_log_file: str) -> List[Path]:
    """
    Get all rotated log files in chronological order (oldest first).

    Handles rotation pattern: trade_monitor.log, trade_monitor.log.1, trade_monitor.log.2, etc.
    Returns files sorted so oldest rotated files are read first, newest last.
    """
    log_path = Path(base_log_file)
    log_dir = log_path.parent
    log_name = log_path.name

    # Find all matching log files
    log_files = []

    # Add rotated files first (oldest to newest: .5, .4, .3, .2, .1)
    for i in range(10, 0, -1):  # Check up to .10 rotation
        rotated = log_dir / f"{log_name}.{i}"
        if rotated.exists():
            log_files.append(rotated)

    # Add current log file last (most recent)
    if log_path.exists():
        log_files.append(log_path)

    return log_files


def parse_trade_logs(trade_id: int, log_file: str = "/app/logs/trade_monitor.log") -> Dict[str, Any]:
    """
    Parse trade monitor logs to extract trailing stop events for a specific trade.

    Reads all rotated log files to find historical data.

    Args:
        trade_id: Trade ID to search for
        log_file: Base path to log file (will also check rotated versions)

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

    # Get all rotated log files
    log_files = get_rotated_log_files(log_file)

    if not log_files:
        events["errors"].append({
            "timestamp": None,
            "message": f"No log files found at {log_file}"
        })
        return events

    # Build search patterns for this trade
    # Match: "Trade 1535", "trade 1535", "Trade 1535:", etc.
    trade_pattern = re.compile(rf'\btrade\s+{trade_id}\b', re.IGNORECASE)

    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Only process lines for this trade
                    if not trade_pattern.search(line):
                        continue

                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else None

                    # Break-even triggers
                    # Matches: "[BREAK-EVEN TRIGGER] Trade 1535: Profit 6pts >= trigger 6pts"
                    if "BREAK-EVEN TRIGGER" in line:
                        profit_match = re.search(r'Profit (\d+)pts >= trigger (\d+)pts', line)
                        if profit_match:
                            events["break_even_triggers"].append({
                                "timestamp": timestamp,
                                "profit_pts": int(profit_match.group(1)),
                                "trigger_pts": int(profit_match.group(2)),
                                "line": line.strip()
                            })

                    # Break-even success confirmation
                    # Matches: "[BREAK-EVEN] Trade 1535 CS.D.AUDUSD.MINI.IP moved to break-even: 0.66320"
                    elif "[BREAK-EVEN]" in line and "moved to break-even" in line:
                        stop_match = re.search(r'moved to break-even:\s*([\d.]+)', line)
                        if stop_match:
                            events["stop_adjustments"].append({
                                "timestamp": timestamp,
                                "new_stop": float(stop_match.group(1)),
                                "type": "break_even",
                                "line": line.strip()
                            })

                    # Stage triggers (stage 1, 2, 3)
                    # Matches: "[STAGE1 TRIGGER]", "[STAGE2 TRIGGER]", "[STAGE3 TRIGGER]"
                    elif "STAGE1 TRIGGER" in line or "STAGE1]" in line:
                        events["stage2_triggers"].append({
                            "timestamp": timestamp,
                            "stage": 1,
                            "line": line.strip()
                        })
                    elif "STAGE2 TRIGGER" in line or "STAGE2]" in line:
                        events["stage2_triggers"].append({
                            "timestamp": timestamp,
                            "stage": 2,
                            "line": line.strip()
                        })
                    elif "STAGE3 TRIGGER" in line or "STAGE3]" in line:
                        events["stage3_triggers"].append({
                            "timestamp": timestamp,
                            "stage": 3,
                            "line": line.strip()
                        })

                    # Profit updates
                    # Matches: "[PROFIT] Trade 1535 CS.D.AUDUSD.MINI.IP BUY: entry=0.66300, current=0.66298, profit=0pts"
                    elif "[PROFIT]" in line and "entry=" in line:
                        profit_match = re.search(r'entry=([\d.]+),\s*current=([\d.]+),\s*profit=(-?\d+)pts', line)
                        if profit_match:
                            events["profit_updates"].append({
                                "timestamp": timestamp,
                                "entry": float(profit_match.group(1)),
                                "current": float(profit_match.group(2)),
                                "profit_pts": int(profit_match.group(3))
                            })

                    # Stop adjustments - multiple patterns
                    # Pattern 1: "IMMEDIATE TRAIL SUCCESS" or "Stop moved to X"
                    # Pattern 2: "moved to break-even: X" (handled above)
                    # Pattern 3: "TRAIL SUCCESS" with new stop price
                    elif "TRAIL SUCCESS" in line or "Stop moved" in line:
                        stop_match = re.search(r'(?:Stop moved to|new stop[:\s]+|trailing.*?to[:\s]+)([\d.]+)', line, re.IGNORECASE)
                        if stop_match:
                            events["stop_adjustments"].append({
                                "timestamp": timestamp,
                                "new_stop": float(stop_match.group(1)),
                                "type": "trail",
                                "line": line.strip()
                            })

                    # Status changes
                    # Matches: "Processing trade 1535 CS.D.AUDUSD.MINI.IP status=tracking"
                    elif "Processing trade" in line and "status=" in line:
                        status_match = re.search(r'status=(\w+)', line)
                        if status_match:
                            # Only add if status changed from last recorded
                            new_status = status_match.group(1)
                            if not events["status_changes"] or events["status_changes"][-1]["status"] != new_status:
                                events["status_changes"].append({
                                    "timestamp": timestamp,
                                    "status": new_status
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
                "message": f"Failed to parse {log_path}: {str(e)}"
            })

    return events


def get_pair_config(symbol: str, is_scalp_trade: bool = False) -> Dict[str, Any]:
    """Get pair-specific trailing configuration"""
    try:
        from config import get_trailing_config_for_epic
        return get_trailing_config_for_epic(symbol, is_scalp_trade=is_scalp_trade)
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

    IMPORTANT: Uses database fields as primary source of truth, not just log parsing.
    The moved_to_breakeven flag and trade status are authoritative.

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

    # Calculate final lock from current trade data (the actual SL in DB)
    final_lock_pts = 0
    if trade.entry_price and trade.sl_price:
        final_lock_pts = calculate_lock_points(trade.sl_price, trade.entry_price, trade.symbol, trade.direction)

    # ✅ PRIMARY SOURCE OF TRUTH: Use database fields, NOT log parsing
    # The moved_to_breakeven flag is set by the trailing stop system when BE is actually executed
    breakeven_activated_db = getattr(trade, 'moved_to_breakeven', False) or False

    # Check trade status for stage indicators
    trade_status = getattr(trade, 'status', '') or ''
    stage1_activated_db = 'stage1' in trade_status.lower() or 'profit_lock' in trade_status.lower()
    stage2_activated_db = 'stage2' in trade_status.lower() or 'profit_protected' in trade_status.lower()
    stage3_activated_db = 'stage3' in trade_status.lower() or 'trailing' in trade_status.lower()

    # ✅ FIXED: Only consider POSITIVE profit values from logs (filter out bug where losses showed as profits)
    # This handles historical logs that had the abs() bug
    positive_profit_updates = [
        event for event in log_events.get("profit_updates", [])
        if event.get("profit_pts", 0) >= 0
    ]

    # Calculate max profit from POSITIVE updates only
    if positive_profit_updates:
        max_profit = max(event["profit_pts"] for event in positive_profit_updates)
    else:
        max_profit = 0

    analysis["breakeven"]["max_profit_reached"] = max_profit

    # ✅ Break-even: Use DB flag as primary, log max_profit as secondary validation
    # Only mark as activated if BOTH the DB flag is set OR stop adjustments show it happened
    if breakeven_activated_db:
        analysis["breakeven"]["activated"] = True
        # Find activation time from logs if available
        for event in positive_profit_updates:
            if event["profit_pts"] >= analysis["breakeven"]["trigger_threshold"]:
                analysis["breakeven"]["activation_time"] = event["timestamp"]
                break
    elif log_events.get("break_even_triggers"):
        # Check if there's an actual break-even trigger event in logs
        analysis["breakeven"]["activated"] = True
        analysis["breakeven"]["activation_time"] = log_events["break_even_triggers"][0].get("timestamp")
    elif log_events.get("stop_adjustments"):
        # Check if stop was actually moved to protect profit
        for adj in log_events["stop_adjustments"]:
            if adj.get("type") == "break_even":
                analysis["breakeven"]["activated"] = True
                analysis["breakeven"]["activation_time"] = adj.get("timestamp")
                break

    # ✅ Stage 1: Use DB status or log triggers
    if stage1_activated_db:
        analysis["stage1"]["activated"] = True
    elif any("STAGE1" in str(event.get("line", "")).upper() or "STAGE 1" in str(event.get("line", "")).upper()
             for event in log_events.get("stage2_triggers", [])):  # stage2_triggers includes stage1 in current code
        analysis["stage1"]["activated"] = True

    # ✅ Stage 2: Use DB status or log triggers
    if stage2_activated_db:
        analysis["stage2"]["activated"] = True
    elif any(event.get("stage") == 2 for event in log_events.get("stage2_triggers", [])):
        analysis["stage2"]["activated"] = True

    # ✅ Stage 3: Use DB status or log triggers
    if stage3_activated_db:
        analysis["stage3"]["activated"] = True
    elif any(event.get("stage") == 3 for event in log_events.get("stage3_triggers", [])):
        analysis["stage3"]["activated"] = True

    # Set final lock from actual trade SL (the real protection level)
    analysis["breakeven"]["final_lock"] = final_lock_pts

    # Add stop adjustment history for transparency
    if log_events.get("stop_adjustments"):
        # Get the last stop adjustment (most recent protection level)
        last_stop_adj = log_events["stop_adjustments"][-1]
        if trade.entry_price:
            last_lock = calculate_lock_points(last_stop_adj["new_stop"], trade.entry_price, trade.symbol, trade.direction)
            analysis["breakeven"]["last_logged_lock"] = last_lock

    # ✅ Add warning if log data seems inconsistent with DB
    if max_profit > 0 and not breakeven_activated_db and max_profit >= analysis["breakeven"]["trigger_threshold"]:
        analysis["breakeven"]["warning"] = f"Log shows max profit {max_profit}pts but moved_to_breakeven=False in DB"

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

    # Always use scalp configs for analysis (per user preference)
    pair_config = get_pair_config(trade.symbol, is_scalp_trade=True)

    # Parse logs
    log_events = parse_trade_logs(trade_id)

    # Analyze stage activation
    stage_analysis = analyze_stage_activation(trade, log_events, pair_config)

    # Calculate distances and metrics
    entry_price = trade.entry_price or 0
    sl_price = trade.sl_price or 0
    tp_price = trade.tp_price or 0

    # Determine multiplier based on epic type
    # CEEM epics already have prices scaled (e.g., 11739.9 for EURUSD 1.17399)
    # MINI epics have standard prices (e.g., 1.17399 for EURUSD)
    if "CEEM" in trade.symbol:
        multiplier = 1
    elif "JPY" in trade.symbol:
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

    # Normalize prices for display (CEEM prices need to be divided by 10000)
    display_entry = entry_price / 10000 if "CEEM" in trade.symbol else entry_price
    display_sl = sl_price / 10000 if "CEEM" in trade.symbol else sl_price
    display_tp = tp_price / 10000 if "CEEM" in trade.symbol else tp_price

    # Build response
    response = {
        "trade_id": trade_id,
        "trade_details": {
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": display_entry,
            "sl_price": display_sl,
            "tp_price": display_tp,
            "status": trade.status,
            "profit_loss": trade.profit_loss,
            "pnl_currency": trade.pnl_currency,
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


def safe_json_parse(data: Any) -> Dict:
    """Safely parse JSON data that might already be a dict or a string"""
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {}
    return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


@router.get("/signal/{trade_id}")
async def get_signal_analysis(trade_id: int, db: Session = Depends(get_db)):
    """
    Get comprehensive signal analysis for a specific trade

    Analyzes the strategy signal that triggered this trade, including:
    - Smart Money Concepts validation
    - Confluence factors
    - Entry timing quality
    - Technical context at entry
    - Risk/reward setup

    Args:
        trade_id: Trade ID to analyze

    Returns:
        Detailed signal analysis with all SMC and technical data
    """
    # Get trade from database
    trade = db.query(TradeLog).filter(TradeLog.id == trade_id).first()

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    # Check if trade has an associated alert
    if not trade.alert_id:
        return {
            "trade_id": trade_id,
            "has_signal": False,
            "message": "This trade has no linked signal (alert_id is null)",
            "trade_details": {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "entry_price": safe_float(trade.entry_price),
                "status": trade.status
            }
        }

    # Get the alert/signal from database
    alert = db.query(AlertHistory).filter(AlertHistory.id == trade.alert_id).first()

    if not alert:
        return {
            "trade_id": trade_id,
            "has_signal": False,
            "message": f"Signal with alert_id={trade.alert_id} not found in database",
            "trade_details": {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "entry_price": safe_float(trade.entry_price),
                "status": trade.status
            }
        }

    # Parse JSON fields
    strategy_indicators = safe_json_parse(alert.strategy_indicators)
    strategy_metadata = safe_json_parse(alert.strategy_metadata)
    market_structure = safe_json_parse(alert.market_structure_analysis)
    order_flow = safe_json_parse(alert.order_flow_analysis)
    confluence = safe_json_parse(alert.confluence_details)
    signal_conditions = safe_json_parse(alert.signal_conditions)

    # Extract SMC data from strategy_indicators (primary source for SMC strategies)
    bos_choch = strategy_indicators.get('bos_choch', {}) or {}
    htf_data = strategy_indicators.get('htf_data', {}) or {}
    sr_data = strategy_indicators.get('sr_data', {}) or {}
    pattern_data = strategy_indicators.get('pattern_data', {}) or {}
    rr_data = strategy_indicators.get('rr_data', {}) or {}
    confidence_breakdown = strategy_indicators.get('confidence_breakdown', {}) or {}
    dataframe_analysis = strategy_indicators.get('dataframe_analysis', {}) or {}
    ema_data = dataframe_analysis.get('ema_data', {}) or {}
    df_sr_data = dataframe_analysis.get('sr_data', {}) or {}

    # Extract confluence factors
    confluence_factors = []
    confluence_score = 0.0

    if confluence:
        # Parse confluence details from SMC strategy
        factors_list = confluence.get('factors', [])
        if isinstance(factors_list, list):
            for factor in factors_list:
                if isinstance(factor, dict):
                    confluence_factors.append({
                        "name": factor.get('name', 'Unknown'),
                        "present": factor.get('present', False),
                        "weight": safe_float(factor.get('weight', 0)),
                        "details": factor.get('details', '')
                    })
        confluence_score = safe_float(confluence.get('total_score', 0))

    # If no structured confluence, build from strategy_indicators and metadata
    if not confluence_factors:
        # Check if this is SMC_SIMPLE strategy (has tier1_ema, tier2_swing, tier3_entry)
        tier1_ema = strategy_indicators.get('tier1_ema', {}) or {}
        tier2_swing = strategy_indicators.get('tier2_swing', {}) or {}
        tier3_entry = strategy_indicators.get('tier3_entry', {}) or {}
        risk_management = strategy_indicators.get('risk_management', {}) or {}

        is_smc_simple = bool(tier1_ema and tier2_swing)

        if is_smc_simple:
            # SMC_SIMPLE: Build confluence factors from 3-tier structure
            has_ema_bias = bool(tier1_ema.get('direction') in ['BULL', 'BEAR'])
            has_swing_break = bool(tier2_swing.get('body_close_confirmed', False))
            has_volume = bool(tier2_swing.get('volume_confirmed', False))
            has_pullback = bool(tier3_entry.get('pullback_depth', 0) > 0)
            has_optimal_zone = bool(tier3_entry.get('in_optimal_zone', False))
            has_rr = bool(safe_float(risk_management.get('rr_ratio', 0)) >= 1.5)

            smc_simple_factors = [
                ('ema_bias', '4H EMA Bias', has_ema_bias, tier1_ema.get('direction', '')),
                ('swing_break', 'Swing Break', has_swing_break, 'Body close confirmed'),
                ('volume', 'Volume Spike', has_volume, 'Above average volume'),
                ('pullback', 'Fib Pullback', has_pullback, f"{safe_float(tier3_entry.get('pullback_depth', 0))*100:.1f}%"),
                ('optimal_zone', 'Optimal Zone', has_optimal_zone, tier3_entry.get('fib_zone', '')),
                ('risk_reward', 'Risk/Reward', has_rr, f"R:R {safe_float(risk_management.get('rr_ratio', 0)):.2f}"),
            ]

            for key, name, present, details in smc_simple_factors:
                confluence_factors.append({
                    "name": name,
                    "present": bool(present),
                    "weight": 1.0 if present else 0.0,
                    "details": str(details) if details else ""
                })
        else:
            # SMC_STRUCTURE: Original confluence factors
            has_htf = bool(htf_data and htf_data.get('trend'))
            has_pattern = bool(pattern_data and pattern_data.get('pattern_type'))
            has_sr = bool(sr_data and sr_data.get('level_price'))
            has_rr = bool(rr_data and safe_float(rr_data.get('rr_ratio', 0)) >= 1.5)
            has_bos = bool(bos_choch and bos_choch.get('htf_direction'))

            smc_factors = [
                ('htf_alignment', 'HTF Alignment', has_htf, htf_data.get('trend', '')),
                ('bos_choch', 'BOS/ChoCH Structure', has_bos, bos_choch.get('structure_type', '')),
                ('pattern', 'Price Pattern', has_pattern, pattern_data.get('pattern_type', '')),
                ('sr_level', 'S/R Level', has_sr, sr_data.get('level_type', '')),
                ('risk_reward', 'Risk/Reward', has_rr, f"R:R {safe_float(rr_data.get('rr_ratio', 0)):.2f}"),
            ]

            # Also check metadata for additional factors
            if strategy_metadata:
                smc_factors.extend([
                    ('order_block', 'Order Block', strategy_metadata.get('has_order_block', False), ''),
                    ('fair_value_gap', 'Fair Value Gap', strategy_metadata.get('has_fvg', False), ''),
                    ('liquidity_sweep', 'Liquidity Sweep', strategy_metadata.get('liquidity_swept', False), ''),
                ])

            for key, name, present, details in smc_factors:
                confluence_factors.append({
                    "name": name,
                    "present": bool(present),
                    "weight": 1.0 if present else 0.0,
                    "details": str(details) if details else ""
                })

        # Calculate confluence score from confidence_breakdown if available
        if confidence_breakdown:
            confluence_score = safe_float(confidence_breakdown.get('total', 0))

    # Extract order block details
    order_block_details = None
    if order_flow:
        ob_data = order_flow.get('order_block', {}) or order_flow.get('nearest_order_block', {})
        if ob_data:
            order_block_details = {
                "type": ob_data.get('type', 'unknown'),
                "strength": ob_data.get('strength', 'unknown'),
                "tested_count": ob_data.get('tested_count', 0),
                "still_valid": ob_data.get('still_valid', ob_data.get('valid', True)),
                "fvg_confluence": safe_float(ob_data.get('fvg_confluence_score', 0)),
                "price_high": safe_float(ob_data.get('high', 0)),
                "price_low": safe_float(ob_data.get('low', 0))
            }

    # Extract FVG details
    fvg_details = None
    if order_flow:
        fvg_data = order_flow.get('fair_value_gap', {}) or order_flow.get('nearest_fvg', {})
        if fvg_data:
            fvg_details = {
                "type": fvg_data.get('type', 'unknown'),
                "size_pips": safe_float(fvg_data.get('size_pips', 0)),
                "status": fvg_data.get('status', 'unknown'),
                "confluence_score": safe_float(fvg_data.get('confluence_score', 0)),
                "price_high": safe_float(fvg_data.get('high', 0)),
                "price_low": safe_float(fvg_data.get('low', 0))
            }

    # Extract HTF alignment info - prefer strategy_indicators over strategy_metadata
    htf_info = strategy_metadata.get('htf_analysis', {}) or {}

    # Extract market intelligence data from strategy_metadata
    market_intel = strategy_metadata.get('market_intelligence', {}) or {}
    regime_analysis = market_intel.get('regime_analysis', {}) or {}
    session_analysis = market_intel.get('session_analysis', {}) or {}
    market_context = market_intel.get('market_context', {}) or {}

    # Detect if this is SMC_SIMPLE strategy
    tier1_ema = strategy_indicators.get('tier1_ema', {}) or {}
    tier2_swing = strategy_indicators.get('tier2_swing', {}) or {}
    tier3_entry = strategy_indicators.get('tier3_entry', {}) or {}
    risk_management = strategy_indicators.get('risk_management', {}) or {}
    is_smc_simple = bool(tier1_ema and tier2_swing)

    # Build entry timing from strategy_indicators
    # SMC_SIMPLE: tier1_ema.direction, SMC_STRUCTURE: htf_data.trend or bos_choch.htf_direction
    if is_smc_simple:
        htf_trend = tier1_ema.get('direction', 'unknown')  # "BULL" or "BEAR"
        ema_distance = safe_float(tier1_ema.get('distance_pips', 0))
        htf_strength = 0.8 if htf_trend in ['BULL', 'BEAR'] and ema_distance < 100 else 0.5 if htf_trend in ['BULL', 'BEAR'] else 0.0
        htf_aligned = htf_trend in ['BULL', 'BEAR']
        pullback_depth = safe_float(tier3_entry.get('pullback_depth', 0))
        in_pullback = pullback_depth > 0
    else:
        htf_trend = htf_data.get('trend', htf_info.get('trend', 'unknown'))
        htf_strength = safe_float(htf_data.get('strength', bos_choch.get('htf_strength', 0)))
        htf_aligned = bool(htf_trend and htf_trend != 'unknown')
        pullback_depth = safe_float(htf_data.get('pullback_depth', 0))
        in_pullback = htf_data.get('in_pullback', False)

    entry_timing = {
        "premium_discount_zone": strategy_metadata.get('zone', 'equilibrium'),
        "zone_quality": safe_float(strategy_metadata.get('zone_quality', 0.5)),
        "htf_aligned": htf_aligned,
        "htf_trend": htf_trend,
        "htf_strength": htf_strength,
        "htf_structure": bos_choch.get('structure_type', htf_info.get('structure', 'unknown')) if not is_smc_simple else tier3_entry.get('entry_type', 'PULLBACK'),
        "htf_bias": bos_choch.get('htf_direction', market_context.get('market_strength', {}).get('market_bias', 'neutral')) if not is_smc_simple else htf_trend,
        "in_pullback": in_pullback,
        "pullback_depth": pullback_depth,
        "mtf_alignment_ratio": safe_float(strategy_metadata.get('mtf_alignment', 0)),
        "entry_quality_score": safe_float(strategy_metadata.get('entry_quality', safe_float(alert.confidence_score)))
    }

    # Build market intelligence section
    market_intelligence = None
    if market_intel:
        market_strength = market_context.get('market_strength', {})
        market_intelligence = {
            "regime": {
                "dominant": regime_analysis.get('dominant_regime', 'unknown'),
                "confidence": safe_float(regime_analysis.get('confidence', 0)),
                "scores": regime_analysis.get('regime_scores', {})
            },
            "session": {
                "current": session_analysis.get('current_session', 'unknown'),
                "volatility": session_analysis.get('session_config', {}).get('volatility', 'unknown'),
                "risk_level": session_analysis.get('session_config', {}).get('risk_level', 'unknown')
            },
            "market_strength": {
                "trend_strength": safe_float(market_strength.get('average_trend_strength', 0)),
                "volatility": safe_float(market_strength.get('average_volatility', 0)),
                "market_bias": market_strength.get('market_bias', 'neutral'),
                "directional_consensus": safe_float(market_strength.get('directional_consensus', 0))
            },
            "correlation": market_context.get('correlation_analysis', {}),
            "intelligence_applied": market_intel.get('intelligence_applied', False)
        }

    # Calculate pip value based on pair
    symbol = trade.symbol or alert.epic or ''
    is_jpy = 'JPY' in symbol.upper()
    pip_multiplier = 100 if is_jpy else 10000

    # Calculate trade outcome
    trade_outcome = {
        "status": trade.status,
        "profit_loss": safe_float(trade.profit_loss),
        "pips_gained": safe_float(trade.pips_gained),
        "duration_minutes": trade.lifecycle_duration_minutes,
        "is_winner": safe_float(trade.profit_loss) > 0 if trade.profit_loss is not None else None,
        "exit_reason": "trailing_stop" if trade.moved_to_breakeven else "unknown"
    }

    # Build the response
    response = {
        "trade_id": trade_id,
        "alert_id": trade.alert_id,
        "has_signal": True,

        "signal_overview": {
            "timestamp": str(alert.alert_timestamp) if alert.alert_timestamp else None,
            "pair": alert.pair,
            "epic": alert.epic,
            "direction": alert.signal_type,
            "strategy": alert.strategy,
            "confidence_score": safe_float(alert.confidence_score),
            "enhanced_confidence": safe_float(alert.enhanced_confidence_score),
            "price_at_signal": safe_float(alert.price),
            "bid_price": safe_float(alert.bid_price),
            "ask_price": safe_float(alert.ask_price),
            "spread_pips": safe_float(alert.spread_pips),
            "timeframe": alert.timeframe,
            "signal_trigger": alert.signal_trigger,
            "market_session": alert.market_session
        },

        "smart_money_analysis": {
            "validated": bool(alert.smart_money_validated) or bool(bos_choch) or is_smc_simple,
            "type": alert.smart_money_type or bos_choch.get('structure_type') or (tier3_entry.get('entry_type') if is_smc_simple else None),
            "score": safe_float(alert.smart_money_score) or safe_float(confidence_breakdown.get('total', 0)),
            "market_structure": {
                "current_structure": bos_choch.get('htf_direction', market_structure.get('current_structure', htf_trend if is_smc_simple else 'unknown')),
                "structure_type": bos_choch.get('structure_type', market_structure.get('structure_type', tier3_entry.get('entry_type', 'unknown') if is_smc_simple else 'unknown')),
                "htf_trend": htf_trend,
                "htf_strength": htf_strength,
                "swing_high": safe_float(market_structure.get('swing_high', tier2_swing.get('swing_level', 0) if is_smc_simple else 0)),
                "swing_low": safe_float(market_structure.get('swing_low', 0)),
                "swing_highs": htf_data.get('swing_highs', [])[:5],  # Last 5 swing highs
                "swing_lows": htf_data.get('swing_lows', [])[:5],    # Last 5 swing lows
                "structure_breaks": market_structure.get('structure_breaks', []),
                "trend_strength": htf_strength
            }
        },

        "confluence_factors": {
            "total_score": confluence_score,
            "factors_present": sum(1 for f in confluence_factors if f.get('present', False)),
            "factors_total": len(confluence_factors),
            "factors": confluence_factors
        },

        "entry_timing": entry_timing,

        "technical_context": {
            # EMA data from strategy_indicators.dataframe_analysis.ema_data
            "ema_9": safe_float(ema_data.get('ema_9', alert.ema_short or 0)),
            "ema_21": safe_float(ema_data.get('ema_21', 0)),
            "ema_50": safe_float(ema_data.get('ema_50', alert.ema_long or 0)),
            "ema_100": safe_float(ema_data.get('ema_100', 0)),
            "ema_200": safe_float(ema_data.get('ema_200', alert.ema_trend or 0)),
            "price_vs_ema_50": "above" if safe_float(alert.price) > safe_float(ema_data.get('ema_50', 0)) else "below",
            "price_vs_ema_200": "above" if safe_float(alert.price) > safe_float(ema_data.get('ema_200', 0)) else "below",
            "macd": {
                "line": safe_float(alert.macd_line) or safe_float(dataframe_analysis.get('macd_data', {}).get('macd_line', 0)),
                "signal": safe_float(alert.macd_signal) or safe_float(dataframe_analysis.get('macd_data', {}).get('macd_signal', 0)),
                "histogram": safe_float(alert.macd_histogram) or safe_float(dataframe_analysis.get('macd_data', {}).get('macd_histogram', 0)),
                "direction": "bullish" if safe_float(alert.macd_histogram or 0) > 0 else "bearish"
            },
            "bollinger_bands": {
                "upper": safe_float(dataframe_analysis.get('other_indicators', {}).get('bb_upper', 0)),
                "middle": safe_float(dataframe_analysis.get('other_indicators', {}).get('bb_middle', 0)),
                "lower": safe_float(dataframe_analysis.get('other_indicators', {}).get('bb_lower', 0))
            },
            "rsi": safe_float(strategy_indicators.get('rsi', 50)),
            "rsi_zone": "overbought" if safe_float(strategy_indicators.get('rsi', 50)) > 70 else (
                "oversold" if safe_float(strategy_indicators.get('rsi', 50)) < 30 else "neutral"
            ),
            "atr": safe_float(strategy_indicators.get('atr', 0)),
            "volume": safe_float(alert.volume),
            "volume_ratio": safe_float(alert.volume_ratio),
            "volume_confirmation": bool(alert.volume_confirmation)
        },

        # Pattern detection from strategy_indicators
        "pattern_analysis": {
            "pattern_type": pattern_data.get('pattern_type'),
            "pattern_strength": safe_float(pattern_data.get('pattern_strength', 0)),
            "rejection_level": safe_float(pattern_data.get('rejection_level', 0)),
            "entry_price": safe_float(pattern_data.get('entry_price', 0))
        } if pattern_data else None,

        # S/R analysis from strategy_indicators
        "support_resistance": {
            "level_price": safe_float(sr_data.get('level_price', 0)),
            "level_type": sr_data.get('level_type', 'unknown'),
            "level_strength": safe_float(sr_data.get('level_strength', 0)),
            "distance_pips": safe_float(sr_data.get('distance_pips', 0)),
            "touch_count": sr_data.get('touch_count', 0),
            # Also include dataframe analysis S/R
            "nearest_support": safe_float(df_sr_data.get('nearest_support', alert.nearest_support or 0)),
            "nearest_resistance": safe_float(df_sr_data.get('nearest_resistance', alert.nearest_resistance or 0)),
            "distance_to_support_pips": safe_float(df_sr_data.get('distance_to_support_pips', alert.distance_to_support_pips or 0)),
            "distance_to_resistance_pips": safe_float(df_sr_data.get('distance_to_resistance_pips', alert.distance_to_resistance_pips or 0))
        },

        # Risk/Reward from strategy_indicators.rr_data (SMC_STRUCTURE) or risk_management (SMC_SIMPLE)
        "risk_reward": {
            "initial_rr": safe_float(rr_data.get('rr_ratio', 0)) or safe_float(risk_management.get('rr_ratio', 0)) or safe_float(alert.risk_reward_ratio or 0),
            "risk_pips": safe_float(rr_data.get('risk_pips', 0)) or safe_float(risk_management.get('risk_pips', 0)),
            "reward_pips": safe_float(rr_data.get('reward_pips', 0)) or safe_float(risk_management.get('reward_pips', 0)),
            "entry_price": safe_float(rr_data.get('entry_price', 0)) or safe_float(tier3_entry.get('entry_price', 0)) or safe_float(trade.entry_price or 0),
            "stop_loss": safe_float(rr_data.get('stop_loss', 0)) or safe_float(risk_management.get('stop_loss', 0)) or safe_float(trade.sl_price or 0),
            "take_profit": safe_float(rr_data.get('take_profit', 0)) or safe_float(risk_management.get('take_profit', 0)) or safe_float(trade.tp_price or 0),
            "partial_tp": safe_float(rr_data.get('partial_tp', 0)),
            "partial_percent": safe_float(rr_data.get('partial_percent', 0))
        },

        # Confidence breakdown from strategy_indicators
        "confidence_breakdown": {
            "total": safe_float(confidence_breakdown.get('total', alert.confidence_score or 0)),
            # SMC_SIMPLE breakdown
            "ema_alignment": safe_float(confidence_breakdown.get('ema_alignment', 0)),
            "volume_bonus": safe_float(confidence_breakdown.get('volume_bonus', 0)),
            "pullback_quality": safe_float(confidence_breakdown.get('pullback_quality', 0)),
            "rr_quality": safe_float(confidence_breakdown.get('rr_quality', 0)),
            "fib_accuracy": safe_float(confidence_breakdown.get('fib_accuracy', 0)),
            # SMC_STRUCTURE breakdown
            "htf_score": safe_float(confidence_breakdown.get('htf_score', 0)),
            "pattern_score": safe_float(confidence_breakdown.get('pattern_score', 0)),
            "sr_score": safe_float(confidence_breakdown.get('sr_score', 0)),
            "rr_score": safe_float(confidence_breakdown.get('rr_score', 0))
        } if confidence_breakdown else None,

        "order_block_details": order_block_details,
        "fair_value_gap_details": fvg_details,

        "market_intelligence": market_intelligence,

        "trade_outcome": trade_outcome,

        "claude_analysis": {
            "score": alert.claude_score,
            "decision": alert.claude_decision,
            "approved": alert.claude_approved,
            "reason": alert.claude_reason,
            "analysis_text": alert.claude_analysis
        },

        "raw_data": {
            "strategy_indicators": strategy_indicators,
            "strategy_metadata": strategy_metadata,
            "market_structure_analysis": market_structure,
            "order_flow_analysis": order_flow,
            "confluence_details": confluence,
            "signal_conditions": signal_conditions
        }
    }

    return response


@router.get("/outcome/{trade_id}")
async def get_trade_outcome_analysis(trade_id: int, db: Session = Depends(get_db)):
    """
    Get comprehensive outcome analysis for a trade - WHY it won or lost.

    This endpoint analyzes the trade to understand:
    - MFE/MAE: How much profit was available vs how much drawdown occurred
    - Entry quality: Was the signal strong?
    - Exit quality: Did we exit optimally?
    - Learning insights: What can we learn from this trade?

    Args:
        trade_id: Trade ID to analyze

    Returns:
        Comprehensive outcome analysis with actionable insights
    """
    # Get trade from database
    trade = db.query(TradeLog).filter(TradeLog.id == trade_id).first()

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    # Check if trade is closed
    if trade.status != "closed":
        return {
            "trade_id": trade_id,
            "status": "TRADE_STILL_OPEN",
            "message": f"Trade is still {trade.status}. Outcome analysis only available for closed trades.",
            "trade_details": {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "entry_price": safe_float(trade.entry_price),
                "status": trade.status
            }
        }

    # Get associated alert for signal data
    alert = None
    strategy_indicators = {}
    if trade.alert_id:
        alert = db.query(AlertHistory).filter(AlertHistory.id == trade.alert_id).first()
        if alert:
            strategy_indicators = safe_json_parse(alert.strategy_indicators)

    # Calculate MFE/MAE from candle data (using IGCandle table with 5m for precision)
    mfe_mae = calculate_mfe_mae(trade, db, IGCandle, timeframe=5)

    # Classify exit type
    exit_type = classify_exit_type(trade)

    # Assess entry quality
    entry_quality = assess_entry_quality(alert, strategy_indicators)

    # Assess exit quality
    exit_quality = assess_exit_quality(trade, mfe_mae, exit_type)

    # Generate learning insights
    insights = generate_learning_insights(
        trade, entry_quality, exit_quality, mfe_mae, strategy_indicators
    )

    # Get outcome summary
    outcome_summary = get_outcome_summary(trade, mfe_mae, exit_type)

    # Fetch candle data for visualization (using IGCandle with 5m candles)
    candle_data = fetch_trade_candles(trade, db, IGCandle, context_candles=10, timeframe=5)

    # Get market context from strategy metadata
    market_context = {}
    if alert:
        strategy_metadata = safe_json_parse(alert.strategy_metadata)
        market_intel = strategy_metadata.get('market_intelligence', {}) or {}
        market_context = {
            "regime_at_entry": market_intel.get('regime_analysis', {}).get('dominant_regime', 'unknown'),
            "session_at_entry": alert.market_session or market_intel.get('session_analysis', {}).get('current_session', 'unknown'),
            "volatility_percentile": safe_float(market_intel.get('market_context', {}).get('market_strength', {}).get('average_volatility', 0)),
            "htf_aligned": entry_quality.get('htf_aligned', False)
        }

    # Build the response
    response = {
        "trade_id": trade_id,
        "status": "ANALYSIS_COMPLETE",

        "outcome_summary": outcome_summary,

        "price_action_analysis": {
            "mfe": {
                "pips": mfe_mae.get('mfe_pips'),
                "price": mfe_mae.get('mfe_price'),
                "timestamp": mfe_mae.get('mfe_timestamp'),
                "time_to_peak_minutes": mfe_mae.get('mfe_time_to_peak_minutes'),
                "percentage_of_tp": mfe_mae.get('percentage_of_tp')
            },
            "mae": {
                "pips": mfe_mae.get('mae_pips'),
                "price": mfe_mae.get('mae_price'),
                "timestamp": mfe_mae.get('mae_timestamp'),
                "time_to_trough_minutes": mfe_mae.get('mae_time_to_trough_minutes'),
                "percentage_of_sl": mfe_mae.get('percentage_of_sl')
            },
            "mfe_mae_ratio": mfe_mae.get('mfe_mae_ratio'),
            "initial_move": mfe_mae.get('initial_move'),
            "immediate_reversal": mfe_mae.get('immediate_reversal'),
            "candle_count": mfe_mae.get('candle_count')
        },

        "entry_quality_assessment": {
            "score": entry_quality.get('score'),
            "max_score": entry_quality.get('max_score'),
            "percentage": entry_quality.get('percentage'),
            "verdict": entry_quality.get('verdict'),
            "factors": entry_quality.get('factors'),
            "htf_aligned": entry_quality.get('htf_aligned'),
            "confluence_count": entry_quality.get('confluence_count')
        },

        "exit_quality_assessment": {
            "exit_type": exit_quality.get('exit_type'),
            "mfe_pips": exit_quality.get('mfe_pips'),
            "actual_pips": exit_quality.get('actual_pips'),
            "missed_profit_pips": exit_quality.get('missed_profit_pips'),
            "exit_efficiency": exit_quality.get('exit_efficiency'),
            "exit_efficiency_pct": exit_quality.get('exit_efficiency_pct'),
            "optimal_exit_price": exit_quality.get('optimal_exit_price'),
            "actual_exit_price": exit_quality.get('actual_exit_price'),
            "stages_activated": exit_quality.get('stages_activated'),
            "verdict": exit_quality.get('verdict'),
            "verdict_details": exit_quality.get('verdict_details')
        },

        "learning_insights": {
            "trade_result": insights.get('trade_result'),
            "primary_factor": insights.get('primary_factor'),
            "pattern_identified": insights.get('pattern_identified'),
            "contributing_factors": insights.get('contributing_factors'),
            "what_went_right": insights.get('what_went_right'),
            "what_went_wrong": insights.get('what_went_wrong'),
            "improvement_suggestions": insights.get('improvement_suggestions'),
            "key_takeaway": insights.get('key_takeaway')
        },

        "market_context": market_context,

        "trade_details": {
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": safe_float(trade.entry_price),
            "exit_price": safe_float(trade.exit_price_calculated),
            "sl_price": safe_float(trade.sl_price),
            "tp_price": safe_float(trade.tp_price),
            "opened_at": str(trade.timestamp) if trade.timestamp else None,
            "closed_at": str(trade.closed_at) if trade.closed_at else None,
            "moved_to_breakeven": trade.moved_to_breakeven,
            "deal_id": trade.deal_id,
            "alert_id": trade.alert_id
        },

        "candle_data": candle_data
    }

    return response
