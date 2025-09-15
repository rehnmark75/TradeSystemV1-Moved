"""
Stream Manager for handling IG trading streams and trade tracking.
This module provides functionality for streaming prices and tracking live trades.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Global variables for tracking streams and trades
tracked_trades_by_epic: Dict[str, Dict[str, Any]] = {}
stream_status = {
    "running": False,
    "epic": None,
    "tick_count": 0
}
active_stream = {
    "client": None,
    "subscription": None
}
task_registry: Dict[str, Any] = {}
active_trades: Dict[str, Any] = {}

logger = logging.getLogger(__name__)

def track_trade_subscription(
    deal_id: str,
    epic: str,
    entry_price: float,
    direction: str,
    headers: Dict[str, str],
    trailing_distance: int = 15
):
    """
    Track a live trade with trailing stop functionality.

    Args:
        deal_id: Unique identifier for the trade
        epic: Market epic (e.g., 'IX.D.DAX.DAILY.IP')
        entry_price: Price at which the trade was entered
        direction: 'BUY' or 'SELL'
        headers: Authentication headers (CST, X-SECURITY-TOKEN, etc.)
        trailing_distance: Distance in points for trailing stop
    """
    try:
        # Store trade information
        if epic not in tracked_trades_by_epic:
            tracked_trades_by_epic[epic] = {}

        tracked_trades_by_epic[epic][deal_id] = {
            "epic": epic,
            "entry_price": entry_price,
            "direction": direction,
            "headers": headers,
            "trailing_distance": trailing_distance,
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }

        # Store in active trades for easy access
        active_trades[deal_id] = {
            "epic": epic,
            "entry_price": entry_price,
            "direction": direction,
            "trailing_distance": trailing_distance
        }

        logger.info(f"Started tracking trade {deal_id} for {epic} at {entry_price} ({direction})")

    except Exception as e:
        logger.error(f"Failed to track trade {deal_id}: {str(e)}")
        raise

async def stream_prices(
    epic: str,
    headers: Dict[str, str],
    entry_price: float,
    direction: str,
    sl_price: float,
    tp_price: float,
    deal_id: str
):
    """
    Stream prices for a specific epic and monitor trade conditions.

    Args:
        epic: Market epic to stream
        headers: Authentication headers
        entry_price: Entry price of the trade
        direction: Trade direction ('BUY' or 'SELL')
        sl_price: Stop loss price
        tp_price: Take profit price
        deal_id: Trade identifier
    """
    try:
        # Update stream status
        stream_status["running"] = True
        stream_status["epic"] = epic
        stream_status["tick_count"] = 0

        logger.info(f"Starting price stream for {epic} (Deal: {deal_id})")

        # Store stream information
        active_stream["epic"] = epic
        active_stream["deal_id"] = deal_id
        active_stream["entry_price"] = entry_price
        active_stream["direction"] = direction
        active_stream["sl_price"] = sl_price
        active_stream["tp_price"] = tp_price

        # This is a placeholder implementation
        # In a real implementation, this would connect to IG's Lightstreamer
        # For now, we'll just simulate the stream being active

        # Simulate some price updates
        for i in range(10):
            if not stream_status["running"]:
                break

            stream_status["tick_count"] += 1
            await asyncio.sleep(1)

            # Log periodic updates
            if i % 5 == 0:
                logger.info(f"Price stream active for {epic} - Tick count: {stream_status['tick_count']}")

        logger.info(f"Price stream completed for {epic}")

    except Exception as e:
        logger.error(f"Error in price stream for {epic}: {str(e)}")
        stream_status["running"] = False
        raise
    finally:
        # Clean up stream status
        if stream_status.get("epic") == epic:
            stream_status["running"] = False

def stop_stream():
    """Stop the active price stream."""
    try:
        stream_status["running"] = False
        if active_stream.get("client"):
            # In a real implementation, disconnect from Lightstreamer
            active_stream["client"] = None
            active_stream["subscription"] = None

        logger.info("Price stream stopped")

    except Exception as e:
        logger.error(f"Error stopping stream: {str(e)}")

def get_tracked_trades():
    """Get all currently tracked trades."""
    return tracked_trades_by_epic

def stop_tracking_trade(deal_id: str):
    """
    Stop tracking a specific trade.

    Args:
        deal_id: Trade identifier to stop tracking
    """
    try:
        # Remove from active trades
        if deal_id in active_trades:
            del active_trades[deal_id]

        # Remove from tracked trades by epic
        for epic, trades in tracked_trades_by_epic.items():
            if deal_id in trades:
                del trades[deal_id]
                logger.info(f"Stopped tracking trade {deal_id}")
                break

    except Exception as e:
        logger.error(f"Error stopping tracking for trade {deal_id}: {str(e)}")

def get_stream_status():
    """Get current stream status."""
    return stream_status.copy()

def get_active_stream_info():
    """Get active stream information."""
    return active_stream.copy()