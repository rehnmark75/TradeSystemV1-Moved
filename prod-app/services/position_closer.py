"""
Position Closer Service - Weekend Protection
Automatically closes all open positions on Fridays at 20:30 UTC to prevent weekend exposure.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx
from .ig_orders import has_open_position
from .ig_auth import get_authenticated_headers
import config

logger = logging.getLogger(__name__)


class PositionCloser:
    """
    Service to automatically close all open positions on Fridays at 20:30 UTC
    for weekend protection.
    """

    def __init__(self):
        self.logger = logger
        self.positions_closed_count = 0
        self.last_closure_time = None
        self.closure_history = []

        # Configuration from config.py
        self.enable_position_closer = getattr(config, 'ENABLE_POSITION_CLOSER', True)
        self.closure_hour = getattr(config, 'POSITION_CLOSURE_HOUR_UTC', 20)
        self.closure_minute = getattr(config, 'POSITION_CLOSURE_MINUTE_UTC', 30)
        self.closure_weekday = getattr(config, 'POSITION_CLOSURE_WEEKDAY', 4)  # 0 = Monday, 4 = Friday

        # Timeout and retry settings
        self.timeout_seconds = getattr(config, 'POSITION_CLOSER_TIMEOUT_SECONDS', 60)
        self.max_retry_attempts = getattr(config, 'POSITION_CLOSER_MAX_RETRY_ATTEMPTS', 3)
        self.retry_delay_seconds = getattr(config, 'POSITION_CLOSER_RETRY_DELAY_SECONDS', 5)

        self.logger.info(f"✅ PositionCloser initialized for {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][self.closure_weekday]} {self.closure_hour:02d}:{self.closure_minute:02d} UTC weekend protection")

    async def should_close_positions_now(self) -> tuple[bool, str]:
        """
        Check if we should close positions now based on current time.

        Returns:
            Tuple of (should_close, reason)
        """
        try:
            now_utc = datetime.now(timezone.utc)
            current_weekday = now_utc.weekday()
            current_hour = now_utc.hour
            current_minute = now_utc.minute

            # Only close positions on the configured weekday (default: Friday)
            if current_weekday != self.closure_weekday:
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                return False, f"Not {weekday_names[self.closure_weekday]} (current weekday: {current_weekday}, need: {self.closure_weekday})"

            # Check if it's the configured closure time (allowing for a 5-minute window)
            if current_hour == self.closure_hour and self.closure_minute <= current_minute <= self.closure_minute + 5:
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                return True, f"{weekday_names[current_weekday]} {current_hour:02d}:{current_minute:02d} UTC - Weekend position closure time"

            return False, f"Not closure time (current: {current_hour:02d}:{current_minute:02d} UTC, target: {self.closure_hour:02d}:{self.closure_minute:02d} UTC)"

        except Exception as e:
            self.logger.error(f"❌ Error checking closure conditions: {e}")
            return False, f"Error checking conditions: {str(e)}"

    async def get_all_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all currently open positions from IG API.

        Returns:
            List of position dictionaries
        """
        try:
            auth_headers = await get_authenticated_headers()
            if not auth_headers:
                self.logger.error("❌ Failed to get authentication headers")
                return []

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{config.API_BASE_URL}/positions",
                    headers=auth_headers,
                    timeout=self.timeout_seconds
                )
                response.raise_for_status()

                data = response.json()
                positions = data.get("positions", [])

                self.logger.info(f"📊 Found {len(positions)} open positions")
                return positions

        except httpx.TimeoutException:
            self.logger.error("❌ Timeout getting open positions")
            return []
        except httpx.HTTPStatusError as e:
            self.logger.error(f"❌ HTTP error getting open positions: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            self.logger.error(f"❌ Error getting open positions: {e}")
            return []

    async def close_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Close a single position.

        Args:
            position: Position dictionary from IG API

        Returns:
            Dictionary with closure result
        """
        try:
            deal_id = position.get("position", {}).get("dealId")
            epic = position.get("market", {}).get("epic", "Unknown")
            direction = position.get("position", {}).get("direction")
            size = position.get("position", {}).get("size")

            if not deal_id:
                return {
                    "success": False,
                    "epic": epic,
                    "error": "No deal ID found in position"
                }

            # Opposite direction for closing
            close_direction = "SELL" if direction == "BUY" else "BUY"

            auth_headers = await get_authenticated_headers()
            if not auth_headers:
                return {
                    "success": False,
                    "epic": epic,
                    "deal_id": deal_id,
                    "error": "Failed to get authentication headers"
                }

            # Close position payload
            payload = {
                "dealId": deal_id,
                "direction": close_direction,
                "orderType": "MARKET",
                "size": size
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{config.API_BASE_URL}/positions/otc",
                    headers=auth_headers,
                    json=payload,
                    timeout=self.timeout_seconds
                )

                if response.status_code in [200, 201]:
                    result_data = response.json()
                    self.logger.info(f"✅ Closed position: {epic} ({direction}) Deal ID: {deal_id}")
                    return {
                        "success": True,
                        "epic": epic,
                        "deal_id": deal_id,
                        "direction": direction,
                        "size": size,
                        "close_direction": close_direction,
                        "response": result_data
                    }
                else:
                    error_msg = response.text
                    self.logger.error(f"❌ Failed to close position {deal_id}: {response.status_code} - {error_msg}")
                    return {
                        "success": False,
                        "epic": epic,
                        "deal_id": deal_id,
                        "error": f"HTTP {response.status_code}: {error_msg}"
                    }

        except httpx.TimeoutException:
            return {
                "success": False,
                "epic": epic,
                "deal_id": deal_id,
                "error": "Timeout closing position"
            }
        except Exception as e:
            self.logger.error(f"❌ Error closing position {deal_id}: {e}")
            return {
                "success": False,
                "epic": position.get("market", {}).get("epic", "Unknown"),
                "deal_id": deal_id,
                "error": str(e)
            }

    async def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions for weekend protection.

        Returns:
            Summary of closure operation
        """
        try:
            start_time = datetime.now(timezone.utc)

            # Check if we should close positions now
            should_close, reason = await self.should_close_positions_now()
            if not should_close:
                return {
                    "action": "skipped",
                    "reason": reason,
                    "timestamp": start_time.isoformat(),
                    "positions_closed": 0,
                    "positions_failed": 0
                }

            self.logger.info(f"🔒 Starting weekend position closure: {reason}")

            # Get all open positions
            positions = await self.get_all_open_positions()
            if not positions:
                self.logger.info("ℹ️ No open positions to close")
                return {
                    "action": "completed",
                    "reason": "No open positions found",
                    "timestamp": start_time.isoformat(),
                    "positions_closed": 0,
                    "positions_failed": 0,
                    "positions_checked": 0
                }

            # Close each position
            closure_results = []
            successful_closures = 0
            failed_closures = 0

            for position in positions:
                result = await self.close_position(position)
                closure_results.append(result)

                if result["success"]:
                    successful_closures += 1
                    self.positions_closed_count += 1
                else:
                    failed_closures += 1

                # Small delay between closures to avoid rate limiting
                await asyncio.sleep(0.5)

            # Update tracking
            self.last_closure_time = start_time

            # Save to history
            closure_summary = {
                "timestamp": start_time.isoformat(),
                "positions_checked": len(positions),
                "positions_closed": successful_closures,
                "positions_failed": failed_closures,
                "results": closure_results
            }

            self.closure_history.append(closure_summary)

            # Keep only last 10 closure events
            if len(self.closure_history) > 10:
                self.closure_history = self.closure_history[-10:]

            self.logger.info(f"🔒 Weekend closure complete: {successful_closures} closed, {failed_closures} failed")

            return {
                "action": "completed",
                "reason": reason,
                "timestamp": start_time.isoformat(),
                "positions_checked": len(positions),
                "positions_closed": successful_closures,
                "positions_failed": failed_closures,
                "results": closure_results,
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
            }

        except Exception as e:
            self.logger.error(f"❌ Error in close_all_positions: {e}")
            return {
                "action": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions_closed": 0,
                "positions_failed": 0
            }

    async def manual_close_all_positions(self) -> Dict[str, Any]:
        """
        Manually close all positions (for testing or emergency use).
        Bypasses the Friday 20:30 UTC check.

        Returns:
            Summary of closure operation
        """
        try:
            self.logger.warning("⚠️ Manual position closure initiated - bypassing time checks")

            # Temporarily disable time check by setting closure to current time
            now_utc = datetime.now(timezone.utc)
            original_hour = self.closure_hour
            original_minute = self.closure_minute
            original_weekday = self.closure_weekday

            # Set to current time to bypass check
            self.closure_hour = now_utc.hour
            self.closure_minute = now_utc.minute
            self.closure_weekday = now_utc.weekday()

            try:
                result = await self.close_all_positions()
                result["manual_override"] = True
                return result
            finally:
                # Restore original settings
                self.closure_hour = original_hour
                self.closure_minute = original_minute
                self.closure_weekday = original_weekday

        except Exception as e:
            self.logger.error(f"❌ Error in manual position closure: {e}")
            return {
                "action": "error",
                "error": str(e),
                "manual_override": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the position closer.

        Returns:
            Status dictionary
        """
        now_utc = datetime.now(timezone.utc)

        return {
            "enabled": self.enable_position_closer,
            "current_time_utc": now_utc.isoformat(),
            "next_closure_time": "Fridays at 20:30 UTC",
            "closure_schedule": {
                "weekday": self.friday_weekday,
                "hour": self.closure_hour,
                "minute": self.closure_minute,
                "timezone": "UTC"
            },
            "statistics": {
                "total_positions_closed": self.positions_closed_count,
                "last_closure_time": self.last_closure_time.isoformat() if self.last_closure_time else None,
                "closure_history_count": len(self.closure_history)
            },
            "next_friday": self._get_next_friday_closure_time(),
            "time_until_next_closure": self._get_time_until_next_closure()
        }

    def _get_next_friday_closure_time(self) -> str:
        """Get the next closure time as ISO string"""
        now_utc = datetime.now(timezone.utc)
        days_until_closure = (self.closure_weekday - now_utc.weekday()) % 7
        if days_until_closure == 0 and (now_utc.hour > self.closure_hour or
                                       (now_utc.hour == self.closure_hour and now_utc.minute >= self.closure_minute)):
            days_until_closure = 7  # Next week if we've passed this week's closure time

        next_closure = now_utc.replace(hour=self.closure_hour, minute=self.closure_minute, second=0, microsecond=0)
        next_closure = next_closure.replace(day=next_closure.day + days_until_closure)
        return next_closure.isoformat()

    def _get_time_until_next_closure(self) -> str:
        """Get human-readable time until next closure"""
        try:
            next_closure = datetime.fromisoformat(self._get_next_friday_closure_time().replace('Z', '+00:00'))
            now_utc = datetime.now(timezone.utc)
            delta = next_closure - now_utc

            days = delta.days
            hours, remainder = divmod(delta.seconds, 3600)
            minutes = remainder // 60

            if days > 0:
                return f"{days} days, {hours} hours, {minutes} minutes"
            elif hours > 0:
                return f"{hours} hours, {minutes} minutes"
            else:
                return f"{minutes} minutes"
        except Exception:
            return "Unable to calculate"

    def get_recent_closures(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent closure history.

        Args:
            limit: Maximum number of recent closures to return

        Returns:
            List of recent closure events
        """
        return self.closure_history[-limit:] if self.closure_history else []


# Global instance
position_closer = PositionCloser()


# Convenience functions for use in FastAPI routes
async def check_and_close_positions() -> Dict[str, Any]:
    """Check if it's time and close all positions if needed."""
    return await position_closer.close_all_positions()


async def manual_close_positions() -> Dict[str, Any]:
    """Manually close all positions (emergency/testing)."""
    return await position_closer.manual_close_all_positions()


def get_position_closer_status() -> Dict[str, Any]:
    """Get position closer status."""
    return position_closer.get_status()


def get_position_closure_history(limit: int = 5) -> List[Dict[str, Any]]:
    """Get recent position closure history."""
    return position_closer.get_recent_closures(limit)