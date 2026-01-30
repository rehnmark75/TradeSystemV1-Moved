# services/stop_verification_service.py
"""
Stop Verification Service - Verify IG actually applied stop updates

CRITICAL: This service addresses the gap where IG returns HTTP 200 OK
but may not have actually changed the stop level (due to spread, min distance, etc.)

Created: Jan 2026 as part of bulletproof trailing system
"""

import logging
import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from config import API_BASE_URL

logger = logging.getLogger("stop_verification")
logger.setLevel(logging.INFO)


@dataclass
class VerificationResult:
    """Result of stop verification check"""
    matched: bool
    expected_stop: Optional[float]
    actual_stop: Optional[float]
    expected_limit: Optional[float]
    actual_limit: Optional[float]
    mismatch_pips: float
    epic: str
    deal_id: str
    verification_time: datetime
    error: Optional[str] = None
    position_exists: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "expected_stop": self.expected_stop,
            "actual_stop": self.actual_stop,
            "expected_limit": self.expected_limit,
            "actual_limit": self.actual_limit,
            "mismatch_pips": self.mismatch_pips,
            "epic": self.epic,
            "deal_id": self.deal_id,
            "verification_time": self.verification_time.isoformat(),
            "error": self.error,
            "position_exists": self.position_exists
        }


class StopVerificationService:
    """
    Service to verify that stop updates were actually applied by IG.

    After sending a stop update to IG, this service reads back the position
    to confirm the stop level matches what was sent.
    """

    def __init__(self):
        self.logger = logger

    def _normalize_ceem_price(self, price: float, epic: str) -> float:
        """
        Normalize CEEM prices from IG format to standard format.
        IG returns CEEM prices scaled (e.g., 11939.6 instead of 1.19396)
        """
        if "CEEM" in epic and price > 1000:
            return price / 10000.0
        return price

    def _denormalize_ceem_price(self, price: float, epic: str) -> float:
        """
        Denormalize price back to CEEM points format for comparison.
        Our system uses standard format (1.19396) but IG uses points (11939.6)
        """
        if "CEEM" in epic and price < 100:
            return price * 10000.0
        return price

    def _calculate_pip_value(self, epic: str) -> float:
        """Get pip value for the instrument"""
        if "JPY" in epic:
            return 0.01
        return 0.0001

    def _calculate_mismatch_pips(
        self,
        expected: float,
        actual: float,
        epic: str
    ) -> float:
        """Calculate mismatch in pips between expected and actual values"""
        if expected is None or actual is None:
            return 0.0

        pip_value = self._calculate_pip_value(epic)
        diff = abs(expected - actual)
        return round(diff / pip_value, 1)

    async def read_position(
        self,
        deal_id: str,
        headers: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Read current position from IG by deal_id.
        Returns position data or None if not found.
        """
        try:
            url = f"{API_BASE_URL}/positions"
            request_headers = {
                "X-IG-API-KEY": headers["X-IG-API-KEY"],
                "CST": headers["CST"],
                "X-SECURITY-TOKEN": headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=request_headers)
                response.raise_for_status()

                positions = response.json().get("positions", [])

                for pos in positions:
                    if pos["position"]["dealId"] == deal_id:
                        return pos

                return None

        except httpx.TimeoutException:
            self.logger.error(f"[VERIFY] Timeout reading position {deal_id}")
            return None
        except Exception as e:
            self.logger.error(f"[VERIFY] Error reading position {deal_id}: {e}")
            return None

    async def verify_stop_update(
        self,
        deal_id: str,
        expected_stop: Optional[float],
        expected_limit: Optional[float],
        epic: str,
        headers: dict,
        tolerance_pips: float = 0.5
    ) -> VerificationResult:
        """
        Verify that a stop update was actually applied by IG.

        Args:
            deal_id: The IG deal ID
            expected_stop: The stop level we sent to IG
            expected_limit: The limit level we sent to IG (optional)
            epic: The instrument epic (for CEEM normalization)
            headers: IG API headers
            tolerance_pips: Acceptable difference in pips (default 0.5)

        Returns:
            VerificationResult with match status and details
        """
        verification_time = datetime.utcnow()

        # Read current position from IG
        position = await self.read_position(deal_id, headers)

        if position is None:
            self.logger.warning(f"[VERIFY] Position {deal_id} not found on IG")
            return VerificationResult(
                matched=False,
                expected_stop=expected_stop,
                actual_stop=None,
                expected_limit=expected_limit,
                actual_limit=None,
                mismatch_pips=0,
                epic=epic,
                deal_id=deal_id,
                verification_time=verification_time,
                error="Position not found",
                position_exists=False
            )

        # Extract actual values from IG
        position_data = position.get("position", {})
        ig_stop_raw = position_data.get("stopLevel")
        ig_limit_raw = position_data.get("limitLevel")

        # Normalize CEEM prices
        actual_stop = self._normalize_ceem_price(float(ig_stop_raw), epic) if ig_stop_raw else None
        actual_limit = self._normalize_ceem_price(float(ig_limit_raw), epic) if ig_limit_raw else None

        # Normalize expected values for comparison (in case they're in points format)
        expected_stop_normalized = self._normalize_ceem_price(expected_stop, epic) if expected_stop else None
        expected_limit_normalized = self._normalize_ceem_price(expected_limit, epic) if expected_limit else None

        # Calculate mismatch
        stop_mismatch_pips = self._calculate_mismatch_pips(
            expected_stop_normalized, actual_stop, epic
        )
        limit_mismatch_pips = self._calculate_mismatch_pips(
            expected_limit_normalized, actual_limit, epic
        )

        # Check if within tolerance
        stop_matched = stop_mismatch_pips <= tolerance_pips if expected_stop else True
        limit_matched = limit_mismatch_pips <= tolerance_pips if expected_limit else True
        matched = stop_matched and limit_matched

        # Calculate total mismatch (use the larger of the two)
        total_mismatch = max(stop_mismatch_pips, limit_mismatch_pips)

        # Log result
        if matched:
            self.logger.info(
                f"[VERIFY OK] {epic} deal={deal_id}: "
                f"Stop expected={expected_stop_normalized}, actual={actual_stop} (diff={stop_mismatch_pips}pips)"
            )
        else:
            self.logger.error(
                f"[VERIFY MISMATCH] {epic} deal={deal_id}: "
                f"Stop expected={expected_stop_normalized}, actual={actual_stop} (diff={stop_mismatch_pips}pips) | "
                f"Limit expected={expected_limit_normalized}, actual={actual_limit} (diff={limit_mismatch_pips}pips)"
            )

        return VerificationResult(
            matched=matched,
            expected_stop=expected_stop_normalized,
            actual_stop=actual_stop,
            expected_limit=expected_limit_normalized,
            actual_limit=actual_limit,
            mismatch_pips=total_mismatch,
            epic=epic,
            deal_id=deal_id,
            verification_time=verification_time,
            error=None,
            position_exists=True
        )

    async def verify_with_retry(
        self,
        deal_id: str,
        expected_stop: Optional[float],
        expected_limit: Optional[float],
        epic: str,
        headers: dict,
        max_retries: int = 3,
        retry_delay_ms: int = 500,
        tolerance_pips: float = 0.5
    ) -> VerificationResult:
        """
        Verify stop update with retries.

        Sometimes IG takes a moment to update the position.
        This method retries verification a few times before failing.
        """
        import asyncio

        last_result = None

        for attempt in range(max_retries):
            if attempt > 0:
                await asyncio.sleep(retry_delay_ms / 1000.0)
                self.logger.info(f"[VERIFY RETRY] Attempt {attempt + 1}/{max_retries} for {deal_id}")

            result = await self.verify_stop_update(
                deal_id=deal_id,
                expected_stop=expected_stop,
                expected_limit=expected_limit,
                epic=epic,
                headers=headers,
                tolerance_pips=tolerance_pips
            )

            last_result = result

            if result.matched:
                return result

            # If position doesn't exist, no point retrying
            if not result.position_exists:
                return result

        # All retries failed
        self.logger.error(
            f"[VERIFY FAILED] {epic} deal={deal_id}: "
            f"Verification failed after {max_retries} attempts. "
            f"Mismatch: {last_result.mismatch_pips} pips"
        )

        return last_result


# Singleton instance for easy import
_verification_service: Optional[StopVerificationService] = None


def get_verification_service() -> StopVerificationService:
    """Get singleton instance of verification service"""
    global _verification_service
    if _verification_service is None:
        _verification_service = StopVerificationService()
    return _verification_service
