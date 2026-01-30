# services/mismatch_detector.py
"""
Mismatch Detector - Detect DB/IG discrepancies during active trailing

CRITICAL: This service addresses the blind spot where sync is skipped during
active trailing, meaning mismatches go undetected.

This detector:
- Compares database sl_price with actual IG stopLevel
- Does NOT auto-fix (trailing system remains source of truth)
- Logs and alerts on mismatches for investigation

Created: Jan 2026 as part of bulletproof trailing system
"""

import logging
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from config import API_BASE_URL
from services.models import TradeLog

logger = logging.getLogger("mismatch_detector")
logger.setLevel(logging.INFO)


class MismatchSeverity(Enum):
    """Severity levels for mismatches"""
    NONE = "none"           # No mismatch or within tolerance
    MINOR = "minor"         # < 2 pips - log only
    MAJOR = "major"         # 2-10 pips - Telegram alert
    CRITICAL = "critical"   # > 10 pips - immediate attention


@dataclass
class MismatchReport:
    """Detailed report of a DB/IG mismatch"""
    trade_id: int
    deal_id: str
    epic: str
    direction: str
    db_stop: Optional[float]
    ig_stop: Optional[float]
    db_limit: Optional[float]
    ig_limit: Optional[float]
    stop_mismatch_pips: float
    limit_mismatch_pips: float
    severity: MismatchSeverity
    detected_at: datetime
    trailing_active: bool
    position_exists: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "deal_id": self.deal_id,
            "epic": self.epic,
            "direction": self.direction,
            "db_stop": self.db_stop,
            "ig_stop": self.ig_stop,
            "db_limit": self.db_limit,
            "ig_limit": self.ig_limit,
            "stop_mismatch_pips": self.stop_mismatch_pips,
            "limit_mismatch_pips": self.limit_mismatch_pips,
            "severity": self.severity.value,
            "detected_at": self.detected_at.isoformat(),
            "trailing_active": self.trailing_active,
            "position_exists": self.position_exists,
            "error": self.error
        }

    def is_mismatched(self) -> bool:
        return self.severity != MismatchSeverity.NONE


class MismatchDetector:
    """
    Detect mismatches between database and IG without auto-fixing.

    The trailing system is the source of truth during active trailing.
    This detector only identifies discrepancies for alerting/investigation.
    """

    # Thresholds for severity classification (in pips)
    MINOR_THRESHOLD = 2.0
    MAJOR_THRESHOLD = 5.0
    CRITICAL_THRESHOLD = 10.0

    def __init__(self):
        self.logger = logger
        self._position_cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 10  # Short TTL for mismatch detection

    def _normalize_ceem_price(self, price: float, epic: str) -> float:
        """Normalize CEEM prices from IG format"""
        if "CEEM" in epic and price and price > 1000:
            return price / 10000.0
        return price

    def _calculate_pip_value(self, epic: str) -> float:
        """Get pip value for the instrument"""
        if "JPY" in epic:
            return 0.01
        return 0.0001

    def _calculate_mismatch_pips(
        self,
        db_value: Optional[float],
        ig_value: Optional[float],
        epic: str
    ) -> float:
        """Calculate mismatch in pips"""
        if db_value is None or ig_value is None:
            return 0.0

        pip_value = self._calculate_pip_value(epic)
        diff = abs(db_value - ig_value)
        return round(diff / pip_value, 1)

    def _classify_severity(self, mismatch_pips: float) -> MismatchSeverity:
        """Classify mismatch severity based on pip difference"""
        if mismatch_pips < 0.5:  # Within tolerance
            return MismatchSeverity.NONE
        elif mismatch_pips < self.MINOR_THRESHOLD:
            return MismatchSeverity.MINOR
        elif mismatch_pips < self.CRITICAL_THRESHOLD:
            return MismatchSeverity.MAJOR
        else:
            return MismatchSeverity.CRITICAL

    async def _get_positions(self, headers: dict) -> List[Dict]:
        """Fetch current positions from IG with short cache"""
        now = datetime.utcnow()

        # Check cache
        if (self._cache_time and
            (now - self._cache_time).total_seconds() < self._cache_ttl_seconds and
            self._position_cache):
            return list(self._position_cache.values())

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

            # Update cache
            self._position_cache = {
                p["position"]["dealId"]: p for p in positions
            }
            self._cache_time = now

            return positions

        except Exception as e:
            self.logger.error(f"[MISMATCH] Error fetching positions: {e}")
            return list(self._position_cache.values()) if self._position_cache else []

    async def check_mismatch(
        self,
        trade: TradeLog,
        headers: dict
    ) -> MismatchReport:
        """
        Check for mismatch between database and IG for a single trade.

        Args:
            trade: TradeLog instance from database
            headers: IG API headers

        Returns:
            MismatchReport with details of any discrepancy
        """
        detected_at = datetime.utcnow()
        trailing_active = (
            getattr(trade, 'moved_to_breakeven', False) or
            getattr(trade, 'early_be_executed', False)
        )

        # Get IG positions
        positions = await self._get_positions(headers)

        # Find matching position
        ig_position = None
        for pos in positions:
            if pos["position"]["dealId"] == trade.deal_id:
                ig_position = pos
                break

        # Position not found on IG
        if ig_position is None:
            self.logger.warning(
                f"[MISMATCH] Trade {trade.id} ({trade.deal_id}) not found on IG - "
                f"may be closed"
            )
            return MismatchReport(
                trade_id=trade.id,
                deal_id=trade.deal_id,
                epic=trade.symbol,
                direction=trade.direction,
                db_stop=trade.sl_price,
                ig_stop=None,
                db_limit=trade.tp_price,
                ig_limit=None,
                stop_mismatch_pips=0,
                limit_mismatch_pips=0,
                severity=MismatchSeverity.NONE,
                detected_at=detected_at,
                trailing_active=trailing_active,
                position_exists=False,
                error="Position not found on IG"
            )

        # Extract IG values
        position_data = ig_position["position"]
        ig_stop_raw = position_data.get("stopLevel")
        ig_limit_raw = position_data.get("limitLevel")

        # Normalize CEEM prices
        ig_stop = self._normalize_ceem_price(float(ig_stop_raw), trade.symbol) if ig_stop_raw else None
        ig_limit = self._normalize_ceem_price(float(ig_limit_raw), trade.symbol) if ig_limit_raw else None

        # Calculate mismatches
        stop_mismatch = self._calculate_mismatch_pips(trade.sl_price, ig_stop, trade.symbol)
        limit_mismatch = self._calculate_mismatch_pips(trade.tp_price, ig_limit, trade.symbol)

        # Use the more severe mismatch
        max_mismatch = max(stop_mismatch, limit_mismatch)
        severity = self._classify_severity(max_mismatch)

        # Log based on severity
        if severity == MismatchSeverity.NONE:
            self.logger.debug(
                f"[MISMATCH OK] Trade {trade.id}: DB stop={trade.sl_price}, "
                f"IG stop={ig_stop} (diff={stop_mismatch}pips)"
            )
        elif severity == MismatchSeverity.MINOR:
            self.logger.info(
                f"[MISMATCH MINOR] Trade {trade.id}: DB stop={trade.sl_price}, "
                f"IG stop={ig_stop} (diff={stop_mismatch}pips)"
            )
        elif severity == MismatchSeverity.MAJOR:
            self.logger.warning(
                f"[MISMATCH MAJOR] Trade {trade.id} {trade.symbol}: "
                f"DB stop={trade.sl_price}, IG stop={ig_stop} (diff={stop_mismatch}pips) | "
                f"trailing_active={trailing_active}"
            )
        else:  # CRITICAL
            self.logger.error(
                f"[MISMATCH CRITICAL] Trade {trade.id} {trade.symbol}: "
                f"DB stop={trade.sl_price}, IG stop={ig_stop} (diff={stop_mismatch}pips) | "
                f"DB limit={trade.tp_price}, IG limit={ig_limit} (diff={limit_mismatch}pips) | "
                f"trailing_active={trailing_active}"
            )

        return MismatchReport(
            trade_id=trade.id,
            deal_id=trade.deal_id,
            epic=trade.symbol,
            direction=trade.direction,
            db_stop=trade.sl_price,
            ig_stop=ig_stop,
            db_limit=trade.tp_price,
            ig_limit=ig_limit,
            stop_mismatch_pips=stop_mismatch,
            limit_mismatch_pips=limit_mismatch,
            severity=severity,
            detected_at=detected_at,
            trailing_active=trailing_active,
            position_exists=True
        )

    async def check_all_trades(
        self,
        trades: List[TradeLog],
        headers: dict
    ) -> List[MismatchReport]:
        """
        Check mismatches for multiple trades efficiently.

        Pre-fetches all positions once, then checks each trade.
        """
        # Pre-warm the cache
        await self._get_positions(headers)

        reports = []
        for trade in trades:
            if trade.deal_id:  # Only check trades with deal_ids
                report = await self.check_mismatch(trade, headers)
                reports.append(report)

        # Summary logging
        mismatches = [r for r in reports if r.is_mismatched()]
        if mismatches:
            by_severity = {}
            for r in mismatches:
                by_severity[r.severity.value] = by_severity.get(r.severity.value, 0) + 1

            self.logger.warning(
                f"[MISMATCH SUMMARY] {len(mismatches)}/{len(reports)} trades with mismatches: "
                f"{by_severity}"
            )
        else:
            self.logger.debug(f"[MISMATCH SUMMARY] All {len(reports)} trades in sync")

        return reports

    def clear_cache(self):
        """Clear position cache to force fresh fetch"""
        self._position_cache = {}
        self._cache_time = None


# Singleton instance
_mismatch_detector: Optional[MismatchDetector] = None


def get_mismatch_detector() -> MismatchDetector:
    """Get singleton instance of mismatch detector"""
    global _mismatch_detector
    if _mismatch_detector is None:
        _mismatch_detector = MismatchDetector()
    return _mismatch_detector
