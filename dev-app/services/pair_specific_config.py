import time

import requests
import httpx
import json
import threading
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from logging.handlers import RotatingFileHandler
from config import API_BASE_URL

from sqlalchemy.orm import Session
from services.db import SessionLocal
from services.models import TradeLog, IGCandle  # ✅ CORRECT: IGCandle model for ig_candles table
from services.shared_types import ApiConfig, TradeMonitorLogger


# Import utils functions
from utils import get_point_value, convert_price_to_points  # ✅ ADDED: Import utils

# Import advanced trailing components with fallback handling
ENHANCED_PROCESSOR_AVAILABLE = False
CombinedTradeProcessor = None
CombinedTrailingConfig = None
SCALPING_CONFIG_WITH_EMA = None

try:
    from enhanced_trade_processor import CombinedTradeProcessor, CombinedTrailingConfig, SCALPING_CONFIG_WITH_EMA
    from dependencies import get_ig_auth_headers
    ENHANCED_PROCESSOR_AVAILABLE = True
    print("✅ Enhanced trade processor imported successfully")
except ImportError as e:
    print(f"Warning: Could not import enhanced trailing components: {e}")
    print("🔄 Loading fallback trailing system...")
    try:
        from trailing_class import EnhancedTradeProcessor, TrailingConfig, SCALPING_CONFIG
        CombinedTradeProcessor = EnhancedTradeProcessor
        CombinedTrailingConfig = TrailingConfig
        SCALPING_CONFIG_WITH_EMA = SCALPING_CONFIG
        from dependencies import get_ig_auth_headers
        print("✅ Fallback trailing system loaded")
    except ImportError as fallback_error:
        print(f"❌ Critical: Could not load fallback trailing system: {fallback_error}")
        print("⚠️ Trade monitoring will be disabled")

# Pair-specific configuration is not implemented yet
# The functions get_fastapi_config_manager, get_config_for_trade, analyze_trade_pair
# do not exist - this feature was planned but not completed
PAIR_CONFIG_AVAILABLE = False

@dataclass
class ApiConfig:
    """Configuration for API calls"""
    base_url: str = "http://fastapi-dev:8000"
    subscription_key: str = "436abe054a074894a0517e5172f0e5b6"
    dry_run: bool = False  # Set to False for live trading
    
    @property
    def adjust_stop_url(self) -> str:
        """Get the full URL for the adjust-stop endpoint"""
        return f"{self.base_url}/orders/adjust-stop"


class PositionCache:
    """Efficient position caching to minimize IG API calls"""
    
    def __init__(self, cache_duration_seconds: int = 30):
        self.cache_duration = cache_duration_seconds
        self._positions_cache = None
        self._last_fetch_time = None
        self._lock = threading.Lock()
        
    def is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._last_fetch_time or not self._positions_cache:
            return False
        
        return (datetime.now() - self._last_fetch_time).total_seconds() < self.cache_duration
    
    async def get_positions(self, trading_headers: dict, force_refresh: bool = False) -> Optional[List[Dict]]:
        """Get positions with caching to minimize API calls"""
        with self._lock:
            if not force_refresh and self.is_cache_valid():
                return self._positions_cache
        
        try:
            import httpx
            
            
            url = f"{API_BASE_URL}/positions"
            headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                positions = data.get("positions", [])
                
                # ✅ FIX: Atomic cache update
                with self._lock:
                    self._positions_cache = positions
                    self._last_fetch_time = datetime.now()
                
                return positions
                
        except Exception as e:
            logging.getLogger("trade_monitor").error(f"[POSITION CACHE ERROR] {e}")
            # Return stale cache if available, but don't update timestamp
            with self._lock:
                return self._positions_cache


class TradeMonitorLogger:
    """Centralized logging setup"""

    def __init__(self, log_file: str = "/app/logs/trade_monitor.log"):
        self.logger = logging.getLogger("trade_monitor")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


class OrderSender:
    """Handles sending orders to the FastAPI /adjust-stop endpoint"""

    def __init__(self, config: ApiConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._last_sent_offsets: Dict[str, Tuple[int, int]] = {}
        self._lock = threading.Lock()

    def _get_adjust_direction(self, trade_direction: str, target: str = "limit") -> str:
        """
        Determine adjustment direction for tightening stops/limits
        """
        if target == "limit":
            return "increase"
        elif target == "stop":
            return "increase" if trade_direction == "BUY" else "decrease"
        return "increase"
    
    async def verify_deal_exists(self, deal_id: str, trading_headers: dict) -> Tuple[bool, str]:
        """
        Verify deal exists with detailed reason codes
        Returns: (exists: bool, reason: str)
        """
        try:
            url = f"{API_BASE_URL}/positions"
            headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:  # ✅ FIX: Longer timeout
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                positions = data.get("positions", [])
                deal_ids = [pos["position"]["dealId"] for pos in positions]
                
                if deal_id in deal_ids:
                    return True, "FOUND_ON_IG"
                else:
                    self.logger.warning(f"[DEAL NOT FOUND] {deal_id} not in {len(deal_ids)} active deals")
                    self.logger.debug(f"[AVAILABLE DEALS] {deal_ids}")
                    return False, "NOT_FOUND_ON_IG"
                    
        except httpx.TimeoutException:
            self.logger.error(f"[TIMEOUT] Deal verification timeout for {deal_id}")
            return True, "TIMEOUT_ASSUME_VALID"  # ✅ FIX: Don't mark as ghost on timeout
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"[HTTP ERROR] Deal verification failed: {e.response.status_code}")
            return True, "HTTP_ERROR_ASSUME_VALID"  # ✅ FIX: Don't mark as ghost on HTTP errors
            
        except Exception as e:
            self.logger.error(f"[VERIFICATION ERROR] {deal_id}: {e}")
            return True, "ERROR_ASSUME_VALID"  # ✅ FIX: Don't mark as ghost on errors

    def send_adjustment(self, epic: str, direction: str, stop_offset_points: int,
                        limit_offset_points: int = 0) -> bool:
        """Send stop/limit adjustment using direct service call (FIXED: No more HTTP authentication issues)"""

        # Validate inputs
        if stop_offset_points <= 0:
            self.logger.warning(f"[INVALID] {epic} Stop offset points must be positive: {stop_offset_points}")
            return False

        current_offsets = (stop_offset_points, limit_offset_points)
        with self._lock:
            if self._last_sent_offsets.get(epic) == current_offsets:
                self.logger.debug(f"[SKIP] Already sent {epic} stop={stop_offset_points}, limit={limit_offset_points}")
                return True

        try:
            # ✅ FIXED: Use direct service call instead of HTTP request
            from services.adjust_stop_service import adjust_stop_sync

            # Log what we're doing
            adjust_direction_stop = self._get_adjust_direction(direction, "stop")
            adjust_direction_limit = self._get_adjust_direction(direction, "limit")

            self.logger.info(f"[TRAILING DIRECT] {epic} stop_offset={stop_offset_points}, limit_offset={limit_offset_points}")
            self.logger.info(f"[TRAILING DIRECT] {epic} stop_direction={adjust_direction_stop}, limit_direction={adjust_direction_limit}")

            # Call the service directly (no HTTP, no authentication issues)
            result = adjust_stop_sync(
                epic=epic,
                stop_offset_points=stop_offset_points,
                limit_offset_points=limit_offset_points,
                adjust_direction_stop=adjust_direction_stop,
                adjust_direction_limit=adjust_direction_limit,
                dry_run=self.config.dry_run
            )

            status = result.get("status", "unknown")

            if status == "updated":
                self.logger.info(f"[✅ DIRECT SUCCESS] {epic} stop={stop_offset_points}, limit={limit_offset_points}")
                self.logger.info(f"[RESPONSE] {epic} {json.dumps(result.get('sentPayload', {}), indent=2)}")

                # Extract actual levels set by IG from the API response
                if 'sentPayload' in result:
                    payload = result.get('sentPayload', {})
                    if 'stopLevel' in payload:
                        actual_stop = payload.get('stopLevel')
                        actual_limit = payload.get('limitLevel')
                        self.logger.info(f"[IG ACTUAL] {epic} stopLevel={actual_stop}, limitLevel={actual_limit}")

                with self._lock:
                    self._last_sent_offsets[epic] = current_offsets
                return True

            elif status == "closed":
                self.logger.info(f"[❌ CLOSED] {epic} {result.get('message', 'Position closed')}")
                self._mark_trade_closed(epic)
                return False

            elif status == "dry_run":
                self.logger.info(f"[DRY RUN] {epic} {json.dumps(result.get('sentPayload', {}), indent=2)}")
                return True

            elif status == "error":
                self.logger.error(f"[DIRECT ERROR] {epic} {result.get('message', 'Unknown error')}")
                return False

            else:
                self.logger.warning(f"[UNEXPECTED] {epic} Status: {status}, Response: {result}")
                return False

        except Exception as e:
            self.logger.error(f"[EXCEPTION] {epic} Failed to send adjustment: {e}")
            return False

    def get_trade_by_deal_id(self, deal_id: str) -> Optional[TradeLog]:
        """Get trade record by deal ID - ADDED MISSING HELPER METHOD"""
        try:
            with SessionLocal() as db:
                return db.query(TradeLog).filter(TradeLog.deal_id == deal_id).first()
        except Exception as e:
            self.logger.error(f"Error getting trade by deal_id {deal_id}: {e}")
            return None

    def update_position_stop(self, deal_id: str, new_stop_price: float, tp_price: float = None, headers: dict = None) -> bool:
        """
        ✅ CRITICAL FIX: Added missing method that profit protection system calls
        This method was being called by the profit protection system but didn't exist
        
        Args:
            deal_id: The IG deal ID for the position
            new_stop_price: The new stop loss price to set
            tp_price: The take profit price (kept unchanged)
            headers: Trading headers for authentication
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"🛡️ [UPDATE POSITION STOP] Deal {deal_id}: new_stop={new_stop_price:.5f}")
            
            # ✅ FIX: Get trade info for proper API call conversion
            trade = self.get_trade_by_deal_id(deal_id)
            if not trade:
                self.logger.error(f"❌ [UPDATE POSITION STOP] Deal {deal_id}: Trade not found in database")
                return False
            
            # Convert price to points adjustment for API compatibility
            point_value = get_point_value(trade.symbol)
            current_stop = trade.sl_price or 0.0
            
            if trade.direction.upper() == "BUY":
                if new_stop_price <= current_stop:
                    self.logger.warning(f"⚠️ [UPDATE POSITION STOP] BUY trade: new stop {new_stop_price:.5f} not higher than current {current_stop:.5f}")
                    return False
                adjustment_distance = new_stop_price - current_stop
            else:  # SELL
                if new_stop_price >= current_stop:
                    self.logger.warning(f"⚠️ [UPDATE POSITION STOP] SELL trade: new stop {new_stop_price:.5f} not lower than current {current_stop:.5f}")
                    return False
                adjustment_distance = current_stop - new_stop_price
            
            adjustment_points = max(1, int(adjustment_distance / point_value))
            
            # Use existing send_adjustment method for consistency
            success = self.send_adjustment(
                epic=trade.symbol,
                direction=trade.direction,
                stop_offset_points=adjustment_points,
                limit_offset_points=0
            )
            
            if success:
                self.logger.info(f"✅ [PROFIT PROTECTION SUCCESS] Deal {deal_id}: Stop updated to {new_stop_price:.5f}")
                # Update trade record with new stop price
                try:
                    with SessionLocal() as db:
                        db_trade = db.query(TradeLog).filter(TradeLog.deal_id == deal_id).with_for_update().first()
                        if db_trade:
                            db_trade.sl_price = new_stop_price
                            db.commit()
                except Exception as db_error:
                    self.logger.error(f"❌ Failed to update trade record: {db_error}")
            else:
                self.logger.error(f"❌ [PROFIT PROTECTION FAILED] Deal {deal_id}: API call failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE POSITION STOP ERROR] Deal {deal_id}: {e}")
            return False

    def _mark_trade_closed(self, epic: str, deal_id: str = None, reason: str = "position_closed"):
        """
        Mark specific trade(s) as closed - ENHANCED VERSION
        
        Args:
            epic: Trading symbol
            deal_id: Specific deal ID to close (preferred)
            reason: Reason for closure (for logging)
        """
        try:
            with SessionLocal() as db:
                if deal_id:
                    # ✅ PREFERRED: Mark specific trade by deal_id
                    updated_count = (db.query(TradeLog)
                                .filter(TradeLog.deal_id == deal_id,
                                        TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "ema_exit_pending"]))
                                .update({
                                    TradeLog.status: "closed",
                                    TradeLog.trigger_time: datetime.utcnow(),
                                    TradeLog.exit_reason: reason
                                }))
                    
                    if updated_count > 0:
                        self.logger.info(f"[✔ CLOSED] Trade with deal_id {deal_id} marked as closed ({reason})")
                    else:
                        self.logger.warning(f"[⚠️ NOT FOUND] No active trade found with deal_id {deal_id}")
                        
                        # Check if trade exists but in different status
                        existing_trade = (db.query(TradeLog)
                                        .filter(TradeLog.deal_id == deal_id)
                                        .first())
                        
                        if existing_trade:
                            self.logger.info(f"[INFO] Trade {existing_trade.id} with deal_id {deal_id} already has status: {existing_trade.status}")
                        else:
                            self.logger.warning(f"[GHOST] No trade found in database with deal_id {deal_id}")
                
                else:
                    # ⚠️ FALLBACK: Mark all trades for symbol (less precise)
                    self.logger.warning(f"[FALLBACK] No deal_id provided - marking ALL {epic} trades as closed")
                    
                    updated_count = (db.query(TradeLog)
                                .filter(TradeLog.symbol == epic,
                                        TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "ema_exit_pending"]))
                                .update({
                                    TradeLog.status: "closed",
                                    TradeLog.trigger_time: datetime.utcnow(),
                                    TradeLog.exit_reason: f"{reason}_by_symbol"
                                }))
                    
                    self.logger.info(f"[✔ BULK CLOSED] Marked {updated_count} trade(s) as closed for {epic} ({reason})")
                
                db.commit()
                return updated_count
                
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to mark trades as closed: {e}")
            self.logger.error(f"   Epic: {epic}, Deal ID: {deal_id}, Reason: {reason}")
            return 0


class TradeMonitor:
    """Main trade monitoring class with efficient validation and pair-specific optimization"""

    def __init__(self, trade_config=None, api_config: ApiConfig = None):
        # ✅ FIX: Handle case where enhanced processor is not available
        if not ENHANCED_PROCESSOR_AVAILABLE or not CombinedTrailingConfig:
            self.logger = TradeMonitorLogger().get_logger()
            self.logger.error("❌ Enhanced trade processor not available - monitoring disabled")
            self.monitoring_enabled = False
            return
        
        self.monitoring_enabled = True
        self.trade_config = trade_config or SCALPING_CONFIG_WITH_EMA
        self.api_config = api_config or ApiConfig()

        logger_setup = TradeMonitorLogger()
        self.logger = logger_setup.get_logger()
        self.order_sender = OrderSender(self.api_config, self.logger)
        
        # ✅ FIX: Use available processor with proper error handling
        try:
            self.trade_processor = CombinedTradeProcessor(self.trade_config, self.order_sender, self.logger)
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trade processor: {e}")
            self.monitoring_enabled = False
            return
        
        # ✅ NEW: Position cache to minimize IG API calls
        self.position_cache = PositionCache(cache_duration_seconds=30)
        self._trading_headers = None
        self._headers_last_refresh = None
        self._running = False
        
        # ✅ NEW: Validation schedule to batch validation checks
        self.validation_interval_seconds = 120  # Validate every 2 minutes
        self._last_validation_time = None

        # ✅ NEW: Pair-specific configuration system
        if PAIR_CONFIG_AVAILABLE:
            try:
                self.pair_config_manager = get_fastapi_config_manager()
                self.logger.info("✅ Pair-specific configuration manager initialized successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize pair config manager: {e}")
                self.pair_config_manager = None
                PAIR_CONFIG_AVAILABLE = False
        else:
            self.pair_config_manager = None
            self.logger.debug("📊 Using default configuration (pair-specific config not available)")
        
        self.logger.info("🔧 TradeMonitor initialized successfully")

    async def refresh_trading_headers(self) -> bool:
        """
        Refresh trading headers with caching
        Returns True if headers are valid, False otherwise
        """
        try:
            # Check if we have cached headers that are still valid (cache for 30 minutes)
            if (self._trading_headers and self._headers_last_refresh and
                (datetime.now() - self._headers_last_refresh).total_seconds() < 1800):
                return True
            
            # Get fresh headers
            if ENHANCED_PROCESSOR_AVAILABLE:
                self._trading_headers = await get_ig_auth_headers()
            else:
                self.logger.error("❌ Cannot refresh headers - dependencies not available")
                return False
                
            self._headers_last_refresh = datetime.now()
            
            if self._trading_headers:
                self.logger.debug("✅ Trading headers refreshed successfully")
                return True
            else:
                self.logger.error("❌ Failed to get trading headers")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error refreshing trading headers: {e}")
            return False

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol - ADDED MISSING METHOD"""
        try:
            if not await self.refresh_trading_headers():
                return None
                
            url = f"{API_BASE_URL}/markets/{symbol}"
            headers = {
                "X-IG-API-KEY": self._trading_headers["X-IG-API-KEY"],
                "CST": self._trading_headers["CST"],
                "X-SECURITY-TOKEN": self._trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "3"
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Return mid price
                bid = float(data["snapshot"]["bid"])
                offer = float(data["snapshot"]["offer"])
                mid_price = (bid + offer) / 2
                
                self.logger.debug(f"📊 [PRICE] {symbol}: bid={bid:.5f}, offer={offer:.5f}, mid={mid_price:.5f}")
                return mid_price
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get current price for {symbol}: {e}")
            return None

    async def validate_open_positions(self) -> Dict[str, int]:
        """
        Validate all open positions against IG API and mark closed ones - ENHANCED WITH RETRY LOGIC
        Returns: Dictionary with validation statistics
        """
        validation_stats = {
            "checked": 0,
            "valid": 0,
            "closed": 0,
            "errors": 0
        }
        
        try:
            # Get fresh headers
            if not await self.refresh_trading_headers():
                self.logger.error("❌ Cannot validate positions - no trading headers")
                validation_stats["errors"] = 1
                return validation_stats
            
            # Get positions from IG (with caching)
            positions = await self.position_cache.get_positions(self._trading_headers, force_refresh=True)
            if positions is None:
                self.logger.warning("⚠️ Skipping position validation - IG API unavailable")
                validation_stats["errors"] = 1
                return validation_stats
            
            # Extract deal IDs from IG
            ig_deal_ids = set(pos["position"]["dealId"] for pos in positions)
            self.logger.info(f"🔍 Found {len(ig_deal_ids)} positions on IG")
            
            # Check database trades
            with SessionLocal() as db:
                active_trades = (db.query(TradeLog)
                               .filter(TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "ema_exit_pending"]))
                               .all())
                
                validation_stats["checked"] = len(active_trades)
                
                for trade in active_trades:
                    if trade.deal_id and trade.deal_id in ig_deal_ids:
                        validation_stats["valid"] += 1
                        # Reset validation failure counter if it exists
                        if hasattr(trade, 'validation_failures'):
                            trade.validation_failures = 0
                    else:
                        # ✅ FIX: Add grace period before marking as closed
                        if not hasattr(trade, 'validation_failures'):
                            trade.validation_failures = 0
                        
                        trade.validation_failures += 1
                        
                        # Only mark closed after 3 consecutive failures (6 minutes of failures)
                        if trade.validation_failures >= 3:
                            trade.status = "closed"
                            trade.trigger_time = datetime.utcnow()
                            trade.exit_reason = "position_not_found_after_retries"
                            validation_stats["closed"] += 1
                            
                            self.logger.warning(f"🚨 Trade {trade.id} ({trade.symbol}) marked as closed after {trade.validation_failures} validation failures")
                        else:
                            self.logger.info(f"⚠️ Trade {trade.id} ({trade.symbol}) validation failure {trade.validation_failures}/3")
                
                db.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Error during position validation: {e}")
            validation_stats["errors"] += 1
        
        return validation_stats

    async def process_single_trade(self, trade: TradeLog) -> bool:
        """Process a single trade with the advanced trailing system - FIXED METHOD CALLS"""
        try:
            if not self.monitoring_enabled:
                self.logger.error("❌ Monitoring disabled - cannot process trades")
                return False
            
            # ✅ FIX: Get current price (was missing)
            current_price = await self.get_current_price(trade.symbol)
            if not current_price:
                self.logger.error(f"❌ Could not get current price for {trade.symbol}")
                return False
            
            # ✅ NEW: Use pair-specific configuration if available
            if self.pair_config_manager:
                try:
                    pair_config = self.pair_config_manager.get_optimized_config_for_trade(trade)
                    self.logger.debug(f"📊 Using pair-specific config for {trade.symbol}")
                except Exception as config_error:
                    self.logger.warning(f"⚠️ Failed to get pair config for {trade.symbol}, using default: {config_error}")
                    pair_config = self.trade_config
            else:
                pair_config = self.trade_config
            
            # ✅ FIX: Use correct method name and pass all required parameters
            with SessionLocal() as db:
                if ENHANCED_PROCESSOR_AVAILABLE and hasattr(self.trade_processor, 'process_trade_with_combined_validation'):
                    # Use enhanced processor with validation
                    success = await self.trade_processor.process_trade_with_combined_validation(
                        trade, current_price, self._trading_headers, db
                    )
                elif hasattr(self.trade_processor, 'process_trade_enhanced'):
                    # Use enhanced processor without validation
                    success = self.trade_processor.process_trade_enhanced(trade, current_price, db)
                elif hasattr(self.trade_processor, 'process_trade_with_advanced_trailing'):
                    # Use basic trailing processor
                    success = self.trade_processor.process_trade_with_advanced_trailing(trade, current_price, db)
                else:
                    self.logger.error(f"❌ No suitable processing method found for trade {trade.id}")
                    return False
            
            if not success:
                self.logger.warning(f"⚠️ Processing failed for trade {trade.id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error processing trade {trade.id}: {e}")
            import traceback
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return False

    async def monitor_trades_once(self) -> Dict[str, int]:
        """
        Run one cycle of trade monitoring
        Returns: Dictionary with processing statistics
        """
        if not self.monitoring_enabled:
            return {"processed": 0, "successful": 0, "failed": 0, "closed": 0, "error": "monitoring_disabled"}
        
        stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "closed": 0
        }
        
        try:
            # Periodic position validation (every 2 minutes)
            current_time = datetime.now()
            if (not self._last_validation_time or 
                (current_time - self._last_validation_time).total_seconds() >= self.validation_interval_seconds):
                
                self.logger.info("🔍 Running position validation...")
                validation_stats = await self.validate_open_positions()
                self._last_validation_time = current_time
                
                self.logger.info(f"📊 Validation complete: {validation_stats}")
                stats["closed"] = validation_stats["closed"]
            
            # Get trades to process
            with SessionLocal() as db:
                # ✅ FIX: Add database locking to prevent concurrent access issues
                active_trades = (db.query(TradeLog)
                               .filter(TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "ema_exit_pending"]))
                               .order_by(TradeLog.id.desc())
                               .limit(50)  # Process max 50 trades per cycle
                               .all())
                
                stats["processed"] = len(active_trades)
                
                if not active_trades:
                    self.logger.debug("📭 No active trades to process")
                    return stats
                
                self.logger.info(f"🔄 Processing {len(active_trades)} active trades...")
                
                # Process each trade
                for trade in active_trades:
                    try:
                        success = await self.process_single_trade(trade)
                        if success:
                            stats["successful"] += 1
                        else:
                            stats["failed"] += 1
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error processing trade {trade.id}: {e}")
                        stats["failed"] += 1
                        
        except Exception as e:
            self.logger.error(f"❌ Error in monitor_trades_once: {e}")
            stats["errors"] = 1
            
        return stats

    async def run_monitoring_loop(self, cycle_interval: int = 30):
        """
        Main monitoring loop
        Args:
            cycle_interval: Seconds between monitoring cycles
        """
        if not self.monitoring_enabled:
            self.logger.error("❌ Monitoring disabled - cannot start monitoring loop")
            return
        
        self.logger.info(f"🚀 Starting trade monitoring loop (interval: {cycle_interval}s)")
        self._running = True
        
        cycle_count = 0
        
        while self._running:
            try:
                cycle_count += 1
                start_time = time.time()
                
                self.logger.info(f"🔄 === Monitoring Cycle #{cycle_count} ===")
                
                # Run monitoring cycle
                stats = await self.monitor_trades_once()
                
                processing_time = time.time() - start_time
                
                # Log cycle summary
                self.logger.info(f"📊 Cycle #{cycle_count} complete in {processing_time:.2f}s: "
                               f"processed={stats['processed']}, successful={stats['successful']}, "
                               f"failed={stats['failed']}, closed={stats['closed']}")
                
                # Wait for next cycle
                import asyncio
                await asyncio.sleep(cycle_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                import asyncio
                await asyncio.sleep(10)  # Wait 10 seconds before retrying
                
        self._running = False
        self.logger.info("🏁 Trade monitoring loop stopped")

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._running = False
        self.logger.info("🛑 Monitoring stop requested")


# Global monitor instance for status tracking
monitor_instance = None

def start_monitoring_thread(seed_data=False, dry_run=False):
    """Start monitoring in background thread - ENHANCED WITH ERROR HANDLING"""
    global monitor_instance
    
    try:
        if not ENHANCED_PROCESSOR_AVAILABLE:
            print("❌ Cannot start monitoring - enhanced processor not available")
            return None
        
        # Create trade monitor instance
        monitor_instance = TradeMonitor()
        
        if not monitor_instance.monitoring_enabled:
            print("❌ Cannot start monitoring - monitor initialization failed")
            return None
        
        def run_monitor():
            import asyncio
            try:
                asyncio.run(monitor_instance.run_monitoring_loop(cycle_interval=30))
            except Exception as e:
                print(f"❌ Monitor thread crashed: {e}")
                import traceback
                print(f"❌ Full traceback: {traceback.format_exc()}")
        
        thread = threading.Thread(target=run_monitor, daemon=True)
        thread.start()
        return thread
        
    except Exception as e:
        print(f"❌ Failed to start monitoring thread: {e}")
        return None

def get_monitor_status():
    """Get current monitor status"""
    global monitor_instance
    
    if not monitor_instance:
        return {"status": "not_started", "monitoring_enabled": False}
    
    if not monitor_instance.monitoring_enabled:
        return {"status": "disabled", "monitoring_enabled": False, "reason": "initialization_failed"}
    
    return {
        "status": "running" if monitor_instance._running else "stopped",
        "monitoring_enabled": True,
        "enhanced_processor_available": ENHANCED_PROCESSOR_AVAILABLE,
        "pair_config_available": PAIR_CONFIG_AVAILABLE
    }


# Main execution
if __name__ == "__main__":
    import asyncio
    
    # Create trade monitor
    monitor = TradeMonitor()
    
    try:
        # Run the monitoring loop
        asyncio.run(monitor.run_monitoring_loop(cycle_interval=30))
    except KeyboardInterrupt:
        print("🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")