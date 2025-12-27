# dev-app/trade_monitor.py
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

# Import utils functions
from utils import get_point_value, convert_price_to_points  # ✅ ADDED: Import
from database_monitoring_fix import DatabaseMonitoringFix  # ✅ CRITICAL FIX: Enhanced monitoring 
# services/shared_types.py
"""
Shared types and configurations to avoid circular imports
This module contains common classes used by both trade_monitor and pair_specific_config
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import threading
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


@dataclass
class ApiConfig:
    """Configuration for API calls"""
    base_url: str = "http://fastapi-dev:8000"  # TODO: Use FASTAPI_DEV_URL from config
    subscription_key: str = "436abe054a074894a0517e5172f0e5b6"
    dry_run: bool = False  # Set to False for live trading
    
    @property
    def adjust_stop_url(self) -> str:
        """Get the full URL for the adjust-stop endpoint"""
        return f"{self.base_url}/orders/adjust-stop"





# Import enhanced trade status manager
try:
    from services.trade_sync import EnhancedTradeStatusManager
    ENHANCED_STATUS_MANAGER_AVAILABLE = True
    print("✅ Enhanced trade status manager imported successfully")
except ImportError as e:
    print(f"Warning: Could not import enhanced status manager: {e}")
    ENHANCED_STATUS_MANAGER_AVAILABLE = False

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

# ✅ NEW: Import pair-specific configuration system if available
try:
    from services.pair_specific_config import (
        get_fastapi_config_manager,
        get_config_for_trade,
        analyze_trade_pair,
        FastAPIPairConfigManager
    )
    PAIR_CONFIG_AVAILABLE = True
    print("✅ Pair-specific configuration system loaded successfully")
except ImportError as e:
    print(f"⚠️ Pair configuration system import failed: {e}")
    print("   This is usually caused by missing dependencies (enhanced_trade_processor)")
    PAIR_CONFIG_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Pair configuration system initialization error: {e}")
    PAIR_CONFIG_AVAILABLE = False

@dataclass
class ApiConfig:
    """Configuration for API calls"""
    base_url: str = "http://fastapi-dev:8000"  # TODO: Use FASTAPI_DEV_URL from config
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
            
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                return result  # Return full result instead of just True

            elif status == "closed":
                self.logger.info(f"[❌ CLOSED] {epic} {result.get('message', 'Position closed')}")
                self._mark_trade_closed(epic)
                return {"status": "closed", "message": result.get('message')}

            elif status == "dry_run":
                self.logger.info(f"[DRY RUN] {epic} {json.dumps(result.get('sentPayload', {}), indent=2)}")
                return result

            elif status == "error":
                self.logger.error(f"[DIRECT ERROR] {epic} {result.get('message', 'Unknown error')}")
                return result

            else:
                self.logger.warning(f"[UNEXPECTED] {epic} Status: {status}, Response: {result}")
                return {"status": "error", "message": f"Unexpected status: {status}"}

        except Exception as e:
            self.logger.error(f"[EXCEPTION] {epic} Failed to send adjustment: {e}")
            return {"status": "error", "message": str(e)}

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
                                        TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "ema_exit_pending", "profit_protected", "partial_closed"]))
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
                                        TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "ema_exit_pending", "partial_closed"]))
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
        print(f"🔧 TradeMonitor.__init__ starting...")
        print(f"   • ENHANCED_PROCESSOR_AVAILABLE: {ENHANCED_PROCESSOR_AVAILABLE}")
        print(f"   • CombinedTrailingConfig available: {CombinedTrailingConfig is not None}")
        print(f"   • SCALPING_CONFIG_WITH_EMA available: {SCALPING_CONFIG_WITH_EMA is not None}")
        print(f"   • ENHANCED_STATUS_MANAGER_AVAILABLE: {ENHANCED_STATUS_MANAGER_AVAILABLE}")
        
        if not ENHANCED_PROCESSOR_AVAILABLE or not CombinedTrailingConfig:
            print("❌ Enhanced trade processor not available - monitoring disabled")
            print(f"   • ENHANCED_PROCESSOR_AVAILABLE: {ENHANCED_PROCESSOR_AVAILABLE}")
            print(f"   • CombinedTrailingConfig: {CombinedTrailingConfig}")
            self.monitoring_enabled = False
            # Still create logger for error reporting
            logger_setup = TradeMonitorLogger()
            self.logger = logger_setup.get_logger()
            self.logger.error("❌ Enhanced trade processor not available - monitoring disabled")
            return
        
        print("✅ Enhanced processor available, continuing initialization...")
        self.monitoring_enabled = True

        # Check master trailing stop disable flag
        try:
            from config import TRAILING_STOPS_ENABLED
            if not TRAILING_STOPS_ENABLED:
                print("⚠️  TRAILING STOPS DISABLED via config.TRAILING_STOPS_ENABLED = False")
                self.monitoring_enabled = False
                # Still create logger for status reporting
                logger_setup = TradeMonitorLogger()
                self.logger = logger_setup.get_logger()
                self.logger.warning("⚠️  Trailing stops disabled via config flag")
                return
        except ImportError:
            pass  # Config not available, continue with monitoring enabled

        # Use standard trailing config (simplified - removed progressive config)
        if trade_config is None:
            self.trade_config = SCALPING_CONFIG_WITH_EMA
            print("🎯 Using standard SCALPING_CONFIG_WITH_EMA")
        else:
            self.trade_config = trade_config
        self.api_config = api_config or ApiConfig()

        print("🔧 Setting up logger...")
        logger_setup = TradeMonitorLogger()
        self.logger = logger_setup.get_logger()

        print("🚨 Initializing CRITICAL monitoring fix...")
        self.monitoring_fix = DatabaseMonitoringFix(self.logger)
        self.last_integrity_check = datetime.now()
        self.integrity_check_interval = timedelta(minutes=5)
        print("✅ Critical monitoring fix initialized")

        print("🔧 Creating order sender...")
        self.order_sender = OrderSender(self.api_config, self.logger)
        
        # ✅ FIX: Use available processor with proper error handling
        print("🔧 Creating trade processor...")
        try:
            self.trade_processor = CombinedTradeProcessor(self.trade_config, self.order_sender, self.logger)
            print("✅ Trade processor created successfully")
        except Exception as e:
            print(f"❌ Failed to initialize trade processor: {e}")
            import traceback
            print(f"❌ Processor traceback: {traceback.format_exc()}")
            self.logger.error(f"❌ Failed to initialize trade processor: {e}")
            self.monitoring_enabled = False
            return
        
        print("🔧 Setting up position cache...")
        # ✅ NEW: Position cache to minimize IG API calls
        self.position_cache = PositionCache(cache_duration_seconds=30)
        self._trading_headers = None
        self._headers_last_refresh = None
        self._running = False
        
        # ✅ NEW: Enhanced status manager for intelligent trade verification
        self.enhanced_status_manager = None
        
        # ✅ NEW: Validation schedule to batch validation checks
        self.validation_interval_seconds = 120  # Validate every 2 minutes
        self._last_validation_time = None

        print("🔧 Setting up pair-specific configuration...")
        # ✅ NEW: Pair-specific configuration system
        if PAIR_CONFIG_AVAILABLE:
            try:
                self.pair_config_manager = get_fastapi_config_manager()
                self.logger.info("✅ Pair-specific configuration manager initialized successfully")
                print("✅ Pair-specific configuration manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize pair config manager: {e}")
                print(f"⚠️ Failed to initialize pair config manager: {e}")
                self.pair_config_manager = None
                # Don't disable monitoring for this
        else:
            self.pair_config_manager = None
            self.logger.debug("📊 Using default configuration (pair-specific config not available)")
            print("📊 Using default configuration (pair-specific config not available)")
        
        print("✅ TradeMonitor initialization completed successfully")
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
                
                # ✅ NEW: Initialize enhanced status manager with fresh headers
                if ENHANCED_STATUS_MANAGER_AVAILABLE and self._trading_headers:
                    self.enhanced_status_manager = EnhancedTradeStatusManager(self._trading_headers)
                    self.logger.debug("✅ Enhanced status manager initialized with fresh headers")
                
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

    async def validate_open_positions_enhanced(self) -> Dict[str, int]:
        """
        ✅ ENHANCED: Validate positions using the new intelligent status management system
        Returns: Dictionary with comprehensive validation statistics
        """
        validation_stats = {
            "checked": 0,
            "still_active": 0,
            "closed": 0,
            "rejected": 0,
            "expired": 0,
            "invalid_deal": 0,
            "missing": 0,
            "verification_errors": 0,
            "api_errors": 0
        }
        
        try:
            # Get fresh headers and initialize enhanced status manager
            if not await self.refresh_trading_headers():
                self.logger.error("❌ Cannot validate positions - no trading headers")
                validation_stats["api_errors"] = 1
                return validation_stats
            
            # ✅ NEW: Use enhanced status manager if available
            if not self.enhanced_status_manager and ENHANCED_STATUS_MANAGER_AVAILABLE:
                self.enhanced_status_manager = EnhancedTradeStatusManager(self._trading_headers)
            
            # Get active trades that need validation - ENHANCED WITH CRITICAL FIX
            with SessionLocal() as db:
                # 🚨 CRITICAL FIX: Use enhanced monitoring to prevent missed trades (like Trade 1161)
                active_trades = self.monitoring_fix.get_active_trades_enhanced(db)
                
                validation_stats["checked"] = len(active_trades)
                
                if not active_trades:
                    self.logger.info("📭 No active trades to validate")
                    return validation_stats
                
                self.logger.info(f"🔍 Validating {len(active_trades)} active trades...")
                
                # ✅ ENHANCED: Use intelligent verification for each trade
                if self.enhanced_status_manager and ENHANCED_STATUS_MANAGER_AVAILABLE:
                    # Use the new enhanced verification system
                    for trade in active_trades:
                        try:
                            final_status = await self.enhanced_status_manager.verify_and_update_trade_status(trade, db)
                            
                            # Count by final status
                            if final_status in ["tracking", "break_even", "trailing", "ema_exit_pending", "profit_protected", "partial_closed"]:
                                validation_stats["still_active"] += 1
                            elif final_status == "closed":
                                validation_stats["closed"] += 1
                            elif final_status in ["rejected", "invalid_deal"]:
                                validation_stats["rejected"] += 1
                            elif final_status == "expired":
                                validation_stats["expired"] += 1
                            elif final_status == "missing_on_ig":
                                validation_stats["missing"] += 1
                            else:
                                validation_stats["verification_errors"] += 1
                                
                        except Exception as e:
                            self.logger.error(f"❌ Error validating trade {trade.id}: {e}")
                            validation_stats["verification_errors"] += 1
                    
                    # Commit all validation changes
                    db.commit()
                    
                else:
                    # ✅ FALLBACK: Use basic position validation with retry logic
                    self.logger.warning("⚠️ Enhanced status manager not available, using basic validation")
                    
                    # Get positions from IG (with caching)
                    positions = await self.position_cache.get_positions(self._trading_headers, force_refresh=True)
                    if positions is None:
                        self.logger.warning("⚠️ Skipping position validation - IG API unavailable")
                        validation_stats["api_errors"] = 1
                        return validation_stats
                    
                    # Extract deal IDs from IG
                    ig_deal_ids = set(pos["position"]["dealId"] for pos in positions)
                    self.logger.info(f"🔍 Found {len(ig_deal_ids)} positions on IG")
                    
                    for trade in active_trades:
                        if trade.deal_id and trade.deal_id in ig_deal_ids:
                            validation_stats["still_active"] += 1
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
                                validation_stats["still_active"] += 1  # Still counting as active until 3 failures
                    
                    db.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Error during enhanced position validation: {e}")
            validation_stats["api_errors"] += 1
        
        return validation_stats

    async def validate_open_positions(self) -> Dict[str, int]:
        """
        Validate all open positions against IG API and mark closed ones - LEGACY WRAPPER
        Returns: Dictionary with validation statistics (maintains backward compatibility)
        """
        enhanced_stats = await self.validate_open_positions_enhanced()
        
        # Convert enhanced stats to legacy format for backward compatibility
        return {
            "checked": enhanced_stats["checked"],
            "valid": enhanced_stats["still_active"],
            "closed": enhanced_stats["closed"] + enhanced_stats["rejected"] + enhanced_stats["expired"] + enhanced_stats["invalid_deal"],
            "errors": enhanced_stats["verification_errors"] + enhanced_stats["api_errors"]
        }

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
            
            # ✅ NEW: Log pair-specific configuration info
            # Note: EnhancedTradeProcessor handles pair-specific configs internally via get_trailing_config_for_epic()
            from config import get_trailing_config_for_epic

            try:
                pair_config = get_trailing_config_for_epic(trade.symbol)
                self.logger.debug(f"📊 [MONITOR] Using pair-specific config for {trade.symbol}: "
                                f"Stage2({pair_config['stage2_trigger_points']}pts→{pair_config['stage2_lock_points']}pts) "
                                f"Stage3({pair_config['stage3_trigger_points']}pts)")
            except Exception as config_error:
                self.logger.warning(f"⚠️ Failed to get pair config for {trade.symbol}: {config_error}")
            
            # ✅ FIX: Use correct method name and pass all required parameters
            with SessionLocal() as db:
                if ENHANCED_PROCESSOR_AVAILABLE and hasattr(self.trade_processor, 'process_trade_with_combined_validation'):
                    # Use enhanced processor with validation
                    success = await self.trade_processor.process_trade_with_combined_validation(
                        trade, current_price, self._trading_headers, db
                    )
                elif hasattr(self.trade_processor, 'process_trade_enhanced'):
                    # Use enhanced processor without validation
                    success = await self.trade_processor.process_trade_enhanced(trade, current_price, db)
                elif hasattr(self.trade_processor, 'process_trade_with_advanced_trailing'):
                    # Use basic trailing processor
                    success = await self.trade_processor.process_trade_with_advanced_trailing(trade, current_price, db)
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
        Run one cycle of trade monitoring - ENHANCED WITH NEW STATUS SYSTEM
        Returns: Dictionary with processing statistics
        """
        if not self.monitoring_enabled:
            return {"processed": 0, "successful": 0, "failed": 0, "closed": 0, "error": "monitoring_disabled"}
        
        stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "closed": 0,
            "rejected": 0,
            "expired": 0,
            "invalid_deal": 0,
            "missing": 0,
            "verification_errors": 0
        }
        
        try:
            # ✅ ENHANCED: Periodic position validation with intelligent status management
            # TEMPORARILY DISABLED to allow progressive trailing to work
            current_time = datetime.now()
            validation_run = False
            if (not self._last_validation_time or
                (current_time - self._last_validation_time).total_seconds() >= (self.validation_interval_seconds * 5)):  # Run validation less frequently

                self.logger.info("🔍 Running enhanced position validation...")
                validation_stats = await self.validate_open_positions_enhanced()
                self._last_validation_time = current_time
                validation_run = True
                
                # ✅ NEW: Log detailed validation results
                self.logger.info(f"📊 Enhanced validation complete:")
                self.logger.info(f"   • Checked: {validation_stats['checked']}")
                self.logger.info(f"   • Still Active: {validation_stats['still_active']}")
                self.logger.info(f"   • Closed: {validation_stats['closed']}")
                self.logger.info(f"   • Rejected: {validation_stats['rejected']}")
                self.logger.info(f"   • Expired: {validation_stats['expired']}")
                self.logger.info(f"   • Invalid Deal: {validation_stats['invalid_deal']}")
                self.logger.info(f"   • Missing: {validation_stats['missing']}")
                self.logger.info(f"   • Errors: {validation_stats['verification_errors']}")

                # Update stats with new categories
                stats.update({
                    "closed": validation_stats["closed"],
                    "rejected": validation_stats["rejected"],
                    "expired": validation_stats["expired"],
                    "invalid_deal": validation_stats["invalid_deal"],
                    "missing": validation_stats["missing"],
                    "verification_errors": validation_stats["verification_errors"]
                })
            else:
                # Skip validation this cycle, just set defaults
                validation_stats = {
                    "checked": 0,
                    "still_active": 0,
                    "closed": 0,
                    "rejected": 0,
                    "expired": 0,
                    "invalid_deal": 0,
                    "missing": 0,
                    "verification_errors": 0
                }
                self.logger.debug("🔄 Skipping validation this cycle - focusing on progressive trailing")
            
            # Get trades to process - ENHANCED WITH CRITICAL FIX
            with SessionLocal() as db:
                # 🚨 CRITICAL FIX: Use enhanced monitoring to prevent missed trades (like Trade 1161)
                all_active_trades = self.monitoring_fix.get_active_trades_enhanced(db)
                # Limit to 50 trades per cycle for performance
                active_trades = all_active_trades[:50]

                # Periodic integrity check to prevent monitoring gaps
                if datetime.now() - self.last_integrity_check > self.integrity_check_interval:
                    report = self.monitoring_fix.validate_monitoring_integrity(db)
                    if report["status"] != "healthy":
                        self.logger.error(f"🚨 MONITORING INTEGRITY ISSUE: {report}")
                    self.last_integrity_check = datetime.now()
                
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
        Main monitoring loop - ENHANCED WITH NEW STATUS REPORTING
        Args:
            cycle_interval: Seconds between monitoring cycles
        """
        if not self.monitoring_enabled:
            self.logger.error("❌ Monitoring disabled - cannot start monitoring loop")
            return
        
        self.logger.info(f"🚀 Starting enhanced trade monitoring loop (interval: {cycle_interval}s)")
        self.logger.info(f"   • Enhanced Status Manager: {'Available' if ENHANCED_STATUS_MANAGER_AVAILABLE else 'Not Available'}")
        self.logger.info(f"   • Pair Config Manager: {'Available' if PAIR_CONFIG_AVAILABLE else 'Using Defaults'}")
        
        self._running = True
        
        cycle_count = 0
        
        while self._running:
            try:
                cycle_count += 1
                start_time = time.time()
                
                self.logger.info(f"🔄 === Enhanced Monitoring Cycle #{cycle_count} ===")
                
                # Run monitoring cycle
                stats = await self.monitor_trades_once()
                
                processing_time = time.time() - start_time
                
                # ✅ ENHANCED: Log cycle summary with new status categories
                self.logger.info(f"📊 Cycle #{cycle_count} complete in {processing_time:.2f}s:")
                self.logger.info(f"   • Processed: {stats['processed']}, Successful: {stats['successful']}, Failed: {stats['failed']}")
                
                # Log validation results if available
                if any(key in stats for key in ["closed", "rejected", "expired", "invalid_deal", "missing"]):
                    self.logger.info(f"   • Validation - Closed: {stats.get('closed', 0)}, Rejected: {stats.get('rejected', 0)}")
                    self.logger.info(f"   • Validation - Expired: {stats.get('expired', 0)}, Invalid: {stats.get('invalid_deal', 0)}, Missing: {stats.get('missing', 0)}")
                    self.logger.info(f"   • Validation - Errors: {stats.get('verification_errors', 0)}")
                
                # Wait for next cycle
                import asyncio
                await asyncio.sleep(cycle_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Enhanced monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in enhanced monitoring loop: {e}")
                import asyncio
                await asyncio.sleep(10)  # Wait 10 seconds before retrying
                
        self._running = False
        self.logger.info("🏁 Enhanced trade monitoring loop stopped")

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._running = False
        self.logger.info("🛑 Enhanced monitoring stop requested")


# Global monitor instance for status tracking
monitor_instance = None

def start_monitoring_thread(seed_data=False, dry_run=False):
    """Start monitoring in background thread - ENHANCED WITH ERROR HANDLING"""
    global monitor_instance
    
    try:
        if not ENHANCED_PROCESSOR_AVAILABLE:
            print("❌ Cannot start monitoring - enhanced processor not available")
            print("   • Check that enhanced_trade_processor.py exists")
            print("   • Check that dependencies.py has get_ig_auth_headers()")
            return None
        
        print("🔧 Creating enhanced trade monitor instance...")
        
        # Create trade monitor instance with detailed error reporting
        try:
            monitor_instance = TradeMonitor()
        except Exception as creation_error:
            print(f"❌ Failed to create TradeMonitor instance: {creation_error}")
            import traceback
            print(f"❌ Creation traceback: {traceback.format_exc()}")
            return None
        
        if not monitor_instance.monitoring_enabled:
            print("❌ Cannot start monitoring - monitor initialization failed")
            print("   • Check TradeMonitor.__init__ logs for specific error")
            print("   • Verify database connection")
            print("   • Verify enhanced processor availability")
            return None
        
        print("✅ Enhanced TradeMonitor instance created successfully")
        print(f"   • Enhanced processor: {'available' if ENHANCED_PROCESSOR_AVAILABLE else 'not available'}")
        print(f"   • Enhanced status manager: {'available' if ENHANCED_STATUS_MANAGER_AVAILABLE else 'not available'}")
        print(f"   • Pair config: {'available' if PAIR_CONFIG_AVAILABLE else 'using defaults'}")
        
        def run_monitor():
            import asyncio
            try:
                print("🚀 Starting enhanced monitor async loop...")
                asyncio.run(monitor_instance.run_monitoring_loop(cycle_interval=30))
            except Exception as e:
                print(f"❌ Enhanced monitor thread crashed: {e}")
                import traceback
                print(f"❌ Monitor traceback: {traceback.format_exc()}")
        
        print("🧵 Starting enhanced monitor thread...")
        thread = threading.Thread(target=run_monitor, daemon=True)
        thread.start()
        print("✅ Enhanced monitor thread started successfully")
        return thread
        
    except Exception as e:
        print(f"❌ Failed to start enhanced monitoring thread: {e}")
        import traceback
        print(f"❌ Startup traceback: {traceback.format_exc()}")
        return None

def get_monitor_status():
    """Get current monitor status - ENHANCED WITH NEW STATUS INFO"""
    global monitor_instance
    
    if not monitor_instance:
        return {"status": "not_started", "monitoring_enabled": False}
    
    if not monitor_instance.monitoring_enabled:
        return {"status": "disabled", "monitoring_enabled": False, "reason": "initialization_failed"}
    
    return {
        "status": "running" if monitor_instance._running else "stopped",
        "monitoring_enabled": True,
        "enhanced_processor_available": ENHANCED_PROCESSOR_AVAILABLE,
        "enhanced_status_manager_available": ENHANCED_STATUS_MANAGER_AVAILABLE,
        "pair_config_available": PAIR_CONFIG_AVAILABLE,
        "features": {
            "intelligent_verification": ENHANCED_STATUS_MANAGER_AVAILABLE,
            "pair_specific_config": PAIR_CONFIG_AVAILABLE,
            "enhanced_trailing": ENHANCED_PROCESSOR_AVAILABLE
        }
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
        print("🛑 Enhanced monitoring stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")