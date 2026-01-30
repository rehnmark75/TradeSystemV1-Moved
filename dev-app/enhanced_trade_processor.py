# dev-app/enhanced_trade_processor.py
"""
Combined trade processor with both trailing and EMA exit systems
UPDATED FOR NEW TRAILING SYSTEM - CRITICAL UPDATES APPLIED
ENHANCED WITH DEAL ID VALIDATION AND SYNC - GHOST TRADE PREVENTION
ENHANCED WITH PROFIT PROTECTION RULE - 15PT → 10PT RULE INTEGRATION
🛡️ FIXED: Profit protection send_adjustment() headers error
🔧 FIXED: Method resolution error for trailing after profit protection
✅ ENHANCED: Integrated with intelligent status management system from trade_sync
"""

from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session
from services.models import TradeLog, AlertHistory
from typing import Optional, Dict
import httpx
from config import API_BASE_URL

# Import existing systems - UPDATED IMPORTS
from trailing_class import EnhancedTradeProcessor, TrailingConfig, TrailingMethod
from ema_exit_system import EMATrendExit
from utils import get_point_value

# ✅ NEW: Import enhanced status management system
try:
    from services.trade_sync import EnhancedTradeStatusManager
    ENHANCED_STATUS_MANAGER_AVAILABLE = True
    print("✅ Enhanced trade status manager imported successfully in trade processor")
except ImportError as e:
    print(f"Warning: Could not import enhanced status manager in trade processor: {e}")
    ENHANCED_STATUS_MANAGER_AVAILABLE = False


@dataclass
class CombinedTrailingConfig(TrailingConfig):
    """Extended config with EMA exit and limit trailing settings + PROFIT PROTECTION RULE"""
    # EMA Exit Settings
    enable_ema_exit: bool = False
    ema_period: int = 21
    ema_confirmation_candles: int = 2
    ema_timeframe: int = 60
    
    # Limit trailing settings - ENHANCED
    enable_limit_trailing: bool = True
    limit_trail_ratio: float = 1.0
    min_limit_distance: int = 3
    
    # ✅ NEW: Universal Profit Protection Rule
    enable_profit_protection_rule: bool = False  # Default disabled - enable per pair in config
    profit_protection_trigger: int = 15  # When profit reaches 15 points
    profit_protection_stop: int = 10     # Move stop to 10 points profit


class CombinedTradeProcessor(EnhancedTradeProcessor):
    """Trade processor with both trailing and EMA exit systems + validation + PROFIT PROTECTION + ENHANCED STATUS MANAGEMENT"""

    def __init__(self, config: CombinedTrailingConfig, order_sender, logger, trading_headers=None):
        # ✅ CRITICAL: Initialize with new trailing config structure
        super().__init__(config, order_sender, logger)
        self.ema_exit = EMATrendExit(logger, order_sender) if config.enable_ema_exit else None
        self.config = config
        self.default_config = config  # Store default config
        self.trading_headers = trading_headers  # Store headers for validation methods

        # ✅ NEW: Initialize enhanced status manager
        self.enhanced_status_manager = None
        if ENHANCED_STATUS_MANAGER_AVAILABLE and trading_headers:
            self.enhanced_status_manager = EnhancedTradeStatusManager(trading_headers)
            self.logger.info("✅ Enhanced status manager initialized in trade processor")

        # Log profit protection rule
        if getattr(config, 'enable_profit_protection_rule', False):
            self.logger.info(f"🛡️ [PROFIT PROTECTION] Enabled: {config.profit_protection_trigger}pt → {config.profit_protection_stop}pt stop")

        if config.enable_ema_exit:
            self.logger.info(f"[EMA EXIT] Enabled with {config.ema_confirmation_candles} candle confirmation on {config.ema_timeframe}min timeframe")
        else:
            self.logger.info("[EMA EXIT] Disabled")

    def get_config_for_trade(self, trade: TradeLog, db: Session = None) -> TrailingConfig:
        """
        Dynamically select trailing config based on trade's scalp flag and ATR.

        v3.2.0 (Jan 2026): ATR-adaptive trailing stops for scalp trades.
        When ATR data is available via alert_history, stage thresholds are scaled
        proportionally to volatility at signal time, with min/max bounds.
        Falls back to static config when ATR data is unavailable.

        Args:
            trade: Trade to get config for
            db: Database session for ATR lookup (optional, falls back to static)

        Returns:
            TrailingConfig with appropriate settings for this trade
        """
        try:
            from config import get_trailing_config_for_epic, compute_atr_trailing_config

            # Check if this is a scalp trade
            is_scalp = getattr(trade, 'is_scalp_trade', False)

            if is_scalp:
                self.logger.debug(f"⚡ [SCALP CONFIG] Trade {trade.id}: Loading scalp-specific trailing config")

            # Get pair-specific static config with scalp flag
            pair_config = get_trailing_config_for_epic(trade.symbol, is_scalp_trade=is_scalp)

            # v3.2.0: ATR-adaptive trailing for scalp trades
            atr_pips = None
            if is_scalp and db is not None and getattr(trade, 'alert_id', None):
                try:
                    alert = db.query(AlertHistory).filter(
                        AlertHistory.id == trade.alert_id
                    ).first()

                    if alert and alert.atr is not None:
                        point_value = get_point_value(trade.symbol)
                        atr_pips = float(alert.atr) / point_value

                        # Compute ATR-proportional config with bounds
                        static_snapshot = dict(pair_config)
                        pair_config = compute_atr_trailing_config(atr_pips, pair_config)

                        self.logger.info(
                            f"📐 [ATR TRAILING] Trade {trade.id} {trade.symbol}: "
                            f"ATR={atr_pips:.1f} pips → "
                            f"earlyBE={pair_config['early_breakeven_trigger_points']}pts "
                            f"(was {static_snapshot.get('early_breakeven_trigger_points')}), "
                            f"S1={pair_config['stage1_trigger_points']}→lock {pair_config['stage1_lock_points']}, "
                            f"S2={pair_config['stage2_trigger_points']}→lock {pair_config['stage2_lock_points']}, "
                            f"S3={pair_config['stage3_trigger_points']}pts"
                        )
                    else:
                        self.logger.debug(
                            f"📐 [ATR TRAILING] Trade {trade.id}: No ATR in alert {trade.alert_id}, using static config"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ [ATR TRAILING] Trade {trade.id}: ATR lookup failed ({e}), using static config"
                    )

            # Create a new TrailingConfig instance with the (possibly ATR-adjusted) values
            config = TrailingConfig(
                epic=trade.symbol,
                stage1_trigger_points=pair_config.get('stage1_trigger_points', self.default_config.stage1_trigger_points),
                stage1_lock_points=pair_config.get('stage1_lock_points', self.default_config.stage1_lock_points),
                stage2_trigger_points=pair_config.get('stage2_trigger_points', self.default_config.stage2_trigger_points),
                stage2_lock_points=pair_config.get('stage2_lock_points', self.default_config.stage2_lock_points),
                stage3_trigger_points=pair_config.get('stage3_trigger_points', self.default_config.stage3_trigger_points),
                stage3_atr_multiplier=pair_config.get('stage3_atr_multiplier', self.default_config.stage3_atr_multiplier),
                stage3_min_distance=pair_config.get('stage3_min_distance', self.default_config.stage3_min_distance),
                min_trail_distance=pair_config.get('min_trail_distance', self.default_config.min_trail_distance),
                break_even_trigger_points=pair_config.get('break_even_trigger_points', self.default_config.break_even_trigger_points),
                initial_trigger_points=pair_config.get('early_breakeven_trigger_points',
                                                      self.default_config.initial_trigger_points),
                partial_close_trigger_points=pair_config.get('partial_close_trigger_points',
                                                             self.default_config.partial_close_trigger_points),
            )

            return config

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load config for trade {trade.id}: {e}, using default")
            return self.default_config
    
    def set_trading_headers(self, trading_headers: dict):
        """Update trading headers for API calls"""
        self.trading_headers = trading_headers
        
        # ✅ NEW: Update enhanced status manager with fresh headers
        if ENHANCED_STATUS_MANAGER_AVAILABLE and trading_headers:
            self.enhanced_status_manager = EnhancedTradeStatusManager(trading_headers)
            self.logger.debug("✅ Enhanced status manager updated with fresh headers")
    
    def calculate_current_profit_points(self, trade: TradeLog, current_price: float) -> float:
        """Calculate current profit in points - ENHANCED"""
        try:
            point_value = get_point_value(trade.symbol)
            direction = trade.direction.upper()
            entry_price = trade.entry_price
            
            if direction == "BUY":
                profit_in_price = current_price - entry_price
            else:  # SELL
                profit_in_price = entry_price - current_price
            
            profit_points = profit_in_price / point_value
            return profit_points
            
        except Exception as e:
            self.logger.error(f"[PROFIT CALC ERROR] Trade {trade.id}: {e}")
            return 0.0
    
    def should_apply_profit_protection_rule(self, trade: TradeLog, current_price: float) -> bool:
        """Check if the 15-point profit protection rule should be applied"""
        try:
            # Rule must be enabled in config
            if not getattr(self.config, 'enable_profit_protection_rule', False):
                return False
            
            # Check if rule already applied for this trade (check status or custom field)
            if getattr(trade, 'status', '') == 'profit_protected':
                self.logger.debug(f"🛡️ [PROFIT PROTECTION SKIP] Trade {trade.id}: Already profit protected")
                return False
            
            # ✅ ENHANCED: Check if stop is already at protection level (manual or automatic)
            protection_stop_points = getattr(self.config, 'profit_protection_stop', 10)
            point_value = get_point_value(trade.symbol)
            direction = trade.direction.upper()
            entry_price = trade.entry_price
            
            # Calculate what the protection stop price should be
            if direction == "BUY":
                expected_protection_stop = entry_price + (protection_stop_points * point_value)
            else:  # SELL
                expected_protection_stop = entry_price - (protection_stop_points * point_value)
            
            current_stop = trade.sl_price or 0.0
            
            # ✅ ENHANCED: Check if stop is already at or better than protection level
            tolerance = 0.5 * point_value
            
            if direction == "BUY":
                # For BUY: current stop at or above protection level means already protected
                stop_is_protected = current_stop >= (expected_protection_stop - tolerance)
            else:  # SELL
                # For SELL: current stop at or below protection level means already protected
                stop_is_protected = current_stop <= (expected_protection_stop + tolerance)
            
            if stop_is_protected:
                # ✅ AUTO-FIX: Update status to prevent future checks
                self.logger.info(f"🛡️ [AUTO-DETECT] Trade {trade.id}: Stop already protected manually "
                               f"(current: {current_stop:.5f}, protection level: {expected_protection_stop:.5f})")
                trade.status = "profit_protected"
                # Commit the status change to database immediately
                try:
                    from services.db import SessionLocal
                    with SessionLocal() as temp_db:
                        temp_db.merge(trade)
                        temp_db.commit()
                except Exception as db_err:
                    self.logger.warning(f"Could not update trade status: {db_err}")
                return False
            
            # Calculate current profit
            profit_points = self.calculate_current_profit_points(trade, current_price)
            trigger_points = getattr(self.config, 'profit_protection_trigger', 15)
            
            # Apply rule if profit >= 15 points (or configured threshold)
            should_apply = profit_points >= trigger_points
            
            if should_apply:
                self.logger.info(f"🛡️ [PROFIT PROTECTION TRIGGER] Trade {trade.id} {trade.symbol}: "
                               f"Profit {profit_points:.1f}pts >= {trigger_points}pts threshold "
                               f"(current stop: {current_stop:.5f}, target: {expected_protection_stop:.5f})")
            
            return should_apply
            
        except Exception as e:
            self.logger.error(f"[PROFIT PROTECTION CHECK ERROR] Trade {trade.id}: {e}")
            return False
    
    def apply_profit_protection_rule(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """Apply the 15-point → 10-point profit protection rule - 🛡️ FIXED METHOD"""
        try:
            protection_stop_points = getattr(self.config, 'profit_protection_stop', 10)
            point_value = get_point_value(trade.symbol)
            direction = trade.direction.upper()
            entry_price = trade.entry_price
            
            # Calculate the new stop loss price (10 points profit)
            if direction == "BUY":
                new_stop_price = entry_price + (protection_stop_points * point_value)
            else:  # SELL
                new_stop_price = entry_price - (protection_stop_points * point_value)
            
            # ✅ ENHANCED: Check if adjustment is meaningful before making API call
            current_stop = trade.sl_price or 0.0
            tolerance = 0.5 * point_value
            
            if abs(new_stop_price - current_stop) <= tolerance:
                # Stop is already at protection level - just update status
                self.logger.info(f"🛡️ [PROFIT PROTECTION SKIP] Trade {trade.id}: Stop already at protection level "
                               f"({current_stop:.5f}), updating status only")
                trade.status = "profit_protected"
                db.commit()
                return True
            
            # ✅ FIXED: Use update_position_stop instead of send_adjustment with headers
            # The send_adjustment method doesn't accept headers parameter
            success = self.order_sender.update_position_stop(
                deal_id=trade.deal_id,
                new_stop_price=new_stop_price,
                tp_price=trade.tp_price,  # Keep current TP
                headers=self.trading_headers
            )
            
            if success:
                # Update trade in database
                trade.sl_price = new_stop_price
                trade.status = "profit_protected"  # New status to track this
                
                db.commit()
                
                current_profit = self.calculate_current_profit_points(trade, current_price)
                
                self.logger.info(f"✅ [PROFIT PROTECTION APPLIED] Trade {trade.id} {trade.symbol}: "
                               f"Stop moved to +{protection_stop_points}pts profit "
                               f"(new stop: {new_stop_price:.5f}, current profit: {current_profit:.1f}pts)")
                
                return True
            else:
                # ✅ IMPROVED: More informative logging for API failures
                self.logger.warning(f"⚠️ [PROFIT PROTECTION SKIP] Trade {trade.id}: Could not update stop via API "
                                  f"(stop may already be optimal or position closed)")
                # Don't treat this as a hard failure - trade can continue processing
                return True
                
        except Exception as e:
            self.logger.error(f"❌ [PROFIT PROTECTION ERROR] Trade {trade.id}: {e}")
            return False
    
    async def process_trade_enhanced(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """
        MAIN PROCESSING METHOD with PROFIT PROTECTION RULE
        Order of operations:
        0. Load correct config based on is_scalp_trade flag (Jan 2026)
        1. Check for 15-point profit protection rule (HIGHEST PRIORITY)
        2. Check EMA exit conditions
        3. Continue with normal trailing logic
        """
        try:
            # ✅ STEP 0: DYNAMIC CONFIG SELECTION (Jan 2026 - Scalp mode support)
            # v3.2.0: Pass db for ATR-adaptive trailing config lookup
            trade_config = self.get_config_for_trade(trade, db=db)
            # Update the trailing manager's config for this trade
            self.trailing_manager.config = trade_config
            self.config = trade_config  # Also update processor's config reference

            # ✅ STEP 1: PROFIT PROTECTION RULE - HIGHEST PRIORITY
            if self.should_apply_profit_protection_rule(trade, current_price):
                self.logger.info(f"🛡️ [PROFIT PROTECTION] Applying rule for trade {trade.id} {trade.symbol}")
                success = self.apply_profit_protection_rule(trade, current_price, db)
                
                if success:
                    # After applying protection, continue with normal processing
                    # This allows trailing to continue from the new protected level
                    self.logger.info(f"🔄 [CONTINUING] Trade {trade.id} will continue with normal trailing from protected level")
                    # ✅ CRITICAL FIX: Set moved_to_breakeven to True to allow trailing to continue
                    trade.moved_to_breakeven = True
                else:
                    return False
            
            # ✅ STEP 2: EMA Exit Check (if enabled and not already protected)
            if self.ema_exit and trade.status in ["tracking", "break_even", "trailing", "profit_protected"]:
                should_exit, exit_reason = self.ema_exit.should_exit_trade(trade, db)
                
                if should_exit:
                    self.logger.warning(f"[EMA EXIT TRIGGERED] Trade {trade.id} {trade.symbol}: {exit_reason}")
                    success = self.ema_exit.execute_ema_exit(trade, exit_reason, db)
                    return success  # Exit early, don't continue with trailing logic
            
            # ✅ STEP 3: Continue with enhanced trailing logic
            return await self.process_trade_with_advanced_trailing(trade, current_price, db)
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED PROCESSING ERROR] Trade {trade.id}: {e}")
            return False
    
    async def verify_deal_exists(self, deal_id: str, trading_headers: dict) -> bool:
        """
        ✅ ENHANCED: Verify that the deal ID actually exists on IG using intelligent verification
        Now uses EnhancedTradeStatusManager if available, otherwise falls back to basic check
        """
        try:
            # ✅ NEW: Use enhanced status manager if available
            if self.enhanced_status_manager and ENHANCED_STATUS_MANAGER_AVAILABLE:
                # Use the comprehensive verification system
                return await self.enhanced_status_manager._check_position_exists(deal_id)
            
            # ✅ FALLBACK: Use basic verification (original method)
            # Get all open positions
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

                # Check if deal ID exists in open positions
                positions = data.get("positions", [])
                deal_exists = any(
                    pos["position"]["dealId"] == deal_id 
                    for pos in positions
                )
                
                if not deal_exists:
                    self.logger.error(f"[GHOST TRADE] Deal ID {deal_id} not found in open positions")
                    
                    # Log available deals for debugging
                    available_deals = [pos["position"]["dealId"] for pos in positions]
                    self.logger.warning(f"[AVAILABLE DEALS] {available_deals}")
                
                return deal_exists
                
        except Exception as e:
            self.logger.error(f"[DEAL VERIFICATION ERROR] {deal_id}: {e}")
            return False  # Assume deal doesn't exist if we can't verify
    
    async def get_real_position_data(self, deal_id: str, trading_headers: dict) -> Optional[Dict]:
        """Get actual position data from IG for validation/sync"""
        try:
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

                # Find the specific deal
                positions = data.get("positions", [])
                for pos in positions:
                    if pos["position"]["dealId"] == deal_id:
                        return {
                            "deal_id": deal_id,
                            "entry_price": pos["position"]["level"],
                            "current_stop": pos["position"].get("stopLevel"),
                            "current_limit": pos["position"].get("limitLevel"),
                            "direction": pos["position"]["direction"],
                            "size": pos["position"]["size"],
                            "epic": pos["market"]["epic"],
                            "current_bid": pos["market"]["bid"],
                            "current_offer": pos["market"]["offer"]
                        }
                
                return None
                
        except Exception as e:
            self.logger.error(f"[POSITION DATA ERROR] {deal_id}: {e}")
            return None
    
    async def sync_trade_with_ig(self, trade: TradeLog, trading_headers: dict, db: Session) -> bool:
        """Sync trade record with actual IG position data"""
        try:
            real_data = await self.get_real_position_data(trade.deal_id, trading_headers)

            if not real_data:
                self.logger.error(f"[SYNC FAILED] Trade {trade.id}: Could not get IG position data")
                return False

            # Normalize CEEM prices from IG API (they return scaled prices like 11988.2 instead of 1.19882)
            epic = real_data.get("epic", trade.symbol)
            if "CEEM" in epic:
                entry_price_raw = float(real_data["entry_price"])
                if entry_price_raw > 1000:  # Clearly scaled
                    real_data["entry_price"] = entry_price_raw / 10000.0
                    self.logger.debug(f"[SYNC CEEM] Normalized entry: {entry_price_raw} → {real_data['entry_price']}")

                if real_data.get("current_stop"):
                    stop_raw = float(real_data["current_stop"])
                    if stop_raw > 1000:
                        real_data["current_stop"] = stop_raw / 10000.0
                        self.logger.debug(f"[SYNC CEEM] Normalized stop: {stop_raw} → {real_data['current_stop']}")

                if real_data.get("current_limit"):
                    limit_raw = float(real_data["current_limit"])
                    if limit_raw > 1000:
                        real_data["current_limit"] = limit_raw / 10000.0
                        self.logger.debug(f"[SYNC CEEM] Normalized limit: {limit_raw} → {real_data['current_limit']}")

            # Check for discrepancies and update if needed
            updates_needed = []

            # CRITICAL FIX (Jan 2026): Skip SL/TP sync when trailing is active
            # The trailing system is the source of truth - syncing from IG's cached data
            # was overwriting valid BE/trailing stops with stale values
            trailing_active = getattr(trade, 'moved_to_breakeven', False) or getattr(trade, 'early_be_executed', False)

            if abs(float(real_data["entry_price"]) - trade.entry_price) > 0.00001:
                updates_needed.append(f"entry_price: {trade.entry_price} → {real_data['entry_price']}")
                trade.entry_price = float(real_data["entry_price"])

            # Only sync SL/TP if trailing is NOT active
            if trailing_active:
                self.logger.debug(f"🔒 [SYNC SKIP] Trade {trade.id}: Trailing active, skipping SL/TP sync from IG")
            else:
                if real_data["current_stop"] and trade.sl_price:
                    if abs(float(real_data["current_stop"]) - trade.sl_price) > 0.00001:
                        updates_needed.append(f"sl_price: {trade.sl_price} → {real_data['current_stop']}")
                        trade.sl_price = float(real_data["current_stop"])

                if real_data["current_limit"] and trade.tp_price:
                    if abs(float(real_data["current_limit"]) - trade.tp_price) > 0.00001:
                        updates_needed.append(f"tp_price: {trade.tp_price} → {real_data['current_limit']}")
                        trade.tp_price = float(real_data["current_limit"])

            if updates_needed:
                self.logger.warning(f"[SYNC UPDATES] Trade {trade.id}: {', '.join(updates_needed)}")
                db.commit()
                return True
            else:
                self.logger.debug(f"[SYNC OK] Trade {trade.id}: No updates needed")
                return True
                
        except Exception as e:
            self.logger.error(f"[SYNC ERROR] Trade {trade.id}: {e}")
            return False
    
    async def comprehensive_trade_verification(self, trade: TradeLog, db: Session) -> bool:
        """
        ✅ NEW: Use enhanced status manager for comprehensive trade verification
        This replaces simple deal existence check with intelligent status determination
        """
        try:
            if not self.enhanced_status_manager or not ENHANCED_STATUS_MANAGER_AVAILABLE:
                self.logger.warning(f"⚠️ [VERIFICATION] Enhanced status manager not available for trade {trade.id}, skipping comprehensive verification")
                return True  # Don't block processing if enhanced system unavailable
            
            self.logger.info(f"🔍 [COMPREHENSIVE VERIFICATION] Trade {trade.id} {trade.symbol}")
            
            # Use the enhanced verification system
            final_status = await self.enhanced_status_manager.verify_and_update_trade_status(trade, db)
            
            # Check if trade should continue processing
            if final_status in ["tracking", "break_even", "trailing", "ema_exit_pending", "profit_protected"]:
                self.logger.debug(f"✅ [VERIFICATION PASSED] Trade {trade.id}: Status {final_status} - continuing processing")
                return True
            elif final_status == "closed":
                self.logger.info(f"📊 [TRADE CLOSED] Trade {trade.id}: Detected as closed - {getattr(trade, 'exit_reason', 'unknown reason')}")
                return False
            elif final_status in ["rejected", "invalid_deal", "expired"]:
                self.logger.warning(f"⚠️ [TRADE INVALID] Trade {trade.id}: Status {final_status} - {getattr(trade, 'exit_reason', 'unknown reason')}")
                return False
            elif final_status == "missing_on_ig":
                self.logger.error(f"❌ [TRADE MISSING] Trade {trade.id}: Genuinely missing from IG - {getattr(trade, 'exit_reason', 'unknown reason')}")
                return False
            else:
                self.logger.error(f"❌ [VERIFICATION ERROR] Trade {trade.id}: Unexpected status {final_status}")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ [COMPREHENSIVE VERIFICATION ERROR] Trade {trade.id}: {e}")
            # Don't block processing on verification errors
            return True
    
    async def process_trade_with_combined_validation(self, trade: TradeLog, current_price: float,
                                                   trading_headers: dict = None, db: Session = None) -> bool:
        """
        ✅ ENHANCED: Processing with comprehensive validation using enhanced status management
        Now uses EnhancedTradeStatusManager for intelligent trade verification instead of simple ghost trade detection
        """

        # Use provided headers or stored headers
        headers = trading_headers or self.trading_headers
        if not headers:
            self.logger.error(f"❌ [NO HEADERS] Trade {trade.id}: No trading headers available for validation")
            # Fallback to enhanced processing without validation
            return await self.process_trade_enhanced(trade, current_price, db)

        # Update headers in status manager if needed
        if headers != self.trading_headers:
            self.set_trading_headers(headers)

        self.logger.info(f"🔧 [COMBINED+ ENHANCED] Processing trade {trade.id} {trade.symbol} status={trade.status}")

        try:
            # ✅ STEP 0: Dynamic Config Selection (Jan 2026 - Scalp mode support)
            # v3.2.0: Pass db for ATR-adaptive trailing config lookup
            trade_config = self.get_config_for_trade(trade, db=db)
            # Update the trailing manager's config for this trade
            self.trailing_manager.config = trade_config
            self.config = trade_config  # Also update processor's config reference
            # ✅ ENHANCED STEP 1: Use comprehensive verification instead of simple deal existence check
            if ENHANCED_STATUS_MANAGER_AVAILABLE and self.enhanced_status_manager:
                # Use the new comprehensive verification system
                verification_passed = await self.comprehensive_trade_verification(trade, db)
                if not verification_passed:
                    self.logger.info(f"📊 [VERIFICATION STOP] Trade {trade.id}: Comprehensive verification determined trade should not continue processing")
                    return False
            else:
                # ✅ FALLBACK: Use basic deal existence check
                if not await self.verify_deal_exists(trade.deal_id, headers):
                    self.logger.error(f"❌ [GHOST TRADE] Trade {trade.id} deal {trade.deal_id} not found on IG")
                    trade.status = "missing_on_ig"
                    trade.exit_reason = "basic_verification_failed"
                    trade.trigger_time = datetime.utcnow()
                    db.commit()
                    return False
            
            # ✅ STEP 2: Sync trade data with IG (prevent data drift)
            sync_success = await self.sync_trade_with_ig(trade, headers, db)
            if not sync_success:
                self.logger.warning(f"⚠️ [SYNC WARNING] Trade {trade.id}: Could not sync with IG")
            
            # ✅ STEP 3: Apply enhanced processing with profit protection
            return await self.process_trade_enhanced(trade, current_price, db)
            
        except Exception as e:
            self.logger.error(f"❌ [COMBINED+ ENHANCED ERROR] Trade {trade.id}: {e}")
            return False

    async def process_trade_with_advanced_trailing(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """Enhanced processing with EMA exit check followed by trailing logic - 🔧 FIXED METHOD"""

        # Diagnostic log
        self.logger.info(f"🔧 [COMBINED] Processing trade {trade.id} {trade.symbol} status={trade.status}")

        try:
            # --- Step -1: Dynamic Config Selection (Jan 2026 - Scalp mode support) ---
            # v3.2.0: Pass db for ATR-adaptive trailing config lookup
            trade_config = self.get_config_for_trade(trade, db=db)
            # Update the trailing manager's config for this trade
            self.trailing_manager.config = trade_config
            self.config = trade_config  # Also update processor's config reference

            # --- Step 0: EMA Trend Reversal Check (if enabled) ---
            if self.ema_exit and trade.status in ["tracking", "break_even", "trailing", "profit_protected"]:
                should_exit, exit_reason = self.ema_exit.should_exit_trade(trade, db)

                if should_exit:
                    self.logger.warning(f"[EMA EXIT TRIGGERED] Trade {trade.id} {trade.symbol}: {exit_reason}")
                    success = self.ema_exit.execute_ema_exit(trade, exit_reason, db)
                    return success  # Exit early, don't continue with trailing logic

            # --- Continue with existing break-even and trailing logic ---
            # ✅ CRITICAL FIX: Call the parent class method using super()
            return await super().process_trade_with_advanced_trailing(trade, current_price, db)
            
        except Exception as e:
            self.logger.error(f"❌ [COMBINED ERROR] Trade {trade.id}: {e}")
            return False

    def calculate_limit_adjustment(self, trade: TradeLog, current_price: float, stop_adjustment_points: int) -> int:
        """Calculate how much to adjust the limit order during trailing - ENHANCED"""
        try:
            direction = trade.direction.upper()
            point_value = get_point_value(trade.symbol)
            
            # Calculate limit adjustment based on ratio
            limit_ratio = getattr(self.config, 'limit_trail_ratio', 1.0)
            base_limit_adjustment = int(stop_adjustment_points * limit_ratio)
            
            # Get minimum limit distance requirement
            min_limit_distance = getattr(self.config, 'min_limit_distance', 3)
            
            # Calculate what the new limit would be
            proposed_limit = self.calculate_new_limit_price(trade, current_price, base_limit_adjustment)
            
            # Validate minimum distance from current price
            distance_from_current = abs(proposed_limit - current_price) / point_value
            
            if distance_from_current < min_limit_distance:
                # Adjust to maintain minimum distance
                if direction == "BUY":
                    safe_limit = current_price + (min_limit_distance * point_value)
                else:
                    safe_limit = current_price - (min_limit_distance * point_value)
                
                # Recalculate adjustment points
                current_limit = trade.tp_price or self.get_default_limit(trade, current_price)
                if direction == "BUY":
                    limit_adjustment_distance = safe_limit - current_limit
                else:
                    limit_adjustment_distance = current_limit - safe_limit
                
                adjusted_limit_points = max(0, int(abs(limit_adjustment_distance) / point_value))
                
                self.logger.debug(f"[LIMIT SAFE] Trade {trade.id} {trade.symbol}: adjusted to maintain {min_limit_distance}pt minimum")
                return adjusted_limit_points
            
            self.logger.debug(f"[LIMIT CALC] Trade {trade.id} {trade.symbol}: {base_limit_adjustment}pts (ratio: {limit_ratio})")
            return max(0, base_limit_adjustment)
            
        except Exception as e:
            self.logger.error(f"[LIMIT CALC ERROR] Trade {trade.id}: {e}")
            return 0

    def calculate_new_limit_price(self, trade: TradeLog, current_price: float, limit_adjustment_points: int) -> float:
        """Calculate the new limit price after adjustment - ENHANCED"""
        try:
            direction = trade.direction.upper()
            point_value = get_point_value(trade.symbol)
            current_limit = trade.tp_price or self.get_default_limit(trade, current_price)
            
            adjustment_distance = limit_adjustment_points * point_value
            
            if direction == "BUY":
                return current_limit + adjustment_distance  # Move limit higher
            else:
                return current_limit - adjustment_distance  # Move limit lower
        except Exception as e:
            self.logger.error(f"[NEW LIMIT CALC ERROR] Trade {trade.id}: {e}")
            return trade.tp_price or current_price

    def get_default_limit(self, trade: TradeLog, current_price: float) -> float:
        """Get default limit if not set in trade - ENHANCED"""
        try:
            direction = trade.direction.upper()
            point_value = get_point_value(trade.symbol)
            default_distance = 10 * point_value  # 10 points default
            
            if direction == "BUY":
                return current_price + default_distance
            else:
                return current_price - default_distance
        except Exception as e:
            self.logger.error(f"[DEFAULT LIMIT ERROR] Trade {trade.id}: {e}")
            return current_price

    # ===============================================================================
    # ADD MISSING TRAILING METHODS
    # ===============================================================================

    def _apply_fixed_points_trailing(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """Apply fixed points trailing using the AdvancedTrailingManager"""
        try:
            self.logger.info(f"🎯 [FIXED POINTS TRAILING] Trade {trade.id}: Applying fixed points trailing")
            
            # Check if we should trail
            if not self.trailing_manager.should_update_trail(trade, current_price, db):
                self.logger.info(f"⏸️ [TRAIL SKIP] Trade {trade.id}: Should not trail at this time")
                return True  # Not an error, just no action needed
            
            # Calculate new trail level
            new_trail_level = self.trailing_manager.calculate_new_trail_level(trade, current_price, db)
            
            if new_trail_level is None:
                self.logger.warning(f"⚠️ [TRAIL CALC] Trade {trade.id}: Could not calculate trail level")
                return True  # Not an error, just no action needed
            
            # Check if this is actually an improvement
            current_stop = trade.sl_price or 0.0
            direction = trade.direction.upper()
            
            is_improvement = False
            if direction == "BUY":
                is_improvement = new_trail_level > current_stop
            else:  # SELL
                is_improvement = new_trail_level < current_stop
            
            if not is_improvement:
                self.logger.info(f"⏸️ [NO IMPROVEMENT] Trade {trade.id}: New trail level {new_trail_level:.5f} not better than current {current_stop:.5f}")
                return True  # Not an error, just no action needed
            
            # Calculate adjustment points
            adjustment_points = self.trailing_manager.get_trail_adjustment_points(trade, current_price, new_trail_level)
            
            if adjustment_points <= 0:
                self.logger.warning(f"⚠️ [INVALID ADJUSTMENT] Trade {trade.id}: Adjustment points = {adjustment_points}")
                return True  # Not an error, just no action needed
            
            # Send the adjustment
            direction_stop = "increase" if direction == "BUY" else "decrease"
            success = self._send_stop_adjustment(trade, adjustment_points, direction_stop, 0)
            
            if success:
                # Update trade record
                trade.sl_price = new_trail_level
                trade.last_trigger_price = current_price
                trade.trigger_time = datetime.utcnow()
                if trade.status != "profit_protected":
                    trade.status = "trailing"
                db.commit()
                
                self.logger.info(f"✅ [TRAILING SUCCESS] Trade {trade.id}: Stop moved to {new_trail_level:.5f} ({adjustment_points}pts)")
                return True
            else:
                self.logger.error(f"❌ [TRAILING FAILED] Trade {trade.id}: Could not send stop adjustment")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ [FIXED POINTS TRAILING ERROR] Trade {trade.id}: {e}")
            return False

    def _apply_percentage_trailing(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """Apply percentage-based trailing - delegates to fixed points method"""
        return self._apply_fixed_points_trailing(trade, current_price, db)

    def _apply_atr_trailing(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """Apply ATR-based trailing - delegates to fixed points method"""
        return self._apply_fixed_points_trailing(trade, current_price, db)

    def _apply_volatility_adaptive_trailing(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """Apply volatility adaptive trailing - delegates to fixed points method"""
        return self._apply_fixed_points_trailing(trade, current_price, db)

    def _send_stop_adjustment(self, trade: TradeLog, stop_points: int,
                            direction_stop: str, limit_points: int,
                            new_stop_level: float = None) -> dict:
        """
        Send stop adjustment to order system - ENHANCED WITH ABSOLUTE LEVEL SUPPORT

        Args:
            trade: The trade to adjust
            stop_points: Points to adjust (for legacy offset mode)
            direction_stop: 'increase' or 'decrease'
            limit_points: Points to adjust limit (for legacy offset mode)
            new_stop_level: PREFERRED - Absolute stop level to set directly

        Returns:
            dict: API response with status and details
        """
        try:
            from services.adjust_stop_service import adjust_stop_sync

            # ✅ CRITICAL FIX: Use absolute stop level when provided
            if new_stop_level is not None:
                self.logger.info(f"📍 [ABSOLUTE STOP] Trade {trade.id} {trade.symbol}: Setting stop directly to {new_stop_level:.5f}")

                result = adjust_stop_sync(
                    epic=trade.symbol,
                    new_stop=new_stop_level,  # Use absolute level - no offset calculation needed
                    stop_offset_points=None,
                    limit_offset_points=None,
                    dry_run=False
                )
            else:
                # Legacy offset-based mode (fallback)
                self.logger.info(f"[OFFSET MODE] Trade {trade.id} {trade.symbol}: Adjusting stop by {stop_points}pts ({direction_stop})")

                result = adjust_stop_sync(
                    epic=trade.symbol,
                    stop_offset_points=stop_points,
                    limit_offset_points=limit_points,
                    adjust_direction_stop=direction_stop,
                    adjust_direction_limit="increase",
                    dry_run=False
                )

            status = result.get("status", "unknown")

            if status == "updated":
                sent_payload = result.get("sentPayload", {})
                actual_stop = sent_payload.get("stopLevel")
                self.logger.info(f"[✅ STOP UPDATED] {trade.symbol} → IG set stopLevel={actual_stop}")
                return result
            elif status == "closed":
                self.logger.warning(f"[❌ POSITION CLOSED] {trade.symbol}: {result.get('message')}")
                return result
            else:
                self.logger.error(f"[❌ ADJUSTMENT FAILED] {trade.symbol}: {result.get('message', 'Unknown error')}")
                return result

        except Exception as e:
            self.logger.error(f"❌ [SEND ERROR] Trade {trade.id}: {e}")
            return {"status": "error", "message": str(e)}

    def validate_stop_level(self, trade: TradeLog, current_price: float, proposed_stop: float) -> bool:
        """Validate that the proposed stop level meets IG's minimum distance requirements"""
        try:
            # Use the trailing manager's validation
            return self.trailing_manager.validate_stop_level(trade, current_price, proposed_stop)
        except Exception as e:
            self.logger.error(f"[VALIDATION ERROR] Trade {trade.id}: {e}")
            return False


# ✅ UPDATED: Pre-configured setups for different trading styles WITH PROFIT PROTECTION
SCALPING_CONFIG_WITH_EMA = CombinedTrailingConfig(
    method=TrailingMethod.FIXED_POINTS,
    initial_trigger_points=6,
    break_even_trigger_points=8,  # ✅ ADDED: Required for new system
    min_trail_distance=2,
    max_trail_distance=20,
    monitor_interval_seconds=60,
    # EMA Exit Settings for Scalping
    enable_ema_exit=False,
    ema_confirmation_candles=2,
    ema_timeframe=60,
    # Limit trailing settings
    enable_limit_trailing=True,
    limit_trail_ratio=1.0,
    min_limit_distance=3,
    # ✅ NEW: Profit Protection Rule
    enable_profit_protection_rule=False,  # DISABLED BY DEFAULT - configure per pair
    profit_protection_trigger=15,
    profit_protection_stop=10
)

# ✅ NEW: Profit protection config that was missing from diagnostic
SCALPING_CONFIG_WITH_PROTECTION = CombinedTrailingConfig(
    method=TrailingMethod.FIXED_POINTS,
    initial_trigger_points=6,
    break_even_trigger_points=8,
    min_trail_distance=2,
    max_trail_distance=20,
    monitor_interval_seconds=60,
    # EMA Exit Settings
    enable_ema_exit=False,
    ema_confirmation_candles=2,
    ema_timeframe=60,
    # Limit trailing settings
    enable_limit_trailing=True,
    limit_trail_ratio=1.0,
    min_limit_distance=3,
    # ✅ PROFIT PROTECTION RULE - This fixes the diagnostic issue
    enable_profit_protection_rule=False,  # DISABLED BY DEFAULT - configure per pair
    profit_protection_trigger=15,    # At 15 points profit
    profit_protection_stop=10        # Move stop to 10 points profit
)

SWING_CONFIG_WITH_EMA = CombinedTrailingConfig(
    method=TrailingMethod.ATR_BASED,
    initial_trigger_points=15,
    break_even_trigger_points=10,  # ✅ ADDED: Required for new system
    atr_multiplier=2.5,
    min_trail_distance=10,
    max_trail_distance=100,
    monitor_interval_seconds=60,
    # EMA Exit Settings for Swing Trading
    enable_ema_exit=False,
    ema_confirmation_candles=3,
    ema_timeframe=240,
    # ✅ NEW: Profit Protection Rule
    enable_profit_protection_rule=False,  # DISABLED BY DEFAULT - configure per pair
    profit_protection_trigger=15,
    profit_protection_stop=10
)

# Config with EMA exit disabled (for testing/comparison)
SCALPING_CONFIG_NO_EMA = CombinedTrailingConfig(
    method=TrailingMethod.FIXED_POINTS,
    initial_trigger_points=5,
    break_even_trigger_points=6,  # ✅ ADDED: Required for new system
    min_trail_distance=4,
    max_trail_distance=15,
    monitor_interval_seconds=60,
    enable_ema_exit=False,
    # ✅ NEW: Profit Protection Rule
    enable_profit_protection_rule=False,  # DISABLED BY DEFAULT - configure per pair
    profit_protection_trigger=15,
    profit_protection_stop=10
)

# Conservative config for high-volatility pairs
CONSERVATIVE_CONFIG_WITH_PROTECTION = CombinedTrailingConfig(
    method=TrailingMethod.FIXED_POINTS,
    initial_trigger_points=12,
    break_even_trigger_points=15,
    min_trail_distance=5,
    max_trail_distance=40,
    monitor_interval_seconds=90,
    enable_ema_exit=False,
    ema_confirmation_candles=4,
    ema_timeframe=240,
    enable_limit_trailing=True,
    limit_trail_ratio=1.2,
    min_limit_distance=8,
    # ✅ NEW: Profit Protection Rule - More conservative for volatile pairs
    enable_profit_protection_rule=False,  # DISABLED BY DEFAULT - configure per pair
    profit_protection_trigger=18,  # Wait a bit longer for volatile pairs
    profit_protection_stop=12      # Secure slightly higher profit
)

# ✅ NEW: Function that pair_specific_config.py was looking for
def get_config_for_trade(trade: TradeLog, logger=None) -> CombinedTrailingConfig:
    """
    Get configuration for a trade - integrates with pair-specific config system
    This function bridges the gap between the diagnostic error and pair-specific config
    """
    try:
        # Try to use pair-specific configuration if available
        from services.pair_specific_config import get_protected_config_for_trade
        return get_protected_config_for_trade(trade, logger)
    except ImportError:
        # Fallback to default protection config
        if logger:
            logger.info("🔄 Using fallback profit protection config")
        return SCALPING_CONFIG_WITH_PROTECTION


def test_profit_protection_logic():
    """Test the profit protection rule logic"""
    print("\n" + "="*80)
    print("🛡️ PROFIT PROTECTION RULE TEST")
    print("="*80)
    print("Testing 15-point → 10-point profit protection logic")
    print("="*80)
    
    # Mock trade data for testing
    class MockTrade:
        def __init__(self):
            self.id = "TEST001"
            from config import DEFAULT_EPICS
            self.symbol = DEFAULT_EPICS['EURUSD']
            self.direction = "BUY"
            self.entry_price = 1.10000
            self.sl_price = 1.09800  # 20 pips stop
            self.tp_price = 1.10400  # 40 pips target
            self.deal_id = "DEAL123"
            self.status = "tracking"
    
    # Test scenarios
    scenarios = [
        {"current_price": 1.10050, "expected_profit": 5.0, "should_trigger": False},
        {"current_price": 1.10100, "expected_profit": 10.0, "should_trigger": False},
        {"current_price": 1.10150, "expected_profit": 15.0, "should_trigger": True},
        {"current_price": 1.10200, "expected_profit": 20.0, "should_trigger": True},
    ]
    
    # Create mock processor with protection rule
    config = SCALPING_CONFIG_WITH_PROTECTION
    
    print(f"📊 Configuration:")
    print(f"   Trigger: {config.profit_protection_trigger} points")
    print(f"   Stop move to: {config.profit_protection_stop} points")
    print(f"   Rule enabled: {config.enable_profit_protection_rule}")
    print(f"   Enhanced status manager: {'Available' if ENHANCED_STATUS_MANAGER_AVAILABLE else 'Not Available'}")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"🧪 Scenario {i}: Price = {scenario['current_price']}")
        
        # Calculate profit points (simplified)
        trade = MockTrade()
        profit_points = (scenario['current_price'] - trade.entry_price) / 0.0001
        
        print(f"   Expected profit: {scenario['expected_profit']} points")
        print(f"   Calculated profit: {profit_points:.1f} points")
        print(f"   Should trigger: {scenario['should_trigger']}")
        print(f"   Actual trigger: {profit_points >= config.profit_protection_trigger}")
        
        if profit_points >= config.profit_protection_trigger:
            new_stop = trade.entry_price + (config.profit_protection_stop * 0.0001)
            print(f"   ✅ New stop would be: {new_stop:.5f} (+{config.profit_protection_stop}pts)")
        else:
            print(f"   ⏳ No action (need {config.profit_protection_trigger - profit_points:.1f} more points)")
        
        print()
    
    print("✅ Enhanced trade processor test completed")
    print("\n🎯 Expected behavior:")
    print("   • At 15+ points profit: Stop moves to +10 points profit")
    print("   • Protects minimum 10-point win instead of risking small wins")
    print("   • Rule applies only once per trade")
    print("   • Normal trailing continues after protection is applied")
    print("   • Enhanced status management prevents ghost trades")
    print("   • Comprehensive verification determines actual trade status")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_profit_protection_logic()