# services/trade_sync.py
"""
IG Trade Sync Task - ENHANCED VERSION WITH INTELLIGENT STATUS MANAGEMENT

FIXED: Now validates trades by DEAL_ID instead of SYMBOL
This prevents false ghost trade detection when multiple trades exist for the same symbol.

ENHANCED: Instead of immediately marking trades as missing_on_ig, now uses comprehensive
verification using deal_id and deal_reference to determine what actually happened to trades.

NEW FALLBACK: When a trade appears missing, fetches ALL open positions to double-check
if the trade still exists (handles API glitches or timing issues).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from sqlalchemy.orm import Session
from services.db import SessionLocal
from services.models import TradeLog
from dependencies import get_ig_auth_headers
import httpx
from config import API_BASE_URL

logger = logging.getLogger("trade_sync")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("/app/logs/trade_sync.log")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class EnhancedTradeStatusManager:
    """
    Enhanced trade status management that uses deal_id and deal_reference 
    to make intelligent decisions about trade status instead of just marking missing_on_ig
    
    NEW: Added fallback mechanism to check ALL open positions when trade appears missing
    """
    
    def __init__(self, trading_headers: dict):
        self.trading_headers = trading_headers
        self.logger = logger
        self._positions_cache = None  # Cache for all positions
        self._cache_timestamp = None
        self._cache_ttl_seconds = 30  # Cache positions for 30 seconds
    
    async def verify_and_update_trade_status(self, trade: TradeLog, db: Session) -> str:
        """
        Comprehensive trade verification using stored deal_id and deal_reference
        Returns the determined status with full reasoning
        """
        try:
            self.logger.info(f"🔍 [VERIFICATION] Trade {trade.id} {trade.symbol} - Starting comprehensive verification")
            
            # Step 1: Check if position still exists (current system)
            position_exists = await self._check_position_exists(trade.deal_id)
            
            if position_exists:
                self.logger.info(f"✅ [ACTIVE] Trade {trade.id} position still active on IG")
                return self._update_trade_status(trade, "tracking", "position_verified_active", db)
            
            # Step 2: Position not found - investigate using deal_reference
            self.logger.info(f"🔎 [INVESTIGATE] Trade {trade.id} position not found, investigating with deal_reference")
            
            # Check deal confirmation to see what happened
            deal_outcome = await self._investigate_deal_outcome(trade.deal_reference)
            
            if deal_outcome:
                return self._handle_deal_outcome(trade, deal_outcome, db)
            
            # Step 3: Check transaction history for this deal
            transaction_result = await self._check_transaction_history(trade.deal_id, trade.deal_reference)
            
            if transaction_result:
                return self._handle_transaction_result(trade, transaction_result, db)
            
            # Step 4: NEW FALLBACK - Get ALL positions and check both deal_id and deal_reference
            self.logger.info(f"🔄 [FALLBACK] Trade {trade.id} - Fetching ALL positions to double-check")
            all_positions_check = await self._comprehensive_position_check(trade)
            
            if all_positions_check:
                self.logger.warning(f"⚠️ [FOUND IN FALLBACK] Trade {trade.id} found in comprehensive check!")
                return self._update_trade_status(trade, "tracking", "found_in_fallback_check", db)
            
            # Step 5: Final fallback - check if deal was ever valid
            deal_validity = await self._verify_deal_validity(trade.deal_reference)
            
            return self._handle_final_determination(trade, deal_validity, db)
            
        except Exception as e:
            self.logger.error(f"❌ [VERIFICATION ERROR] Trade {trade.id}: {e}")
            return self._update_trade_status(trade, "verification_error", f"Error during verification: {str(e)}", db)
    
    async def _get_all_positions(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get all open positions from IG with caching
        """
        now = datetime.utcnow()
        
        # Check if we have valid cached positions
        if (not force_refresh and 
            self._positions_cache is not None and 
            self._cache_timestamp and 
            (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds):
            self.logger.debug(f"📦 Using cached positions (age: {(now - self._cache_timestamp).total_seconds():.1f}s)")
            return self._positions_cache
        
        # Fetch fresh positions
        try:
            url = f"{API_BASE_URL}/positions"
            headers = {
                "X-IG-API-KEY": self.trading_headers["X-IG-API-KEY"],
                "CST": self.trading_headers["CST"],
                "X-SECURITY-TOKEN": self.trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                positions = response.json().get("positions", [])
                
                # Update cache
                self._positions_cache = positions
                self._cache_timestamp = now
                
                self.logger.info(f"📊 Fetched {len(positions)} open positions from IG")
                return positions
                
        except Exception as e:
            self.logger.error(f"❌ [GET ALL POSITIONS ERROR]: {e}")
            # Return cached positions if available, even if expired
            if self._positions_cache is not None:
                self.logger.warning("⚠️ Using expired cache due to API error")
                return self._positions_cache
            return []
    
    async def _comprehensive_position_check(self, trade: TradeLog) -> bool:
        """
        NEW: Comprehensive check of ALL positions looking for the trade
        Checks both deal_id and deal_reference in all position data
        """
        try:
            # Get all positions (will use cache if recent)
            all_positions = await self._get_all_positions()
            
            if not all_positions:
                self.logger.warning(f"⚠️ No positions available for comprehensive check")
                return False
            
            # Check by deal_id first
            for pos in all_positions:
                position_data = pos.get("position", {})
                
                # Direct deal_id match
                if position_data.get("dealId") == trade.deal_id:
                    self.logger.info(f"✅ [FALLBACK HIT] Found trade {trade.id} by deal_id: {trade.deal_id}")
                    return True
                
                # Check if deal_reference matches (sometimes stored in different field)
                if position_data.get("dealReference") == trade.deal_reference:
                    self.logger.info(f"✅ [FALLBACK HIT] Found trade {trade.id} by deal_reference: {trade.deal_reference}")
                    # Update the deal_id if it's different
                    if position_data.get("dealId") != trade.deal_id:
                        self.logger.warning(f"⚠️ Updating deal_id from {trade.deal_id} to {position_data.get('dealId')}")
                        trade.deal_id = position_data.get("dealId")
                    return True
            
            # Extended check: Look for matching symbol, direction, and approximate entry price
            for pos in all_positions:
                position_data = pos.get("position", {})
                market_data = pos.get("market", {})
                
                # Check if this could be our trade based on multiple factors
                if (market_data.get("epic") == trade.symbol and
                    position_data.get("direction") == trade.direction):
                    
                    # Check if entry prices are close (within 0.1%)
                    pos_entry = float(position_data.get("level", 0))
                    if pos_entry > 0 and trade.entry_price:
                        price_diff_pct = abs(pos_entry - trade.entry_price) / trade.entry_price * 100
                        if price_diff_pct < 0.1:  # Within 0.1%
                            self.logger.warning(f"⚠️ [POSSIBLE MATCH] Found possible trade match by symbol/direction/price")
                            self.logger.warning(f"   Position: {position_data.get('dealId')} vs Trade: {trade.deal_id}")
                            self.logger.warning(f"   Entry: {pos_entry} vs {trade.entry_price} (diff: {price_diff_pct:.3f}%)")
                            # Don't auto-update here, just log for investigation
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [COMPREHENSIVE CHECK ERROR] Trade {trade.id}: {e}")
            return False
    
    async def verify_missing_trades(self, db: Session) -> Dict[str, int]:
        """
        NEW: Specifically check trades marked as missing_on_ig against all open positions
        This is a batch operation to efficiently verify multiple missing trades
        """
        stats = {
            "checked": 0,
            "recovered": 0,
            "still_missing": 0,
            "errors": 0
        }
        
        try:
            # Get all trades marked as missing
            missing_trades = db.query(TradeLog).filter(
                TradeLog.status == "missing_on_ig",
                TradeLog.endpoint == "dev"
            ).all()
            
            if not missing_trades:
                self.logger.info("📭 No missing trades to verify")
                return stats
            
            self.logger.info(f"🔍 Verifying {len(missing_trades)} trades marked as missing_on_ig")
            stats["checked"] = len(missing_trades)
            
            # Get all open positions (fresh fetch)
            all_positions = await self._get_all_positions(force_refresh=True)
            
            if not all_positions:
                self.logger.warning("⚠️ No open positions found on IG")
                stats["still_missing"] = len(missing_trades)
                return stats
            
            # Create lookup sets for faster checking
            ig_deal_ids = {pos["position"]["dealId"] for pos in all_positions}
            ig_deal_refs = {pos["position"].get("dealReference") for pos in all_positions if pos["position"].get("dealReference")}
            
            # Also create a mapping of epic+direction to positions for fuzzy matching
            position_map = {}
            for pos in all_positions:
                key = f"{pos['market']['epic']}_{pos['position']['direction']}"
                if key not in position_map:
                    position_map[key] = []
                position_map[key].append(pos)
            
            # Check each missing trade
            for trade in missing_trades:
                try:
                    found = False
                    
                    # Check by deal_id
                    if trade.deal_id in ig_deal_ids:
                        self.logger.info(f"✅ [RECOVERED] Trade {trade.id} found by deal_id: {trade.deal_id}")
                        trade.status = "tracking"
                        trade.exit_reason = "recovered_from_missing"
                        trade.trigger_time = datetime.utcnow()
                        stats["recovered"] += 1
                        found = True
                    
                    # Check by deal_reference
                    elif trade.deal_reference in ig_deal_refs:
                        self.logger.info(f"✅ [RECOVERED] Trade {trade.id} found by deal_reference: {trade.deal_reference}")
                        # Find the matching position to update deal_id
                        for pos in all_positions:
                            if pos["position"].get("dealReference") == trade.deal_reference:
                                trade.deal_id = pos["position"]["dealId"]
                                self.logger.info(f"   Updated deal_id to: {trade.deal_id}")
                                break
                        trade.status = "tracking"
                        trade.exit_reason = "recovered_from_missing_by_reference"
                        trade.trigger_time = datetime.utcnow()
                        stats["recovered"] += 1
                        found = True
                    
                    # Fuzzy check by symbol+direction+price
                    elif not found:
                        key = f"{trade.symbol}_{trade.direction}"
                        if key in position_map:
                            for pos in position_map[key]:
                                pos_entry = float(pos["position"]["level"])
                                if trade.entry_price:
                                    price_diff_pct = abs(pos_entry - trade.entry_price) / trade.entry_price * 100
                                    if price_diff_pct < 0.05:  # Within 0.05% - very close match
                                        self.logger.warning(f"⚠️ [LIKELY MATCH] Trade {trade.id} likely matches position {pos['position']['dealId']}")
                                        self.logger.warning(f"   Symbol: {trade.symbol}, Direction: {trade.direction}")
                                        self.logger.warning(f"   Entry prices: {pos_entry} vs {trade.entry_price} (diff: {price_diff_pct:.4f}%)")
                                        # Update the trade with the correct deal_id
                                        trade.deal_id = pos["position"]["dealId"]
                                        trade.status = "tracking"
                                        trade.exit_reason = "recovered_by_fuzzy_match"
                                        trade.trigger_time = datetime.utcnow()
                                        stats["recovered"] += 1
                                        found = True
                                        break
                    
                    if not found:
                        stats["still_missing"] += 1
                        self.logger.info(f"❌ Trade {trade.id} still missing after comprehensive check")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error checking trade {trade.id}: {e}")
                    stats["errors"] += 1
            
            # Commit all changes
            db.commit()
            
            self.logger.info(f"📊 Missing trade verification complete:")
            self.logger.info(f"   Checked: {stats['checked']}")
            self.logger.info(f"   Recovered: {stats['recovered']}")
            self.logger.info(f"   Still Missing: {stats['still_missing']}")
            self.logger.info(f"   Errors: {stats['errors']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in verify_missing_trades: {e}")
            stats["errors"] += 1
        
        return stats
    
    async def _check_position_exists(self, deal_id: str) -> bool:
        """Check if position still exists in open positions"""
        try:
            # Use cached positions if available
            positions = await self._get_all_positions()
            return any(pos["position"]["dealId"] == deal_id for pos in positions)
                
        except Exception as e:
            self.logger.error(f"❌ [POSITION CHECK ERROR] {deal_id}: {e}")
            return False
    
    async def _investigate_deal_outcome(self, deal_reference: str) -> Optional[Dict]:
        """Use deal_reference to get the final outcome of the deal"""
        if not deal_reference:
            self.logger.warning("⚠️ [DEAL OUTCOME] No deal_reference provided")
            return None
            
        try:
            url = f"{API_BASE_URL}/confirms/{deal_reference}"
            headers = {
                "X-IG-API-KEY": self.trading_headers["X-IG-API-KEY"],
                "CST": self.trading_headers["CST"],
                "X-SECURITY-TOKEN": self.trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "1"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(f"✅ [DEAL OUTCOME] {deal_reference}: {data.get('status', 'Unknown')}")
                    return data
                elif response.status_code == 404:
                    self.logger.warning(f"⚠️ [DEAL OUTCOME] {deal_reference}: Deal reference not found (expired or invalid)")
                    return {"status": "NOT_FOUND", "reason": "Deal reference expired or invalid"}
                else:
                    self.logger.warning(f"⚠️ [DEAL OUTCOME] {deal_reference}: HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ [DEAL OUTCOME ERROR] {deal_reference}: {e}")
            return None
    
    async def _check_transaction_history(self, deal_id: str, deal_reference: str) -> Optional[Dict]:
        """Check transaction history to see if trade was closed"""
        try:
            # Get last 7 days of transactions
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)
            
            url = f"{API_BASE_URL}/history/transactions"
            headers = {
                "X-IG-API-KEY": self.trading_headers["X-IG-API-KEY"],
                "CST": self.trading_headers["CST"],
                "X-SECURITY-TOKEN": self.trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }
            
            params = {
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "maxSpanSeconds": 604800,  # 7 days
                "pageSize": 500
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                transactions = response.json().get("transactions", [])
                
                # Look for transactions related to this deal
                for tx in transactions:
                    tx_deal_id = tx.get("dealId") or tx.get("dealid")
                    tx_reference = tx.get("dealReference") or tx.get("reference")
                    
                    if tx_deal_id == deal_id or tx_reference == deal_reference:
                        self.logger.info(f"📋 [TRANSACTION FOUND] Deal {deal_id}: {tx.get('transactionType', 'Unknown')} - {tx.get('size', 'Unknown')} - {tx.get('profit', 'Unknown')}")
                        return {
                            "found": True,
                            "transaction_type": tx.get("transactionType"),
                            "profit": tx.get("profit"),
                            "size": tx.get("size"),
                            "close_level": tx.get("level"),
                            "date": tx.get("date")
                        }
                
                self.logger.info(f"📋 [NO TRANSACTION] No transaction found for deal {deal_id}")
                return {"found": False}
                
        except Exception as e:
            self.logger.error(f"❌ [TRANSACTION CHECK ERROR] {deal_id}: {e}")
            return None
    
    async def _verify_deal_validity(self, deal_reference: str) -> Dict:
        """Final check to see if deal was ever valid"""
        if not deal_reference or len(deal_reference) < 10:
            return {"valid": False, "reason": "Invalid deal reference format"}
        
        # Check if deal reference format appears valid
        if not deal_reference.replace("-", "").replace("_", "").isalnum():
            return {"valid": False, "reason": "Deal reference contains invalid characters"}
        
        # Check length (IG deal references are typically 12-20 characters)
        if len(deal_reference) > 25:
            return {"valid": False, "reason": "Deal reference too long"}
        
        return {"valid": True, "reason": "Deal reference format appears valid"}
    
    def _handle_deal_outcome(self, trade: TradeLog, deal_outcome: Dict, db: Session) -> str:
        """Handle the result from deal confirmation"""
        status = deal_outcome.get("status", "").upper()
        
        if status == "ACCEPTED":
            # Deal was accepted but position not found - likely closed
            return self._update_trade_status(trade, "closed", "deal_accepted_but_position_closed", db)
        
        elif status == "REJECTED":
            reason = deal_outcome.get("reason", "Unknown rejection reason")
            return self._update_trade_status(trade, "rejected", f"deal_rejected: {reason}", db)
        
        elif status == "NOT_FOUND":
            return self._update_trade_status(trade, "expired", "deal_reference_expired", db)
        
        else:
            return self._update_trade_status(trade, "unknown_outcome", f"deal_status: {status}", db)
    
    def _handle_transaction_result(self, trade: TradeLog, transaction_result: Dict, db: Session) -> str:
        """Handle transaction history findings"""
        if transaction_result.get("found"):
            tx_type = transaction_result.get("transaction_type", "").upper()
            profit = transaction_result.get("profit", 0)
            
            # Trade was found in transaction history - it was closed
            if "CLOSE" in tx_type or "STOP" in tx_type or "LIMIT" in tx_type:
                exit_reason = f"closed_via_{tx_type.lower()}"
                if profit:
                    exit_reason += f"_profit_{profit}"
                
                return self._update_trade_status(trade, "closed", exit_reason, db)
            else:
                return self._update_trade_status(trade, "closed", f"transaction_found_{tx_type.lower()}", db)
        else:
            # No transaction found - this is genuinely problematic
            return self._update_trade_status(trade, "missing_on_ig", "no_position_no_transaction", db)
    
    def _handle_final_determination(self, trade: TradeLog, deal_validity: Dict, db: Session) -> str:
        """Make final determination about trade status"""
        if not deal_validity.get("valid"):
            reason = deal_validity.get("reason", "Invalid deal")
            return self._update_trade_status(trade, "invalid_deal", reason, db)
        
        # If we reach here, we have a valid deal but can't find it anywhere
        # This could indicate a temporary IG API issue or genuine missing trade
        return self._update_trade_status(trade, "missing_on_ig", "valid_deal_but_not_found", db)
    
    def _update_trade_status(self, trade: TradeLog, status: str, exit_reason: str, db: Session) -> str:
        """Update trade status with detailed reasoning"""
        old_status = trade.status
        trade.status = status
        trade.exit_reason = exit_reason
        trade.trigger_time = datetime.utcnow()
        
        # Don't commit here - let the caller handle it
        self.logger.info(f"📊 [STATUS UPDATE] Trade {trade.id}: {old_status} → {status} (reason: {exit_reason})")
        return status

async def sync_trades_with_ig():
    """
    ENHANCED: Now uses comprehensive verification instead of just marking missing_on_ig
    Also includes periodic recovery check for trades marked as missing
    """
    try:
        with SessionLocal() as db:
            # Get active trades with deal_ids
            open_trades = db.query(TradeLog).filter(
                TradeLog.status.in_(["pending", "tracking", "break_even", "trailing"]),
                TradeLog.deal_id.isnot(None),  # Only trades with deal_ids
                TradeLog.endpoint == "dev"
            ).all()
            
            if not open_trades:
                logger.info("[SYNC] No active trades to validate")
            else:
                logger.info(f"[SYNC] Validating {len(open_trades)} active trades with deal_ids...")

                # Get IG positions
                trading_headers = await get_ig_auth_headers()
                
                # Initialize enhanced status manager
                status_manager = EnhancedTradeStatusManager(trading_headers)
                
                # Track results
                validated = 0
                investigated = 0
                still_active = 0
                closed = 0
                rejected = 0
                missing = 0
                errors = 0
                
                for trade in open_trades:
                    try:
                        # Use enhanced verification for each trade
                        final_status = await status_manager.verify_and_update_trade_status(trade, db)
                        investigated += 1
                        
                        # Count by final status
                        if final_status in ["tracking", "break_even", "trailing"]:
                            still_active += 1
                        elif final_status == "closed":
                            closed += 1
                        elif final_status in ["rejected", "invalid_deal"]:
                            rejected += 1
                        elif final_status == "missing_on_ig":
                            missing += 1
                        else:
                            errors += 1
                            
                    except Exception as e:
                        logger.error(f"❌ [TRADE ERROR] Trade {trade.id}: {e}")
                        errors += 1

                # Commit all changes at once
                if investigated > 0:
                    db.commit()
                    logger.info(f"[✅ ENHANCED SYNC] {validated} validated, {investigated} investigated")
                    logger.info(f"   Results: {still_active} active, {closed} closed, {rejected} rejected, {missing} missing, {errors} errors")
                else:
                    logger.info(f"[✅ SYNC] All trades processed successfully")
            
            # NEW: Periodically check trades marked as missing_on_ig
            # Run this check every 5 minutes to recover trades that might have been incorrectly marked
            current_minute = datetime.utcnow().minute
            if current_minute % 5 == 0:  # Every 5 minutes
                logger.info("[RECOVERY] Running periodic check for missing trades...")
                trading_headers = await get_ig_auth_headers()
                status_manager = EnhancedTradeStatusManager(trading_headers)
                recovery_stats = await status_manager.verify_missing_trades(db)
                
                if recovery_stats["recovered"] > 0:
                    logger.info(f"🎉 [RECOVERY SUCCESS] Recovered {recovery_stats['recovered']} trades!")
                
    except Exception as e:
        logger.exception("[❌ SYNC FAILED]")

async def periodic_trade_sync(interval_seconds: int = 300):
    """Periodic sync function - unchanged"""
    while True:
        await sync_trades_with_ig()
        await asyncio.sleep(interval_seconds)

# Test function to verify the fix
async def test_sync_logic():
    """Test function to verify the corrected sync logic"""
    print("🧪 Testing corrected sync logic...")
    
    try:
        with SessionLocal() as db:
            # Get some sample trades
            sample_trades = db.query(TradeLog).filter(
                TradeLog.deal_id.isnot(None)
            ).limit(5).all()
            
            if not sample_trades:
                print("❌ No trades with deal_ids found for testing")
                return
            
            print(f"📊 Found {len(sample_trades)} sample trades:")
            for trade in sample_trades:
                print(f"   Trade {trade.id}: {trade.deal_id} - {trade.symbol} {trade.direction} ({trade.status})")
            
            # Get IG positions
            trading_headers = await get_ig_auth_headers()
            headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }

            async with httpx.AsyncClient() as client:
                r = await client.get(f"{API_BASE_URL}/positions", headers=headers)
                r.raise_for_status()
                ig_positions = r.json()["positions"]

            ig_deal_ids = {pos["position"]["dealId"] for pos in ig_positions}
            print(f"📊 IG has {len(ig_deal_ids)} active deal_ids")
            
            # Test the logic
            for trade in sample_trades:
                if trade.deal_id in ig_deal_ids:
                    print(f"✅ Trade {trade.id} ({trade.deal_id}) would be validated")
                else:
                    print(f"🔍 Trade {trade.id} ({trade.deal_id}) would be investigated with enhanced verification")
            
            print("✅ Test completed successfully")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

# Enhanced test function for the new verification system
async def test_enhanced_verification():
    """Test the enhanced verification system"""
    print("🧪 Testing enhanced verification system...")
    
    try:
        with SessionLocal() as db:
            # Get some sample trades that might be missing
            missing_trades = db.query(TradeLog).filter(
                TradeLog.status == "missing_on_ig"
            ).limit(3).all()
            
            if not missing_trades:
                print("📊 No trades with missing_on_ig status found for testing")
                return
            
            print(f"📊 Found {len(missing_trades)} trades with missing_on_ig status:")
            
            # Get trading headers
            trading_headers = await get_ig_auth_headers()
            status_manager = EnhancedTradeStatusManager(trading_headers)
            
            for trade in missing_trades:
                print(f"\n🔍 Testing trade {trade.id}:")
                print(f"   Deal ID: {trade.deal_id}")
                print(f"   Deal Reference: {trade.deal_reference}")
                print(f"   Symbol: {trade.symbol}")
                print(f"   Current Status: {trade.status}")
                
                # Test the enhanced verification
                try:
                    final_status = await status_manager.verify_and_update_trade_status(trade, db)
                    print(f"   ✅ Enhanced verification result: {final_status}")
                    print(f"   Exit reason: {getattr(trade, 'exit_reason', 'Not set')}")
                except Exception as e:
                    print(f"   ❌ Verification failed: {e}")
            
            print("\n✅ Enhanced verification test completed")
            
    except Exception as e:
        print(f"❌ Enhanced verification test failed: {e}")

async def test_missing_trade_recovery():
    """Test the batch recovery of missing trades"""
    try:
        with SessionLocal() as db:
            # Get trading headers
            trading_headers = await get_ig_auth_headers()
            status_manager = EnhancedTradeStatusManager(trading_headers)
            
            print("🔍 Running missing trade recovery check...")
            stats = await status_manager.verify_missing_trades(db)
            
            print("\n📊 Recovery Results:")
            print(f"   Trades Checked: {stats['checked']}")
            print(f"   Trades Recovered: {stats['recovered']}")
            print(f"   Still Missing: {stats['still_missing']}")
            print(f"   Errors: {stats['errors']}")
            
    except Exception as e:
        print(f"❌ Recovery test failed: {e}")

if __name__ == "__main__":
    # Run both tests
    print("Running basic sync logic test...")
    asyncio.run(test_sync_logic())
    
    print("\nRunning enhanced verification test...")
    asyncio.run(test_enhanced_verification())
    
    print("\nRunning missing trade recovery test...")
    asyncio.run(test_missing_trade_recovery())