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

    def _parse_ig_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse IG API date string to datetime object.
        IG uses formats: '2025-12-29T09:48:15' or '2025/12/29 09:48:15' or '29/12/25'
        """
        if not date_str:
            return None

        try:
            # Try ISO format first (most common from IG)
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
            # Try slash format with time
            elif '/' in date_str and ':' in date_str:
                return datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
            # Try DD/MM/YY format (common in IG activity)
            elif '/' in date_str and len(date_str) <= 10:
                return datetime.strptime(date_str, "%d/%m/%y")
            # Try dash format without T
            elif '-' in date_str and ':' in date_str:
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            # Try dash date only
            elif '-' in date_str:
                return datetime.strptime(date_str, "%Y-%m-%d")
            return None
        except (ValueError, TypeError) as e:
            self.logger.debug(f"⚠️ Could not parse IG date '{date_str}': {e}")
            return None
    
    async def verify_and_update_trade_status(self, trade: TradeLog, db: Session) -> str:
        """
        Comprehensive trade verification using stored deal_id and deal_reference
        Returns the determined status with full reasoning

        IMPORTANT: We must check ALL sources before marking a trade as expired/invalid.
        The deal confirmation API returning NOT_FOUND does NOT mean the position is closed -
        it could be a timing issue or API quirk. The position endpoint is the source of truth.
        """
        try:
            self.logger.info(f"🔍 [VERIFICATION] Trade {trade.id} {trade.symbol} - Starting comprehensive verification")

            # Step 1: Check if position still exists by deal_id
            position_data = await self._check_position_exists(trade.deal_id)

            if position_data:
                self.logger.info(f"✅ [ACTIVE] Trade {trade.id} position still active on IG")
                return self._update_trade_status(trade, "tracking", "position_verified_active", db, position_data)

            # Step 2: CRITICAL - Check ALL positions before investigating deal outcome
            # This is the most reliable check - positions endpoint is the source of truth
            self.logger.info(f"🔄 [ALL POSITIONS CHECK] Trade {trade.id} - Checking ALL open positions")
            all_positions_check = await self._comprehensive_position_check(trade)

            if all_positions_check:
                self.logger.info(f"✅ [FOUND IN ALL POSITIONS] Trade {trade.id} found in comprehensive position check!")
                return self._update_trade_status(trade, "tracking", "found_in_comprehensive_check", db)

            # Step 3: Check transaction history BEFORE deal outcome
            # If trade was closed, it will appear in transaction history
            self.logger.info(f"📋 [TRANSACTION CHECK] Trade {trade.id} - Checking transaction history")
            transaction_result = await self._check_transaction_history(trade.deal_id, trade.deal_reference)

            if transaction_result and transaction_result.get("found"):
                self.logger.info(f"📋 [TRANSACTION FOUND] Trade {trade.id} found in transaction history")
                return self._handle_transaction_result(trade, transaction_result, db)

            # Step 3b: Check activity history for close events
            # The close activity has a DIFFERENT deal_id, so we need to search by position reference
            self.logger.info(f"📋 [ACTIVITY CHECK] Trade {trade.id} - Checking activity history for close events")
            activity_result = await self._check_activity_for_close(trade)

            if activity_result and activity_result.get("closed"):
                self.logger.info(f"📋 [ACTIVITY CLOSE FOUND] Trade {trade.id} was closed via activity: {activity_result.get('description')}")
                return self._update_trade_status(
                    trade, "closed",
                    f"closed_via_activity_{activity_result.get('close_deal_id', 'unknown')}",
                    db,
                    activity_close_data=activity_result
                )

            # Step 4: Now check deal confirmation outcome
            # Only use this for ACCEPTED/REJECTED status, NOT for NOT_FOUND
            self.logger.info(f"🔎 [DEAL OUTCOME] Trade {trade.id} - Investigating deal confirmation")
            deal_outcome = await self._investigate_deal_outcome(trade.deal_reference)

            if deal_outcome:
                status = deal_outcome.get("status", "").upper()
                # Only act on definitive statuses (ACCEPTED with no position = closed, REJECTED = rejected)
                # Do NOT act on NOT_FOUND - this is unreliable and caused trade 1524 to lose 58 pips
                if status == "ACCEPTED":
                    # Deal was accepted but not in positions and not in transactions = likely just closed
                    self.logger.info(f"📊 [DEAL ACCEPTED] Trade {trade.id} deal accepted but position not found - likely closed")
                    return self._update_trade_status(trade, "closed", "deal_accepted_position_closed", db)
                elif status == "REJECTED":
                    reason = deal_outcome.get("reason", "Unknown rejection reason")
                    self.logger.warning(f"❌ [DEAL REJECTED] Trade {trade.id} deal was rejected: {reason}")
                    return self._update_trade_status(trade, "rejected", f"deal_rejected: {reason}", db)
                elif status == "NOT_FOUND":
                    # NOT_FOUND is unreliable - the deal confirmation API is not the source of truth
                    # Log a warning but DO NOT mark as expired - continue to allow trailing
                    self.logger.warning(f"⚠️ [DEAL NOT_FOUND] Trade {trade.id} deal confirmation returned NOT_FOUND - this is UNRELIABLE, continuing to track")
                    # Return tracking status to allow trailing stop to continue working
                    return self._update_trade_status(trade, "tracking", "deal_not_found_but_continuing", db)

            # Step 5: Final fallback - check if deal reference format is valid
            deal_validity = await self._verify_deal_validity(trade.deal_reference)

            if not deal_validity.get("valid"):
                reason = deal_validity.get("reason", "Invalid deal")
                self.logger.error(f"❌ [INVALID DEAL] Trade {trade.id}: {reason}")
                return self._update_trade_status(trade, "invalid_deal", reason, db)

            # Step 6: If we get here, we have a valid deal but can't find it anywhere
            # This is genuinely problematic - but still give benefit of the doubt for first occurrence
            # Track how many times we've seen this trade as "uncertain"
            uncertain_count = getattr(trade, '_verification_uncertain_count', 0) + 1
            trade._verification_uncertain_count = uncertain_count

            if uncertain_count < 3:
                self.logger.warning(f"⚠️ [UNCERTAIN] Trade {trade.id}: Cannot verify status (attempt {uncertain_count}/3) - continuing to track")
                return self._update_trade_status(trade, "tracking", f"uncertain_status_attempt_{uncertain_count}", db)
            else:
                self.logger.error(f"❌ [MISSING] Trade {trade.id}: Failed to verify after 3 attempts - marking as missing")
                return self._update_trade_status(trade, "missing_on_ig", "failed_verification_3_attempts", db)

        except Exception as e:
            self.logger.error(f"❌ [VERIFICATION ERROR] Trade {trade.id}: {e}")
            # On error, DON'T block processing - continue to track the trade
            self.logger.warning(f"⚠️ [VERIFICATION ERROR] Trade {trade.id}: Continuing to track despite error")
            return "tracking"
    
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
                TradeLog.endpoint.in_(["dev", "dev-limit"])  # Include limit orders
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
    
    async def _check_position_exists(self, deal_id: str) -> Optional[Dict]:
        """Check if position still exists in open positions and return position data"""
        try:
            # Use cached positions if available
            positions = await self._get_all_positions()
            for pos in positions:
                if pos["position"]["dealId"] == deal_id:
                    return pos  # Return the full position data
            return None  # Position not found
                
        except Exception as e:
            self.logger.error(f"❌ [POSITION CHECK ERROR] {deal_id}: {e}")
            return None
    
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
            
            async with httpx.AsyncClient(timeout=30.0) as client:
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

            async with httpx.AsyncClient(timeout=30.0) as client:
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

    async def _check_activity_for_close(self, trade: TradeLog) -> Optional[Dict]:
        """
        Check activity history for close events using position reference.

        The close activity has a DIFFERENT deal_id than the open, so we need to:
        1. Extract position reference from the original deal_id (last 8 chars after DIAAAAV3)
        2. Search activities for "stängd" (closed) descriptions containing that reference
        """
        try:
            # Extract position reference from deal_id
            # Format: DIAAAAV3TABV7AF -> position ref is TABV7AF (or 3TABV7AF)
            if not trade.deal_id or len(trade.deal_id) < 10:
                return None

            # The position reference is typically the last 7-8 characters
            position_ref = trade.deal_id[-7:]  # e.g., "TABV7AF" from "DIAAAAV3TABV7AF"

            # Get activities from last 2 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=2)

            url = f"{API_BASE_URL}/history/activity"
            headers = {
                "X-IG-API-KEY": self.trading_headers["X-IG-API-KEY"],
                "CST": self.trading_headers["CST"],
                "X-SECURITY-TOKEN": self.trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "3"
            }

            params = {
                "from": start_date.strftime("%Y-%m-%dT00:00:00"),
                "to": end_date.strftime("%Y-%m-%dT23:59:59"),
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                activities = response.json().get("activities", [])

                # Look for close activities containing our position reference
                for act in activities:
                    description = act.get("description", "")
                    epic = act.get("epic", "")

                    # Check if this is a close activity for our position
                    # Swedish: "Position(er) stängd(a): 3TABV7AF"
                    # English: "Position(s) closed: 3TABV7AF"
                    if ("stängd" in description.lower() or "closed" in description.lower()):
                        # Check if our position reference is in the description
                        if position_ref in description or f"3{position_ref}" in description:
                            # Verify it's the same symbol
                            if epic == trade.symbol:
                                self.logger.info(f"📋 [ACTIVITY CLOSE] Found close activity for {trade.deal_id}: {description}")
                                return {
                                    "closed": True,
                                    "close_deal_id": act.get("dealId"),
                                    "close_date": act.get("date"),
                                    "description": description,
                                    "epic": epic
                                }

                self.logger.debug(f"📋 [NO ACTIVITY CLOSE] No close activity found for position ref {position_ref}")
                return {"closed": False}

        except Exception as e:
            self.logger.error(f"❌ [ACTIVITY CHECK ERROR] Trade {trade.id}: {e}")
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
        """
        Handle the result from deal confirmation

        IMPORTANT: NOT_FOUND status is UNRELIABLE and should NOT mark trades as expired.
        Trade 1524 was lost because NOT_FOUND caused premature expiration while the position
        was still active and went to +58 pips before reversing.
        """
        status = deal_outcome.get("status", "").upper()

        if status == "ACCEPTED":
            # Deal was accepted but position not found - likely closed
            return self._update_trade_status(trade, "closed", "deal_accepted_but_position_closed", db)

        elif status == "REJECTED":
            reason = deal_outcome.get("reason", "Unknown rejection reason")
            return self._update_trade_status(trade, "rejected", f"deal_rejected: {reason}", db)

        elif status == "NOT_FOUND":
            # CRITICAL FIX: Do NOT mark as expired - this is unreliable
            # The deal confirmation API returning NOT_FOUND does NOT mean the position is closed
            # Continue tracking to allow trailing stop to work
            self.logger.warning(f"⚠️ [DEAL NOT_FOUND] Trade {trade.id}: Deal confirmation returned NOT_FOUND - continuing to track (NOT expiring)")
            return self._update_trade_status(trade, "tracking", "deal_not_found_continuing", db)

        else:
            # For unknown statuses, also continue tracking rather than blocking
            self.logger.warning(f"⚠️ [UNKNOWN STATUS] Trade {trade.id}: Deal confirmation returned unknown status '{status}' - continuing to track")
            return self._update_trade_status(trade, "tracking", f"unknown_deal_status_{status}", db)
    
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
    
    def _update_trade_status(self, trade: TradeLog, status: str, exit_reason: str, db: Session, ig_position_data: dict = None, activity_close_data: dict = None) -> str:
        """Update trade status with detailed reasoning and sync position data"""
        old_status = trade.status
        trade.status = status
        trade.exit_reason = exit_reason
        trade.trigger_time = datetime.utcnow()

        # NEW: Sync position data from IG if provided
        # CRITICAL FIX (Jan 2026): Do NOT sync SL from IG when trailing is active
        # The trailing system is the source of truth - syncing from IG's cached data
        # was overwriting valid BE/trailing stops with stale values
        if ig_position_data and status in ["tracking", "break_even", "trailing"]:
            position = ig_position_data.get("position", {})

            # SKIP SL sync if trailing is active (moved_to_breakeven or early_be_executed)
            # The trailing system manages the stop - don't overwrite with stale IG data
            trailing_active = getattr(trade, 'moved_to_breakeven', False) or getattr(trade, 'early_be_executed', False)

            if trailing_active:
                self.logger.debug(f"🔒 [SYNC SKIP] Trade {trade.id}: Trailing active, skipping SL/TP sync from IG")
            else:
                # Only sync SL/TP for trades NOT being managed by trailing system
                ig_stop_level = position.get("stopLevel")
                if ig_stop_level and abs(float(ig_stop_level) - (trade.sl_price or 0)) > 0.0001:
                    old_sl = trade.sl_price
                    trade.sl_price = float(ig_stop_level)
                    self.logger.info(f"🔄 [SYNC SL] Trade {trade.id}: SL {old_sl} → {trade.sl_price}")

                # Update take profit if different
                ig_limit_level = position.get("limitLevel")
                if ig_limit_level and abs(float(ig_limit_level) - (trade.tp_price or 0)) > 0.0001:
                    old_tp = trade.tp_price
                    trade.tp_price = float(ig_limit_level)
                    self.logger.info(f"🔄 [SYNC TP] Trade {trade.id}: TP {old_tp} → {trade.tp_price}")

        # Store activity close data for P/L correlation
        if activity_close_data and status == "closed":
            close_deal_id = activity_close_data.get("close_deal_id")
            if close_deal_id:
                trade.activity_close_deal_id = close_deal_id
                self.logger.info(f"📋 [CLOSE DEAL ID] Trade {trade.id}: activity_close_deal_id = {close_deal_id}")

            # Extract and store position reference from deal_id
            if trade.deal_id and len(trade.deal_id) >= 8:
                position_ref = trade.deal_id[-8:]  # e.g., "3TABV7AF"
                trade.position_reference = position_ref
                self.logger.info(f"📋 [POSITION REF] Trade {trade.id}: position_reference = {position_ref}")

            # Set closed_at timestamp using actual IG close time if available
            close_date_str = activity_close_data.get("close_date")
            if close_date_str:
                actual_close_time = self._parse_ig_date(close_date_str)
                if actual_close_time:
                    trade.closed_at = actual_close_time
                    self.logger.info(f"🕐 [CLOSE TIME] Trade {trade.id}: Using IG close time {actual_close_time}")
                else:
                    trade.closed_at = datetime.utcnow()
                    self.logger.info(f"🕐 [CLOSE TIME] Trade {trade.id}: Could not parse IG date, using current time")
            else:
                trade.closed_at = datetime.utcnow()
                self.logger.info(f"🕐 [CLOSE TIME] Trade {trade.id}: No IG close date, using current time")

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
                TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "ema_exit_pending", "profit_protected", "partial_closed"]),
                TradeLog.deal_id.isnot(None),  # Only trades with deal_ids
                TradeLog.endpoint.in_(["dev", "dev-limit"])  # Include limit orders that have filled
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

                    # Auto-correlate P/L for newly closed trades
                    if closed > 0:
                        try:
                            from services.trade_pnl_correlator import TradePnLCorrelator
                            logger.info(f"[P/L CORRELATION] Running P/L correlator for {closed} newly closed trades...")
                            pnl_correlator = TradePnLCorrelator(db_session=db)
                            pnl_result = await pnl_correlator.correlate_and_update_pnl(trading_headers, days_back=7)
                            updated_count = pnl_result.get("summary", {}).get("updated_count", 0)
                            if updated_count > 0:
                                logger.info(f"[✅ P/L UPDATED] Updated P/L for {updated_count} trades")
                        except Exception as pnl_error:
                            logger.error(f"[P/L CORRELATION ERROR] {pnl_error}")
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

            async with httpx.AsyncClient(timeout=30.0) as client:
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