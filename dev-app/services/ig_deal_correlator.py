# services/ig_deal_correlator.py
"""
IG Deal ID Correlator - ENHANCED Deal ID correlation system
Fetches activity data and correlates deal IDs with transactions and trade_log
Updates trade_log with actual P&L from broker transactions
"""

import logging
import json
import httpx
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text

from services.broker_transaction_analyzer import BrokerTransactionAnalyzer, BrokerTransaction
from services.models import TradeLog
from services.db import get_db


@dataclass
class IGActivity:
    """Data class for IG activity"""
    date: str
    epic: str
    period: str
    deal_id: str
    channel: str
    type: str
    status: str
    description: str
    details: Optional[str] = None

    # Derived fields
    position_reference: Optional[str] = None
    action: Optional[str] = None  # 'OPENED' or 'CLOSED'
    date_utc: Optional[datetime] = None  # Parsed UTC datetime for accurate closed_at


@dataclass
class DealCorrelation:
    """Data class for deal correlation results"""
    deal_id: str
    transaction_reference: Optional[str] = None
    trade_log_id: Optional[int] = None
    profit_loss: Optional[float] = None
    correlation_status: str = "pending"  # 'matched', 'partial', 'missing'
    activity_found: bool = False
    transaction_found: bool = False
    trade_log_found: bool = False
    close_time: Optional[datetime] = None  # Actual close time from IG


class IGDealCorrelator:
    """
    Enhanced IG Deal ID correlation system
    Fetches activity data and correlates with transactions and trade_log
    """
    
    def __init__(self, db_session: Session = None, logger=None):
        self.db_session = db_session
        self.logger = logger or logging.getLogger(__name__)
        self.transaction_analyzer = BrokerTransactionAnalyzer(db_manager=db_session, logger=logger)
        
        # IG API configuration
        from config import API_BASE_URL
        self.ig_api_base = API_BASE_URL
        
        self.logger.info("🔗 IGDealCorrelator initialized for deal ID correlation")

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

    async def _fetch_ig_transactions(self, trading_headers: dict, days_back: int) -> dict:
        """Fetch transactions using the same method as existing endpoint"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp_ms = int(start_time.timestamp() * 1000)
            
            ig_url = f"{self.ig_api_base}/history/transactions/ALL/{start_timestamp_ms}"
            
            self.logger.info(f"🔍 Fetching IG transactions from {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"📡 IG Transaction API URL: {ig_url}")
            
            ig_headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "1"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(ig_url, headers=ig_headers)
                response.raise_for_status()
                
                ig_data = response.json()
                transactions_list = ig_data.get('transactions', [])
                
                self.logger.info(f"📥 Fetched {len(transactions_list)} transactions from IG API")
                
                return ig_data
                    
        except Exception as e:
            self.logger.error(f"❌ Error fetching transactions: {e}")
            return {"transactions": []}

    async def _fetch_ig_activities(self, trading_headers: dict, days_back: int = 7) -> List[IGActivity]:
        """
        Fetch activity data from IG API
        """
        try:
            # Calculate timestamp for the lookback period
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp_ms = int(start_time.timestamp() * 1000)
            
            # Construct IG API URL for activities
            activity_url = f"{self.ig_api_base}/history/activity/{start_timestamp_ms}"
            
            self.logger.info(f"🔍 Fetching IG activities from {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"📡 IG Activity API URL: {activity_url}")
            
            # Prepare headers
            ig_headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "3"  # Use version 3 for activity endpoint
            }
            
            # Fetch from IG API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(activity_url, headers=ig_headers)
                response.raise_for_status()
                activity_data = response.json()
            
            # Parse activities
            activities = self._parse_ig_activities(activity_data)
            
            self.logger.info(f"✅ Fetched {len(activities)} activities from IG API")
            return activities
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching IG activities: {e}")
            return []
    
    def _parse_ig_activities(self, activity_data: dict) -> List[IGActivity]:
        """Parse IG activity JSON data into structured objects"""
        try:
            activities = []
            raw_activities = activity_data.get('activities', [])
            
            self.logger.info(f"🔍 Processing {len(raw_activities)} raw activities from IG API")
            
            # Debug: Log sample activity structure
            if raw_activities:
                sample_activity = raw_activities[0]
                available_fields = list(sample_activity.keys())
                self.logger.info(f"📋 Available IG activity fields: {available_fields}")
                self.logger.debug(f"📋 Sample activity: {sample_activity}")
            
            for i, activity in enumerate(raw_activities):
                try:
                    # Extract position reference from description
                    description = activity.get('description', '')
                    position_reference = self._extract_position_reference(description)

                    # Determine action from description
                    action = self._determine_action(description)

                    # Parse date for accurate timestamp
                    date_utc = self._parse_ig_date(activity.get('date', ''))

                    ig_activity = IGActivity(
                        date=activity.get('date', ''),
                        epic=activity.get('epic', ''),
                        period=activity.get('period', ''),
                        deal_id=activity.get('dealId', ''),
                        channel=activity.get('channel', ''),
                        type=activity.get('type', ''),
                        status=activity.get('status', ''),
                        description=description,
                        details=activity.get('details'),
                        position_reference=position_reference,
                        action=action,
                        date_utc=date_utc
                    )

                    activities.append(ig_activity)

                    self.logger.debug(f"✅ Parsed activity: {ig_activity.deal_id} - {ig_activity.action} @ {date_utc}")
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Skipping activity {i+1} (parsing error): {e}")
                    continue
            
            return activities
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing IG activities: {e}")
            return []
    
    def _extract_position_reference(self, description: str) -> Optional[str]:
        """
        Extract position reference from Swedish IG description
        Example: "Position(er) stängd(a): HG75C5AE" -> "HG75C5AE"
        """
        try:
            if ':' in description:
                # Split by colon and take the last part, strip whitespace
                reference = description.split(':')[-1].strip()
                if reference and len(reference) > 4:  # Basic validation
                    return reference
            return None
        except:
            return None
    
    def _determine_action(self, description: str) -> Optional[str]:
        """
        Determine action from description
        """
        description_lower = description.lower()
        if 'stängd' in description_lower or 'closed' in description_lower:
            return 'CLOSED'
        elif 'öppnad' in description_lower or 'opened' in description_lower:
            return 'OPENED'
        return None
    
    async def correlate_deals_with_transactions_and_trades(
        self, 
        trading_headers: dict, 
        days_back: int = 7
    ) -> Dict[str, any]:
        """
        Complete correlation workflow:
        1. Fetch activities from IG
        2. Fetch transactions from IG  
        3. Correlate deal IDs
        4. Update trade_log with P&L
        """
        try:
            self.logger.info(f"🔗 Starting comprehensive deal correlation for last {days_back} days")
            
            # Step 1: Fetch activities
            activities = await self.fetch_ig_activities(trading_headers, days_back)
            
            # Step 2: Fetch transactions using existing analyzer
            transactions_data = await self._fetch_ig_transactions(trading_headers, days_back)
            transactions = self.transaction_analyzer.parse_broker_transactions(transactions_data)
            
            # Step 3: Perform correlation
            correlations = self._correlate_deals_fuzzy(activities, transactions)

            
            # Step 4: Update trade_log with P&L
            updated_trades = await self._update_trade_log_with_pnl(correlations)
            
            # Step 5: Generate summary report
            summary = self._generate_correlation_summary(correlations, updated_trades)
            
            self.logger.info(f"✅ Correlation completed: {summary['total_correlations']} deals processed")
            
            return {
                "status": "success",
                "summary": summary,
                "correlations": correlations,
                "updated_trades": updated_trades,
                "activities_found": len(activities),
                "transactions_found": len(transactions)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in deal correlation workflow: {e}")
            return {
                "status": "error",
                "error": str(e),
                "correlations": [],
                "summary": {}
            }
    
    def _extract_position_reference(self, description: str) -> Optional[str]:
        """
        Extract position reference from Swedish IG description  
        Example: "Position(er) stängd(a): HGWD6UAS" -> "HGWD6UAS"
        """
        try:
            if ':' in description:
                # Split by colon and take the last part, strip whitespace
                reference = description.split(':')[-1].strip()
                if reference and len(reference) > 4:  # Basic validation
                    self.logger.debug(f"✅ Extracted position reference: {reference}")
                    return reference
            
            # Try alternative patterns if colon method fails
            import re
            
            # Pattern for Swedish: "Position(er) stängd(a): REFERENCE"
            pattern = r'Position\(er\)\s+stängd\(a\):\s*([A-Z0-9]+)'
            match = re.search(pattern, description)
            if match:
                reference = match.group(1).strip()
                self.logger.debug(f"✅ Extracted position reference via regex: {reference}")
                return reference
            
            # Pattern for English: "Position closed: REFERENCE" 
            pattern = r'Position\s+closed:\s*([A-Z0-9]+)'
            match = re.search(pattern, description)
            if match:
                reference = match.group(1).strip()
                self.logger.debug(f"✅ Extracted position reference via English regex: {reference}")
                return reference
            
            self.logger.debug(f"⚠️ Could not extract reference from: {description}")
            return None
            
        except Exception as e:
            self.logger.debug(f"❌ Error extracting position reference: {e}")
            return None

    async def fetch_ig_activities(self, trading_headers: dict, days_back: int = 7) -> List[IGActivity]:
        """
        Fetch activity data from IG API using correct URL parameters (from/to)
        """
        try:
            # Calculate date range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Format dates for IG API (ISO format: YYYY-MM-DDTHH:MM:SS)
            from_date = start_time.strftime('%Y-%m-%dT00:00:00')
            to_date = end_time.strftime('%Y-%m-%dT23:59:59')
            
            # Construct IG API URL with parameters
            activity_url = f"{self.ig_api_base}/history/activity"
            
            self.logger.info(f"🔍 Fetching IG activities from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
            self.logger.info(f"📡 IG Activity API URL: {activity_url}")
            self.logger.info(f"📅 Date range: from={from_date}, to={to_date}")
            
            # Prepare headers
            ig_headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "3"  # Use version 3 as it's most common for activity
            }
            
            # Prepare URL parameters
            params = {
                "from": from_date,
                "to": to_date
            }
            
            # Fetch from IG API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(activity_url, headers=ig_headers, params=params)
                
                self.logger.info(f"📡 Full request URL: {response.url}")
                self.logger.info(f"📊 Response status: {response.status_code}")
                
                if response.status_code == 200:
                    activity_data = response.json()
                    
                    # Log the raw response structure for debugging
                    self.logger.info(f"📋 Raw response keys: {list(activity_data.keys())}")
                    activities_list = activity_data.get('activities', [])
                    self.logger.info(f"📥 Raw activities fetched from IG API: {len(activities_list)}")
                    
                    # Parse activities
                    activities = self._parse_ig_activities(activity_data)
                    
                    self.logger.info(f"✅ Fetched {len(activities)} activities from IG API")
                    
                    # Log sample activities for debugging
                    if activities:
                        self.logger.info("📋 Sample activities found:")
                        for i, activity in enumerate(activities[:3]):
                            self.logger.info(f"   {i+1}. Deal: {activity.deal_id} | Epic: {activity.epic}")
                            self.logger.info(f"      Description: {activity.description}")
                            self.logger.info(f"      Position Ref: {activity.position_reference}")
                            self.logger.info(f"      Action: {activity.action}")
                    
                    return activities
                    
                elif response.status_code == 400:
                    self.logger.error(f"❌ Bad Request (400): {response.text}")
                    self.logger.error("💡 Check date format or parameters")
                    return []
                elif response.status_code == 401:
                    self.logger.error(f"❌ Unauthorized (401): Check IG authentication")
                    return []
                else:
                    self.logger.error(f"❌ IG Activity API error: {response.status_code} - {response.text}")
                    return []
                    
        except httpx.TimeoutException:
            self.logger.error("❌ IG Activity API request timeout")
            return []
        except Exception as e:
            self.logger.error(f"❌ Error fetching IG activities: {e}")
            return []
    
    def _correlate_deals(self, activities: List[IGActivity], transactions: List[BrokerTransaction]) -> List[DealCorrelation]:
        """
        Correlate activities with transactions and existing trade_log entries
        """
        correlations = []
        
        # Create lookup maps for efficiency
        activity_by_deal_id = {act.deal_id: act for act in activities if act.deal_id}
        transaction_by_deal_id = {tx.deal_id: tx for tx in transactions if tx.deal_id}
        transaction_by_reference = {tx.reference: tx for tx in transactions if tx.reference}
        
        # Get all deal IDs from both sources
        all_deal_ids = set()
        all_deal_ids.update(activity_by_deal_id.keys())
        all_deal_ids.update(transaction_by_deal_id.keys())
        
        self.logger.info(f"🔍 Correlating {len(all_deal_ids)} unique deal IDs")
        
        for deal_id in all_deal_ids:
            if not deal_id:
                continue
                
            correlation = DealCorrelation(deal_id=deal_id)

            # Check if we have activity data
            if deal_id in activity_by_deal_id:
                correlation.activity_found = True
                activity = activity_by_deal_id[deal_id]
                # Capture close time from activity
                correlation.close_time = activity.date_utc

                # Try to match with transaction by position reference
                if activity.position_reference:
                    if activity.position_reference in transaction_by_reference:
                        transaction = transaction_by_reference[activity.position_reference]
                        correlation.transaction_reference = activity.position_reference
                        correlation.profit_loss = transaction.profit_loss
                        correlation.transaction_found = True

            # Check if we have transaction data directly
            if deal_id in transaction_by_deal_id:
                transaction = transaction_by_deal_id[deal_id]
                correlation.transaction_found = True
                correlation.transaction_reference = transaction.reference
                correlation.profit_loss = transaction.profit_loss

            # Check if we have trade_log entry
            trade_log_entry = self._find_trade_log_by_deal_id(deal_id)
            if trade_log_entry:
                correlation.trade_log_found = True
                correlation.trade_log_id = trade_log_entry.id

            # Determine correlation status
            if correlation.activity_found and correlation.transaction_found and correlation.trade_log_found:
                correlation.correlation_status = "matched"
            elif correlation.transaction_found and correlation.trade_log_found:
                correlation.correlation_status = "partial"
            else:
                correlation.correlation_status = "missing"

            correlations.append(correlation)
            
            # Log correlation details
            status_emoji = "✅" if correlation.correlation_status == "matched" else "⚠️" if correlation.correlation_status == "partial" else "❌"
            self.logger.debug(f"{status_emoji} {deal_id}: Activity={correlation.activity_found}, Transaction={correlation.transaction_found}, TradeLog={correlation.trade_log_found}")
        
        return correlations
    
    def _find_trade_log_by_deal_id(self, deal_id: str) -> Optional[TradeLog]:
        """Find trade_log entry by deal_id"""
        try:
            if not self.db_session:
                return None
                
            trade_log = self.db_session.query(TradeLog).filter(
                TradeLog.deal_id == deal_id
            ).first()
            
            return trade_log
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error finding trade_log for deal_id {deal_id}: {e}")
            return None
    
    async def _update_trade_log_with_pnl(self, correlations: List[DealCorrelation]) -> List[Dict]:
        """
        Update trade_log entries with P&L from broker transactions
        """
        updated_trades = []
        
        for correlation in correlations:
            if (correlation.trade_log_found and 
                correlation.transaction_found and 
                correlation.profit_loss is not None):
                
                try:
                    # Find the trade_log entry
                    trade_log = self.db_session.query(TradeLog).filter(
                        TradeLog.deal_id == correlation.deal_id
                    ).first()
                    
                    if trade_log:
                        # Add P&L fields to trade_log if they don't exist
                        if not hasattr(trade_log, 'profit_loss'):
                            # Add new columns to trade_log table
                            self._add_pnl_columns_to_trade_log()

                        # Use actual close time from IG activity, fallback to current time
                        actual_close_time = correlation.close_time or datetime.utcnow()
                        current_utc = datetime.utcnow()
                        self.logger.info(f"🕐 Using IG close time: {actual_close_time.strftime('%Y-%m-%d %H:%M:%S')} UTC for deal {correlation.deal_id}")

                        # Update the trade with actual P&L
                        self.db_session.execute(
                            text("""
                                UPDATE trade_log
                                SET profit_loss = :pnl,
                                    pnl_currency = :currency,
                                    status = 'closed',
                                    closed_at = :closed_at,
                                    updated_at = :updated_at
                                WHERE deal_id = :deal_id
                            """),
                            {
                                "pnl": correlation.profit_loss,
                                "currency": "SEK",  # Assuming SEK based on your transaction data
                                "deal_id": correlation.deal_id,
                                "closed_at": actual_close_time,  # Use actual IG close time
                                "updated_at": current_utc
                            }
                        )
                        
                        updated_trades.append({
                            "trade_log_id": trade_log.id,
                            "deal_id": correlation.deal_id,
                            "symbol": trade_log.symbol,
                            "profit_loss": correlation.profit_loss,
                            "entry_price": trade_log.entry_price,
                            "direction": trade_log.direction
                        })
                        
                        self.logger.info(f"✅ Updated trade_log {trade_log.id} with P&L: {correlation.profit_loss}")
                
                except Exception as e:
                    self.logger.error(f"❌ Error updating trade_log for deal_id {correlation.deal_id}: {e}")
                    continue
        
        # Commit all updates
        try:
            self.db_session.commit()
            self.logger.info(f"✅ Committed {len(updated_trades)} trade_log updates")
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"❌ Error committing trade_log updates: {e}")
        
        return updated_trades
    
    def _add_pnl_columns_to_trade_log(self):
        """Add P&L tracking columns to trade_log table"""
        try:
            # Add new columns if they don't exist
            alter_queries = [
                "ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS profit_loss DECIMAL(12, 2)",
                "ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS pnl_currency VARCHAR(10) DEFAULT 'SEK'",
                "ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS pnl_updated_at TIMESTAMP"
            ]
            
            for query in alter_queries:
                self.db_session.execute(text(query))
            
            self.db_session.commit()
            self.logger.info("✅ Enhanced trade_log table with P&L columns")
            
        except Exception as e:
            self.logger.debug(f"Note: P&L columns may already exist: {e}")
            self.db_session.rollback()
    
    def _generate_correlation_summary(self, correlations: List[DealCorrelation], updated_trades: List[Dict]) -> Dict:
        """Generate summary statistics for correlation results"""
        
        total_correlations = len(correlations)
        matched_count = sum(1 for c in correlations if c.correlation_status == "matched")
        partial_count = sum(1 for c in correlations if c.correlation_status == "partial")
        missing_count = sum(1 for c in correlations if c.correlation_status == "missing")
        
        total_pnl = sum(trade.get('profit_loss', 0) for trade in updated_trades)
        winning_trades = sum(1 for trade in updated_trades if trade.get('profit_loss', 0) > 0)
        losing_trades = sum(1 for trade in updated_trades if trade.get('profit_loss', 0) < 0)
        
        return {
            "total_correlations": total_correlations,
            "matched_deals": matched_count,
            "partial_matches": partial_count,
            "missing_data": missing_count,
            "correlation_rate": round(matched_count / total_correlations * 100, 2) if total_correlations > 0 else 0,
            "trades_updated": len(updated_trades),
            "total_pnl": round(total_pnl, 2),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(winning_trades / len(updated_trades) * 100, 2) if updated_trades else 0
        }
    
    def _correlate_deals_fuzzy(self, activities: List[IGActivity], transactions: List[BrokerTransaction]) -> List[DealCorrelation]:
        """
        Enhanced correlation with fuzzy matching for references
        """
        correlations = []
        
        # Create lookup maps for efficiency
        activity_by_deal_id = {act.deal_id: act for act in activities if act.deal_id}
        transaction_by_deal_id = {tx.deal_id: tx for tx in transactions if tx.deal_id}
        transaction_by_reference = {tx.reference: tx for tx in transactions if tx.reference}
        
        # Get all deal IDs from both sources
        all_deal_ids = set()
        all_deal_ids.update(activity_by_deal_id.keys())
        all_deal_ids.update(transaction_by_deal_id.keys())
        
        self.logger.info(f"🔍 Correlating {len(all_deal_ids)} unique deal IDs with fuzzy matching")
        
        for deal_id in all_deal_ids:
            if not deal_id:
                continue
                
            correlation = DealCorrelation(deal_id=deal_id)

            # Check if we have activity data
            if deal_id in activity_by_deal_id:
                correlation.activity_found = True
                activity = activity_by_deal_id[deal_id]
                # Capture close time from activity
                correlation.close_time = activity.date_utc

                # Try exact match first
                if activity.position_reference and activity.position_reference in transaction_by_reference:
                    transaction = transaction_by_reference[activity.position_reference]
                    correlation.transaction_reference = activity.position_reference
                    correlation.profit_loss = transaction.profit_loss
                    correlation.transaction_found = True
                    self.logger.debug(f"✅ Exact match: {activity.position_reference}")

                # Try fuzzy matching if exact match failed
                elif activity.position_reference:
                    fuzzy_match = self._find_fuzzy_reference_match(activity.position_reference, transactions)
                    if fuzzy_match:
                        correlation.transaction_reference = fuzzy_match.reference
                        correlation.profit_loss = fuzzy_match.profit_loss
                        correlation.transaction_found = True
                        self.logger.info(f"🎯 Fuzzy match: {activity.position_reference} → {fuzzy_match.reference}")

            # Check if we have transaction data directly
            if deal_id in transaction_by_deal_id:
                transaction = transaction_by_deal_id[deal_id]
                correlation.transaction_found = True
                correlation.transaction_reference = transaction.reference
                correlation.profit_loss = transaction.profit_loss

            # Check if we have trade_log entry
            trade_log_entry = self._find_trade_log_by_deal_id(deal_id)
            if trade_log_entry:
                correlation.trade_log_found = True
                correlation.trade_log_id = trade_log_entry.id

            # Determine correlation status
            if correlation.activity_found and correlation.transaction_found and correlation.trade_log_found:
                correlation.correlation_status = "matched"
            elif correlation.transaction_found and correlation.trade_log_found:
                correlation.correlation_status = "partial"
            else:
                correlation.correlation_status = "missing"
            
            correlations.append(correlation)
            
            # Log correlation details
            status_emoji = "✅" if correlation.correlation_status == "matched" else "⚠️" if correlation.correlation_status == "partial" else "❌"
            self.logger.debug(f"{status_emoji} {deal_id}: Activity={correlation.activity_found}, Transaction={correlation.transaction_found}, TradeLog={correlation.trade_log_found}")
        
        return correlations

    def _find_fuzzy_reference_match(self, activity_ref: str, transactions: List[BrokerTransaction]):
        """
        Find fuzzy matches for activity references in transaction data
        """
        if not activity_ref:
            return None
        
        # Try different fuzzy matching strategies
        
        # Strategy 1: Same prefix (first 2-3 chars)
        prefix = activity_ref[:3] if len(activity_ref) >= 3 else activity_ref[:2]
        for tx in transactions:
            if tx.reference and tx.reference.startswith(prefix):
                self.logger.debug(f"🎯 Prefix match: {activity_ref} → {tx.reference}")
                return tx
        
        # Strategy 2: Contains substring (last 4-6 chars)
        if len(activity_ref) >= 6:
            suffix = activity_ref[-6:]
            for tx in transactions:
                if tx.reference and suffix in tx.reference:
                    self.logger.debug(f"🎯 Suffix match: {activity_ref} ({suffix}) → {tx.reference}")
                    return tx
        
        # Strategy 3: Edit distance (allow 1-2 character differences)
        if len(activity_ref) >= 6:
            for tx in transactions:
                if tx.reference and len(tx.reference) == len(activity_ref):
                    # Simple edit distance check
                    differences = sum(c1 != c2 for c1, c2 in zip(activity_ref, tx.reference))
                    if differences <= 2:  # Allow up to 2 character differences
                        self.logger.debug(f"🎯 Edit distance match: {activity_ref} → {tx.reference} (diff: {differences})")
                        return tx
        
        # Strategy 4: Alphanumeric core match (remove letters, compare numbers or vice versa)
        activity_nums = ''.join(c for c in activity_ref if c.isdigit())
        activity_chars = ''.join(c for c in activity_ref if c.isalpha())
        
        if len(activity_nums) >= 3:  # If we have enough numbers to match on
            for tx in transactions:
                if tx.reference:
                    tx_nums = ''.join(c for c in tx.reference if c.isdigit())
                    if activity_nums == tx_nums:
                        self.logger.debug(f"🎯 Number match: {activity_ref} ({activity_nums}) → {tx.reference} ({tx_nums})")
                        return tx
        
        return None


# Factory function for easy integration
def create_deal_correlator(db_session: Session = None, logger=None) -> IGDealCorrelator:
    """Factory function to create IGDealCorrelator"""
    return IGDealCorrelator(db_session=db_session, logger=logger)


# Standalone function for direct use
async def correlate_ig_deals(trading_headers: dict, days_back: int = 7, db_session: Session = None) -> Dict:
    """Standalone function to perform complete deal correlation"""
    correlator = IGDealCorrelator(db_session=db_session)
    return await correlator.correlate_deals_with_transactions_and_trades(trading_headers, days_back)


if __name__ == "__main__":
    # Test the IGDealCorrelator
    print("🧪 Testing IG Deal ID Correlator...")
    
    # This would require actual IG credentials and database connection
    # For testing, you can run this with your actual setup
    
    print("✅ IG Deal ID Correlator ready for integration!")
    print("🔗 Features:")
    print("   - Fetches activity data from IG API")
    print("   - Correlates deal IDs across activities, transactions, and trade_log")
    print("   - Updates trade_log with actual P&L from broker")
    print("   - Handles Swedish IG descriptions")
    print("   - Provides comprehensive correlation reporting")