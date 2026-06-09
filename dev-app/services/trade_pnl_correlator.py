# services/trade_pnl_correlator.py - FIXED to use activity_close_deal_id
"""
Trade P&L Correlation Service - FIXED VERSION
FIXED: Now uses activity_close_deal_id instead of deal_id (open deal ID)
Uses transaction references directly from /history/transactions endpoint
Matches close_deal_ids (DIAAAAUXXXXXXX) with transaction references (XXXXXXX)
"""

import logging
import httpx
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text

# Import existing models and database utilities
try:
    from services.models import TradeLog
    from services.db import get_db_session
except ImportError:
    # Alternative import paths for different project structures
    try:
        from .models import TradeLog
        from .db import get_db_session
    except ImportError:
        print("⚠️ Warning: Import paths may need adjustment for your project structure")


@dataclass
class TransactionMatch:
    """Represents a matched transaction from IG API"""
    reference: str
    profit_loss: float
    currency: str
    instrument_name: str
    transaction_date: str
    transaction_type: str
    open_level: Optional[float] = None
    close_level: Optional[float] = None
    position_size: Optional[float] = None
    date_utc: Optional[datetime] = None  # Parsed UTC datetime for accurate closed_at


@dataclass
class TradeCorrelation:
    """Correlation between trade_log entry and transaction data"""
    trade_log_id: int
    close_deal_id: str
    extracted_reference: str
    symbol: str
    
    # Transaction data
    transaction_found: bool = False
    transaction_match: Optional[TransactionMatch] = None
    
    # Update status
    updated: bool = False
    error_message: Optional[str] = None


class TradePnLCorrelator:
    """
    Service to correlate trade_log entries with IG transaction data
    FIXED: Now uses activity_close_deal_id for proper correlation
    """
    
    def __init__(self, db_session: Session = None, logger=None):
        self.db_session = db_session
        self.logger = logger or logging.getLogger(__name__)
        
        # IG API configuration
        from config import API_BASE_URL
        self.api_base_url = API_BASE_URL  # Change to live API for production
        
    async def correlate_and_update_pnl(
        self, 
        trading_headers: dict, 
        days_back: int = 7,
        specific_deal_ids: List[str] = None
    ) -> Dict:
        """
        Main workflow to correlate trade_log with IG transactions and update P&L
        FIXED: Now uses activity_close_deal_id instead of deal_id
        """
        try:
            self.logger.info(f"🔄 Starting trade P&L correlation for last {days_back} days...")
            
            # Step 1: Get trade_log entries with close_deal_ids
            trade_entries = self._get_trade_entries_with_close_deal_ids(days_back, specific_deal_ids)
            self.logger.info(f"📋 Found {len(trade_entries)} trade entries to process")
            
            if not trade_entries:
                return {
                    "status": "success",
                    "message": "No trade entries found with close_deal_ids",
                    "total_processed": 0,
                    "updated_count": 0
                }
            
            # Step 2: Extract references and create correlations
            correlations = []
            for trade in trade_entries:
                correlation = self._create_correlation_from_trade(trade)
                correlations.append(correlation)
            
            # Step 3: Fetch transaction data from IG API
            transactions = await self._fetch_transactions_from_ig(trading_headers, days_back)
            self.logger.info(f"📊 Fetched {len(transactions)} transactions from IG API")
            
            # Step 4: Match transactions with trade entries
            self._match_transactions_to_trades(correlations, transactions)
            
            # Step 5: Update trade_log with P&L data
            updated_trades = await self._update_trade_log_with_pnl(correlations)
            
            # Step 6: Generate summary
            summary = self._generate_correlation_summary(correlations, updated_trades)
            
            self.logger.info(f"✅ Correlation completed: {summary['updated_count']}/{summary['total_processed']} trades updated")
            
            return {
                "status": "success",
                "message": "Trade P&L correlation completed successfully",
                "summary": summary,
                "correlations": self._serialize_correlations(correlations),
                "updated_trades": updated_trades
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in trade P&L correlation: {e}")
            import traceback
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_trade_entries_with_close_deal_ids(
        self, 
        days_back: int, 
        specific_deal_ids: List[str] = None
    ) -> List[TradeLog]:
        """
        Get trade_log entries that have close_deal_ids
        FIXED: Now queries activity_close_deal_id instead of deal_id
        """
        try:
            if not self.db_session:
                return []
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # 🔥 FIXED: Use activity_close_deal_id instead of deal_id
            query = self.db_session.query(TradeLog).filter(
                TradeLog.activity_close_deal_id.isnot(None),
                TradeLog.activity_close_deal_id != '',
                TradeLog.timestamp >= start_date
            )
            
            # Filter by specific deal IDs if provided
            if specific_deal_ids:
                query = query.filter(TradeLog.activity_close_deal_id.in_(specific_deal_ids))

            # Process trades that need P/L update (including those with status='expired')
            # Don't skip trades with existing P&L=0 as they might need recalculation
            query = query.filter(
                (TradeLog.profit_loss.is_(None)) |
                (TradeLog.profit_loss == 0) |
                (TradeLog.status == 'expired')  # Include expired trades for status fix
            )

            trades = query.all()
            
            self.logger.info(f"📋 Found {len(trades)} trade entries with close_deal_ids needing P&L update")
            return trades
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching trade entries: {e}")
            return []
    
    def _create_correlation_from_trade(self, trade: TradeLog) -> TradeCorrelation:
        """
        Create correlation object from trade_log entry
        FIXED: Now uses activity_close_deal_id
        """
        # 🔥 FIXED: Extract reference from activity_close_deal_id instead of deal_id
        close_deal_id = trade.activity_close_deal_id
        extracted_reference = self._extract_reference_from_deal_id(close_deal_id)
        
        return TradeCorrelation(
            trade_log_id=trade.id,
            close_deal_id=close_deal_id,
            extracted_reference=extracted_reference,
            symbol=trade.symbol
        )
    
    def _extract_reference_from_deal_id(self, close_deal_id: str) -> str:
        """
        Extract transaction reference from close_deal_id
        Format: DIAAAAVA4EZYRAQ -> A4EZYRAQ (last 8 characters)
        Format: DIAAAAUQJJCCFAM -> QJJCCFAM (last 8 characters)

        IG deal IDs have variable-length prefixes (DIAAAAU, DIAAAAVA, etc.)
        but transaction references are always the last 8 characters
        """
        if not close_deal_id:
            return ""

        # Extract last 8 characters (standard IG reference length)
        if len(close_deal_id) >= 8:
            extracted = close_deal_id[-8:]
            self.logger.debug(f"🔍 Extracted reference: {close_deal_id} -> {extracted}")
            return extracted
        else:
            # If shorter than 8 chars, use the whole string
            self.logger.warning(f"⚠️ Unexpected close_deal_id length: {close_deal_id}")
            return close_deal_id
    
    async def _fetch_transactions_from_ig(
        self,
        trading_headers: dict,
        days_back: int
    ) -> List[TransactionMatch]:
        """Fetch transaction data from IG history/transactions endpoint using V2 API"""
        try:
            # Calculate date range for API
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)

            # Use V2 API with proper date filtering
            url = f"{self.api_base_url}/history/transactions"

            headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"  # V2 supports date filtering
            }

            # V2 API parameters with date range.
            # IG treats `to` as exclusive midnight, so closes on the current day
            # (date-only) get cut off. Push `to` to tomorrow so today is included.
            params = {
                "from": start_time.strftime("%Y-%m-%d"),
                "to": (end_time + timedelta(days=1)).strftime("%Y-%m-%d"),
                "maxSpanSeconds": (days_back + 1) * 86400,
                "pageSize": 500
            }

            self.logger.info(f"🌐 Fetching transactions from IG API: {params['from']} to {params['to']}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    transactions = self._parse_transaction_data(data)
                    self.logger.info(f"✅ Successfully fetched {len(transactions)} transactions")
                    return transactions
                else:
                    self.logger.error(f"❌ IG API error: {response.status_code} - {response.text}")
                    return []

        except Exception as e:
            self.logger.error(f"❌ Error fetching transactions from IG API: {e}")
            return []
    
    def _parse_transaction_data(self, api_data: dict) -> List[TransactionMatch]:
        """Parse IG API transaction data into TransactionMatch objects"""
        transactions = []
        
        try:
            raw_transactions = api_data.get('transactions', [])
            self.logger.info(f"🔍 Processing {len(raw_transactions)} raw transactions from IG API")
            
            # Log available fields for debugging
            if raw_transactions:
                sample_tx = raw_transactions[0]
                available_fields = list(sample_tx.keys())
                self.logger.info(f"📋 Available IG transaction fields: {available_fields}")
            
            for tx in raw_transactions:
                # Skip non-trading transactions (fees, interest, etc.)
                transaction_type = tx.get('transactionType', '')
                
                # Handle Swedish/international transaction types
                if transaction_type.lower() not in ['position', 'handel', 'trade']:
                    continue
                
                # Parse profit/loss amount (handle Swedish format like "SK6.91")
                profit_loss_str = tx.get('profitAndLoss', '0')
                profit_loss = self._parse_currency_amount(profit_loss_str)
                
                # Extract currency
                currency = tx.get('currency', 'SEK')
                
                # Parse dateUtc for accurate closed_at timestamp
                date_utc = self._parse_ig_date_utc(tx.get('dateUtc', ''))

                # Create transaction match object
                transaction = TransactionMatch(
                    reference=tx.get('reference', ''),
                    profit_loss=profit_loss,
                    currency=currency,
                    instrument_name=tx.get('instrumentName', ''),
                    transaction_date=tx.get('date', ''),
                    transaction_type=transaction_type,
                    open_level=self._safe_float(tx.get('openLevel')),
                    close_level=self._safe_float(tx.get('closeLevel')),
                    position_size=self._safe_float(tx.get('size')),
                    date_utc=date_utc
                )

                transactions.append(transaction)
                self.logger.debug(f"📊 Parsed transaction: {transaction.reference} -> {transaction.profit_loss} {transaction.currency} @ {date_utc}")
                
        except Exception as e:
            self.logger.error(f"❌ Error parsing transaction data: {e}")
        
        self.logger.info(f"✅ Parsed {len(transactions)} trading transactions (filtered out non-trading)")
        return transactions
    
    def _parse_currency_amount(self, amount_str: str) -> float:
        """Parse currency amount string (e.g., 'SK6.91', '-SK45.20') to float"""
        if not amount_str:
            return 0.0
        
        try:
            # Remove currency symbols and convert to float
            cleaned = amount_str
            # Remove common currency prefixes
            for prefix in ['SK', 'SEK', '$', '€', '£', 'USD', 'EUR', 'GBP']:
                cleaned = cleaned.replace(prefix, '')
            
            # Remove spaces and convert
            cleaned = cleaned.strip()
            return float(cleaned)
            
        except (ValueError, TypeError):
            self.logger.warning(f"⚠️ Could not parse currency amount: {amount_str}")
            return 0.0
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def _parse_ig_date_utc(self, date_str: str) -> Optional[datetime]:
        """
        Parse IG API dateUtc string to datetime object.
        IG uses format: '2025-12-29T09:48:15' or '2025/12/29 09:48:15'
        """
        if not date_str:
            return None

        try:
            # Try ISO format first (most common from IG)
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
            # Try slash format
            elif '/' in date_str:
                return datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
            # Try dash format without T
            else:
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError) as e:
            self.logger.warning(f"⚠️ Could not parse IG date '{date_str}': {e}")
            return None
    
    def _match_transactions_to_trades(
        self, 
        correlations: List[TradeCorrelation], 
        transactions: List[TransactionMatch]
    ):
        """Match transaction data to trade correlations by reference"""
        
        # Create lookup dictionary for efficient matching
        transaction_lookup = {tx.reference: tx for tx in transactions}
        
        matched_count = 0
        
        for correlation in correlations:
            reference = correlation.extracted_reference
            
            if reference in transaction_lookup:
                correlation.transaction_found = True
                correlation.transaction_match = transaction_lookup[reference]
                matched_count += 1
                
                self.logger.debug(f"✅ Matched: {correlation.close_deal_id} -> {reference} (P&L: {correlation.transaction_match.profit_loss} {correlation.transaction_match.currency})")
            else:
                correlation.transaction_found = False
                self.logger.debug(f"❌ No match: {correlation.close_deal_id} -> {reference}")
        
        self.logger.info(f"🎯 Matched {matched_count}/{len(correlations)} trades with transactions")
    
    async def _update_trade_log_with_pnl(self, correlations: List[TradeCorrelation]) -> List[Dict]:
        """Update trade_log entries with P&L from matched transactions"""
        updated_trades = []
        
        for correlation in correlations:
            if correlation.transaction_found and correlation.transaction_match:
                try:
                    # Ensure P&L columns exist
                    await self._ensure_pnl_columns_exist()
                    
                    # Update the trade_log entry
                    # Status should be 'closed' since we have transaction P/L
                    # Use IG's dateUtc for accurate closed_at (when trade actually closed on IG)
                    actual_close_time = correlation.transaction_match.date_utc or datetime.utcnow()
                    self.logger.info(f"🕐 Using IG close time: {actual_close_time} for trade {correlation.trade_log_id}")

                    update_data = {
                        "profit_loss": correlation.transaction_match.profit_loss,
                        "pnl_currency": correlation.transaction_match.currency,
                        "status": "closed",  # Change from expired/tracking to closed
                        "closed_at": actual_close_time,  # Use actual IG close time, not detection time
                        "updated_at": datetime.utcnow(),
                        "pnl_updated_at": datetime.utcnow()
                    }
                    
                    self.db_session.execute(
                        text("""
                            UPDATE trade_log
                            SET profit_loss = :profit_loss,
                                pnl_currency = :pnl_currency,
                                status = :status,
                                closed_at = :closed_at,
                                updated_at = :updated_at,
                                pnl_updated_at = :pnl_updated_at
                            WHERE id = :trade_log_id
                        """),
                        {**update_data, "trade_log_id": correlation.trade_log_id}
                    )
                    
                    correlation.updated = True
                    
                    updated_trades.append({
                        "trade_log_id": correlation.trade_log_id,
                        "close_deal_id": correlation.close_deal_id,
                        "reference": correlation.extracted_reference,
                        "symbol": correlation.symbol,
                        "profit_loss": correlation.transaction_match.profit_loss,
                        "currency": correlation.transaction_match.currency
                    })
                    
                    self.logger.info(f"✅ Updated trade_log {correlation.trade_log_id} with P&L: {correlation.transaction_match.profit_loss} {correlation.transaction_match.currency}")
                    
                except Exception as e:
                    correlation.error_message = str(e)
                    self.logger.error(f"❌ Error updating trade_log {correlation.trade_log_id}: {e}")
                    continue
        
        # Commit all updates
        if updated_trades:
            try:
                self.db_session.commit()
                self.logger.info(f"✅ Committed {len(updated_trades)} trade_log updates")
                # Post-mortems are now ON-DEMAND only — generated lazily via
                # GET /trade/{id}/postmortem when a trade is actually opened.
                # Auto-firing one per closed trade was disabled (Jun 9 2026):
                # single-trade (n=1) critiques produced noisy, ungrounded config
                # recommendations (e.g. wrong "filter RANGE_FADE to 20-22 UTC").
                # Strategy/config decisions belong to aggregate, on-demand review.
                # self._schedule_postmortems(updated_trades)
            except Exception as e:
                self.db_session.rollback()
                self.logger.error(f"❌ Error committing updates: {e}")

        return updated_trades

    def _schedule_postmortems(self, updated_trades: list) -> None:
        """Fire off postmortem tasks for each updated trade (non-blocking)."""
        try:
            from services.trade_postmortem_service import schedule_postmortem
            from config import TRADING_ENVIRONMENT
            for trade in updated_trades:
                trade_id = trade.get("trade_log_id")
                if trade_id:
                    schedule_postmortem(trade_id, TRADING_ENVIRONMENT)
        except Exception as e:
            self.logger.warning(f"Could not schedule postmortems: {e}")
    
    async def _ensure_pnl_columns_exist(self):
        """Ensure trade_log table has required P&L columns"""
        try:
            alter_queries = [
                "ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS profit_loss DECIMAL(12, 2)",
                "ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS pnl_currency VARCHAR(10) DEFAULT 'SEK'",
                "ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ]
            
            for query in alter_queries:
                self.db_session.execute(text(query))
            
            self.db_session.commit()
            self.logger.debug("✅ Ensured P&L columns exist in trade_log table")
            
        except Exception as e:
            self.logger.debug(f"Note: P&L columns may already exist: {e}")
            self.db_session.rollback()
    
    def _generate_correlation_summary(
        self, 
        correlations: List[TradeCorrelation], 
        updated_trades: List[Dict]
    ) -> Dict:
        """Generate summary statistics for correlation results"""
        
        total_processed = len(correlations)
        transactions_found = sum(1 for c in correlations if c.transaction_found)
        updated_count = len(updated_trades)
        
        total_pnl = sum(trade.get('profit_loss', 0) for trade in updated_trades)
        winning_trades = sum(1 for trade in updated_trades if trade.get('profit_loss', 0) > 0)
        losing_trades = sum(1 for trade in updated_trades if trade.get('profit_loss', 0) < 0)
        
        return {
            "total_processed": total_processed,
            "transactions_found": transactions_found,
            "updated_count": updated_count,
            "match_rate": round((transactions_found / total_processed * 100), 2) if total_processed > 0 else 0,
            "update_rate": round((updated_count / total_processed * 100), 2) if total_processed > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round((winning_trades / updated_count * 100), 2) if updated_count > 0 else 0
        }
    
    def _serialize_correlations(self, correlations: List[TradeCorrelation]) -> List[Dict]:
        """Serialize correlations for JSON response"""
        serialized = []
        
        for correlation in correlations:
            data = {
                "trade_log_id": correlation.trade_log_id,
                "close_deal_id": correlation.close_deal_id,
                "extracted_reference": correlation.extracted_reference,
                "symbol": correlation.symbol,
                "transaction_found": correlation.transaction_found,
                "updated": correlation.updated
            }
            
            if correlation.transaction_match:
                data["transaction_data"] = {
                    "reference": correlation.transaction_match.reference,
                    "profit_loss": correlation.transaction_match.profit_loss,
                    "currency": correlation.transaction_match.currency,
                    "instrument_name": correlation.transaction_match.instrument_name,
                    "transaction_date": correlation.transaction_match.transaction_date
                }
            
            if correlation.error_message:
                data["error"] = correlation.error_message
            
            serialized.append(data)
        
        return serialized


# Factory function for easy integration
def create_trade_pnl_correlator(db_session: Session = None, logger=None) -> TradePnLCorrelator:
    """Factory function to create TradePnLCorrelator instance"""
    return TradePnLCorrelator(db_session=db_session, logger=logger)


# Standalone function for integration with existing automation
async def update_trade_pnl_from_transactions(
    trading_headers: dict,
    days_back: int = 7,
    db_session: Session = None,
    logger=None
) -> Dict:
    """
    Standalone function to update trade P&L from IG transactions
    FIXED: Now uses activity_close_deal_id for proper correlation
    """
    correlator = TradePnLCorrelator(db_session=db_session, logger=logger)
    return await correlator.correlate_and_update_pnl(trading_headers, days_back)


if __name__ == "__main__":
    """
    Test the TradePnLCorrelator service with close deal ID
    """
    import asyncio
    
    async def test_correlator():
        print("🧪 Testing Trade P&L Correlator with CLOSE DEAL ID...")
        
        # Test reference extraction with real examples
        correlator = TradePnLCorrelator()
        
        test_deal_ids = [
            "DIAAAAUQJJCCFAM",  # Should extract QJJCCFAM
            "DIAAAAU P874HAAM",  # Should extract P874HAAM  
            "DIAAAAUABCDEF123"   # Should extract ABCDEF123
        ]
        
        for deal_id in test_deal_ids:
            reference = correlator._extract_reference_from_deal_id(deal_id)
            print(f"   {deal_id} -> {reference}")
        
        print("✅ Trade P&L Correlator test completed!")
        print("🔧 Ready for integration - now using activity_close_deal_id!")
        
    # Run test
    asyncio.run(test_correlator())