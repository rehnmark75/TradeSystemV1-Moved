# services/trade_automation_service.py
"""
Trade Automation Integration Service
Integrates the new P&L correlation with existing trade data automation

This service can be called from:
1. Scheduled jobs/cron tasks
2. Manual API endpoints  
3. Docker container automation
4. Integration with existing activity endpoint fetching

Usage:
    from services.trade_automation_service import TradeAutomationService
    
    service = TradeAutomationService()
    await service.run_complete_trade_sync(trading_headers)
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text

# Import the new P&L correlator
from services.trade_pnl_correlator import TradePnLCorrelator, update_trade_pnl_from_transactions

# Import existing services (adjust paths as needed)
try:
    from services.activity_pnl_correlator import ActivityPnLCorrelator
    from services.ig_deal_correlator import IGDealCorrelator
    from services.broker_transaction_analyzer import BrokerTransactionAnalyzer
    from services.db import get_db_session
    from services.models import TradeLog
except ImportError:
    print("⚠️ Warning: Adjust import paths based on your project structure")


class TradeAutomationService:
    """
    Unified service for automating all trade data synchronization
    Combines activity data, transaction data, and P&L correlation
    """
    
    def __init__(self, db_session: Session = None, logger=None):
        self.db_session = db_session or get_db_session()
        self.logger = logger or self._setup_logger()
        
        # Initialize component services
        self.pnl_correlator = TradePnLCorrelator(db_session=self.db_session, logger=self.logger)
        self.activity_correlator = ActivityPnLCorrelator(db_session=self.db_session, logger=self.logger)
        self.deal_correlator = IGDealCorrelator(db_session=self.db_session, logger=self.logger)
        self.transaction_analyzer = BrokerTransactionAnalyzer(logger=self.logger)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the service"""
        logger = logging.getLogger("TradeAutomationService")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def run_complete_trade_sync(
        self, 
        trading_headers: dict, 
        days_back: int = 7,
        include_activity_correlation: bool = True,
        include_transaction_correlation: bool = True
    ) -> Dict:
        """
        Run complete trade data synchronization including P&L correlation
        
        This is the main entry point for automating all trade data updates
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting complete trade synchronization...")
        
        results = {
            "status": "success",
            "start_time": start_time.isoformat(),
            "summary": {},
            "components": {}
        }
        
        try:
            # Component 1: Update trade P&L from IG transactions (NEW!)
            if include_transaction_correlation:
                self.logger.info("📊 Step 1: Updating trade P&L from IG transactions...")
                pnl_result = await self.pnl_correlator.correlate_and_update_pnl(
                    trading_headers=trading_headers,
                    days_back=days_back
                )
                results["components"]["pnl_correlation"] = pnl_result
                self.logger.info(f"✅ P&L correlation: {pnl_result.get('summary', {}).get('updated_count', 0)} trades updated")
            
            # Component 2: Activity-based correlation (existing)
            if include_activity_correlation:
                self.logger.info("🔄 Step 2: Running activity-based correlation...")
                activity_result = await self.activity_correlator.correlate_trade_log_with_activities(
                    trading_headers=trading_headers,
                    days_back=days_back,
                    update_trade_log=True
                )
                results["components"]["activity_correlation"] = activity_result
                self.logger.info(f"✅ Activity correlation: {len(activity_result.get('updated_trades', []))} trades updated")
            
            # Component 3: Generate combined summary
            combined_summary = self._generate_combined_summary(results["components"])
            results["summary"] = combined_summary
            
            # Component 4: Database cleanup and optimization
            cleanup_result = await self._perform_database_cleanup()
            results["components"]["database_cleanup"] = cleanup_result
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results["end_time"] = end_time.isoformat()
            results["duration_seconds"] = duration
            
            self.logger.info(f"🎉 Complete trade synchronization finished in {duration:.2f}s")
            self.logger.info(f"📊 Total trades updated: {combined_summary.get('total_trades_updated', 0)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in complete trade sync: {e}")
            import traceback
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    async def run_pnl_sync_only(
        self, 
        trading_headers: dict, 
        days_back: int = 7,
        specific_deal_ids: List[str] = None
    ) -> Dict:
        """
        Run only the P&L synchronization (new functionality)
        Useful for targeted updates or when only P&L data is needed
        """
        self.logger.info("💰 Running P&L synchronization only...")
        
        return await self.pnl_correlator.correlate_and_update_pnl(
            trading_headers=trading_headers,
            days_back=days_back,
            specific_deal_ids=specific_deal_ids
        )
    
    async def get_sync_status(self, days_back: int = 7) -> Dict:
        """
        Get current synchronization status for trade data
        Shows what trades have been correlated and what's missing
        """
        try:
            self.logger.info(f"📊 Checking sync status for last {days_back} days...")
            
            # Get trade statistics
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Query trade_log for status information
            from sqlalchemy import text
            
            status_query = text("""
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN deal_id IS NOT NULL AND deal_id != '' THEN 1 END) as with_deal_ids,
                    COUNT(CASE WHEN profit_loss IS NOT NULL THEN 1 END) as with_pnl,
                    COUNT(CASE WHEN position_reference IS NOT NULL THEN 1 END) as with_position_ref,
                    COUNT(CASE WHEN activity_correlated = true THEN 1 END) as activity_correlated,
                    COUNT(CASE WHEN status = 'closed' THEN 1 END) as closed_trades,
                    COALESCE(SUM(profit_loss), 0) as total_pnl,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losing_trades
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
            """)
            
            result = self.db_session.execute(status_query, {
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()
            
            if result:
                total_trades = result[0]
                status = {
                    "period": f"Last {days_back} days",
                    "total_trades": total_trades,
                    "with_deal_ids": result[1],
                    "with_pnl": result[2],
                    "with_position_reference": result[3],
                    "activity_correlated": result[4],
                    "closed_trades": result[5],
                    "total_pnl": float(result[6]),
                    "winning_trades": result[7],
                    "losing_trades": result[8]
                }
                
                # Calculate percentages
                if total_trades > 0:
                    status["deal_id_coverage"] = round((result[1] / total_trades) * 100, 2)
                    status["pnl_coverage"] = round((result[2] / total_trades) * 100, 2)
                    status["activity_coverage"] = round((result[4] / total_trades) * 100, 2)
                    status["win_rate"] = round((result[7] / result[2]) * 100, 2) if result[2] > 0 else 0
                else:
                    status.update({
                        "deal_id_coverage": 0,
                        "pnl_coverage": 0,
                        "activity_coverage": 0,
                        "win_rate": 0
                    })
                
                # Determine sync health
                if status["pnl_coverage"] >= 80:
                    status["sync_health"] = "excellent"
                elif status["pnl_coverage"] >= 60:
                    status["sync_health"] = "good"
                elif status["pnl_coverage"] >= 40:
                    status["sync_health"] = "fair"
                else:
                    status["sync_health"] = "poor"
                
                return {
                    "status": "success",
                    "sync_status": status,
                    "recommendations": self._generate_sync_recommendations(status)
                }
            else:
                return {
                    "status": "success",
                    "sync_status": {"total_trades": 0},
                    "message": "No trades found in the specified period"
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error checking sync status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_combined_summary(self, components: Dict) -> Dict:
        """Generate combined summary from all correlation components"""
        summary = {
            "total_trades_updated": 0,
            "pnl_trades_updated": 0,
            "activity_trades_updated": 0,
            "total_pnl": 0.0,
            "components_run": []
        }
        
        # P&L correlation summary
        if "pnl_correlation" in components:
            pnl_data = components["pnl_correlation"]
            if pnl_data.get("status") == "success":
                pnl_summary = pnl_data.get("summary", {})
                summary["pnl_trades_updated"] = pnl_summary.get("updated_count", 0)
                summary["total_pnl"] += pnl_summary.get("total_pnl", 0)
                summary["components_run"].append("pnl_correlation")
        
        # Activity correlation summary
        if "activity_correlation" in components:
            activity_data = components["activity_correlation"]
            if activity_data.get("status") == "success":
                summary["activity_trades_updated"] = len(activity_data.get("updated_trades", []))
                summary["components_run"].append("activity_correlation")
        
        # Calculate total unique trades updated
        summary["total_trades_updated"] = summary["pnl_trades_updated"] + summary["activity_trades_updated"]
        
        return summary
    
    def _generate_sync_recommendations(self, status: Dict) -> List[str]:
        """Generate recommendations based on sync status"""
        recommendations = []
        
        if status.get("pnl_coverage", 0) < 80:
            recommendations.append("Run P&L synchronization to improve profit/loss data coverage")
        
        if status.get("deal_id_coverage", 0) < 90:
            recommendations.append("Check deal_id population in trade_log entries")
        
        if status.get("activity_coverage", 0) < 70:
            recommendations.append("Run activity correlation to improve position reference mapping")
        
        if status.get("sync_health") == "poor":
            recommendations.append("Consider running complete trade synchronization")
        
        if not recommendations:
            recommendations.append("Synchronization status is healthy - no immediate actions needed")
        
        return recommendations
    
    async def _perform_database_cleanup(self) -> Dict:
        """Perform database cleanup and optimization"""
        try:
            cleanup_operations = []
            
            # Remove duplicate entries (if any)
            duplicate_query = text("""
                DELETE FROM trade_log t1 
                USING trade_log t2 
                WHERE t1.id < t2.id 
                AND t1.deal_id = t2.deal_id 
                AND t1.deal_id IS NOT NULL 
                AND t1.deal_id != ''
            """)
            
            result = self.db_session.execute(duplicate_query)
            duplicates_removed = result.rowcount
            cleanup_operations.append(f"Removed {duplicates_removed} duplicate trade entries")
            
            # Update trade status for closed trades with P&L
            status_update_query = text("""
                UPDATE trade_log 
                SET status = 'closed' 
                WHERE profit_loss IS NOT NULL 
                AND status != 'closed'
            """)
            
            result = self.db_session.execute(status_update_query)
            status_updates = result.rowcount
            cleanup_operations.append(f"Updated status for {status_updates} closed trades")
            
            self.db_session.commit()
            
            return {
                "status": "success",
                "operations": cleanup_operations,
                "duplicates_removed": duplicates_removed,
                "status_updates": status_updates
            }
            
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"❌ Error in database cleanup: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Convenience functions for easy integration

async def run_automated_trade_sync(trading_headers: dict, days_back: int = 7) -> Dict:
    """
    Convenience function to run automated trade synchronization
    Can be called from cron jobs, scheduled tasks, or API endpoints
    """
    service = TradeAutomationService()
    return await service.run_complete_trade_sync(trading_headers, days_back)


async def sync_trade_pnl_only(trading_headers: dict, days_back: int = 7) -> Dict:
    """
    Convenience function to sync only P&L data
    Useful for quick updates or when only profit/loss data is needed
    """
    service = TradeAutomationService()
    return await service.run_pnl_sync_only(trading_headers, days_back)


def get_trade_sync_status(days_back: int = 7) -> Dict:
    """
    Convenience function to check current sync status
    """
    service = TradeAutomationService()
    return asyncio.run(service.get_sync_status(days_back))


# Integration with existing Docker/cron automation
class TradeDataScheduler:
    """
    Scheduler for automated trade data synchronization
    Can be integrated with existing Docker containers or cron jobs
    """
    
    def __init__(self, trading_headers: dict, sync_interval_hours: int = 6):
        self.trading_headers = trading_headers
        self.sync_interval_hours = sync_interval_hours
        self.logger = logging.getLogger("TradeDataScheduler")
        self.service = TradeAutomationService()
    
    async def start_automated_sync(self):
        """Start automated synchronization loop"""
        self.logger.info(f"🔄 Starting automated trade sync every {self.sync_interval_hours} hours...")
        
        while True:
            try:
                self.logger.info("⏰ Running scheduled trade synchronization...")
                
                result = await self.service.run_complete_trade_sync(
                    trading_headers=self.trading_headers,
                    days_back=7  # Always sync last 7 days
                )
                
                if result["status"] == "success":
                    summary = result.get("summary", {})
                    self.logger.info(f"✅ Scheduled sync completed: {summary.get('total_trades_updated', 0)} trades updated")
                else:
                    self.logger.error(f"❌ Scheduled sync failed: {result.get('error', 'Unknown error')}")
                
                # Wait for next sync interval
                await asyncio.sleep(self.sync_interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"❌ Error in automated sync: {e}")
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)


if __name__ == "__main__":
    """
    Test the Trade Automation Service
    """
    async def test_service():
        print("🧪 Testing Trade Automation Service...")
        
        # Mock trading headers (replace with real ones for testing)
        mock_headers = {
            "X-IG-API-KEY": "your_api_key",
            "CST": "your_cst_token", 
            "X-SECURITY-TOKEN": "your_security_token"
        }
        
        service = TradeAutomationService()
        
        # Test sync status
        print("📊 Testing sync status check...")
        status = await service.get_sync_status(days_back=7)
        print(f"   Status: {status.get('sync_status', {}).get('sync_health', 'unknown')}")
        
        print("✅ Trade Automation Service test completed!")
        print("🔧 Ready for integration with existing automation")
        print("💡 Usage examples:")
        print("   - Add to Docker containers: await run_automated_trade_sync(headers)")
        print("   - Schedule with cron: TradeDataScheduler(headers).start_automated_sync()")
        print("   - Manual API calls: service.run_pnl_sync_only(headers)")
        
    # Run test
    asyncio.run(test_service())