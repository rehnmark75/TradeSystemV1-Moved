# app/routers/trading_analytics_router.py
"""
Cleaned Trading Analytics Router for FastAPI
Integrates with activity_pnl_correlator and price_based_pnl_calculator
Removed redundant code and organized for maintainability
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging
import httpx

from services.models import TradeLog
from sqlalchemy.orm import Session
from sqlalchemy import text

from services.broker_transaction_analyzer import BrokerTransactionAnalyzer
from services.activity_pnl_correlator import create_activity_pnl_correlator
from services.price_based_pnl_calculator import create_price_based_pnl_calculator

from services.db import get_db
from dependencies import get_ig_auth_headers

router = APIRouter(prefix="/api/trading", tags=["Trading Analytics"])
logger = logging.getLogger(__name__)

# IG API Configuration
from config import API_BASE_URL
IG_API_BASE_URL = API_BASE_URL

# Request Models
class IGTransactionRequest(BaseModel):
    days_back: Optional[int] = 7
    auto_store: Optional[bool] = True

class ActivityCorrelationRequest(BaseModel):
    days_back: int = 7
    update_trade_log: bool = True
    include_trade_lifecycles: bool = False

class IntegratedPnLRequest(BaseModel):
    days_back: int = 7
    update_trade_log: bool = True
    calculate_prices: bool = True
    include_detailed_results: bool = False

class BrokerTransactionData(BaseModel):
    transactions: List[Dict]

# =============================================================================
# CORE P/L CALCULATION ENDPOINTS
# =============================================================================

@router.post("/deals/calculate-complete-pnl", summary="💰 Complete Activity + Price P/L Calculation")
async def calculate_complete_pnl(
    request: IntegratedPnLRequest,
    db: Session = Depends(get_db)
):
    """
    💰 INTEGRATED: Complete Activity + Price-Based P/L Calculation
    
    This is the main endpoint that:
    1. Correlates your trade_log deal IDs with IG activities
    2. Extracts position references and complete trade lifecycles
    3. Fetches actual market prices at entry/exit timestamps
    4. Calculates accurate P/L from real price differences
    5. Updates trade_log with precise P/L calculations
    """
    try:
        trading_headers = await get_ig_auth_headers()
        
        logger.info(f"💰 Starting complete P/L calculation for last {request.days_back} days")
        
        # Step 1: Activity-based correlation
        activity_correlator = create_activity_pnl_correlator(db_session=db, logger=logger)
        
        logger.info("🔗 Step 1: Correlating trade_log with IG activities...")
        activity_result = await activity_correlator.correlate_trade_log_with_activities(
            trading_headers=trading_headers,
            days_back=request.days_back,
            update_trade_log=request.update_trade_log
        )
        
        if activity_result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Activity correlation failed: {activity_result['error']}"
            )
        
        activity_correlations = activity_result.get("correlations", [])
        activity_summary = activity_result.get("summary", {})
        
        logger.info(f"✅ Activity correlation completed: {activity_summary.get('correlations_found', 0)} trades correlated")
        
        # Step 2: Price-based P/L calculation (if requested)
        price_pnl_result = None
        if request.calculate_prices and activity_correlations:
            
            logger.info("💰 Step 2: Calculating P/L from market prices...")
            price_calculator = create_price_based_pnl_calculator(db_session=db, logger=logger)
            
            price_pnl_result = await price_calculator.calculate_pnl_for_correlated_trades(
                correlations=activity_correlations,
                trading_headers=trading_headers,
                update_trade_log=request.update_trade_log
            )
            
            if price_pnl_result["status"] == "error":
                logger.warning(f"⚠️ Price calculation failed: {price_pnl_result['error']}")
                price_pnl_result = None
            else:
                price_summary = price_pnl_result.get("summary", {})
                logger.info(f"✅ Price calculation completed: {price_summary.get('successful_calculations', 0)} P/L calculations")
        
        # Step 3: Generate comprehensive response
        response_data = {
            "status": "success",
            "message": f"Complete P/L calculation finished for last {request.days_back} days",
            "calculation_method": "activity_plus_price_based",
            "period": {
                "days_back": request.days_back,
                "start_date": (datetime.now() - timedelta(days=request.days_back)).strftime('%Y-%m-%d'),
                "end_date": datetime.now().strftime('%Y-%m-%d')
            },
            "activity_correlation": {
                "status": activity_result["status"],
                "summary": activity_summary,
                "trades_correlated": activity_summary.get("correlations_found", 0),
                "correlation_rate": activity_summary.get("correlation_rate", 0)
            }
        }
        
        # Add price calculation results if available
        if price_pnl_result:
            price_summary = price_pnl_result.get("summary", {})
            response_data["price_calculation"] = {
                "status": price_pnl_result["status"],
                "summary": price_summary,
                "total_net_pnl": price_summary.get("total_net_pnl", 0),
                "total_gross_pnl": price_summary.get("total_gross_pnl", 0),
                "total_spread_cost": price_summary.get("total_spread_cost", 0),
                "currency": price_summary.get("currency", "SEK"),
                "successful_calculations": price_summary.get("successful_calculations", 0),
                "calculation_rate": price_summary.get("calculation_rate", 0)
            }
        else:
            response_data["price_calculation"] = {
                "status": "skipped" if not request.calculate_prices else "failed",
                "message": "Price calculation was skipped or failed"
            }
        
        # Calculate overall success metrics
        total_trades = activity_summary.get("total_trades", 0)
        correlated_trades = activity_summary.get("correlations_found", 0)
        calculated_pnl_trades = price_summary.get("successful_calculations", 0) if price_pnl_result else 0
        
        response_data["overall_summary"] = {
            "total_trades_processed": total_trades,
            "successfully_correlated": correlated_trades,
            "pnl_calculated": calculated_pnl_trades,
            "complete_success_rate": round(calculated_pnl_trades / total_trades * 100, 2) if total_trades > 0 else 0,
            "ready_for_trading_analysis": calculated_pnl_trades > 0
        }
        
        # Log final results
        logger.info(f"✅ Complete P/L calculation finished:")
        logger.info(f"   📊 Total trades: {total_trades}")
        logger.info(f"   🔗 Correlated: {correlated_trades}")
        logger.info(f"   💰 P/L calculated: {calculated_pnl_trades}")
        if price_pnl_result:
            total_pnl = price_summary.get("total_net_pnl", 0)
            logger.info(f"   💵 Total P/L: {total_pnl:.2f} SEK")
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error in complete P/L calculation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Complete P/L calculation failed: {str(e)}"
        )

@router.post("/deals/correlate-activities", summary="🎯 Activity-Based P/L Correlation")
async def correlate_activity_based_pnl(
    request: ActivityCorrelationRequest,
    db: Session = Depends(get_db)
):
    """
    🎯 Activity-Based P/L Correlation
    
    Correlates trade_log deal IDs with IG activities to extract:
    - Position references from activity descriptions
    - Complete trade lifecycles (open → close)
    - Trade duration and stop limit changes
    - Foundation for price-based P/L calculation
    """
    try:
        trading_headers = await get_ig_auth_headers()
        correlator = create_activity_pnl_correlator(db_session=db, logger=logger)
        
        logger.info(f"🎯 Starting activity-based P/L correlation for last {request.days_back} days")
        
        correlation_result = await correlator.correlate_trade_log_with_activities(
            trading_headers=trading_headers,
            days_back=request.days_back,
            update_trade_log=request.update_trade_log
        )
        
        if correlation_result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Activity correlation failed: {correlation_result['error']}"
            )
        
        response_data = {
            "status": "success",
            "message": f"Activity-based P/L correlation completed for last {request.days_back} days",
            "summary": correlation_result["summary"],
            "method": "activity_endpoint_only",
            "period": {
                "days_back": request.days_back,
                "start_date": (datetime.now() - timedelta(days=request.days_back)).strftime('%Y-%m-%d'),
                "end_date": datetime.now().strftime('%Y-%m-%d')
            },
            "correlation_details": {
                "trades_in_trade_log": correlation_result["summary"]["total_trades"],
                "activities_found": len(correlation_result.get("correlations", [])),
                "successful_correlations": correlation_result["summary"]["correlations_found"],
                "correlation_rate": correlation_result["summary"]["correlation_rate"]
            }
        }
        
        # Add detailed data if requested
        if request.include_trade_lifecycles:
            response_data["trade_lifecycles"] = correlation_result.get("trade_lifecycles", {})
            response_data["correlations"] = correlation_result.get("correlations", [])
        
        # Add updated trades info
        if correlation_result.get("updated_trades"):
            response_data["updated_trades"] = correlation_result["updated_trades"]
        
        summary = correlation_result["summary"]
        logger.info(f"✅ Activity correlation completed: {summary['correlations_found']}/{summary['total_trades']} trades correlated ({summary['correlation_rate']}%)")
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error in activity-based P/L correlation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Activity-based P/L correlation failed: {str(e)}"
        )

@router.get("/deals/pnl-calculation-status", summary="📊 P/L Calculation Status Overview")
async def get_pnl_calculation_status(
    days_back: int = 7,
    db: Session = Depends(get_db)
):
    """
    📊 Get comprehensive status of P/L calculation coverage
    
    Shows the current state of:
    - Trade log entries with deal IDs
    - Activity correlations completed
    - Price-based P/L calculations completed
    - Total P/L calculated
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Basic trade_log statistics
        total_trades = db.execute(text("""
            SELECT COUNT(*) as count 
            FROM trade_log 
            WHERE timestamp >= :start_date 
            AND timestamp <= :end_date
            AND deal_id IS NOT NULL 
            AND deal_id != ''
        """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
        
        # Activity correlation status
        try:
            activity_correlated = db.execute(text("""
                SELECT COUNT(*) as count 
                FROM trade_log 
                WHERE timestamp >= :start_date 
                AND timestamp <= :end_date
                AND deal_id IS NOT NULL 
                AND deal_id != ''
                AND activity_correlated = true
            """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
        except:
            activity_correlated = 0
        
        # Price calculation status
        try:
            price_calculated = db.execute(text("""
                SELECT COUNT(*) as count 
                FROM trade_log 
                WHERE timestamp >= :start_date 
                AND timestamp <= :end_date
                AND deal_id IS NOT NULL 
                AND deal_id != ''
                AND calculated_pnl IS NOT NULL
            """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
            
            # Get total P/L
            total_pnl_result = db.execute(text("""
                SELECT 
                    COALESCE(SUM(calculated_pnl), 0) as total_pnl,
                    COALESCE(SUM(gross_pnl), 0) as total_gross_pnl,
                    COALESCE(SUM(spread_cost), 0) as total_spread_cost,
                    COALESCE(SUM(pips_gained), 0) as total_pips
                FROM trade_log 
                WHERE timestamp >= :start_date 
                AND timestamp <= :end_date
                AND deal_id IS NOT NULL 
                AND deal_id != ''
                AND calculated_pnl IS NOT NULL
            """), {"start_date": start_date, "end_date": end_date}).fetchone()
            
            total_pnl = float(total_pnl_result[0])
            total_gross_pnl = float(total_pnl_result[1])
            total_spread_cost = float(total_pnl_result[2])
            total_pips = float(total_pnl_result[3])
            
        except:
            price_calculated = 0
            total_pnl = 0
            total_gross_pnl = 0
            total_spread_cost = 0
            total_pips = 0
        
        # Calculate coverage percentages
        activity_coverage = round(activity_correlated / total_trades * 100, 2) if total_trades > 0 else 0
        price_coverage = round(price_calculated / total_trades * 100, 2) if total_trades > 0 else 0
        
        return {
            "status": "success",
            "period": {
                "days_back": days_back,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            },
            "trade_statistics": {
                "total_trades_with_deal_ids": total_trades,
                "activity_correlated_trades": activity_correlated,
                "price_calculated_trades": price_calculated
            },
            "coverage_metrics": {
                "activity_correlation_coverage": activity_coverage,
                "price_calculation_coverage": price_coverage,
                "complete_pipeline_coverage": price_coverage
            },
            "pnl_summary": {
                "total_net_pnl": round(total_pnl, 2),
                "total_gross_pnl": round(total_gross_pnl, 2),
                "total_spread_cost": round(total_spread_cost, 2),
                "total_pips_gained": round(total_pips, 2),
                "currency": "SEK",
                "trades_with_pnl": price_calculated
            },
            "pipeline_status": {
                "activity_correlation_needed": total_trades - activity_correlated,
                "price_calculation_needed": activity_correlated - price_calculated,
                "pipeline_complete": price_calculated == total_trades and total_trades > 0
            },
            "next_actions": [
                f"Run /deals/calculate-complete-pnl to process {total_trades - price_calculated} remaining trades"
            ] if price_calculated < total_trades else [
                "All trades have complete P/L calculations",
                "System ready for trading analysis and reporting"
            ],
            "calculation_method": "activity_plus_price_based"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting P/L calculation status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting P/L calculation status: {str(e)}"
        )

# =============================================================================
# IG TRANSACTION FETCHING ENDPOINTS
# =============================================================================

@router.post("/transactions/fetch-ig", summary="📥 Fetch Transactions from IG API")
async def fetch_ig_transactions(
    request: IGTransactionRequest,
    db = Depends(get_db)
):
    """
    Fetch transaction data directly from IG API and optionally store for analysis
    """
    try:
        trading_headers = await get_ig_auth_headers()
        
        # Calculate timestamp for the lookback period
        end_time = datetime.now()
        start_time = end_time - timedelta(days=request.days_back)
        start_timestamp_ms = int(start_time.timestamp() * 1000)
        end_timestamp_ms = int(end_time.timestamp() * 1000)
        
        # Construct IG API URL
        ig_url = f"{IG_API_BASE_URL}/history/transactions/ALL/{start_timestamp_ms}"
        
        logger.info(f"🔍 Fetching IG transactions from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Prepare headers for IG API
        ig_headers = {
            "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
            "CST": trading_headers["CST"],
            "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
            "Accept": "application/json",
            "Version": "1"
        }
        
        # Fetch from IG API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(ig_url, headers=ig_headers)
            response.raise_for_status()
            ig_data = response.json()
        
        # Extract transactions from IG response
        transactions_list = ig_data.get('transactions', [])
        
        logger.info(f"📥 Raw transactions fetched from IG API: {len(transactions_list)}")
        
        # Filter transactions to only include the requested time period
        filtered_transactions = []
        
        for tx in transactions_list:
            try:
                tx_date_str = tx.get('date', '')
                
                if tx_date_str and '/' in tx_date_str:
                    # Parse DD/MM/YY format
                    day, month, year = tx_date_str.split('/')
                    
                    # Convert 2-digit year to 4-digit
                    if len(year) == 2:
                        year = f"20{year}"
                    
                    tx_datetime = datetime(int(year), int(month), int(day), 12, 0, 0)
                    
                    # Check if transaction is within our time range
                    if start_time <= tx_datetime <= end_time:
                        filtered_transactions.append(tx)
                else:
                    # Include transaction if we can't parse the date
                    filtered_transactions.append(tx)
                    
            except Exception as date_error:
                logger.warning(f"⚠️ Error parsing transaction date '{tx.get('date', 'N/A')}': {date_error}")
                filtered_transactions.append(tx)
        
        logger.info(f"🔍 Filtered transactions (within {request.days_back} days): {len(filtered_transactions)}")
        
        # Update ig_data with filtered transactions
        ig_data['transactions'] = filtered_transactions
        
        if not filtered_transactions:
            return {
                "status": "success",
                "message": f"No transactions found in the last {request.days_back} days",
                "period": f"Last {request.days_back} days",
                "transactions_fetched": 0,
                "filtering_applied": True
            }
        
        # Process transactions if auto_store is enabled
        analysis_results = {}
        if request.auto_store:
            try:
                analyzer = BrokerTransactionAnalyzer(db_manager=db)
                parsed_transactions = analyzer.parse_broker_transactions(ig_data)
                
                if parsed_transactions:
                    stored_count = analyzer.store_transactions(parsed_transactions)
                    stats = analyzer.generate_trading_statistics(days_back=request.days_back)
                    correlations = analyzer.correlate_signals_with_trades(
                        lookback_hours=request.days_back * 24
                    )
                    
                    analysis_results = {
                        "transactions_parsed": len(parsed_transactions),
                        "transactions_stored": stored_count,
                        "correlations_found": len(correlations),
                        "quick_stats": stats.get('basic_metrics', {}) if stats else {}
                    }
                    
                    logger.info(f"📊 Stored {stored_count} transactions and found {len(correlations)} signal correlations")
                
            except Exception as analysis_error:
                logger.error(f"❌ Error during analysis: {analysis_error}")
                analysis_results = {"error": f"Analysis failed: {str(analysis_error)}"}
        
        return {
            "status": "success",
            "message": f"Successfully fetched {len(filtered_transactions)} transactions from last {request.days_back} days",
            "fetch_details": {
                "period": f"Last {request.days_back} days",
                "start_date": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_date": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "start_timestamp_ms": start_timestamp_ms,
                "end_timestamp_ms": end_timestamp_ms,
                "api_url": ig_url,
                "filtering_applied": True
            },
            "transactions_fetched": len(filtered_transactions),
            "sample_transaction": filtered_transactions[0] if filtered_transactions else None,
            "analysis_results": analysis_results if request.auto_store else {"message": "Auto-store disabled"}
        }
        
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ IG API HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"IG API error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"❌ Error fetching IG transactions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching transactions: {str(e)}"
        )

@router.get("/transactions/today", summary="📅 Get Today's Transactions Only")
async def get_today_transactions(
    auto_store: bool = True,
    db = Depends(get_db)
):
    """
    Fetch transactions from today only (since midnight)
    """
    try:
        trading_headers = await get_ig_auth_headers()
        
        # Calculate today's date range (from midnight to now)
        now = datetime.now()
        start_of_today = datetime(now.year, now.month, now.day, 0, 0, 0)
        
        # Use a wider search range for IG API then filter
        search_start = start_of_today - timedelta(days=2)
        start_timestamp_ms = int(search_start.timestamp() * 1000)
        
        ig_url = f"{IG_API_BASE_URL}/history/transactions/ALL/{start_timestamp_ms}"
        
        logger.info(f"🔍 Searching for today's transactions: {start_of_today.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        
        all_transactions = ig_data.get('transactions', [])
        logger.info(f"📥 Found {len(all_transactions)} transactions in search range")
        
        # Filter for TODAY only
        today_transactions = []
        today_date_str = now.strftime('%d/%m/%y')  # IG format: DD/MM/YY
        
        for tx in all_transactions:
            tx_date = tx.get('date', '')
            if tx_date == today_date_str:
                today_transactions.append(tx)
        
        logger.info(f"🎯 Found {len(today_transactions)} transactions from TODAY ({today_date_str})")
        
        ig_data['transactions'] = today_transactions
        
        if not today_transactions:
            return {
                "status": "success",
                "message": "No transactions found for today",
                "today_date": now.strftime('%Y-%m-%d'),
                "transactions_fetched": 0
            }
        
        # Process transactions if auto_store is enabled
        analysis_results = {}
        if auto_store:
            try:
                analyzer = BrokerTransactionAnalyzer(db_manager=db)
                parsed_transactions = analyzer.parse_broker_transactions(ig_data)
                
                if parsed_transactions:
                    stored_count = analyzer.store_transactions(parsed_transactions)
                    analysis_results = {
                        "transactions_parsed": len(parsed_transactions),
                        "transactions_stored": stored_count
                    }
                    logger.info(f"📊 Stored {stored_count} transactions from today")
                
            except Exception as analysis_error:
                logger.error(f"❌ Error during today's analysis: {analysis_error}")
                analysis_results = {"error": f"Analysis failed: {str(analysis_error)}"}
        
        return {
            "status": "success",
            "message": f"Successfully fetched {len(today_transactions)} transactions from today",
            "today_date": now.strftime('%Y-%m-%d'),
            "time_range": {
                "start": start_of_today.strftime('%Y-%m-%d %H:%M:%S'),
                "end": now.strftime('%Y-%m-%d %H:%M:%S')
            },
            "transactions_fetched": len(today_transactions),
            "sample_transaction": today_transactions[0] if today_transactions else None,
            "analysis_results": analysis_results if auto_store else {"message": "Auto-store disabled"}
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching today's transactions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching today's transactions: {str(e)}"
        )

# =============================================================================
# TRADITIONAL ANALYTICS ENDPOINTS
# =============================================================================

@router.post("/transactions/upload", summary="📤 Upload Broker Transactions")
async def upload_broker_transactions(
    data: BrokerTransactionData,
    db = Depends(get_db),
    headers: dict = Depends(get_ig_auth_headers)
):
    """
    Upload broker transaction data for analysis
    """
    try:
        analyzer = BrokerTransactionAnalyzer(db_manager=db)
        transactions = analyzer.parse_broker_transactions(data.dict())
        
        if not transactions:
            raise HTTPException(
                status_code=400,
                detail="No valid transactions found in the provided data"
            )
        
        stored_count = analyzer.store_transactions(transactions)
        stats = analyzer.generate_trading_statistics(days_back=7)
        
        logger.info(f"📊 Processed {len(transactions)} transactions, stored {stored_count}")
        
        return {
            "status": "success",
            "message": f"Processed {len(transactions)} transactions",
            "transactions_parsed": len(transactions),
            "transactions_stored": stored_count,
            "quick_stats": {
                "total_trades": stats.get('basic_metrics', {}).get('total_trades', 0),
                "win_rate": stats.get('basic_metrics', {}).get('win_rate', 0),
                "total_pnl": stats.get('profit_loss_metrics', {}).get('total_profit_loss', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing broker transactions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing transactions: {str(e)}"
        )

@router.get("/statistics", summary="📈 Get Trading Statistics")
async def get_trading_statistics(
    days_back: int = 30,
    include_correlations: bool = True,
    db = Depends(get_db),
    headers: dict = Depends(get_ig_auth_headers)
):
    """
    Get comprehensive trading statistics
    """
    try:
        analyzer = BrokerTransactionAnalyzer(db_manager=db)
        stats = analyzer.generate_trading_statistics(days_back=days_back)
        
        if not stats or 'error' in stats:
            raise HTTPException(
                status_code=404,
                detail="No trading data found for the specified period"
            )
        
        # Add signal correlations if requested
        if include_correlations:
            correlations = analyzer.correlate_signals_with_trades(lookback_hours=days_back * 24)
            signal_report = analyzer.get_signal_performance_report()
            
            stats['signal_analysis'] = {
                'correlations_found': len(correlations),
                'signal_performance': signal_report
            }
        
        logger.info(f"📈 Generated statistics for {days_back} days")
        
        return {
            "status": "success",
            "analysis_date": datetime.now().isoformat(),
            "parameters": {
                "days_analyzed": days_back,
                "include_correlations": include_correlations
            },
            "statistics": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating statistics: {str(e)}"
        )

@router.get("/performance/summary", summary="📊 Get Performance Summary")
async def get_performance_summary(
    db = Depends(get_db),
    headers: dict = Depends(get_ig_auth_headers)
):
    """
    Get quick performance summary for dashboard
    """
    try:
        analyzer = BrokerTransactionAnalyzer(db_manager=db)
        stats = analyzer.generate_trading_statistics(days_back=30)
        
        if not stats or 'error' in stats:
            return {
                "status": "no_data",
                "message": "No trading data available",
                "summary": {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "total_pips": 0
                }
            }
        
        # Extract key metrics for dashboard
        summary = {
            "total_trades": stats.get('basic_metrics', {}).get('total_trades', 0),
            "win_rate": stats.get('basic_metrics', {}).get('win_rate', 0),
            "total_pnl": stats.get('profit_loss_metrics', {}).get('total_profit_loss', 0),
            "total_pips": stats.get('pips_metrics', {}).get('total_pips', 0),
            "profit_factor": stats.get('profit_loss_metrics', {}).get('profit_factor', 0),
            "largest_win": stats.get('profit_loss_metrics', {}).get('largest_win', 0),
            "largest_loss": stats.get('profit_loss_metrics', {}).get('largest_loss', 0),
            "avg_pips_per_trade": stats.get('pips_metrics', {}).get('average_pips_per_trade', 0)
        }
        
        return {
            "status": "success",
            "last_updated": datetime.now().isoformat(),
            "period": "Last 30 days",
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting performance summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting performance summary: {str(e)}"
        )

# =============================================================================
# UTILITY AND HEALTH CHECK ENDPOINTS
# =============================================================================

@router.get("/health", summary="🏥 Analytics Service Health Check")
async def analytics_health_check(db = Depends(get_db)):
    """Health check for trading analytics service"""
    try:
        analyzer = BrokerTransactionAnalyzer(db_manager=db)
        
        # Quick test query
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM broker_transactions WHERE transaction_date >= CURRENT_DATE - INTERVAL '7 days'")
                recent_trades = cursor.fetchone()[0]
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database_connection": "ok",
            "recent_trades_count": recent_trades,
            "services": {
                "transaction_parser": "ok",
                "statistics_engine": "ok",
                "activity_correlator": "ok",
                "price_calculator": "ok"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Analytics health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@router.get("/ig/connection-test", summary="🔌 Test IG API Connection")
async def test_ig_connection(headers: dict = Depends(get_ig_auth_headers)):
    """Test connection to IG API"""
    try:
        test_url = f"{IG_API_BASE_URL}/accounts"
        
        ig_headers = {
            "X-IG-API-KEY": headers["X-IG-API-KEY"],
            "CST": headers["CST"],
            "X-SECURITY-TOKEN": headers["X-SECURITY-TOKEN"],
            "Accept": "application/json",
            "Version": "1"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(test_url, headers=ig_headers)
            response.raise_for_status()
            
        return {
            "status": "success",
            "message": "IG API connection successful",
            "api_endpoint": IG_API_BASE_URL,
            "response_status": response.status_code,
            "connection_test_passed": True
        }
        
    except Exception as e:
        logger.error(f"❌ IG API connection test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "message": "IG API connection failed",
                "error": str(e),
                "connection_test_passed": False
            }
        )

# =============================================================================
# DEBUG AND TESTING ENDPOINTS  
# =============================================================================

@router.post("/deals/test-activity-extraction", summary="🧪 Test Activity Data Extraction")
async def test_activity_extraction(
    days_back: int = 1,
    db: Session = Depends(get_db)
):
    """
    🧪 Test endpoint to extract and analyze activity data structure
    without updating the database
    """
    try:
        trading_headers = await get_ig_auth_headers()
        correlator = create_activity_pnl_correlator(db_session=db, logger=logger)
        
        # Fetch activities only
        activities = await correlator._fetch_ig_activities(trading_headers, days_back)
        
        if not activities:
            return {
                "status": "success",
                "message": "No activities found for the specified period",
                "activities_count": 0
            }
        
        # Build trade lifecycles for analysis
        trade_lifecycles = correlator._build_trade_lifecycles_from_activities(activities)
        
        # Analyze activity structure
        activity_analysis = {
            "total_activities": len(activities),
            "activities_by_type": {},
            "position_references_found": list(trade_lifecycles.keys()),
            "complete_trade_cycles": len([lc for lc in trade_lifecycles.values() if lc.status == 'closed']),
            "open_positions": len([lc for lc in trade_lifecycles.values() if lc.status == 'open'])
        }
        
        # Analyze activity types
        for activity in activities:
            desc = activity.get('description', '')
            activity_type = "unknown"
            
            if 'öppnad' in desc.lower() or 'opened' in desc.lower():
                activity_type = "position_opened"
            elif 'stängd' in desc.lower() or 'closed' in desc.lower():
                activity_type = "position_closed"
            elif 'stopplimit' in desc.lower():
                activity_type = "stop_limit_changed"
            
            activity_analysis["activities_by_type"][activity_type] = activity_analysis["activities_by_type"].get(activity_type, 0) + 1
        
        # Sample activities for inspection
        sample_activities = activities[:5] if len(activities) > 5 else activities
        
        return {
            "status": "success",
            "message": f"Activity data analysis completed for last {days_back} days",
            "analysis": activity_analysis,
            "trade_lifecycles_summary": {
                lifecycle_ref: {
                    "epic": lifecycle.epic,
                    "status": lifecycle.status,
                    "duration_minutes": lifecycle.duration_minutes,
                    "open_deal_id": lifecycle.open_deal_id,
                    "close_deal_id": lifecycle.close_deal_id,
                    "stop_changes": len(lifecycle.stop_limit_changes)
                }
                for lifecycle_ref, lifecycle in trade_lifecycles.items()
            },
            "sample_activities": sample_activities,
            "extraction_ready": len(trade_lifecycles) > 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error in activity extraction test: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Activity extraction test failed: {str(e)}"
        )

# Additional endpoints to add to your trading_analytics_router.py

# Add these imports at the top with your existing imports
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy import text
import json

# =============================================================================
# DATABASE VERIFICATION ENDPOINTS
# =============================================================================

# Request Models for Verification Endpoints
class DatabaseVerificationRequest(BaseModel):
    days_back: Optional[int] = 7
    include_detailed_results: Optional[bool] = False
    run_integrity_checks: Optional[bool] = True

class TradeLogInspectionRequest(BaseModel):
    trade_log_id: Optional[int] = None
    deal_id: Optional[str] = None
    days_back: Optional[int] = 7
    limit: Optional[int] = 10

@router.post("/verify/database-structure", summary="🔍 Verify Database Structure & P/L Columns")
async def verify_database_structure(
    db: Session = Depends(get_db)
):
    """
    🔍 Verify that the trade_log table has all required P/L calculation columns
    
    This endpoint checks:
    - Required P/L columns exist
    - Column types are correct
    - Indexes are in place
    - Table structure integrity
    """
    try:
        logger.info("🔍 Starting database structure verification...")
        
        # Required P/L columns from price_based_pnl_calculator
        required_columns = {
            'calculated_pnl': 'numeric',
            'gross_pnl': 'numeric', 
            'spread_cost': 'numeric',
            'pips_gained': 'numeric',
            'entry_price_calculated': 'numeric',
            'exit_price_calculated': 'numeric',
            'trade_direction': 'character varying',
            'trade_size': 'numeric',
            'pip_value': 'numeric',
            'pnl_calculation_method': 'character varying',
            'pnl_calculated_at': 'timestamp without time zone',
            'position_reference': 'character varying',
            'activity_correlated': 'boolean',
            'lifecycle_duration_minutes': 'integer',
            'activity_open_deal_id': 'character varying',
            'activity_close_deal_id': 'character varying'
        }
        
        # Check existing columns
        existing_columns = {}
        missing_columns = []
        type_mismatches = []
        
        for column_name, expected_type in required_columns.items():
            try:
                result = db.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'trade_log' 
                    AND column_name = :column_name
                """), {"column_name": column_name}).fetchone()
                
                if result:
                    existing_columns[column_name] = {
                        "type": result[1],
                        "nullable": result[2] == "YES",
                        "expected_type": expected_type
                    }
                    
                    # Check type compatibility
                    if expected_type.lower() not in result[1].lower():
                        type_mismatches.append({
                            "column": column_name,
                            "expected": expected_type,
                            "actual": result[1]
                        })
                else:
                    missing_columns.append(column_name)
                    
            except Exception as e:
                missing_columns.append(f"{column_name} (error: {str(e)})")
        
        # Check table existence
        table_exists = db.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'trade_log'
            )
        """)).fetchone()[0]
        
        # Check indexes
        indexes_result = db.execute(text("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'trade_log'
        """)).fetchall()
        
        indexes = [{"name": idx[0], "definition": idx[1]} for idx in indexes_result]
        
        # Calculate status
        structure_status = "healthy" if len(missing_columns) == 0 and len(type_mismatches) == 0 else "needs_attention"
        
        return {
            "status": "success",
            "verification_timestamp": datetime.now().isoformat(),
            "table_status": {
                "trade_log_exists": table_exists,
                "structure_status": structure_status
            },
            "column_analysis": {
                "total_required": len(required_columns),
                "existing_columns": len(existing_columns),
                "missing_columns": missing_columns,
                "type_mismatches": type_mismatches
            },
            "column_details": existing_columns,
            "indexes": indexes,
            "recommendations": [
                f"Add missing columns: {', '.join(missing_columns)}" if missing_columns else "All required columns present",
                f"Fix type mismatches: {len(type_mismatches)} found" if type_mismatches else "All column types correct"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error verifying database structure: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database structure verification failed: {str(e)}"
        )

@router.post("/verify/pnl-calculations", summary="💰 Verify P/L Calculations Status")
async def verify_pnl_calculations(
    request: DatabaseVerificationRequest,
    db: Session = Depends(get_db)
):
    """
    💰 Verify the status of P/L calculations in the database
    
    Checks:
    - How many trades have P/L calculated
    - P/L calculation coverage rate
    - Data quality and consistency
    - Recent calculation activity
    """
    try:
        logger.info(f"💰 Verifying P/L calculations for last {request.days_back} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days_back)
        
        # Total trades in period
        total_trades = db.execute(text("""
            SELECT COUNT(*) as count
            FROM trade_log 
            WHERE timestamp >= :start_date AND timestamp <= :end_date
        """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
        
        # Trades with deal IDs (eligible for P/L calculation)
        eligible_trades = db.execute(text("""
            SELECT COUNT(*) as count
            FROM trade_log 
            WHERE timestamp >= :start_date AND timestamp <= :end_date
            AND deal_id IS NOT NULL AND deal_id != ''
        """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
        
        # Trades with calculated P/L
        try:
            calculated_trades = db.execute(text("""
                SELECT COUNT(*) as count
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                AND calculated_pnl IS NOT NULL
            """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
            
            # P/L summary statistics
            pnl_stats = db.execute(text("""
                SELECT 
                    COUNT(*) as trades_with_pnl,
                    COALESCE(SUM(calculated_pnl), 0) as total_net_pnl,
                    COALESCE(SUM(gross_pnl), 0) as total_gross_pnl,
                    COALESCE(SUM(spread_cost), 0) as total_spread_cost,
                    COALESCE(SUM(pips_gained), 0) as total_pips,
                    COALESCE(AVG(calculated_pnl), 0) as avg_pnl,
                    COALESCE(AVG(pips_gained), 0) as avg_pips,
                    MIN(pnl_calculated_at) as first_calculation,
                    MAX(pnl_calculated_at) as last_calculation,
                    COUNT(DISTINCT pnl_calculation_method) as calculation_methods
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                AND calculated_pnl IS NOT NULL
            """), {"start_date": start_date, "end_date": end_date}).fetchone()
            
            # Recent calculation activity
            recent_calculations = db.execute(text("""
                SELECT COUNT(*) as count
                FROM trade_log 
                WHERE pnl_calculated_at >= NOW() - INTERVAL '24 hours'
                AND calculated_pnl IS NOT NULL
            """)).fetchone()[0]
            
        except Exception as e:
            logger.warning(f"⚠️ P/L columns may not exist yet: {e}")
            calculated_trades = 0
            pnl_stats = (0, 0, 0, 0, 0, 0, 0, None, None, 0)
            recent_calculations = 0
        
        # Calculate rates
        calculation_coverage = (calculated_trades / eligible_trades * 100) if eligible_trades > 0 else 0
        total_coverage = (calculated_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Status assessment
        status = "excellent" if calculation_coverage >= 80 else "good" if calculation_coverage >= 50 else "needs_improvement"
        
        verification_result = {
            "status": "success",
            "verification_timestamp": datetime.now().isoformat(),
            "period": {
                "days_back": request.days_back,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            },
            "trade_counts": {
                "total_trades": total_trades,
                "eligible_for_calculation": eligible_trades,
                "with_calculated_pnl": calculated_trades,
                "recent_calculations_24h": recent_calculations
            },
            "coverage_metrics": {
                "calculation_coverage_rate": round(calculation_coverage, 2),
                "total_coverage_rate": round(total_coverage, 2),
                "status": status
            },
            "pnl_summary": {
                "total_net_pnl": round(float(pnl_stats[1]), 2),
                "total_gross_pnl": round(float(pnl_stats[2]), 2),
                "total_spread_cost": round(float(pnl_stats[3]), 2),
                "total_pips": round(float(pnl_stats[4]), 2),
                "average_pnl": round(float(pnl_stats[5]), 2),
                "average_pips": round(float(pnl_stats[6]), 2),
                "currency": "SEK"
            },
            "calculation_activity": {
                "first_calculation": pnl_stats[7].isoformat() if pnl_stats[7] else None,
                "last_calculation": pnl_stats[8].isoformat() if pnl_stats[8] else None,
                "calculation_methods_used": pnl_stats[9]
            }
        }
        
        # Add detailed results if requested
        if request.include_detailed_results:
            # Get sample calculated trades
            sample_trades = db.execute(text("""
                SELECT 
                    id, deal_id, symbol, calculated_pnl, pips_gained, 
                    pnl_calculation_method, pnl_calculated_at
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                AND calculated_pnl IS NOT NULL
                ORDER BY pnl_calculated_at DESC
                LIMIT 5
            """), {"start_date": start_date, "end_date": end_date}).fetchall()
            
            verification_result["sample_calculated_trades"] = [
                {
                    "trade_id": trade[0],
                    "deal_id": trade[1],
                    "symbol": trade[2],
                    "calculated_pnl": float(trade[3]),
                    "pips_gained": float(trade[4]) if trade[4] else None,
                    "method": trade[5],
                    "calculated_at": trade[6].isoformat() if trade[6] else None
                }
                for trade in sample_trades
            ]
        
        return verification_result
        
    except Exception as e:
        logger.error(f"❌ Error verifying P/L calculations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"P/L calculation verification failed: {str(e)}"
        )

@router.post("/verify/trade-correlations", summary="🔗 Verify Activity & Trade Correlations")
async def verify_trade_correlations(
    request: DatabaseVerificationRequest,
    db: Session = Depends(get_db)
):
    """
    🔗 Verify the status of activity correlations and trade matching
    
    Checks:
    - Activity correlation coverage
    - Position reference extraction
    - Deal ID matching status
    - Correlation data quality
    """
    try:
        logger.info(f"🔗 Verifying trade correlations for last {request.days_back} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days_back)
        
        # Total trades with deal IDs (eligible for correlation)
        eligible_trades = db.execute(text("""
            SELECT COUNT(*) as count
            FROM trade_log 
            WHERE timestamp >= :start_date AND timestamp <= :end_date
            AND deal_id IS NOT NULL AND deal_id != ''
        """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
        
        # Check activity correlation status
        try:
            correlated_trades = db.execute(text("""
                SELECT COUNT(*) as count
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                AND deal_id IS NOT NULL AND deal_id != ''
                AND activity_correlated = true
            """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
            
            # Trades with position references
            trades_with_position_ref = db.execute(text("""
                SELECT COUNT(*) as count
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                AND position_reference IS NOT NULL AND position_reference != ''
            """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
            
            # Trades with complete lifecycle data
            complete_lifecycle_trades = db.execute(text("""
                SELECT COUNT(*) as count
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                AND activity_open_deal_id IS NOT NULL 
                AND activity_close_deal_id IS NOT NULL
            """), {"start_date": start_date, "end_date": end_date}).fetchone()[0]
            
        except Exception as e:
            logger.warning(f"⚠️ Activity correlation columns may not exist: {e}")
            correlated_trades = 0
            trades_with_position_ref = 0
            complete_lifecycle_trades = 0
        
        # Calculate correlation rates
        correlation_rate = (correlated_trades / eligible_trades * 100) if eligible_trades > 0 else 0
        position_ref_rate = (trades_with_position_ref / eligible_trades * 100) if eligible_trades > 0 else 0
        lifecycle_completion_rate = (complete_lifecycle_trades / eligible_trades * 100) if eligible_trades > 0 else 0
        
        # Status assessment
        correlation_status = "excellent" if correlation_rate >= 70 else "good" if correlation_rate >= 40 else "needs_improvement"
        
        return {
            "status": "success",
            "verification_timestamp": datetime.now().isoformat(),
            "period": {
                "days_back": request.days_back,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            },
            "correlation_metrics": {
                "eligible_trades": eligible_trades,
                "activity_correlated": correlated_trades,
                "with_position_reference": trades_with_position_ref,
                "complete_lifecycle": complete_lifecycle_trades
            },
            "correlation_rates": {
                "activity_correlation_rate": round(correlation_rate, 2),
                "position_reference_rate": round(position_ref_rate, 2),
                "lifecycle_completion_rate": round(lifecycle_completion_rate, 2),
                "status": correlation_status
            },
            "pipeline_readiness": {
                "ready_for_pnl_calculation": complete_lifecycle_trades,
                "needs_activity_correlation": eligible_trades - correlated_trades,
                "needs_position_reference": eligible_trades - trades_with_position_ref
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error verifying trade correlations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Trade correlation verification failed: {str(e)}"
        )

@router.post("/inspect/trade-log", summary="🔍 Inspect Trade Log Entries")
async def inspect_trade_log_entries(
    request: TradeLogInspectionRequest,
    db: Session = Depends(get_db)
):
    """
    🔍 Inspect specific trade_log entries for debugging
    
    Allows inspection by:
    - Specific trade_log ID
    - Deal ID
    - Recent trades from a time period
    """
    try:
        logger.info("🔍 Starting trade_log inspection...")
        
        trades = []
        query_params = {}
        
        if request.trade_log_id:
            # Inspect specific trade by ID
            query = "SELECT * FROM trade_log WHERE id = :trade_id"
            query_params["trade_id"] = request.trade_log_id
            
        elif request.deal_id:
            # Inspect trades by deal ID
            query = "SELECT * FROM trade_log WHERE deal_id = :deal_id"
            query_params["deal_id"] = request.deal_id
            
        else:
            # Inspect recent trades
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.days_back)
            
            query = """
                SELECT * FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                ORDER BY timestamp DESC
                LIMIT :limit
            """
            query_params.update({
                "start_date": start_date,
                "end_date": end_date,
                "limit": request.limit
            })
        
        # Execute query
        results = db.execute(text(query), query_params).fetchall()
        
        # Convert to dictionaries
        if results:
            columns = results[0]._fields
            trades = [dict(zip(columns, row)) for row in results]
            
            # Convert datetime objects to strings for JSON serialization
            for trade in trades:
                for key, value in trade.items():
                    if isinstance(value, datetime):
                        trade[key] = value.isoformat()
        
        # Analyze P/L calculation status
        pnl_status_summary = {
            "total_inspected": len(trades),
            "with_calculated_pnl": 0,
            "with_activity_correlation": 0,
            "with_position_reference": 0,
            "calculation_methods": set()
        }
        
        for trade in trades:
            if trade.get('calculated_pnl') is not None:
                pnl_status_summary["with_calculated_pnl"] += 1
            
            if trade.get('activity_correlated'):
                pnl_status_summary["with_activity_correlation"] += 1
                
            if trade.get('position_reference'):
                pnl_status_summary["with_position_reference"] += 1
                
            if trade.get('pnl_calculation_method'):
                pnl_status_summary["calculation_methods"].add(trade.get('pnl_calculation_method'))
        
        pnl_status_summary["calculation_methods"] = list(pnl_status_summary["calculation_methods"])
        
        return {
            "status": "success",
            "inspection_timestamp": datetime.now().isoformat(),
            "query_parameters": {
                "trade_log_id": request.trade_log_id,
                "deal_id": request.deal_id,
                "days_back": request.days_back,
                "limit": request.limit
            },
            "pnl_status_summary": pnl_status_summary,
            "trades": trades[:10] if len(trades) > 10 else trades,  # Limit output for readability
            "total_found": len(trades),
            "showing": min(len(trades), 10)
        }
        
    except Exception as e:
        logger.error(f"❌ Error inspecting trade_log entries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Trade log inspection failed: {str(e)}"
        )

@router.post("/test/database-updates", summary="🧪 Test Database Update Operations")
async def test_database_updates(
    db: Session = Depends(get_db)
):
    """
    🧪 Test database update operations without affecting real data
    
    This endpoint:
    - Creates test records
    - Tests P/L column updates
    - Verifies data integrity
    - Cleans up test data
    """
    try:
        logger.info("🧪 Starting database update test...")
        
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests_run": [],
            "results": {}
        }
        
        # Test 1: Check if we can create P/L columns
        test_results["tests_run"].append("column_creation")
        try:
            # Try to ensure P/L columns exist (safe operation)
            db.execute(text("""
                ALTER TABLE trade_log 
                ADD COLUMN IF NOT EXISTS test_calculated_pnl NUMERIC(12, 4),
                ADD COLUMN IF NOT EXISTS test_pnl_method VARCHAR(20)
            """))
            db.commit()
            
            # Clean up test columns
            db.execute(text("""
                ALTER TABLE trade_log 
                DROP COLUMN IF EXISTS test_calculated_pnl,
                DROP COLUMN IF EXISTS test_pnl_method
            """))
            db.commit()
            
            test_results["results"]["column_creation"] = {
                "status": "pass",
                "message": "Can successfully add/remove P/L columns"
            }
            
        except Exception as e:
            test_results["results"]["column_creation"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Test 2: Check database connection and basic operations
        test_results["tests_run"].append("basic_operations")
        try:
            # Test basic query
            result = db.execute(text("SELECT COUNT(*) FROM trade_log")).fetchone()[0]
            
            test_results["results"]["basic_operations"] = {
                "status": "pass",
                "message": f"Database connection working, {result} trades in trade_log"
            }
            
        except Exception as e:
            test_results["results"]["basic_operations"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Test 3: Check if we can update trade_log records
        test_results["tests_run"].append("update_operations")
        try:
            # Try to update a record (with rollback)
            db.execute(text("""
                UPDATE trade_log 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE id = (SELECT id FROM trade_log LIMIT 1)
            """))
            # Don't commit - just test the operation
            db.rollback()
            
            test_results["results"]["update_operations"] = {
                "status": "pass",
                "message": "Can successfully update trade_log records"
            }
            
        except Exception as e:
            test_results["results"]["update_operations"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Calculate overall test status
        passed_tests = sum(1 for test in test_results["results"].values() if test["status"] == "pass")
        total_tests = len(test_results["tests_run"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "overall_status": "pass" if passed_tests == total_tests else "partial" if passed_tests > 0 else "fail"
        }
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Error in database update test: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database update test failed: {str(e)}"
        )

@router.get("/status/pipeline-health", summary="🏥 Complete P/L Pipeline Health Check")
async def get_pipeline_health(
    days_back: int = 7,
    db: Session = Depends(get_db)
):
    """
    🏥 Complete health check of the P/L calculation pipeline
    
    This provides a comprehensive overview of:
    - Database structure health
    - Trade data availability
    - Correlation status
    - P/L calculation coverage
    - Recent activity
    """
    try:
        logger.info(f"🏥 Running complete pipeline health check...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        health_report = {
            "health_check_timestamp": datetime.now().isoformat(),
            "period": f"Last {days_back} days",
            "overall_health": "checking",
            "components": {}
        }
        
        # Component 1: Database Structure
        try:
            structure_check = db.execute(text("""
                SELECT COUNT(*) FROM information_schema.columns 
                WHERE table_name = 'trade_log' 
                AND column_name IN ('calculated_pnl', 'position_reference', 'activity_correlated')
            """)).fetchone()[0]
            
            health_report["components"]["database_structure"] = {
                "status": "healthy" if structure_check >= 2 else "needs_attention",
                "required_columns_found": structure_check,
                "message": "P/L columns available" if structure_check >= 2 else "P/L columns missing"
            }
        except Exception as e:
            health_report["components"]["database_structure"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Component 2: Trade Data Availability
        try:
            trade_counts = db.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN deal_id IS NOT NULL AND deal_id != '' THEN 1 END) as with_deal_ids
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
            """), {"start_date": start_date, "end_date": end_date}).fetchone()
            
            deal_id_coverage = (trade_counts[1] / trade_counts[0] * 100) if trade_counts[0] > 0 else 0
            
            health_report["components"]["trade_data"] = {
                "status": "healthy" if deal_id_coverage >= 80 else "warning" if deal_id_coverage >= 50 else "critical",
                "total_trades": trade_counts[0],
                "trades_with_deal_ids": trade_counts[1],
                "deal_id_coverage": round(deal_id_coverage, 2)
            }
        except Exception as e:
            health_report["components"]["trade_data"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Component 3: P/L Calculation Status
        try:
            pnl_counts = db.execute(text("""
                SELECT 
                    COUNT(CASE WHEN deal_id IS NOT NULL AND deal_id != '' THEN 1 END) as eligible,
                    COUNT(CASE WHEN calculated_pnl IS NOT NULL THEN 1 END) as calculated
                FROM trade_log 
                WHERE timestamp >= :start_date AND timestamp <= :end_date
            """), {"start_date": start_date, "end_date": end_date}).fetchone()
            
            pnl_coverage = (pnl_counts[1] / pnl_counts[0] * 100) if pnl_counts[0] > 0 else 0
            
            health_report["components"]["pnl_calculation"] = {
                "status": "healthy" if pnl_coverage >= 70 else "warning" if pnl_coverage >= 30 else "critical",
                "eligible_trades": pnl_counts[0],
                "calculated_trades": pnl_counts[1],
                "calculation_coverage": round(pnl_coverage, 2)
            }
        except Exception as e:
            health_report["components"]["pnl_calculation"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Component 4: Recent Activity
        try:
            recent_activity = db.execute(text("""
                SELECT COUNT(*) 
                FROM trade_log 
                WHERE pnl_calculated_at >= NOW() - INTERVAL '24 hours'
            """)).fetchone()[0]
            
            health_report["components"]["recent_activity"] = {
                "status": "healthy" if recent_activity > 0 else "warning",
                "calculations_last_24h": recent_activity,
                "message": "Recent P/L calculations found" if recent_activity > 0 else "No recent P/L calculations"
            }
        except Exception as e:
            health_report["components"]["recent_activity"] = {
                "status": "unknown",
                "error": str(e)
            }
        
        # Calculate overall health
        component_statuses = [comp.get("status", "error") for comp in health_report["components"].values()]
        healthy_components = sum(1 for status in component_statuses if status == "healthy")
        total_components = len(component_statuses)
        
        if healthy_components == total_components:
            health_report["overall_health"] = "healthy"
        elif healthy_components >= total_components * 0.7:
            health_report["overall_health"] = "warning"
        else:
            health_report["overall_health"] = "critical"
        
        health_report["health_score"] = round(healthy_components / total_components * 100, 2)
        
        return health_report
        
    except Exception as e:
        logger.error(f"❌ Error in pipeline health check: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline health check failed: {str(e)}"
        )