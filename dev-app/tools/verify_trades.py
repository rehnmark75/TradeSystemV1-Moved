#!/usr/bin/env python3
"""
Manual recovery script to immediately check and recover trades marked as missing_on_ig
Run this script to recover trades without waiting for automatic sync
"""

import asyncio
import logging
from datetime import datetime
from services.db import SessionLocal
from services.models import TradeLog
from dependencies import get_ig_auth_headers
import httpx
from config import API_BASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def manual_recover_missing_trades():
    """
    Manually check and recover trades marked as missing_on_ig
    """
    try:
        logger.info("=" * 60)
        logger.info("🔍 MANUAL RECOVERY PROCESS STARTED")
        logger.info("=" * 60)
        
        with SessionLocal() as db:
            # Step 1: Find all trades marked as missing_on_ig
            missing_trades = db.query(TradeLog).filter(
                TradeLog.status == "missing_on_ig"
            ).all()
            
            if not missing_trades:
                logger.info("✅ No trades marked as missing_on_ig - nothing to recover")
                return
            
            logger.info(f"📊 Found {len(missing_trades)} trades marked as missing_on_ig:")
            for trade in missing_trades:
                logger.info(f"   - Trade {trade.id}: {trade.symbol} {trade.direction}")
                logger.info(f"     Deal ID: {trade.deal_id}")
                logger.info(f"     Deal Ref: {trade.deal_reference}")
            
            # Step 2: Get all open positions from IG
            logger.info("\n🔄 Fetching all open positions from IG...")
            trading_headers = await get_ig_auth_headers()
            
            headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{API_BASE_URL}/positions", headers=headers)
                response.raise_for_status()
                positions = response.json().get("positions", [])
            
            logger.info(f"📊 Found {len(positions)} open positions on IG")
            
            # Create lookup dictionaries
            ig_positions_by_deal_id = {}
            ig_positions_by_deal_ref = {}
            
            for pos in positions:
                deal_id = pos["position"]["dealId"]
                deal_ref = pos["position"].get("dealReference")
                
                ig_positions_by_deal_id[deal_id] = pos
                if deal_ref:
                    ig_positions_by_deal_ref[deal_ref] = pos
                    
                # Log the position details for debugging
                logger.info(f"   IG Position: {deal_id} - {pos['market']['epic']} {pos['position']['direction']}")
            
            # Step 3: Check each missing trade
            logger.info("\n🔍 Checking each missing trade against IG positions...")
            recovered = 0
            still_missing = 0
            
            for trade in missing_trades:
                logger.info(f"\n📌 Checking Trade {trade.id}:")
                found = False
                
                # Check by deal_id
                if trade.deal_id in ig_positions_by_deal_id:
                    logger.info(f"   ✅ FOUND by deal_id: {trade.deal_id}")
                    pos = ig_positions_by_deal_id[trade.deal_id]
                    
                    # Update the trade
                    trade.status = "tracking"
                    trade.exit_reason = "recovered_from_missing_by_deal_id"
                    trade.trigger_time = datetime.utcnow()
                    
                    logger.info(f"   ✅ Updated status to 'tracking'")
                    recovered += 1
                    found = True
                    
                # Check by deal_reference if not found by deal_id
                elif trade.deal_reference and trade.deal_reference in ig_positions_by_deal_ref:
                    logger.info(f"   ✅ FOUND by deal_reference: {trade.deal_reference}")
                    pos = ig_positions_by_deal_ref[trade.deal_reference]
                    
                    # Update the trade with correct deal_id
                    correct_deal_id = pos["position"]["dealId"]
                    logger.info(f"   📝 Updating deal_id from {trade.deal_id} to {correct_deal_id}")
                    
                    trade.deal_id = correct_deal_id
                    trade.status = "tracking"
                    trade.exit_reason = "recovered_from_missing_by_reference"
                    trade.trigger_time = datetime.utcnow()
                    
                    logger.info(f"   ✅ Updated status to 'tracking' and fixed deal_id")
                    recovered += 1
                    found = True
                
                # Fuzzy check by symbol and direction
                else:
                    logger.info(f"   🔍 Not found by exact match, checking fuzzy match...")
                    
                    for pos in positions:
                        if (pos["market"]["epic"] == trade.symbol and 
                            pos["position"]["direction"] == trade.direction):
                            
                            pos_entry = float(pos["position"]["level"])
                            if trade.entry_price:
                                price_diff = abs(pos_entry - trade.entry_price)
                                price_diff_pct = (price_diff / trade.entry_price) * 100
                                
                                if price_diff_pct < 0.1:  # Within 0.1%
                                    logger.info(f"   ⚠️ POSSIBLE MATCH found:")
                                    logger.info(f"      Position: {pos['position']['dealId']}")
                                    logger.info(f"      Entry: {pos_entry} vs {trade.entry_price} (diff: {price_diff_pct:.4f}%)")
                                    
                                    # Update with the matched position
                                    trade.deal_id = pos["position"]["dealId"]
                                    trade.status = "tracking"
                                    trade.exit_reason = "recovered_from_missing_by_fuzzy_match"
                                    trade.trigger_time = datetime.utcnow()
                                    
                                    logger.info(f"   ✅ Updated based on fuzzy match")
                                    recovered += 1
                                    found = True
                                    break
                
                if not found:
                    logger.info(f"   ❌ Trade still not found - genuinely missing")
                    still_missing += 1
            
            # Step 4: Commit all changes
            if recovered > 0:
                db.commit()
                logger.info(f"\n✅ Database updated successfully")
            
            # Step 5: Summary
            logger.info("\n" + "=" * 60)
            logger.info("📊 RECOVERY SUMMARY:")
            logger.info(f"   Total Checked: {len(missing_trades)}")
            logger.info(f"   Recovered: {recovered} ✅")
            logger.info(f"   Still Missing: {still_missing} ❌")
            logger.info("=" * 60)
            
            # Show specific deal we're looking for
            logger.info("\n🔍 Looking for specific deal DIAAAAUQWGKUCAB:")
            if "DIAAAAUQWGKUCAB" in ig_positions_by_deal_id:
                logger.info("   ✅ This deal IS in open positions!")
                pos = ig_positions_by_deal_id["DIAAAAUQWGKUCAB"]
                logger.info(f"   Symbol: {pos['market']['epic']}")
                logger.info(f"   Direction: {pos['position']['direction']}")
                logger.info(f"   Entry: {pos['position']['level']}")
            else:
                logger.info("   ❌ This deal NOT found in open positions")
            
            return {"recovered": recovered, "still_missing": still_missing}
            
    except Exception as e:
        logger.error(f"❌ Error in manual recovery: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

async def check_sync_status():
    """
    Check if the periodic sync is actually running
    """
    logger.info("\n🔍 Checking if periodic sync is running...")
    
    # Check the log file for recent sync activity
    try:
        with open("/app/logs/trade_sync.log", "r") as f:
            lines = f.readlines()[-50:]  # Last 50 lines
            
        recent_syncs = [line for line in lines if "[SYNC]" in line or "[RECOVERY]" in line]
        
        if recent_syncs:
            logger.info("📊 Recent sync activity found:")
            for line in recent_syncs[-5:]:  # Show last 5 sync entries
                logger.info(f"   {line.strip()}")
        else:
            logger.warning("⚠️ No recent sync activity found in logs")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not check sync logs: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting manual recovery process...")
    
    # Check sync status first
    asyncio.run(check_sync_status())
    
    # Run manual recovery
    result = asyncio.run(manual_recover_missing_trades())
    
    if result:
        if result["recovered"] > 0:
            logger.info(f"\n🎉 SUCCESS! Recovered {result['recovered']} trades")
        else:
            logger.info("\n⚠️ No trades were recovered - they may be genuinely missing")