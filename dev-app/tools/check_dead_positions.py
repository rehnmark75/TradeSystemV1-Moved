#!/usr/bin/env python3
"""
Standalone script to check for dead positions and close them in the database.
Run this from your container: python check_dead_positions.py
"""

import sys
import os
import logging
from datetime import datetime

# Add your project path if needed
sys.path.append('/app')  # Adjust this to your project path

from sqlalchemy.orm import Session
from services.db import SessionLocal
from services.models import TradeLog

# Import your existing classes
from trailing_class import TrailingConfig, SCALPING_CONFIG
import requests
import json


class PositionChecker:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        from config import FASTAPI_DEV_URL
        self.base_url = FASTAPI_DEV_URL
        self.subscription_key = "436abe054a074894a0517e5172f0e5b6"
        
        # Setup basic logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
        self.logger = logging.getLogger(__name__)

    def test_position_exists(self, epic: str, direction: str) -> bool:
        """Test if a position still exists by sending a minimal adjustment"""
        
        payload = {
            "epic": epic,
            "adjustDirectionStop": "increase" if direction == "BUY" else "decrease",
            "adjustDirectionLimit": "increase",
            "stop_offset_points": 1,  # Minimal adjustment
            "limit_offset_points": 0,
            "dry_run": self.dry_run
        }
        
        headers = {
            "X-APIM-Gateway": "verified",
            "X-API-KEY": self.subscription_key,
            "Content-Type": "application/json"
        }
        
        try:
            url = f"{self.base_url}/orders/adjust-stop"
            self.logger.info(f"[TESTING] {epic} - Checking position existence...")
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                
                if status == "closed":
                    self.logger.warning(f"[CLOSED] {epic} - Position no longer exists")
                    return False
                elif status in ["updated", "dry_run"]:
                    self.logger.info(f"[EXISTS] {epic} - Position still active")
                    return True
                else:
                    self.logger.warning(f"[UNKNOWN] {epic} - Status: {status}")
                    return True  # Assume exists if unknown
            else:
                self.logger.error(f"[ERROR] {epic} - HTTP {response.status_code}: {response.text}")
                return True  # Assume exists if error
                
        except Exception as e:
            self.logger.error(f"[EXCEPTION] {epic} - Error checking position: {e}")
            return True  # Assume exists if exception

    def check_and_close_dead_positions(self):
        """Check all active trades and close those with dead positions"""
        
        with SessionLocal() as db:
            try:
                # Get all active trades
                active_trades = (db.query(TradeLog)
                               .filter(TradeLog.status.in_(["pending", "tracking", "break_even", "trailing"]),
                                       TradeLog.endpoint == "dev")
                               .all())
                
                self.logger.info(f"[START] Checking {len(active_trades)} active trades")
                
                if not active_trades:
                    self.logger.info("[RESULT] No active trades found")
                    return
                
                closed_count = 0
                
                for trade in active_trades:
                    self.logger.info(f"[CHECKING] Trade {trade.id} - {trade.symbol} ({trade.direction}) - Status: {trade.status}")
                    
                    # Test if position still exists
                    position_exists = self.test_position_exists(trade.symbol, trade.direction)
                    
                    if not position_exists:
                        self.logger.warning(f"[CLOSING] Trade {trade.id} - {trade.symbol} - Position dead")
                        trade.status = "closed"
                        trade.trigger_time = datetime.utcnow()
                        closed_count += 1
                    else:
                        self.logger.info(f"[ACTIVE] Trade {trade.id} - {trade.symbol} - Position still exists")
                
                # Commit changes
                if closed_count > 0:
                    db.commit()
                    self.logger.info(f"[RESULT] Closed {closed_count} dead positions")
                else:
                    self.logger.info(f"[RESULT] All {len(active_trades)} positions are still active")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] Error checking positions: {e}")
                db.rollback()

    def show_active_trades(self):
        """Show current active trades"""
        with SessionLocal() as db:
            active_trades = (db.query(TradeLog)
                           .filter(TradeLog.status.in_(["pending", "tracking", "break_even", "trailing"]),
                                   TradeLog.endpoint == "dev")
                           .all())
            
            print(f"\n=== ACTIVE TRADES ({len(active_trades)}) ===")
            for trade in active_trades:
                print(f"Trade {trade.id}: {trade.symbol} {trade.direction} - Status: {trade.status}")
                print(f"  Entry: {trade.entry_price}, SL: {trade.sl_price}")
                print(f"  Moved to BE: {getattr(trade, 'moved_to_breakeven', 'Unknown')}")
                print()


def main():
    print("=== Position Checker ===")
    
    # Create checker (dry_run=False to actually test positions)
    checker = PositionChecker(dry_run=False)  # Set to True for testing without real API calls
    
    # Show current active trades
    checker.show_active_trades()
    
    # Ask for confirmation
    response = input("\nDo you want to check for dead positions? (y/N): ")
    if response.lower() == 'y':
        checker.check_and_close_dead_positions()
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()