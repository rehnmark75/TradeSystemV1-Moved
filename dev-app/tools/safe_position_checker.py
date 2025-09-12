#!/usr/bin/env python3
"""
Safe position checker that only verifies without making changes
"""

import sys
import requests
import json
sys.path.append('/app')

from services.db import SessionLocal
from services.models import TradeLog

def check_position_via_fastapi(epic: str, dry_run: bool = True):
    """Check if position exists using your FastAPI endpoint"""
    
    payload = {
        "epic": epic,
        "adjustDirectionStop": "increase",
        "adjustDirectionLimit": "increase", 
        "stop_offset_points": 1,
        "limit_offset_points": 0,
        "dry_run": dry_run  # Always use dry_run for checking
    }
    
    headers = {
        "X-APIM-Gateway": "verified",
        "X-API-KEY": "436abe054a074894a0517e5172f0e5b6",
        "Content-Type": "application/json"
    }
    
    try:
        from config import ADJUST_STOP_URL
        url = ADJUST_STOP_URL
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")
            message = result.get("message", "")
            
            return {
                "exists": status != "closed",
                "status": status,
                "message": message,
                "response": result
            }
        else:
            return {
                "exists": None,
                "status": "error",
                "message": f"HTTP {response.status_code}: {response.text}",
                "response": None
            }
            
    except Exception as e:
        return {
            "exists": None,
            "status": "exception", 
            "message": str(e),
            "response": None
        }

def verify_all_positions():
    """Verify all active positions without making changes"""
    
    with SessionLocal() as db:
        active_trades = (db.query(TradeLog)
                       .filter(TradeLog.status.in_(["pending", "tracking", "break_even", "trailing"]),
                               TradeLog.endpoint == "dev")
                       .all())
        
        print(f"=== POSITION VERIFICATION ({len(active_trades)} trades) ===\n")
        
        for trade in active_trades:
            print(f"🔍 Trade {trade.id}: {trade.symbol} ({trade.direction})")
            print(f"   Status: {trade.status}")
            print(f"   Entry: {trade.entry_price}, SL: {trade.sl_price}")
            
            # Check position with dry_run=True (safe)
            result = check_position_via_fastapi(trade.symbol, dry_run=True)
            
            if result["exists"] is True:
                print(f"   ✅ Position EXISTS on IG")
            elif result["exists"] is False:
                print(f"   ❌ Position CLOSED on IG: {result['message']}")
            else:
                print(f"   ⚠️  Cannot verify: {result['message']}")
            
            print(f"   API Status: {result['status']}")
            print()

def check_specific_epic(epic: str):
    """Check a specific epic"""
    print(f"Checking {epic}...")
    result = check_position_via_fastapi(epic, dry_run=True)
    
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Full response: {json.dumps(result['response'], indent=2)}")

if __name__ == "__main__":
    print("=== SAFE Position Verification ===")
    print("This script only CHECKS positions, it does NOT make changes.\n")
    
    choice = input("1) Verify all active trades\n2) Check specific epic\n3) Exit\nChoice: ")
    
    if choice == "1":
        verify_all_positions()
    elif choice == "2":
        from config import DEFAULT_EPICS
        print(f"Available test epics: {list(DEFAULT_EPICS.values())}")
        epic = input("Enter epic (e.g., CS.D.GBPUSD.MINI.IP): ")
        check_specific_epic(epic)
    else:
        print("Exiting...")