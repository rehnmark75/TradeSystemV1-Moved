#!/usr/bin/env python3
"""
Test script to verify Ichimoku ALL EPICS fix
"""
import requests
import json

def test_ichimoku_all_epics():
    url = "http://task-worker:8007/api/backtest/run"
    headers = {
        "Content-Type": "application/json",
        "x-apim-gateway": "verified"
    }
    payload = {
        "strategy_name": "ichimoku",
        "epic": None,  # ALL EPICS mode
        "days": 1,
        "timeframe": "15m",
        "show_signals": True,
        "parameters": {}
    }

    print("🧪 Testing Ichimoku strategy with ALL EPICS mode...")
    print(f"📤 Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                print(f"✅ SUCCESS: Found {data.get('total_signals', 0)} signals")
                print(f"📈 Strategy: {data.get('strategy_name')}")
                print(f"🎯 Epic: {data.get('epic', 'ALL EPICS')}")
                print(f"⏱️ Execution time: {data.get('execution_time', 0):.2f}s")
                return True
            else:
                print(f"❌ FAILED: {data.get('error_message', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_ichimoku_all_epics()
    exit(0 if success else 1)