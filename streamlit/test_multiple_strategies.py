#!/usr/bin/env python3
"""
Test script to verify ALL EPICS mode works with multiple strategies
"""
import requests
import json

def test_strategy_all_epics(strategy_name):
    url = "http://task-worker:8007/api/backtest/run"
    headers = {
        "Content-Type": "application/json",
        "x-apim-gateway": "verified"
    }
    payload = {
        "strategy_name": strategy_name,
        "epic": None,  # ALL EPICS mode
        "days": 1,
        "timeframe": "15m",
        "show_signals": True,
        "parameters": {}
    }

    print(f"🧪 Testing {strategy_name} strategy with ALL EPICS mode...")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                print(f"✅ SUCCESS: {strategy_name} - Found {data.get('total_signals', 0)} signals")
                print(f"⏱️ Execution time: {data.get('execution_time', 0):.2f}s")
                return True
            else:
                print(f"❌ FAILED: {strategy_name} - {data.get('error_message', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP ERROR: {strategy_name} - {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {strategy_name} - {str(e)}")
        return False

if __name__ == "__main__":
    strategies_to_test = ["ema", "macd", "ichimoku"]
    results = {}

    for strategy in strategies_to_test:
        results[strategy] = test_strategy_all_epics(strategy)
        print("-" * 50)

    print(f"\n📈 Summary:")
    for strategy, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {strategy}: {status}")

    all_passed = all(results.values())
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    exit(0 if all_passed else 1)