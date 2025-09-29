#!/usr/bin/env python3
"""
Test script for level cluster detection functionality
Tests the scenario from the user's screenshot where buy signals below resistance clusters should be rejected
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the worker app path to Python path
sys.path.append('/home/hr/Projects/TradeSystemV1/worker/app')

from forex_scanner.core.detection.enhanced_support_resistance_validator import (
    EnhancedSupportResistanceValidator,
    LevelClusterType,
    LevelType
)

def create_test_data_with_resistance_cluster():
    """
    Create test data that mimics the scenario from the user's screenshot:
    - Current price around 1.10500
    - Buy signal generated
    - Multiple resistance levels stacked above (cluster)
    - Should result in trade rejection due to poor risk/reward
    """

    # Create 200 bars of sample data
    n_bars = 200
    base_price = 1.10000

    # Generate realistic price movement with volatility
    np.random.seed(42)  # For reproducible results
    price_changes = np.random.normal(0, 0.0008, n_bars)  # ~8 pip standard deviation

    # Create price series with some trend and support/resistance levels
    prices = [base_price]
    for i in range(1, n_bars):
        # Add some trending behavior
        trend_component = 0.00002 * i if i < 100 else -0.00002 * (i - 100)
        new_price = prices[-1] + price_changes[i] + trend_component
        prices.append(max(new_price, 1.09500))  # Floor at 1.09500

    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        # Add some intrabar volatility
        high = price + abs(np.random.normal(0, 0.0003))
        low = price - abs(np.random.normal(0, 0.0003))
        open_price = price + np.random.normal(0, 0.0001)
        close_price = price + np.random.normal(0, 0.0001)

        # Ensure OHLC logic
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        data.append({
            'timestamp': datetime.now() - timedelta(minutes=(n_bars-i) * 15),
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close_price, 5),
            'volume': np.random.randint(100, 1000)
        })

    # Create artificial resistance cluster around 1.10600-1.10700
    # This simulates the scenario where multiple resistance levels are stacked
    resistance_zone_bars = [120, 125, 130, 135, 140, 145]  # Multiple touch points
    for bar_idx in resistance_zone_bars:
        if bar_idx < len(data):
            # Create resistance touches
            resistance_price = 1.10600 + np.random.uniform(-0.00020, 0.00020)
            data[bar_idx]['high'] = max(data[bar_idx]['high'], resistance_price)
            data[bar_idx]['close'] = resistance_price - 0.00010  # Rejection from resistance

    return pd.DataFrame(data)

def create_test_signal():
    """Create a test BUY signal at current price"""
    return {
        'signal_type': 'BUY',
        'epic': 'CS.D.EURUSD.MINI.IP',
        'entry_price': 1.10520,  # Below the resistance cluster
        'timestamp': datetime.now(),
        'confidence': 0.75,
        'strategy': 'test_cluster_scenario'
    }

def test_cluster_detection():
    """Test the cluster detection system"""
    print("🧪 Testing Level Cluster Detection System")
    print("=" * 60)

    # Create test data
    df = create_test_data_with_resistance_cluster()
    test_signal = create_test_signal()

    print(f"📊 Test Data Created:")
    print(f"   Bars: {len(df)}")
    print(f"   Current Price: {test_signal['entry_price']}")
    print(f"   Signal Type: {test_signal['signal_type']}")
    print(f"   Price Range: {df['low'].min():.5f} - {df['high'].max():.5f}")

    # Initialize validator with cluster detection enabled
    try:
        validator = EnhancedSupportResistanceValidator(
            level_tolerance_pips=8.0,
            min_level_strength=0.3,
            enhanced_mode=True,
            enable_mtf_validation=False,  # Disable for testing
            enable_cluster_detection=True,  # Enable cluster detection
            max_cluster_radius_pips=15.0,
            min_levels_per_cluster=2,
            cluster_density_threshold=0.8,
            min_risk_reward_with_clusters=2.0
        )

        print(f"\n🔧 Validator Configuration:")
        print(f"   Enhanced Mode: ✅")
        print(f"   Cluster Detection: ✅")
        print(f"   Max Cluster Radius: {validator.max_cluster_radius_pips} pips")
        print(f"   Min Levels per Cluster: {validator.min_levels_per_cluster}")
        print(f"   Min Risk/Reward with Clusters: {validator.min_risk_reward_with_clusters}")

    except Exception as e:
        print(f"❌ Failed to initialize validator: {e}")
        return False

    # Test cluster detection
    try:
        print(f"\n🔍 Running Cluster Detection Test...")

        # Validate the trade
        is_valid, reason, details = validator.validate_trade_direction(
            test_signal, df, test_signal['epic']
        )

        print(f"\n📊 Cluster Detection Results:")
        print(f"   Trade Valid: {'✅ YES' if is_valid else '❌ NO'}")
        print(f"   Reason: {reason}")

        # Check if cluster information is present
        if 'cluster_risk_assessment' in details:
            cluster_risk = details['cluster_risk_assessment']
            print(f"\n🎯 Cluster Risk Assessment:")
            print(f"   Cluster Density Warning: {'⚠️ YES' if cluster_risk.cluster_density_warning else '✅ NO'}")
            print(f"   Distance to Cluster: {cluster_risk.cluster_distance_pips:.1f} pips")
            print(f"   Cluster Impact Score: {cluster_risk.cluster_impact_score:.2f}")
            print(f"   Expected Risk/Reward: {cluster_risk.expected_risk_reward:.1f}")
            print(f"   Intervening Levels: {cluster_risk.intervening_levels_count}")

            if cluster_risk.nearest_cluster:
                cluster = cluster_risk.nearest_cluster
                print(f"   Nearest Cluster Type: {cluster.cluster_type.value}")
                print(f"   Cluster Center: {cluster.center_price:.5f}")
                print(f"   Cluster Strength: {cluster.total_strength:.2f}")
                print(f"   Levels in Cluster: {len(cluster.levels)}")

        if 'clusters_detected' in details:
            print(f"   Total Clusters Detected: {details['clusters_detected']}")

        # Expected result check
        print(f"\n🎯 Expected vs Actual Results:")
        print(f"   Expected: Trade should be REJECTED due to resistance cluster above")
        print(f"   Actual: Trade {'REJECTED ✅' if not is_valid else 'ACCEPTED ❌'}")

        if not is_valid and 'cluster' in reason.lower():
            print(f"   Result: ✅ CORRECT - Cluster detection working as expected!")
        elif not is_valid:
            print(f"   Result: ⚠️ PARTIALLY CORRECT - Trade rejected but not for cluster reasons")
        else:
            print(f"   Result: ❌ INCORRECT - Trade should have been rejected due to cluster")

        return not is_valid and 'cluster' in reason.lower()

    except Exception as e:
        print(f"❌ Cluster detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_cluster_scenario():
    """Test a scenario where no clusters exist and trade should be allowed"""
    print(f"\n🧪 Testing No-Cluster Scenario")
    print("-" * 40)

    # Create simple test data without clusters
    n_bars = 100
    base_price = 1.10000

    data = []
    for i in range(n_bars):
        price = base_price + (i * 0.00001)  # Gentle uptrend
        data.append({
            'timestamp': datetime.now() - timedelta(minutes=(n_bars-i) * 15),
            'open': round(price - 0.00005, 5),
            'high': round(price + 0.00008, 5),
            'low': round(price - 0.00012, 5),
            'close': round(price, 5),
            'volume': 500
        })

    df = pd.DataFrame(data)
    test_signal = {
        'signal_type': 'BUY',
        'epic': 'CS.D.EURUSD.MINI.IP',
        'entry_price': base_price + 0.00050,
        'timestamp': datetime.now(),
        'confidence': 0.75
    }

    validator = EnhancedSupportResistanceValidator(
        enable_cluster_detection=True,
        min_levels_per_cluster=2,
        enhanced_mode=True
    )

    is_valid, reason, details = validator.validate_trade_direction(
        test_signal, df, test_signal['epic']
    )

    print(f"   No-Cluster Test Result:")
    print(f"   Trade Valid: {'✅ YES' if is_valid else '❌ NO'}")
    print(f"   Reason: {reason}")
    print(f"   Clusters Detected: {details.get('clusters_detected', 0)}")
    print(f"   Expected: Trade should be ALLOWED (no clusters)")
    print(f"   Result: {'✅ CORRECT' if is_valid else '❌ INCORRECT'}")

    return is_valid

if __name__ == "__main__":
    print("🚀 Level Cluster Detection Test Suite")
    print("="*60)

    # Test 1: Resistance cluster scenario (should reject)
    test1_passed = test_cluster_detection()

    # Test 2: No cluster scenario (should allow)
    test2_passed = test_no_cluster_scenario()

    print(f"\n🏁 Test Summary:")
    print(f"   Cluster Detection Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   No-Cluster Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print(f"\n🎉 All tests PASSED! Cluster detection is working correctly.")
        print(f"✅ The system will now reject trades with poor risk/reward due to level clusters")
        print(f"✅ This addresses the scenario from the user's screenshot")
    else:
        print(f"\n⚠️ Some tests FAILED. Check the implementation.")

    print(f"\n📋 Next Steps:")
    print(f"   1. Integration test with real market data")
    print(f"   2. Fine-tune cluster parameters based on backtesting")
    print(f"   3. Monitor cluster detection performance in live trading")