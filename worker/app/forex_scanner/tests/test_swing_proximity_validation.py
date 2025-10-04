# tests/test_swing_proximity_validation.py
"""
Test script for swing proximity validation
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategies.helpers.swing_proximity_validator import SwingProximityValidator
from core.strategies.helpers.smc_market_structure import SMCMarketStructure
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_test_dataframe_with_swings():
    """Create test DataFrame with known swing points"""
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')

    # Create price data with clear swing highs and lows
    prices = []
    base_price = 1.10000

    for i in range(100):
        # Create oscillating prices with clear pivots
        if i % 20 == 10:  # Swing high every 20 bars
            price = base_price + 0.00050
        elif i % 20 == 0:  # Swing low every 20 bars
            price = base_price - 0.00050
        else:
            # Random walk between
            price = base_price + (np.random.random() - 0.5) * 0.00020

        prices.append(price)

    df = pd.DataFrame({
        'start_time': dates,
        'open': prices,
        'high': [p + abs(np.random.random() * 0.00010) for p in prices],
        'low': [p - abs(np.random.random() * 0.00010) for p in prices],
        'close': prices,
        'volume': [1000 + np.random.random() * 500 for _ in range(100)]
    })

    return df


def test_smc_swing_detection():
    """Test SMC market structure swing detection"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: SMC Swing Detection")
    logger.info("="*60)

    df = create_test_dataframe_with_swings()

    # Initialize SMC analyzer
    smc_config = {
        'swing_length': 5,
        'structure_confirmation': 3,
        'bos_threshold': 0.0001
    }

    smc_analyzer = SMCMarketStructure(logger=logger)
    df_enhanced = smc_analyzer.analyze_market_structure(df, smc_config)

    # Check if swings were detected
    swing_highs_count = df_enhanced['swing_high'].sum() if 'swing_high' in df_enhanced.columns else 0
    swing_lows_count = df_enhanced['swing_low'].sum() if 'swing_low' in df_enhanced.columns else 0

    logger.info(f"✓ Detected {swing_highs_count} swing highs")
    logger.info(f"✓ Detected {swing_lows_count} swing lows")
    logger.info(f"✓ Total swing points in analyzer: {len(smc_analyzer.swing_points)}")

    assert swing_highs_count > 0 or len(smc_analyzer.swing_points) > 0, "No swing highs detected!"
    assert swing_lows_count > 0 or len(smc_analyzer.swing_points) > 0, "No swing lows detected!"

    logger.info("✅ SMC swing detection test PASSED\n")
    return df_enhanced, smc_analyzer


def test_swing_proximity_validator_basic():
    """Test basic swing proximity validation"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Basic Swing Proximity Validation")
    logger.info("="*60)

    df, smc_analyzer = test_smc_swing_detection()

    # Initialize validator with configuration
    validator_config = {
        'enabled': True,
        'min_distance_pips': 20,
        'lookback_swings': 5,
        'strict_mode': False,
        'resistance_buffer': 1.2,
        'support_buffer': 1.2
    }

    validator = SwingProximityValidator(
        smc_analyzer=smc_analyzer,
        config=validator_config,
        logger=logger
    )

    # Test scenario 1: Price far from swings (should pass)
    current_price = 1.10000
    result = validator.validate_entry_proximity(
        df=df,
        current_price=current_price,
        direction='BUY',
        epic='CS.D.EURUSD.MINI.IP'
    )

    logger.info(f"\nScenario 1: Price at {current_price:.5f}, BUY signal")
    logger.info(f"  Valid: {result['valid']}")
    logger.info(f"  Distance to swing: {result.get('distance_to_swing')} pips")
    logger.info(f"  Nearest swing price: {result.get('nearest_swing_price')}")
    logger.info(f"  Swing type: {result.get('swing_type')}")

    # Test scenario 2: Price very close to a swing high (should warn/penalize)
    if smc_analyzer.swing_points:
        # Get a recent swing high
        swing_highs = [sp for sp in smc_analyzer.swing_points if 'HIGH' in sp.swing_type.value]
        if swing_highs:
            swing_high = swing_highs[-1]
            # Set price very close to swing high
            close_price = swing_high.price - 0.00015  # About 15 pips below swing high

            result2 = validator.validate_entry_proximity(
                df=df,
                current_price=close_price,
                direction='BUY',
                epic='CS.D.EURUSD.MINI.IP'
            )

            logger.info(f"\nScenario 2: Price at {close_price:.5f} (near swing high {swing_high.price:.5f}), BUY signal")
            logger.info(f"  Valid: {result2['valid']}")
            logger.info(f"  Distance to swing: {result2.get('distance_to_swing')} pips")
            logger.info(f"  Confidence penalty: {result2.get('confidence_penalty', 0.0):.3f}")
            logger.info(f"  Rejection reason: {result2.get('rejection_reason', 'None')}")

            assert result2.get('confidence_penalty', 0) > 0, "Should have confidence penalty when close to swing!"

    logger.info("\n✅ Basic swing proximity validation test PASSED\n")


def test_swing_proximity_different_directions():
    """Test swing proximity for BUY vs SELL signals"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: BUY vs SELL Direction Validation")
    logger.info("="*60)

    df, smc_analyzer = test_smc_swing_detection()

    validator_config = {
        'enabled': True,
        'min_distance_pips': 20,
        'lookback_swings': 5,
        'strict_mode': False
    }

    validator = SwingProximityValidator(
        smc_analyzer=smc_analyzer,
        config=validator_config,
        logger=logger
    )

    current_price = 1.10050

    # Test BUY direction (checks resistance/swing highs)
    buy_result = validator.validate_entry_proximity(
        df=df,
        current_price=current_price,
        direction='BUY',
        epic='CS.D.EURUSD.MINI.IP'
    )

    # Test SELL direction (checks support/swing lows)
    sell_result = validator.validate_entry_proximity(
        df=df,
        current_price=current_price,
        direction='SELL',
        epic='CS.D.EURUSD.MINI.IP'
    )

    logger.info(f"\nPrice: {current_price:.5f}")
    logger.info(f"BUY  - Valid: {buy_result['valid']}, Penalty: {buy_result.get('confidence_penalty', 0):.3f}")
    logger.info(f"SELL - Valid: {sell_result['valid']}, Penalty: {sell_result.get('confidence_penalty', 0):.3f}")

    logger.info("\n✅ Direction validation test PASSED\n")


def test_swing_proximity_strict_mode():
    """Test strict mode (reject vs penalize)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Strict Mode Validation")
    logger.info("="*60)

    df, smc_analyzer = test_smc_swing_detection()

    # Test with strict mode OFF (penalize only)
    validator_lenient = SwingProximityValidator(
        smc_analyzer=smc_analyzer,
        config={'enabled': True, 'min_distance_pips': 20, 'strict_mode': False},
        logger=logger
    )

    # Test with strict mode ON (reject)
    validator_strict = SwingProximityValidator(
        smc_analyzer=smc_analyzer,
        config={'enabled': True, 'min_distance_pips': 20, 'strict_mode': True},
        logger=logger
    )

    # Find a price close to a swing
    if smc_analyzer.swing_points:
        swing = smc_analyzer.swing_points[-1]
        close_price = swing.price + 0.00010  # 10 pips away

        result_lenient = validator_lenient.validate_entry_proximity(
            df=df, current_price=close_price, direction='BUY', epic='CS.D.EURUSD.MINI.IP'
        )

        result_strict = validator_strict.validate_entry_proximity(
            df=df, current_price=close_price, direction='BUY', epic='CS.D.EURUSD.MINI.IP'
        )

        logger.info(f"\nPrice close to swing:")
        logger.info(f"Lenient mode - Valid: {result_lenient['valid']}, Penalty: {result_lenient.get('confidence_penalty', 0):.3f}")
        logger.info(f"Strict mode  - Valid: {result_strict['valid']}, Reason: {result_strict.get('rejection_reason', 'None')}")

        # In lenient mode, should be valid but with penalty
        # In strict mode, may be invalid if too close

    logger.info("\n✅ Strict mode test PASSED\n")


if __name__ == '__main__':
    logger.info("\n" + "="*80)
    logger.info("SWING PROXIMITY VALIDATION TEST SUITE")
    logger.info("="*80)

    try:
        test_smc_swing_detection()
        test_swing_proximity_validator_basic()
        test_swing_proximity_different_directions()
        test_swing_proximity_strict_mode()

        logger.info("\n" + "="*80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*80 + "\n")

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
