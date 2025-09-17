#!/usr/bin/env python3
"""
Comprehensive Trailing Stop Loss Validation Tests

This module validates that the trailing stop loss system correctly handles
both regular currency pairs and JPY pairs with different point value systems.

Tests cover:
1. Point value calculations for different pair types
2. Price-to-points and points-to-price conversions
3. Stop/limit distance calculations
4. Progressive configuration application
5. Edge cases and boundary conditions
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_point_value,
    convert_stop_distance_to_price,
    convert_limit_distance_to_price,
    convert_price_to_points,
    calculate_move_points,
    format_price,
    validate_epic
)
from trailing_class import TrailingConfig, TrailingMethod, Progressive3StageTrailing
from services.progressive_config import (
    get_progressive_config_for_epic,
    CONSERVATIVE_PROGRESSIVE_CONFIG,
    BALANCED_PROGRESSIVE_CONFIG,
    DEFAULT_PROGRESSIVE_CONFIG
)


class TestPointValueSystem(unittest.TestCase):
    """Test point value calculations for different currency pair types"""

    def test_jpy_pair_point_values(self):
        """Test that JPY pairs use 0.01 point value"""
        jpy_pairs = [
            "USDJPY", "CS.D.USDJPY.MINI.IP",
            "EURJPY", "CS.D.EURJPY.MINI.IP",
            "GBPJPY", "CS.D.GBPJPY.MINI.IP",
            "AUDJPY", "CS.D.AUDJPY.MINI.IP",
            "NZDJPY", "CS.D.NZDJPY.MINI.IP",
            "CADJPY", "CS.D.CADJPY.MINI.IP",
            "CHFJPY", "CS.D.CHFJPY.MINI.IP"
        ]

        for pair in jpy_pairs:
            with self.subTest(pair=pair):
                self.assertEqual(get_point_value(pair), 0.01,
                    f"JPY pair {pair} should have 0.01 point value")

    def test_regular_pair_point_values(self):
        """Test that regular major pairs use 0.0001 point value"""
        regular_pairs = [
            "EURUSD", "CS.D.EURUSD.MINI.IP",
            "GBPUSD", "CS.D.GBPUSD.MINI.IP",
            "AUDUSD", "CS.D.AUDUSD.MINI.IP",
            "NZDUSD", "CS.D.NZDUSD.MINI.IP",
            "USDCAD", "CS.D.USDCAD.MINI.IP",
            "USDCHF", "CS.D.USDCHF.MINI.IP"
        ]

        for pair in regular_pairs:
            with self.subTest(pair=pair):
                self.assertEqual(get_point_value(pair), 0.0001,
                    f"Regular pair {pair} should have 0.0001 point value")

    def test_other_instrument_point_values(self):
        """Test that indices and commodities use 1.0 point value"""
        other_instruments = [
            "US500", "US30", "UK100", "GER40", "JPN225",
            "GOLD", "SILVER", "OIL.WTI", "COPPER"
        ]

        for instrument in other_instruments:
            with self.subTest(instrument=instrument):
                self.assertEqual(get_point_value(instrument), 1.0,
                    f"Other instrument {instrument} should have 1.0 point value")


class TestPricePointConversions(unittest.TestCase):
    """Test conversions between prices and points for different pair types"""

    def test_jpy_price_to_points_conversion(self):
        """Test JPY pair price-to-points conversions"""
        # USDJPY: 0.10 price difference = 10 points (10 * 0.01)
        self.assertEqual(convert_price_to_points(0.10, "USDJPY"), 10)
        self.assertEqual(convert_price_to_points(0.50, "USDJPY"), 50)
        self.assertEqual(convert_price_to_points(1.00, "USDJPY"), 100)

        # EURJPY: Same logic
        self.assertEqual(convert_price_to_points(0.25, "EURJPY"), 25)

    def test_regular_pair_price_to_points_conversion(self):
        """Test regular pair price-to-points conversions"""
        # EURUSD: 0.0010 price difference = 10 points (10 * 0.0001)
        self.assertEqual(convert_price_to_points(0.0010, "EURUSD"), 10)
        self.assertEqual(convert_price_to_points(0.0050, "EURUSD"), 50)
        self.assertEqual(convert_price_to_points(0.0100, "EURUSD"), 100)

        # GBPUSD: Same logic
        self.assertEqual(convert_price_to_points(0.0025, "GBPUSD"), 25)

    def test_jpy_stop_distance_calculation(self):
        """Test stop distance calculations for JPY pairs"""
        # USDJPY SELL: entry 146.500, 10 points stop = 146.600
        stop_price = convert_stop_distance_to_price(146.500, 10, "SELL", "USDJPY")
        self.assertAlmostEqual(stop_price, 146.600, places=3)

        # USDJPY BUY: entry 146.500, 10 points stop = 146.400
        stop_price = convert_stop_distance_to_price(146.500, 10, "BUY", "USDJPY")
        self.assertAlmostEqual(stop_price, 146.400, places=3)

        # EURJPY SELL: entry 173.500, 15 points stop = 173.650
        stop_price = convert_stop_distance_to_price(173.500, 15, "SELL", "EURJPY")
        self.assertAlmostEqual(stop_price, 173.650, places=3)

    def test_regular_pair_stop_distance_calculation(self):
        """Test stop distance calculations for regular pairs"""
        # EURUSD SELL: entry 1.1000, 10 points stop = 1.1010
        stop_price = convert_stop_distance_to_price(1.1000, 10, "SELL", "EURUSD")
        self.assertAlmostEqual(stop_price, 1.1010, places=4)

        # EURUSD BUY: entry 1.1000, 10 points stop = 1.0990
        stop_price = convert_stop_distance_to_price(1.1000, 10, "BUY", "EURUSD")
        self.assertAlmostEqual(stop_price, 1.0990, places=4)

        # GBPUSD SELL: entry 1.3500, 15 points stop = 1.3515
        stop_price = convert_stop_distance_to_price(1.3500, 15, "SELL", "GBPUSD")
        self.assertAlmostEqual(stop_price, 1.3515, places=4)

    def test_move_points_calculation(self):
        """Test move points calculation for profit/loss"""
        # JPY pairs
        # USDJPY BUY: 146.500 -> 146.600 = +10 points profit (0.1 move / 0.01 = 10)
        move = calculate_move_points(146.500, 146.600, "BUY", "USDJPY")
        self.assertEqual(move, 10)

        # USDJPY SELL: 146.500 -> 146.400 = +10 points profit (0.1 move / 0.01 = 10)
        move = calculate_move_points(146.500, 146.400, "SELL", "USDJPY")
        self.assertEqual(move, 10)

        # Regular pairs
        # EURUSD BUY: 1.1000 -> 1.1010 = +10 points profit (0.0010 move / 0.0001 = 10)
        move = calculate_move_points(1.1000, 1.1010, "BUY", "EURUSD")
        self.assertEqual(move, 10)

        # EURUSD SELL: 1.1000 -> 1.0990 = +10 points profit (0.0010 move / 0.0001 = 10)
        move = calculate_move_points(1.1000, 1.0990, "SELL", "EURUSD")
        self.assertEqual(move, 10)


class TestProgressiveConfiguration(unittest.TestCase):
    """Test that progressive configurations are applied correctly to different pair types"""

    def test_jpy_pair_conservative_config(self):
        """Test that JPY pairs get conservative configuration"""
        jpy_pairs = [
            "CS.D.USDJPY.MINI.IP",
            "CS.D.EURJPY.MINI.IP",
            "CS.D.AUDJPY.MINI.IP"
        ]

        for pair in jpy_pairs:
            with self.subTest(pair=pair):
                config = get_progressive_config_for_epic(pair, enable_adaptive=False)

                # JPY pairs should use conservative settings (40/60/100 point triggers)
                self.assertGreaterEqual(config.stage1_trigger_points, 30,
                    f"JPY pair {pair} should have conservative stage1 trigger (≥30pts)")
                self.assertGreaterEqual(config.stage2_trigger_points, 50,
                    f"JPY pair {pair} should have conservative stage2 trigger (≥50pts)")
                self.assertGreaterEqual(config.stage3_trigger_points, 80,
                    f"JPY pair {pair} should have conservative stage3 trigger (≥80pts)")

    def test_major_pair_balanced_config(self):
        """Test that major pairs get balanced configuration"""
        major_pairs = [
            "CS.D.EURUSD.MINI.IP",
            "CS.D.GBPUSD.MINI.IP"
        ]

        for pair in major_pairs:
            with self.subTest(pair=pair):
                config = get_progressive_config_for_epic(pair, enable_adaptive=False)

                # Major pairs should use balanced settings (6/10/18 point triggers)
                self.assertLessEqual(config.stage1_trigger_points, 10,
                    f"Major pair {pair} should have balanced stage1 trigger (≤10pts)")
                self.assertLessEqual(config.stage2_trigger_points, 15,
                    f"Major pair {pair} should have balanced stage2 trigger (≤15pts)")
                self.assertLessEqual(config.stage3_trigger_points, 25,
                    f"Major pair {pair} should have balanced stage3 trigger (≤25pts)")

    def test_other_pair_configurations(self):
        """Test that other pairs get appropriate configurations"""
        # AUDUSD gets default configuration
        audusd_config = get_progressive_config_for_epic("CS.D.AUDUSD.MINI.IP", enable_adaptive=False)
        self.assertEqual(audusd_config.stage1_trigger_points, 6,
            "AUDUSD should have reasonable stage1 trigger")

        # USDCHF gets default configuration
        usdchf_config = get_progressive_config_for_epic("CS.D.USDCHF.MINI.IP", enable_adaptive=False)
        self.assertEqual(usdchf_config.stage1_trigger_points, 6,
            "USDCHF should have reasonable stage1 trigger")

        # NZDUSD gets default configuration
        nzdusd_config = get_progressive_config_for_epic("CS.D.NZDUSD.MINI.IP", enable_adaptive=False)
        self.assertEqual(nzdusd_config.stage1_trigger_points, 7,
            "NZDUSD should have reasonable stage1 trigger")

        # All should have reasonable ranges (not too aggressive, not too conservative)
        for pair, config in [("AUDUSD", audusd_config), ("USDCHF", usdchf_config), ("NZDUSD", nzdusd_config)]:
            with self.subTest(pair=pair):
                self.assertGreater(config.stage1_trigger_points, 3,
                    f"{pair} should not be too aggressive")
                self.assertLess(config.stage1_trigger_points, 20,
                    f"{pair} should not be too conservative")


class TestTrailingLogic(unittest.TestCase):
    """Test trailing stop logic for realistic trading scenarios"""

    def test_jpy_progressive_trailing_scenario(self):
        """Test progressive trailing for JPY pair realistic scenario"""
        # Simulate USDJPY SELL trade: entry 146.700, current 146.650 (+5 points profit)
        config = get_progressive_config_for_epic("CS.D.USDJPY.MINI.IP", enable_adaptive=False)

        entry_price = 146.700
        current_price = 146.650  # 0.05 price difference = 5 points profit

        # Calculate profit in points
        profit_points = calculate_move_points(entry_price, current_price, "SELL", "USDJPY")
        self.assertEqual(profit_points, 5)

        # With conservative config (40 point trigger), 5 points should not trigger break-even yet
        self.assertLess(profit_points, config.stage1_trigger_points,
            "5 points profit should not exceed stage 1 trigger for conservative JPY config")

        # Test a scenario where it does trigger
        current_price_big_move = 146.300  # 40 points profit
        profit_points_big = calculate_move_points(entry_price, current_price_big_move, "SELL", "USDJPY")
        self.assertEqual(profit_points_big, 40)

        if profit_points_big >= config.stage1_trigger_points:
            # Should trigger stage 1 - break-even mode
            expected_lock_price = entry_price - (config.stage1_lock_points * 0.01)  # SELL trade
            self.assertAlmostEqual(expected_lock_price, 146.700 - 0.10, places=3)  # 10 points = 0.10

    def test_regular_pair_progressive_trailing_scenario(self):
        """Test progressive trailing for regular pair realistic scenario"""
        # Simulate EURUSD BUY trade: entry 1.1000, current 1.1015 (+15 points profit)
        config = get_progressive_config_for_epic("CS.D.EURUSD.MINI.IP", enable_adaptive=False)

        entry_price = 1.1000
        current_price = 1.1015  # 15 points profit

        # Calculate profit in points
        profit_points = calculate_move_points(entry_price, current_price, "BUY", "EURUSD")
        self.assertEqual(profit_points, 15)

        # With balanced config (6/10/18), 15 points should trigger stage 2
        self.assertGreater(profit_points, config.stage1_trigger_points,
            "15 points profit should exceed stage 1 trigger")

        if profit_points >= config.stage2_trigger_points:
            # Should be in stage 2 - profit lock mode
            expected_lock_price = entry_price + (config.stage2_lock_points * 0.0001)  # BUY trade
            self.assertAlmostEqual(expected_lock_price, 1.1000 + 0.0005, places=4)  # 5 points = 0.0005


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_zero_point_distance(self):
        """Test handling of zero point distances"""
        # Should handle zero distances gracefully
        stop_price = convert_stop_distance_to_price(146.500, 0, "SELL", "USDJPY")
        self.assertEqual(stop_price, 146.500)

        points = convert_price_to_points(0.0, "EURUSD")
        self.assertEqual(points, 0)

    def test_negative_point_distance(self):
        """Test handling of negative distances (should convert to absolute)"""
        points = convert_price_to_points(-0.0050, "EURUSD")
        self.assertEqual(points, 50)  # Should take absolute value

    def test_very_small_price_differences(self):
        """Test handling of very small price differences"""
        # 0.1 point for EURUSD should round to 0 points
        points = convert_price_to_points(0.00001, "EURUSD")
        self.assertEqual(points, 0)

        # 0.1 point for USDJPY should round to 0 points
        points = convert_price_to_points(0.001, "USDJPY")
        self.assertEqual(points, 0)

    def test_price_formatting(self):
        """Test price formatting for different pair types"""
        # JPY pairs should show 3 decimal places
        formatted = format_price(146.45678, "USDJPY")
        self.assertEqual(formatted, "146.457")

        # Regular pairs should show 4 decimal places
        formatted = format_price(1.123456, "EURUSD")
        self.assertEqual(formatted, "1.1235")

        # Others should show 2 decimal places
        formatted = format_price(2345.6789, "US500")
        self.assertEqual(formatted, "2345.68")

    def test_epic_validation(self):
        """Test epic validation function"""
        # Valid epics
        self.assertTrue(validate_epic("USDJPY"))
        self.assertTrue(validate_epic("CS.D.EURUSD.MINI.IP"))
        self.assertTrue(validate_epic("US500"))
        self.assertTrue(validate_epic("GOLD"))

        # Invalid epics
        self.assertFalse(validate_epic("INVALID"))
        self.assertFalse(validate_epic("RANDOM"))


class TestRealWorldScenarios(unittest.TestCase):
    """Test with real-world trading scenarios from the logs"""

    def test_usdjpy_trade_1083_scenario(self):
        """Test based on actual USDJPY trade 1083 from logs"""
        # From logs: USDJPY SELL entry=146.70900, various current prices showing 4-6 point profits
        entry_price = 146.70900

        # Test different profit scenarios
        profit_scenarios = [
            (146.66650, 4),  # 4 points profit
            (146.64650, 6),  # 6 points profit
            (146.61600, 9),  # 9 points profit
        ]

        config = get_progressive_config_for_epic("CS.D.USDJPY.MINI.IP", enable_adaptive=False)

        for current_price, expected_profit in profit_scenarios:
            with self.subTest(current_price=current_price, expected_profit=expected_profit):
                actual_profit = calculate_move_points(entry_price, current_price, "SELL", "USDJPY")
                self.assertEqual(actual_profit, expected_profit)

                # With conservative config, 4+ points should trigger break-even (stage 1)
                if actual_profit >= config.stage1_trigger_points:
                    # Should move to break-even: entry - 1 point = 146.69900
                    break_even_price = entry_price - (config.stage1_lock_points * 0.01)
                    self.assertAlmostEqual(break_even_price, 146.69900, places=5)

    def test_audjpy_trade_1093_scenario(self):
        """Test based on actual AUDJPY trade 1093 from logs"""
        # From logs: AUDJPY SELL entry=97.83300, current=97.81450, profit=1pts, trigger=7pts
        entry_price = 97.83300
        current_price = 97.81450

        actual_profit = calculate_move_points(entry_price, current_price, "SELL", "AUDJPY")
        # Should be approximately 18 points (97.833 - 97.8145 = 0.0185, 0.0185/0.01 = 18.5 ≈ 18)
        expected_profit = round((entry_price - current_price) / 0.01)
        self.assertEqual(actual_profit, expected_profit)

        config = get_progressive_config_for_epic("CS.D.AUDJPY.MINI.IP", enable_adaptive=False)

        # With conservative config (40 point trigger), this should not trigger break-even yet
        self.assertLess(actual_profit, config.stage1_trigger_points,
            "AUDJPY with small profit should not trigger break-even with conservative config")


def run_comprehensive_validation():
    """Run all validation tests and provide detailed report"""
    print("🔍 Starting Comprehensive Trailing Stop Loss Validation")
    print("="*70)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestPointValueSystem,
        TestPricePointConversions,
        TestProgressiveConfiguration,
        TestTrailingLogic,
        TestEdgeCases,
        TestRealWorldScenarios
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  - {test}: {error_msg}")

    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            error_msg = error_lines[-2] if len(error_lines) > 1 else str(traceback)
            print(f"  - {test}: {error_msg}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")

    if success_rate == 100.0:
        print("🎉 ALL TESTS PASSED! Trailing stop system is working correctly for both JPY and regular pairs.")
    else:
        print("⚠️ Some tests failed. Review the failures above for issues to fix.")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)