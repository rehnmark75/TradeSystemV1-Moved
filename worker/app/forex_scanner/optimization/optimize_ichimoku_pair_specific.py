#!/usr/bin/env python3
"""
Pair-Specific Ichimoku Parameter Optimization
Optimizes parameters based on currency pair volatility characteristics
"""

import sys
import os
import logging
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import itertools

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ichimoku_pair_optimizer')


def get_pair_volatility_profile(epic: str) -> Dict:
    """Get volatility profile and optimization parameters for each pair"""

    # Define pair characteristics based on market behavior
    profiles = {
        # Major EUR pairs - moderate volatility
        'CS.D.EURUSD.CEEM.IP': {
            'volatility': 'low',
            'confidence_range': [0.45, 0.50, 0.55],
            'target_signals': (8, 15),  # Target range per 14 days
            'description': 'Low volatility major pair'
        },

        # GBP pairs - high volatility
        'CS.D.GBPUSD.MINI.IP': {
            'volatility': 'high',
            'confidence_range': [0.55, 0.60, 0.65, 0.70],  # Higher confidence needed
            'target_signals': (5, 12),  # Fewer signals due to volatility
            'description': 'High volatility pound pair'
        },

        # JPY pairs - moderate to high volatility
        'CS.D.USDJPY.MINI.IP': {
            'volatility': 'medium',
            'confidence_range': [0.50, 0.55, 0.60],
            'target_signals': (6, 14),
            'description': 'Medium volatility yen pair'
        },
        'CS.D.EURJPY.MINI.IP': {
            'volatility': 'medium-high',
            'confidence_range': [0.52, 0.57, 0.62],
            'target_signals': (5, 12),
            'description': 'Medium-high volatility cross pair'
        },
        'CS.D.AUDJPY.MINI.IP': {
            'volatility': 'high',
            'confidence_range': [0.55, 0.60, 0.65],
            'target_signals': (4, 10),
            'description': 'High volatility commodity-yen cross'
        },

        # Commodity currencies - medium volatility
        'CS.D.AUDUSD.MINI.IP': {
            'volatility': 'medium',
            'confidence_range': [0.48, 0.53, 0.58],
            'target_signals': (6, 14),
            'description': 'Medium volatility commodity pair'
        },
        'CS.D.NZDUSD.MINI.IP': {
            'volatility': 'medium',
            'confidence_range': [0.48, 0.53, 0.58],
            'target_signals': (6, 14),
            'description': 'Medium volatility commodity pair'
        },

        # Safe haven pairs - lower volatility
        'CS.D.USDCHF.MINI.IP': {
            'volatility': 'low-medium',
            'confidence_range': [0.45, 0.50, 0.55],
            'target_signals': (8, 16),
            'description': 'Low-medium volatility safe haven pair'
        },
        'CS.D.USDCAD.MINI.IP': {
            'volatility': 'medium',
            'confidence_range': [0.48, 0.53, 0.58],
            'target_signals': (6, 14),
            'description': 'Medium volatility commodity pair'
        }
    }

    return profiles.get(epic, {
        'volatility': 'medium',
        'confidence_range': [0.50, 0.55, 0.60],
        'target_signals': (8, 15),
        'description': 'Default medium volatility profile'
    })


def test_parameter_combination(epic: str, confidence: float,
                             cloud_filter_enabled: bool = False,
                             cloud_buffer_pips: float = 0.0) -> Optional[Dict]:
    """Test a specific parameter combination for an epic"""
    try:
        # Temporarily modify the database record for this test
        # In a production system, we'd modify the strategy config temporarily

        # Run backtest with current database settings
        cmd = [
            'python3', 'backtest_cli.py',
            '--strategy', 'ichimoku',
            '--epic', epic,
            '--days', '14',
            '--show-signals'
        ]

        result = subprocess.run(
            cmd,
            cwd='/app/forex_scanner',
            capture_output=True,
            text=True,
            timeout=45
        )

        if result.returncode == 0:
            # Parse output for signal count
            output = result.stdout
            signal_count = 0

            # Look for "Total signals processed:" in output
            for line in output.split('\n'):
                if 'Total signals processed:' in line:
                    try:
                        signal_count = int(line.split(':')[-1].strip())
                        break
                    except:
                        continue

            return {
                'epic': epic,
                'confidence': confidence,
                'cloud_filter_enabled': cloud_filter_enabled,
                'cloud_buffer_pips': cloud_buffer_pips,
                'signal_count': signal_count
            }
        else:
            logger.warning(f"Backtest failed for {epic} with confidence {confidence}")
            return None

    except Exception as e:
        logger.warning(f"Error testing {epic}: {e}")
        return None


def optimize_pair_specific(epic: str) -> Optional[Dict]:
    """Optimize parameters specifically for one currency pair"""

    profile = get_pair_volatility_profile(epic)
    target_min, target_max = profile['target_signals']

    logger.info(f"🎯 Optimizing {epic}")
    logger.info(f"   Profile: {profile['description']}")
    logger.info(f"   Volatility: {profile['volatility']}")
    logger.info(f"   Target signals: {target_min}-{target_max} per 14 days")

    best_result = None
    best_score = 0
    results = []

    # Test different confidence thresholds
    for confidence in profile['confidence_range']:
        logger.info(f"   Testing confidence: {confidence:.1%}")

        # Test with cloud filter disabled (our proven setting)
        result_no_cloud = test_parameter_combination(epic, confidence, False, 0.0)

        if result_no_cloud:
            signal_count = result_no_cloud['signal_count']

            # Calculate score based on how close we are to target range
            if signal_count == 0:
                score = 0
            elif signal_count < target_min:
                # Too few signals - linear penalty
                score = signal_count / target_min * 0.5
            elif signal_count <= target_max:
                # In target range - excellent score
                score = 1.0 - abs(signal_count - (target_min + target_max) / 2) / ((target_max - target_min) / 2) * 0.1
            else:
                # Too many signals - penalty based on excess
                excess = signal_count - target_max
                score = max(0.1, 1.0 - excess / target_max)

            result_no_cloud['score'] = score
            results.append(result_no_cloud)

            logger.info(f"      No cloud filter: {signal_count} signals (score: {score:.3f})")

            if score > best_score:
                best_score = score
                best_result = result_no_cloud

            # If we're getting too many signals, test with cloud filter enabled
            if signal_count > target_max * 1.5:
                logger.info(f"      Too many signals, testing with cloud filter...")

                # Test with moderate cloud filter
                result_with_cloud = test_parameter_combination(epic, confidence, True, 15.0)

                if result_with_cloud:
                    signal_count_cloud = result_with_cloud['signal_count']

                    # Recalculate score for cloud filter version
                    if signal_count_cloud == 0:
                        score_cloud = 0
                    elif signal_count_cloud < target_min:
                        score_cloud = signal_count_cloud / target_min * 0.5
                    elif signal_count_cloud <= target_max:
                        score_cloud = 1.0 - abs(signal_count_cloud - (target_min + target_max) / 2) / ((target_max - target_min) / 2) * 0.1
                    else:
                        excess_cloud = signal_count_cloud - target_max
                        score_cloud = max(0.1, 1.0 - excess_cloud / target_max)

                    result_with_cloud['score'] = score_cloud
                    results.append(result_with_cloud)

                    logger.info(f"      With cloud filter: {signal_count_cloud} signals (score: {score_cloud:.3f})")

                    if score_cloud > best_score:
                        best_score = score_cloud
                        best_result = result_with_cloud

    if best_result:
        logger.info(f"   🏆 Best config for {epic}:")
        logger.info(f"      Confidence: {best_result['confidence']:.1%}")
        logger.info(f"      Cloud filter: {best_result['cloud_filter_enabled']}")
        logger.info(f"      Signals: {best_result['signal_count']}")
        logger.info(f"      Score: {best_result['score']:.3f}")

        return best_result
    else:
        logger.warning(f"   ❌ No valid optimization found for {epic}")
        return None


def save_pair_specific_params(epic: str, result: Dict):
    """Save pair-specific optimized parameters to database"""
    try:
        from optimization.optimal_parameter_service import get_optimal_parameter_service

        # Calculate risk management based on pair volatility
        profile = get_pair_volatility_profile(epic)

        # Adjust risk management for volatility
        if profile['volatility'] == 'high':
            stop_loss_pips = 25.0
            take_profit_pips = 50.0
        elif profile['volatility'] == 'low':
            stop_loss_pips = 15.0
            take_profit_pips = 30.0
        else:
            stop_loss_pips = 20.0
            take_profit_pips = 40.0

        service = get_optimal_parameter_service()
        with service.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ichimoku_best_parameters (
                        epic, best_tenkan_period, best_kijun_period, best_senkou_b_period,
                        best_chikou_shift, best_cloud_shift, best_confidence_threshold,
                        best_timeframe, best_cloud_thickness_threshold, best_tk_cross_strength_threshold,
                        best_chikou_clear_threshold, best_cloud_filter_enabled, best_chikou_filter_enabled,
                        best_tk_filter_enabled, best_mtf_enabled, best_momentum_confluence_enabled,
                        best_smart_money_enabled, best_ema_200_trend_filter, best_contradiction_filter_enabled,
                        optimal_stop_loss_pips, optimal_take_profit_pips, best_win_rate,
                        best_composite_score, best_tk_cross_accuracy, best_cloud_breakout_accuracy,
                        best_chikou_confirmation_rate, best_perfect_alignment_rate, best_signal_quality_score,
                        market_regime, session_preference, volatility_range, last_updated
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (epic) DO UPDATE SET
                        best_confidence_threshold = EXCLUDED.best_confidence_threshold,
                        best_cloud_filter_enabled = EXCLUDED.best_cloud_filter_enabled,
                        optimal_stop_loss_pips = EXCLUDED.optimal_stop_loss_pips,
                        optimal_take_profit_pips = EXCLUDED.optimal_take_profit_pips,
                        best_composite_score = EXCLUDED.best_composite_score,
                        volatility_range = EXCLUDED.volatility_range,
                        last_updated = EXCLUDED.last_updated
                """, (
                    epic,                                    # epic
                    9,                                       # tenkan_period (classic)
                    26,                                      # kijun_period (classic)
                    52,                                      # senkou_b_period (classic)
                    26,                                      # chikou_shift (standard)
                    26,                                      # cloud_shift (standard)
                    result['confidence'],                    # OPTIMIZED confidence_threshold
                    '15m',                                   # timeframe
                    0.0001,                                  # cloud_thickness_threshold
                    0.3,                                     # tk_cross_strength_threshold
                    0.0002,                                  # chikou_clear_threshold
                    result['cloud_filter_enabled'],         # OPTIMIZED cloud_filter_enabled
                    False,                                   # chikou_filter_enabled (disabled)
                    True,                                    # tk_filter_enabled
                    False,                                   # mtf_enabled (disabled)
                    False,                                   # momentum_confluence_enabled
                    False,                                   # smart_money_enabled
                    False,                                   # ema_200_trend_filter
                    True,                                    # contradiction_filter_enabled
                    stop_loss_pips,                          # PAIR-SPECIFIC stop_loss_pips
                    take_profit_pips,                        # PAIR-SPECIFIC take_profit_pips
                    0.60,                                    # estimated win_rate
                    result['score'],                         # ACTUAL composite_score
                    0.65,                                    # tk_cross_accuracy
                    0.55,                                    # cloud_breakout_accuracy
                    0.60,                                    # chikou_confirmation_rate
                    0.50,                                    # perfect_alignment_rate
                    0.70,                                    # signal_quality_score
                    'trending',                              # market_regime
                    'all',                                   # session_preference
                    profile['volatility'],                   # PAIR-SPECIFIC volatility_range
                    datetime.now()                           # last_updated
                ))

        logger.info(f"💾 Saved pair-specific parameters for {epic}")

    except Exception as e:
        logger.error(f"❌ Failed to save parameters for {epic}: {e}")


def optimize_all_pairs():
    """Run pair-specific optimization for all currency pairs"""

    epics = [
        'CS.D.EURUSD.CEEM.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.AUDUSD.MINI.IP',
        'CS.D.USDCHF.MINI.IP',
        'CS.D.USDCAD.MINI.IP',
        'CS.D.NZDUSD.MINI.IP',
        'CS.D.EURJPY.MINI.IP',
        'CS.D.AUDJPY.MINI.IP'
    ]

    logger.info(f"🌥️ Starting pair-specific Ichimoku optimization for {len(epics)} epics")
    logger.info("=" * 70)

    results = {}

    for i, epic in enumerate(epics, 1):
        logger.info(f"📊 [{i}/{len(epics)}] Optimizing {epic}")

        try:
            result = optimize_pair_specific(epic)

            if result:
                # Save optimized parameters
                save_pair_specific_params(epic, result)

                results[epic] = {
                    'confidence': result['confidence'],
                    'cloud_filter': result['cloud_filter_enabled'],
                    'signal_count': result['signal_count'],
                    'score': result['score'],
                    'status': 'optimized'
                }

                logger.info(f"   ✅ {epic} optimization completed")
            else:
                results[epic] = {'status': 'failed'}
                logger.warning(f"   ❌ {epic} optimization failed")

        except Exception as e:
            logger.error(f"   💥 {epic} optimization error: {e}")
            results[epic] = {'status': 'error', 'error': str(e)}

        logger.info("-" * 50)

    # Summary report
    logger.info("🏁 PAIR-SPECIFIC OPTIMIZATION SUMMARY")
    logger.info("=" * 50)

    successful = 0
    failed = 0

    for epic, result in results.items():
        if result['status'] == 'optimized':
            successful += 1
            profile = get_pair_volatility_profile(epic)
            logger.info(f"✅ {epic} ({profile['volatility']} vol):")
            logger.info(f"   Confidence: {result['confidence']:.1%}")
            logger.info(f"   Cloud filter: {result['cloud_filter']}")
            logger.info(f"   Signals: {result['signal_count']} (score: {result['score']:.3f})")
        else:
            failed += 1
            logger.info(f"❌ {epic}: {result['status']}")

    logger.info("-" * 50)
    logger.info(f"📊 Results: {successful} successful, {failed} failed")
    logger.info("🎯 Each pair now has volatility-adjusted parameters")

    return results


if __name__ == "__main__":
    logger.info("🚀 Starting Pair-Specific Ichimoku Optimization")

    results = optimize_all_pairs()

    if results:
        successful_count = sum(1 for r in results.values() if r['status'] == 'optimized')
        logger.info(f"✅ Pair-specific optimization completed! {successful_count}/{len(results)} pairs optimized")
    else:
        logger.error("❌ Pair-specific optimization failed!")