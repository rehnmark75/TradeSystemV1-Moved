#!/usr/bin/env python3
"""
Ichimoku Parameter Optimization System
Comprehensive optimization engine for Ichimoku Cloud strategy parameters
"""

import sys
import os
import argparse
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import itertools
import psycopg2

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the existing backtest system
from core.backtest_scanner import BacktestScanner
from core.database import DatabaseManager

# Configuration imports
from configdata import config as strategy_config
try:
    import config
except ImportError:
    from forex_scanner import config


class IchimokuParameterOptimizer:
    """
    Advanced Ichimoku parameter optimization engine with comprehensive testing
    """

    def __init__(self, fast_mode: bool = False, smart_presets: bool = False):
        self.logger = logging.getLogger('ichimoku_optimizer')
        self.setup_logging()
        self.db_manager = DatabaseManager(config.DATABASE_URL)

        # Optimization tracking
        self.current_run_id = None
        self.optimization_results = []

        # Mode configuration
        self.fast_mode = fast_mode
        self.smart_presets = smart_presets
        self.parameter_grid = self._get_parameter_grid()

        # Ichimoku-specific settings
        self.min_backtest_days = 60  # Ichimoku needs more data for cloud formation
        self.min_signals_required = 5  # Minimum signals to consider valid

        if fast_mode:
            self.logger.info("🚀 FAST MODE ENABLED: Reduced validation for ultra-fast optimization")
        elif smart_presets:
            self.logger.info("🎯 SMART PRESETS MODE: Using intelligent parameter selection")

    def _get_parameter_grid(self) -> Dict:
        """Get parameter grid based on optimization mode"""

        if self.smart_presets:
            # Smart presets optimized for 15m timeframe based on trading analysis
            return {
                'ichimoku_configs': [
                    {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'name': 'classic'},
                    {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'name': 'faster'},
                    {'tenkan': 12, 'kijun': 30, 'senkou_b': 60, 'name': 'smoother'},
                    {'tenkan': 9, 'kijun': 24, 'senkou_b': 48, 'name': 'balanced'},
                    {'tenkan': 10, 'kijun': 28, 'senkou_b': 56, 'name': 'moderate'}
                ],
                'confidence_levels': [0.45, 0.50, 0.55, 0.60],  # Balanced confidence range
                'timeframes': ['15m'],
                'min_bars_options': [60, 70, 80],  # Ichimoku needs sufficient data
                'cloud_filter_options': [True, False],  # Test with/without cloud filter
                'cloud_buffer_pips': [0, 10, 20, 30],  # Different cloud buffer levels
                'tk_cross_strength_thresholds': [0.1, 0.3, 0.5],  # TK cross validation
                'chikou_filter_options': [True, False],  # Chikou span validation
                'stop_loss_levels': [15, 20, 25],  # Ichimoku stop levels
                'take_profit_levels': [30, 40, 50]  # Ichimoku target levels
            }
        elif self.fast_mode:
            # Fast mode with core Ichimoku parameters
            return {
                'ichimoku_configs': [
                    {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'name': 'classic'},
                    {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'name': 'faster'},
                    {'tenkan': 12, 'kijun': 30, 'senkou_b': 60, 'name': 'smoother'}
                ],
                'confidence_levels': [0.45, 0.50, 0.55],
                'timeframes': ['15m'],
                'min_bars_options': [60, 80],
                'cloud_filter_options': [True, False],
                'cloud_buffer_pips': [0, 20],
                'tk_cross_strength_thresholds': [0.3],
                'chikou_filter_options': [False],  # Disable for simplicity
                'stop_loss_levels': [20],
                'take_profit_levels': [40]
            }
        else:
            # Full grid search
            return {
                'ichimoku_configs': [
                    {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'name': 'fast'},
                    {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'name': 'classic'},
                    {'tenkan': 12, 'kijun': 30, 'senkou_b': 60, 'name': 'smooth'},
                    {'tenkan': 9, 'kijun': 24, 'senkou_b': 48, 'name': 'balanced'},
                    {'tenkan': 10, 'kijun': 28, 'senkou_b': 56, 'name': 'moderate'},
                    {'tenkan': 8, 'kijun': 24, 'senkou_b': 48, 'name': 'responsive'}
                ],
                'confidence_levels': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
                'timeframes': ['15m'],
                'min_bars_options': [60, 70, 80, 90],
                'cloud_filter_options': [True, False],
                'cloud_buffer_pips': [0, 10, 20, 30, 40],
                'tk_cross_strength_thresholds': [0.1, 0.2, 0.3, 0.5],
                'chikou_filter_options': [True, False],
                'stop_loss_levels': [15, 20, 25, 30],
                'take_profit_levels': [30, 40, 50, 60]
            }

    def setup_logging(self):
        """Configure logging for optimization"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'ichimoku_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )

    def optimize_epic(self, epic: str, custom_params: Dict = None) -> Dict:
        """
        Optimize Ichimoku parameters for a specific epic

        Args:
            epic: Trading pair to optimize (e.g., 'CS.D.EURUSD.CEEM.IP')
            custom_params: Optional custom parameter overrides

        Returns:
            Dict with optimization results
        """
        self.logger.info(f"🌥️ Starting Ichimoku optimization for {epic}")

        # Use custom params if provided, otherwise use grid
        param_grid = custom_params if custom_params else self.parameter_grid

        # Calculate total combinations
        total_combinations = (
            len(param_grid['ichimoku_configs']) *
            len(param_grid['confidence_levels']) *
            len(param_grid['min_bars_options']) *
            len(param_grid['cloud_filter_options']) *
            len(param_grid['cloud_buffer_pips']) *
            len(param_grid['tk_cross_strength_thresholds']) *
            len(param_grid['chikou_filter_options']) *
            len(param_grid['stop_loss_levels']) *
            len(param_grid['take_profit_levels'])
        )

        self.logger.info(f"📊 Testing {total_combinations} parameter combinations")

        best_result = None
        best_score = 0
        results = []
        combination_count = 0

        # Start optimization timer
        start_time = time.time()

        # Test all combinations
        for (ichimoku_config, confidence, min_bars, cloud_filter, cloud_buffer,
             tk_strength, chikou_filter, stop_loss, take_profit) in itertools.product(
            param_grid['ichimoku_configs'],
            param_grid['confidence_levels'],
            param_grid['min_bars_options'],
            param_grid['cloud_filter_options'],
            param_grid['cloud_buffer_pips'],
            param_grid['tk_cross_strength_thresholds'],
            param_grid['chikou_filter_options'],
            param_grid['stop_loss_levels'],
            param_grid['take_profit_levels']
        ):
            combination_count += 1

            try:
                # Test this parameter combination
                result = self._test_parameter_combination(
                    epic=epic,
                    tenkan_period=ichimoku_config['tenkan'],
                    kijun_period=ichimoku_config['kijun'],
                    senkou_b_period=ichimoku_config['senkou_b'],
                    confidence_threshold=confidence,
                    min_bars=min_bars,
                    cloud_filter_enabled=cloud_filter,
                    cloud_buffer_pips=cloud_buffer,
                    tk_cross_strength_threshold=tk_strength,
                    chikou_filter_enabled=chikou_filter,
                    stop_loss_pips=stop_loss,
                    take_profit_pips=take_profit
                )

                if result and result['composite_score'] > best_score:
                    best_score = result['composite_score']
                    best_result = result
                    self.logger.info(f"🎯 New best score: {best_score:.4f} for {ichimoku_config['name']}")

                results.append(result)

                # Progress logging
                if combination_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = combination_count / elapsed
                    eta = (total_combinations - combination_count) / rate
                    self.logger.info(
                        f"📊 Progress: {combination_count}/{total_combinations} "
                        f"({combination_count/total_combinations*100:.1f}%) "
                        f"Rate: {rate:.1f}/sec ETA: {eta/60:.1f}min"
                    )

            except Exception as e:
                self.logger.warning(f"⚠️ Error testing combination {combination_count}: {e}")
                continue

        # Save best result to database
        if best_result:
            self._save_best_parameters(epic, best_result)
            self.logger.info(f"✅ Optimization completed! Best score: {best_score:.4f}")
        else:
            self.logger.error("❌ No valid optimization results found")

        return {
            'epic': epic,
            'best_result': best_result,
            'total_tested': combination_count,
            'optimization_time': time.time() - start_time
        }

    def _test_parameter_combination(self, epic: str, **params) -> Optional[Dict]:
        """Test a specific parameter combination"""
        try:
            # Create backtest with these parameters
            backtest = EMABacktest(
                epic=epic,
                timeframe='15m',
                days=self.min_backtest_days,
                strategy_override='ichimoku',
                custom_params={
                    'tenkan_period': params['tenkan_period'],
                    'kijun_period': params['kijun_period'],
                    'senkou_b_period': params['senkou_b_period'],
                    'confidence_threshold': params['confidence_threshold'],
                    'min_bars': params['min_bars'],
                    'cloud_filter_enabled': params['cloud_filter_enabled'],
                    'cloud_buffer_pips': params['cloud_buffer_pips'],
                    'tk_cross_strength_threshold': params['tk_cross_strength_threshold'],
                    'chikou_filter_enabled': params['chikou_filter_enabled'],
                    'stop_loss_pips': params['stop_loss_pips'],
                    'take_profit_pips': params['take_profit_pips']
                }
            )

            # Run backtest
            results = backtest.run()

            if not results or results.get('total_signals', 0) < self.min_signals_required:
                return None

            # Calculate performance metrics
            win_rate = results.get('win_rate', 0)
            total_signals = results.get('total_signals', 0)
            avg_profit = results.get('avg_profit_per_trade', 0)
            risk_reward = params['take_profit_pips'] / params['stop_loss_pips']

            # Calculate composite score
            composite_score = self._calculate_composite_score(
                win_rate, total_signals, avg_profit, risk_reward
            )

            return {
                **params,
                'win_rate': win_rate,
                'total_signals': total_signals,
                'avg_profit_per_trade': avg_profit,
                'risk_reward_ratio': risk_reward,
                'composite_score': composite_score,
                'backtest_results': results
            }

        except Exception as e:
            self.logger.debug(f"Parameter test failed: {e}")
            return None

    def _calculate_composite_score(self, win_rate: float, signal_count: int,
                                 avg_profit: float, risk_reward: float) -> float:
        """Calculate composite optimization score"""
        # Normalize metrics
        win_rate_norm = min(win_rate, 1.0)  # Cap at 100%
        signal_frequency_norm = min(signal_count / 50.0, 1.0)  # Normalize around 50 signals
        profit_norm = max(0, min(avg_profit / 10.0, 1.0))  # Normalize around 10 pip profit
        rr_norm = min(risk_reward / 3.0, 1.0)  # Normalize around 3:1 RR

        # Weighted composite score
        score = (
            win_rate_norm * 0.35 +          # Win rate is crucial
            signal_frequency_norm * 0.25 +   # Need reasonable signal frequency
            profit_norm * 0.25 +             # Profitability matters
            rr_norm * 0.15                   # Risk-reward balance
        )

        return score

    def _save_best_parameters(self, epic: str, result: Dict):
        """Save best parameters to database"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO ichimoku_best_parameters (
                            epic, best_tenkan_period, best_kijun_period, best_senkou_b_period,
                            best_confidence_threshold, best_min_bars, best_cloud_filter_enabled,
                            best_cloud_buffer_pips, best_tk_cross_strength_threshold,
                            best_chikou_filter_enabled, best_win_rate, best_composite_score,
                            optimal_stop_loss_pips, optimal_take_profit_pips,
                            best_total_signals, best_avg_profit_per_trade, last_updated
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (epic) DO UPDATE SET
                            best_tenkan_period = EXCLUDED.best_tenkan_period,
                            best_kijun_period = EXCLUDED.best_kijun_period,
                            best_senkou_b_period = EXCLUDED.best_senkou_b_period,
                            best_confidence_threshold = EXCLUDED.best_confidence_threshold,
                            best_min_bars = EXCLUDED.best_min_bars,
                            best_cloud_filter_enabled = EXCLUDED.best_cloud_filter_enabled,
                            best_cloud_buffer_pips = EXCLUDED.best_cloud_buffer_pips,
                            best_tk_cross_strength_threshold = EXCLUDED.best_tk_cross_strength_threshold,
                            best_chikou_filter_enabled = EXCLUDED.best_chikou_filter_enabled,
                            best_win_rate = EXCLUDED.best_win_rate,
                            best_composite_score = EXCLUDED.best_composite_score,
                            optimal_stop_loss_pips = EXCLUDED.optimal_stop_loss_pips,
                            optimal_take_profit_pips = EXCLUDED.optimal_take_profit_pips,
                            best_total_signals = EXCLUDED.best_total_signals,
                            best_avg_profit_per_trade = EXCLUDED.best_avg_profit_per_trade,
                            last_updated = EXCLUDED.last_updated
                    """, (
                        epic,
                        result['tenkan_period'],
                        result['kijun_period'],
                        result['senkou_b_period'],
                        result['confidence_threshold'],
                        result['min_bars'],
                        result['cloud_filter_enabled'],
                        result['cloud_buffer_pips'],
                        result['tk_cross_strength_threshold'],
                        result['chikou_filter_enabled'],
                        result['win_rate'],
                        result['composite_score'],
                        result['stop_loss_pips'],
                        result['take_profit_pips'],
                        result['total_signals'],
                        result['avg_profit_per_trade'],
                        datetime.now()
                    ))

            self.logger.info(f"💾 Saved optimal parameters for {epic}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save parameters: {e}")


def main():
    """Main optimization script"""
    parser = argparse.ArgumentParser(description='Ichimoku Parameter Optimization')
    parser.add_argument('--epic', default='CS.D.EURUSD.CEEM.IP', help='Epic to optimize')
    parser.add_argument('--fast', action='store_true', help='Fast mode with reduced parameters')
    parser.add_argument('--smart', action='store_true', help='Smart presets mode')
    parser.add_argument('--all-epics', action='store_true', help='Optimize all configured epics')

    args = parser.parse_args()

    # Create optimizer
    optimizer = IchimokuParameterOptimizer(
        fast_mode=args.fast,
        smart_presets=args.smart
    )

    if args.all_epics:
        # Optimize all epics
        epics = getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.CEEM.IP'])
        for epic in epics:
            print(f"\n🌥️ Optimizing {epic}...")
            result = optimizer.optimize_epic(epic)
            if result['best_result']:
                print(f"✅ {epic}: Score {result['best_result']['composite_score']:.4f}")
            else:
                print(f"❌ {epic}: Optimization failed")
    else:
        # Optimize single epic
        result = optimizer.optimize_epic(args.epic)
        if result['best_result']:
            print(f"✅ Optimization completed with score: {result['best_result']['composite_score']:.4f}")
        else:
            print("❌ Optimization failed")


if __name__ == "__main__":
    main()