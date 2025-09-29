#!/usr/bin/env python3
# validation/run_ichimoku_quality_validation.py
"""
Ichimoku Quality Optimization Validation Script
Execute comprehensive validation of quality vs quantity optimization

This script:
1. Loads historical data for testing
2. Runs comparative backtesting using the quality backtest framework
3. Validates statistical significance of improvements
4. Generates comprehensive performance reports
5. Provides actionable recommendations

Usage:
    python run_ichimoku_quality_validation.py [options]
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.ichimoku_quality_backtest import (
    IchimokuQualityBacktest, BacktestConfig, TradeResult, PerformanceMetrics
)
from core.strategies.helpers.ichimoku_statistical_filter import IchimokuStatisticalFilter
from core.strategies.helpers.ichimoku_bayesian_confidence import IchimokuBayesianConfidence
from core.strategies.helpers.ichimoku_adaptive_periods import IchimokuAdaptivePeriods


class IchimokuQualityValidator:
    """Comprehensive validation system for Ichimoku quality optimization"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or self._setup_logger()

        # Initialize optimization components
        self.statistical_filter = IchimokuStatisticalFilter(self.logger)
        self.bayesian_confidence = IchimokuBayesianConfidence(self.logger)
        self.adaptive_periods = IchimokuAdaptivePeriods(self.logger)
        self.backtest_engine = IchimokuQualityBacktest(self.logger)

        # Validation parameters
        self.validation_params = {
            'test_period_months': 6,        # Months of data for testing
            'min_trades_required': 20,      # Minimum trades for valid test
            'significance_level': 0.05,     # Statistical significance threshold
            'improvement_threshold': 0.10,  # Minimum improvement to consider meaningful
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger('IchimokuQualityValidator')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_comprehensive_validation(self, data_sources: Dict[str, str] = None,
                                   output_dir: str = None) -> Dict:
        """
        Run comprehensive validation of Ichimoku quality optimization

        Args:
            data_sources: Dictionary of {epic: data_file_path}
            output_dir: Directory to save validation reports

        Returns:
            Dictionary with comprehensive validation results
        """
        try:
            self.logger.info("=== Starting Ichimoku Quality Optimization Validation ===")

            validation_start_time = datetime.now()

            # Setup output directory
            if output_dir is None:
                output_dir = f"ichimoku_validation_{validation_start_time.strftime('%Y%m%d_%H%M%S')}"

            os.makedirs(output_dir, exist_ok=True)

            # Load and prepare test data
            self.logger.info("Loading and preparing test data...")
            test_data = self._load_test_data(data_sources)

            if not test_data:
                raise ValueError("No valid test data loaded")

            # Setup backtest configuration
            backtest_config = self._create_backtest_config()

            # Run comparative backtest
            self.logger.info("Running comparative backtest...")
            backtest_results = self.backtest_engine.run_comparative_backtest(
                test_data, backtest_config
            )

            # Validate results and generate comprehensive report
            validation_results = self._validate_and_analyze_results(backtest_results)

            # Save results to files
            self._save_validation_results(validation_results, output_dir)

            # Generate executive summary
            summary = self._generate_executive_summary(validation_results)

            validation_end_time = datetime.now()
            validation_duration = (validation_end_time - validation_start_time).total_seconds()

            final_results = {
                'validation_info': {
                    'start_time': validation_start_time.isoformat(),
                    'end_time': validation_end_time.isoformat(),
                    'duration_seconds': validation_duration,
                    'data_sources': len(test_data),
                    'output_directory': output_dir
                },
                'backtest_results': backtest_results,
                'validation_analysis': validation_results,
                'executive_summary': summary,
                'recommendations': self._generate_recommendations(validation_results)
            }

            self.logger.info(f"=== Validation completed in {validation_duration:.1f} seconds ===")
            return final_results

        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _load_test_data(self, data_sources: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
        """Load test data for validation"""
        try:
            test_data = {}

            if data_sources:
                # Load from provided data sources
                for epic, file_path in data_sources.items():
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        df['start_time'] = pd.to_datetime(df['start_time'])
                        test_data[epic] = df
                        self.logger.info(f"Loaded {len(df)} bars for {epic}")
            else:
                # Generate synthetic test data for demonstration
                test_data = self._generate_synthetic_test_data()

            return test_data

        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            return {}

    def _generate_synthetic_test_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic test data for demonstration purposes"""
        try:
            self.logger.info("Generating synthetic test data for demonstration...")

            # Generate data for multiple currency pairs
            currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
            test_data = {}

            for pair in currency_pairs:
                # Generate 6 months of hourly data
                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=180),
                    end=datetime.now(),
                    freq='H'
                )

                # Generate realistic forex price data with trends and volatility
                n_points = len(dates)

                # Base price movement (trending component)
                trend = np.cumsum(np.random.normal(0, 0.0001, n_points))

                # Add volatility clusters
                volatility = np.abs(np.random.normal(0.001, 0.0005, n_points))
                volatility = pd.Series(volatility).rolling(window=24).mean().fillna(volatility)

                # Generate OHLC data
                returns = np.random.normal(0, 1, n_points) * volatility + trend
                prices = 1.1000 + np.cumsum(returns)  # Start around 1.1000 for EUR/USD-like

                # Generate OHLC from price series
                opens = prices
                high_noise = np.abs(np.random.normal(0, volatility/4))
                low_noise = np.abs(np.random.normal(0, volatility/4))
                close_noise = np.random.normal(0, volatility/8)

                highs = prices + high_noise
                lows = prices - low_noise
                closes = prices + close_noise

                # Ensure OHLC relationships are maintained
                highs = np.maximum(highs, np.maximum(opens, closes))
                lows = np.minimum(lows, np.minimum(opens, closes))

                df = pd.DataFrame({
                    'start_time': dates,
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': np.random.uniform(1000, 10000, n_points)
                })

                test_data[pair] = df
                self.logger.info(f"Generated {len(df)} synthetic bars for {pair}")

            return test_data

        except Exception as e:
            self.logger.error(f"Error generating synthetic test data: {e}")
            return {}

    def _create_backtest_config(self) -> BacktestConfig:
        """Create backtest configuration"""
        return BacktestConfig(
            start_date=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            initial_capital=10000.0,
            position_size=0.02,
            transaction_costs=0.0002,
            slippage=0.0001,
            max_concurrent_trades=3,
            risk_free_rate=0.02
        )

    def _validate_and_analyze_results(self, backtest_results: Dict) -> Dict:
        """Validate and analyze backtest results"""
        try:
            validation_analysis = {}

            # Extract key results
            qty_results = backtest_results.get('quantity_results', {})
            qual_results = backtest_results.get('quality_results', {})
            comparison = backtest_results.get('comparison_analysis', {})
            stats_tests = backtest_results.get('statistical_tests', {})

            # Validate result quality
            validation_analysis['data_quality'] = self._validate_data_quality(
                qty_results, qual_results
            )

            # Analyze performance improvements
            validation_analysis['performance_analysis'] = self._analyze_performance_improvements(
                comparison
            )

            # Validate statistical significance
            validation_analysis['significance_analysis'] = self._analyze_statistical_significance(
                stats_tests
            )

            # Risk-adjusted performance analysis
            validation_analysis['risk_analysis'] = self._analyze_risk_adjustments(
                qty_results, qual_results
            )

            # Signal quality analysis
            validation_analysis['signal_quality'] = self._analyze_signal_quality_improvements(
                qty_results, qual_results
            )

            # Regime-specific performance
            validation_analysis['regime_performance'] = self._analyze_regime_effectiveness(
                backtest_results.get('regime_analysis', {})
            )

            return validation_analysis

        except Exception as e:
            self.logger.error(f"Error validating and analyzing results: {e}")
            return {'error': str(e)}

    def _validate_data_quality(self, qty_results: Dict, qual_results: Dict) -> Dict:
        """Validate the quality of backtest data and results"""
        try:
            qty_trades = len(qty_results.get('portfolio', {}).get('trades', []))
            qual_trades = len(qual_results.get('portfolio', {}).get('trades', []))

            min_required = self.validation_params['min_trades_required']

            validation = {
                'quantity_trade_count': qty_trades,
                'quality_trade_count': qual_trades,
                'minimum_required': min_required,
                'quantity_sufficient': qty_trades >= min_required,
                'quality_sufficient': qual_trades >= min_required,
                'both_sufficient': qty_trades >= min_required and qual_trades >= min_required,
                'data_quality_score': 0.0
            }

            # Calculate data quality score
            if validation['both_sufficient']:
                validation['data_quality_score'] = 1.0
            elif validation['quantity_sufficient'] or validation['quality_sufficient']:
                validation['data_quality_score'] = 0.6
            else:
                validation['data_quality_score'] = 0.3

            validation['validation_status'] = (
                'PASS' if validation['both_sufficient'] else
                'MARGINAL' if validation['data_quality_score'] >= 0.6 else
                'FAIL'
            )

            return validation

        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return {'error': str(e)}

    def _analyze_performance_improvements(self, comparison: Dict) -> Dict:
        """Analyze performance improvements from quality optimization"""
        try:
            perf_comp = comparison.get('performance_comparison', {})

            improvements = {}
            threshold = self.validation_params['improvement_threshold']

            for metric in ['win_rate', 'total_return', 'sharpe_ratio', 'profit_factor']:
                metric_data = perf_comp.get(metric, {})
                change_pct = metric_data.get('change_pct', 0) / 100.0  # Convert to decimal

                improvements[metric] = {
                    'change_pct': change_pct,
                    'meets_threshold': abs(change_pct) >= threshold,
                    'direction': 'improvement' if change_pct > 0 else 'degradation' if change_pct < 0 else 'neutral',
                    'magnitude': 'significant' if abs(change_pct) >= threshold else 'marginal'
                }

            # Overall improvement score
            improvement_count = sum(1 for m in improvements.values() if m['direction'] == 'improvement')
            significant_count = sum(1 for m in improvements.values() if m['magnitude'] == 'significant')

            overall_score = (improvement_count + significant_count) / (len(improvements) * 2)

            return {
                'individual_metrics': improvements,
                'overall_improvement_score': overall_score,
                'improvement_summary': f"{improvement_count}/{len(improvements)} metrics improved",
                'significance_summary': f"{significant_count}/{len(improvements)} changes significant"
            }

        except Exception as e:
            self.logger.error(f"Error analyzing performance improvements: {e}")
            return {'error': str(e)}

    def _analyze_statistical_significance(self, stats_tests: Dict) -> Dict:
        """Analyze statistical significance of improvements"""
        try:
            significance_level = self.validation_params['significance_level']

            analysis = {}

            for test_name, test_results in stats_tests.items():
                if isinstance(test_results, dict) and 'p_value' in test_results:
                    p_value = test_results['p_value']
                    significant = p_value < significance_level

                    analysis[test_name] = {
                        'p_value': p_value,
                        'significant': significant,
                        'confidence_level': (1 - p_value) * 100,
                        'interpretation': test_results.get('interpretation', 'No interpretation available')
                    }

            # Overall significance assessment
            significant_tests = sum(1 for t in analysis.values() if t['significant'])
            total_tests = len(analysis)

            overall_significance = {
                'significant_tests': significant_tests,
                'total_tests': total_tests,
                'significance_ratio': significant_tests / total_tests if total_tests > 0 else 0,
                'overall_assessment': (
                    'HIGHLY_SIGNIFICANT' if significant_tests == total_tests else
                    'MODERATELY_SIGNIFICANT' if significant_tests >= total_tests / 2 else
                    'NOT_SIGNIFICANT'
                )
            }

            return {
                'individual_tests': analysis,
                'overall_significance': overall_significance
            }

        except Exception as e:
            self.logger.error(f"Error analyzing statistical significance: {e}")
            return {'error': str(e)}

    def _analyze_risk_adjustments(self, qty_results: Dict, qual_results: Dict) -> Dict:
        """Analyze risk-adjusted performance improvements"""
        try:
            qty_perf = qty_results.get('performance')
            qual_perf = qual_results.get('performance')

            if not qty_perf or not qual_perf:
                return {'error': 'Missing performance data'}

            risk_analysis = {
                'sharpe_ratio_improvement': qual_perf.sharpe_ratio - qty_perf.sharpe_ratio,
                'volatility_change': qual_perf.volatility - qty_perf.volatility,
                'max_drawdown_improvement': qty_perf.max_drawdown - qual_perf.max_drawdown,
                'risk_adjusted_return_ratio': (
                    qual_perf.total_return / qual_perf.volatility if qual_perf.volatility > 0 else 0
                ) / (
                    qty_perf.total_return / qty_perf.volatility if qty_perf.volatility > 0 else 1
                ) if qty_perf.volatility > 0 else 1
            }

            # Risk improvement score
            improvements = 0
            if risk_analysis['sharpe_ratio_improvement'] > 0:
                improvements += 1
            if risk_analysis['volatility_change'] < 0:  # Lower volatility is better
                improvements += 1
            if risk_analysis['max_drawdown_improvement'] > 0:  # Less drawdown is better
                improvements += 1
            if risk_analysis['risk_adjusted_return_ratio'] > 1:
                improvements += 1

            risk_analysis['risk_improvement_score'] = improvements / 4
            risk_analysis['risk_assessment'] = (
                'EXCELLENT' if improvements >= 3 else
                'GOOD' if improvements >= 2 else
                'MARGINAL' if improvements >= 1 else
                'POOR'
            )

            return risk_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing risk adjustments: {e}")
            return {'error': str(e)}

    def _analyze_signal_quality_improvements(self, qty_results: Dict, qual_results: Dict) -> Dict:
        """Analyze signal quality improvements"""
        try:
            qty_trades = qty_results.get('portfolio', {}).get('trades', [])
            qual_trades = qual_results.get('portfolio', {}).get('trades', [])

            if not qty_trades or not qual_trades:
                return {'error': 'Insufficient trade data'}

            # Signal efficiency metrics
            qty_win_rate = len([t for t in qty_trades if t.pnl > 0]) / len(qty_trades)
            qual_win_rate = len([t for t in qual_trades if t.pnl > 0]) / len(qual_trades)

            qty_avg_trade = sum(t.pnl for t in qty_trades) / len(qty_trades)
            qual_avg_trade = sum(t.pnl for t in qual_trades) / len(qual_trades)

            signal_analysis = {
                'trade_reduction': len(qty_trades) - len(qual_trades),
                'trade_reduction_pct': (len(qty_trades) - len(qual_trades)) / len(qty_trades) * 100,
                'win_rate_improvement': qual_win_rate - qty_win_rate,
                'avg_trade_improvement': qual_avg_trade - qty_avg_trade,
                'signal_efficiency_ratio': (qual_win_rate * qual_avg_trade) / (qty_win_rate * qty_avg_trade) if qty_win_rate > 0 and qty_avg_trade != 0 else 1,
                'quality_score': 0.0
            }

            # Calculate overall quality score
            quality_factors = []

            # Reward trade reduction (fewer, better trades)
            if signal_analysis['trade_reduction'] > 0:
                quality_factors.append(0.25)

            # Reward win rate improvement
            if signal_analysis['win_rate_improvement'] > 0:
                quality_factors.append(0.25)

            # Reward average trade improvement
            if signal_analysis['avg_trade_improvement'] > 0:
                quality_factors.append(0.25)

            # Reward efficiency improvement
            if signal_analysis['signal_efficiency_ratio'] > 1:
                quality_factors.append(0.25)

            signal_analysis['quality_score'] = sum(quality_factors)
            signal_analysis['quality_assessment'] = (
                'EXCELLENT' if signal_analysis['quality_score'] >= 0.75 else
                'GOOD' if signal_analysis['quality_score'] >= 0.50 else
                'MARGINAL' if signal_analysis['quality_score'] >= 0.25 else
                'POOR'
            )

            return signal_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing signal quality improvements: {e}")
            return {'error': str(e)}

    def _analyze_regime_effectiveness(self, regime_analysis: Dict) -> Dict:
        """Analyze effectiveness across different market regimes"""
        try:
            if not regime_analysis:
                return {'error': 'No regime analysis data available'}

            # Placeholder for regime analysis
            # In a real implementation, this would analyze performance
            # across trending/ranging/volatile market conditions

            return {
                'regime_adaptability': 'analysis_placeholder',
                'best_performing_regime': 'unknown',
                'worst_performing_regime': 'unknown',
                'regime_consistency_score': 0.5
            }

        except Exception as e:
            self.logger.error(f"Error analyzing regime effectiveness: {e}")
            return {'error': str(e)}

    def _generate_executive_summary(self, validation_results: Dict) -> str:
        """Generate executive summary of validation results"""
        try:
            performance = validation_results.get('performance_analysis', {})
            significance = validation_results.get('significance_analysis', {})
            risk = validation_results.get('risk_analysis', {})
            signal_quality = validation_results.get('signal_quality', {})

            summary = f"""
ICHIMOKU QUALITY OPTIMIZATION VALIDATION SUMMARY
================================================

OVERALL ASSESSMENT: {self._get_overall_assessment(validation_results)}

KEY FINDINGS:
• Performance Improvements: {performance.get('improvement_summary', 'N/A')}
• Statistical Significance: {significance.get('overall_significance', {}).get('overall_assessment', 'N/A')}
• Risk Adjustment: {risk.get('risk_assessment', 'N/A')}
• Signal Quality: {signal_quality.get('quality_assessment', 'N/A')}

PERFORMANCE METRICS:
• Signal Efficiency Ratio: {signal_quality.get('signal_efficiency_ratio', 'N/A'):.2f}
• Trade Reduction: {signal_quality.get('trade_reduction_pct', 'N/A'):.1f}%
• Win Rate Improvement: {signal_quality.get('win_rate_improvement', 'N/A'):.1%}
• Risk Improvement Score: {risk.get('risk_improvement_score', 'N/A'):.2f}

STATISTICAL VALIDATION:
• Significant Tests: {significance.get('overall_significance', {}).get('significant_tests', 'N/A')}/{significance.get('overall_significance', {}).get('total_tests', 'N/A')}
• Overall Improvement Score: {performance.get('overall_improvement_score', 'N/A'):.2f}

RECOMMENDATION: {self._get_recommendation(validation_results)}
"""

            return summary

        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return f"Error generating summary: {str(e)}"

    def _get_overall_assessment(self, validation_results: Dict) -> str:
        """Get overall assessment of optimization effectiveness"""
        try:
            scores = []

            # Collect all available scores
            if 'performance_analysis' in validation_results:
                scores.append(validation_results['performance_analysis'].get('overall_improvement_score', 0))

            if 'risk_analysis' in validation_results:
                scores.append(validation_results['risk_analysis'].get('risk_improvement_score', 0))

            if 'signal_quality' in validation_results:
                scores.append(validation_results['signal_quality'].get('quality_score', 0))

            if not scores:
                return 'INSUFFICIENT_DATA'

            overall_score = sum(scores) / len(scores)

            if overall_score >= 0.75:
                return 'HIGHLY_SUCCESSFUL'
            elif overall_score >= 0.50:
                return 'SUCCESSFUL'
            elif overall_score >= 0.25:
                return 'MARGINAL_IMPROVEMENT'
            else:
                return 'NO_SIGNIFICANT_IMPROVEMENT'

        except Exception as e:
            self.logger.error(f"Error getting overall assessment: {e}")
            return 'ERROR'

    def _get_recommendation(self, validation_results: Dict) -> str:
        """Get recommendation based on validation results"""
        try:
            assessment = self._get_overall_assessment(validation_results)

            recommendations = {
                'HIGHLY_SUCCESSFUL': 'IMPLEMENT IMMEDIATELY - Quality optimization shows excellent results across all metrics.',
                'SUCCESSFUL': 'IMPLEMENT - Quality optimization shows good improvements with acceptable trade-offs.',
                'MARGINAL_IMPROVEMENT': 'CONSIDER IMPLEMENTATION - Some improvements shown, evaluate based on specific requirements.',
                'NO_SIGNIFICANT_IMPROVEMENT': 'DO NOT IMPLEMENT - No meaningful improvements detected.',
                'INSUFFICIENT_DATA': 'RERUN WITH MORE DATA - Insufficient data for reliable recommendation.',
                'ERROR': 'INVESTIGATE ERRORS - Technical issues prevent reliable assessment.'
            }

            return recommendations.get(assessment, 'MANUAL REVIEW REQUIRED')

        except Exception as e:
            self.logger.error(f"Error getting recommendation: {e}")
            return 'ERROR GENERATING RECOMMENDATION'

    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate specific recommendations based on validation results"""
        recommendations = []

        try:
            # Based on performance analysis
            performance = validation_results.get('performance_analysis', {})
            if performance.get('overall_improvement_score', 0) >= 0.5:
                recommendations.append("Implement quality-focused configuration for improved performance")

            # Based on risk analysis
            risk = validation_results.get('risk_analysis', {})
            if risk.get('risk_improvement_score', 0) >= 0.5:
                recommendations.append("Quality optimization successfully reduces risk while maintaining returns")

            # Based on signal quality
            signal_quality = validation_results.get('signal_quality', {})
            if signal_quality.get('trade_reduction_pct', 0) > 0:
                recommendations.append(f"Expect {signal_quality.get('trade_reduction_pct', 0):.1f}% reduction in trade frequency")

            # Based on statistical significance
            significance = validation_results.get('significance_analysis', {})
            if significance.get('overall_significance', {}).get('overall_assessment') == 'HIGHLY_SIGNIFICANT':
                recommendations.append("Changes are statistically significant and reliable")

            if not recommendations:
                recommendations.append("No clear recommendations - consider further optimization or different approaches")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations - manual review required"]

    def _save_validation_results(self, validation_results: Dict, output_dir: str):
        """Save validation results to files"""
        try:
            # Save main results as JSON
            results_file = os.path.join(output_dir, 'validation_results.json')
            with open(results_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)

            # Save summary report
            summary_file = os.path.join(output_dir, 'executive_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(validation_results.get('executive_summary', 'No summary available'))

            # Save recommendations
            rec_file = os.path.join(output_dir, 'recommendations.txt')
            with open(rec_file, 'w') as f:
                recommendations = validation_results.get('recommendations', [])
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")

            self.logger.info(f"Validation results saved to {output_dir}")

        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")


def main():
    """Main function for running validation"""
    parser = argparse.ArgumentParser(description='Run Ichimoku Quality Optimization Validation')
    parser.add_argument('--data-dir', help='Directory containing test data files')
    parser.add_argument('--output-dir', help='Directory to save validation results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Initialize validator
    validator = IchimokuQualityValidator()

    # Run validation
    print("Starting Ichimoku Quality Optimization Validation...")
    print("=" * 60)

    results = validator.run_comprehensive_validation(
        data_sources=None,  # Use synthetic data for demonstration
        output_dir=args.output_dir
    )

    if 'error' in results:
        print(f"Validation failed: {results['error']}")
        return 1

    # Print executive summary
    print("\nEXECUTIVE SUMMARY:")
    print("=" * 60)
    print(results.get('executive_summary', 'No summary available'))

    # Print recommendations
    print("\nRECOMMENDATIONS:")
    print("=" * 60)
    recommendations = results.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print(f"\nDetailed results saved to: {results['validation_info']['output_directory']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())