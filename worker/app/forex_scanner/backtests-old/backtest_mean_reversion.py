#!/usr/bin/env python3
"""
Mean Reversion Strategy Backtest with Multi-Oscillator Confluence Analysis
Run: python backtest_mean_reversion.py --epic CS.D.EURUSD.CEEM.IP --days 7 --timeframe 15m

FEATURES:
- Multi-oscillator confluence approach based on RAG analysis findings
- LuxAlgo Premium Oscillator (primary mean reversion engine)
- Multi-timeframe RSI analysis for confluence confirmation
- RSI-EMA divergence detection for reversal pattern identification
- Squeeze Momentum Indicator for timing optimization
- Mean reversion zone validation with statistical support/resistance
- Market regime filtering to avoid inappropriate market conditions
- Database optimization parameter integration
- Enhanced signal validation and performance analysis
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))

# If we're in backtests/ subdirectory, go up one level to project root
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    # If we're in project root
    project_root = script_dir

sys.path.insert(0, project_root)

try:
    from backtests.backtest_base import BacktestBase
    from core.strategies.mean_reversion_strategy import MeanReversionStrategy, create_mean_reversion_strategy
except ImportError:
    from forex_scanner.backtests.backtest_base import BacktestBase
    from forex_scanner.core.strategies.mean_reversion_strategy import MeanReversionStrategy, create_mean_reversion_strategy

try:
    import config
except ImportError:
    from forex_scanner import config


class MeanReversionBacktest(BacktestBase):
    """Mean Reversion Strategy Backtesting with Multi-Oscillator Confluence Analysis"""

    def __init__(self, use_optimal_parameters: bool = True):
        super().__init__('mean_reversion', use_optimal_parameters)
        self.logger = logging.getLogger('mean_reversion_backtest')
        self.setup_logging()

        # Performance tracking for mean reversion specific metrics
        self.mr_performance_stats = {
            'oscillator_confluence_signals': 0,
            'divergence_signals': 0,
            'zone_validation_passes': 0,
            'regime_filter_passes': 0,
            'extreme_oscillator_signals': 0
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def initialize_strategy(self, epic: str = None) -> MeanReversionStrategy:
        """Initialize the mean reversion strategy with optimal parameters"""
        try:
            # Create strategy with optimal parameters and enhanced validation
            strategy = create_mean_reversion_strategy(
                data_fetcher=self.data_fetcher,
                backtest_mode=True,
                epic=epic,
                timeframe='15m',
                use_optimized_parameters=self.use_optimal_parameters,
                pipeline_mode=True  # Enable our enhanced validation logic
            )

            if self.use_optimal_parameters:
                self.logger.info(f"✅ Mean Reversion strategy initialized with optimal parameters for {epic}")
            else:
                self.logger.info(f"📊 Mean Reversion strategy initialized with default parameters for {epic}")

            return strategy

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Mean Reversion strategy: {e}")
            raise

    def run_strategy_backtest(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """
        Run mean reversion strategy backtest on given data

        Args:
            df: DataFrame with price data and indicators
            epic: Trading pair epic
            spread_pips: Spread in pips
            timeframe: Trading timeframe

        Returns:
            List of signal dictionaries
        """
        try:
            self.logger.info(f"🎯 Running Mean Reversion backtest for {epic}")
            self.logger.info(f"   Data points: {len(df)}")
            self.logger.info(f"   Timeframe: {timeframe}")
            self.logger.info(f"   Spread: {spread_pips} pips")

            # Initialize strategy for this epic
            strategy = self.initialize_strategy(epic)

            # Detect signals using strategy
            signal = strategy.detect_signal(
                df=df,
                epic=epic,
                spread_pips=spread_pips,
                timeframe=timeframe
            )

            signals = []
            if signal:
                # Enhance signal with backtest-specific information
                enhanced_signal = self._enhance_signal_for_backtest(signal, df, epic)
                signals.append(enhanced_signal)

                # Update performance statistics
                self._update_performance_stats(enhanced_signal)

                self.logger.info(f"✅ Found Mean Reversion signal: {signal['signal_type']} at "
                               f"{signal.get('price', 'N/A')} (confidence: {signal.get('confidence', 0):.1%})")
            else:
                self.logger.debug(f"   No signals detected for {epic}")

            return signals

        except Exception as e:
            self.logger.error(f"❌ Mean Reversion backtest failed for {epic}: {e}")
            return []

    def _enhance_signal_for_backtest(self, signal: Dict, df: pd.DataFrame, epic: str) -> Dict:
        """
        Enhance signal with additional backtest analysis information

        Args:
            signal: Original signal dictionary
            df: DataFrame with price data
            epic: Trading pair epic

        Returns:
            Enhanced signal dictionary
        """
        try:
            enhanced_signal = signal.copy()

            # Add timestamp if available
            if 'signal_time' not in enhanced_signal and len(df) > 0:
                # Try to get timestamp from the last row
                last_row = df.iloc[-1]
                for time_col in ['start_time', 'datetime', 'timestamp']:
                    if time_col in last_row:
                        enhanced_signal['signal_time'] = last_row[time_col]
                        break

            # Add mean reversion specific analysis
            enhanced_signal['mean_reversion_analysis'] = self._analyze_mean_reversion_conditions(signal, df)

            # Add oscillator confluence details
            enhanced_signal['oscillator_confluence'] = {
                'luxalgo_oscillator': signal.get('luxalgo_oscillator', 50),
                'bull_score': signal.get('oscillator_bull_score', 0),
                'bear_score': signal.get('oscillator_bear_score', 0),
                'mtf_alignment': signal.get('mtf_bull_alignment', 0) if signal.get('signal_type') == 'BULL'
                                else signal.get('mtf_bear_alignment', 0),
                'divergence_present': signal.get('rsi_ema_divergence_bull', False) or signal.get('rsi_ema_divergence_bear', False),
                'divergence_strength': signal.get('divergence_strength', 0),
                'squeeze_momentum': signal.get('squeeze_momentum', 0),
                'squeeze_active': signal.get('squeeze_on', False)
            }

            # Add execution guidance analysis
            execution_guidance = signal.get('execution_guidance', {})
            enhanced_signal['execution_analysis'] = {
                'market_regime': execution_guidance.get('market_regime', 'unknown'),
                'oscillator_extremity': execution_guidance.get('oscillator_extremity', 'unknown'),
                'risk_level': execution_guidance.get('risk_level', 'medium'),
                'recommended_position_size': execution_guidance.get('recommended_position_size', 'medium')
            }

            # Add performance predictions based on signal characteristics
            enhanced_signal['performance_prediction'] = self._predict_signal_performance(enhanced_signal)

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"Error enhancing signal for backtest: {e}")
            return signal

    def _analyze_mean_reversion_conditions(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Analyze mean reversion market conditions"""
        try:
            if len(df) == 0:
                return {'error': 'No data available'}

            latest_row = df.iloc[-1]

            # Analyze market structure for mean reversion suitability
            analysis = {
                'price_position': self._analyze_price_position(df),
                'volatility_state': self._analyze_volatility_state(df),
                'trend_strength': self._analyze_trend_strength(latest_row),
                'oscillator_readings': self._analyze_oscillator_readings(latest_row),
                'mean_reversion_probability': 0.0
            }

            # Calculate mean reversion probability
            probability_factors = []

            # Factor 1: Oscillator extremity (higher extremity = higher reversion probability)
            luxalgo_osc = latest_row.get('luxalgo_oscillator', 50)
            if signal.get('signal_type') == 'BULL':
                if luxalgo_osc < 10:
                    probability_factors.append(0.8)  # Very high probability
                elif luxalgo_osc < 20:
                    probability_factors.append(0.6)  # High probability
                else:
                    probability_factors.append(0.4)  # Moderate probability
            else:  # BEAR
                if luxalgo_osc > 90:
                    probability_factors.append(0.8)  # Very high probability
                elif luxalgo_osc > 80:
                    probability_factors.append(0.6)  # High probability
                else:
                    probability_factors.append(0.4)  # Moderate probability

            # Factor 2: Market regime (ranging markets favor mean reversion)
            adx = latest_row.get('adx', 25)
            if adx < 20:
                probability_factors.append(0.7)  # Ranging market
            elif adx < 30:
                probability_factors.append(0.5)  # Moderate trend
            else:
                probability_factors.append(0.3)  # Strong trend

            # Factor 3: Divergence presence (increases reversion probability)
            if signal.get('divergence_strength', 0) > 0.5:
                probability_factors.append(0.7)
            else:
                probability_factors.append(0.4)

            # Calculate weighted average
            analysis['mean_reversion_probability'] = sum(probability_factors) / len(probability_factors)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing mean reversion conditions: {e}")
            return {'error': str(e)}

    def _analyze_price_position(self, df: pd.DataFrame) -> str:
        """Analyze price position relative to recent range"""
        try:
            if len(df) < 20:
                return 'insufficient_data'

            recent_data = df.tail(20)
            high_range = recent_data['high'].max()
            low_range = recent_data['low'].min()
            current_price = df.iloc[-1]['close']

            range_size = high_range - low_range
            if range_size == 0:
                return 'no_range'

            position_pct = (current_price - low_range) / range_size

            if position_pct > 0.8:
                return 'top_of_range'
            elif position_pct > 0.6:
                return 'upper_range'
            elif position_pct > 0.4:
                return 'middle_range'
            elif position_pct > 0.2:
                return 'lower_range'
            else:
                return 'bottom_of_range'

        except Exception:
            return 'analysis_error'

    def _analyze_volatility_state(self, df: pd.DataFrame) -> str:
        """Analyze current volatility state"""
        try:
            if 'atr' in df.columns and len(df) >= 20:
                current_atr = df.iloc[-1]['atr']
                avg_atr = df['atr'].tail(20).mean()

                if current_atr > avg_atr * 1.5:
                    return 'high_volatility'
                elif current_atr < avg_atr * 0.7:
                    return 'low_volatility'
                else:
                    return 'normal_volatility'
            else:
                return 'volatility_unknown'

        except Exception:
            return 'volatility_error'

    def _analyze_trend_strength(self, row: pd.Series) -> str:
        """Analyze trend strength from ADX"""
        try:
            adx = row.get('adx', 25)

            if adx > 50:
                return 'very_strong_trend'
            elif adx > 30:
                return 'strong_trend'
            elif adx > 20:
                return 'moderate_trend'
            else:
                return 'weak_trend_ranging'

        except Exception:
            return 'trend_unknown'

    def _analyze_oscillator_readings(self, row: pd.Series) -> Dict:
        """Analyze oscillator readings for confluence"""
        try:
            return {
                'luxalgo_oscillator': row.get('luxalgo_oscillator', 50),
                'rsi_14': row.get('rsi_14', 50),
                'mtf_bull_alignment': row.get('mtf_bull_alignment', 0),
                'mtf_bear_alignment': row.get('mtf_bear_alignment', 0),
                'squeeze_momentum': row.get('squeeze_momentum', 0),
                'squeeze_on': row.get('squeeze_on', False)
            }

        except Exception:
            return {'error': 'oscillator_analysis_failed'}

    def _predict_signal_performance(self, signal: Dict) -> Dict:
        """Predict signal performance based on characteristics"""
        try:
            # Start with base expectation
            win_probability = 0.6  # Base mean reversion win rate

            # Adjust based on signal characteristics
            confluence = signal.get('oscillator_confluence', {})
            execution = signal.get('execution_analysis', {})
            mr_analysis = signal.get('mean_reversion_analysis', {})

            # Factor 1: Oscillator confluence strength
            if signal.get('signal_type') == 'BULL':
                confluence_score = confluence.get('bull_score', 0)
            else:
                confluence_score = confluence.get('bear_score', 0)

            if confluence_score > 0.8:
                win_probability += 0.15
            elif confluence_score > 0.65:
                win_probability += 0.1
            elif confluence_score < 0.5:
                win_probability -= 0.1

            # Factor 2: Divergence presence
            if confluence.get('divergence_present', False):
                divergence_strength = confluence.get('divergence_strength', 0)
                win_probability += divergence_strength * 0.15

            # Factor 3: Market regime
            regime = execution.get('market_regime', 'unknown')
            if regime == 'ranging':
                win_probability += 0.1
            elif regime == 'strong_trend':
                win_probability -= 0.15

            # Factor 4: Mean reversion probability
            mr_probability = mr_analysis.get('mean_reversion_probability', 0.5)
            win_probability += (mr_probability - 0.5) * 0.2

            # Factor 5: Risk level
            risk_level = execution.get('risk_level', 'medium')
            if risk_level == 'low':
                win_probability += 0.05
            elif risk_level == 'high':
                win_probability -= 0.1

            # Clamp to reasonable bounds
            win_probability = max(0.3, min(0.85, win_probability))

            return {
                'predicted_win_probability': win_probability,
                'confidence_category': self._categorize_confidence(win_probability),
                'factors_analyzed': {
                    'confluence_score': confluence_score,
                    'divergence_present': confluence.get('divergence_present', False),
                    'market_regime': regime,
                    'mr_probability': mr_probability,
                    'risk_level': risk_level
                }
            }

        except Exception as e:
            return {'error': str(e), 'predicted_win_probability': 0.6}

    def _categorize_confidence(self, win_probability: float) -> str:
        """Categorize win probability into confidence levels"""
        if win_probability >= 0.75:
            return 'very_high'
        elif win_probability >= 0.65:
            return 'high'
        elif win_probability >= 0.55:
            return 'medium'
        elif win_probability >= 0.45:
            return 'low'
        else:
            return 'very_low'

    def _update_performance_stats(self, signal: Dict):
        """Update mean reversion specific performance statistics"""
        try:
            # Track oscillator confluence signals
            confluence = signal.get('oscillator_confluence', {})
            if confluence.get('bull_score', 0) > 0.65 or confluence.get('bear_score', 0) > 0.65:
                self.mr_performance_stats['oscillator_confluence_signals'] += 1

            # Track divergence signals
            if confluence.get('divergence_present', False):
                self.mr_performance_stats['divergence_signals'] += 1

            # Track extreme oscillator readings
            luxalgo_osc = confluence.get('luxalgo_oscillator', 50)
            if luxalgo_osc < 15 or luxalgo_osc > 85:
                self.mr_performance_stats['extreme_oscillator_signals'] += 1

            # Track regime filtering
            execution = signal.get('execution_analysis', {})
            if execution.get('market_regime') in ['ranging', 'weak_trend_ranging']:
                self.mr_performance_stats['regime_filter_passes'] += 1

        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")

    def display_mean_reversion_performance(self):
        """Display mean reversion specific performance metrics"""
        self.logger.info(f"\n📊 Mean Reversion Strategy Performance Details:")
        self.logger.info(f"   Oscillator Confluence Signals: {self.mr_performance_stats['oscillator_confluence_signals']}")
        self.logger.info(f"   Divergence Signals: {self.mr_performance_stats['divergence_signals']}")
        self.logger.info(f"   Extreme Oscillator Signals: {self.mr_performance_stats['extreme_oscillator_signals']}")
        self.logger.info(f"   Regime Filter Passes: {self.mr_performance_stats['regime_filter_passes']}")

        # Calculate percentages if we have signals
        total_signals = sum(self.mr_performance_stats.values())
        if total_signals > 0:
            self.logger.info(f"\n📈 Signal Quality Breakdown:")
            self.logger.info(f"   Confluence Rate: {self.mr_performance_stats['oscillator_confluence_signals']/total_signals:.1%}")
            self.logger.info(f"   Divergence Rate: {self.mr_performance_stats['divergence_signals']/total_signals:.1%}")
            self.logger.info(f"   Extreme Reading Rate: {self.mr_performance_stats['extreme_oscillator_signals']/total_signals:.1%}")


def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Mean Reversion Strategy Backtest')
    parser.add_argument('--epic', type=str, help='Epic to backtest (e.g., CS.D.EURUSD.CEEM.IP)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to backtest (default: 7)')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe (default: 15m)')
    parser.add_argument('--show-signals', action='store_true', help='Show detailed signal information')
    parser.add_argument('--disable-optimization', action='store_true', help='Disable optimization parameters')

    args = parser.parse_args()

    # Initialize backtest
    backtest = MeanReversionBacktest(use_optimal_parameters=not args.disable_optimization)

    try:
        # Run the backtest
        success = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=args.show_signals
        )

        if success:
            # Display mean reversion specific performance
            backtest.display_mean_reversion_performance()
            backtest.logger.info("✅ Mean Reversion backtest completed successfully")
        else:
            backtest.logger.error("❌ Mean Reversion backtest failed")

    except KeyboardInterrupt:
        backtest.logger.info("🛑 Backtest interrupted by user")
    except Exception as e:
        backtest.logger.error(f"❌ Backtest failed with error: {e}")
        raise


if __name__ == "__main__":
    main()