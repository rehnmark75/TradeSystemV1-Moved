#!/usr/bin/env python3
"""
Ranging Market Strategy Backtest with Multi-Oscillator Confluence Analysis
Run: python backtest_ranging_market.py --epic CS.D.GBPUSD.MINI.IP --days 7 --timeframe 15m

FEATURES:
- Multi-oscillator confluence approach optimized for ranging/sideways markets
- Squeeze Momentum Indicator (LazyBear - primary ranging detection engine)
- Wave Trend Oscillator for hybrid trend/momentum analysis
- Bollinger Bands + Keltner Channels for dynamic support/resistance
- RSI with divergence detection for mean reversion signals
- Relative Vigor Index for momentum confirmation
- Dynamic support/resistance zone validation
- Market regime filtering (optimized for ranging conditions)
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
    from core.strategies.ranging_market_strategy import RangingMarketStrategy, create_ranging_market_strategy
except ImportError:
    from forex_scanner.backtests.backtest_base import BacktestBase
    from forex_scanner.core.strategies.ranging_market_strategy import RangingMarketStrategy, create_ranging_market_strategy

try:
    import config
except ImportError:
    from forex_scanner import config


class RangingMarketBacktest(BacktestBase):
    """Ranging Market Strategy Backtesting with Multi-Oscillator Confluence Analysis"""

    def __init__(self, use_optimal_parameters: bool = True):
        super().__init__('ranging_market', use_optimal_parameters)
        self.logger = logging.getLogger('ranging_market_backtest')
        self.setup_logging()

        # Performance tracking for ranging market specific metrics
        self.rm_performance_stats = {
            'squeeze_momentum_signals': 0,
            'wave_trend_signals': 0,
            'bollinger_keltner_signals': 0,
            'rsi_divergence_signals': 0,
            'rvi_signals': 0,
            'oscillator_confluence_signals': 0,
            'zone_validation_passes': 0,
            'regime_filter_passes': 0,
            'squeeze_release_signals': 0,
            'dynamic_sr_hits': 0
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def initialize_strategy(self, epic: str = None) -> RangingMarketStrategy:
        """Initialize the ranging market strategy with optimal parameters"""
        try:
            # Create strategy with optimal parameters
            strategy = create_ranging_market_strategy(
                data_fetcher=self.data_fetcher,
                backtest_mode=True,
                epic=epic,
                timeframe='15m',
                use_optimized_parameters=self.use_optimal_parameters
            )

            if self.use_optimal_parameters:
                self.logger.info(f"✅ Ranging Market strategy initialized with optimal parameters for {epic}")
            else:
                self.logger.info(f"📊 Ranging Market strategy initialized with default parameters for {epic}")

            return strategy

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Ranging Market strategy: {e}")
            raise

    def run_strategy_backtest(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """
        Run ranging market strategy backtest on given data - scanning all historical periods

        Args:
            df: DataFrame with price data and indicators
            epic: Trading pair epic
            spread_pips: Spread in pips
            timeframe: Trading timeframe

        Returns:
            List of signal dictionaries
        """
        try:
            self.logger.info(f"🎯 Running Ranging Market backtest for {epic}")
            self.logger.info(f"   Data points: {len(df)}")
            self.logger.info(f"   Timeframe: {timeframe}")
            self.logger.info(f"   Spread: {spread_pips} pips")

            # Initialize strategy for this epic
            strategy = self.initialize_strategy(epic)

            signals = []
            min_bars = 50  # Minimum bars needed for analysis
            last_signal_time = None

            # Scan through all historical data bar by bar
            for i in range(min_bars, len(df)):
                try:
                    # Get data up to current point
                    current_df = df.iloc[:i+1].copy()

                    # Check if we should skip based on signal spacing
                    current_time = current_df.iloc[-1].name if hasattr(current_df.iloc[-1], 'name') else None

                    if last_signal_time and current_time:
                        try:
                            if isinstance(current_time, str):
                                current_time = pd.to_datetime(current_time)
                            if isinstance(last_signal_time, str):
                                last_signal_time = pd.to_datetime(last_signal_time)

                            hours_diff = (current_time - last_signal_time).total_seconds() / 3600
                            min_spacing = strategy.config.get('signal_filter_min_signal_spacing', 12)

                            if hours_diff < min_spacing:
                                continue
                        except:
                            pass  # Skip spacing check if timestamp parsing fails

                    # Detect signal at this point in time
                    signal = strategy.detect_signal(
                        df=current_df,
                        epic=epic,
                        spread_pips=spread_pips,
                        timeframe=timeframe
                    )

                    if signal:
                        # Enhance signal with backtest-specific information
                        enhanced_signal = self._enhance_signal_for_backtest(signal, current_df, epic)
                        signals.append(enhanced_signal)

                        # Update performance statistics
                        self._update_performance_stats(enhanced_signal)

                        # Update last signal time for spacing
                        last_signal_time = current_time

                        self.logger.info(f"✅ Found Ranging Market signal: {signal['signal_type']} at "
                                       f"{signal.get('price', 'N/A')} (confidence: {signal.get('confidence', 0):.1%})")

                        # Continue scanning for more signals within time/daily limits

                except Exception as e:
                    self.logger.debug(f"Error processing bar {i}: {e}")
                    continue

            self.logger.info(f"   Found {len(signals)} signals")
            return signals

        except Exception as e:
            self.logger.error(f"❌ Ranging Market backtest failed for {epic}: {e}")
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

            # Add ranging market specific analysis
            enhanced_signal['ranging_market_analysis'] = self._analyze_ranging_market_conditions(signal, df)

            # Add oscillator confluence details
            oscillator_signals = signal.get('oscillator_signals', {})
            enhanced_signal['oscillator_confluence'] = {
                'squeeze_momentum': oscillator_signals.get('squeeze_momentum', {}).get('value', 0),
                'wave_trend_wt1': oscillator_signals.get('wave_trend', {}).get('wt1', 0),
                'wave_trend_wt2': oscillator_signals.get('wave_trend', {}).get('wt2', 0),
                'rsi_value': oscillator_signals.get('rsi', {}).get('value', 50),
                'rvi_value': oscillator_signals.get('rvi', {}).get('value', 0),
                'bull_score': signal.get('confluence_result', {}).get('bull_score', 0),
                'bear_score': signal.get('confluence_result', {}).get('bear_score', 0),
                'bull_confirmations': signal.get('confluence_result', {}).get('bull_confirmations', 0),
                'bear_confirmations': signal.get('confluence_result', {}).get('bear_confirmations', 0),
                'squeeze_active': oscillator_signals.get('squeeze_momentum', {}).get('squeeze_active', False),
                'rsi_divergence_bull': oscillator_signals.get('rsi', {}).get('divergence_bull', False),
                'rsi_divergence_bear': oscillator_signals.get('rsi', {}).get('divergence_bear', False)
            }

            # Add zone validation details
            zone_validation = signal.get('zone_validation', {})
            enhanced_signal['zone_analysis'] = {
                'zone_valid': zone_validation.get('valid', False),
                'zone_type': zone_validation.get('zone_type', 'unknown'),
                'zone_level': zone_validation.get('zone_level', 0),
                'distance_pips': zone_validation.get('distance_pips', 0)
            }

            # Add execution guidance analysis
            execution_guidance = signal.get('execution_guidance', {})
            enhanced_signal['execution_analysis'] = {
                'market_regime': execution_guidance.get('market_regime', 'unknown'),
                'oscillator_confluence_score': execution_guidance.get('oscillator_confluence', 0),
                'zone_proximity': execution_guidance.get('zone_proximity', 0),
                'squeeze_active': execution_guidance.get('squeeze_active', False),
                'recommended_position_size': execution_guidance.get('recommended_position_size', 'medium')
            }

            # Add performance predictions based on signal characteristics
            enhanced_signal['performance_prediction'] = self._predict_signal_performance(enhanced_signal)

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"Error enhancing signal for backtest: {e}")
            return signal

    def _analyze_ranging_market_conditions(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Analyze ranging market conditions for signal quality assessment"""
        try:
            if len(df) == 0:
                return {'error': 'No data available'}

            latest_row = df.iloc[-1]

            # Analyze market structure for ranging market suitability
            analysis = {
                'range_characteristics': self._analyze_range_characteristics(df),
                'volatility_profile': self._analyze_volatility_profile(df),
                'trend_strength': self._analyze_trend_strength(latest_row),
                'oscillator_readings': self._analyze_oscillator_readings(latest_row, signal),
                'ranging_probability': 0.0
            }

            # Calculate ranging market probability
            probability_factors = []

            # Factor 1: ADX reading (lower is better for ranging)
            adx = latest_row.get('adx', 25)
            if adx < 15:
                probability_factors.append(0.9)  # Very high probability
            elif adx < 20:
                probability_factors.append(0.7)  # High probability
            elif adx < 25:
                probability_factors.append(0.5)  # Moderate probability
            else:
                probability_factors.append(0.3)  # Lower probability

            # Factor 2: Squeeze state (squeeze indicates ranging)
            oscillator_confluence = signal.get('oscillator_confluence', {})
            if oscillator_confluence.get('squeeze_active', False):
                probability_factors.append(0.8)
            else:
                probability_factors.append(0.5)

            # Factor 3: Range stability
            range_char = analysis.get('range_characteristics', {})
            range_stability = range_char.get('stability_score', 0.5)
            probability_factors.append(range_stability)

            # Factor 4: Zone proximity
            zone_analysis = signal.get('zone_analysis', {})
            if zone_analysis.get('zone_valid', False):
                distance_pips = zone_analysis.get('distance_pips', 0)
                if distance_pips <= 5:
                    probability_factors.append(0.8)  # Very close to zone
                elif distance_pips <= 10:
                    probability_factors.append(0.6)  # Close to zone
                else:
                    probability_factors.append(0.4)  # Moderate distance
            else:
                probability_factors.append(0.3)

            # Calculate weighted average
            analysis['ranging_probability'] = sum(probability_factors) / len(probability_factors)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing ranging market conditions: {e}")
            return {'error': str(e)}

    def _analyze_range_characteristics(self, df: pd.DataFrame) -> Dict:
        """Analyze characteristics of the current price range"""
        try:
            if len(df) < 30:
                return {'error': 'insufficient_data'}

            # Use recent 30 bars for range analysis
            recent_data = df.tail(30)

            range_high = recent_data['high'].max()
            range_low = recent_data['low'].min()
            range_size = range_high - range_low
            current_price = df.iloc[-1]['close']

            if range_size == 0:
                return {'error': 'no_range_detected'}

            # Calculate position within range
            position_in_range = (current_price - range_low) / range_size

            # Calculate range stability (how consistently price stays within range)
            breakout_bars = 0
            for _, row in recent_data.iterrows():
                if row['high'] > range_high * 1.02 or row['low'] < range_low * 0.98:
                    breakout_bars += 1

            stability_score = 1.0 - (breakout_bars / len(recent_data))

            # Calculate oscillation frequency
            price_changes = recent_data['close'].diff().abs()
            avg_change = price_changes.mean()
            oscillation_frequency = avg_change / range_size if range_size > 0 else 0

            return {
                'range_size_pips': range_size * 10000,  # Assuming 4-digit broker
                'position_in_range': position_in_range,
                'stability_score': stability_score,
                'oscillation_frequency': oscillation_frequency,
                'range_high': range_high,
                'range_low': range_low,
                'current_position': 'upper_range' if position_in_range > 0.7 else
                                  'lower_range' if position_in_range < 0.3 else 'middle_range'
            }

        except Exception as e:
            return {'error': str(e)}

    def _analyze_volatility_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility profile for ranging market assessment"""
        try:
            if 'atr' not in df.columns or len(df) < 20:
                return {'error': 'insufficient_volatility_data'}

            current_atr = df.iloc[-1]['atr']
            avg_atr = df['atr'].tail(20).mean()
            atr_stability = 1.0 - (df['atr'].tail(20).std() / avg_atr) if avg_atr > 0 else 0

            # Classify volatility state
            if current_atr > avg_atr * 1.3:
                volatility_state = 'high'
            elif current_atr < avg_atr * 0.7:
                volatility_state = 'low'
            else:
                volatility_state = 'normal'

            return {
                'current_atr': current_atr,
                'average_atr': avg_atr,
                'volatility_state': volatility_state,
                'atr_stability': atr_stability,
                'ranging_favorable': volatility_state in ['low', 'normal'] and atr_stability > 0.6
            }

        except Exception as e:
            return {'error': str(e)}

    def _analyze_trend_strength(self, row: pd.Series) -> Dict:
        """Analyze trend strength indicators"""
        try:
            adx = row.get('adx', 25)

            if adx < 15:
                trend_state = 'very_weak_ranging'
                ranging_favorability = 0.9
            elif adx < 20:
                trend_state = 'weak_ranging'
                ranging_favorability = 0.7
            elif adx < 25:
                trend_state = 'moderate'
                ranging_favorability = 0.5
            elif adx < 30:
                trend_state = 'trending'
                ranging_favorability = 0.3
            else:
                trend_state = 'strong_trend'
                ranging_favorability = 0.1

            return {
                'adx_value': adx,
                'trend_state': trend_state,
                'ranging_favorability': ranging_favorability
            }

        except Exception as e:
            return {'error': str(e)}

    def _analyze_oscillator_readings(self, row: pd.Series, signal: Dict) -> Dict:
        """Analyze oscillator readings for confluence assessment"""
        try:
            oscillator_signals = signal.get('oscillator_signals', {})

            return {
                'squeeze_momentum': oscillator_signals.get('squeeze_momentum', {}).get('value', 0),
                'squeeze_active': oscillator_signals.get('squeeze_momentum', {}).get('squeeze_active', False),
                'wave_trend_wt1': oscillator_signals.get('wave_trend', {}).get('wt1', 0),
                'wave_trend_wt2': oscillator_signals.get('wave_trend', {}).get('wt2', 0),
                'rsi': oscillator_signals.get('rsi', {}).get('value', 50),
                'rvi': oscillator_signals.get('rvi', {}).get('value', 0),
                'confluence_score': max(
                    signal.get('confluence_result', {}).get('bull_score', 0),
                    signal.get('confluence_result', {}).get('bear_score', 0)
                )
            }

        except Exception as e:
            return {'error': str(e)}

    def _predict_signal_performance(self, signal: Dict) -> Dict:
        """Predict signal performance based on ranging market characteristics"""
        try:
            # Start with base expectation for ranging market strategy
            win_probability = 0.62  # Base ranging market win rate

            # Adjust based on signal characteristics
            confluence = signal.get('oscillator_confluence', {})
            execution = signal.get('execution_analysis', {})
            rm_analysis = signal.get('ranging_market_analysis', {})
            zone_analysis = signal.get('zone_analysis', {})

            # Factor 1: Oscillator confluence strength
            confluence_score = execution.get('oscillator_confluence_score', 0)
            if confluence_score > 0.8:
                win_probability += 0.18
            elif confluence_score > 0.7:
                win_probability += 0.12
            elif confluence_score < 0.6:
                win_probability -= 0.08

            # Factor 2: Squeeze state (active squeeze favors ranging strategy)
            if confluence.get('squeeze_active', False):
                win_probability += 0.15

            # Factor 3: Zone validation
            if zone_analysis.get('zone_valid', False):
                distance_pips = zone_analysis.get('distance_pips', 0)
                if distance_pips <= 5:
                    win_probability += 0.12
                elif distance_pips <= 10:
                    win_probability += 0.08

            # Factor 4: Ranging market probability
            ranging_probability = rm_analysis.get('ranging_probability', 0.5)
            win_probability += (ranging_probability - 0.5) * 0.25

            # Factor 5: ADX reading (lower is better for ranging)
            trend_analysis = rm_analysis.get('trend_strength', {})
            ranging_favorability = trend_analysis.get('ranging_favorability', 0.5)
            win_probability += (ranging_favorability - 0.5) * 0.15

            # Factor 6: Divergence presence
            if confluence.get('rsi_divergence_bull', False) or confluence.get('rsi_divergence_bear', False):
                win_probability += 0.10

            # Factor 7: Range characteristics
            range_char = rm_analysis.get('range_characteristics', {})
            if isinstance(range_char, dict) and 'stability_score' in range_char:
                stability = range_char.get('stability_score', 0.5)
                win_probability += (stability - 0.5) * 0.20

            # Clamp to reasonable bounds
            win_probability = max(0.35, min(0.88, win_probability))

            return {
                'predicted_win_probability': win_probability,
                'confidence_category': self._categorize_confidence(win_probability),
                'factors_analyzed': {
                    'confluence_score': confluence_score,
                    'squeeze_active': confluence.get('squeeze_active', False),
                    'zone_valid': zone_analysis.get('zone_valid', False),
                    'zone_distance_pips': zone_analysis.get('distance_pips', 0),
                    'ranging_probability': ranging_probability,
                    'ranging_favorability': ranging_favorability,
                    'divergence_present': confluence.get('rsi_divergence_bull', False) or confluence.get('rsi_divergence_bear', False)
                }
            }

        except Exception as e:
            return {'error': str(e), 'predicted_win_probability': 0.62}

    def _categorize_confidence(self, win_probability: float) -> str:
        """Categorize win probability into confidence levels"""
        if win_probability >= 0.78:
            return 'very_high'
        elif win_probability >= 0.68:
            return 'high'
        elif win_probability >= 0.58:
            return 'medium'
        elif win_probability >= 0.48:
            return 'low'
        else:
            return 'very_low'

    def _update_performance_stats(self, signal: Dict):
        """Update ranging market specific performance statistics"""
        try:
            # Track oscillator signals
            oscillator_confluence = signal.get('oscillator_confluence', {})

            if oscillator_confluence.get('squeeze_active', False):
                self.rm_performance_stats['squeeze_momentum_signals'] += 1

            if abs(oscillator_confluence.get('wave_trend_wt1', 0)) > 20:
                self.rm_performance_stats['wave_trend_signals'] += 1

            if oscillator_confluence.get('rsi_divergence_bull', False) or oscillator_confluence.get('rsi_divergence_bear', False):
                self.rm_performance_stats['rsi_divergence_signals'] += 1

            if abs(oscillator_confluence.get('rvi_value', 0)) > 0.3:
                self.rm_performance_stats['rvi_signals'] += 1

            # Track confluence signals
            confluence_score = max(oscillator_confluence.get('bull_score', 0), oscillator_confluence.get('bear_score', 0))
            if confluence_score > 0.7:
                self.rm_performance_stats['oscillator_confluence_signals'] += 1

            # Track zone validation
            zone_analysis = signal.get('zone_analysis', {})
            if zone_analysis.get('zone_valid', False):
                self.rm_performance_stats['zone_validation_passes'] += 1
                if zone_analysis.get('distance_pips', 0) <= 8:
                    self.rm_performance_stats['dynamic_sr_hits'] += 1

            # Track regime filtering
            execution = signal.get('execution_analysis', {})
            if execution.get('market_regime') == 'ranging':
                self.rm_performance_stats['regime_filter_passes'] += 1

        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")

    def display_ranging_market_performance(self):
        """Display ranging market specific performance metrics"""
        self.logger.info(f"\n📊 Ranging Market Strategy Performance Details:")
        self.logger.info(f"   Squeeze Momentum Signals: {self.rm_performance_stats['squeeze_momentum_signals']}")
        self.logger.info(f"   Wave Trend Signals: {self.rm_performance_stats['wave_trend_signals']}")
        self.logger.info(f"   RSI Divergence Signals: {self.rm_performance_stats['rsi_divergence_signals']}")
        self.logger.info(f"   RVI Signals: {self.rm_performance_stats['rvi_signals']}")
        self.logger.info(f"   Oscillator Confluence Signals: {self.rm_performance_stats['oscillator_confluence_signals']}")
        self.logger.info(f"   Zone Validation Passes: {self.rm_performance_stats['zone_validation_passes']}")
        self.logger.info(f"   Dynamic S/R Hits: {self.rm_performance_stats['dynamic_sr_hits']}")
        self.logger.info(f"   Regime Filter Passes: {self.rm_performance_stats['regime_filter_passes']}")

        # Calculate percentages if we have signals
        total_signals = sum(self.rm_performance_stats.values())
        if total_signals > 0:
            self.logger.info(f"\n📈 Signal Quality Breakdown:")
            self.logger.info(f"   Squeeze Detection Rate: {self.rm_performance_stats['squeeze_momentum_signals']/total_signals:.1%}")
            self.logger.info(f"   Confluence Rate: {self.rm_performance_stats['oscillator_confluence_signals']/total_signals:.1%}")
            self.logger.info(f"   Zone Validation Rate: {self.rm_performance_stats['zone_validation_passes']/total_signals:.1%}")
            self.logger.info(f"   S/R Hit Rate: {self.rm_performance_stats['dynamic_sr_hits']/total_signals:.1%}")


def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Ranging Market Strategy Backtest')
    parser.add_argument('--epic', type=str, help='Epic to backtest (e.g., CS.D.GBPUSD.MINI.IP)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to backtest (default: 7)')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe (default: 15m)')
    parser.add_argument('--show-signals', action='store_true', help='Show detailed signal information')
    parser.add_argument('--disable-optimization', action='store_true', help='Disable optimization parameters')

    args = parser.parse_args()

    # Initialize backtest
    backtest = RangingMarketBacktest(use_optimal_parameters=not args.disable_optimization)

    try:
        # Run the backtest
        success = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=args.show_signals
        )

        if success:
            # Display ranging market specific performance
            backtest.display_ranging_market_performance()
            backtest.logger.info("✅ Ranging Market backtest completed successfully")
        else:
            backtest.logger.error("❌ Ranging Market backtest failed")

    except KeyboardInterrupt:
        backtest.logger.info("🛑 Backtest interrupted by user")
    except Exception as e:
        backtest.logger.error(f"❌ Backtest failed with error: {e}")
        raise


if __name__ == "__main__":
    main()