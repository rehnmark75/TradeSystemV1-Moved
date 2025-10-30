#!/usr/bin/env python3
"""
SMC Pure Structure Strategy
Structure-based trading using Smart Money Concepts (pure price action)

Strategy Logic:
1. Identify HTF trend (4H structure)
2. Wait for pullback to S/R
3. Confirm rejection pattern (pin bar, engulfing, etc.)
4. Enter with structure (not indicators)
5. Stop beyond structure (invalidation point)
6. Target next structure (supply/demand zone)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Import helper modules
from .helpers.smc_trend_structure import SMCTrendStructure
from .helpers.smc_support_resistance import SMCSupportResistance
from .helpers.smc_candlestick_patterns import SMCCandlestickPatterns


class SMCStructureStrategy:
    """
    Pure structure-based strategy using Smart Money Concepts

    Entry Requirements (ALL must be met):
    1. HTF trend identified (4H showing clear HH/HL or LH/LL)
    2. Price pulled back to key S/R level
    3. Rejection pattern confirmed (pin bar, engulfing, etc.)
    4. Signal direction aligns with HTF trend
    5. Structure-based stop loss calculated
    6. R:R meets minimum requirement
    """

    def __init__(self, config, logger=None):
        """Initialize SMC Structure Strategy"""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize helper modules
        self.trend_analyzer = SMCTrendStructure(logger=self.logger)
        self.sr_detector = SMCSupportResistance(logger=self.logger)
        self.pattern_detector = SMCCandlestickPatterns(logger=self.logger)

        # Load configuration
        self._load_config()

        self.logger.info("✅ SMC Structure Strategy initialized")
        self.logger.info(f"   HTF Timeframe: {self.htf_timeframe}")
        self.logger.info(f"   Min Pattern Strength: {self.min_pattern_strength}")
        self.logger.info(f"   Min R:R Ratio: {self.min_rr_ratio}")
        self.logger.info(f"   SR Proximity: {self.sr_proximity_pips} pips")

    def _load_config(self):
        """Load strategy configuration"""
        # Higher timeframe for trend analysis
        self.htf_timeframe = getattr(self.config, 'SMC_HTF_TIMEFRAME', '4h')
        self.htf_lookback = getattr(self.config, 'SMC_HTF_LOOKBACK', 100)

        # Support/Resistance detection
        self.sr_lookback = getattr(self.config, 'SMC_SR_LOOKBACK', 100)
        self.sr_proximity_pips = getattr(self.config, 'SMC_SR_PROXIMITY_PIPS', 20)

        # Pattern detection
        self.min_pattern_strength = getattr(self.config, 'SMC_MIN_PATTERN_STRENGTH', 0.70)
        self.pattern_lookback_bars = getattr(self.config, 'SMC_PATTERN_LOOKBACK_BARS', 5)

        # Risk management
        self.sl_buffer_pips = getattr(self.config, 'SMC_SL_BUFFER_PIPS', 5)
        self.min_rr_ratio = getattr(self.config, 'SMC_MIN_RR_RATIO', 2.0)

        # Partial profit settings
        self.partial_profit_enabled = getattr(self.config, 'SMC_PARTIAL_PROFIT_ENABLED', True)
        self.partial_profit_percent = getattr(self.config, 'SMC_PARTIAL_PROFIT_PERCENT', 50)
        self.partial_profit_rr = getattr(self.config, 'SMC_PARTIAL_PROFIT_RR', 1.5)

    def detect_signal(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        epic: str,
        pair: str
    ) -> Optional[Dict]:
        """
        Detect SMC structure-based trading signal

        Args:
            df_1h: 1H timeframe OHLCV data (entry timeframe)
            df_4h: 4H timeframe OHLCV data (higher timeframe for trend)
            epic: IG Markets epic code
            pair: Currency pair name

        Returns:
            Signal dict or None if no valid signal
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔍 SMC Structure Strategy - Signal Detection")
        self.logger.info(f"   Pair: {pair} ({epic})")
        self.logger.info(f"   Entry TF: 1H | HTF: {self.htf_timeframe}")
        self.logger.info(f"{'='*70}")

        # Get pip value
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        try:
            # STEP 1: Identify HTF trend (4H structure)
            self.logger.info(f"\n📊 STEP 1: Analyzing HTF Trend Structure ({self.htf_timeframe})")

            trend_analysis = self.trend_analyzer.analyze_trend(
                df=df_4h,
                epic=epic,
                lookback=self.htf_lookback
            )

            self.logger.info(f"   Trend: {trend_analysis['trend']}")
            self.logger.info(f"   Strength: {trend_analysis['strength']*100:.0f}%")
            self.logger.info(f"   Structure: {trend_analysis['structure_type']}")
            self.logger.info(f"   Swing Highs: {len(trend_analysis['swing_highs'])}")
            self.logger.info(f"   Swing Lows: {len(trend_analysis['swing_lows'])}")

            if trend_analysis['in_pullback']:
                self.logger.info(f"   ✅ In Pullback: {trend_analysis['pullback_depth']*100:.0f}% retracement")

            # Must have clear trend
            if trend_analysis['trend'] not in ['BULL', 'BEAR']:
                self.logger.info(f"   ❌ No clear trend - SIGNAL REJECTED")
                return None

            # Must have minimum strength
            if trend_analysis['strength'] < 0.50:
                self.logger.info(f"   ❌ Trend too weak ({trend_analysis['strength']*100:.0f}% < 50%) - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ HTF Trend confirmed: {trend_analysis['trend']}")

            # STEP 2: Detect S/R levels and check proximity
            self.logger.info(f"\n🎯 STEP 2: Detecting Support/Resistance Levels")

            levels = self.sr_detector.detect_levels(
                df=df_1h,
                epic=epic,
                lookback=self.sr_lookback
            )

            self.logger.info(f"   Support levels: {len(levels['support'])}")
            self.logger.info(f"   Resistance levels: {len(levels['resistance'])}")
            self.logger.info(f"   Demand zones: {len(levels['demand_zones'])}")
            self.logger.info(f"   Supply zones: {len(levels['supply_zones'])}")

            # Get current price
            current_price = df_1h['close'].iloc[-1]

            # Check which levels we're near based on trend direction
            if trend_analysis['trend'] == 'BULL':
                # For bullish trend, look for pullback to support/demand zones
                relevant_levels = levels['support'] + levels['demand_zones']
                level_type = 'support/demand'
            else:
                # For bearish trend, look for pullback to resistance/supply zones
                relevant_levels = levels['resistance'] + levels['supply_zones']
                level_type = 'resistance/supply'

            nearest_level = self.sr_detector.find_nearest_level(
                current_price=current_price,
                levels=relevant_levels,
                proximity_pips=self.sr_proximity_pips,
                pip_value=pip_value
            )

            if not nearest_level:
                self.logger.info(f"   ❌ Not near any {level_type} level (within {self.sr_proximity_pips} pips) - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ Near {level_type} level:")
            self.logger.info(f"      Price: {nearest_level['price']:.5f}")
            self.logger.info(f"      Distance: {nearest_level['distance_pips']:.1f} pips")
            self.logger.info(f"      Strength: {nearest_level['strength']*100:.0f}%")
            self.logger.info(f"      Touches: {nearest_level['touch_count']}")

            # STEP 3: Confirm rejection pattern
            self.logger.info(f"\n📍 STEP 3: Detecting Rejection Pattern")

            # Get recent bars for pattern detection
            recent_bars = df_1h.tail(self.pattern_lookback_bars)

            rejection_pattern = self.pattern_detector.detect_rejection_pattern(
                df=recent_bars,
                direction=trend_analysis['trend'],
                min_strength=self.min_pattern_strength
            )

            if not rejection_pattern:
                self.logger.info(f"   ❌ No strong rejection pattern (min strength {self.min_pattern_strength*100:.0f}%) - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ Rejection pattern detected:")
            self.logger.info(f"      Type: {rejection_pattern['pattern_type']}")
            self.logger.info(f"      Strength: {rejection_pattern['strength']*100:.0f}%")
            self.logger.info(f"      Entry: {rejection_pattern['entry_price']:.5f}")
            self.logger.info(f"      Rejection Level: {rejection_pattern['rejection_level']:.5f}")
            self.logger.info(f"      Description: {rejection_pattern['description']}")

            # STEP 4: Calculate structure-based stop loss
            self.logger.info(f"\n🛑 STEP 4: Calculating Structure-Based Stop Loss")

            # Stop loss goes beyond the structure that would invalidate the trade
            if trend_analysis['trend'] == 'BULL':
                # For longs, stop below rejection level (swing low)
                structure_invalidation = rejection_pattern['rejection_level']
                stop_loss = structure_invalidation - (self.sl_buffer_pips * pip_value)
            else:
                # For shorts, stop above rejection level (swing high)
                structure_invalidation = rejection_pattern['rejection_level']
                stop_loss = structure_invalidation + (self.sl_buffer_pips * pip_value)

            entry_price = rejection_pattern['entry_price']
            risk_pips = abs(entry_price - stop_loss) / pip_value

            # Validate entry vs stop loss relationship (catch logic errors)
            if trend_analysis['trend'] == 'BULL':
                if entry_price <= stop_loss:
                    self.logger.error(f"❌ Invalid BULL entry: entry {entry_price:.5f} <= stop {stop_loss:.5f}")
                    self.logger.error(f"   This indicates a bug in pattern entry logic!")
                    return None
            else:  # BEAR
                if entry_price >= stop_loss:
                    self.logger.error(f"❌ Invalid BEAR entry: entry {entry_price:.5f} >= stop {stop_loss:.5f}")
                    self.logger.error(f"   This indicates a bug in pattern entry logic!")
                    return None

            self.logger.info(f"   ✅ Entry price validation passed")
            self.logger.info(f"   Structure Invalidation: {structure_invalidation:.5f}")
            self.logger.info(f"   Stop Loss: {stop_loss:.5f}")
            self.logger.info(f"   Risk: {risk_pips:.1f} pips")

            # STEP 5: Calculate take profit (next structure level)
            self.logger.info(f"\n🎯 STEP 5: Calculating Structure-Based Take Profit")

            # Find next structure level in direction of trade
            if trend_analysis['trend'] == 'BULL':
                # For longs, target next resistance/supply zone
                target_levels = levels['resistance'] + levels['supply_zones']
                # Filter levels above entry
                target_levels = [l for l in target_levels if l['price'] > entry_price]
            else:
                # For shorts, target next support/demand zone
                target_levels = levels['support'] + levels['demand_zones']
                # Filter levels below entry
                target_levels = [l for l in target_levels if l['price'] < entry_price]

            if not target_levels:
                self.logger.info(f"   ⚠️  No structure level for TP, using minimum R:R of {self.min_rr_ratio}")
                reward_pips = risk_pips * self.min_rr_ratio
                if trend_analysis['trend'] == 'BULL':
                    take_profit = entry_price + (reward_pips * pip_value)
                else:
                    take_profit = entry_price - (reward_pips * pip_value)
            else:
                # Sort by distance and take nearest
                if trend_analysis['trend'] == 'BULL':
                    target_levels.sort(key=lambda x: x['price'])
                else:
                    target_levels.sort(key=lambda x: x['price'], reverse=True)

                target_level = target_levels[0]
                take_profit = target_level['price']
                reward_pips = abs(take_profit - entry_price) / pip_value

                self.logger.info(f"   Target Level: {target_level['price']:.5f}")
                self.logger.info(f"   Strength: {target_level['strength']*100:.0f}%")

            # Calculate R:R ratio
            rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

            self.logger.info(f"   Take Profit: {take_profit:.5f}")
            self.logger.info(f"   Reward: {reward_pips:.1f} pips")
            self.logger.info(f"   R:R Ratio: {rr_ratio:.2f}")

            # STEP 6: Validate R:R ratio
            if rr_ratio < self.min_rr_ratio:
                self.logger.info(f"   ❌ R:R too low ({rr_ratio:.2f} < {self.min_rr_ratio}) - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ R:R meets minimum requirement ({rr_ratio:.2f} >= {self.min_rr_ratio})")

            # Calculate partial profit if enabled
            partial_tp = None
            if self.partial_profit_enabled:
                partial_reward_pips = risk_pips * self.partial_profit_rr
                if trend_analysis['trend'] == 'BULL':
                    partial_tp = entry_price + (partial_reward_pips * pip_value)
                else:
                    partial_tp = entry_price - (partial_reward_pips * pip_value)

                self.logger.info(f"\n💰 Partial Profit Settings:")
                self.logger.info(f"   Enabled: Yes")
                self.logger.info(f"   Partial TP: {partial_tp:.5f}")
                self.logger.info(f"   Partial R:R: {self.partial_profit_rr}")
                self.logger.info(f"   Close Percent: {self.partial_profit_percent}%")

            # BUILD SIGNAL
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ VALID SMC STRUCTURE SIGNAL DETECTED")
            self.logger.info(f"{'='*70}")

            signal = {
                'strategy': 'SMC_STRUCTURE',
                'signal': trend_analysis['trend'],
                'epic': epic,
                'pair': pair,
                'timeframe': '1h',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'partial_tp': partial_tp,
                'partial_percent': self.partial_profit_percent if self.partial_profit_enabled else None,
                'risk_pips': risk_pips,
                'reward_pips': reward_pips,
                'rr_ratio': rr_ratio,
                'timestamp': datetime.now(),

                # Signal details
                'htf_trend': trend_analysis['trend'],
                'htf_strength': trend_analysis['strength'],
                'htf_structure': trend_analysis['structure_type'],
                'in_pullback': trend_analysis['in_pullback'],
                'pullback_depth': trend_analysis['pullback_depth'],

                'sr_level': nearest_level['price'],
                'sr_type': nearest_level['type'],
                'sr_strength': nearest_level['strength'],
                'sr_distance_pips': nearest_level['distance_pips'],

                'pattern_type': rejection_pattern['pattern_type'],
                'pattern_strength': rejection_pattern['strength'],
                'rejection_level': rejection_pattern['rejection_level'],

                # Readable description
                'description': self._build_signal_description(
                    trend_analysis, nearest_level, rejection_pattern, rr_ratio
                )
            }

            self.logger.info(f"\n📋 Signal Summary:")
            self.logger.info(f"   Direction: {signal['signal']}")
            self.logger.info(f"   Entry: {signal['entry_price']:.5f}")
            self.logger.info(f"   Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
            self.logger.info(f"   Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
            if partial_tp:
                self.logger.info(f"   Partial TP: {partial_tp:.5f} ({self.partial_profit_percent}% at {self.partial_profit_rr}R)")
            self.logger.info(f"   R:R Ratio: {signal['rr_ratio']:.2f}")
            self.logger.info(f"\n   {signal['description']}")
            self.logger.info(f"{'='*70}\n")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting SMC signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_signal_description(
        self,
        trend_analysis: Dict,
        sr_level: Dict,
        pattern: Dict,
        rr_ratio: float
    ) -> str:
        """Build human-readable signal description"""

        desc_parts = []

        # Trend context
        desc_parts.append(f"{trend_analysis['trend']} trend ({trend_analysis['structure_type']})")

        # Pullback context
        if trend_analysis['in_pullback']:
            desc_parts.append(f"pullback to {sr_level['type']}")
        else:
            desc_parts.append(f"at {sr_level['type']}")

        # Pattern
        pattern_name = pattern['pattern_type'].replace('_', ' ')
        desc_parts.append(f"with {pattern_name}")

        # R:R
        desc_parts.append(f"({rr_ratio:.1f}R)")

        return ", ".join(desc_parts).capitalize()

    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return "SMC_STRUCTURE"

    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return "Pure structure-based strategy using Smart Money Concepts (price action only)"


def create_smc_structure_strategy(config=None, **kwargs) -> SMCStructureStrategy:
    """
    Factory function to create SMC Structure strategy instance

    Args:
        config: Configuration module (if None, imports config_smc_structure)
        **kwargs: Additional arguments passed to strategy

    Returns:
        SMCStructureStrategy instance
    """
    if config is None:
        # Import config using absolute import from configdata
        # (backtest_cli adds /app/forex_scanner to path, so 'configdata' is accessible)
        from configdata.strategies import config_smc_structure as config

    return SMCStructureStrategy(config=config, **kwargs)
