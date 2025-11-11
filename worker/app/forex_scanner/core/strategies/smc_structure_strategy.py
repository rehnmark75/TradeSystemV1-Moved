#!/usr/bin/env python3
"""
SMC Pure Structure Strategy - BARE ESSENTIALS (Refactored v2.5.0)
Structure-based trading using Smart Money Concepts (core logic only)

VERSION: 2.5.0 (Stripped to Bare Essentials)
DATE: 2025-11-11
STATUS: Baseline Measurement - All Filters Removed

REFACTORING GOAL: Remove 99% of signal-blocking filters to establish baseline.

Strategy Logic (SIMPLIFIED):
1. Identify HTF trend (4H structure) - swing analysis
2. Detect BOS/CHoCH on 15m timeframe - break of structure
3. Validate HTF alignment (15m BOS matches 4H trend)
4. Generate signal with simple structure-based stops

ALL FILTERS REMOVED:
- Order Block re-entry logic (blocked 95% of signals)
- BOS quality scoring (buggy - wrong candle)
- Re-entry zone checks (all variants)
- Premium/discount zone filters
- Pattern detection (pin bars, engulfing, etc.)
- S/R level detection
- Session filters
- Momentum filters
- Cooldown system
- Signal deduplication
- Confidence filters

Expected Outcome:
- 50-70% code reduction
- 20-50+ signals in 30 days (vs 3 with filters)
- Ready for data-driven optimization

Version History:
- v2.5.0 (2025-11-11): Stripped to bare essentials - removed all filters
- v2.4.0 (2025-11-03): Last profitable baseline
- v2.2.0 (2025-11-03): Order Block Re-entry implementation (over-engineered)
- v2.1.1 (2025-11-03): Added session filter (disabled), fixed timestamp bug
- v2.1.0 (2025-11-02): Phase 2.1 baseline - HTF alignment enabled
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

# Import ONLY required helper modules
from .helpers.smc_trend_structure import SMCTrendStructure
from .helpers.smc_market_structure import SMCMarketStructure


class SMCStructureStrategy:
    """
    Pure structure-based strategy using Smart Money Concepts (BARE ESSENTIALS)

    Entry Requirements (ALL must be met):
    1. HTF trend identified (4H showing clear HH/HL or LH/LL)
    2. BOS/CHoCH detected on 15m timeframe
    3. HTF alignment confirmed (15m BOS matches 4H trend)
    4. Structure-based stop loss calculated
    """

    def __init__(self, config, logger=None, decision_logger=None):
        """Initialize SMC Structure Strategy"""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        # decision_logger parameter ignored in v2.5.0 - only for compatibility

        # Initialize ONLY required helper modules
        self.trend_analyzer = SMCTrendStructure(logger=self.logger)
        self.market_structure = SMCMarketStructure(logger=self.logger)

        # Load configuration
        self._load_config()

        self.logger.info("✅ SMC Structure Strategy initialized (v2.5.2 - Improved Swing Detection)")
        self.logger.info(f"   HTF Timeframe: {self.htf_timeframe}")
        self.logger.info(f"   BOS/CHoCH Timeframe: {self.bos_choch_timeframe}")
        self.logger.info(f"   HTF Alignment Lookback: {self.htf_alignment_lookback}")
        self.logger.info(f"   Min R:R Ratio: {self.min_rr_ratio}")
        self.logger.info(f"   Stop Loss Buffer: {self.sl_buffer_pips} pips")

    def _load_config(self):
        """Load strategy configuration (SIMPLIFIED)"""
        # Higher timeframe for trend analysis
        self.htf_timeframe = getattr(self.config, 'SMC_HTF_TIMEFRAME', '4h')
        self.htf_lookback = getattr(self.config, 'SMC_HTF_LOOKBACK', 100)

        # BOS/CHoCH detection
        self.bos_choch_timeframe = getattr(self.config, 'SMC_BOS_CHOCH_TIMEFRAME', '15m')
        self.require_1h_alignment = getattr(self.config, 'SMC_REQUIRE_1H_ALIGNMENT', True)
        self.require_4h_alignment = getattr(self.config, 'SMC_REQUIRE_4H_ALIGNMENT', True)
        self.htf_alignment_lookback = getattr(self.config, 'SMC_HTF_ALIGNMENT_LOOKBACK', 50)

        # Risk management (BASIC)
        self.sl_buffer_pips = getattr(self.config, 'SMC_SL_BUFFER_PIPS', 5)
        self.min_rr_ratio = getattr(self.config, 'SMC_MIN_RR_RATIO', 2.0)

        # Partial profit settings (OPTIONAL)
        self.partial_profit_enabled = getattr(self.config, 'SMC_PARTIAL_PROFIT_ENABLED', True)
        self.partial_profit_percent = getattr(self.config, 'SMC_PARTIAL_PROFIT_PERCENT', 50)
        self.partial_profit_rr = getattr(self.config, 'SMC_PARTIAL_PROFIT_RR', 1.5)

        # Pair blacklist - exclude pairs with proven failure modes
        self.blacklisted_pairs = getattr(self.config, 'SMC_BLACKLISTED_PAIRS', [])

    def detect_signal(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        epic: str,
        pair: str,
        df_15m: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Detect SMC structure-based trading signal (BARE ESSENTIALS)

        Args:
            df_1h: 1H timeframe OHLCV data (for entry calculation)
            df_4h: 4H timeframe OHLCV data (higher timeframe for trend)
            epic: IG Markets epic code
            pair: Currency pair name
            df_15m: 15m timeframe for BOS/CHoCH detection

        Returns:
            Signal dict or None if no valid signal
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔍 SMC Structure Strategy - Signal Detection (v2.5.2 - Improved Swing Detection)")
        self.logger.info(f"   Pair: {pair} ({epic})")
        self.logger.info(f"   HTF: {self.htf_timeframe} | BOS TF: {self.bos_choch_timeframe}")
        self.logger.info(f"{'='*70}")

        # OPTIMIZATION #2: Check pair blacklist (immediate loss elimination)
        # NZDUSD: 0% WR (96/96 losses) - excluded until pair-specific stops implemented
        if pair in self.blacklisted_pairs:
            self.logger.info(f"   ❌ Pair {pair} is BLACKLISTED (proven failure mode) - SIGNAL REJECTED")
            return None

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

            # Detect BOS/CHoCH on HTF to determine trend direction
            self.logger.info(f"\n🔍 Detecting BOS/CHoCH on HTF ({self.htf_timeframe}) for trend direction...")

            df_4h_with_structure = self.market_structure.analyze_market_structure(
                df=df_4h,
                epic=epic,
                config=vars(self.config) if hasattr(self.config, '__dict__') else {}
            )

            bos_choch_direction = self.market_structure.get_last_bos_choch_direction(df_4h_with_structure)

            # Use BOS/CHoCH direction as trend
            if bos_choch_direction in ['bullish', 'bearish']:
                final_trend = 'BULL' if bos_choch_direction == 'bullish' else 'BEAR'
                final_strength = trend_analysis['strength']  # Use swing strength

                self.logger.info(f"   ✅ HTF BOS/CHoCH: {bos_choch_direction.upper()} → {final_trend}")
                self.logger.info(f"   ✅ Trend Strength: {final_strength*100:.0f}%")
            else:
                # No BOS/CHoCH found - reject signal
                self.logger.info(f"   ❌ No BOS/CHoCH detected on HTF - SIGNAL REJECTED")
                return None

            # PHASE 1 OPTIMIZATION: Raised HTF strength threshold 50% → 60%
            # Filters 20-30 marginal trend signals
            # Expected: PF 2.04 → 2.35-2.55 (+15-25%)
            if final_strength < 0.60:
                self.logger.info(f"   ❌ Trend too weak ({final_strength*100:.0f}% < 60%) - SIGNAL REJECTED")
                return None

            # STEP 2: Detect BOS/CHoCH on 15m timeframe
            if df_15m is None or len(df_15m) == 0:
                self.logger.info(f"\n❌ No 15m data available - SIGNAL REJECTED")
                return None

            self.logger.info(f"\n🔄 STEP 2: Detecting BOS/CHoCH on 15m Timeframe")

            bos_choch_info = self._detect_bos_choch_15m(df_15m, epic)

            if not bos_choch_info:
                self.logger.info(f"   ❌ No BOS/CHoCH detected on 15m - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ BOS/CHoCH detected:")
            self.logger.info(f"      Direction: {bos_choch_info['direction']}")
            self.logger.info(f"      Level: {bos_choch_info['level']:.5f}")
            self.logger.info(f"      Type: {bos_choch_info['type']}")

            # STEP 3: Validate HTF alignment
            self.logger.info(f"\n🔍 STEP 3: Validating HTF Alignment")

            htf_aligned = self._validate_htf_alignment(
                bos_direction=bos_choch_info['direction'],
                df_1h=df_1h,
                df_4h=df_4h,
                epic=epic
            )

            if not htf_aligned:
                self.logger.info(f"   ❌ HTF not aligned - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ HTF Alignment Confirmed")

            # STEP 4: Calculate entry and stop loss
            self.logger.info(f"\n💰 STEP 4: Calculating Entry and Stop Loss")

            current_price = float(df_15m['close'].iloc[-1])
            entry_price = current_price

            # Simple structure-based stop loss
            if bos_choch_info['direction'] == 'bullish':
                # For longs, stop below BOS level
                structure_level = bos_choch_info['level']
                stop_loss = structure_level - (self.sl_buffer_pips * pip_value)
                direction_str = 'bullish'
            else:
                # For shorts, stop above BOS level
                structure_level = bos_choch_info['level']
                stop_loss = structure_level + (self.sl_buffer_pips * pip_value)
                direction_str = 'bearish'

            risk_pips = abs(entry_price - stop_loss) / pip_value

            # Validate entry vs stop loss relationship
            if bos_choch_info['direction'] == 'bullish':
                if entry_price <= stop_loss:
                    self.logger.error(f"   ❌ Invalid BULL entry: entry {entry_price:.5f} <= stop {stop_loss:.5f}")
                    return None
            else:
                if entry_price >= stop_loss:
                    self.logger.error(f"   ❌ Invalid BEAR entry: entry {entry_price:.5f} >= stop {stop_loss:.5f}")
                    return None

            self.logger.info(f"   ✅ Entry: {entry_price:.5f}")
            self.logger.info(f"   ✅ Stop Loss: {stop_loss:.5f}")
            self.logger.info(f"   ✅ Risk: {risk_pips:.1f} pips")

            # STEP 5: Calculate take profit (simple R:R based)
            self.logger.info(f"\n🎯 STEP 5: Calculating Take Profit (R:R Based)")

            reward_pips = risk_pips * self.min_rr_ratio
            if bos_choch_info['direction'] == 'bullish':
                take_profit = entry_price + (reward_pips * pip_value)
            else:
                take_profit = entry_price - (reward_pips * pip_value)

            rr_ratio = self.min_rr_ratio

            self.logger.info(f"   ✅ Take Profit: {take_profit:.5f}")
            self.logger.info(f"   ✅ Reward: {reward_pips:.1f} pips")
            self.logger.info(f"   ✅ R:R Ratio: {rr_ratio:.2f}")

            # Calculate partial profit if enabled
            partial_tp = None
            if self.partial_profit_enabled:
                partial_reward_pips = risk_pips * self.partial_profit_rr
                if bos_choch_info['direction'] == 'bullish':
                    partial_tp = entry_price + (partial_reward_pips * pip_value)
                else:
                    partial_tp = entry_price - (partial_reward_pips * pip_value)

                self.logger.info(f"\n💰 Partial Profit:")
                self.logger.info(f"   Partial TP: {partial_tp:.5f}")
                self.logger.info(f"   Partial R:R: {self.partial_profit_rr}")
                self.logger.info(f"   Close Percent: {self.partial_profit_percent}%")

            # Calculate simple confidence score
            # Based on: HTF strength (100%) - no other factors
            confidence = final_strength

            # BUILD SIGNAL
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ VALID SMC STRUCTURE SIGNAL DETECTED (BARE ESSENTIALS)")
            self.logger.info(f"{'='*70}")

            signal = {
                'strategy': 'SMC_STRUCTURE',
                'signal_type': final_trend,
                'signal': final_trend,
                'confidence_score': round(confidence, 2),
                'epic': epic,
                'pair': pair,
                'timeframe': '15m',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'partial_tp': partial_tp,
                'partial_percent': self.partial_profit_percent if self.partial_profit_enabled else None,
                'risk_pips': risk_pips,
                'reward_pips': reward_pips,
                'rr_ratio': rr_ratio,
                'timestamp': datetime.now(),

                # Signal details (SIMPLIFIED)
                'htf_trend': final_trend,
                'htf_strength': final_strength,
                'htf_structure': trend_analysis['structure_type'],
                'bos_direction': bos_choch_info['direction'],
                'bos_level': bos_choch_info['level'],

                # Readable description
                'description': f"{final_trend} trend ({trend_analysis['structure_type']}), {bos_choch_info['direction']} BOS on 15m ({rr_ratio:.1f}R)"
            }

            self.logger.info(f"\n📋 Signal Summary:")
            self.logger.info(f"   Direction: {signal['signal']}")
            self.logger.info(f"   Entry: {signal['entry_price']:.5f}")
            self.logger.info(f"   Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
            self.logger.info(f"   Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
            if partial_tp:
                self.logger.info(f"   Partial TP: {partial_tp:.5f} ({self.partial_profit_percent}% at {self.partial_profit_rr}R)")
            self.logger.info(f"   R:R Ratio: {signal['rr_ratio']:.2f}")
            self.logger.info(f"   Confidence: {signal['confidence_score']*100:.0f}%")
            self.logger.info(f"\n   {signal['description']}")
            self.logger.info(f"{'='*70}\n")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting SMC signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_bos_choch_15m(self, df_15m: pd.DataFrame, epic: str) -> Optional[Dict]:
        """
        Detect BOS/CHoCH on 15m timeframe (SIMPLIFIED)

        Args:
            df_15m: 15m DataFrame
            epic: Currency pair

        Returns:
            Dict with BOS/CHoCH info or None if no break detected
        """
        # Analyze market structure
        df_15m_with_structure = self.market_structure.analyze_market_structure(
            df=df_15m,
            config=vars(self.config) if hasattr(self.config, '__dict__') else {},
            epic=epic,
            timeframe='15m'
        )

        # Get BOS/CHoCH direction
        bos_choch_direction = self.market_structure.get_last_bos_choch_direction(df_15m_with_structure)

        if not bos_choch_direction:
            return None

        # Get current price for level
        current_price = float(df_15m['close'].iloc[-1])

        return {
            'type': 'BOS',
            'direction': bos_choch_direction,
            'level': current_price,
            'timestamp': df_15m.index[-1] if hasattr(df_15m.index[-1], 'to_pydatetime') else None
        }

    def _validate_htf_alignment(self, bos_direction: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame, epic: str) -> bool:
        """
        Validate that 1H and 4H timeframes align with BOS/CHoCH direction

        REVERTED: Back to v2.5.0 baseline (strict HTF alignment)
        Priority 1 fix (allowing RANGING) caused catastrophic regression:
        - Profit Factor: 1.90 → 0.23 (-88%)
        - Win Rate: 56.3% → 35.3% (-37%)
        - Expectancy: +3.2 → -16.8 pips

        Args:
            bos_direction: 'bullish' or 'bearish' from BOS/CHoCH
            df_1h: 1H DataFrame
            df_4h: 4H DataFrame
            epic: Currency pair

        Returns:
            True if HTF alignment confirmed, False otherwise
        """
        self.logger.info(f"   Checking HTF alignment for {bos_direction} BOS...")

        # Check 1H alignment (if required)
        if self.require_1h_alignment:
            trend_1h = self.trend_analyzer.analyze_trend(
                df=df_1h,
                epic=epic,
                lookback=self.htf_alignment_lookback
            )

            expected_trend_1h = 'BULL' if bos_direction == 'bullish' else 'BEAR'

            if trend_1h['trend'] != expected_trend_1h:
                self.logger.info(f"      ❌ 1H trend mismatch: {trend_1h['trend']} vs expected {expected_trend_1h}")
                return False

            self.logger.info(f"      ✅ 1H aligned: {trend_1h['trend']} ({trend_1h['strength']*100:.0f}%)")

        # Check 4H alignment (if required)
        if self.require_4h_alignment:
            trend_4h = self.trend_analyzer.analyze_trend(
                df=df_4h,
                epic=epic,
                lookback=self.htf_alignment_lookback
            )

            expected_trend_4h = 'BULL' if bos_direction == 'bullish' else 'BEAR'

            if trend_4h['trend'] != expected_trend_4h:
                self.logger.info(f"      ❌ 4H trend mismatch: {trend_4h['trend']} vs expected {expected_trend_4h}")
                return False

            self.logger.info(f"      ✅ 4H aligned: {trend_4h['trend']} ({trend_4h['strength']*100:.0f}%)")

        return True

    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return "SMC_STRUCTURE"

    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return "Pure structure-based strategy using Smart Money Concepts (v2.5.2 - Improved Swing Detection)"


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
        from configdata.strategies import config_smc_structure as config

    return SMCStructureStrategy(config=config, **kwargs)
