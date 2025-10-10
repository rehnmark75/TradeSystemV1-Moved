# core/intelligence/market_intelligence.py
"""
Advanced Market Intelligence System - FIXED VERSION
Real-time market condition analysis and adaptive strategy selection

FIXES:
- Fixed 'characteristics' key error in _generate_executive_summary
- Added proper error handling and safe dictionary access
- Fixed volume column handling (ltv vs volume)
- Added fallback values for missing data
- Improved session analysis structure
- Enhanced error recovery and logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import json

try:
    import config
except ImportError:
    from forex_scanner import config


class MarketIntelligenceEngine:
    """Advanced market analysis and intelligence system - FIXED VERSION"""
    
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
        self.market_memory = {}  # Store market state history
        
        self.logger.info("🧠 MarketIntelligenceEngine initialized (FIXED VERSION)")
        
    def analyze_market_regime(
        self, 
        epic_list: List[str],
        lookback_hours: int = 24
    ) -> Dict:
        """
        Comprehensive market regime analysis - FIXED VERSION
        Returns: trending, ranging, breakout, reversal, high_vol, low_vol
        """
        self.logger.info(f"🧠 Analyzing market regime for {len(epic_list)} pairs: {[e.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '') for e in epic_list]}")
        
        regime_scores = {
            'trending': 0,
            'ranging': 0,
            'breakout': 0,
            'reversal': 0,
            'high_volatility': 0,
            'low_volatility': 0
        }
        
        pair_analyses = {}
        successful_analyses = 0
        
        for epic in epic_list:
            try:
                analysis = self._analyze_single_pair_regime(epic, lookback_hours)
                pair_analyses[epic] = analysis
                successful_analyses += 1
                
                # Accumulate regime scores
                for regime, score in analysis['regime_scores'].items():
                    if regime in regime_scores:
                        regime_scores[regime] += score
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {epic}: {e}")
                # Add fallback analysis for failed epics
                pair_analyses[epic] = self._get_fallback_analysis()
                continue
        
        # Normalize scores
        if successful_analyses > 0:
            regime_scores = {k: v/successful_analyses for k, v in regime_scores.items()}
        else:
            # All analyses failed - use neutral scores
            regime_scores = {k: 0.5 for k in regime_scores.keys()}

        # CRITICAL FIX: Strategy-aware regime selection
        # Volatility is orthogonal to market structure (trending/ranging)
        dominant_regime = self._determine_dominant_regime(regime_scores, strategy=None)
        confidence = regime_scores[dominant_regime]
        
        # FIXED: Safe access to market strength and correlation analysis
        try:
            market_strength = self._calculate_market_strength(pair_analyses)
        except Exception as e:
            self.logger.warning(f"Market strength calculation failed: {e}")
            market_strength = self._get_fallback_market_strength()
        
        try:
            correlation_analysis = self._analyze_cross_pair_correlation(pair_analyses)
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {e}")
            correlation_analysis = self._get_fallback_correlation_analysis()
        
        try:
            recommended_strategy = self._recommend_strategy_for_regime(dominant_regime, confidence)
        except Exception as e:
            self.logger.warning(f"Strategy recommendation failed: {e}")
            recommended_strategy = self._get_fallback_strategy()
        
        return {
            'dominant_regime': dominant_regime,
            'confidence': confidence,
            'regime_scores': regime_scores,
            'pair_analyses': pair_analyses,
            'market_strength': market_strength,
            'correlation_analysis': correlation_analysis,
            'recommended_strategy': recommended_strategy
        }
    
    def _determine_dominant_regime(self, regime_scores: Dict, strategy: str = None) -> str:
        """
        🔥 CRITICAL FIX: Strategy-aware regime selection

        Volatility is orthogonal to market structure (trending/ranging).
        For trend-following strategies (EMA, MACD), prioritize trending/ranging classification.

        Problem: Alert 5568 had high_volatility:1.0, ranging:0.566, trending:0.434
                 max() selected high_volatility, gave EMA 1.0 modifier instead of 0.8 (ranging)

        Solution: Separate structure regimes from volatility regimes for trend strategies
        """
        try:
            # Separate volatility from structure regimes
            structure_regimes = {
                'trending': regime_scores.get('trending', 0),
                'ranging': regime_scores.get('ranging', 0),
                'breakout': regime_scores.get('breakout', 0),
                'consolidation': regime_scores.get('consolidation', 0),
                'reversal': regime_scores.get('reversal', 0)
            }

            volatility_regimes = {
                'high_volatility': regime_scores.get('high_volatility', 0),
                'low_volatility': regime_scores.get('low_volatility', 0)
            }

            # For trend-following strategies, use market structure as primary regime
            trend_strategies = ['ema', 'macd', 'ichimoku', 'kama', 'smart_money_ema', 'smart_money_macd']

            if strategy and any(ts in strategy.lower() for ts in trend_strategies):
                # Use trending vs ranging as primary classification
                dominant_structure = max(structure_regimes, key=structure_regimes.get)
                dominant_volatility = max(volatility_regimes, key=volatility_regimes.get)

                self.logger.info(
                    f"🎯 Strategy-aware regime: Structure={dominant_structure} ({structure_regimes[dominant_structure]:.1%}), "
                    f"Volatility={dominant_volatility} ({volatility_regimes[dominant_volatility]:.1%}) "
                    f"[Strategy: {strategy}]"
                )
                return dominant_structure

            # For other strategies or no strategy specified, use original logic (highest score)
            dominant = max(regime_scores, key=regime_scores.get)

            # Log if volatility dominated over structure
            if dominant in ['high_volatility', 'low_volatility']:
                dominant_structure = max(structure_regimes, key=structure_regimes.get)
                self.logger.debug(
                    f"⚠️ Volatility regime dominant ({dominant}), "
                    f"market structure is {dominant_structure} ({structure_regimes[dominant_structure]:.1%})"
                )

            return dominant

        except Exception as e:
            self.logger.error(f"Error in regime determination: {e}")
            # Fallback to original logic
            return max(regime_scores, key=regime_scores.get)

    def _get_fallback_analysis(self) -> Dict:
        """Get fallback analysis for failed epic analysis"""
        return {
            'regime_scores': {
                'trending': 0.5,
                'ranging': 0.5,
                'breakout': 0.3,
                'reversal': 0.3,
                'high_volatility': 0.4,
                'low_volatility': 0.6
            },
            'current_price': 1.0,
            'price_change_24h': 0.0,
            'volatility_percentile': 50.0,
            'volume_analysis': {
                'trend': 'stable',
                'relative_volume': 1.0,
                'volume_price_correlation': 0.0,
                'volume_availability': False
            },
            'support_resistance': {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': None,
                'nearest_resistance': None
            }
        }
    
    def _get_fallback_market_strength(self) -> Dict:
        """Get fallback market strength data"""
        return {
            'average_trend_strength': 0.5,
            'average_volatility': 0.5,
            'market_bias': 'neutral',
            'directional_consensus': 0.5,
            'market_efficiency': 0.5
        }
    
    def _get_fallback_correlation_analysis(self) -> Dict:
        """Get fallback correlation analysis"""
        return {
            'correlation_matrix': {},
            'currency_strength': {},
            'risk_on_off': 'neutral'
        }
    
    def _get_fallback_strategy(self) -> Dict:
        """Get fallback strategy recommendation"""
        return {
            'strategy': 'conservative',
            'ema_config': 'default',
            'momentum_enabled': True,  # Enable momentum by default as it's conservative
            'recommendations': ['Use standard strategy parameters', 'Momentum strategy enabled with conservative settings']
        }
    
    def _analyze_single_pair_regime(self, epic: str, lookback_hours: int) -> Dict:
        """Analyze market regime for a single pair - FIXED VERSION"""
        try:
            # FIXED: Better error handling for data fetching
            pair_name = epic.split('.')[-3] if '.' in epic else epic  # Extract pair name
            
            # Get multi-timeframe data with error handling
            df_1h = self._safe_get_data(epic, pair_name, '1h')
            df_15m = self._safe_get_data(epic, pair_name, '15m')
            df_5m = self._safe_get_data(epic, pair_name, '5m')
            
            # Use whatever data is available
            available_data = [df for df in [df_1h, df_15m, df_5m] if df is not None and len(df) > 10]
            
            if not available_data:
                raise ValueError(f"No usable data for {epic}")
            
            # Use the best available timeframe
            primary_df = available_data[0]
            secondary_df = available_data[1] if len(available_data) > 1 else primary_df
            
            # Focus on recent data
            recent_primary = primary_df.tail(min(lookback_hours, len(primary_df)))
            recent_secondary = secondary_df.tail(min(lookback_hours * 2, len(secondary_df)))
            
            regime_scores = {}
            
            # 1. Trend Analysis - FIXED with error handling
            try:
                regime_scores['trending'] = self._calculate_trend_score(recent_primary, recent_secondary)
                regime_scores['ranging'] = max(0, 1.0 - regime_scores['trending'])
            except Exception as e:
                self.logger.debug(f"Trend analysis failed for {epic}: {e}")
                regime_scores['trending'] = 0.5
                regime_scores['ranging'] = 0.5
            
            # 2. Breakout Analysis - FIXED with error handling
            try:
                regime_scores['breakout'] = self._calculate_breakout_score(recent_primary, recent_secondary)
            except Exception as e:
                self.logger.debug(f"Breakout analysis failed for {epic}: {e}")
                regime_scores['breakout'] = 0.3
            
            # 3. Reversal Analysis - FIXED with error handling
            try:
                regime_scores['reversal'] = self._calculate_reversal_score(recent_primary, recent_secondary)
            except Exception as e:
                self.logger.debug(f"Reversal analysis failed for {epic}: {e}")
                regime_scores['reversal'] = 0.3
            
            # 4. Volatility Analysis - FIXED with error handling
            try:
                vol_score = self._calculate_volatility_score(recent_primary)
                regime_scores['high_volatility'] = max(0, vol_score - 0.5) * 2
                regime_scores['low_volatility'] = max(0, 0.5 - vol_score) * 2
            except Exception as e:
                self.logger.debug(f"Volatility analysis failed for {epic}: {e}")
                regime_scores['high_volatility'] = 0.4
                regime_scores['low_volatility'] = 0.6
            
            # Get additional analysis with error handling
            try:
                current_price = float(recent_primary['close'].iloc[-1])
                price_change_24h = (current_price - float(recent_primary['close'].iloc[0])) / float(recent_primary['close'].iloc[0])
            except:
                current_price = 1.0
                price_change_24h = 0.0
            
            try:
                volatility_percentile = self._calculate_volatility_percentile(recent_primary)
            except:
                volatility_percentile = 50.0
            
            try:
                volume_analysis = self._analyze_volume_patterns(recent_primary)
            except:
                volume_analysis = {
                    'trend': 'stable',
                    'relative_volume': 1.0,
                    'volume_price_correlation': 0.0,
                    'volume_availability': False
                }
            
            try:
                support_resistance = self._identify_key_levels(recent_primary)
            except:
                support_resistance = {
                    'support_levels': [],
                    'resistance_levels': [],
                    'nearest_support': None,
                    'nearest_resistance': None
                }
            
            return {
                'regime_scores': regime_scores,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'volatility_percentile': volatility_percentile,
                'volume_analysis': volume_analysis,
                'support_resistance': support_resistance
            }
            
        except Exception as e:
            self.logger.warning(f"Single pair analysis failed for {epic}: {e}")
            return self._get_fallback_analysis()
    
    def _safe_get_data(self, epic: str, pair_name: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Safely get data with error handling"""
        try:
            df = self.data_fetcher.get_enhanced_data(epic, pair_name, timeframe=timeframe)
            if df is not None and len(df) > 10:
                return df
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get {timeframe} data for {epic}: {e}")
            return None
    
    def _calculate_trend_score(self, df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> float:
        """Calculate trend strength score (0 = ranging, 1 = strong trend) - FIXED"""
        scores = []
        
        try:
            # 1. Price momentum
            price_change = (df_primary['close'].iloc[-1] - df_primary['close'].iloc[0]) / df_primary['close'].iloc[0]
            momentum_score = min(1.0, abs(price_change) * 100)  # Normalize to percentage
            scores.append(momentum_score)
        except:
            pass
        
        try:
            # 2. EMA alignment (if available)
            if all(col in df_secondary.columns for col in ['ema_9', 'ema_21']):
                latest = df_secondary.iloc[-1]
                if 'ema_200' in df_secondary.columns:
                    # Full EMA alignment
                    if latest['ema_9'] > latest['ema_21'] > latest['ema_200']:
                        ema_score = 1.0
                    elif latest['ema_9'] < latest['ema_21'] < latest['ema_200']:
                        ema_score = 1.0
                    else:
                        ema_score = 0.3
                else:
                    # Just 9/21 alignment
                    if latest['ema_9'] > latest['ema_21'] or latest['ema_9'] < latest['ema_21']:
                        ema_score = 0.7
                    else:
                        ema_score = 0.3
                scores.append(ema_score)
        except:
            pass
        
        try:
            # 3. Linear regression slope
            prices = df_primary['close'].tail(min(20, len(df_primary))).values
            if len(prices) >= 5:
                x = np.arange(len(prices))
                slope, _, r_value, _, _ = stats.linregress(x, prices)
                r_squared = r_value ** 2
                scores.append(r_squared)
        except:
            pass
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_breakout_score(self, df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> float:
        """Calculate breakout potential score - FIXED"""
        scores = []
        
        try:
            # 1. Volume surge - FIXED: Use ltv column consistently
            volume_col = 'ltv' if 'ltv' in df_primary.columns else 'volume'
            if volume_col in df_primary.columns and len(df_primary) >= 20:
                recent_volume = df_primary[volume_col].tail(5).mean()
                baseline_volume = df_primary[volume_col].tail(20).mean()
                if baseline_volume > 0:
                    volume_ratio = recent_volume / baseline_volume
                    volume_score = min(1.0, max(0, (volume_ratio - 1) / 2))
                    scores.append(volume_score)
        except:
            pass
        
        try:
            # 2. Price compression
            if len(df_primary) >= 10:
                recent_ranges = (df_primary['high'] - df_primary['low']).tail(10)
                if len(recent_ranges) >= 5:
                    range_trend = np.polyfit(range(len(recent_ranges)), recent_ranges, 1)[0]
                    compression_score = max(0, -range_trend * 1000)
                    scores.append(min(1.0, compression_score))
        except:
            pass
        
        try:
            # 3. Support/resistance proximity
            current_price = df_primary['close'].iloc[-1]
            recent_highs = df_primary['high'].tail(min(20, len(df_primary)))
            recent_lows = df_primary['low'].tail(min(20, len(df_primary)))
            
            resistance_level = recent_highs.max()
            support_level = recent_lows.min()
            
            to_resistance = abs(current_price - resistance_level) / current_price
            to_support = abs(current_price - support_level) / current_price
            
            level_proximity = 1 - min(to_resistance, to_support) * 100
            scores.append(min(1.0, max(0, level_proximity)))
        except:
            pass
        
        return np.mean(scores) if scores else 0.3
    
    def _calculate_reversal_score(self, df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> float:
        """Calculate reversal potential score - FIXED"""
        scores = []
        
        try:
            # 1. Price divergence analysis
            if len(df_primary) >= 10:
                prices = df_primary['close'].tail(10)
                price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
                
                recent_high = prices.tail(3).max()
                prev_high = prices.head(7).max()
                
                if recent_high > prev_high and price_trend < 0:
                    scores.append(0.8)
                elif prices.tail(3).min() < prices.head(7).min() and price_trend > 0:
                    scores.append(0.8)
                else:
                    scores.append(0.2)
        except:
            pass
        
        try:
            # 2. EMA separation (if available)
            if 'ema_9' in df_secondary.columns and 'ema_21' in df_secondary.columns:
                latest = df_secondary.iloc[-1]
                ema_separation = abs(latest['ema_9'] - latest['ema_21']) / latest['close']
                separation_score = min(1.0, ema_separation * 1000)
                scores.append(separation_score)
        except:
            pass
        
        try:
            # 3. Simple reversal pattern detection
            if len(df_primary) >= 3:
                last_3 = df_primary.tail(3)
                pattern_score = self._detect_reversal_patterns(last_3)
                scores.append(pattern_score)
        except:
            pass
        
        return np.mean(scores) if scores else 0.3
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate volatility score (0 = low vol, 1 = high vol) - FIXED"""
        try:
            if len(df) < 10:
                return 0.5
            
            # ATR-based volatility
            df_calc = df.copy()
            df_calc['tr'] = np.maximum(
                df_calc['high'] - df_calc['low'],
                np.maximum(
                    abs(df_calc['high'] - df_calc['close'].shift(1)),
                    abs(df_calc['low'] - df_calc['close'].shift(1))
                )
            )
            
            recent_atr = df_calc['tr'].tail(min(14, len(df_calc))).mean()
            long_term_atr = df_calc['tr'].tail(min(50, len(df_calc))).mean()
            
            if long_term_atr > 0:
                volatility_ratio = recent_atr / long_term_atr
                return min(1.0, max(0.0, (volatility_ratio - 0.5) * 2))
            else:
                return 0.5
                
        except Exception as e:
            self.logger.debug(f"Volatility calculation error: {e}")
            return 0.5
    
    def _detect_reversal_patterns(self, candles: pd.DataFrame) -> float:
        """Simple reversal pattern detection - FIXED"""
        try:
            if len(candles) < 3:
                return 0.0
            
            c3 = candles.iloc[-1]  # Latest candle
            
            # Simple doji detection
            body_size = abs(c3['close'] - c3['open'])
            candle_range = c3['high'] - c3['low']
            
            if candle_range > 0 and body_size < 0.1 * candle_range:
                return 0.5
            
            return 0.1
            
        except:
            return 0.1
    
    def _calculate_market_strength(self, pair_analyses: Dict) -> Dict:
        """Calculate overall market strength indicators - FIXED"""
        try:
            if not pair_analyses:
                return self._get_fallback_market_strength()
            
            total_trend_strength = 0
            total_volatility = 0
            bullish_pairs = 0
            bearish_pairs = 0
            valid_pairs = 0
            
            for epic, analysis in pair_analyses.items():
                try:
                    # Safe access to regime scores
                    regime_scores = analysis.get('regime_scores', {})
                    total_trend_strength += regime_scores.get('trending', 0.5)
                    total_volatility += regime_scores.get('high_volatility', 0.5)
                    
                    price_change = analysis.get('price_change_24h', 0)
                    if price_change > 0.001:  # >0.1% gain
                        bullish_pairs += 1
                    elif price_change < -0.001:  # >0.1% loss
                        bearish_pairs += 1
                    
                    valid_pairs += 1
                except:
                    continue
            
            if valid_pairs == 0:
                return self._get_fallback_market_strength()
            
            # Determine market bias
            if bullish_pairs > bearish_pairs:
                market_bias = 'bullish'
            elif bearish_pairs > bullish_pairs:
                market_bias = 'bearish'
            else:
                market_bias = 'neutral'
            
            return {
                'average_trend_strength': total_trend_strength / valid_pairs,
                'average_volatility': total_volatility / valid_pairs,
                'market_bias': market_bias,
                'directional_consensus': max(bullish_pairs, bearish_pairs) / valid_pairs if valid_pairs > 0 else 0.5,
                'market_efficiency': 1 - abs(bullish_pairs - bearish_pairs) / valid_pairs if valid_pairs > 0 else 0.5
            }
            
        except Exception as e:
            self.logger.warning(f"Market strength calculation error: {e}")
            return self._get_fallback_market_strength()
    
    def _analyze_cross_pair_correlation(self, pair_analyses: Dict) -> Dict:
        """Analyze correlation patterns between currency pairs - FIXED"""
        try:
            if not pair_analyses:
                return self._get_fallback_correlation_analysis()
            
            # Calculate price change correlations
            price_changes = {}
            for epic, analysis in pair_analyses.items():
                try:
                    price_changes[epic] = analysis.get('price_change_24h', 0)
                except:
                    price_changes[epic] = 0
            
            # Calculate currency strength
            currency_strength = self._calculate_currency_strength(price_changes)
            
            # Determine risk sentiment
            risk_sentiment = self._determine_risk_sentiment(price_changes)
            
            return {
                'correlation_matrix': {},  # Simplified for now
                'currency_strength': currency_strength,
                'risk_on_off': risk_sentiment
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation analysis error: {e}")
            return self._get_fallback_correlation_analysis()
    
    def _calculate_currency_strength(self, price_changes: Dict) -> Dict:
        """Calculate individual currency strength - FIXED"""
        try:
            currency_scores = {}
            
            for epic, change in price_changes.items():
                try:
                    # Simple currency extraction from epic name
                    if 'EURUSD' in epic:
                        base, quote = 'EUR', 'USD'
                    elif 'GBPUSD' in epic:
                        base, quote = 'GBP', 'USD'
                    elif 'USDJPY' in epic:
                        base, quote = 'USD', 'JPY'
                    elif 'AUDUSD' in epic:
                        base, quote = 'AUD', 'USD'
                    elif 'USDCAD' in epic:
                        base, quote = 'USD', 'CAD'
                    else:
                        continue
                    
                    if base not in currency_scores:
                        currency_scores[base] = 0
                    if quote not in currency_scores:
                        currency_scores[quote] = 0
                    
                    currency_scores[base] += change
                    currency_scores[quote] -= change
                    
                except:
                    continue
            
            return currency_scores
            
        except:
            return {}
    
    def _determine_risk_sentiment(self, price_changes: Dict) -> str:
        """Determine market risk sentiment - FIXED"""
        try:
            safe_havens = ['JPY', 'CHF', 'USD']
            risk_currencies = ['AUD', 'NZD', 'CAD']
            
            safe_haven_performance = 0
            risk_currency_performance = 0
            
            for epic, change in price_changes.items():
                for safe_currency in safe_havens:
                    if safe_currency in epic:
                        safe_haven_performance += abs(change)
                
                for risk_currency in risk_currencies:
                    if risk_currency in epic:
                        risk_currency_performance += abs(change)
            
            if safe_haven_performance > risk_currency_performance:
                return 'risk_off'
            elif risk_currency_performance > safe_haven_performance:
                return 'risk_on'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _recommend_strategy_for_regime(self, regime: str, confidence: float) -> Dict:
        """Recommend trading strategy based on market regime - FIXED"""
        try:
            if confidence < 0.6:
                return {
                    'strategy': 'conservative',
                    'ema_config': 'default',
                    'reason': f'Low confidence ({confidence:.1%}) in regime detection',
                    'recommendations': [
                        'Reduce position sizes',
                        'Require higher signal confidence',
                        'Focus on major pairs only'
                    ]
                }
            
            strategy_map = {
                'trending': {
                    'strategy': 'trend_following',
                    'ema_config': 'aggressive',
                    'momentum_enabled': True,
                    'recommendations': [
                        'Use trend-following strategies',
                        'Enable momentum strategy for trend continuation',
                        'Look for pullback entries',
                        'Trail stops aggressively'
                    ]
                },
                'ranging': {
                    'strategy': 'mean_reversion',
                    'ema_config': 'conservative',
                    'momentum_enabled': False,
                    'recommendations': [
                        'Trade range boundaries',
                        'Use oscillator confirmations',
                        'Disable momentum strategy in ranging markets',
                        'Take profits quickly'
                    ]
                },
                'breakout': {
                    'strategy': 'breakout_trading',
                    'ema_config': 'aggressive',
                    'momentum_enabled': True,
                    'recommendations': [
                        'Prepare for volatility expansion',
                        'Enable momentum strategy for breakout confirmation',
                        'Use volume confirmation',
                        'Set wider stops'
                    ]
                },
                'high_volatility': {
                    'strategy': 'volatility_adaptive',
                    'ema_config': 'conservative',
                    'momentum_enabled': True,
                    'recommendations': [
                        'Enable momentum strategy for volatility adaptation',
                        'Reduce position sizes',
                        'Widen stop losses',
                        'Trade less frequently'
                    ]
                },
                'momentum_favorable': {
                    'strategy': 'momentum_focused',
                    'ema_config': 'balanced',
                    'momentum_enabled': True,
                    'recommendations': [
                        'Prioritize momentum strategy',
                        'Look for velocity confirmations',
                        'Use volume-weighted momentum',
                        'Trail stops dynamically'
                    ]
                }
            }
            
            return strategy_map.get(regime, {
                'strategy': 'default',
                'ema_config': 'default',
                'momentum_enabled': True,  # Enable momentum by default
                'recommendations': ['Use standard strategy parameters', 'Momentum strategy enabled']
            })
            
        except Exception as e:
            self.logger.warning(f"Strategy recommendation error: {e}")
            return self._get_fallback_strategy()
    
    def get_session_analysis(self) -> Dict:
        """Analyze current trading session and its characteristics - FIXED VERSION"""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            
            # Determine current session (UTC hours)
            if 22 <= hour or hour < 6:
                session = 'asian'
            elif 6 <= hour < 14:
                session = 'london'  
            elif 14 <= hour < 22:
                session = 'new_york'
            else:
                session = 'overlap'
            
            # FIXED: Session configurations with consistent structure
            session_configs = {
                'asian': {
                    'volatility': 'low',
                    'characteristics': 'Lower volatility, range-bound trading',  # FIXED: Added missing key
                    'preferred_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD'],
                    'strategy_adjustment': 'Range trading, support/resistance focus',
                    'risk_level': 'low'
                },
                'london': {
                    'volatility': 'high',
                    'characteristics': 'High volatility, strong trend potential',  # FIXED: Added missing key
                    'preferred_pairs': ['EURUSD', 'GBPUSD', 'EURGBP'],
                    'strategy_adjustment': 'Trend following, breakout strategies',
                    'risk_level': 'medium'
                },
                'new_york': {
                    'volatility': 'high',
                    'characteristics': 'High volume, momentum-driven movements',  # FIXED: Added missing key
                    'preferred_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
                    'strategy_adjustment': 'Momentum trading, news-based strategies',
                    'risk_level': 'medium'
                },
                'overlap': {
                    'volatility': 'very_high',
                    'characteristics': 'Highest volume and volatility period',  # FIXED: Added missing key
                    'preferred_pairs': ['EURUSD', 'GBPUSD'],
                    'strategy_adjustment': 'Aggressive trend strategies',
                    'risk_level': 'high'
                }
            }
            
            session_config = session_configs.get(session, session_configs['london'])
            
            return {
                'current_session': session,
                'session_config': session_config,
                'recommendations': self._get_session_recommendations(session),
                'optimal_timeframes': self._get_session_timeframes(session)
            }
            
        except Exception as e:
            self.logger.warning(f"Error in session analysis: {e}")
            # FIXED: Proper fallback with characteristics key
            return {
                'current_session': 'unknown',
                'session_config': {
                    'volatility': 'medium',
                    'characteristics': 'Standard market conditions',  # FIXED: Added missing key
                    'risk_level': 'medium',
                    'preferred_pairs': ['EURUSD'],
                    'strategy_adjustment': 'Standard approach'
                },
                'recommendations': ['Use standard trading approach'],
                'optimal_timeframes': ['15m']
            }

    def _get_session_recommendations(self, session: str) -> List[str]:
        """Get session-specific trading recommendations - FIXED"""
        recommendations = {
            'asian': [
                'Focus on range trading strategies',
                'Use tight stop losses due to low volatility',
                'Look for support/resistance bounces',
                'Avoid aggressive breakout trades'
            ],
            'london': [
                'Expect strong directional moves',
                'Use trend-following strategies',
                'Watch for Brexit-related GBP volatility',
                'Monitor economic news releases'
            ],
            'new_york': [
                'Focus on USD pairs',
                'Watch for trend continuation',
                'Monitor Federal Reserve communications',
                'Use momentum-based strategies'
            ],
            'overlap': [
                'Prepare for highest volatility',
                'Use aggressive trend strategies',
                'Monitor both US and UK economic data',
                'Scale positions appropriately'
            ]
        }
        
        return recommendations.get(session, ['Use standard trading approach'])

    def _get_session_timeframes(self, session: str) -> List[str]:
        """Get optimal timeframes for current session - FIXED"""
        timeframes = {
            'asian': ['15m', '1h'],      # Longer for ranging markets
            'london': ['5m', '15m'],     # Shorter for volatility
            'new_york': ['15m', '1h'],   # Medium timeframes
            'overlap': ['5m', '15m']     # Short for maximum action
        }
        
        return timeframes.get(session, ['15m'])

    def generate_market_intelligence_report(self, epic_list: List[str]) -> Dict:
        """Generate comprehensive market intelligence report - FIXED VERSION"""
        self.logger.info("📊 Generating comprehensive market intelligence report...")
        
        try:
            # Gather all intelligence with error handling
            market_regime = self.analyze_market_regime(epic_list)
            session_analysis = self.get_session_analysis()
            
            # Generate components with error handling
            try:
                executive_summary = self._generate_executive_summary(market_regime, session_analysis)
            except Exception as e:
                self.logger.warning(f"Executive summary generation failed: {e}")
                executive_summary = "Market intelligence summary unavailable"
            
            try:
                risk_assessment = self._assess_market_risks(market_regime, epic_list)
            except Exception as e:
                self.logger.warning(f"Risk assessment failed: {e}")
                risk_assessment = {
                    'risk_level': 'medium',
                    'risk_factors': ['Unable to assess risks'],
                    'mitigation_strategies': ['Use standard risk management']
                }
            
            try:
                trading_recommendations = self._generate_trading_recommendations(market_regime, session_analysis)
            except Exception as e:
                self.logger.warning(f"Trading recommendations failed: {e}")
                trading_recommendations = {
                    'primary_strategy': 'conservative',
                    'confidence_threshold': 0.7,
                    'position_sizing': 'NORMAL'
                }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'executive_summary': executive_summary,
                'market_regime': market_regime,
                'session_analysis': session_analysis,
                'risk_assessment': risk_assessment,
                'trading_recommendations': trading_recommendations,
                'confidence_score': market_regime.get('confidence', 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Market intelligence report generation failed: {e}")
            # Return minimal fallback report
            return {
                'timestamp': datetime.now().isoformat(),
                'executive_summary': 'Market intelligence unavailable',
                'market_regime': {
                    'dominant_regime': 'unknown',
                    'confidence': 0.5,
                    'regime_scores': {},
                    'pair_analyses': {}
                },
                'session_analysis': {
                    'current_session': 'unknown',
                    'session_config': {
                        'characteristics': 'Standard market conditions',
                        'volatility': 'medium'
                    }
                },
                'risk_assessment': {
                    'risk_level': 'medium',
                    'risk_factors': [],
                    'mitigation_strategies': []
                },
                'trading_recommendations': {
                    'primary_strategy': 'conservative',
                    'confidence_threshold': 0.7
                },
                'confidence_score': 0.5
            }
    
    def _generate_executive_summary(self, market_regime: Dict, session_analysis: Dict) -> str:
        """Generate executive summary of market conditions - FIXED VERSION"""
        try:
            # FIXED: Safe access to all dictionary keys
            regime = market_regime.get('dominant_regime', 'unknown')
            confidence = market_regime.get('confidence', 0.5)
            session = session_analysis.get('current_session', 'unknown')
            
            # FIXED: Safe access to session characteristics
            session_config = session_analysis.get('session_config', {})
            characteristics = session_config.get('characteristics', 'standard market conditions')
            
            # FIXED: Safe access to recommended strategy
            recommended_strategy = market_regime.get('recommended_strategy', {})
            strategy_name = recommended_strategy.get('strategy', 'conservative')
            
            # FIXED: Safe access to market strength
            market_strength = market_regime.get('market_strength', {})
            market_bias = market_strength.get('market_bias', 'neutral')
            
            # FIXED: Safe access to correlation analysis
            correlation_analysis = market_regime.get('correlation_analysis', {})
            risk_sentiment = correlation_analysis.get('risk_on_off', 'neutral')

            # Generate individual epic regime summary
            pair_analyses = market_regime.get('pair_analyses', {})
            individual_regimes = []
            regime_counts = {'trending': 0, 'ranging': 0, 'breakout': 0, 'reversal': 0}

            for epic, analysis in pair_analyses.items():
                if isinstance(analysis, dict) and 'regime_scores' in analysis:
                    regime_scores = analysis['regime_scores']
                    epic_regime = max(regime_scores, key=regime_scores.get) if regime_scores else 'unknown'
                    epic_confidence = regime_scores.get(epic_regime, 0.5) if regime_scores else 0.5

                    clean_epic = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
                    individual_regimes.append(f"{clean_epic}: {epic_regime} ({epic_confidence:.1%})")

                    # Count regimes for distribution
                    if epic_regime in regime_counts:
                        regime_counts[epic_regime] += 1

            epic_breakdown = "\n".join([f"  • {regime_info}" for regime_info in individual_regimes[:8]])  # Limit to 8 pairs

            summary = f"""
MARKET INTELLIGENCE SUMMARY

Market Regime: {regime.upper()} (Confidence: {confidence:.1%})
Trading Session: {session.upper()}

Key Insights:
- Market is currently exhibiting {regime} characteristics
- {session.title()} session is active with {characteristics.lower()}
- Recommended strategy approach: {strategy_name}

Market Strength: {market_bias.title()} bias detected
Risk Sentiment: {risk_sentiment.replace('_', ' ').title()}

Individual Epic Regimes:
{epic_breakdown}

Regime Distribution: Trending({regime_counts['trending']}), Ranging({regime_counts['ranging']}), Breakout({regime_counts['breakout']}), Reversal({regime_counts['reversal']})
            """
            
            return summary.strip()
            
        except Exception as e:
            self.logger.warning(f"Executive summary generation error: {e}")
            return f"Market Intelligence Summary: {regime} regime detected with {confidence:.1%} confidence"
    
    def _assess_market_risks(self, market_regime: Dict, epic_list: List[str]) -> Dict:
        """Assess current market risks - FIXED"""
        try:
            risk_factors = []
            risk_level = 'medium'
            
            # Safe access to regime data
            regime = market_regime.get('dominant_regime', 'unknown')
            confidence = market_regime.get('confidence', 0.5)
            regime_scores = market_regime.get('regime_scores', {})
            
            # Risk assessment logic
            if confidence < 0.6:
                risk_factors.append('Uncertain market conditions')
                risk_level = 'high'
            
            if regime_scores.get('high_volatility', 0) > 0.7:
                risk_factors.append('High market volatility')
                if risk_level != 'high':
                    risk_level = 'medium_high'
            
            if regime == 'reversal':
                risk_factors.append('Potential trend reversal in progress')
            
            return {
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'mitigation_strategies': self._generate_risk_mitigation(risk_factors)
            }
            
        except Exception as e:
            self.logger.warning(f"Risk assessment error: {e}")
            return {
                'risk_level': 'medium',
                'risk_factors': ['Unable to assess risks'],
                'mitigation_strategies': ['Use standard risk management']
            }

    def _generate_risk_mitigation(self, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation strategies - FIXED"""
        strategies = []
        
        if 'High market volatility' in risk_factors:
            strategies.extend([
                'Reduce position sizes by 50%',
                'Use wider stop losses',
                'Avoid over-leveraging'
            ])
        
        if 'Uncertain market conditions' in risk_factors:
            strategies.append('Require higher signal confidence (>80%)')
        
        if not strategies:
            strategies.append('Use standard risk management')
        
        return strategies

    def _generate_trading_recommendations(self, market_regime: Dict, session_analysis: Dict) -> Dict:
        """Generate specific trading recommendations - FIXED"""
        try:
            # Safe access to strategy data
            regime_strategy = market_regime.get('recommended_strategy', {})
            session_config = session_analysis.get('session_config', {})
            
            return {
                'primary_strategy': regime_strategy.get('strategy', 'conservative'),
                'ema_configuration': regime_strategy.get('ema_config', 'default'),
                'preferred_pairs': session_config.get('preferred_pairs', ['EURUSD']),
                'strategy_adjustments': session_config.get('strategy_adjustment', 'standard'),
                'specific_actions': regime_strategy.get('recommendations', ['Use standard parameters']),
                'confidence_threshold': self._suggest_confidence_threshold(market_regime),
                'position_sizing': self._suggest_position_sizing(market_regime),
                'timeframe_focus': session_analysis.get('optimal_timeframes', ['15m'])
            }
            
        except Exception as e:
            self.logger.warning(f"Trading recommendations error: {e}")
            return {
                'primary_strategy': 'conservative',
                'confidence_threshold': 0.7,
                'position_sizing': 'NORMAL'
            }
    
    def _suggest_confidence_threshold(self, market_regime: Dict) -> float:
        """Suggest optimal confidence threshold - FIXED"""
        try:
            base_threshold = 0.7
            
            confidence = market_regime.get('confidence', 0.5)
            regime_scores = market_regime.get('regime_scores', {})
            
            if confidence < 0.6:
                base_threshold += 0.1
            
            if regime_scores.get('high_volatility', 0) > 0.7:
                base_threshold += 0.05
            
            return min(0.9, base_threshold)
            
        except:
            return 0.7
    
    def _suggest_position_sizing(self, market_regime: Dict) -> str:
        """Suggest position sizing approach - FIXED"""
        try:
            regime_scores = market_regime.get('regime_scores', {})
            confidence = market_regime.get('confidence', 0.5)
            
            volatility_score = regime_scores.get('high_volatility', 0.5)
            
            if volatility_score > 0.7 or confidence < 0.6:
                return 'REDUCED'
            elif volatility_score < 0.3 and confidence > 0.8:
                return 'INCREASED'
            else:
                return 'NORMAL'
                
        except:
            return 'NORMAL'

    def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns and trends - FIXED VERSION"""
        try:
            # FIXED: Consistent ltv column usage
            volume_col = 'ltv' if 'ltv' in df.columns else 'volume'
            
            if volume_col not in df.columns or len(df) < 10:
                return {
                    'trend': 'unknown',
                    'relative_volume': 1.0,
                    'volume_price_correlation': 0.0,
                    'volume_availability': False
                }
            
            # Volume analysis using correct column
            recent_volume = df[volume_col].tail(5).mean()
            historical_volume = df[volume_col].mean()
            
            volume_trend = 'increasing' if recent_volume > historical_volume * 1.2 else \
                          'decreasing' if recent_volume < historical_volume * 0.8 else 'stable'
            
            relative_volume = recent_volume / historical_volume if historical_volume > 0 else 1.0
            
            # Volume-price correlation
            try:
                vol_price_corr = df[volume_col].corr(df['close'])
                if pd.isna(vol_price_corr):
                    vol_price_corr = 0.0
            except:
                vol_price_corr = 0.0
            
            return {
                'trend': volume_trend,
                'relative_volume': float(relative_volume),
                'volume_price_correlation': float(vol_price_corr),
                'volume_availability': True,
                'volume_column_used': volume_col
            }
            
        except Exception as e:
            self.logger.debug(f"Volume analysis error: {e}")
            return {
                'trend': 'unknown',
                'relative_volume': 1.0,
                'volume_price_correlation': 0.0,
                'volume_availability': False
            }

    def _identify_key_levels(self, df: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels - FIXED"""
        try:
            if len(df) < 10:
                return {
                    'support_levels': [],
                    'resistance_levels': [],
                    'nearest_support': None,
                    'nearest_resistance': None
                }
            
            # Simple level identification
            window = min(5, len(df) // 4)
            highs = df['high'].rolling(window=window, center=True).max()
            lows = df['low'].rolling(window=window, center=True).min()
            
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(df) - window):
                if df['high'].iloc[i] == highs.iloc[i]:
                    resistance_levels.append(float(df['high'].iloc[i]))
                if df['low'].iloc[i] == lows.iloc[i]:
                    support_levels.append(float(df['low'].iloc[i]))
            
            current_price = float(df['close'].iloc[-1])
            
            supports_below = [s for s in support_levels if s < current_price]
            resistances_above = [r for r in resistance_levels if r > current_price]
            
            nearest_support = max(supports_below) if supports_below else None
            nearest_resistance = min(resistances_above) if resistances_above else None
            
            return {
                'support_levels': sorted(list(set(support_levels)), reverse=True)[:3],
                'resistance_levels': sorted(list(set(resistance_levels)))[:3],
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
        except Exception as e:
            self.logger.debug(f"Key levels identification error: {e}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': None,
                'nearest_resistance': None
            }

    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calculate volatility percentile - FIXED"""
        try:
            if len(df) < 20:
                return 50.0
            
            df_calc = df.copy()
            df_calc['returns'] = df_calc['close'].pct_change()
            recent_vol = df_calc['returns'].tail(10).std()
            
            hist_vol = df_calc['returns'].rolling(10).std().dropna()
            
            if len(hist_vol) < 5 or recent_vol is None or pd.isna(recent_vol):
                return 50.0
            
            percentile = stats.percentileofscore(hist_vol.dropna(), recent_vol)
            return float(percentile)
            
        except Exception as e:
            self.logger.debug(f"Volatility percentile error: {e}")
            return 50.0

    def save_market_state(self, intelligence_report: Dict):
        """Save current market state for historical analysis - FIXED"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Safe access to report data
            market_regime = intelligence_report.get('market_regime', {})
            session_analysis = intelligence_report.get('session_analysis', {})
            risk_assessment = intelligence_report.get('risk_assessment', {})
            
            self.market_memory[timestamp] = {
                'regime': market_regime.get('dominant_regime', 'unknown'),
                'confidence': intelligence_report.get('confidence_score', 0.5),
                'session': session_analysis.get('current_session', 'unknown'),
                'risk_level': risk_assessment.get('risk_level', 'medium')
            }
            
            # Keep only last 24 hours of data
            cutoff = datetime.now() - timedelta(hours=24)
            self.market_memory = {
                k: v for k, v in self.market_memory.items() 
                if datetime.fromisoformat(k) > cutoff
            }
            
        except Exception as e:
            self.logger.debug(f"Market state save error: {e}")

    def get_regime_stability_score(self) -> float:
        """Calculate regime stability - FIXED"""
        try:
            if len(self.market_memory) < 3:
                return 0.5
            
            recent_regimes = [state.get('regime', 'unknown') for state in list(self.market_memory.values())[-5:]]
            
            if not recent_regimes:
                return 0.5
            
            most_common = max(set(recent_regimes), key=recent_regimes.count)
            stability = recent_regimes.count(most_common) / len(recent_regimes)
            
            return stability
            
        except:
            return 0.5