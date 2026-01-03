# core/strategies/helpers/smc_premium_discount.py
"""
Smart Money Concepts - Premium/Discount Pricing Model
Implements market maker model for identifying optimal trade entry zones

Premium/Discount Theory:
- Daily/Weekly ranges are divided into zones
- Premium zones (70-100%): Price is expensive, look for sells
- Discount zones (0-30%): Price is cheap, look for buys  
- Equilibrium (40-60%): Neutral zone, avoid entries
- Optimal Trade Entry (OTE): 62%-79% retracement levels

Key Concepts:
- Market makers seek liquidity at premium/discount extremes
- Institutions buy at discount, sell at premium
- Golden ratio (0.618, 0.382) levels for optimal entries
- Session highs/lows create intraday premium/discount zones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class PremiumDiscountZone(Enum):
    """Premium/Discount zone classifications"""
    EXTREME_PREMIUM = "extreme_premium"      # 90-100%
    PREMIUM = "premium"                      # 70-89%
    EQUILIBRIUM_HIGH = "equilibrium_high"    # 60-69%
    EQUILIBRIUM = "equilibrium"              # 40-59%
    EQUILIBRIUM_LOW = "equilibrium_low"      # 31-39%
    DISCOUNT = "discount"                    # 11-30%
    EXTREME_DISCOUNT = "extreme_discount"    # 0-10%


class RangeType(Enum):
    """Types of ranges for premium/discount calculation"""
    DAILY = "daily"
    WEEKLY = "weekly"  
    SESSION = "session"
    SWING = "swing"


@dataclass
class PremiumDiscountLevel:
    """Represents a premium/discount level"""
    zone: PremiumDiscountZone
    percentage: float
    price_level: float
    range_type: RangeType
    range_high: float
    range_low: float
    confidence: float
    session: str = None
    optimal_entry: bool = False


class SMCPremiumDiscount:
    """Smart Money Concepts Premium/Discount Analyzer"""
    
    def __init__(self, logger: logging.Logger = None, data_fetcher=None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
        
        # Golden ratio levels for optimal trade entry
        self.golden_ratios = {
            'fibonacci_618': 0.618,
            'fibonacci_382': 0.382,
            'fibonacci_705': 0.705,  # Optimal Trade Entry zone
            'fibonacci_79': 0.79     # OTE upper bound
        }
        
        # Premium/Discount thresholds
        self.zone_thresholds = {
            PremiumDiscountZone.EXTREME_PREMIUM: (0.90, 1.00),
            PremiumDiscountZone.PREMIUM: (0.70, 0.89),
            PremiumDiscountZone.EQUILIBRIUM_HIGH: (0.60, 0.69),
            PremiumDiscountZone.EQUILIBRIUM: (0.40, 0.59),
            PremiumDiscountZone.EQUILIBRIUM_LOW: (0.31, 0.39),
            PremiumDiscountZone.DISCOUNT: (0.11, 0.30),
            PremiumDiscountZone.EXTREME_DISCOUNT: (0.00, 0.10)
        }
    
    def analyze_premium_discount(
        self, 
        df: pd.DataFrame, 
        config: Dict,
        epic: str = None,
        current_price: float = None
    ) -> Dict:
        """
        Analyze premium/discount levels for current price
        
        Args:
            df: OHLCV DataFrame
            config: SMC configuration
            epic: Epic code for additional data fetching
            current_price: Current market price
            
        Returns:
            Premium/discount analysis results
        """
        try:
            if len(df) < 20:
                return self._get_empty_analysis()
            
            current_price = current_price or df.iloc[-1]['close']
            
            # Get premium/discount analysis for multiple timeframes
            analysis_results = {
                'current_price': current_price,
                'daily_analysis': self._analyze_daily_range(df, current_price, config),
                'weekly_analysis': self._analyze_weekly_range(df, current_price, config, epic),
                'session_analysis': self._analyze_session_range(df, current_price, config),
                'swing_analysis': self._analyze_swing_range(df, current_price, config),
                'optimal_entry_zones': [],
                'market_maker_bias': None,
                'confluence_score': 0.0
            }
            
            # Calculate market maker bias and optimal entry zones
            analysis_results = self._calculate_market_maker_bias(analysis_results, config)
            analysis_results = self._identify_optimal_entry_zones(analysis_results, config)
            analysis_results = self._calculate_pd_confluence_score(analysis_results, config)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Premium/discount analysis failed: {e}")
            return self._get_empty_analysis()
    
    def _analyze_daily_range(self, df: pd.DataFrame, current_price: float, config: Dict) -> Dict:
        """Analyze daily range premium/discount levels"""
        try:
            # Get today's range (or most recent complete session)
            daily_data = self._get_daily_range_data(df)
            
            if not daily_data:
                return None
            
            daily_high = daily_data['high']
            daily_low = daily_data['low']
            daily_range = daily_high - daily_low
            
            if daily_range == 0:
                return None
            
            # Calculate position within daily range
            position_in_range = (current_price - daily_low) / daily_range
            zone = self._classify_premium_discount_zone(position_in_range)
            
            return {
                'range_type': RangeType.DAILY.value,
                'range_high': daily_high,
                'range_low': daily_low,
                'range_size': daily_range,
                'position_percentage': position_in_range,
                'zone': zone.value,
                'confidence': self._calculate_range_confidence(daily_data, 'daily'),
                'golden_ratio_levels': self._calculate_golden_ratio_levels(daily_high, daily_low),
                'optimal_entry_zone': self._is_optimal_entry_zone(position_in_range),
                'market_maker_bias': self._get_market_maker_bias(zone, position_in_range)
            }
            
        except Exception as e:
            self.logger.error(f"Daily range analysis failed: {e}")
            return None
    
    def _analyze_weekly_range(
        self, 
        df: pd.DataFrame, 
        current_price: float, 
        config: Dict,
        epic: str = None
    ) -> Dict:
        """Analyze weekly range premium/discount levels"""
        try:
            # Try to get weekly data if data_fetcher available
            weekly_data = None
            
            if self.data_fetcher and epic:
                weekly_data = self._get_weekly_range_data(epic, config)
            
            # Fallback to calculating from available data
            if not weekly_data:
                weekly_data = self._estimate_weekly_range_from_daily(df)
            
            if not weekly_data:
                return None
            
            weekly_high = weekly_data['high']
            weekly_low = weekly_data['low']  
            weekly_range = weekly_high - weekly_low
            
            if weekly_range == 0:
                return None
            
            # Calculate position within weekly range
            position_in_range = (current_price - weekly_low) / weekly_range
            zone = self._classify_premium_discount_zone(position_in_range)
            
            return {
                'range_type': RangeType.WEEKLY.value,
                'range_high': weekly_high,
                'range_low': weekly_low,
                'range_size': weekly_range,
                'position_percentage': position_in_range,
                'zone': zone.value,
                'confidence': self._calculate_range_confidence(weekly_data, 'weekly'),
                'golden_ratio_levels': self._calculate_golden_ratio_levels(weekly_high, weekly_low),
                'optimal_entry_zone': self._is_optimal_entry_zone(position_in_range),
                'market_maker_bias': self._get_market_maker_bias(zone, position_in_range),
                'weekly_multiplier': config.get('weekly_range_multiplier', 0.382)
            }
            
        except Exception as e:
            self.logger.error(f"Weekly range analysis failed: {e}")
            return None
    
    def _analyze_session_range(self, df: pd.DataFrame, current_price: float, config: Dict) -> Dict:
        """Analyze current session range premium/discount levels"""
        try:
            # Get current session data
            session_data = self._get_current_session_range(df)
            
            if not session_data:
                return None
            
            session_high = session_data['high']
            session_low = session_data['low']
            session_range = session_high - session_low
            session_name = session_data.get('session', 'Unknown')
            
            if session_range == 0:
                return None
            
            # Calculate position within session range
            position_in_range = (current_price - session_low) / session_range
            zone = self._classify_premium_discount_zone(position_in_range)
            
            return {
                'range_type': RangeType.SESSION.value,
                'range_high': session_high,
                'range_low': session_low,
                'range_size': session_range,
                'position_percentage': position_in_range,
                'zone': zone.value,
                'session_name': session_name,
                'confidence': self._calculate_range_confidence(session_data, 'session'),
                'golden_ratio_levels': self._calculate_golden_ratio_levels(session_high, session_low),
                'optimal_entry_zone': self._is_optimal_entry_zone(position_in_range),
                'market_maker_bias': self._get_market_maker_bias(zone, position_in_range)
            }
            
        except Exception as e:
            self.logger.error(f"Session range analysis failed: {e}")
            return None
    
    def _analyze_swing_range(self, df: pd.DataFrame, current_price: float, config: Dict) -> Dict:
        """Analyze recent swing range premium/discount levels"""
        try:
            # Get recent swing high/low range
            swing_data = self._get_recent_swing_range(df, config.get('swing_length', 5))
            
            if not swing_data:
                return None
            
            swing_high = swing_data['high']
            swing_low = swing_data['low']
            swing_range = swing_high - swing_low
            
            if swing_range == 0:
                return None
            
            # Calculate position within swing range
            position_in_range = (current_price - swing_low) / swing_range
            zone = self._classify_premium_discount_zone(position_in_range)
            
            return {
                'range_type': RangeType.SWING.value,
                'range_high': swing_high,
                'range_low': swing_low,
                'range_size': swing_range,
                'position_percentage': position_in_range,
                'zone': zone.value,
                'confidence': self._calculate_range_confidence(swing_data, 'swing'),
                'golden_ratio_levels': self._calculate_golden_ratio_levels(swing_high, swing_low),
                'optimal_entry_zone': self._is_optimal_entry_zone(position_in_range),
                'market_maker_bias': self._get_market_maker_bias(zone, position_in_range),
                'swing_period': swing_data.get('period', 20)
            }
            
        except Exception as e:
            self.logger.error(f"Swing range analysis failed: {e}")
            return None
    
    def _get_daily_range_data(self, df: pd.DataFrame) -> Dict:
        """Get daily range high/low data"""
        try:
            # Use recent 24-hour period or available data
            recent_period = min(24 * 4, len(df))  # 24 hours in 15m bars, or available data
            recent_data = df.tail(recent_period)
            
            return {
                'high': recent_data['high'].max(),
                'low': recent_data['low'].min(),
                'period': len(recent_data),
                'confidence': min(len(recent_data) / (24 * 4), 1.0)  # Full confidence for 24h data
            }
            
        except Exception:
            return None
    
    def _get_weekly_range_data(self, epic: str, config: Dict) -> Dict:
        """Get weekly range data using data_fetcher"""
        try:
            if not self.data_fetcher:
                return None
            
            # Extract pair from epic
            pair = self._extract_pair_from_epic(epic)
            
            # Fetch weekly data (7 days)
            weekly_data = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='4h',  # Use 4h data for weekly analysis
                lookback_hours=7 * 24
            )
            
            if weekly_data is None or weekly_data.empty:
                return None
            
            return {
                'high': weekly_data['high'].max(),
                'low': weekly_data['low'].min(),
                'period': len(weekly_data),
                'confidence': min(len(weekly_data) / (7 * 6), 1.0)  # 7 days * 6 bars per day (4h)
            }
            
        except Exception as e:
            self.logger.debug(f"Weekly range data fetch failed: {e}")
            return None
    
    def _estimate_weekly_range_from_daily(self, df: pd.DataFrame) -> Dict:
        """Estimate weekly range from available daily data"""
        try:
            # Use available data up to 7 days worth
            weekly_period = min(7 * 24 * 4, len(df))  # 7 days in 15m bars
            weekly_data = df.tail(weekly_period)
            
            return {
                'high': weekly_data['high'].max(),
                'low': weekly_data['low'].min(),
                'period': len(weekly_data),
                'confidence': len(weekly_data) / (7 * 24 * 4),  # Partial confidence
                'estimated': True
            }
            
        except Exception:
            return None
    
    def _get_current_session_range(self, df: pd.DataFrame) -> Dict:
        """Get current trading session range"""
        try:
            # Identify current session based on time
            current_time = datetime.utcnow()
            session_info = self._identify_trading_session(current_time)
            
            # Get session data (last 8 hours max)
            session_period = min(8 * 4, len(df))  # 8 hours in 15m bars
            session_data = df.tail(session_period)
            
            return {
                'high': session_data['high'].max(),
                'low': session_data['low'].min(),
                'period': len(session_data),
                'session': session_info['session'],
                'confidence': min(len(session_data) / (8 * 4), 1.0)
            }
            
        except Exception:
            return None
    
    def _get_recent_swing_range(self, df: pd.DataFrame, swing_length: int) -> Dict:
        """Get recent swing high/low range"""
        try:
            # Look for swing range over recent period
            swing_period = min(swing_length * 8, len(df))  # 8x swing length
            swing_data = df.tail(swing_period)
            
            return {
                'high': swing_data['high'].max(),
                'low': swing_data['low'].min(),
                'period': len(swing_data),
                'confidence': min(len(swing_data) / (swing_length * 8), 1.0)
            }
            
        except Exception:
            return None
    
    def _classify_premium_discount_zone(self, position_percentage: float) -> PremiumDiscountZone:
        """Classify position percentage into premium/discount zone"""
        try:
            for zone, (min_pct, max_pct) in self.zone_thresholds.items():
                if min_pct <= position_percentage <= max_pct:
                    return zone
            
            # Fallback
            if position_percentage > 0.9:
                return PremiumDiscountZone.EXTREME_PREMIUM
            elif position_percentage < 0.1:
                return PremiumDiscountZone.EXTREME_DISCOUNT
            else:
                return PremiumDiscountZone.EQUILIBRIUM
                
        except Exception:
            return PremiumDiscountZone.EQUILIBRIUM
    
    def _calculate_golden_ratio_levels(self, high: float, low: float) -> Dict:
        """Calculate golden ratio levels for the range"""
        try:
            range_size = high - low
            
            return {
                'fibonacci_382': low + (range_size * self.golden_ratios['fibonacci_382']),
                'fibonacci_618': low + (range_size * self.golden_ratios['fibonacci_618']),
                'fibonacci_705': low + (range_size * self.golden_ratios['fibonacci_705']),
                'fibonacci_79': low + (range_size * self.golden_ratios['fibonacci_79']),
                'equilibrium': low + (range_size * 0.5),
                'premium_entry': low + (range_size * 0.75),
                'discount_entry': low + (range_size * 0.25)
            }
            
        except Exception:
            return {}
    
    def _is_optimal_entry_zone(self, position_percentage: float) -> bool:
        """Check if current position is in optimal trade entry zone"""
        try:
            # OTE zones: 62%-79% (premium for sells), 21%-38% (discount for buys)
            ote_premium = 0.62 <= position_percentage <= 0.79
            ote_discount = 0.21 <= position_percentage <= 0.38
            
            return ote_premium or ote_discount
            
        except Exception:
            return False
    
    def _get_market_maker_bias(self, zone: PremiumDiscountZone, position_percentage: float) -> str:
        """Get market maker bias based on premium/discount zone"""
        try:
            if zone in [PremiumDiscountZone.EXTREME_PREMIUM, PremiumDiscountZone.PREMIUM]:
                return 'bearish'  # Sell at premium
            elif zone in [PremiumDiscountZone.EXTREME_DISCOUNT, PremiumDiscountZone.DISCOUNT]:
                return 'bullish'  # Buy at discount
            elif zone == PremiumDiscountZone.EQUILIBRIUM_HIGH and position_percentage > 0.65:
                return 'bearish'  # Lean bearish in high equilibrium
            elif zone == PremiumDiscountZone.EQUILIBRIUM_LOW and position_percentage < 0.35:
                return 'bullish'  # Lean bullish in low equilibrium
            else:
                return 'neutral'  # Neutral in true equilibrium
                
        except Exception:
            return 'neutral'
    
    def _calculate_range_confidence(self, range_data: Dict, range_type: str) -> float:
        """Calculate confidence level for range analysis"""
        try:
            base_confidence = range_data.get('confidence', 0.5)
            
            # Adjust confidence based on range type and data quality
            if range_type == 'weekly':
                # Weekly ranges are more significant
                confidence_multiplier = 1.2
            elif range_type == 'daily':
                # Daily ranges are standard
                confidence_multiplier = 1.0
            elif range_type == 'session':
                # Session ranges are less significant
                confidence_multiplier = 0.8
            else:  # swing
                # Swing ranges are least significant
                confidence_multiplier = 0.6
            
            # Consider data completeness
            period = range_data.get('period', 0)
            expected_periods = {
                'weekly': 7 * 6,    # 7 days * 6 4h bars
                'daily': 24 * 4,    # 24 hours * 4 15m bars
                'session': 8 * 4,   # 8 hours * 4 15m bars
                'swing': 40         # Typical swing period
            }
            
            expected = expected_periods.get(range_type, 40)
            completeness = min(period / expected, 1.0) if expected > 0 else 1.0
            
            final_confidence = base_confidence * confidence_multiplier * completeness
            return min(max(final_confidence, 0.1), 1.0)  # Clamp 0.1-1.0
            
        except Exception:
            return 0.5
    
    def _calculate_market_maker_bias(self, analysis: Dict, config: Dict) -> Dict:
        """Calculate overall market maker bias from all ranges"""
        try:
            bias_scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            total_weight = 0
            
            # Weight different range types
            range_weights = {
                'weekly_analysis': 0.4,    # Highest weight
                'daily_analysis': 0.3,     # High weight
                'session_analysis': 0.2,   # Medium weight
                'swing_analysis': 0.1      # Lowest weight
            }
            
            for range_key, weight in range_weights.items():
                range_analysis = analysis.get(range_key)
                
                if range_analysis:
                    bias = range_analysis.get('market_maker_bias', 'neutral')
                    confidence = range_analysis.get('confidence', 0.5)
                    
                    # Weight by confidence and range importance
                    weighted_score = weight * confidence
                    bias_scores[bias] += weighted_score
                    total_weight += weight
            
            if total_weight == 0:
                analysis['market_maker_bias'] = 'neutral'
                return analysis
            
            # Normalize scores
            for bias in bias_scores:
                bias_scores[bias] /= total_weight
            
            # Determine dominant bias
            max_bias = max(bias_scores, key=bias_scores.get)
            max_score = bias_scores[max_bias]
            
            # Require minimum confidence for non-neutral bias
            if max_score < 0.6:
                analysis['market_maker_bias'] = 'neutral'
            else:
                analysis['market_maker_bias'] = max_bias
            
            analysis['bias_scores'] = bias_scores
            analysis['bias_confidence'] = max_score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market maker bias calculation failed: {e}")
            analysis['market_maker_bias'] = 'neutral'
            return analysis
    
    def _identify_trading_session(self, current_time: datetime) -> Dict:
        """Identify current trading session"""
        try:
            utc_hour = current_time.hour
            
            if 8 <= utc_hour <= 16:
                if 13 <= utc_hour <= 16:
                    return {'session': 'London/NY Overlap', 'priority': 'high'}
                else:
                    return {'session': 'London', 'priority': 'medium'}
            elif 13 <= utc_hour <= 22:
                return {'session': 'New York', 'priority': 'medium'}
            elif 22 <= utc_hour or utc_hour <= 8:
                return {'session': 'Asian', 'priority': 'low'}
            else:
                return {'session': 'Off Hours', 'priority': 'very_low'}
                
        except Exception:
            return {'session': 'Unknown', 'priority': 'low'}
    
    def _identify_optimal_entry_zones(self, analysis: Dict, config: Dict) -> Dict:
        """Identify optimal trade entry zones across all ranges"""
        try:
            optimal_zones = []
            
            for range_key in ['daily_analysis', 'weekly_analysis', 'session_analysis', 'swing_analysis']:
                range_analysis = analysis.get(range_key)
                
                if range_analysis and range_analysis.get('optimal_entry_zone', False):
                    zone_info = {
                        'range_type': range_analysis.get('range_type'),
                        'zone': range_analysis.get('zone'),
                        'position_percentage': range_analysis.get('position_percentage'),
                        'confidence': range_analysis.get('confidence'),
                        'market_maker_bias': range_analysis.get('market_maker_bias'),
                        'golden_ratio_levels': range_analysis.get('golden_ratio_levels', {})
                    }
                    optimal_zones.append(zone_info)
            
            analysis['optimal_entry_zones'] = optimal_zones
            analysis['in_optimal_zone'] = len(optimal_zones) > 0
            analysis['optimal_zone_count'] = len(optimal_zones)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Optimal entry zone identification failed: {e}")
            analysis['optimal_entry_zones'] = []
            return analysis
    
    def _calculate_pd_confluence_score(self, analysis: Dict, config: Dict) -> Dict:
        """Calculate premium/discount confluence score"""
        try:
            confluence_factors = []
            
            # Factor 1: Market maker bias alignment
            bias = analysis.get('market_maker_bias', 'neutral')
            bias_confidence = analysis.get('bias_confidence', 0)
            
            if bias != 'neutral' and bias_confidence > 0.6:
                confluence_factors.append({
                    'factor': f'market_maker_bias_{bias}',
                    'weight': bias_confidence * 0.4,
                    'description': f'Market maker bias: {bias}'
                })
            
            # Factor 2: Multiple optimal entry zones
            optimal_zones = len(analysis.get('optimal_entry_zones', []))
            if optimal_zones > 0:
                zone_score = min(optimal_zones / 3.0, 1.0) * 0.3
                confluence_factors.append({
                    'factor': f'optimal_entry_zones_{optimal_zones}',
                    'weight': zone_score,
                    'description': f'{optimal_zones} optimal entry zones identified'
                })
            
            # Factor 3: Weekly/Daily alignment
            weekly_analysis = analysis.get('weekly_analysis')
            daily_analysis = analysis.get('daily_analysis')
            
            if weekly_analysis and daily_analysis:
                weekly_bias = weekly_analysis.get('market_maker_bias', 'neutral')
                daily_bias = daily_analysis.get('market_maker_bias', 'neutral')
                
                if weekly_bias != 'neutral' and weekly_bias == daily_bias:
                    alignment_score = 0.25
                    confluence_factors.append({
                        'factor': f'weekly_daily_alignment_{weekly_bias}',
                        'weight': alignment_score,
                        'description': 'Weekly/Daily premium-discount alignment'
                    })
            
            # Factor 4: Golden ratio confluence
            golden_ratio_confluence = self._check_golden_ratio_confluence(analysis)
            if golden_ratio_confluence > 0:
                confluence_factors.append({
                    'factor': 'golden_ratio_confluence',
                    'weight': golden_ratio_confluence * 0.2,
                    'description': 'Golden ratio level confluence'
                })
            
            # Calculate total confluence score
            total_score = sum(factor['weight'] for factor in confluence_factors)
            
            analysis['pd_confluence_factors'] = confluence_factors
            analysis['confluence_score'] = min(total_score, 1.0)
            analysis['confluence_grade'] = self._get_confluence_grade(total_score)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"PD confluence score calculation failed: {e}")
            analysis['confluence_score'] = 0.0
            return analysis
    
    def _check_golden_ratio_confluence(self, analysis: Dict) -> float:
        """Check for golden ratio level confluence across ranges"""
        try:
            current_price = analysis.get('current_price', 0)
            if not current_price:
                return 0.0
            
            confluence_count = 0
            total_ranges = 0
            
            for range_key in ['daily_analysis', 'weekly_analysis', 'session_analysis']:
                range_analysis = analysis.get(range_key)
                
                if range_analysis:
                    total_ranges += 1
                    golden_levels = range_analysis.get('golden_ratio_levels', {})
                    
                    # Check if current price is near any golden ratio level
                    tolerance = current_price * 0.001  # 0.1% tolerance
                    
                    for level_name, level_price in golden_levels.items():
                        if abs(current_price - level_price) <= tolerance:
                            confluence_count += 1
                            break
            
            if total_ranges == 0:
                return 0.0
            
            return confluence_count / total_ranges
            
        except Exception:
            return 0.0
    
    def _get_confluence_grade(self, score: float) -> str:
        """Get confluence grade based on score"""
        if score >= 0.8:
            return 'A+'
        elif score >= 0.7:
            return 'A'
        elif score >= 0.6:
            return 'B+'
        elif score >= 0.5:
            return 'B'
        elif score >= 0.4:
            return 'C+'
        elif score >= 0.3:
            return 'C'
        else:
            return 'D'
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic"""
        try:
            if '.D.' in epic and '.MINI.IP' in epic:
                parts = epic.split('.D.')
                if len(parts) > 1:
                    pair_part = parts[1].split('.MINI.IP')[0]
                    return pair_part
            return 'EURUSD'
        except Exception:
            return 'EURUSD'
    
    def _get_empty_analysis(self) -> Dict:
        """Get empty analysis result"""
        return {
            'current_price': 0.0,
            'daily_analysis': None,
            'weekly_analysis': None,
            'session_analysis': None,
            'swing_analysis': None,
            'optimal_entry_zones': [],
            'market_maker_bias': 'neutral',
            'confluence_score': 0.0,
            'error': 'Analysis failed'
        }
    
    def get_premium_discount_signal(
        self, 
        analysis: Dict, 
        signal_direction: str,
        config: Dict
    ) -> Optional[Dict]:
        """Get premium/discount based trading signal validation"""
        try:
            if not analysis or analysis.get('error'):
                return None
            
            bias = analysis.get('market_maker_bias', 'neutral')
            confluence_score = analysis.get('confluence_score', 0.0)
            optimal_zones = analysis.get('optimal_entry_zones', [])
            
            # Check if signal direction aligns with market maker bias
            direction_alignment = False
            if signal_direction == 'bullish' and bias == 'bullish':
                direction_alignment = True
            elif signal_direction == 'bearish' and bias == 'bearish':
                direction_alignment = True
            
            # Require minimum confluence for premium/discount signals
            min_pd_confluence = config.get('min_pd_confluence', 0.4)
            
            if not direction_alignment or confluence_score < min_pd_confluence:
                return None
            
            # Create premium/discount signal
            pd_signal = {
                'pd_validation': True,
                'market_maker_bias': bias,
                'confluence_score': confluence_score,
                'confluence_grade': analysis.get('confluence_grade', 'D'),
                'optimal_entry_zones': len(optimal_zones),
                'premium_discount_factors': analysis.get('pd_confluence_factors', []),
                'bias_confidence': analysis.get('bias_confidence', 0.0)
            }
            
            # Add range-specific data
            for range_type in ['daily', 'weekly', 'session', 'swing']:
                range_analysis = analysis.get(f'{range_type}_analysis')
                if range_analysis:
                    pd_signal[f'{range_type}_zone'] = range_analysis.get('zone')
                    pd_signal[f'{range_type}_position'] = range_analysis.get('position_percentage')
            
            return pd_signal
            
        except Exception as e:
            self.logger.error(f"Premium/discount signal generation failed: {e}")
            return None