# core/trading/market_monitor.py
"""
Market Monitor - Extracted from TradingOrchestrator
Monitors market conditions and trading environment
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
try:
    import config
except ImportError:
    from forex_scanner import config


class MarketMonitor:
    """
    Monitors market conditions and trading environment
    Extracted from TradingOrchestrator to provide focused market monitoring
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 timezone_manager=None):
        
        self.logger = logger or logging.getLogger(__name__)
        self.timezone_manager = timezone_manager
        
        # Market session definitions
        self.market_sessions = {
            'SYDNEY': {'open': 21, 'close': 6},    # UTC hours
            'TOKYO': {'open': 0, 'close': 9},
            'LONDON': {'open': 8, 'close': 17},
            'NEW_YORK': {'open': 13, 'close': 22}
        }
        
        # Market status tracking
        self.market_status = {
            'is_open': False,
            'active_sessions': [],
            'next_open': None,
            'next_close': None,
            'volatility_level': 'NORMAL',
            'liquidity_level': 'NORMAL'
        }
        
        # Volatility thresholds
        self.volatility_thresholds = {
            'LOW': getattr(config, 'LOW_VOLATILITY_THRESHOLD', 0.5),
            'NORMAL': getattr(config, 'NORMAL_VOLATILITY_THRESHOLD', 1.0),
            'HIGH': getattr(config, 'HIGH_VOLATILITY_THRESHOLD', 2.0),
            'EXTREME': getattr(config, 'EXTREME_VOLATILITY_THRESHOLD', 3.0)
        }
        
        # Spread monitoring
        self.spread_thresholds = {
            'TIGHT': getattr(config, 'TIGHT_SPREAD_THRESHOLD', 2.0),
            'NORMAL': getattr(config, 'NORMAL_SPREAD_THRESHOLD', 3.0),
            'WIDE': getattr(config, 'WIDE_SPREAD_THRESHOLD', 5.0)
        }
        
        # Economic calendar events (placeholder)
        self.economic_events = []
        
        # Market condition cache
        self.condition_cache = {}
        self.cache_timestamp = None
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
        self.logger.info("🌍 MarketMonitor initialized")
        self.logger.info(f"   Monitoring sessions: {list(self.market_sessions.keys())}")
    
    def check_market_hours(self) -> Tuple[bool, str]:
        """
        Check if markets are currently open
        
        Returns:
            Tuple of (is_open, status_message)
        """
        try:
            current_time = datetime.utcnow()
            current_hour = current_time.hour
            
            active_sessions = []
            
            # Check each major session
            for session_name, times in self.market_sessions.items():
                open_hour = times['open']
                close_hour = times['close']
                
                # Handle sessions that cross midnight
                if open_hour > close_hour:
                    is_open = current_hour >= open_hour or current_hour < close_hour
                else:
                    is_open = open_hour <= current_hour < close_hour
                
                if is_open:
                    active_sessions.append(session_name)
            
            # Update market status
            self.market_status['active_sessions'] = active_sessions
            self.market_status['is_open'] = len(active_sessions) > 0
            
            if active_sessions:
                session_list = ", ".join(active_sessions)
                return True, f"Markets open: {session_list}"
            else:
                # Find next session opening
                next_open = self._find_next_session_open(current_hour)
                return False, f"Markets closed. Next open: {next_open}"
                
        except Exception as e:
            self.logger.error(f"❌ Error checking market hours: {e}")
            return True, "Market hours check failed, assuming open"  # Fail-safe
    
    def _find_next_session_open(self, current_hour: int) -> str:
        """Find the next market session opening"""
        try:
            next_opens = []
            
            for session_name, times in self.market_sessions.items():
                open_hour = times['open']
                
                if open_hour > current_hour:
                    hours_until = open_hour - current_hour
                else:
                    hours_until = (24 - current_hour) + open_hour
                
                next_opens.append((hours_until, session_name, open_hour))
            
            # Find the earliest opening
            next_opens.sort()
            hours_until, session_name, open_hour = next_opens[0]
            
            return f"{session_name} in {hours_until}h ({open_hour:02d}:00 UTC)"
            
        except Exception as e:
            self.logger.error(f"❌ Error finding next session: {e}")
            return "Unknown"
    
    def assess_market_volatility(self, epic: str = None, price_data: List[Dict] = None) -> Dict:
        """
        Assess current market volatility
        
        Args:
            epic: Optional specific currency pair
            price_data: Optional historical price data
            
        Returns:
            Volatility assessment dict
        """
        try:
            # Use cache if available and fresh
            if self._is_cache_valid():
                cache_key = f"volatility_{epic or 'general'}"
                if cache_key in self.condition_cache:
                    return self.condition_cache[cache_key]
            
            volatility_assessment = {
                'level': 'NORMAL',
                'score': 1.0,
                'factors': [],
                'recommendation': 'Normal trading conditions',
                'epic': epic,
                'timestamp': datetime.utcnow()
            }
            
            # Check time-based volatility factors
            current_hour = datetime.utcnow().hour
            
            # Higher volatility during session overlaps
            if self._is_session_overlap(current_hour):
                volatility_assessment['score'] *= 1.3
                volatility_assessment['factors'].append('Session overlap')
            
            # Lower volatility during lunch hours
            if 11 <= current_hour <= 13:  # London lunch
                volatility_assessment['score'] *= 0.8
                volatility_assessment['factors'].append('London lunch hours')
            
            # Higher volatility at session opens/closes
            if self._is_session_boundary(current_hour):
                volatility_assessment['score'] *= 1.2
                volatility_assessment['factors'].append('Session boundary')
            
            # Check for economic events
            if self._has_major_news_events():
                volatility_assessment['score'] *= 1.5
                volatility_assessment['factors'].append('Major news events')
            
            # Determine volatility level
            if volatility_assessment['score'] >= self.volatility_thresholds['EXTREME']:
                volatility_assessment['level'] = 'EXTREME'
                volatility_assessment['recommendation'] = 'Reduce position sizes, increase stops'
            elif volatility_assessment['score'] >= self.volatility_thresholds['HIGH']:
                volatility_assessment['level'] = 'HIGH'
                volatility_assessment['recommendation'] = 'Use tighter stops, reduce risk'
            elif volatility_assessment['score'] <= self.volatility_thresholds['LOW']:
                volatility_assessment['level'] = 'LOW'
                volatility_assessment['recommendation'] = 'May use wider stops, normal risk'
            else:
                volatility_assessment['level'] = 'NORMAL'
                volatility_assessment['recommendation'] = 'Normal trading conditions'
            
            # Cache the result
            cache_key = f"volatility_{epic or 'general'}"
            self.condition_cache[cache_key] = volatility_assessment
            self.cache_timestamp = datetime.utcnow()
            
            return volatility_assessment
            
        except Exception as e:
            self.logger.error(f"❌ Error assessing volatility: {e}")
            return {
                'level': 'UNKNOWN',
                'score': 1.0,
                'factors': ['Assessment error'],
                'recommendation': 'Use default risk parameters',
                'epic': epic,
                'timestamp': datetime.utcnow()
            }
    
    def monitor_spread_conditions(self, epic: str, current_spread: float = None) -> Dict:
        """
        Monitor spread conditions for trading
        
        Args:
            epic: Currency pair to monitor
            current_spread: Current spread in pips
            
        Returns:
            Spread condition assessment
        """
        try:
            spread_assessment = {
                'epic': epic,
                'current_spread': current_spread,
                'condition': 'UNKNOWN',
                'is_tradable': True,
                'recommendation': 'Monitor spreads',
                'timestamp': datetime.utcnow()
            }
            
            if current_spread is not None:
                # Determine spread condition
                if current_spread <= self.spread_thresholds['TIGHT']:
                    spread_assessment['condition'] = 'TIGHT'
                    spread_assessment['recommendation'] = 'Excellent trading conditions'
                elif current_spread <= self.spread_thresholds['NORMAL']:
                    spread_assessment['condition'] = 'NORMAL'
                    spread_assessment['recommendation'] = 'Good trading conditions'
                elif current_spread <= self.spread_thresholds['WIDE']:
                    spread_assessment['condition'] = 'WIDE'
                    spread_assessment['recommendation'] = 'Consider reduced position size'
                else:
                    spread_assessment['condition'] = 'EXTREME'
                    spread_assessment['is_tradable'] = False
                    spread_assessment['recommendation'] = 'Avoid trading - spreads too wide'
            
            # Check for time-based spread widening
            current_hour = datetime.utcnow().hour
            
            # Spreads typically wider during:
            # - Market closes/opens
            # - Low liquidity periods
            # - Major news events
            
            if not self.market_status['is_open']:
                spread_assessment['recommendation'] += ' (Markets closed - expect wider spreads)'
            
            if self._has_major_news_events():
                spread_assessment['recommendation'] += ' (News events - monitor spreads closely)'
            
            return spread_assessment
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring spreads: {e}")
            return {
                'epic': epic,
                'condition': 'UNKNOWN',
                'is_tradable': True,
                'recommendation': 'Monitor spreads carefully',
                'timestamp': datetime.utcnow()
            }
    
    def check_news_events(self, hours_ahead: int = 2) -> Dict:
        """
        Check for upcoming major news events
        
        Args:
            hours_ahead: Hours to look ahead for events
            
        Returns:
            News events assessment
        """
        try:
            current_time = datetime.utcnow()
            cutoff_time = current_time + timedelta(hours=hours_ahead)
            
            # In production, this would connect to economic calendar API
            # For now, return mock data based on common news times
            
            upcoming_events = []
            current_hour = current_time.hour
            
            # Common news release times (UTC)
            news_hours = {
                8: 'UK Economic Data',
                12: 'EU Economic Data', 
                13: 'US Market Open',
                17: 'US Economic Data',
                21: 'FOMC/Fed Announcements'
            }
            
            for hour, event_type in news_hours.items():
                if current_hour <= hour <= current_hour + hours_ahead:
                    upcoming_events.append({
                        'time': f"{hour:02d}:00 UTC",
                        'event': event_type,
                        'impact': 'MEDIUM',
                        'currencies': ['USD', 'EUR', 'GBP']
                    })
            
            # Check for major events (placeholder)
            major_events = self._check_major_scheduled_events()
            upcoming_events.extend(major_events)
            
            return {
                'upcoming_events': upcoming_events,
                'has_major_events': len([e for e in upcoming_events if e.get('impact') == 'HIGH']) > 0,
                'recommendation': self._get_news_recommendation(upcoming_events),
                'hours_ahead': hours_ahead,
                'timestamp': current_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error checking news events: {e}")
            return {
                'upcoming_events': [],
                'has_major_events': False,
                'recommendation': 'Monitor news carefully',
                'timestamp': datetime.utcnow()
            }
    
    def validate_trading_conditions(self, signal: Dict) -> Tuple[bool, str]:
        """
        Validate overall trading conditions for a signal
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            Tuple of (conditions_ok, status_message)
        """
        try:
            epic = signal.get('epic', '')
            conditions = []
            warnings = []
            
            # Check market hours
            market_open, market_msg = self.check_market_hours()
            if not market_open:
                return False, f"Market closed: {market_msg}"
            else:
                conditions.append("Market open")
            
            # Check volatility
            volatility = self.assess_market_volatility(epic)
            if volatility['level'] == 'EXTREME':
                warnings.append(f"Extreme volatility: {volatility['recommendation']}")
            else:
                conditions.append(f"Volatility {volatility['level'].lower()}")
            
            # Check news events
            news = self.check_news_events()
            if news['has_major_events']:
                warnings.append("Major news events approaching")
            
            # Check spread conditions (if available)
            current_spread = signal.get('spread')
            if current_spread:
                spread_info = self.monitor_spread_conditions(epic, current_spread)
                if not spread_info['is_tradable']:
                    return False, f"Spreads too wide: {spread_info['recommendation']}"
                else:
                    conditions.append(f"Spreads {spread_info['condition'].lower()}")
            
            # Compile status message
            status_parts = []
            if conditions:
                status_parts.append(f"Conditions: {', '.join(conditions)}")
            if warnings:
                status_parts.append(f"Warnings: {', '.join(warnings)}")
            
            status_message = "; ".join(status_parts) if status_parts else "Trading conditions validated"
            
            return True, status_message
            
        except Exception as e:
            self.logger.error(f"❌ Error validating trading conditions: {e}")
            return True, "Condition validation failed, allowing trade"  # Fail-safe
    
    def get_market_status(self) -> Dict:
        """
        Get comprehensive market status
        
        Returns:
            Dict containing market status information
        """
        try:
            # Update market hours
            market_open, market_msg = self.check_market_hours()
            
            # Get volatility assessment
            volatility = self.assess_market_volatility()
            
            # Check news events
            news = self.check_news_events()
            
            return {
                'market_hours': {
                    'is_open': market_open,
                    'status': market_msg,
                    'active_sessions': self.market_status['active_sessions']
                },
                'volatility': {
                    'level': volatility['level'],
                    'score': volatility['score'],
                    'recommendation': volatility['recommendation']
                },
                'news_events': {
                    'upcoming_count': len(news['upcoming_events']),
                    'has_major_events': news['has_major_events'],
                    'next_events': news['upcoming_events'][:3]  # Next 3 events
                },
                'trading_recommendation': self._get_overall_trading_recommendation(
                    market_open, volatility, news
                ),
                'timestamp': datetime.utcnow(),
                'market_sessions': self.market_sessions
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow()
            }
    
    def _is_session_overlap(self, hour: int) -> bool:
        """Check if current hour has overlapping trading sessions"""
        # Major overlaps:
        # London-New York: 13:00-17:00 UTC
        # Tokyo-London: 8:00-9:00 UTC
        # Sydney-Tokyo: 0:00-6:00 UTC
        
        overlaps = [
            (13, 17),  # London-New York
            (8, 9),    # Tokyo-London
            (0, 6)     # Sydney-Tokyo
        ]
        
        for start, end in overlaps:
            if start <= hour < end:
                return True
        return False
    
    def _is_session_boundary(self, hour: int) -> bool:
        """Check if current hour is near session open/close"""
        boundary_hours = set()
        
        for session in self.market_sessions.values():
            boundary_hours.add(session['open'])
            boundary_hours.add(session['close'])
            # Add hour before and after for boundary effect
            boundary_hours.add((session['open'] - 1) % 24)
            boundary_hours.add((session['open'] + 1) % 24)
            boundary_hours.add((session['close'] - 1) % 24)
            boundary_hours.add((session['close'] + 1) % 24)
        
        return hour in boundary_hours
    
    def _has_major_news_events(self) -> bool:
        """Check if there are major news events in the next 2 hours"""
        # Placeholder implementation
        # In production, this would check real economic calendar
        
        current_hour = datetime.utcnow().hour
        
        # High-impact news hours (UTC)
        high_impact_hours = [8, 12, 13, 17, 21]  # Major data releases
        
        return current_hour in high_impact_hours
    
    def _check_major_scheduled_events(self) -> List[Dict]:
        """Check for major scheduled events (FOMC, NFP, etc.)"""
        # Placeholder for major scheduled events
        # In production, this would connect to economic calendar API
        
        return []
    
    def _get_news_recommendation(self, events: List[Dict]) -> str:
        """Get trading recommendation based on news events"""
        if not events:
            return "No major events scheduled"
        
        high_impact_count = len([e for e in events if e.get('impact') == 'HIGH'])
        
        if high_impact_count > 0:
            return "Avoid trading or use reduced position sizes"
        elif len(events) > 2:
            return "Monitor news closely, consider tighter stops"
        else:
            return "Normal trading, be aware of scheduled events"
    
    def _get_overall_trading_recommendation(self, market_open: bool, 
                                          volatility: Dict, news: Dict) -> str:
        """Get overall trading recommendation"""
        if not market_open:
            return "AVOID - Markets closed"
        
        if volatility['level'] == 'EXTREME':
            return "CAUTION - Extreme volatility"
        
        if news['has_major_events']:
            return "CAUTION - Major news events"
        
        if volatility['level'] == 'HIGH':
            return "MODERATE - High volatility"
        
        if len(self.market_status['active_sessions']) >= 2:
            return "FAVORABLE - Multiple sessions active"
        
        return "NORMAL - Standard trading conditions"
    
    def _is_cache_valid(self) -> bool:
        """Check if condition cache is still valid"""
        if not self.cache_timestamp:
            return False
        
        return datetime.utcnow() - self.cache_timestamp < self.cache_duration
    
    def clear_cache(self):
        """Clear market condition cache"""
        self.condition_cache.clear()
        self.cache_timestamp = None
        self.logger.info("🔄 Market condition cache cleared")
    
    def update_economic_events(self, events: List[Dict]):
        """Update economic calendar events"""
        self.economic_events = events
        self.logger.info(f"📅 Updated economic events: {len(events)} events")
    
    def set_volatility_thresholds(self, **thresholds):
        """Update volatility thresholds"""
        for level, threshold in thresholds.items():
            if level.upper() in self.volatility_thresholds:
                self.volatility_thresholds[level.upper()] = float(threshold)
        
        self.logger.info(f"🔧 Volatility thresholds updated: {thresholds}")
    
    def set_spread_thresholds(self, **thresholds):
        """Update spread thresholds"""
        for level, threshold in thresholds.items():
            if level.upper() in self.spread_thresholds:
                self.spread_thresholds[level.upper()] = float(threshold)
        
        self.logger.info(f"🔧 Spread thresholds updated: {thresholds}")