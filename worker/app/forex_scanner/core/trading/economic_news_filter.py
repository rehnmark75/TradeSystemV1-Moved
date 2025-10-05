# core/trading/economic_news_filter.py
"""
Economic News Filter for Trade Validator
Integrates with economic-calendar service to filter signals based on upcoming economic events

FEATURES:
- Real-time news risk assessment for currency pairs
- Pre-trade news filtering to avoid high-risk periods
- Configurable impact levels and time windows
- Graceful degradation when news service unavailable
- Caching for performance optimization
"""

import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

try:
    import config
except ImportError:
    from forex_scanner import config


class EconomicNewsFilter:
    """
    Economic news filter for trade validation
    Integrates with economic-calendar service to assess news risk before trading
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Configuration
        self.news_service_url = getattr(config, 'ECONOMIC_CALENDAR_URL', 'http://economic-calendar:8091')
        self.enable_news_filtering = getattr(config, 'ENABLE_NEWS_FILTERING', True)

        # Risk assessment parameters
        self.high_impact_buffer_minutes = getattr(config, 'NEWS_HIGH_IMPACT_BUFFER_MINUTES', 30)
        self.medium_impact_buffer_minutes = getattr(config, 'NEWS_MEDIUM_IMPACT_BUFFER_MINUTES', 15)
        self.max_lookahead_hours = getattr(config, 'NEWS_LOOKAHEAD_HOURS', 4)

        # Risk tolerance levels
        self.block_before_high_impact = getattr(config, 'BLOCK_TRADES_BEFORE_HIGH_IMPACT_NEWS', True)
        self.block_before_medium_impact = getattr(config, 'BLOCK_TRADES_BEFORE_MEDIUM_IMPACT_NEWS', False)
        self.reduce_confidence_near_news = getattr(config, 'REDUCE_CONFIDENCE_NEAR_NEWS', True)

        # Major economic events that always trigger high risk
        self.critical_events = getattr(config, 'CRITICAL_ECONOMIC_EVENTS', [
            'Non-Farm Employment Change', 'NFP', 'FOMC', 'Federal Funds Rate', 'ECB Press Conference',
            'Interest Rate Decision', 'CPI', 'Core CPI', 'GDP', 'Employment', 'Unemployment'
        ])

        # Cache for performance
        self.news_cache = {}
        self.cache_expiry = {}
        self.cache_duration_minutes = getattr(config, 'NEWS_CACHE_DURATION_MINUTES', 5)

        # Connection settings
        self.request_timeout = getattr(config, 'NEWS_SERVICE_TIMEOUT_SECONDS', 5)
        self.fail_safe_on_error = getattr(config, 'NEWS_FILTER_FAIL_SAFE', True)

        # Statistics
        self.stats = {
            'total_checks': 0,
            'signals_blocked': 0,
            'high_impact_blocks': 0,
            'medium_impact_blocks': 0,
            'confidence_reductions': 0,
            'service_errors': 0,
            'cache_hits': 0
        }

        self.logger.info("📰 Economic News Filter initialized")
        self.logger.info(f"   Service URL: {self.news_service_url}")
        self.logger.info(f"   High impact buffer: {self.high_impact_buffer_minutes} minutes")
        self.logger.info(f"   Medium impact buffer: {self.medium_impact_buffer_minutes} minutes")
        self.logger.info(f"   Lookahead window: {self.max_lookahead_hours} hours")
        self.logger.info(f"   Block high impact: {self.block_before_high_impact}")
        self.logger.info(f"   Block medium impact: {self.block_before_medium_impact}")
        self.logger.info(f"   Reduce confidence: {self.reduce_confidence_near_news}")

    def validate_signal_against_news(self, signal: Dict) -> Tuple[bool, str, Dict]:
        """
        Main validation method - checks signal against upcoming economic news

        Args:
            signal: Trading signal to validate

        Returns:
            Tuple of (is_valid, reason, news_context)
        """
        self.stats['total_checks'] += 1

        if not self.enable_news_filtering:
            return True, "News filtering disabled", {}

        try:
            # Extract currency pair from signal
            currency_pair = self._extract_currency_pair(signal)
            if not currency_pair:
                return True, "Cannot determine currency pair", {}

            # Get upcoming news events
            news_events = self._get_upcoming_news(currency_pair)
            if not news_events:
                return True, "No upcoming news events", {}

            # Assess news risk
            risk_assessment = self._assess_news_risk(news_events, signal)

            # Make filtering decision
            is_valid, reason = self._make_filtering_decision(risk_assessment, signal)

            # Log the decision
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')

            if not is_valid:
                self.stats['signals_blocked'] += 1
                if risk_assessment['highest_impact'] == 'high':
                    self.stats['high_impact_blocks'] += 1
                elif risk_assessment['highest_impact'] == 'medium':
                    self.stats['medium_impact_blocks'] += 1

                self.logger.info(f"📰 NEWS BLOCK: {epic} {signal_type} - {reason}")
            else:
                self.logger.debug(f"📰 News check passed: {epic} {signal_type} - {reason}")

            return is_valid, reason, risk_assessment

        except Exception as e:
            self.stats['service_errors'] += 1
            self.logger.error(f"❌ Economic news filter error: {e}")

            if self.fail_safe_on_error:
                return True, f"News filter error (allowing): {str(e)}", {}
            else:
                return False, f"News filter error (blocking): {str(e)}", {}

    def _extract_currency_pair(self, signal: Dict) -> Optional[str]:
        """Extract currency pair from signal epic"""
        try:
            epic = signal.get('epic', '')
            if not epic:
                return None

            # Handle IG Markets format: CS.D.EURUSD.CEEM.IP -> EURUSD
            if 'CS.D.' in epic:
                parts = epic.split('.')
                if len(parts) >= 3:
                    return parts[2]  # EURUSD

            # Handle other formats
            # Remove common prefixes/suffixes
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.', '')

            # Validate currency pair format (6 characters for major pairs)
            if len(pair) == 6 and pair.isalpha():
                return pair.upper()

            return None

        except Exception as e:
            self.logger.error(f"Error extracting currency pair from {epic}: {e}")
            return None

    def _get_upcoming_news(self, currency_pair: str) -> List[Dict]:
        """Get upcoming news events for currency pair"""
        try:
            # Check cache first
            cache_key = f"news_{currency_pair}"
            current_time = datetime.now()

            if (cache_key in self.news_cache and
                cache_key in self.cache_expiry and
                current_time < self.cache_expiry[cache_key]):

                self.stats['cache_hits'] += 1
                return self.news_cache[cache_key]

            # Extract base and quote currencies
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:]

            # Make API request
            url = f"{self.news_service_url}/api/v1/events/upcoming"
            params = {
                'currencies': f"{base_currency},{quote_currency}",
                'hours': self.max_lookahead_hours,
                'impact_level': 'medium'  # Get medium and high impact events
            }

            response = requests.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()

            data = response.json()
            events = data.get('upcoming_events', [])

            # Cache the results
            self.news_cache[cache_key] = events
            self.cache_expiry[cache_key] = current_time + timedelta(minutes=self.cache_duration_minutes)

            # Clean old cache entries
            self._cleanup_cache(current_time)

            self.logger.debug(f"📰 Retrieved {len(events)} upcoming news events for {currency_pair}")
            return events

        except requests.RequestException as e:
            self.logger.warning(f"⚠️ Failed to fetch news events: {e}")
            return []
        except Exception as e:
            self.logger.error(f"❌ Error getting news events: {e}")
            return []

    def _assess_news_risk(self, news_events: List[Dict], signal: Dict) -> Dict:
        """Assess risk level based on upcoming news events"""
        current_time = datetime.now()
        risk_assessment = {
            'events_count': len(news_events),
            'high_impact_events': [],
            'medium_impact_events': [],
            'critical_events': [],
            'nearest_event': None,
            'highest_impact': 'low',
            'risk_score': 0.0,
            'time_to_nearest_high_impact': None,
            'currencies_affected': set()
        }

        try:
            for event in news_events:
                event_date_str = event.get('event_date', '')
                event_name = event.get('event_name', '')
                impact_level = event.get('impact_level', 'low')
                currency = event.get('currency', '')

                # Parse event time
                try:
                    event_time = datetime.fromisoformat(event_date_str.replace('Z', '+00:00'))
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=current_time.tzinfo)
                except:
                    continue

                # Calculate time until event
                time_until_event = (event_time - current_time).total_seconds() / 60  # minutes

                if time_until_event < 0:
                    continue  # Skip past events

                # Categorize by impact
                if impact_level == 'high' or any(critical in event_name for critical in self.critical_events):
                    risk_assessment['high_impact_events'].append({
                        'event': event,
                        'time_until': time_until_event
                    })
                    risk_assessment['highest_impact'] = 'high'

                    if risk_assessment['time_to_nearest_high_impact'] is None:
                        risk_assessment['time_to_nearest_high_impact'] = time_until_event
                    else:
                        risk_assessment['time_to_nearest_high_impact'] = min(
                            risk_assessment['time_to_nearest_high_impact'], time_until_event
                        )

                    # Check for critical events
                    if any(critical in event_name for critical in self.critical_events):
                        risk_assessment['critical_events'].append(event)

                elif impact_level == 'medium':
                    risk_assessment['medium_impact_events'].append({
                        'event': event,
                        'time_until': time_until_event
                    })
                    if risk_assessment['highest_impact'] != 'high':
                        risk_assessment['highest_impact'] = 'medium'

                # Track currencies affected
                risk_assessment['currencies_affected'].add(currency)

                # Track nearest event overall
                if (risk_assessment['nearest_event'] is None or
                    time_until_event < risk_assessment['nearest_event']['time_until']):
                    risk_assessment['nearest_event'] = {
                        'event': event,
                        'time_until': time_until_event
                    }

            # Calculate overall risk score (0.0 - 1.0)
            risk_score = 0.0

            # High impact events contribute more to risk
            for high_event in risk_assessment['high_impact_events']:
                time_factor = max(0, (60 - high_event['time_until']) / 60)  # Risk increases as event approaches
                risk_score += 0.4 * time_factor

            # Medium impact events contribute less
            for medium_event in risk_assessment['medium_impact_events']:
                time_factor = max(0, (30 - medium_event['time_until']) / 30)
                risk_score += 0.2 * time_factor

            # Critical events add significant risk
            if risk_assessment['critical_events']:
                risk_score += 0.3

            risk_assessment['risk_score'] = min(1.0, risk_score)

            self.logger.debug(f"📰 News risk assessment: {risk_assessment['highest_impact']} impact, "
                            f"score: {risk_assessment['risk_score']:.2f}, "
                            f"events: {risk_assessment['events_count']}")

            return risk_assessment

        except Exception as e:
            self.logger.error(f"❌ Error assessing news risk: {e}")
            return risk_assessment

    def _make_filtering_decision(self, risk_assessment: Dict, signal: Dict) -> Tuple[bool, str]:
        """Make the final filtering decision based on risk assessment"""
        try:
            # No events = no risk
            if risk_assessment['events_count'] == 0:
                return True, "No upcoming news events"

            # Check high impact events
            if risk_assessment['high_impact_events'] and self.block_before_high_impact:
                for high_event in risk_assessment['high_impact_events']:
                    if high_event['time_until'] <= self.high_impact_buffer_minutes:
                        event_name = high_event['event'].get('event_name', 'Unknown')
                        currency = high_event['event'].get('currency', 'Unknown')
                        return False, f"High impact news in {high_event['time_until']:.0f}min: {currency} {event_name}"

            # Check medium impact events
            if risk_assessment['medium_impact_events'] and self.block_before_medium_impact:
                for medium_event in risk_assessment['medium_impact_events']:
                    if medium_event['time_until'] <= self.medium_impact_buffer_minutes:
                        event_name = medium_event['event'].get('event_name', 'Unknown')
                        currency = medium_event['event'].get('currency', 'Unknown')
                        return False, f"Medium impact news in {medium_event['time_until']:.0f}min: {currency} {event_name}"

            # Check critical events (always block regardless of time)
            if risk_assessment['critical_events']:
                critical_event = risk_assessment['critical_events'][0]
                event_name = critical_event.get('event_name', 'Unknown')
                currency = critical_event.get('currency', 'Unknown')
                nearest_time = risk_assessment.get('time_to_nearest_high_impact', 0)

                if nearest_time <= 60:  # Block if critical event within 1 hour
                    return False, f"Critical economic event approaching: {currency} {event_name} in {nearest_time:.0f}min"

            # Signal passes news filter
            if risk_assessment['highest_impact'] == 'high':
                return True, f"Signal allowed (high impact news in {risk_assessment['time_to_nearest_high_impact']:.0f}min)"
            elif risk_assessment['highest_impact'] == 'medium':
                nearest = risk_assessment['nearest_event']
                return True, f"Signal allowed (medium impact news in {nearest['time_until']:.0f}min)"
            else:
                return True, "No significant news risk detected"

        except Exception as e:
            self.logger.error(f"❌ Error making filtering decision: {e}")
            if self.fail_safe_on_error:
                return True, f"Decision error (allowing): {str(e)}"
            else:
                return False, f"Decision error (blocking): {str(e)}"

    def _cleanup_cache(self, current_time: datetime):
        """Clean up expired cache entries"""
        try:
            expired_keys = [
                key for key, expiry_time in self.cache_expiry.items()
                if current_time > expiry_time
            ]

            for key in expired_keys:
                self.news_cache.pop(key, None)
                self.cache_expiry.pop(key, None)

            if expired_keys:
                self.logger.debug(f"🧹 Cleaned {len(expired_keys)} expired news cache entries")

        except Exception as e:
            self.logger.error(f"❌ Error cleaning news cache: {e}")

    def adjust_confidence_for_news(self, signal: Dict, confidence: float) -> Tuple[float, str]:
        """
        Adjust signal confidence based on news risk

        Args:
            signal: Trading signal
            confidence: Original confidence score

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        if not self.reduce_confidence_near_news:
            return confidence, "News confidence adjustment disabled"

        try:
            currency_pair = self._extract_currency_pair(signal)
            if not currency_pair:
                return confidence, "Cannot determine currency pair"

            news_events = self._get_upcoming_news(currency_pair)
            if not news_events:
                return confidence, "No upcoming news events"

            risk_assessment = self._assess_news_risk(news_events, signal)

            # Apply confidence reduction based on risk
            adjustment_factor = 1.0

            if risk_assessment['highest_impact'] == 'high':
                adjustment_factor = 0.7  # 30% reduction for high impact news
                self.stats['confidence_reductions'] += 1
            elif risk_assessment['highest_impact'] == 'medium':
                adjustment_factor = 0.85  # 15% reduction for medium impact news
                self.stats['confidence_reductions'] += 1

            # Additional reduction for critical events
            if risk_assessment['critical_events']:
                adjustment_factor *= 0.8  # Additional 20% reduction

            adjusted_confidence = confidence * adjustment_factor

            if adjustment_factor < 1.0:
                return adjusted_confidence, f"Confidence reduced {(1-adjustment_factor)*100:.0f}% due to {risk_assessment['highest_impact']} impact news"
            else:
                return confidence, "No confidence adjustment needed"

        except Exception as e:
            self.logger.error(f"❌ Error adjusting confidence for news: {e}")
            return confidence, f"Confidence adjustment error: {str(e)}"

    def get_news_context_for_signal(self, signal: Dict) -> Dict:
        """Get complete news context for a signal (for logging/analysis)"""
        try:
            currency_pair = self._extract_currency_pair(signal)
            if not currency_pair:
                return {}

            news_events = self._get_upcoming_news(currency_pair)
            if not news_events:
                return {}

            risk_assessment = self._assess_news_risk(news_events, signal)

            # Add summary information
            context = {
                'currency_pair': currency_pair,
                'events_found': len(news_events),
                'risk_level': risk_assessment['highest_impact'],
                'risk_score': risk_assessment['risk_score'],
                'upcoming_high_impact': len(risk_assessment['high_impact_events']),
                'upcoming_medium_impact': len(risk_assessment['medium_impact_events']),
                'critical_events': len(risk_assessment['critical_events']),
                'time_to_nearest_high': risk_assessment.get('time_to_nearest_high_impact'),
                'currencies_affected': list(risk_assessment['currencies_affected'])
            }

            return context

        except Exception as e:
            self.logger.error(f"❌ Error getting news context: {e}")
            return {}

    def test_service_connection(self) -> Tuple[bool, str]:
        """Test connection to economic calendar service"""
        try:
            url = f"{self.news_service_url}/health"
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()

            data = response.json()
            if data.get('status') == 'healthy':
                return True, "Economic calendar service is healthy"
            else:
                return False, f"Service unhealthy: {data.get('status', 'unknown')}"

        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_filter_statistics(self) -> Dict:
        """Get filtering statistics for monitoring"""
        total_checks = max(1, self.stats['total_checks'])

        return {
            'enabled': self.enable_news_filtering,
            'service_url': self.news_service_url,
            'configuration': {
                'high_impact_buffer_minutes': self.high_impact_buffer_minutes,
                'medium_impact_buffer_minutes': self.medium_impact_buffer_minutes,
                'lookahead_hours': self.max_lookahead_hours,
                'block_high_impact': self.block_before_high_impact,
                'block_medium_impact': self.block_before_medium_impact,
                'reduce_confidence': self.reduce_confidence_near_news,
                'fail_safe': self.fail_safe_on_error
            },
            'statistics': self.stats,
            'performance': {
                'block_rate': f"{(self.stats['signals_blocked'] / total_checks) * 100:.1f}%",
                'high_impact_block_rate': f"{(self.stats['high_impact_blocks'] / total_checks) * 100:.1f}%",
                'medium_impact_block_rate': f"{(self.stats['medium_impact_blocks'] / total_checks) * 100:.1f}%",
                'confidence_reduction_rate': f"{(self.stats['confidence_reductions'] / total_checks) * 100:.1f}%",
                'error_rate': f"{(self.stats['service_errors'] / total_checks) * 100:.1f}%",
                'cache_hit_rate': f"{(self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + len(self.news_cache))) * 100:.1f}%"
            }
        }