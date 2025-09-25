"""
Economic Calendar Service Integration
Fetches relevant economic events for trading pairs from the economic-calendar API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class EconomicCalendarService:
    def __init__(self, api_base_url: str = "http://economic-calendar:8091/api/v1"):
        self.api_base_url = api_base_url

    def extract_currencies_from_epic(self, epic: str) -> Tuple[str, str]:
        """
        Extract base and quote currencies from epic format
        Examples:
        - CS.D.EURUSD.CFD.IP -> (EUR, USD)
        - CS.D.GBPJPY.CFD.IP -> (GBP, JPY)
        - CS.D.AUDUSD.CFD.IP -> (AUD, USD)
        """
        # Common patterns for currency pair extraction
        patterns = [
            r'\.([A-Z]{3})([A-Z]{3})\.',  # Standard 6-letter pairs
            r'([A-Z]{3})([A-Z]{3})',      # Direct 6-letter pairs
            r'([A-Z]{3})/([A-Z]{3})',     # Slash separated
        ]

        for pattern in patterns:
            match = re.search(pattern, epic.upper())
            if match:
                base_currency = match.group(1)
                quote_currency = match.group(2)
                return base_currency, quote_currency

        # Fallback: try to find any 3-letter currency codes
        currencies = re.findall(r'[A-Z]{3}', epic.upper())
        if len(currencies) >= 2:
            return currencies[0], currencies[1]

        logger.warning(f"Could not extract currencies from epic: {epic}")
        return "", ""

    def get_relevant_events(
        self,
        epic: str,
        hours_ahead: int = 48,
        impact_levels: List[str] = None
    ) -> List[Dict]:
        """
        Get economic calendar events relevant to the selected trading pair
        """
        try:
            base_currency, quote_currency = self.extract_currencies_from_epic(epic)
            if not base_currency or not quote_currency:
                return []

            currencies = [base_currency, quote_currency]

            # Calculate date range
            start_date = datetime.now().date()
            end_date = (datetime.now() + timedelta(hours=hours_ahead)).date()

            # Build API request parameters - use higher limit to get more data
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'limit': 200  # Increase limit to get more events
            }

            # Make API request
            response = requests.get(f"{self.api_base_url}/events", params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            events = data.get('events', [])

            # Filter events for relevant currencies
            relevant_events = []
            for event in events:
                event_currency = event.get('currency', '').upper()
                if event_currency in currencies:
                    # Parse event datetime
                    event_datetime_str = event.get('event_date', '')
                    try:
                        event_dt = pd.to_datetime(event_datetime_str)
                        event['parsed_datetime'] = event_dt
                        event['time_until'] = self._calculate_time_until(event_dt)
                        event['is_relevant'] = True
                        event['pair_currencies'] = currencies

                        # Filter by impact level if specified
                        if impact_levels is None or event.get('impact_level', '').lower() in [level.lower() for level in impact_levels]:
                            relevant_events.append(event)

                    except Exception as e:
                        logger.warning(f"Failed to parse event datetime: {event_datetime_str}, error: {e}")
                        continue

            # Sort by datetime
            relevant_events.sort(key=lambda x: x.get('parsed_datetime', datetime.now()))

            # Check if we have upcoming events
            upcoming_events = [e for e in relevant_events if e.get('time_until', 'Past') != 'Past']

            # If no upcoming real events, add demo events for demonstration
            if len(upcoming_events) == 0:
                try:
                    from services.economic_calendar_demo import get_demo_economic_events, is_demo_mode_enabled
                    if is_demo_mode_enabled():
                        demo_events = get_demo_economic_events(epic)
                        logger.info(f"Added {len(demo_events)} demo events for {epic} (no real upcoming events)")
                        relevant_events.extend(demo_events)
                except ImportError:
                    logger.warning("Demo mode not available")

            logger.info(f"Found {len(relevant_events)} relevant events for {epic} ({base_currency}/{quote_currency})")
            return relevant_events

        except requests.RequestException as e:
            logger.error(f"Failed to fetch economic calendar data: {e}")
            # Fallback to demo data if API fails
            try:
                from services.economic_calendar_demo import get_demo_economic_events
                demo_events = get_demo_economic_events(epic)
                logger.info(f"Using demo data: {len(demo_events)} events for {epic}")
                return demo_events
            except ImportError:
                return []
        except Exception as e:
            logger.error(f"Unexpected error in get_relevant_events: {e}")
            return []

    def get_upcoming_high_impact_events(self, epic: str, hours_ahead: int = 24) -> List[Dict]:
        """Get only high-impact events coming up"""
        return self.get_relevant_events(
            epic,
            hours_ahead=hours_ahead,
            impact_levels=['high']
        )

    def get_events_this_week(self, epic: str) -> List[Dict]:
        """Get all relevant events for the current week"""
        return self.get_relevant_events(epic, hours_ahead=168)  # 7 days * 24 hours

    def _calculate_time_until(self, event_datetime: pd.Timestamp) -> str:
        """Calculate human-readable time until event"""
        try:
            from datetime import timezone

            # Convert event to UTC timestamp
            if isinstance(event_datetime, pd.Timestamp):
                if event_datetime.tz is None:
                    # Assume UTC if no timezone
                    event_dt = event_datetime.tz_localize('UTC')
                else:
                    event_dt = event_datetime.tz_convert('UTC')
            else:
                event_dt = pd.to_datetime(event_datetime).tz_localize('UTC')

            # Get current time in UTC
            now_utc = datetime.now(timezone.utc)

            time_diff = event_dt.to_pydatetime() - now_utc

            if time_diff.total_seconds() < 0:
                return "Past"
            elif time_diff.total_seconds() < 3600:  # Less than 1 hour
                minutes = int(time_diff.total_seconds() / 60)
                return f"In {minutes} min"
            elif time_diff.total_seconds() < 86400:  # Less than 1 day
                hours = int(time_diff.total_seconds() / 3600)
                minutes = int((time_diff.total_seconds() % 3600) / 60)
                return f"In {hours}h {minutes}m"
            else:  # More than 1 day
                days = time_diff.days
                hours = int(time_diff.total_seconds() % 86400 / 3600)
                return f"In {days}d {hours}h"

        except Exception as e:
            logger.warning(f"Failed to calculate time until event: {e}")
            return "Unknown"

    def get_impact_color(self, impact_level: str) -> str:
        """Get color for impact level display"""
        colors = {
            'high': '#FF4444',    # Red
            'medium': '#FFA500',  # Orange
            'low': '#4CAF50'      # Green
        }
        return colors.get(impact_level.lower(), '#666666')

    def get_impact_icon(self, impact_level: str) -> str:
        """Get emoji icon for impact level"""
        icons = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }
        return icons.get(impact_level.lower(), '⚪')

    def format_event_for_display(self, event: Dict) -> Dict:
        """Format event data for UI display"""
        return {
            'name': event.get('event_name', 'Unknown Event'),
            'currency': event.get('currency', ''),
            'time_until': event.get('time_until', ''),
            'impact_level': event.get('impact_level', 'low'),
            'impact_color': self.get_impact_color(event.get('impact_level', 'low')),
            'impact_icon': self.get_impact_icon(event.get('impact_level', 'low')),
            'previous_value': event.get('previous_value', ''),
            'forecast_value': event.get('forecast_value', ''),
            'actual_value': event.get('actual_value', ''),
            'event_date': event.get('event_date', ''),
            'parsed_datetime': event.get('parsed_datetime'),
            'is_today': self._is_today(event.get('parsed_datetime')),
            'is_tomorrow': self._is_tomorrow(event.get('parsed_datetime'))
        }

    def _is_today(self, event_datetime: pd.Timestamp) -> bool:
        """Check if event is today"""
        if not event_datetime:
            return False
        try:
            return event_datetime.date() == datetime.now().date()
        except:
            return False

    def _is_tomorrow(self, event_datetime: pd.Timestamp) -> bool:
        """Check if event is tomorrow"""
        if not event_datetime:
            return False
        try:
            tomorrow = datetime.now().date() + timedelta(days=1)
            return event_datetime.date() == tomorrow
        except:
            return False

# Create singleton instance
_economic_calendar_service = None

def get_economic_calendar_service() -> EconomicCalendarService:
    """Get singleton instance of economic calendar service"""
    global _economic_calendar_service
    if _economic_calendar_service is None:
        _economic_calendar_service = EconomicCalendarService()
    return _economic_calendar_service