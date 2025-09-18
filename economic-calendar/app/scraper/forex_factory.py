import requests
from bs4 import BeautifulSoup
import logging
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
from urllib.parse import urljoin
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential

from models import EconomicEvent, ScrapeLog, ImpactLevel, EventStatus, ScrapeStatus
from database.connection import db_manager
from config import config

logger = logging.getLogger(__name__)


class ForexFactoryScraper:
    """Scraper for Forex Factory economic calendar"""

    def __init__(self):
        self.session = requests.Session()
        self.base_url = config.FOREX_FACTORY_BASE_URL
        self.calendar_url = config.FOREX_FACTORY_CALENDAR_URL
        self.user_agents = config.USER_AGENTS
        self.setup_session()

    def setup_session(self):
        """Configure HTTP session with headers and settings"""
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def rotate_user_agent(self):
        """Rotate User-Agent to avoid detection"""
        self.session.headers['User-Agent'] = random.choice(self.user_agents)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def fetch_calendar_page(self, week_offset: int = 0) -> Optional[str]:
        """
        Fetch calendar page HTML

        Args:
            week_offset: Weeks from current week (0=current, 1=next week, -1=last week)
        """
        try:
            # Calculate date for the week
            today = datetime.now()
            target_date = today + timedelta(weeks=week_offset)
            date_str = target_date.strftime("%b%%d.%Y").lower()

            # Build URL with date parameter
            url = f"{self.calendar_url}?week={date_str}"

            logger.info(f"Fetching calendar data from: {url}")

            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1.0, config.REQUEST_DELAY_SECONDS))

            # Rotate user agent
            self.rotate_user_agent()

            response = self.session.get(
                url,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True
            )

            response.raise_for_status()

            if response.status_code == 200:
                logger.info(f"Successfully fetched calendar page ({len(response.content)} bytes)")
                return response.text
            else:
                logger.warning(f"Unexpected status code: {response.status_code}")
                return None

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching calendar: {e}")
            raise

    def parse_impact_level(self, impact_element) -> ImpactLevel:
        """Parse impact level from HTML element"""
        if not impact_element:
            return ImpactLevel.LOW

        # Check for CSS classes or icons that indicate impact
        classes = impact_element.get('class', [])
        title = impact_element.get('title', '').lower()

        if 'high' in classes or 'red' in classes or 'high impact' in title:
            return ImpactLevel.HIGH
        elif 'medium' in classes or 'orange' in classes or 'medium impact' in title:
            return ImpactLevel.MEDIUM
        elif 'holiday' in classes or 'holiday' in title:
            return ImpactLevel.HOLIDAY
        else:
            return ImpactLevel.LOW

    def parse_currency(self, currency_element) -> Optional[str]:
        """Parse currency from HTML element"""
        if not currency_element:
            return None

        currency = currency_element.get_text(strip=True).upper()

        # Validate currency code
        if len(currency) == 3 and currency.isalpha():
            return currency
        return None

    def parse_event_time(self, time_element, date_str: str) -> Tuple[Optional[datetime], Optional[str]]:
        """Parse event date and time"""
        if not time_element:
            return None, None

        time_text = time_element.get_text(strip=True)

        # Handle special cases
        if time_text.lower() in ['all day', 'tentative', 'tbd']:
            return None, time_text

        # Parse time (format: "9:30am", "2:00pm", etc.)
        time_match = re.match(r'(\d{1,2}):(\d{2})(am|pm)', time_text.lower())
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            period = time_match.group(3)

            # Convert to 24-hour format
            if period == 'pm' and hour != 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0

            # Combine with date
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                event_datetime = event_date.replace(hour=hour, minute=minute)

                # Convert to UTC (assuming ET timezone for Forex Factory)
                et_tz = pytz.timezone('US/Eastern')
                utc_tz = pytz.UTC

                event_datetime_et = et_tz.localize(event_datetime)
                event_datetime_utc = event_datetime_et.astimezone(utc_tz)

                return event_datetime_utc, f"{hour:02d}:{minute:02d}"
            except ValueError:
                return None, time_text

        return None, time_text

    def parse_economic_value(self, value_text: str) -> Tuple[Optional[str], Optional[float]]:
        """Parse economic value and convert to numeric if possible"""
        if not value_text or value_text.strip() == '':
            return None, None

        cleaned_value = value_text.strip()

        # Handle common patterns
        if cleaned_value.lower() in ['n/a', 'na', '', '--', 'tbd']:
            return None, None

        # Try to extract numeric value
        # Remove common suffixes and prefixes
        numeric_text = re.sub(r'[%$£€¥,]', '', cleaned_value)
        numeric_text = re.sub(r'[KMBkmb]$', '', numeric_text)  # Remove K, M, B suffixes

        try:
            # Handle percentage values
            if '%' in cleaned_value:
                numeric_value = float(numeric_text)
                return cleaned_value, numeric_value

            # Handle currency values
            if any(symbol in cleaned_value for symbol in ['$', '£', '€', '¥']):
                numeric_value = float(numeric_text)
                # Apply multipliers for K, M, B
                if cleaned_value.upper().endswith('K'):
                    numeric_value *= 1000
                elif cleaned_value.upper().endswith('M'):
                    numeric_value *= 1000000
                elif cleaned_value.upper().endswith('B'):
                    numeric_value *= 1000000000
                return cleaned_value, numeric_value

            # Try direct conversion
            numeric_value = float(numeric_text)
            return cleaned_value, numeric_value

        except (ValueError, TypeError):
            return cleaned_value, None

    def parse_calendar_html(self, html_content: str, target_date: datetime) -> List[Dict]:
        """Parse HTML content and extract economic events"""
        soup = BeautifulSoup(html_content, 'html.parser')
        events = []

        try:
            # Find the calendar table
            calendar_table = soup.find('table', class_=re.compile(r'calendar'))
            if not calendar_table:
                logger.warning("Could not find calendar table in HTML")
                return events

            # Process table rows
            current_date = None

            for row in calendar_table.find_all('tr'):
                # Check if this is a date header row
                date_cell = row.find('td', class_=re.compile(r'date'))
                if date_cell:
                    date_text = date_cell.get_text(strip=True)
                    if date_text:
                        try:
                            # Parse date (format varies)
                            current_date = self.parse_date_header(date_text, target_date)
                        except ValueError:
                            continue

                # Check if this is an event row
                time_cell = row.find('td', class_=re.compile(r'time'))
                currency_cell = row.find('td', class_=re.compile(r'currency'))
                impact_cell = row.find('td', class_=re.compile(r'impact'))
                event_cell = row.find('td', class_=re.compile(r'event'))

                if time_cell and currency_cell and event_cell:
                    event_data = self.parse_event_row(
                        row, current_date, time_cell, currency_cell,
                        impact_cell, event_cell
                    )
                    if event_data:
                        events.append(event_data)

        except Exception as e:
            logger.error(f"Error parsing calendar HTML: {e}")

        logger.info(f"Parsed {len(events)} events from calendar")
        return events

    def parse_date_header(self, date_text: str, reference_date: datetime) -> datetime:
        """Parse date header text into datetime object"""
        # Common formats: "Friday, January 15", "Fri Jan 15", etc.
        date_patterns = [
            r'(\w+),\s*(\w+)\s+(\d+)',  # "Friday, January 15"
            r'(\w+)\s+(\w+)\s+(\d+)',   # "Fri Jan 15"
            r'(\w+)\s+(\d+)',           # "Jan 15"
        ]

        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                try:
                    if len(match.groups()) == 3:
                        month_str = match.group(2)
                        day_str = match.group(3)
                    else:
                        month_str = match.group(1)
                        day_str = match.group(2)

                    # Parse month
                    month_abbrevs = {
                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                    }

                    month = month_abbrevs.get(month_str.lower()[:3])
                    if not month:
                        continue

                    day = int(day_str)
                    year = reference_date.year

                    # Adjust year if needed (for dates in next year)
                    parsed_date = datetime(year, month, day)
                    if parsed_date < reference_date - timedelta(days=7):
                        parsed_date = parsed_date.replace(year=year + 1)

                    return parsed_date

                except (ValueError, KeyError):
                    continue

        raise ValueError(f"Could not parse date: {date_text}")

    def parse_event_row(self, row, current_date: datetime, time_cell, currency_cell,
                       impact_cell, event_cell) -> Optional[Dict]:
        """Parse individual event row"""
        try:
            # Extract basic event information
            currency = self.parse_currency(currency_cell)
            if not currency or currency not in config.FOCUS_CURRENCIES:
                return None  # Skip non-focus currencies

            impact_level = self.parse_impact_level(impact_cell)
            event_name = event_cell.get_text(strip=True)

            if not event_name:
                return None

            # Parse time - ensure we always have a valid date
            if current_date:
                date_str = current_date.strftime("%Y-%m-%d")
                base_date = current_date
            else:
                # Fallback to today if no current_date
                base_date = datetime.now()
                date_str = base_date.strftime("%Y-%m-%d")

            event_datetime, time_str = self.parse_event_time(time_cell, date_str)

            # Ensure we have a valid event_date - fallback to base_date if parsing failed
            final_event_date = event_datetime or base_date

            # Extract actual, forecast, and previous values
            actual_cell = row.find('td', class_=re.compile(r'actual'))
            forecast_cell = row.find('td', class_=re.compile(r'forecast'))
            previous_cell = row.find('td', class_=re.compile(r'previous'))

            actual_value, actual_numeric = None, None
            forecast_value, forecast_numeric = None, None
            previous_value, previous_numeric = None, None

            if actual_cell:
                actual_value, actual_numeric = self.parse_economic_value(actual_cell.get_text(strip=True))
            if forecast_cell:
                forecast_value, forecast_numeric = self.parse_economic_value(forecast_cell.get_text(strip=True))
            if previous_cell:
                previous_value, previous_numeric = self.parse_economic_value(previous_cell.get_text(strip=True))

            return {
                'event_name': event_name,
                'currency': currency,
                'event_date': final_event_date,
                'event_time': time_str,
                'impact_level': impact_level,
                'actual_value': actual_value,
                'actual_numeric': actual_numeric,
                'forecast_value': forecast_value,
                'forecast_numeric': forecast_numeric,
                'previous_value': previous_value,
                'previous_numeric': previous_numeric,
                'source': 'forex_factory',
                'scraped_at': datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Error parsing event row: {e}")
            return None

    def scrape_week(self, week_offset: int = 0) -> Tuple[List[Dict], ScrapeLog]:
        """
        Scrape economic calendar for a specific week

        Args:
            week_offset: Weeks from current week (0=current, 1=next week)

        Returns:
            Tuple of (events_list, scrape_log)
        """
        start_time = datetime.utcnow()
        scrape_log = ScrapeLog(
            scrape_date=start_time,
            data_source='forex_factory',
            scrape_type='weekly',
            start_time=start_time,
            status=ScrapeStatus.IN_PROGRESS
        )

        events = []

        try:
            # Calculate target date
            target_date = datetime.now() + timedelta(weeks=week_offset)

            # Fetch calendar page
            html_content = self.fetch_calendar_page(week_offset)
            if not html_content:
                raise Exception("Failed to fetch calendar page")

            # Parse events
            events = self.parse_calendar_html(html_content, target_date)

            # Update scrape log
            scrape_log.end_time = datetime.utcnow()
            scrape_log.duration_seconds = (scrape_log.end_time - scrape_log.start_time).total_seconds()
            scrape_log.events_found = len(events)
            scrape_log.status = ScrapeStatus.SUCCESS
            scrape_log.date_from = target_date - timedelta(days=7)
            scrape_log.date_to = target_date

            logger.info(f"Successfully scraped {len(events)} events for week {week_offset}")

        except Exception as e:
            scrape_log.end_time = datetime.utcnow()
            scrape_log.duration_seconds = (scrape_log.end_time - scrape_log.start_time).total_seconds()
            scrape_log.status = ScrapeStatus.FAILED
            scrape_log.error_message = str(e)
            scrape_log.error_count = 1

            logger.error(f"Scraping failed: {e}")

        return events, scrape_log

    def save_events_to_database(self, events_data: List[Dict], scrape_log: ScrapeLog) -> None:
        """Save scraped events to database"""
        try:
            with db_manager.get_session() as session:
                # Create a new scrape log instance to avoid session issues
                new_scrape_log = ScrapeLog(
                    scrape_date=scrape_log.scrape_date,
                    data_source=scrape_log.data_source,
                    scrape_type=scrape_log.scrape_type,
                    start_time=scrape_log.start_time,
                    status=scrape_log.status,
                    end_time=scrape_log.end_time,
                    duration_seconds=scrape_log.duration_seconds,
                    date_from=scrape_log.date_from,
                    date_to=scrape_log.date_to
                )

                # Save scrape log first
                session.add(new_scrape_log)
                session.flush()  # Get the ID

                events_new = 0
                events_updated = 0
                events_failed = 0

                for event_data in events_data:
                    try:
                        # Skip events with invalid required data
                        if not event_data.get('event_name') or not event_data.get('currency') or not event_data.get('event_date'):
                            logger.warning(f"Skipping event with missing required data: {event_data.get('event_name', 'Unknown')}")
                            events_failed += 1
                            continue

                        # Check if event already exists
                        existing_event = session.query(EconomicEvent).filter(
                            EconomicEvent.event_name == event_data['event_name'],
                            EconomicEvent.currency == event_data['currency'],
                            EconomicEvent.event_date == event_data['event_date']
                        ).first()

                        if existing_event:
                            # Update existing event
                            for key, value in event_data.items():
                                if hasattr(existing_event, key):
                                    setattr(existing_event, key, value)
                            existing_event.updated_at = datetime.utcnow()
                            events_updated += 1
                        else:
                            # Create new event
                            event = EconomicEvent(**event_data)
                            session.add(event)
                            events_new += 1

                    except Exception as e:
                        logger.error(f"Failed to save event {event_data.get('event_name', 'unknown')}: {e}")
                        events_failed += 1

                # Update scrape log with final counts
                new_scrape_log.events_new = events_new
                new_scrape_log.events_updated = events_updated
                new_scrape_log.events_failed = events_failed

                session.commit()

                # Update the original scrape_log for return reference
                scrape_log.events_new = events_new
                scrape_log.events_updated = events_updated
                scrape_log.events_failed = events_failed

                logger.info(f"Saved events to database: {events_new} new, {events_updated} updated, {events_failed} failed")

        except Exception as e:
            logger.error(f"Failed to save events to database: {e}")
            raise

    def cleanup_old_data(self, retention_days: int = None) -> None:
        """Clean up old events and scrape logs"""
        retention_days = retention_days or config.EVENT_RETENTION_DAYS
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        try:
            with db_manager.get_session() as session:
                # Clean up old events
                old_events = session.query(EconomicEvent).filter(
                    EconomicEvent.event_date < cutoff_date
                ).delete()

                # Clean up old scrape logs
                log_retention_days = config.SCRAPE_LOG_RETENTION_DAYS
                log_cutoff_date = datetime.utcnow() - timedelta(days=log_retention_days)

                old_logs = session.query(ScrapeLog).filter(
                    ScrapeLog.scrape_date < log_cutoff_date
                ).delete()

                session.commit()

                logger.info(f"Cleaned up {old_events} old events and {old_logs} old scrape logs")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            raise