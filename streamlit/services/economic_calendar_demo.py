"""
Economic Calendar Demo Data
Provides sample economic events for demonstration when live data is not available
"""

from datetime import datetime, timedelta
from typing import List, Dict


def get_demo_economic_events(epic: str) -> List[Dict]:
    """
    Generate demo economic events for the selected trading pair
    Shows how the integration will look with real upcoming events
    """
    from services.economic_calendar_service import get_economic_calendar_service

    service = get_economic_calendar_service()
    base_currency, quote_currency = service.extract_currencies_from_epic(epic)

    if not base_currency or not quote_currency:
        return []

    # Current time for relative calculations
    now = datetime.now()

    # Create sample events for the currency pair
    demo_events = []

    # Define sample events for major currencies
    sample_events = {
        'EUR': [
            {'name': 'ECB Interest Rate Decision', 'impact': 'high', 'hours_ahead': 2.5},
            {'name': 'Eurozone CPI m/m', 'impact': 'high', 'hours_ahead': 26},
            {'name': 'German Manufacturing PMI', 'impact': 'medium', 'hours_ahead': 50},
            {'name': 'ECB President Lagarde Speech', 'impact': 'medium', 'hours_ahead': 74},
        ],
        'USD': [
            {'name': 'US Core CPI m/m', 'impact': 'high', 'hours_ahead': 8.5},
            {'name': 'Federal Funds Rate', 'impact': 'high', 'hours_ahead': 32},
            {'name': 'US Retail Sales m/m', 'impact': 'medium', 'hours_ahead': 56},
            {'name': 'FOMC Meeting Minutes', 'impact': 'medium', 'hours_ahead': 80},
        ],
        'GBP': [
            {'name': 'BoE Interest Rate Decision', 'impact': 'high', 'hours_ahead': 4},
            {'name': 'UK CPI y/y', 'impact': 'high', 'hours_ahead': 28},
            {'name': 'UK Employment Change', 'impact': 'medium', 'hours_ahead': 52},
        ],
        'JPY': [
            {'name': 'BoJ Interest Rate Decision', 'impact': 'high', 'hours_ahead': 12},
            {'name': 'Japan Core CPI y/y', 'impact': 'medium', 'hours_ahead': 36},
            {'name': 'Japan Manufacturing PMI', 'impact': 'low', 'hours_ahead': 60},
        ],
        'AUD': [
            {'name': 'RBA Interest Rate Decision', 'impact': 'high', 'hours_ahead': 6},
            {'name': 'Australia Employment Change', 'impact': 'medium', 'hours_ahead': 30},
            {'name': 'Australia CPI q/q', 'impact': 'medium', 'hours_ahead': 54},
        ],
        'CAD': [
            {'name': 'BoC Interest Rate Decision', 'impact': 'high', 'hours_ahead': 10},
            {'name': 'Canada CPI m/m', 'impact': 'medium', 'hours_ahead': 34},
            {'name': 'Canada Employment Change', 'impact': 'medium', 'hours_ahead': 58},
        ],
        'CHF': [
            {'name': 'SNB Interest Rate Decision', 'impact': 'high', 'hours_ahead': 14},
            {'name': 'Switzerland CPI m/m', 'impact': 'low', 'hours_ahead': 38},
        ]
    }

    # Create events for base and quote currencies
    for currency in [base_currency, quote_currency]:
        if currency in sample_events:
            for event_data in sample_events[currency]:
                event_time = now + timedelta(hours=event_data['hours_ahead'])

                # Create demo event in the same format as real events
                demo_event = {
                    'id': f"demo_{currency}_{event_data['name'].replace(' ', '_').lower()}",
                    'event_name': event_data['name'],
                    'currency': currency,
                    'country': None,
                    'event_date': event_time.isoformat(),
                    'event_time': event_time.strftime('%H:%M'),
                    'impact_level': event_data['impact'],
                    'previous_value': get_demo_previous_value(event_data['name']),
                    'forecast_value': get_demo_forecast_value(event_data['name']),
                    'actual_value': None,
                    'category': None,
                    'source': 'demo',
                    'market_moving': event_data['impact'] in ['high', 'medium'],
                    'created_at': now.isoformat()
                }

                # Add parsed fields using pandas and service methods
                import pandas as pd
                demo_event['parsed_datetime'] = pd.to_datetime(event_time)
                demo_event['time_until'] = service._calculate_time_until(pd.to_datetime(event_time))
                demo_event['is_relevant'] = True
                demo_event['pair_currencies'] = [base_currency, quote_currency]

                demo_events.append(demo_event)

    # Sort by event time
    demo_events.sort(key=lambda x: x['parsed_datetime'])

    return demo_events


def get_demo_previous_value(event_name: str) -> str:
    """Generate realistic previous values for demo events"""
    value_map = {
        'Interest Rate': '4.00%',
        'CPI': '0.2%',
        'Employment': '15.2K',
        'PMI': '52.1',
        'Retail Sales': '0.4%',
        'Speech': '',
        'Minutes': '',
        'Decision': '4.25%'
    }

    for key, value in value_map.items():
        if key in event_name:
            return value
    return '0.1%'


def get_demo_forecast_value(event_name: str) -> str:
    """Generate realistic forecast values for demo events"""
    forecast_map = {
        'Interest Rate': '4.25%',
        'CPI': '0.3%',
        'Employment': '18.5K',
        'PMI': '52.8',
        'Retail Sales': '0.2%',
        'Speech': '',
        'Minutes': '',
        'Decision': '4.50%'
    }

    for key, value in forecast_map.items():
        if key in event_name:
            return value
    return '0.2%'


def is_demo_mode_enabled() -> bool:
    """Check if demo mode should be enabled (when no real upcoming events available)"""
    import os
    return os.getenv('ECONOMIC_CALENDAR_DEMO', 'true').lower() == 'true'