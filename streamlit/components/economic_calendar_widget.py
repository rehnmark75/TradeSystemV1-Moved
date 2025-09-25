"""
Economic Calendar UI Components for TradingView Chart Page
Provides sidebar widgets and chart markers for economic events
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from services.economic_calendar_service import get_economic_calendar_service

logger = logging.getLogger(__name__)

def render_economic_calendar_sidebar(selected_epic: str) -> Optional[List[Dict]]:
    """
    Render economic calendar sidebar section for the selected trading pair
    Returns list of events for potential chart integration
    """
    try:
        calendar_service = get_economic_calendar_service()

        # Extract currencies for display
        base_currency, quote_currency = calendar_service.extract_currencies_from_epic(selected_epic)
        if not base_currency or not quote_currency:
            return None

        # Create collapsible section
        with st.sidebar.expander(f"📅 Economic Calendar - {base_currency}/{quote_currency}", expanded=True):

            # Quick filter tabs
            tab1, tab2, tab3 = st.columns(3)

            with tab1:
                show_today = st.checkbox("Today", value=True, help="Show today's events")
            with tab2:
                show_tomorrow = st.checkbox("Tomorrow", value=True, help="Show tomorrow's events")
            with tab3:
                show_week = st.checkbox("This Week", value=False, help="Show all events this week")

            # Impact level filter
            impact_filter = st.multiselect(
                "Impact Level",
                options=['High', 'Medium', 'Low'],
                default=['High', 'Medium'],
                help="Filter events by market impact"
            )

            # Fetch events based on timeframe
            if show_week:
                events = calendar_service.get_events_this_week(selected_epic)
            else:
                hours_ahead = 48 if show_tomorrow else 24
                events = calendar_service.get_relevant_events(
                    selected_epic,
                    hours_ahead=hours_ahead,
                    impact_levels=impact_filter if impact_filter else None
                )

            if not events:
                st.info("No relevant economic events found.")
                return None

            # Filter events by day selection
            filtered_events = []
            for event in events:
                formatted_event = calendar_service.format_event_for_display(event)

                # Apply day filters
                include_event = False
                if show_today and formatted_event['is_today']:
                    include_event = True
                elif show_tomorrow and formatted_event['is_tomorrow']:
                    include_event = True
                elif show_week:
                    include_event = True

                if include_event:
                    filtered_events.append(formatted_event)

            if not filtered_events:
                st.info("No events match your filters.")
                return None

            # Display event count
            st.caption(f"Found {len(filtered_events)} relevant events")

            # Render events
            for event in filtered_events[:10]:  # Limit to 10 most relevant
                render_event_card(event)

            # Show chart markers toggle
            show_markers = st.checkbox(
                "Show on Chart",
                value=False,
                help="Add event markers to the price chart"
            )

            if show_markers:
                return events  # Return raw events for chart integration

        return None

    except Exception as e:
        st.sidebar.error(f"Economic Calendar Error: {str(e)[:50]}...")
        logger.error(f"Error rendering economic calendar sidebar: {e}")
        return None

def render_event_card(event: Dict):
    """Render individual event card in sidebar"""
    try:
        # Event container with impact color
        with st.container():
            # Header with impact icon and currency
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                st.write(event['impact_icon'])
            with col2:
                st.write(f"**{event['currency']}**")
            with col3:
                st.caption(event['time_until'])

            # Event name
            st.write(event['name'])

            # Values row if available
            if event['previous_value'] or event['forecast_value'] or event['actual_value']:
                value_cols = st.columns(3)

                if event['previous_value']:
                    with value_cols[0]:
                        st.caption(f"Prev: {event['previous_value']}")
                if event['forecast_value']:
                    with value_cols[1]:
                        st.caption(f"Fcst: {event['forecast_value']}")
                if event['actual_value']:
                    with value_cols[2]:
                        st.caption(f"Act: {event['actual_value']}")

            # Add visual separator
            st.divider()

    except Exception as e:
        logger.error(f"Error rendering event card: {e}")
        st.error("Error displaying event")

def render_upcoming_alerts(selected_epic: str):
    """Render upcoming high-impact event alerts"""
    try:
        calendar_service = get_economic_calendar_service()

        # Get high-impact events in next 6 hours
        high_impact_events = calendar_service.get_upcoming_high_impact_events(
            selected_epic,
            hours_ahead=6
        )

        if not high_impact_events:
            return

        # Show alert for imminent high-impact events
        for event in high_impact_events[:2]:  # Max 2 alerts
            formatted_event = calendar_service.format_event_for_display(event)

            # Only show if event is within next 6 hours
            if 'In' in formatted_event['time_until'] and ('min' in formatted_event['time_until'] or 'h' in formatted_event['time_until']):
                st.sidebar.warning(
                    f"⚠️ **{formatted_event['currency']} High Impact**\n\n"
                    f"{formatted_event['name']}\n\n"
                    f"**{formatted_event['time_until']}**"
                )

    except Exception as e:
        logger.error(f"Error rendering upcoming alerts: {e}")

def generate_chart_markers(events: List[Dict], visible_candle_times: List[int], timeframe_minutes: int) -> List[Dict]:
    """
    Generate chart markers for economic events
    Returns markers in the format expected by TradingView charts
    """
    try:
        markers = []
        calendar_service = get_economic_calendar_service()

        for event in events:
            try:
                event_datetime = event.get('parsed_datetime')
                if not event_datetime:
                    continue

                # Convert to timestamp
                event_timestamp = int(event_datetime.timestamp())

                # Find nearest candle time
                nearest_candle_time = find_nearest_candle_time(
                    event_timestamp,
                    visible_candle_times,
                    timeframe_minutes
                )

                if not nearest_candle_time:
                    continue

                # Create marker
                impact_level = event.get('impact_level', 'low').lower()

                marker = {
                    "time": nearest_candle_time,
                    "position": "aboveBar",
                    "color": calendar_service.get_impact_color(impact_level),
                    "shape": "arrowDown",
                    "text": f"{event.get('currency', '')} {event.get('event_name', '')[:20]}...",
                    "size": 2 if impact_level == 'high' else 1
                }

                markers.append(marker)

            except Exception as e:
                logger.warning(f"Failed to create marker for event: {e}")
                continue

        logger.info(f"Generated {len(markers)} economic calendar markers")
        return markers

    except Exception as e:
        logger.error(f"Error generating chart markers: {e}")
        return []

def find_nearest_candle_time(event_timestamp: int, candle_times: List[int], timeframe_minutes: int) -> Optional[int]:
    """Find the nearest candle time to an event timestamp"""
    try:
        if not candle_times:
            return None

        # Find the candle that would contain this event
        sorted_times = sorted(candle_times)

        for i, candle_time in enumerate(sorted_times):
            # Check if event falls within this candle's time range
            if i < len(sorted_times) - 1:
                next_candle_time = sorted_times[i + 1]
                if candle_time <= event_timestamp < next_candle_time:
                    return candle_time
            else:
                # Last candle - check if event is within timeframe
                if candle_time <= event_timestamp < candle_time + (timeframe_minutes * 60):
                    return candle_time

        # If not within any candle range, find closest
        closest_time = min(sorted_times, key=lambda x: abs(x - event_timestamp))
        return closest_time

    except Exception as e:
        logger.error(f"Error finding nearest candle time: {e}")
        return None

def render_market_intelligence_enhancement(selected_epic: str, existing_regime_data: Dict = None):
    """
    Enhance existing market intelligence with economic calendar context
    """
    try:
        calendar_service = get_economic_calendar_service()

        # Get high-impact events in next 24 hours
        upcoming_events = calendar_service.get_upcoming_high_impact_events(
            selected_epic,
            hours_ahead=24
        )

        if not upcoming_events:
            return

        # Get the most imminent high-impact event
        next_event = upcoming_events[0]
        formatted_event = calendar_service.format_event_for_display(next_event)

        # Enhance market intelligence display
        if existing_regime_data:
            regime_name = existing_regime_data.get('regime_name', 'unknown')
            confidence = existing_regime_data.get('confidence', 0)

            # Create enhanced message
            enhanced_message = (
                f"🎯 **{regime_name.upper()} REGIME** ({confidence:.1%}) | "
                f"📅 **{formatted_event['currency']} {formatted_event['name'][:30]}** {formatted_event['time_until']}"
            )

            st.warning(enhanced_message)

            # Add volatility warning
            if formatted_event['impact_level'] == 'high':
                st.info("⚡ **High volatility expected** - Consider adjusting position sizes and stop levels")

    except Exception as e:
        logger.error(f"Error enhancing market intelligence: {e}")

def render_economic_calendar_summary():
    """Render a compact summary of today's key events"""
    try:
        calendar_service = get_economic_calendar_service()

        # This would be called with the current epic
        # For now, show general high-impact events
        st.sidebar.subheader("📊 Today's Key Events")

        # Placeholder for summary - would be integrated with actual epic selection
        st.sidebar.info("Economic calendar integration ready")

    except Exception as e:
        logger.error(f"Error rendering economic calendar summary: {e}")
        st.sidebar.error("Calendar service unavailable")