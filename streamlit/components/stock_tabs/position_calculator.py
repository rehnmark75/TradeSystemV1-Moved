"""
Position Size Calculator Component

Provides risk-based position sizing for trades:
- Account size from broker (auto-loaded) or manual input
- Risk per trade (% or $)
- Entry price and stop loss
- Calculates shares to buy and position value
- Warns on concentration risk
"""

import streamlit as st
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

# Import broker service for account balance
try:
    from services.broker_analytics_service import get_broker_service
    BROKER_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import broker service: {e}")
    BROKER_SERVICE_AVAILABLE = False
    get_broker_service = None


def _get_broker_account_value() -> Optional[float]:
    """
    Get the total account value from broker data.

    Returns:
        Total account value in USD, or None if not available
    """
    if not BROKER_SERVICE_AVAILABLE:
        logger.debug("Broker service not available")
        return None

    try:
        broker_service = get_broker_service()
        if not broker_service.is_configured:
            logger.warning("Broker service database not configured")
            return None

        balance = broker_service.get_account_balance()
        logger.debug(f"Broker balance response: {balance}")

        if balance and not balance.get('error'):
            total_value = balance.get('total_value', 0.0)
            if total_value > 0:
                logger.info(f"Loaded broker account value: ${total_value:,.2f}")
                return float(total_value)
        elif balance and balance.get('error'):
            logger.warning(f"Broker balance error: {balance.get('error')}")
    except Exception as e:
        logger.error(f"Error fetching broker account value: {e}")

    return None


def render_position_calculator(
    service: Any = None,
    ticker: str = None,
    entry_price: float = None,
    stop_loss: float = None,
    in_sidebar: bool = False
) -> Optional[Dict]:
    """
    Render a position size calculator widget.

    Args:
        service: StockAnalyticsService for fetching stock data
        ticker: Pre-selected ticker
        entry_price: Pre-populated entry price
        stop_loss: Pre-populated stop loss
        in_sidebar: Whether to render in sidebar (more compact)

    Returns:
        Dict with calculated values if valid, None otherwise
    """
    # Initialize session state for account size (fallback)
    if 'account_size' not in st.session_state:
        st.session_state.account_size = 25000.0

    # Try to get account value from broker first
    # Re-fetch if not loaded yet OR if still at default value (previous fetch may have failed)
    should_fetch_broker = (
        'broker_account_loaded' not in st.session_state or
        (st.session_state.get('account_source') != 'broker' and
         st.session_state.account_size == 25000.0)
    )

    if should_fetch_broker:
        broker_value = _get_broker_account_value()
        if broker_value and broker_value > 0:
            st.session_state.account_size = broker_value
            st.session_state.broker_account_loaded = True
            st.session_state.account_source = 'broker'
        else:
            st.session_state.broker_account_loaded = True
            st.session_state.account_source = 'manual'
    if 'risk_percent' not in st.session_state:
        st.session_state.risk_percent = 1.0
    if 'portfolio_heat' not in st.session_state:
        st.session_state.portfolio_heat = 0.0  # Total exposure

    container = st.sidebar if in_sidebar else st

    with container:
        if not in_sidebar:
            account_source = st.session_state.get('account_source', 'manual')
            if account_source == 'broker':
                st.markdown("### Position Size Calculator <span style='font-size: 0.7em; color: #4CAF50;'>(Broker)</span>", unsafe_allow_html=True)
            else:
                st.markdown("### Position Size Calculator")

        # Account settings
        col1, col2 = st.columns(2)

        with col1:
            account_label = "Broker Account ($)" if st.session_state.get('account_source') == 'broker' else "Account Size ($)"
            account_size = st.number_input(
                account_label,
                min_value=1000.0,
                max_value=10000000.0,
                value=st.session_state.account_size,
                step=1000.0,
                format="%.0f",
                key="pos_calc_account"
            )
            st.session_state.account_size = account_size

        with col2:
            risk_pct = st.number_input(
                "Risk Per Trade (%)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.risk_percent,
                step=0.1,
                format="%.1f",
                key="pos_calc_risk"
            )
            st.session_state.risk_percent = risk_pct

        st.divider()

        # Trade parameters
        if ticker and service:
            # Fetch data for ticker
            pos_data = service.get_position_sizing_data(ticker)
            metrics = pos_data.get('metrics', {})
            signal = pos_data.get('signal', {})

            default_entry = entry_price or float(signal.get('entry_price') or metrics.get('current_price') or 0)
            default_stop = stop_loss or float(signal.get('stop_loss') or pos_data.get('suggested_stop_long') or 0)

            st.markdown(f"**Ticker: {ticker}**")
            if metrics.get('current_price'):
                st.caption(f"Current: ${float(metrics['current_price']):.2f} | ATR: {float(metrics.get('atr_percent') or 0):.1f}%")
        else:
            default_entry = entry_price or 0.0
            default_stop = stop_loss or 0.0

        col1, col2 = st.columns(2)

        with col1:
            entry = st.number_input(
                "Entry Price ($)",
                min_value=0.01,
                value=max(default_entry, 0.01),
                step=0.01,
                format="%.2f",
                key="pos_calc_entry"
            )

        with col2:
            stop = st.number_input(
                "Stop Loss ($)",
                min_value=0.01,
                value=max(default_stop, 0.01),
                step=0.01,
                format="%.2f",
                key="pos_calc_stop"
            )

        # Calculate position size
        if entry > 0 and stop > 0 and entry != stop:
            risk_per_share = abs(entry - stop)
            risk_dollars = account_size * (risk_pct / 100)
            shares = int(risk_dollars / risk_per_share)
            position_value = shares * entry
            position_pct = (position_value / account_size) * 100

            # Determine trade direction
            is_long = stop < entry

            # Risk/Reward placeholder (target = 2x risk)
            if is_long:
                target_price = entry + (2 * risk_per_share)
            else:
                target_price = entry - (2 * risk_per_share)

            st.divider()

            # Results display
            st.markdown("#### Calculation Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Shares to Buy", f"{shares:,}")
                st.metric("Position Value", f"${position_value:,.0f}")

            with col2:
                st.metric("Dollar Risk", f"${risk_dollars:,.0f}")
                st.metric("Position Size", f"{position_pct:.1f}%")

            # Warnings
            warnings = []

            if position_pct > 25:
                warnings.append(("Position > 25% of account - HIGH CONCENTRATION", "error"))
            elif position_pct > 15:
                warnings.append(("Position > 15% of account - moderate concentration", "warning"))

            if shares < 1:
                warnings.append(("Cannot buy even 1 share with this risk amount", "error"))

            if risk_per_share / entry > 0.15:
                warnings.append(("Stop is >15% from entry - wide stop loss", "warning"))

            for msg, level in warnings:
                if level == "error":
                    st.error(msg)
                else:
                    st.warning(msg)

            # Additional info
            st.markdown("---")
            st.markdown(f"""
            **Trade Details:**
            - Direction: {'Long' if is_long else 'Short'}
            - Risk per share: ${risk_per_share:.2f}
            - Risk %: {(risk_per_share/entry)*100:.1f}%
            - Target (2R): ${target_price:.2f}
            """)

            return {
                'ticker': ticker,
                'entry': entry,
                'stop': stop,
                'shares': shares,
                'position_value': position_value,
                'risk_dollars': risk_dollars,
                'position_pct': position_pct,
                'is_long': is_long,
                'target_2r': target_price
            }

        else:
            st.info("Enter valid entry and stop loss prices to calculate position size.")
            return None


def render_position_calculator_modal(service: Any, ticker: str, signal_data: dict = None) -> None:
    """
    Render position calculator in an expander/modal style.

    Args:
        service: StockAnalyticsService
        ticker: Stock ticker
        signal_data: Optional dict with entry_price, stop_loss from signal
    """
    with st.expander(f"Calculate Position Size for {ticker}", expanded=False):
        entry = float(signal_data.get('entry_price', 0)) if signal_data else None
        stop = float(signal_data.get('stop_loss', 0)) if signal_data else None

        result = render_position_calculator(
            service=service,
            ticker=ticker,
            entry_price=entry,
            stop_loss=stop,
            in_sidebar=False
        )

        if result and result.get('shares', 0) > 0:
            st.success(f"Buy {result['shares']:,} shares of {ticker} at ${result['entry']:.2f}")


def render_quick_position_calc(entry: float, stop: float, account_size: float = 25000, risk_pct: float = 1.0) -> Dict:
    """
    Quick position calculation without UI.

    Args:
        entry: Entry price
        stop: Stop loss price
        account_size: Account value
        risk_pct: Risk percentage

    Returns:
        Dict with shares, position_value, risk_dollars
    """
    if entry <= 0 or stop <= 0 or entry == stop:
        return {'shares': 0, 'position_value': 0, 'risk_dollars': 0}

    risk_per_share = abs(entry - stop)
    risk_dollars = account_size * (risk_pct / 100)
    shares = int(risk_dollars / risk_per_share)
    position_value = shares * entry

    return {
        'shares': shares,
        'position_value': position_value,
        'risk_dollars': risk_dollars,
        'position_pct': (position_value / account_size) * 100 if account_size > 0 else 0
    }


def render_inline_position_calculator(
    ticker: str,
    current_price: float = None,
    suggested_entry: float = None,
    suggested_stop: float = None,
    atr_percent: float = None,
) -> Optional[Dict]:
    """
    Render a compact inline position calculator for the chart analysis section.

    Uses total account value from broker data (if available) or session state.
    Designed to fit within the analysis flow without taking too much space.

    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        suggested_entry: Suggested entry price (from signal or current)
        suggested_stop: Suggested stop loss price
        atr_percent: ATR as percentage for default stop calculation

    Returns:
        Dict with calculated position details or None
    """
    # Initialize session state for account settings (fallback)
    if 'account_size' not in st.session_state:
        st.session_state.account_size = 25000.0
    if 'risk_percent' not in st.session_state:
        st.session_state.risk_percent = 1.0

    # Try to get account value from broker first
    # Re-fetch if not loaded yet OR if still at default value (previous fetch may have failed)
    should_fetch_broker = (
        'broker_account_loaded' not in st.session_state or
        (st.session_state.get('account_source') != 'broker' and
         st.session_state.account_size == 25000.0)
    )

    if should_fetch_broker:
        broker_value = _get_broker_account_value()
        if broker_value and broker_value > 0:
            st.session_state.account_size = broker_value
            st.session_state.broker_account_loaded = True
            st.session_state.account_source = 'broker'
        else:
            st.session_state.broker_account_loaded = True
            st.session_state.account_source = 'manual'

    # Header with source indicator
    account_source = st.session_state.get('account_source', 'manual')
    if account_source == 'broker':
        st.markdown("### Position Calculator <span style='font-size: 0.7em; color: #4CAF50;'>(Broker Account)</span>", unsafe_allow_html=True)
    else:
        st.markdown("### Position Calculator")

    # Calculate default stop if not provided (using ATR)
    if suggested_stop is None and current_price and atr_percent:
        # Default stop = 1.5x ATR below entry for long
        suggested_stop = current_price * (1 - (atr_percent * 1.5 / 100))

    # Use current price as default entry if not provided
    default_entry = suggested_entry or current_price or 0.0
    default_stop = suggested_stop or (default_entry * 0.95 if default_entry > 0 else 0.0)

    # Account settings row
    col1, col2, col3, col4 = st.columns([1.5, 1, 1.5, 1])

    with col1:
        # Show account input with broker value as default
        account_label = "Total Account ($)"
        if account_source == 'broker':
            account_label = "Broker Account ($)"

        account_size = st.number_input(
            account_label,
            min_value=1000.0,
            max_value=10000000.0,
            value=st.session_state.account_size,
            step=1000.0,
            format="%.0f",
            key=f"inline_pos_account_{ticker}",
            help="Your total trading account value (auto-loaded from broker if available)"
        )
        st.session_state.account_size = account_size

    with col2:
        risk_pct = st.number_input(
            "Risk %",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.risk_percent,
            step=0.25,
            format="%.2f",
            key=f"inline_pos_risk_{ticker}",
            help="Percentage of account to risk per trade"
        )
        st.session_state.risk_percent = risk_pct

    with col3:
        entry = st.number_input(
            "Entry Price ($)",
            min_value=0.01,
            value=max(default_entry, 0.01),
            step=0.01,
            format="%.2f",
            key=f"inline_pos_entry_{ticker}"
        )

    with col4:
        stop = st.number_input(
            "Stop Loss ($)",
            min_value=0.01,
            value=max(default_stop, 0.01),
            step=0.01,
            format="%.2f",
            key=f"inline_pos_stop_{ticker}"
        )

    # Calculate and display results
    if entry > 0 and stop > 0 and entry != stop:
        risk_per_share = abs(entry - stop)
        risk_dollars = account_size * (risk_pct / 100)
        shares = int(risk_dollars / risk_per_share)
        position_value = shares * entry
        position_pct = (position_value / account_size) * 100 if account_size > 0 else 0
        is_long = stop < entry

        # Calculate targets
        if is_long:
            target_1r = entry + risk_per_share
            target_2r = entry + (2 * risk_per_share)
            target_3r = entry + (3 * risk_per_share)
        else:
            target_1r = entry - risk_per_share
            target_2r = entry - (2 * risk_per_share)
            target_3r = entry - (3 * risk_per_share)

        # Results in compact format
        result_cols = st.columns(5)

        with result_cols[0]:
            st.metric("Shares", f"{shares:,}" if shares > 0 else "0")

        with result_cols[1]:
            st.metric("Position $", f"${position_value:,.0f}")

        with result_cols[2]:
            # Color-code position size
            pct_color = "normal"
            if position_pct > 25:
                pct_color = "inverse"  # Red warning
            elif position_pct > 15:
                pct_color = "off"  # Neutral warning
            st.metric("Position %", f"{position_pct:.1f}%", delta=None)

        with result_cols[3]:
            st.metric("Risk $", f"${risk_dollars:,.0f}")

        with result_cols[4]:
            risk_per_share_pct = (risk_per_share / entry) * 100
            st.metric("Risk/Share", f"{risk_per_share_pct:.1f}%")

        # Targets row
        st.markdown(f"""
        <div style="background-color: #1a1a2e; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <span style="color: #888;">Direction:</span> <strong>{'LONG' if is_long else 'SHORT'}</strong> |
            <span style="color: #888;">Target 1R:</span> <strong>${target_1r:.2f}</strong> |
            <span style="color: #888;">Target 2R:</span> <strong>${target_2r:.2f}</strong> |
            <span style="color: #888;">Target 3R:</span> <strong>${target_3r:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Warnings
        if shares < 1:
            st.error("Cannot buy even 1 share with this risk amount. Increase account size or risk %.")
        elif position_pct > 25:
            st.warning(f"Position is {position_pct:.1f}% of account - HIGH CONCENTRATION risk!")
        elif position_pct > 15:
            st.info(f"Position is {position_pct:.1f}% of account - moderate concentration.")

        return {
            'ticker': ticker,
            'entry': entry,
            'stop': stop,
            'shares': shares,
            'position_value': position_value,
            'risk_dollars': risk_dollars,
            'position_pct': position_pct,
            'is_long': is_long,
            'target_1r': target_1r,
            'target_2r': target_2r,
            'target_3r': target_3r,
            'account_size': account_size,
            'risk_percent': risk_pct
        }
    else:
        st.caption("Enter valid entry and stop loss prices to calculate position size.")
        return None
