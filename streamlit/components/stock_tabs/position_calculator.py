"""
Position Size Calculator Component

Provides risk-based position sizing for trades:
- Account size input (persisted in session)
- Risk per trade (% or $)
- Entry price and stop loss
- Calculates shares to buy and position value
- Warns on concentration risk
"""

import streamlit as st
from typing import Any, Optional, Dict


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
    # Initialize session state for account size
    if 'account_size' not in st.session_state:
        st.session_state.account_size = 25000.0
    if 'risk_percent' not in st.session_state:
        st.session_state.risk_percent = 1.0
    if 'portfolio_heat' not in st.session_state:
        st.session_state.portfolio_heat = 0.0  # Total exposure

    container = st.sidebar if in_sidebar else st

    with container:
        if not in_sidebar:
            st.markdown("### Position Size Calculator")

        # Account settings
        col1, col2 = st.columns(2)

        with col1:
            account_size = st.number_input(
                "Account Size ($)",
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
