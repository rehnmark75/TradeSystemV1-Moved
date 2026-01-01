"""
Custom Screener Builder Tab

Allows users to build custom stock screens using flexible filter criteria:
- Technical filters (EMA, RSI, MACD, Volume)
- Relative Strength filters (RS percentile, RS trend)
- Trend filters (trend strength, MA alignment)
- Fundamental filters (sector, market cap)

Screens can be saved and loaded for reuse.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List


# Available filter fields with their types and options
FILTER_FIELDS = {
    'rs_percentile': {
        'label': 'RS Percentile',
        'type': 'range',
        'min': 0,
        'max': 100,
        'description': 'Relative Strength vs SPY (0-100)'
    },
    'rs_trend': {
        'label': 'RS Trend',
        'type': 'select',
        'options': ['improving', 'stable', 'deteriorating'],
        'description': 'RS momentum direction'
    },
    'trend_strength': {
        'label': 'Trend Strength',
        'type': 'select',
        'options': ['strong_up', 'up', 'sideways', 'down', 'strong_down'],
        'description': 'Current trend classification'
    },
    'current_price': {
        'label': 'Current Price',
        'type': 'range',
        'min': 0,
        'max': 10000,
        'description': 'Stock price filter'
    },
    'rsi_14': {
        'label': 'RSI (14)',
        'type': 'range',
        'min': 0,
        'max': 100,
        'description': 'Relative Strength Index'
    },
    'atr_percent': {
        'label': 'ATR %',
        'type': 'range',
        'min': 0,
        'max': 20,
        'description': 'Average True Range as % of price'
    },
    'relative_volume': {
        'label': 'Relative Volume',
        'type': 'range',
        'min': 0,
        'max': 10,
        'description': 'Volume vs 20-day average'
    },
    'price_vs_ema_50': {
        'label': 'Price vs EMA 50',
        'type': 'select',
        'options': ['above', 'below'],
        'description': 'Price position relative to EMA 50'
    },
    'price_vs_ema_200': {
        'label': 'Price vs EMA 200',
        'type': 'select',
        'options': ['above', 'below'],
        'description': 'Price position relative to EMA 200'
    },
    'ma_alignment': {
        'label': 'MA Alignment',
        'type': 'select',
        'options': ['bullish', 'bearish', 'mixed'],
        'description': 'EMA stack alignment'
    },
    'sector': {
        'label': 'Sector',
        'type': 'multi_select',
        'options': [
            'Technology', 'Health Care', 'Financials', 'Consumer Discretionary',
            'Communication Services', 'Industrials', 'Consumer Staples',
            'Energy', 'Utilities', 'Real Estate', 'Materials'
        ],
        'description': 'Sector filter'
    },
    'smc_trend': {
        'label': 'SMC Trend',
        'type': 'select',
        'options': ['Bullish', 'Bearish', 'Neutral'],
        'description': 'Smart Money Concepts trend'
    }
}

# Preset screeners
PRESET_SCREENERS = {
    'rs_leaders': {
        'name': 'RS Leaders',
        'description': 'Top RS stocks in uptrend',
        'filters': [
            {'field': 'rs_percentile', 'operator': '>=', 'value': 80},
            {'field': 'trend_strength', 'operator': 'in', 'value': ['strong_up', 'up']},
            {'field': 'price_vs_ema_200', 'operator': '=', 'value': 'above'}
        ]
    },
    'pullback_candidates': {
        'name': 'Pullback Candidates',
        'description': 'Strong stocks pulling back to support',
        'filters': [
            {'field': 'rs_percentile', 'operator': '>=', 'value': 70},
            {'field': 'rsi_14', 'operator': '<=', 'value': 45},
            {'field': 'price_vs_ema_50', 'operator': '=', 'value': 'above'},
            {'field': 'ma_alignment', 'operator': '=', 'value': 'bullish'}
        ]
    },
    'high_volatility_momentum': {
        'name': 'High Volatility Momentum',
        'description': 'Volatile stocks with strong momentum',
        'filters': [
            {'field': 'atr_percent', 'operator': '>=', 'value': 4},
            {'field': 'relative_volume', 'operator': '>=', 'value': 1.5},
            {'field': 'trend_strength', 'operator': 'in', 'value': ['strong_up', 'up']}
        ]
    },
    'smc_bullish_discount': {
        'name': 'SMC Bullish Discount',
        'description': 'Bullish SMC trend in discount zone',
        'filters': [
            {'field': 'smc_trend', 'operator': '=', 'value': 'Bullish'},
            {'field': 'rs_percentile', 'operator': '>=', 'value': 60}
        ]
    }
}


def render_screener_builder_tab(service: Any) -> None:
    """Render the Custom Screener Builder tab."""
    st.markdown("""
    <div class="main-header">
        <h2>Custom Screener Builder</h2>
        <p>Build and save custom stock screens using flexible filter criteria</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for filters
    if 'custom_filters' not in st.session_state:
        st.session_state.custom_filters = []

    # Preset selector or custom builder
    tab1, tab2 = st.tabs(["Preset Screeners", "Custom Builder"])

    with tab1:
        _render_preset_screeners(service)

    with tab2:
        _render_custom_builder(service)


def _render_preset_screeners(service: Any) -> None:
    """Render preset screener selection and execution."""
    st.markdown("### Select a Preset Screener")

    cols = st.columns(2)

    for i, (key, preset) in enumerate(PRESET_SCREENERS.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"**{preset['name']}**")
                st.caption(preset['description'])

                # Show filter summary
                filter_summary = []
                for f in preset['filters']:
                    field_info = FILTER_FIELDS.get(f['field'], {})
                    label = field_info.get('label', f['field'])
                    filter_summary.append(f"{label} {f['operator']} {f['value']}")

                st.markdown("Filters: " + " | ".join(filter_summary))

                if st.button(f"Run {preset['name']}", key=f"run_preset_{key}"):
                    with st.spinner(f"Running {preset['name']} screener..."):
                        results = service.get_custom_screen_results(
                            filters=preset['filters'],
                            sort_by='rs_percentile',
                            limit=100
                        )
                        if results is not None and not results.empty:
                            st.success(f"Found {len(results)} matching stocks")
                            _display_screener_results(results)
                        else:
                            st.info("No stocks match the current criteria")

                st.markdown("---")


def _render_custom_builder(service: Any) -> None:
    """Render custom filter builder interface."""
    st.markdown("### Build Custom Screen")

    # Add filter button
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_field = st.selectbox(
            "Add Filter",
            list(FILTER_FIELDS.keys()),
            format_func=lambda x: FILTER_FIELDS[x]['label'],
            key="new_filter_field"
        )

    with col2:
        if st.button("Add Filter", key="add_filter_btn"):
            st.session_state.custom_filters.append({
                'field': selected_field,
                'operator': '>=',
                'value': None
            })

    # Display and edit active filters
    st.markdown("### Active Filters")

    if not st.session_state.custom_filters:
        st.info("No filters added yet. Add filters above to build your screen.")
    else:
        filters_to_remove = []

        for i, filter_item in enumerate(st.session_state.custom_filters):
            field = filter_item['field']
            field_info = FILTER_FIELDS.get(field, {})

            col1, col2, col3, col4 = st.columns([2, 1, 2, 0.5])

            with col1:
                st.markdown(f"**{field_info.get('label', field)}**")
                st.caption(field_info.get('description', ''))

            with col2:
                field_type = field_info.get('type', 'range')
                if field_type in ['select', 'multi_select']:
                    operator = st.selectbox(
                        "Operator",
                        ['=', 'in', '!='],
                        key=f"op_{i}",
                        label_visibility="collapsed"
                    )
                else:
                    operator = st.selectbox(
                        "Operator",
                        ['>=', '<=', '=', '>', '<'],
                        key=f"op_{i}",
                        label_visibility="collapsed"
                    )
                st.session_state.custom_filters[i]['operator'] = operator

            with col3:
                if field_type == 'range':
                    value = st.number_input(
                        "Value",
                        min_value=float(field_info.get('min', 0)),
                        max_value=float(field_info.get('max', 100)),
                        value=float(filter_item.get('value') or field_info.get('min', 0)),
                        key=f"val_{i}",
                        label_visibility="collapsed"
                    )
                elif field_type == 'select':
                    value = st.selectbox(
                        "Value",
                        field_info.get('options', []),
                        key=f"val_{i}",
                        label_visibility="collapsed"
                    )
                elif field_type == 'multi_select':
                    value = st.multiselect(
                        "Value",
                        field_info.get('options', []),
                        key=f"val_{i}",
                        label_visibility="collapsed"
                    )
                else:
                    value = st.text_input(
                        "Value",
                        value=str(filter_item.get('value', '')),
                        key=f"val_{i}",
                        label_visibility="collapsed"
                    )
                st.session_state.custom_filters[i]['value'] = value

            with col4:
                if st.button("X", key=f"remove_{i}"):
                    filters_to_remove.append(i)

        # Remove marked filters
        for idx in reversed(filters_to_remove):
            st.session_state.custom_filters.pop(idx)
            st.rerun()

    # Run screener
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        sort_by = st.selectbox(
            "Sort By",
            ['rs_percentile', 'current_price', 'relative_volume', 'atr_percent', 'rsi_14'],
            format_func=lambda x: FILTER_FIELDS.get(x, {}).get('label', x),
            key="sort_by"
        )

    with col2:
        limit = st.number_input("Max Results", min_value=10, max_value=500, value=100, key="result_limit")

    with col3:
        if st.button("Run Custom Screen", type="primary", key="run_custom_screen"):
            if st.session_state.custom_filters:
                with st.spinner("Running custom screener..."):
                    # Build filters list for service
                    filters = []
                    for f in st.session_state.custom_filters:
                        if f.get('value') is not None:
                            filters.append({
                                'field': f['field'],
                                'operator': f['operator'],
                                'value': f['value']
                            })

                    results = service.get_custom_screen_results(
                        filters=filters,
                        sort_by=sort_by,
                        limit=int(limit)
                    )

                    if results is not None and not results.empty:
                        st.success(f"Found {len(results)} matching stocks")
                        _display_screener_results(results)
                    else:
                        st.info("No stocks match the current criteria")
            else:
                st.warning("Add at least one filter to run the screener")

    # Clear filters button
    if st.session_state.custom_filters:
        if st.button("Clear All Filters", key="clear_filters"):
            st.session_state.custom_filters = []
            st.rerun()


def _display_screener_results(df: pd.DataFrame) -> None:
    """Display screener results in a formatted table."""
    st.markdown("### Results")

    # Format display columns
    display_df = df.copy()

    # Format common columns if they exist
    if 'current_price' in display_df.columns:
        display_df['Price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else '-')

    if 'rs_percentile' in display_df.columns:
        display_df['RS'] = display_df['rs_percentile'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else '-')

    if 'rsi_14' in display_df.columns:
        display_df['RSI'] = display_df['rsi_14'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else '-')

    if 'atr_percent' in display_df.columns:
        display_df['ATR%'] = display_df['atr_percent'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else '-')

    if 'relative_volume' in display_df.columns:
        display_df['RVol'] = display_df['relative_volume'].apply(lambda x: f"{x:.1f}x" if pd.notnull(x) else '-')

    if 'trend_strength' in display_df.columns:
        display_df['Trend'] = display_df['trend_strength']

    # Select display columns
    display_cols = ['ticker']
    for col in ['Price', 'RS', 'RSI', 'ATR%', 'RVol', 'Trend', 'sector', 'name']:
        if col in display_df.columns:
            display_cols.append(col)

    result_df = display_df[[c for c in display_cols if c in display_df.columns]]
    result_df = result_df.rename(columns={'ticker': 'Ticker', 'sector': 'Sector', 'name': 'Name'})

    st.dataframe(result_df, use_container_width=True, hide_index=True, height=400)

    # Export
    col1, col2 = st.columns(2)
    with col1:
        csv = result_df.to_csv(index=False)
        st.download_button(
            "Export CSV",
            csv,
            f"custom_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )

    with col2:
        if 'Ticker' in result_df.columns:
            tickers = ','.join(result_df['Ticker'].tolist())
            st.download_button(
                "Export Ticker List",
                tickers,
                f"tickers_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                "text/plain"
            )
