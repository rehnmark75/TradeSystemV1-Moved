"""
MAE (Maximum Adverse Excursion) Analysis Tab

Displays real-time tick-level MAE tracking for scalp trades to help
optimize Virtual Stop Loss (VSL) settings per currency pair.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from services.trading_analytics_service import TradingAnalyticsService


def render_mae_analysis_tab():
    """Render the MAE analysis tab for scalp trades"""
    service = TradingAnalyticsService()

    st.header("MAE Analysis (Scalp Trades)")
    st.markdown("""
    **Maximum Adverse Excursion (MAE)** tracks the worst drawdown during each trade before
    it becomes profitable. Use this data to optimize your Virtual Stop Loss (VSL) settings.
    """)

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        days_filter = st.selectbox(
            "Time Period",
            [1, 3, 7, 14, 30],
            index=2,
            key="mae_days_filter",
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
        )

    with col2:
        if st.button("Refresh", key="mae_refresh"):
            st.rerun()

    # Fetch data
    mae_df = service.fetch_scalp_mae_analysis(days_filter)
    summary_df = service.fetch_mae_summary_by_pair(days_filter)

    if mae_df.empty:
        st.warning(f"No scalp trade data found for the last {days_filter} days")
        return

    # Summary Statistics by Pair
    st.subheader("MAE Summary by Currency Pair")

    if not summary_df.empty:
        # Display summary table
        display_cols = ['symbol_short', 'total_trades', 'win_rate', 'avg_mae_pips',
                       'median_mae_pips', 'p75_mae_pips', 'p90_mae_pips', 'max_mae_pips',
                       'avg_mfe_pips', 'avg_vsl_setting']
        display_cols = [c for c in display_cols if c in summary_df.columns]

        st.dataframe(
            summary_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol_short": "Pair",
                "total_trades": "Trades",
                "win_rate": st.column_config.NumberColumn("Win %", format="%.1f%%"),
                "avg_mae_pips": st.column_config.NumberColumn("Avg MAE", format="%.1f"),
                "median_mae_pips": st.column_config.NumberColumn("Median MAE", format="%.1f"),
                "p75_mae_pips": st.column_config.NumberColumn("75th %ile", format="%.1f"),
                "p90_mae_pips": st.column_config.NumberColumn("90th %ile", format="%.1f"),
                "max_mae_pips": st.column_config.NumberColumn("Max MAE", format="%.1f"),
                "avg_mfe_pips": st.column_config.NumberColumn("Avg MFE", format="%.1f"),
                "avg_vsl_setting": st.column_config.NumberColumn("VSL Setting", format="%.1f"),
            }
        )

        # VSL Optimization Insights
        st.subheader("VSL Optimization Insights")

        insights_cols = st.columns(len(summary_df) if len(summary_df) <= 4 else 4)
        for idx, (_, row) in enumerate(summary_df.head(4).iterrows()):
            with insights_cols[idx]:
                pair = row['symbol_short']
                p90_mae = row.get('p90_mae_pips', 0) or 0
                current_vsl = row.get('avg_vsl_setting', 0) or 0

                # Recommendation: VSL should cover 90th percentile MAE
                if current_vsl > 0 and p90_mae > 0:
                    if p90_mae > current_vsl:
                        status = "too_tight"
                        color = "red"
                        recommendation = f"Consider {p90_mae:.1f}+ pips"
                    elif p90_mae < current_vsl * 0.7:
                        status = "can_tighten"
                        color = "green"
                        recommendation = f"Could reduce to {p90_mae:.1f}"
                    else:
                        status = "optimal"
                        color = "blue"
                        recommendation = "VSL looks good"
                else:
                    color = "gray"
                    recommendation = "Insufficient data"

                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; border: 1px solid {color};">
                    <h4 style="margin: 0;">{pair}</h4>
                    <p style="margin: 5px 0;">90th %ile MAE: <b>{p90_mae:.1f}</b> pips</p>
                    <p style="margin: 5px 0;">Current VSL: <b>{current_vsl:.1f}</b> pips</p>
                    <p style="margin: 5px 0; color: {color};">{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

        # MAE Distribution Chart
        st.subheader("MAE Distribution")

        if 'mae_pips' in mae_df.columns and mae_df['mae_pips'].notna().any():
            fig = px.histogram(
                mae_df[mae_df['mae_pips'] > 0],
                x='mae_pips',
                color='symbol_short',
                nbins=20,
                title='MAE Distribution by Pair',
                labels={'mae_pips': 'MAE (pips)', 'symbol_short': 'Pair'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # MAE vs MFE Scatter
        st.subheader("MAE vs MFE (Risk vs Reward)")

        if 'mfe_pips' in mae_df.columns and 'mae_pips' in mae_df.columns:
            scatter_df = mae_df[(mae_df['mae_pips'] > 0) | (mae_df['mfe_pips'] > 0)].copy()
            if not scatter_df.empty:
                fig = px.scatter(
                    scatter_df,
                    x='mae_pips',
                    y='mfe_pips',
                    color='result',
                    symbol='symbol_short',
                    hover_data=['id', 'direction', 'entry_price'],
                    title='MAE vs MFE by Trade Outcome',
                    labels={'mae_pips': 'MAE (pips)', 'mfe_pips': 'MFE (pips)'}
                )
                # Add diagonal line (1:1 risk/reward)
                max_val = max(scatter_df['mae_pips'].max(), scatter_df['mfe_pips'].max(), 10)
                fig.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='1:1 R:R'
                ))
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

    # Individual Trades Table
    st.subheader("Individual Scalp Trades")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        pair_filter = st.multiselect(
            "Filter by Pair",
            options=mae_df['symbol_short'].unique(),
            default=mae_df['symbol_short'].unique(),
            key="mae_pair_filter"
        )
    with col2:
        result_filter = st.multiselect(
            "Filter by Result",
            options=['WIN', 'LOSS', 'OPEN', 'PENDING'],
            default=['WIN', 'LOSS', 'OPEN'],
            key="mae_result_filter"
        )

    # Apply filters
    filtered_df = mae_df[
        (mae_df['symbol_short'].isin(pair_filter)) &
        (mae_df['result'].isin(result_filter))
    ]

    if not filtered_df.empty:
        display_cols = ['id', 'timestamp', 'symbol_short', 'direction', 'result',
                       'mae_pips', 'mfe_pips', 'mae_pct_of_vsl', 'virtual_sl_pips',
                       'vsl_stage', 'profit_loss']
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": "Trade ID",
                "timestamp": st.column_config.DatetimeColumn("Time", format="DD/MM HH:mm"),
                "symbol_short": "Pair",
                "direction": "Dir",
                "result": "Result",
                "mae_pips": st.column_config.NumberColumn("MAE", format="%.1f"),
                "mfe_pips": st.column_config.NumberColumn("MFE", format="%.1f"),
                "mae_pct_of_vsl": st.column_config.NumberColumn("MAE % VSL", format="%.0f%%"),
                "virtual_sl_pips": st.column_config.NumberColumn("VSL", format="%.1f"),
                "vsl_stage": "Stage",
                "profit_loss": st.column_config.NumberColumn("P&L", format="%.2f"),
            }
        )
    else:
        st.info("No trades match the selected filters")
