"""
Entry Timing Analysis Tab

Analyzes entry timing quality to identify if trades are entering at poor levels.
Key metrics:
- Entry type performance (PULLBACK, MOMENTUM, MICRO_PULLBACK)
- Signal trigger performance (SWING_PULLBACK, SWING_PULLBACK+PIN, etc.)
- Zero MFE trades (price went straight against entry)
- Slippage analysis (signal price vs fill price)
- Time to MAE (how quickly price moved against entry)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from services.trading_analytics_service import TradingAnalyticsService


def render_entry_timing_analysis_tab():
    """Render the Entry Timing Analysis tab"""
    service = TradingAnalyticsService()

    st.header("Entry Timing Analysis")
    st.markdown("""
    Analyze entry quality to identify timing issues. **Zero MFE** trades (price moved immediately
    against entry with no favorable movement) indicate poor entry timing - entering at local
    highs/lows instead of optimal pullback levels.
    """)

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        days_filter = st.selectbox(
            "Time Period",
            [1, 3, 7, 14, 30],
            index=2,
            key="entry_timing_days_filter",
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
        )

    with col2:
        if st.button("Refresh", key="entry_timing_refresh"):
            st.rerun()

    # Fetch data
    timing_df = service.fetch_entry_timing_analysis(days_filter)
    summary_df = service.fetch_entry_timing_summary(days_filter)
    trigger_df = service.fetch_entry_timing_by_trigger(days_filter)

    if timing_df.empty:
        st.warning(f"No trade data found for the last {days_filter} days")
        return

    # =====================================================================
    # SECTION 1: Entry Type Performance Summary
    # =====================================================================
    st.subheader("Performance by Entry Type")

    if not summary_df.empty:
        # Display summary with color coding
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "entry_type": "Entry Type",
                "total_trades": "Trades",
                "wins": "Wins",
                "losses": "Losses",
                "win_rate": st.column_config.NumberColumn("Win %", format="%.1f%%"),
                "avg_pnl": st.column_config.NumberColumn("Avg P&L", format="%.2f"),
                "total_pnl": st.column_config.NumberColumn("Total P&L", format="%.2f"),
                "avg_mae_pips": st.column_config.NumberColumn("Avg MAE", format="%.1f"),
                "avg_mfe_pips": st.column_config.NumberColumn("Avg MFE", format="%.1f"),
                "zero_mfe_pct": st.column_config.NumberColumn("Zero MFE %", format="%.1f%%"),
                "avg_confidence": st.column_config.NumberColumn("Avg Conf", format="%.2f"),
                "avg_pullback_depth": st.column_config.NumberColumn("Avg Pullback", format="%.2f"),
            }
        )

        # Insights based on entry type
        st.markdown("#### Entry Type Insights")

        for _, row in summary_df.iterrows():
            entry_type = row['entry_type']
            win_rate = row['win_rate']
            zero_mfe_pct = row['zero_mfe_pct']
            total_trades = row['total_trades']

            if total_trades < 5:
                continue  # Skip small samples

            # Color code based on performance
            if win_rate < 30:
                emoji = "🔴"
                assessment = "Poor"
            elif win_rate < 50:
                emoji = "🟡"
                assessment = "Below Average"
            else:
                emoji = "🟢"
                assessment = "Good"

            # Zero MFE warning
            mfe_warning = ""
            if zero_mfe_pct > 70:
                mfe_warning = f" ⚠️ **{zero_mfe_pct:.0f}% of trades had zero favorable movement** - entries at local extremes"
            elif zero_mfe_pct > 50:
                mfe_warning = f" ⚠️ {zero_mfe_pct:.0f}% zero MFE trades"

            st.markdown(f"{emoji} **{entry_type}**: {win_rate:.1f}% win rate ({total_trades} trades){mfe_warning}")

    # =====================================================================
    # SECTION 2: Performance by Signal Trigger
    # =====================================================================
    st.subheader("Performance by Signal Trigger")

    if not trigger_df.empty:
        st.dataframe(
            trigger_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "signal_trigger": "Trigger",
                "entry_type": "Entry Type",
                "total_trades": "Trades",
                "win_rate": st.column_config.NumberColumn("Win %", format="%.1f%%"),
                "avg_pnl": st.column_config.NumberColumn("Avg P&L", format="%.2f"),
                "total_pnl": st.column_config.NumberColumn("Total P&L", format="%.2f"),
                "avg_mae_pips": st.column_config.NumberColumn("Avg MAE", format="%.1f"),
                "avg_mfe_pips": st.column_config.NumberColumn("Avg MFE", format="%.1f"),
                "zero_mfe_pct": st.column_config.NumberColumn("Zero MFE %", format="%.1f%%"),
            }
        )

    # =====================================================================
    # SECTION 3: Zero MFE Analysis (Bad Timing Indicator)
    # =====================================================================
    st.subheader("Zero MFE Analysis (Entry Timing Quality)")

    # Calculate overall zero MFE stats
    closed_trades = timing_df[timing_df['result'].isin(['WIN', 'LOSS'])]
    if not closed_trades.empty:
        total_closed = len(closed_trades)
        zero_mfe_trades = closed_trades[closed_trades['zero_mfe'] == True]
        zero_mfe_count = len(zero_mfe_trades)
        zero_mfe_pct = (zero_mfe_count / total_closed * 100) if total_closed > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Closed Trades", total_closed)

        with col2:
            st.metric("Zero MFE Trades", zero_mfe_count)

        with col3:
            delta_color = "inverse" if zero_mfe_pct > 50 else "normal"
            st.metric("Zero MFE %", f"{zero_mfe_pct:.1f}%",
                     delta=f"{'Bad' if zero_mfe_pct > 50 else 'OK'}")

        with col4:
            # Win rate among zero MFE trades
            zero_mfe_wins = len(zero_mfe_trades[zero_mfe_trades['result'] == 'WIN'])
            zero_mfe_win_rate = (zero_mfe_wins / zero_mfe_count * 100) if zero_mfe_count > 0 else 0
            st.metric("Zero MFE Win Rate", f"{zero_mfe_win_rate:.1f}%")

        # Zero MFE breakdown by result
        if zero_mfe_count > 0:
            zero_mfe_losses = len(zero_mfe_trades[zero_mfe_trades['result'] == 'LOSS'])

            if zero_mfe_pct > 50:
                st.error(f"""
                **Entry Timing Problem Detected**: {zero_mfe_pct:.0f}% of trades had no favorable
                movement before moving against entry. This suggests entries are occurring at local
                price extremes (peaks for buys, troughs for sells) rather than at optimal pullback levels.

                - Zero MFE Wins: {zero_mfe_wins}
                - Zero MFE Losses: {zero_mfe_losses}

                **Recommendation**: Switch to LIMIT orders (better price entry) or wait for deeper pullbacks.
                """)
            elif zero_mfe_pct > 30:
                st.warning(f"""
                **Moderate Timing Issues**: {zero_mfe_pct:.0f}% of trades had zero MFE.
                Consider using LIMIT orders to get better entry prices.
                """)

    # =====================================================================
    # SECTION 4: Slippage Analysis
    # =====================================================================
    st.subheader("Slippage Analysis (Signal vs Entry Price)")

    slippage_data = timing_df[timing_df['slippage_pips'].notna()]
    if not slippage_data.empty:
        avg_slippage = slippage_data['slippage_pips'].mean()
        median_slippage = slippage_data['slippage_pips'].median()
        max_slippage = slippage_data['slippage_pips'].max()

        col1, col2, col3 = st.columns(3)

        with col1:
            delta_color = "inverse" if avg_slippage > 2 else "normal"
            st.metric("Avg Slippage", f"{avg_slippage:.1f} pips",
                     delta=f"{'High' if avg_slippage > 3 else 'Low' if avg_slippage < 1 else 'Normal'}")

        with col2:
            st.metric("Median Slippage", f"{median_slippage:.1f} pips")

        with col3:
            st.metric("Max Slippage", f"{max_slippage:.1f} pips")

        # Slippage by entry type
        slippage_by_type = slippage_data.groupby('entry_type')['slippage_pips'].agg(['mean', 'median', 'count']).reset_index()
        slippage_by_type.columns = ['Entry Type', 'Avg Slippage', 'Median Slippage', 'Count']

        if not slippage_by_type.empty:
            st.markdown("**Slippage by Entry Type:**")
            st.dataframe(
                slippage_by_type,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg Slippage": st.column_config.NumberColumn(format="%.2f"),
                    "Median Slippage": st.column_config.NumberColumn(format="%.2f"),
                }
            )

        # Slippage histogram
        fig = px.histogram(
            slippage_data,
            x='slippage_pips',
            color='result',
            nbins=30,
            title="Slippage Distribution by Trade Result",
            labels={'slippage_pips': 'Slippage (pips)', 'result': 'Result'},
            color_discrete_map={'WIN': 'green', 'LOSS': 'red', 'OPEN': 'blue'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================================
    # SECTION 5: MFE vs MAE Scatter by Entry Type
    # =====================================================================
    st.subheader("MFE vs MAE by Entry Type")

    plot_data = timing_df[
        (timing_df['mfe_pips'].notna()) &
        (timing_df['mae_pips'].notna()) &
        (timing_df['entry_type'].notna())
    ]

    if not plot_data.empty:
        fig = px.scatter(
            plot_data,
            x='mae_pips',
            y='mfe_pips',
            color='entry_type',
            symbol='result',
            hover_data=['symbol_short', 'direction', 'profit_loss', 'signal_trigger'],
            title="MFE vs MAE by Entry Type (Good entries: high MFE, low MAE)",
            labels={
                'mae_pips': 'MAE (pips) - Worst Drawdown',
                'mfe_pips': 'MFE (pips) - Best Profit',
                'entry_type': 'Entry Type',
                'result': 'Result'
            }
        )
        # Add reference line (MFE = MAE)
        max_val = max(plot_data['mae_pips'].max(), plot_data['mfe_pips'].max(), 5)
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='MFE = MAE',
            line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Reading this chart:**
        - Points **above** the diagonal line had more favorable than adverse movement (better timing)
        - Points **below** the diagonal had more adverse movement (worse timing)
        - Points clustered at **low MFE, high MAE** indicate poor entry timing
        """)

    # =====================================================================
    # SECTION 6: Individual Trade Details
    # =====================================================================
    st.subheader("Individual Trade Details")

    # Filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        result_filter = st.multiselect(
            "Result",
            options=['WIN', 'LOSS', 'OPEN'],
            default=['WIN', 'LOSS'],
            key="entry_timing_result_filter"
        )

    with col2:
        entry_type_options = timing_df['entry_type'].dropna().unique().tolist()
        entry_type_filter = st.multiselect(
            "Entry Type",
            options=entry_type_options,
            default=entry_type_options,
            key="entry_timing_type_filter"
        )

    with col3:
        zero_mfe_only = st.checkbox("Show Zero MFE Only", key="entry_timing_zero_mfe")

    # Apply filters
    filtered_df = timing_df.copy()
    if result_filter:
        filtered_df = filtered_df[filtered_df['result'].isin(result_filter)]
    if entry_type_filter:
        filtered_df = filtered_df[filtered_df['entry_type'].isin(entry_type_filter)]
    if zero_mfe_only:
        filtered_df = filtered_df[filtered_df['zero_mfe'] == True]

    # Display columns - include pattern and divergence info
    display_cols = [
        'id', 'symbol_short', 'direction', 'entry_type', 'signal_trigger',
        'pattern_type', 'rsi_divergence_detected',
        'result', 'profit_loss', 'mfe_pips', 'mae_pips', 'slippage_pips',
        'zero_mfe', 'confidence_score', 'trade_timestamp'
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": "Trade ID",
            "symbol_short": "Pair",
            "direction": "Dir",
            "entry_type": "Entry Type",
            "signal_trigger": "Trigger",
            "pattern_type": "Pattern",
            "rsi_divergence_detected": st.column_config.CheckboxColumn("RSI Div"),
            "result": "Result",
            "profit_loss": st.column_config.NumberColumn("P&L", format="%.2f"),
            "mfe_pips": st.column_config.NumberColumn("MFE", format="%.1f"),
            "mae_pips": st.column_config.NumberColumn("MAE", format="%.1f"),
            "slippage_pips": st.column_config.NumberColumn("Slippage", format="%.1f"),
            "zero_mfe": st.column_config.CheckboxColumn("Zero MFE"),
            "confidence_score": st.column_config.NumberColumn("Conf", format="%.2f"),
            "trade_timestamp": st.column_config.DatetimeColumn("Time", format="MM/DD HH:mm"),
        }
    )

    st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df)} trades")

    # =====================================================================
    # SECTION 7: Recommendations
    # =====================================================================
    st.subheader("Recommendations")

    # Generate recommendations based on data
    recommendations = []

    # Check zero MFE percentage
    if not closed_trades.empty:
        zero_mfe_pct = (len(closed_trades[closed_trades['zero_mfe'] == True]) / len(closed_trades) * 100)
        if zero_mfe_pct > 70:
            recommendations.append("🔴 **Critical**: Over 70% of trades have zero MFE. Entry timing is severely problematic. Consider switching to LIMIT orders for better entry prices.")
        elif zero_mfe_pct > 50:
            recommendations.append("🟡 **Warning**: Over 50% of trades have zero MFE. Entry timing needs improvement. Use LIMIT orders or wait for deeper pullbacks.")

    # Check entry type performance
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            if row['total_trades'] >= 5:
                if row['entry_type'] == 'PULLBACK' and row['win_rate'] < 30:
                    recommendations.append(f"🔴 **PULLBACK entries** have {row['win_rate']:.0f}% win rate. Consider disabling or requiring deeper pullbacks.")
                if row['entry_type'] == 'MICRO_PULLBACK' and row['win_rate'] < 30:
                    recommendations.append(f"🔴 **MICRO_PULLBACK entries** have {row['win_rate']:.0f}% win rate. These entries may be too aggressive.")
                if row['entry_type'] == 'MOMENTUM' and row['win_rate'] > 40:
                    recommendations.append(f"🟢 **MOMENTUM entries** performing well at {row['win_rate']:.0f}% win rate. Consider focusing on this entry type.")

    # Check slippage
    if not slippage_data.empty and avg_slippage > 3:
        recommendations.append(f"🟡 **High slippage**: Average {avg_slippage:.1f} pips. LIMIT orders would reduce this significantly.")

    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.info("No specific recommendations at this time. Continue monitoring entry quality.")
