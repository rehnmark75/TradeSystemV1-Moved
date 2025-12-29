"""
Breakeven Optimizer Tab Component

Renders the Breakeven & Stop-Loss Optimizer tab with:
- MFE/MAE pattern analysis
- Breakeven trigger recommendations
- Stop-loss level recommendations
- Per-epic detailed analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib


def render_breakeven_optimizer_tab():
    """Render the Breakeven Optimizer analysis tab."""
    st.header("Breakeven & Stop-Loss Optimizer")
    st.markdown("""
    *Analyze historical trade MFE/MAE patterns to find optimal breakeven triggers and stop-loss levels.*

    This tool examines your recent trades per epic/direction to recommend:
    - **Breakeven Triggers**: When to move stop-loss to breakeven
    - **Stop-Loss Levels**: Optimal initial stop-loss distance based on MAE analysis
    """)

    # Import service
    try:
        from services.breakeven_analysis_service import BreakevenAnalysisService, format_epic_display
        service = BreakevenAnalysisService()
    except ImportError as e:
        st.error(f"Failed to load Breakeven Analysis Service: {e}")
        return

    # Configuration section
    st.subheader("Analysis Configuration")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        trades_per_group = st.slider(
            "Trades per Epic/Dir",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of recent trades to analyze per epic/direction combination"
        )

    with col2:
        # Get available epics
        available_epics = service.get_available_epics()
        epic_filter = st.multiselect(
            "Filter Epics (optional)",
            options=available_epics,
            default=[],
            help="Leave empty to analyze all epics with trades"
        )

    with col3:
        use_cache = st.checkbox("Use cached results", value=True,
                               help="Load from cache for faster display (updated hourly)")

    with col4:
        analyze_btn = st.button("Run Fresh Analysis", type="primary")

    st.markdown("---")

    # Main display area
    if analyze_btn:
        # Run fresh analysis
        with st.spinner("Analyzing trade patterns... This may take a minute for large datasets."):
            analyses = service.run_full_analysis(
                trades_per_group=trades_per_group,
                epic_filter=epic_filter if epic_filter else None,
                save_to_cache=True
            )

        if not analyses:
            st.warning("No trades found for analysis. Ensure you have closed trades with entry/exit data.")
            return

        _display_breakeven_results(service, analyses)

    elif use_cache:
        # Load from cache
        cached_df = service.get_cached_analysis(max_age_hours=24)

        if cached_df.empty:
            st.info("No cached analysis found. Click 'Run Fresh Analysis' to generate recommendations.")
            return

        # Display cache info
        if 'analyzed_at' in cached_df.columns:
            latest_analysis = cached_df['analyzed_at'].max()
            st.caption(f"Last analyzed: {latest_analysis}")

        _display_breakeven_results_from_cache(cached_df)
    else:
        st.info("Click 'Run Fresh Analysis' to analyze your trade data and get breakeven recommendations.")


def _display_breakeven_results(service, analyses):
    """Display breakeven analysis results from fresh analysis."""
    from services.breakeven_analysis_service import format_epic_display

    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Epic/Direction Pairs", len(analyses))
    with col2:
        total_trades = sum(a.trade_count for a in analyses)
        st.metric("Total Trades Analyzed", total_trades)
    with col3:
        high_priority_be = sum(1 for a in analyses if a.priority == 'high')
        st.metric("High Priority BE", high_priority_be,
                 delta=None if high_priority_be == 0 else f"{high_priority_be} need attention",
                 delta_color="inverse")
    with col4:
        high_priority_sl = sum(1 for a in analyses if a.sl_priority == 'high')
        st.metric("High Priority SL", high_priority_sl,
                 delta=None if high_priority_sl == 0 else f"{high_priority_sl} need attention",
                 delta_color="inverse")
    with col5:
        avg_win_rate = np.mean([a.win_rate for a in analyses])
        st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")

    # Main results table - split into BE and SL sections
    st.subheader("Breakeven Recommendations")
    df = service.get_summary_dataframe(analyses)

    # BE columns subset
    be_cols = ['Epic', 'Dir', 'Trades', 'Win%', 'Avg MFE', 'Med MFE', 'Avg MAE',
               'Optimal BE', 'Current BE', 'BE Diff', 'BE Action', 'BE Priority', 'Confidence']
    be_df = df[be_cols].copy()

    def highlight_be_priority(row):
        if row['BE Priority'] == 'HIGH':
            return ['background-color: rgba(255, 99, 71, 0.3)'] * len(row)
        elif row['BE Priority'] == 'MEDIUM':
            return ['background-color: rgba(255, 193, 7, 0.3)'] * len(row)
        return [''] * len(row)

    styled_be_df = be_df.style.apply(highlight_be_priority, axis=1)
    st.dataframe(styled_be_df, use_container_width=True, height=300)

    # Stop-Loss recommendations table
    st.subheader("Stop-Loss Recommendations")
    sl_cols = ['Epic', 'Dir', 'Trades', 'Win%', 'Avg MAE', 'P95 MAE',
               'Optimal SL', 'Config SL', 'Actual SL', 'SL Diff', 'SL Action', 'SL Priority']
    sl_df = df[sl_cols].copy()

    def highlight_sl_priority(row):
        if row['SL Priority'] == 'HIGH':
            return ['background-color: rgba(255, 99, 71, 0.3)'] * len(row)
        elif row['SL Priority'] == 'MEDIUM':
            return ['background-color: rgba(255, 193, 7, 0.3)'] * len(row)
        return [''] * len(row)

    styled_sl_df = sl_df.style.apply(highlight_sl_priority, axis=1)
    st.dataframe(styled_sl_df, use_container_width=True, height=300)

    # Detailed view expanders
    st.subheader("Detailed Analysis")

    for analysis in analyses:
        epic_display = format_epic_display(analysis.epic)
        # Combined priority indicator - show highest priority
        max_priority = 'high' if analysis.priority == 'high' or analysis.sl_priority == 'high' else \
                      'medium' if analysis.priority == 'medium' or analysis.sl_priority == 'medium' else 'low'
        priority_icon = "!" if max_priority == 'high' else "~" if max_priority == 'medium' else "+"

        with st.expander(f"{priority_icon} {epic_display} - {analysis.direction} ({analysis.trade_count} trades)"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**MFE Distribution (pips)**")
                st.markdown(f"- Average: **{analysis.avg_mfe:.1f}**")
                st.markdown(f"- Median: **{analysis.median_mfe:.1f}**")
                st.markdown(f"- 25th Percentile: **{analysis.percentile_25_mfe:.1f}**")
                st.markdown(f"- 75th Percentile: **{analysis.percentile_75_mfe:.1f}**")

                st.markdown("**MAE Distribution (pips)**")
                st.markdown(f"- Median: **{analysis.median_mae:.1f}**")
                st.markdown(f"- 75th Percentile: **{analysis.percentile_75_mae:.1f}**")
                st.markdown(f"- 95th Percentile: **{analysis.percentile_95_mae:.1f}**")
                st.markdown(f"- Maximum: **{analysis.max_mae:.1f}**")

            with col2:
                st.markdown("**Breakeven Analysis**")
                st.markdown(f"- Current Trigger: **{analysis.current_be_trigger:.0f}** pips")
                st.markdown(f"- Optimal Trigger: **{analysis.optimal_be_trigger:.1f}** pips")
                st.markdown(f"- Conservative: **{analysis.conservative_be_trigger:.1f}** pips")

                be_diff = analysis.optimal_be_trigger - analysis.current_be_trigger
                if abs(be_diff) > 1:
                    direction = "v Lower by" if be_diff < 0 else "^ Raise by"
                    st.markdown(f"- Suggested Change: **{direction} {abs(be_diff):.0f} pips**")

                st.markdown(f"- Action: **{analysis.recommendation}**")
                st.markdown(f"- Priority: **{analysis.priority.upper()}**")

            with col3:
                st.markdown("**Stop-Loss Analysis**")
                st.markdown(f"- Current SL: **{analysis.current_stop_loss:.0f}** pips")
                st.markdown(f"- Optimal SL: **{analysis.optimal_stop_loss:.1f}** pips")

                sl_diff = analysis.optimal_stop_loss - analysis.current_stop_loss
                if abs(sl_diff) > 1:
                    direction = "v Tighten by" if sl_diff < 0 else "^ Widen by"
                    st.markdown(f"- Suggested Change: **{direction} {abs(sl_diff):.0f} pips**")

                st.markdown(f"- Action: **{analysis.sl_recommendation}**")
                st.markdown(f"- Priority: **{analysis.sl_priority.upper()}**")

                st.markdown("---")
                st.markdown(f"- Win Rate: **{analysis.win_rate:.0f}%**")
                st.markdown(f"- Confidence: **{analysis.confidence.upper()}**")

            # Efficiency metrics
            st.markdown("**Efficiency Metrics**")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("BE Reach Rate", f"{analysis.be_reach_rate:.0f}%",
                         help="Percentage of trades that reached the BE trigger level")
            with metric_col2:
                st.metric("BE Protection Rate", f"{analysis.be_protection_rate:.0f}%",
                         help="Percentage of BE trades that exited at exactly breakeven")
            with metric_col3:
                st.metric("BE Profit Rate", f"{analysis.be_profit_rate:.0f}%",
                         help="Percentage of BE trades that continued to profit")

            # Analysis notes
            if analysis.analysis_notes:
                st.info(f"{analysis.analysis_notes}")

    # Export option
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            "Export CSV",
            csv,
            "breakeven_analysis.csv",
            "text/csv"
        )


def _display_breakeven_results_from_cache(df):
    """Display breakeven analysis results from cached DataFrame."""
    # Re-read current BE config to get latest values (not cached values)
    try:
        import trailing_config
        importlib.reload(trailing_config)
        current_config = trailing_config.PAIR_TRAILING_CONFIGS
    except ImportError:
        current_config = {}

    # Update current_be_trigger from live config instead of cached values
    def get_current_be_from_config(epic):
        """Get current BE trigger from live config."""
        config = current_config.get(epic, {})
        if config:
            return config.get('break_even_trigger_points', 25)
        # Try partial match
        for key, cfg in current_config.items():
            if epic in key or key in epic:
                return cfg.get('break_even_trigger_points', 25)
        return 25  # Default

    df = df.copy()
    df['current_be_trigger'] = df['epic'].apply(get_current_be_from_config)

    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Epic/Direction Pairs", len(df))
    with col2:
        total_trades = df['trade_count'].sum()
        st.metric("Total Trades Analyzed", int(total_trades))
    with col3:
        high_priority_be = (df['priority'] == 'high').sum()
        st.metric("High Priority BE", int(high_priority_be),
                 delta=None if high_priority_be == 0 else f"{high_priority_be} need attention",
                 delta_color="inverse")
    with col4:
        # Check if sl_priority column exists (for backwards compatibility)
        if 'sl_priority' in df.columns:
            high_priority_sl = (df['sl_priority'] == 'high').sum()
        else:
            high_priority_sl = 0
        st.metric("High Priority SL", int(high_priority_sl),
                 delta=None if high_priority_sl == 0 else f"{high_priority_sl} need attention",
                 delta_color="inverse")
    with col5:
        avg_win_rate = df['win_rate'].mean()
        st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")

    # Check if SL columns exist (for backwards compatibility)
    has_sl_data = 'optimal_stop_loss' in df.columns and df['optimal_stop_loss'].notna().any()

    # Breakeven Recommendations Table
    st.subheader("Breakeven Recommendations")

    # Create BE display dataframe
    be_display_df = df[['epic', 'direction', 'trade_count', 'win_rate',
                    'avg_mfe', 'median_mfe', 'avg_mae',
                    'optimal_be_trigger', 'current_be_trigger',
                    'recommendation', 'priority', 'confidence']].copy()

    # Clean epic names
    be_display_df['epic'] = be_display_df['epic'].apply(
        lambda x: x.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
    )

    # Calculate diff
    be_display_df['diff'] = be_display_df['optimal_be_trigger'] - be_display_df['current_be_trigger']
    be_display_df['diff'] = be_display_df['diff'].apply(lambda x: f"+{x:.0f}" if x > 0 else f"{x:.0f}")

    # Format columns
    be_display_df['win_rate'] = be_display_df['win_rate'].apply(lambda x: f"{x:.0f}%")
    be_display_df['avg_mfe'] = be_display_df['avg_mfe'].apply(lambda x: f"{x:.0f}")
    be_display_df['median_mfe'] = be_display_df['median_mfe'].apply(lambda x: f"{x:.0f}")
    be_display_df['avg_mae'] = be_display_df['avg_mae'].apply(lambda x: f"{x:.0f}")
    be_display_df['optimal_be_trigger'] = be_display_df['optimal_be_trigger'].apply(lambda x: f"{x:.0f}")
    be_display_df['current_be_trigger'] = be_display_df['current_be_trigger'].apply(lambda x: f"{x:.0f}")
    be_display_df['priority'] = be_display_df['priority'].str.upper()
    be_display_df['confidence'] = be_display_df['confidence'].str.upper()

    # Rename columns for display
    be_display_df.columns = ['Epic', 'Dir', 'Trades', 'Win%', 'Avg MFE', 'Med MFE', 'Avg MAE',
                         'Optimal BE', 'Current BE', 'Action', 'Priority', 'Confidence', 'Diff']

    # Reorder columns
    be_display_df = be_display_df[['Epic', 'Dir', 'Trades', 'Win%', 'Avg MFE', 'Med MFE', 'Avg MAE',
                            'Optimal BE', 'Current BE', 'Diff', 'Action', 'Priority', 'Confidence']]

    # Style the dataframe
    def highlight_be_priority(row):
        if row['Priority'] == 'HIGH':
            return ['background-color: rgba(255, 99, 71, 0.3)'] * len(row)
        elif row['Priority'] == 'MEDIUM':
            return ['background-color: rgba(255, 193, 7, 0.3)'] * len(row)
        return [''] * len(row)

    styled_be_df = be_display_df.style.apply(highlight_be_priority, axis=1)
    st.dataframe(styled_be_df, use_container_width=True, height=300)

    # Stop-Loss Recommendations Table (if data exists)
    if has_sl_data:
        st.subheader("Stop-Loss Recommendations")

        sl_display_df = df[['epic', 'direction', 'trade_count', 'win_rate', 'avg_mae']].copy()

        # Add SL columns if they exist
        if 'percentile_95_mae' in df.columns:
            sl_display_df['p95_mae'] = df['percentile_95_mae']
        else:
            sl_display_df['p95_mae'] = 0

        sl_display_df['optimal_sl'] = df['optimal_stop_loss']
        sl_display_df['current_sl'] = df['current_stop_loss']
        sl_display_df['sl_action'] = df['sl_recommendation'] if 'sl_recommendation' in df.columns else 'N/A'
        sl_display_df['sl_priority'] = df['sl_priority'] if 'sl_priority' in df.columns else 'low'

        # Clean epic names
        sl_display_df['epic'] = sl_display_df['epic'].apply(
            lambda x: x.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
        )

        # Calculate SL diff
        sl_display_df['sl_diff'] = sl_display_df['optimal_sl'] - sl_display_df['current_sl']
        sl_display_df['sl_diff'] = sl_display_df['sl_diff'].apply(lambda x: f"+{x:.0f}" if x > 0 else f"{x:.0f}")

        # Format columns
        sl_display_df['win_rate'] = sl_display_df['win_rate'].apply(lambda x: f"{x:.0f}%")
        sl_display_df['avg_mae'] = sl_display_df['avg_mae'].apply(lambda x: f"{x:.0f}")
        sl_display_df['p95_mae'] = sl_display_df['p95_mae'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        sl_display_df['optimal_sl'] = sl_display_df['optimal_sl'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        sl_display_df['current_sl'] = sl_display_df['current_sl'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        sl_display_df['sl_priority'] = sl_display_df['sl_priority'].str.upper()

        # Rename columns
        sl_display_df.columns = ['Epic', 'Dir', 'Trades', 'Win%', 'Avg MAE', 'P95 MAE',
                                 'Optimal SL', 'Current SL', 'Action', 'Priority', 'Diff']

        # Reorder
        sl_display_df = sl_display_df[['Epic', 'Dir', 'Trades', 'Win%', 'Avg MAE', 'P95 MAE',
                                       'Optimal SL', 'Current SL', 'Diff', 'Action', 'Priority']]

        def highlight_sl_priority(row):
            if row['Priority'] == 'HIGH':
                return ['background-color: rgba(255, 99, 71, 0.3)'] * len(row)
            elif row['Priority'] == 'MEDIUM':
                return ['background-color: rgba(255, 193, 7, 0.3)'] * len(row)
            return [''] * len(row)

        styled_sl_df = sl_display_df.style.apply(highlight_sl_priority, axis=1)
        st.dataframe(styled_sl_df, use_container_width=True, height=300)

    # Detailed expanders
    st.subheader("Detailed Analysis")

    for _, row in df.iterrows():
        epic_display = row['epic'].replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
        be_priority = row['priority']
        sl_priority = row.get('sl_priority', 'low') or 'low'

        # Combined priority indicator
        max_priority = 'high' if be_priority == 'high' or sl_priority == 'high' else \
                      'medium' if be_priority == 'medium' or sl_priority == 'medium' else 'low'
        priority_icon = "!" if max_priority == 'high' else "~" if max_priority == 'medium' else "+"

        with st.expander(f"{priority_icon} {epic_display} - {row['direction']} ({int(row['trade_count'])} trades)"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**MFE Distribution (pips)**")
                st.markdown(f"- Average: **{row['avg_mfe']:.1f}**")
                st.markdown(f"- Median: **{row['median_mfe']:.1f}**")
                st.markdown(f"- 25th Percentile: **{row['percentile_25_mfe']:.1f}**")
                st.markdown(f"- 75th Percentile: **{row['percentile_75_mfe']:.1f}**")

                st.markdown("**MAE Distribution (pips)**")
                st.markdown(f"- Median: **{row['median_mae']:.1f}**")
                st.markdown(f"- 75th Percentile: **{row['percentile_75_mae']:.1f}**")
                if 'percentile_95_mae' in row and pd.notna(row.get('percentile_95_mae')):
                    st.markdown(f"- 95th Percentile: **{row['percentile_95_mae']:.1f}**")
                if 'max_mae' in row and pd.notna(row.get('max_mae')):
                    st.markdown(f"- Maximum: **{row['max_mae']:.1f}**")

            with col2:
                st.markdown("**Breakeven Analysis**")
                st.markdown(f"- Current Trigger: **{row['current_be_trigger']:.0f}** pips")
                st.markdown(f"- Optimal Trigger: **{row['optimal_be_trigger']:.1f}** pips")
                st.markdown(f"- Conservative: **{row['conservative_be_trigger']:.1f}** pips")

                be_diff = row['optimal_be_trigger'] - row['current_be_trigger']
                if abs(be_diff) > 1:
                    direction = "v Lower by" if be_diff < 0 else "^ Raise by"
                    st.markdown(f"- Suggested Change: **{direction} {abs(be_diff):.0f} pips**")

                st.markdown(f"- Action: **{row['recommendation']}**")
                st.markdown(f"- Priority: **{row['priority'].upper()}**")

            with col3:
                st.markdown("**Stop-Loss Analysis**")
                if has_sl_data and pd.notna(row.get('optimal_stop_loss')):
                    st.markdown(f"- Current SL: **{row['current_stop_loss']:.0f}** pips")
                    st.markdown(f"- Optimal SL: **{row['optimal_stop_loss']:.1f}** pips")

                    sl_diff = row['optimal_stop_loss'] - row['current_stop_loss']
                    if abs(sl_diff) > 1:
                        direction = "v Tighten by" if sl_diff < 0 else "^ Widen by"
                        st.markdown(f"- Suggested Change: **{direction} {abs(sl_diff):.0f} pips**")

                    sl_action = row.get('sl_recommendation', 'N/A')
                    st.markdown(f"- Action: **{sl_action}**")
                    st.markdown(f"- Priority: **{sl_priority.upper()}**")
                else:
                    st.markdown("*No SL data available. Run fresh analysis.*")

                st.markdown("---")
                st.markdown(f"- Win Rate: **{row['win_rate']:.0f}%**")
                st.markdown(f"- Confidence: **{row['confidence'].upper()}**")

            # Efficiency metrics
            st.markdown("**Efficiency Metrics**")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("BE Reach Rate", f"{row['be_reach_rate']:.0f}%")
            with metric_col2:
                st.metric("BE Protection Rate", f"{row['be_protection_rate']:.0f}%")
            with metric_col3:
                st.metric("BE Profit Rate", f"{row['be_profit_rate']:.0f}%")

            # Analysis notes
            if row.get('analysis_notes'):
                st.info(f"{row['analysis_notes']}")

    # Export option
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        csv = be_display_df.to_csv(index=False)
        st.download_button(
            "Export CSV",
            csv,
            "breakeven_analysis.csv",
            "text/csv"
        )
