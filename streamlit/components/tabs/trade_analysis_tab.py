"""
Trade Analysis Tab Component

Renders the Individual Trade Analysis tab with:
- Trade selection dropdown
- Trailing Stop Analysis sub-tab
- Signal Analysis sub-tab
- Outcome Analysis sub-tab
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import traceback

from services.trade_analysis_service import TradeAnalysisService


def render_trade_analysis_tab():
    """Render the Trade Analysis tab with sub-tabs for trailing stop and signal analysis."""
    service = TradeAnalysisService()

    st.header("Individual Trade Analysis")
    st.markdown("*Comprehensive analysis of trade execution and entry signals*")

    # Fetch only filled trades (closed or tracking) - exclude unfilled limit orders
    filled_trades = service.fetch_filled_trades_for_analysis()

    if filled_trades.empty:
        st.warning("No filled trades found for analysis.")
        return

    # Create trade selection options
    trade_options = {
        f"#{row['id']} | {row['symbol_short']} | {row['direction']} | {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | {row['pnl_display']}": row['id']
        for _, row in filled_trades.iterrows()
    }

    # Input for trade selection (shared between sub-tabs)
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_trade = st.selectbox(
            "Select Trade to Analyze",
            options=list(trade_options.keys()),
            key="trade_select_input",
            help="Only showing filled trades (closed or currently open)"
        )
        trade_id = trade_options[selected_trade]

    with col2:
        analyze_btn = st.button("Analyze Trade", type="primary")

    # Sub-tabs for different analysis types
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Trailing Stop Analysis",
        "Signal Analysis",
        "Outcome Analysis"
    ])

    with sub_tab1:
        _render_trailing_stop_analysis(service, trade_id, analyze_btn)

    with sub_tab2:
        _render_signal_analysis(service, trade_id, analyze_btn)

    with sub_tab3:
        _render_outcome_analysis(service, trade_id, analyze_btn)


def _render_trailing_stop_analysis(service: TradeAnalysisService, trade_id: int, analyze_btn: bool):
    """Render the trailing stop stage analysis sub-tab."""
    st.subheader("Trailing Stop Stage Analysis")
    st.markdown("*Analyze break-even triggers and profit lock stages*")

    if analyze_btn or trade_id:
        with st.spinner(f"Analyzing trade {trade_id}..."):
            data = service.get_trailing_stop_analysis(trade_id)

        if not data:
            st.error("Failed to get analysis data")
            return

        if "error" in data:
            if data["error"] == "not_found":
                st.error(f"Trade {trade_id} not found in database")
            else:
                st.error(f"Error: {data['message']}")
            return

        # Trade Details Section
        st.subheader("Trade Details")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Symbol", data['trade_details']['symbol'].replace('CS.D.', '').replace('.MINI.IP', ''))
            st.metric("Direction", data['trade_details']['direction'])

        with col2:
            st.metric("Entry Price", f"{data['trade_details']['entry_price']:.5f}")
            st.metric("Stop Loss", f"{data['trade_details']['sl_price']:.5f}")

        with col3:
            st.metric("Take Profit", f"{data['trade_details']['tp_price']:.5f}")
            st.metric("Status", data['trade_details']['status'].upper())

        with col4:
            metrics = data['calculated_metrics']
            sl_icon = "+" if metrics['sl_above_entry'] else "-"
            st.metric("SL Distance", f"{metrics['sl_distance_pts']:.1f} pts")
            st.metric("Protection", f"{sl_icon}{metrics['sl_distance_pts']:.1f}")

        # Pair Configuration Section
        st.subheader("Pair-Specific Configuration")
        cfg = data['pair_configuration']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Break-Even</h4>
                <p><strong>Trigger:</strong> {cfg['break_even_trigger_points']} points</p>
                <p><strong>Lock:</strong> 0 points (entry)</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Stage 1 (Profit Lock)</h4>
                <p><strong>Trigger:</strong> {cfg['stage1_trigger_points']} points</p>
                <p><strong>Lock:</strong> {cfg['stage1_lock_points']} points profit</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Stage 2 (Profit Lock)</h4>
                <p><strong>Trigger:</strong> {cfg['stage2_trigger_points']} points</p>
                <p><strong>Lock:</strong> {cfg['stage2_lock_points']} points profit</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Stage 3 (ATR Trailing)</h4>
                <p><strong>Trigger:</strong> {cfg['stage3_trigger_points']} points</p>
                <p><strong>ATR:</strong> {cfg['stage3_atr_multiplier']}x multiplier</p>
            </div>
            """, unsafe_allow_html=True)

        # Stage Activation Analysis
        st.subheader("Stage Activation Analysis")
        stages = data['stage_analysis']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            be_status = 'ACTIVATED' if stages['breakeven']['activated'] else 'NOT REACHED'
            be_icon = "[+]" if stages['breakeven']['activated'] else "[-]"
            be_bg = '#d4edda' if stages['breakeven']['activated'] else '#f8d7da'

            be_extra = ""
            if stages['breakeven']['activated']:
                be_extra = f"<p><strong>Time:</strong> {stages['breakeven']['activation_time']}</p><p><strong>Highest Profit:</strong> {stages['breakeven']['max_profit_reached']} pts</p><p><strong>Final Lock:</strong> +{stages['breakeven']['final_lock']} pts</p>"
            else:
                be_extra = f"<p><strong>Required:</strong> {stages['breakeven']['trigger_threshold']} pts</p>"

            st.markdown(f"""
            <div class="metric-card" style="background: {be_bg};">
                <h4>{be_icon} Break-Even</h4>
                <p><strong>Status:</strong> {be_status}</p>
                {be_extra}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            stage1_status = 'ACTIVATED' if stages['stage1']['activated'] else 'NOT REACHED'
            stage1_icon = "[+]" if stages['stage1']['activated'] else "[-]"
            stage1_bg = '#d4edda' if stages['stage1']['activated'] else '#fff3cd'

            stage1_extra = ""
            if stages['stage1']['activated']:
                actual_lock = stages['stage1'].get('actual_lock', stages['stage1']['lock_amount'])
                stage1_extra = f"<p><strong>Time:</strong> {stages['stage1']['activation_time']}</p><p><strong>Actual Lock:</strong> +{actual_lock} pts</p><p><strong>Expected:</strong> +{stages['stage1']['lock_amount']} pts</p>"
            else:
                stage1_extra = f"<p><strong>Required:</strong> {stages['stage1']['trigger_threshold']} pts</p><p><strong>Would Lock:</strong> +{stages['stage1']['lock_amount']} pts</p>"

            st.markdown(f"""
            <div class="metric-card" style="background: {stage1_bg};">
                <h4>{stage1_icon} Stage 1: Profit Lock</h4>
                <p><strong>Status:</strong> {stage1_status}</p>
                {stage1_extra}
            </div>
            """, unsafe_allow_html=True)

        with col3:
            stage2_status = 'ACTIVATED' if stages['stage2']['activated'] else 'NOT REACHED'
            stage2_icon = "[+]" if stages['stage2']['activated'] else "[-]"
            stage2_bg = '#d4edda' if stages['stage2']['activated'] else '#fff3cd'

            stage2_extra = ""
            if stages['stage2']['activated']:
                stage2_extra = f"<p><strong>Time:</strong> {stages['stage2']['activation_time']}</p>"
            else:
                stage2_extra = f"<p><strong>Required:</strong> {stages['stage2']['trigger_threshold']} pts</p><p><strong>Would Lock:</strong> +{stages['stage2']['lock_amount']} pts</p>"

            st.markdown(f"""
            <div class="metric-card" style="background: {stage2_bg};">
                <h4>{stage2_icon} Stage 2: Profit Lock</h4>
                <p><strong>Status:</strong> {stage2_status}</p>
                {stage2_extra}
            </div>
            """, unsafe_allow_html=True)

        with col4:
            stage3_status = 'ACTIVATED' if stages['stage3']['activated'] else 'NOT REACHED'
            stage3_icon = "[+]" if stages['stage3']['activated'] else "[-]"
            stage3_bg = '#d4edda' if stages['stage3']['activated'] else '#fff3cd'

            stage3_extra = ""
            if stages['stage3']['activated']:
                stage3_extra = f"<p><strong>Time:</strong> {stages['stage3']['activation_time']}</p>"
            else:
                stage3_extra = f"<p><strong>Required:</strong> {stages['stage3']['trigger_threshold']} pts</p>"

            st.markdown(f"""
            <div class="metric-card" style="background: {stage3_bg};">
                <h4>{stage3_icon} Stage 3: ATR Trailing</h4>
                <p><strong>Status:</strong> {stage3_status}</p>
                {stage3_extra}
            </div>
            """, unsafe_allow_html=True)

        # Performance Summary
        st.subheader("Performance Summary")
        summary = data['summary']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Stages Activated", f"{summary['stages_activated']}/3")

        with col2:
            st.metric("Highest Trigger Reached", f"{summary['max_profit_reached']} pts")

        with col3:
            profit_protected = summary['final_protection']
            protection_icon = "+" if profit_protected > 0 else "-"
            st.metric("Final Protection", f"{protection_icon}{profit_protected:.1f} pts")

        with col4:
            fully_trailed = "Yes" if summary['fully_trailed'] else "No"
            st.metric("Fully Trailed", fully_trailed)

        # Profit Timeline Chart
        if data['timeline']['profit_progression']:
            st.subheader("Profit Progression Timeline")

            profit_data = data['timeline']['profit_progression']
            timestamps = [p['timestamp'] for p in profit_data]
            profits = [p['profit_pts'] for p in profit_data]

            fig = go.Figure()

            # Add profit line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=profits,
                mode='lines+markers',
                name='Profit (pts)',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            # Add stage threshold lines
            fig.add_hline(y=cfg['break_even_trigger_points'], line_dash="dash",
                         line_color="green", annotation_text="Stage 1 Trigger")
            fig.add_hline(y=cfg['stage2_trigger_points'], line_dash="dash",
                         line_color="orange", annotation_text="Stage 2 Trigger")
            fig.add_hline(y=cfg['stage3_trigger_points'], line_dash="dash",
                         line_color="red", annotation_text="Stage 3 Trigger")

            fig.update_layout(
                title=f"Trade {trade_id} - Profit Evolution",
                xaxis_title="Time",
                yaxis_title="Profit (points)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Event Log
        st.subheader("Event Log")

        # Break-even events
        if data['timeline']['break_even_events']:
            st.markdown("**Break-Even Triggers:**")
            for event in data['timeline']['break_even_events']:
                st.info(f"@ {event['timestamp']}: Profit {event['profit_pts']}pts >= Trigger {event['trigger_pts']}pts")

        # Stop adjustments
        if data['timeline']['stop_adjustments']:
            st.markdown("**Stop Adjustments:**")
            for event in data['timeline']['stop_adjustments']:
                st.success(f"@ {event['timestamp']}: Stop moved to {event['new_stop']:.5f}")

        # Raw Data (Expandable)
        with st.expander("View Raw Analysis Data"):
            st.json(data)


def _render_signal_analysis(service: TradeAnalysisService, trade_id: int, analyze_btn: bool):
    """Render the strategy signal analysis sub-tab."""
    st.subheader("Entry Signal Analysis")
    st.markdown("*Analyze the strategy signal that triggered this trade*")

    if analyze_btn or trade_id:
        with st.spinner(f"Analyzing signal for trade {trade_id}..."):
            data = service.get_signal_analysis(trade_id)

        if not data:
            st.error("Failed to get signal analysis data")
            return

        if "error" in data:
            if data["error"] == "not_found":
                st.error(f"Trade {trade_id} not found in database")
            else:
                st.error(f"Error: {data['message']}")
            return

        # Check if signal exists
        if not data.get('has_signal', False):
            st.warning(f"{data.get('message', 'No signal data available for this trade')}")
            if 'trade_details' in data:
                td = data['trade_details']
                st.info(f"Trade: {td.get('symbol')} | {td.get('direction')} | Entry: {td.get('entry_price'):.5f}")
            return

        # Signal Overview Section
        st.subheader("Signal Overview")
        sig = data['signal_overview']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pair_display = sig['pair'] or sig['epic'].replace('CS.D.', '').replace('.MINI.IP', '')
            st.metric("Pair", pair_display)
            direction_icon = "+" if sig['direction'] == "BUY" else "-"
            st.metric("Direction", f"{direction_icon} {sig['direction']}")

        with col2:
            st.metric("Strategy", sig['strategy'] or "Unknown")
            st.metric("Timeframe", sig['timeframe'] or "N/A")

        with col3:
            conf_pct = sig['confidence_score'] * 100 if sig['confidence_score'] < 1 else sig['confidence_score']
            st.metric("Confidence", f"{conf_pct:.1f}%")
            st.metric("Price at Signal", f"{sig['price_at_signal']:.5f}")

        with col4:
            st.metric("Spread", f"{sig['spread_pips']:.1f} pips" if sig['spread_pips'] else "N/A")
            st.metric("Session", sig['market_session'] or "Unknown")

        # Detect strategy type from signal overview
        strategy_name = (sig.get('strategy') or '').upper()
        is_smc_simple = 'SMC_SIMPLE' in strategy_name

        # Get raw strategy indicators for SMC_SIMPLE display
        raw_data = data.get('raw_data', {})
        strategy_indicators = raw_data.get('strategy_indicators', {})

        # Smart Money / Strategy Validation Section
        if is_smc_simple:
            _render_smc_simple_analysis(data, strategy_indicators)
        else:
            _render_smc_structure_analysis(data)

        # Common sections for all strategies
        _render_confluence_factors(data)
        _render_entry_timing(data, is_smc_simple, strategy_indicators)
        _render_technical_context(data)
        _render_risk_reward(data)
        _render_trade_outcome_correlation(data)

        # Raw Data Expander
        with st.expander("View Raw Signal Data"):
            st.json(data.get('raw_data', {}))


def _render_smc_simple_analysis(data, strategy_indicators):
    """Render SMC Simple 3-tier analysis."""
    st.subheader("SMC Simple - 3-Tier Analysis")

    # Extract tier data from raw strategy_indicators
    tier1 = strategy_indicators.get('tier1_ema', {})
    tier2 = strategy_indicators.get('tier2_swing', {})
    tier3 = strategy_indicators.get('tier3_entry', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        # TIER 1: EMA Bias
        ema_dir = tier1.get('direction', 'Unknown')
        dir_icon = "^" if ema_dir == 'BULL' else ("v" if ema_dir == 'BEAR' else "-")
        dir_bg = '#d4edda' if ema_dir in ['BULL', 'BEAR'] else '#f8d7da'
        ema_value = tier1.get('ema_value', 0)
        ema_distance = tier1.get('distance_pips', 0)
        st.markdown(f"""
        <div class="metric-card" style="background: {dir_bg};">
            <h4>{dir_icon} TIER 1: 4H EMA Bias</h4>
            <p><strong>Direction:</strong> {ema_dir}</p>
            <p><strong>EMA {tier1.get('ema_period', 50)}:</strong> {ema_value:.5f}</p>
            <p><strong>Distance:</strong> {ema_distance:.1f} pips</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # TIER 2: Swing Break
        swing_level = tier2.get('swing_level', 0)
        body_confirmed = tier2.get('body_close_confirmed', False)
        vol_confirmed = tier2.get('volume_confirmed', False)
        body_icon = "[+]" if body_confirmed else "[-]"
        vol_icon = "[+]" if vol_confirmed else "[ ]"
        st.markdown(f"""
        <div class="metric-card">
            <h4>TIER 2: Swing Break</h4>
            <p><strong>Swing Level:</strong> {swing_level:.5f}</p>
            <p><strong>Body Close:</strong> {body_icon} {'Confirmed' if body_confirmed else 'Not confirmed'}</p>
            <p><strong>Volume:</strong> {vol_icon} {'Confirmed' if vol_confirmed else 'No spike'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # TIER 3: Entry Timing
        entry_price = tier3.get('entry_price', 0)
        pullback = tier3.get('pullback_depth', 0)
        fib_zone = tier3.get('fib_zone', 'N/A')
        in_optimal = tier3.get('in_optimal_zone', False)
        optimal_icon = "[*]" if in_optimal else "[ ]"
        st.markdown(f"""
        <div class="metric-card">
            <h4>TIER 3: Entry Timing</h4>
            <p><strong>Entry:</strong> {entry_price:.5f}</p>
            <p><strong>Pullback:</strong> {pullback*100:.1f}%</p>
            <p><strong>Fib Zone:</strong> {fib_zone}</p>
            <p><strong>Optimal:</strong> {optimal_icon} {'Yes' if in_optimal else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)


def _render_smc_structure_analysis(data):
    """Render original SMC Structure analysis."""
    st.subheader("Smart Money Validation")
    smc = data['smart_money_analysis']

    col1, col2, col3 = st.columns(3)

    with col1:
        validated = smc['validated']
        val_icon = "[+]" if validated else "[-]"
        val_bg = '#d4edda' if validated else '#f8d7da'
        st.markdown(f"""
        <div class="metric-card" style="background: {val_bg};">
            <h4>{val_icon} SMC Validated</h4>
            <p><strong>Type:</strong> {smc['type'] or 'Unknown'}</p>
            <p><strong>Score:</strong> {smc['score']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ms = smc['market_structure']
        structure = ms['current_structure'] or ms['structure_type'] or 'Unknown'
        struct_icon = "^" if 'bullish' in structure.lower() else ("v" if 'bearish' in structure.lower() else "-")
        st.markdown(f"""
        <div class="metric-card">
            <h4>{struct_icon} Market Structure</h4>
            <p><strong>Type:</strong> {structure.upper()}</p>
            <p><strong>Trend Strength:</strong> {ms['trend_strength']*100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Structure breaks
        breaks = ms.get('structure_breaks', [])
        break_count = len(breaks) if isinstance(breaks, list) else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Structure Details</h4>
            <p><strong>Swing High:</strong> {ms['swing_high']:.5f}</p>
            <p><strong>Swing Low:</strong> {ms['swing_low']:.5f}</p>
            <p><strong>Breaks:</strong> {break_count}</p>
        </div>
        """, unsafe_allow_html=True)


def _render_confluence_factors(data):
    """Render confluence factors section."""
    st.subheader("Confluence Factors")
    conf = data['confluence_factors']

    factors_present = conf['factors_present']
    factors_total = conf['factors_total']

    # Progress bar for confluence
    if factors_total > 0:
        progress = factors_present / factors_total
        st.progress(progress, text=f"Confluence: {factors_present}/{factors_total} factors present")

    # Display individual factors
    if conf['factors']:
        cols = st.columns(min(len(conf['factors']), 6))
        for i, factor in enumerate(conf['factors']):
            with cols[i % len(cols)]:
                icon = "[+]" if factor['present'] else "[-]"
                bg = '#d4edda' if factor['present'] else '#fff3cd'
                st.markdown(f"""
                <div style="background: {bg}; padding: 0.5rem; border-radius: 5px; margin: 0.25rem 0; text-align: center;">
                    {icon} {factor['name']}
                </div>
                """, unsafe_allow_html=True)


def _render_entry_timing(data, is_smc_simple, strategy_indicators):
    """Render entry timing quality section."""
    st.subheader("Entry Timing Quality")
    timing = data['entry_timing']

    if is_smc_simple:
        # SMC_SIMPLE: Display Fibonacci pullback and entry quality
        tier3 = strategy_indicators.get('tier3_entry', {})
        risk_mgmt = strategy_indicators.get('risk_management', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pullback = tier3.get('pullback_depth', 0)
            # Determine pullback quality based on Fib levels
            if 0.382 <= pullback <= 0.500:
                pullback_quality = "Optimal (38.2-50%)"
                pullback_color = '#28a745'
            elif 0.236 <= pullback <= 0.618:
                pullback_quality = "Good (Fib zone)"
                pullback_color = '#ffc107'
            else:
                pullback_quality = "Extended"
                pullback_color = '#dc3545'
            st.markdown(f"""
            <div class="metric-card">
                <h4>Pullback Depth</h4>
                <h3 style="color: {pullback_color};">{pullback*100:.1f}%</h3>
                <small>{pullback_quality}</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fib_zone = tier3.get('fib_zone', 'N/A')
            in_optimal = tier3.get('in_optimal_zone', False)
            zone_icon = "[*]" if in_optimal else "[ ]"
            zone_bg = '#d4edda' if in_optimal else '#f0f0f0'
            st.markdown(f"""
            <div class="metric-card" style="background: {zone_bg};">
                <h4>{zone_icon} Fibonacci Zone</h4>
                <p><strong>Range:</strong> {fib_zone}</p>
                <p><strong>Optimal:</strong> {'Yes' if in_optimal else 'No'}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            rr_ratio = risk_mgmt.get('rr_ratio', 0)
            rr_color = '#28a745' if rr_ratio >= 2 else ('#ffc107' if rr_ratio >= 1.5 else '#dc3545')
            st.markdown(f"""
            <div class="metric-card">
                <h4>Risk:Reward</h4>
                <h3 style="color: {rr_color};">{rr_ratio:.2f}</h3>
                <small>{'Excellent' if rr_ratio >= 2 else ('Good' if rr_ratio >= 1.5 else 'Fair')}</small>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            entry_quality = timing.get('entry_quality_score', 0) * 100
            quality_color = '#28a745' if entry_quality >= 70 else ('#ffc107' if entry_quality >= 50 else '#dc3545')
            st.markdown(f"""
            <div class="metric-card">
                <h4>Entry Quality</h4>
                <h3 style="color: {quality_color};">{entry_quality:.0f}%</h3>
                <small>Overall score</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Original SMC_STRUCTURE: Premium/Discount zones
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            zone = timing['premium_discount_zone']
            zone_icon = "+" if zone == 'discount' else ("-" if zone == 'premium' else "~")
            zone_quality = "Good for BUY" if zone == 'discount' else ("Good for SELL" if zone == 'premium' else "Neutral")
            st.markdown(f"""
            <div class="metric-card">
                <h4>Price Zone</h4>
                <h3>{zone_icon} {zone.upper()}</h3>
                <small>{zone_quality}</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            htf_aligned = timing['htf_aligned']
            htf_icon = "[+]" if htf_aligned else "[-]"
            st.markdown(f"""
            <div class="metric-card">
                <h4>HTF Alignment</h4>
                <h3>{htf_icon} {'Aligned' if htf_aligned else 'Not Aligned'}</h3>
                <small>Structure: {timing['htf_structure']}</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            mtf_ratio = timing['mtf_alignment_ratio'] * 100
            htf_strength = timing.get('htf_strength', 0) * 100
            if mtf_ratio > 0:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>MTF Consensus</h4>
                    <h3>{mtf_ratio:.0f}%</h3>
                    <small>Timeframes aligned</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                htf_color = '#28a745' if htf_strength >= 60 else ('#ffc107' if htf_strength >= 40 else '#dc3545')
                st.markdown(f"""
                <div class="metric-card">
                    <h4>HTF Strength</h4>
                    <h3 style="color: {htf_color};">{htf_strength:.0f}%</h3>
                    <small>4H trend strength</small>
                </div>
                """, unsafe_allow_html=True)

        with col4:
            entry_quality = timing['entry_quality_score'] * 100
            quality_color = '#28a745' if entry_quality >= 70 else ('#ffc107' if entry_quality >= 50 else '#dc3545')
            st.markdown(f"""
            <div class="metric-card">
                <h4>Entry Quality</h4>
                <h3 style="color: {quality_color};">{entry_quality:.0f}%</h3>
                <small>Overall score</small>
            </div>
            """, unsafe_allow_html=True)


def _render_technical_context(data):
    """Render technical context section."""
    st.subheader("Technical Context at Entry")
    tech = data['technical_context']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ema_50 = tech.get('ema_50', 0)
        ema_200 = tech.get('ema_200', 0)
        price_pos = tech.get('price_vs_ema_50', 'unknown')
        st.markdown(f"""
        <div class="metric-card">
            <h4>EMA Analysis</h4>
            <p><strong>EMA 50:</strong> {ema_50:.5f}</p>
            <p><strong>EMA 200:</strong> {ema_200:.5f}</p>
            <p><strong>Price vs EMA50:</strong> {price_pos.upper()}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        macd = tech.get('macd', {})
        macd_dir = macd.get('direction', 'unknown')
        macd_icon = "^" if macd_dir == 'bullish' else "v"
        hist_val = macd.get('histogram', 0)
        hist_color = '#28a745' if hist_val > 0 else '#dc3545'
        st.markdown(f"""
        <div class="metric-card">
            <h4>{macd_icon} MACD</h4>
            <p><strong>Line:</strong> {macd.get('line', 0):.6f}</p>
            <p><strong>Signal:</strong> {macd.get('signal', 0):.6f}</p>
            <p style="color: {hist_color};"><strong>Histogram:</strong> {hist_val:.6f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        bb = tech.get('bollinger_bands', {})
        if bb and bb.get('upper', 0) > 0:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Bollinger Bands</h4>
                <p><strong>Upper:</strong> {bb.get('upper', 0):.5f}</p>
                <p><strong>Middle:</strong> {bb.get('middle', 0):.5f}</p>
                <p><strong>Lower:</strong> {bb.get('lower', 0):.5f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            rsi = tech.get('rsi', 50)
            rsi_zone = tech.get('rsi_zone', 'neutral')
            rsi_color = '#dc3545' if rsi_zone == 'overbought' else ('#28a745' if rsi_zone == 'oversold' else '#6c757d')
            st.markdown(f"""
            <div class="metric-card">
                <h4>RSI</h4>
                <h3 style="color: {rsi_color};">{rsi:.1f}</h3>
                <small>{rsi_zone.upper()}</small>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        vol_ratio = tech.get('volume_ratio', 0)
        vol_conf = tech.get('volume_confirmation', False)
        vol_icon = "[+]" if vol_conf else "[ ]"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volume & ATR</h4>
            <p><strong>Ratio:</strong> {vol_ratio:.2f}x</p>
            <p><strong>Confirmed:</strong> {vol_icon}</p>
            <p><strong>ATR:</strong> {tech.get('atr', 0):.5f}</p>
        </div>
        """, unsafe_allow_html=True)


def _render_risk_reward(data):
    """Render risk/reward section."""
    st.subheader("Risk/Reward Setup")
    rr = data.get('risk_reward', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        initial_rr = rr.get('initial_rr', 0)
        rr_color = '#28a745' if initial_rr >= 2 else ('#ffc107' if initial_rr >= 1.5 else '#dc3545')
        st.markdown(f"""
        <div class="metric-card">
            <h4>Initial R:R Ratio</h4>
            <h2 style="color: {rr_color};">{initial_rr:.2f}</h2>
            <small>{'Excellent' if initial_rr >= 2 else ('Good' if initial_rr >= 1.5 else 'Poor')}</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Risk/Reward Pips</h4>
            <p><strong>Risk:</strong> {rr.get('risk_pips', 0):.1f} pips</p>
            <p><strong>Reward:</strong> {rr.get('reward_pips', 0):.1f} pips</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Trade Levels</h4>
            <p><strong>Entry:</strong> {rr.get('entry_price', 0):.5f}</p>
            <p><strong>SL:</strong> {rr.get('stop_loss', 0):.5f}</p>
            <p><strong>TP:</strong> {rr.get('take_profit', 0):.5f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        partial_tp = rr.get('partial_tp', 0)
        partial_pct = rr.get('partial_percent', 0)
        if partial_tp > 0:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Partial Take Profit</h4>
                <p><strong>Level:</strong> {partial_tp:.5f}</p>
                <p><strong>Size:</strong> {partial_pct:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h4>No Partial TP</h4>
                <p>Full position to TP</p>
            </div>
            """, unsafe_allow_html=True)


def _render_trade_outcome_correlation(data):
    """Render trade outcome correlation section."""
    st.subheader("Trade Outcome Correlation")
    outcome = data['trade_outcome']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = outcome['status']
        status_icon = "[+]" if status == 'closed' else ("~" if status == 'tracking' else "?")
        st.metric("Status", f"{status_icon} {status.upper()}")

    with col2:
        is_winner = outcome['is_winner']
        if is_winner is not None:
            result_icon = "WIN" if is_winner else "LOSS"
            result_bg = '#d4edda' if is_winner else '#f8d7da'
            result_color = '#28a745' if is_winner else '#dc3545'
            st.markdown(f"""
            <div class="metric-card" style="background: {result_bg};">
                <h4>Result</h4>
                <h2 style="color: {result_color};">{result_icon}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Result", "Pending")

    with col3:
        pnl = outcome['profit_loss']
        pnl_color = '#28a745' if pnl > 0 else '#dc3545'
        st.markdown(f"""
        <div class="metric-card">
            <h4>P&L</h4>
            <h3 style="color: {pnl_color};">{'+' if pnl > 0 else ''}{pnl:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        pips = outcome['pips_gained']
        pips_color = '#28a745' if pips > 0 else '#dc3545'
        duration = outcome['duration_minutes']
        duration_str = f"{duration // 60}h {duration % 60}m" if duration else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Pips & Duration</h4>
            <p style="color: {pips_color};"><strong>{'+' if pips > 0 else ''}{pips:.1f} pips</strong></p>
            <small>Duration: {duration_str}</small>
        </div>
        """, unsafe_allow_html=True)


def _render_outcome_analysis(service: TradeAnalysisService, trade_id: int, analyze_btn: bool):
    """Render the trade outcome analysis sub-tab - WHY the trade won or lost."""
    st.subheader("Trade Outcome Analysis")
    st.markdown("*Understand WHY this trade won or lost - Learn from every trade*")

    if analyze_btn or trade_id:
        with st.spinner(f"Analyzing outcome for trade {trade_id}..."):
            data = service.get_outcome_analysis(trade_id)

        if not data:
            st.error("Failed to get outcome analysis data")
            return

        if "error" in data:
            if data["error"] == "not_found":
                st.error(f"Trade {trade_id} not found in database")
            else:
                st.error(f"Error: {data['message']}")
            return

        # Check if trade is still open
        if data.get('status') == 'TRADE_STILL_OPEN':
            st.warning(f"{data.get('message', 'Trade is still open')}")
            if 'trade_details' in data:
                td = data['trade_details']
                st.info(f"Trade: {td.get('symbol')} | {td.get('direction')} | Entry: {td.get('entry_price'):.5f} | Status: {td.get('status')}")
            return

        # ===== OUTCOME SUMMARY SECTION =====
        _render_outcome_summary(data)

        # ===== MFE/MAE ANALYSIS SECTION =====
        _render_mfe_mae_analysis(data)

        # ===== ENTRY QUALITY SECTION =====
        _render_entry_quality_assessment(data)

        # ===== EXIT QUALITY SECTION =====
        _render_exit_quality_assessment(data)

        # ===== LEARNING INSIGHTS SECTION =====
        _render_learning_insights(data)

        # ===== TRADE DETAILS SECTION =====
        with st.expander("Trade Details"):
            td = data['trade_details']
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Symbol", td['symbol'].replace('CS.D.', '').replace('.MINI.IP', ''))
                st.metric("Direction", td['direction'])
                st.metric("Deal ID", td.get('deal_id', 'N/A'))

            with col2:
                st.metric("Entry Price", f"{td['entry_price']:.5f}")
                st.metric("Exit Price", f"{td['exit_price']:.5f}" if td.get('exit_price') else "N/A")
                st.metric("Moved to BE", "Yes" if td['moved_to_breakeven'] else "No")

            with col3:
                st.metric("Stop Loss", f"{td['sl_price']:.5f}")
                st.metric("Take Profit", f"{td['tp_price']:.5f}")
                st.metric("Alert ID", td.get('alert_id', 'N/A'))

            st.write(f"**Opened:** {td.get('opened_at', 'N/A')}")
            st.write(f"**Closed:** {td.get('closed_at', 'N/A')}")

        # Raw Data Expander
        with st.expander("View Raw Outcome Data"):
            st.json(data)


def _render_outcome_summary(data):
    """Render outcome summary section."""
    st.subheader("Outcome Summary")
    summary = data['outcome_summary']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        result = summary['result']
        if result == "WIN":
            result_bg = '#d4edda'
            result_icon = "WIN"
            result_color = '#28a745'
        elif result == "LOSS":
            result_bg = '#f8d7da'
            result_icon = "LOSS"
            result_color = '#dc3545'
        else:
            result_bg = '#fff3cd'
            result_icon = "BE"
            result_color = '#856404'
        st.markdown(f"""
        <div class="metric-card" style="background: {result_bg}; text-align: center;">
            <h4>Result</h4>
            <h2 style="color: {result_color};">{result_icon}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pips = summary['pips_gained']
        profit_loss = summary['profit_loss']
        pnl_color = '#28a745' if profit_loss > 0 else '#dc3545'
        if pips != 0:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4>P&L</h4>
                <h3 style="color: {pnl_color};">{'+' if pips > 0 else ''}{pips:.1f} pips</h3>
                <small>{'+' if profit_loss > 0 else ''}{profit_loss:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4>P&L</h4>
                <h3 style="color: {pnl_color};">{'+' if profit_loss > 0 else ''}{profit_loss:.2f}</h3>
                <small>pips not recorded</small>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        r_mult = summary['r_multiple']
        if r_mult == 0 and summary.get('mfe_pips', 0) > 0:
            mfe_mae_ratio = summary.get('mfe_mae_ratio', 0)
            if result == "WIN":
                r_display = f"~{mfe_mae_ratio:.1f}R*"
                r_color = '#28a745'
            elif result == "LOSS":
                r_display = f"-1R*"
                r_color = '#dc3545'
            else:
                r_display = "0R"
                r_color = '#856404'
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4>R-Multiple</h4>
                <h3 style="color: {r_color};">{r_display}</h3>
                <small>*estimated</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            r_color = '#28a745' if r_mult > 0 else '#dc3545'
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4>R-Multiple</h4>
                <h3 style="color: {r_color};">{'+' if r_mult > 0 else ''}{r_mult:.2f}R</h3>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        exit_type = summary['exit_type']
        exit_icon = "TP" if exit_type == "TP_HIT" else ("SL" if exit_type == "SL_HIT" else "?")
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h4>Exit Type</h4>
            <h3>{exit_icon} {exit_type.replace('_', ' ')}</h3>
            <small>Duration: {summary['duration_display']}</small>
        </div>
        """, unsafe_allow_html=True)


def _render_mfe_mae_analysis(data):
    """Render MFE/MAE analysis section."""
    st.subheader("MFE/MAE Analysis")
    st.markdown("*Maximum Favorable Excursion (MFE) vs Maximum Adverse Excursion (MAE)*")

    price_action = data['price_action_analysis']
    mfe = price_action['mfe']
    mae = price_action['mae']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mfe_pips = mfe['pips']
        st.markdown(f"""
        <div class="metric-card" style="background: #d4edda;">
            <h4>MFE (Max Profit)</h4>
            <h3 style="color: #28a745;">+{mfe_pips:.1f} pips</h3>
            <small>Peak: {mfe['time_to_peak_minutes']}m</small>
            <p><small>{mfe['percentage_of_tp']:.0f}% of TP distance</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        mae_pips = mae['pips']
        st.markdown(f"""
        <div class="metric-card" style="background: #f8d7da;">
            <h4>MAE (Max Drawdown)</h4>
            <h3 style="color: #dc3545;">-{mae_pips:.1f} pips</h3>
            <small>Trough: {mae['time_to_trough_minutes']}m</small>
            <p><small>{mae['percentage_of_sl']:.0f}% of SL distance</small></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        ratio = price_action['mfe_mae_ratio']
        ratio_color = '#28a745' if ratio > 2 else ('#ffc107' if ratio > 1 else '#dc3545')
        ratio_verdict = "Excellent" if ratio > 3 else ("Good" if ratio > 2 else ("Fair" if ratio > 1 else "Poor"))
        st.markdown(f"""
        <div class="metric-card">
            <h4>MFE/MAE Ratio</h4>
            <h3 style="color: {ratio_color};">{ratio:.2f}</h3>
            <small>{ratio_verdict}</small>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        initial = price_action['initial_move']
        initial_icon = "+" if initial == "FAVORABLE" else ("-" if initial == "ADVERSE" else "~")
        reversal = price_action['immediate_reversal']
        reversal_icon = "!" if reversal else "[+]"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Initial Move</h4>
            <h3>{initial_icon} {initial}</h3>
            <small>{reversal_icon} {'Immediate Reversal' if reversal else 'No Quick Reversal'}</small>
        </div>
        """, unsafe_allow_html=True)


def _render_entry_quality_assessment(data):
    """Render entry quality assessment section."""
    st.subheader("Entry Quality Assessment")
    entry_q = data['entry_quality_assessment']

    score = entry_q['score']
    verdict = entry_q['verdict']
    verdict_color = '#28a745' if verdict == "GOOD_ENTRY" else ('#ffc107' if verdict in ["AVERAGE_ENTRY", "BELOW_AVERAGE_ENTRY"] else '#dc3545')
    verdict_bg = '#d4edda' if verdict == "GOOD_ENTRY" else ('#fff3cd' if verdict in ["AVERAGE_ENTRY", "BELOW_AVERAGE_ENTRY"] else '#f8d7da')

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: {verdict_bg}; text-align: center;">
            <h4>Entry Score</h4>
            <h2 style="color: {verdict_color};">{score:.0f}/100</h2>
            <p><strong>{verdict.replace('_', ' ')}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        factors = entry_q.get('factors', [])
        if factors:
            for factor in factors:
                points = factor.get('points', 0)
                max_pts = factor.get('max_points', 0)
                pct = (points / max_pts * 100) if max_pts > 0 else 0
                bar_color = '#28a745' if pct >= 70 else ('#ffc107' if pct >= 40 else '#dc3545')
                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{factor['name']}: {factor['value']}</span>
                        <span><strong>+{points:.0f}/{max_pts}</strong></span>
                    </div>
                    <div style="background: #e9ecef; border-radius: 4px; height: 8px;">
                        <div style="background: {bar_color}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def _render_exit_quality_assessment(data):
    """Render exit quality assessment section."""
    st.subheader("Exit Quality Assessment")
    exit_q = data['exit_quality_assessment']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        exit_verdict = exit_q['verdict']
        exit_v_color = '#28a745' if exit_verdict in ["OPTIMAL_EXIT", "EXCELLENT_EXIT", "GOOD_EXIT"] else ('#ffc107' if exit_verdict == "ACCEPTABLE_EXIT" else '#dc3545')
        exit_v_bg = '#d4edda' if exit_verdict in ["OPTIMAL_EXIT", "EXCELLENT_EXIT", "GOOD_EXIT"] else ('#fff3cd' if exit_verdict == "ACCEPTABLE_EXIT" else '#f8d7da')
        st.markdown(f"""
        <div class="metric-card" style="background: {exit_v_bg}; text-align: center;">
            <h4>Exit Verdict</h4>
            <h3 style="color: {exit_v_color};">{exit_verdict.replace('_', ' ')}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        efficiency = exit_q['exit_efficiency_pct']
        eff_color = '#28a745' if efficiency >= 70 else ('#ffc107' if efficiency >= 40 else '#dc3545')
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h4>Exit Efficiency</h4>
            <h3 style="color: {eff_color};">{efficiency:.0f}%</h3>
            <small>of MFE captured</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        actual_pips = exit_q['actual_pips']
        summary = data['outcome_summary']
        if actual_pips != 0:
            actual_display = f"{'+' if actual_pips > 0 else ''}{actual_pips:.1f} pips"
        else:
            actual_pnl = summary['profit_loss']
            actual_display = f"{'+' if actual_pnl > 0 else ''}{actual_pnl:.2f}"
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h4>Comparison</h4>
            <p>MFE: <strong>+{exit_q['mfe_pips']:.1f}</strong> pips</p>
            <p>Actual: <strong>{actual_display}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        missed = exit_q['missed_profit_pips']
        missed_color = '#dc3545' if missed > 10 else ('#ffc107' if missed > 5 else '#28a745')
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h4>Missed Profit</h4>
            <h3 style="color: {missed_color};">{missed:.1f} pips</h3>
            <small>{exit_q['verdict_details']}</small>
        </div>
        """, unsafe_allow_html=True)


def _render_learning_insights(data):
    """Render learning insights section."""
    st.subheader("Learning Insights")
    insights = data['learning_insights']

    result_color = '#28a745' if insights['trade_result'] == "WIN" else ('#dc3545' if insights['trade_result'] == "LOSS" else '#856404')
    result_bg = '#d4edda' if insights['trade_result'] == "WIN" else ('#f8d7da' if insights['trade_result'] == "LOSS" else '#fff3cd')

    # Primary Factor
    st.markdown(f"""
    <div class="metric-card" style="background: {result_bg}; border-left: 4px solid {result_color};">
        <h4>PRIMARY FACTOR</h4>
        <p style="font-size: 1.1rem;"><strong>{insights['primary_factor']}</strong></p>
        <small>Pattern: <em>{insights.get('pattern_identified', 'Unknown').replace('_', ' ')}</em></small>
    </div>
    """, unsafe_allow_html=True)

    # What went right/wrong
    col1, col2 = st.columns(2)

    with col1:
        if insights.get('what_went_right'):
            st.markdown("#### What Went Right")
            for item in insights['what_went_right']:
                st.success(f"* {item}")
        elif insights['trade_result'] == "WIN":
            st.markdown("#### What Went Right")
            st.success("* Trade reached target as planned")

    with col2:
        if insights.get('what_went_wrong'):
            st.markdown("#### What Went Wrong")
            for item in insights['what_went_wrong']:
                st.error(f"* {item}")

    # Contributing Factors
    if insights.get('contributing_factors'):
        st.markdown("#### Contributing Factors")
        for factor in insights['contributing_factors']:
            st.info(f"* {factor}")

    # Improvement Suggestions
    if insights.get('improvement_suggestions'):
        st.markdown("#### Improvement Suggestions")
        for suggestion in insights['improvement_suggestions']:
            st.warning(f"* {suggestion}")

    # Key Takeaway
    if insights.get('key_takeaway'):
        st.markdown("---")
        st.markdown(f"""
        <div style="background: #e7f1ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0066cc;">
            <h4>KEY TAKEAWAY</h4>
            <p style="font-size: 1.1rem; margin: 0;"><em>{insights['key_takeaway']}</em></p>
        </div>
        """, unsafe_allow_html=True)
