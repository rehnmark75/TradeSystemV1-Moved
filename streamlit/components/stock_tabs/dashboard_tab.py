"""
Stock Scanner Dashboard Tab

Quick snapshot of the scanner system with:
- Key metrics (active signals, high quality, Claude analyzed, today's signals)
- Scanner leaderboard by profit factor
- Top A+/A signals
- Quality tier and scanner distribution charts
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


def render_dashboard_tab(service):
    """Render the Dashboard tab - quick snapshot of scanner system."""
    with st.spinner("Loading dashboard..."):
        stats = service.get_dashboard_stats()
        leaderboard = service.get_scanner_leaderboard()
        top_signals = service.get_top_signals(limit=5, min_tier='A')

    if not stats:
        st.info("No dashboard data available. Run the scanner to generate signals.")
        return

    # Header
    last_scan = stats.get('last_scan', 'Unknown')
    if isinstance(last_scan, datetime):
        last_scan = last_scan.strftime('%Y-%m-%d %H:%M')

    st.markdown(f"""
    <div class="main-header">
        <h2>Stock Scanner Dashboard</h2>
        <p>Scanning {stats.get('total_stocks', 3454):,} US stocks via 7 backtested strategies | Last scan: {last_scan}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics - Row 1
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Signals", stats.get('active_signals', 0))

    with col2:
        st.metric("High Quality (A+/A)", stats.get('high_quality', 0),
                  help="Signals with quality tier A+ or A")

    with col3:
        st.metric("Claude Analyzed", stats.get('claude_analyzed', 0),
                  help="Signals with AI analysis complete")

    with col4:
        st.metric("Today's Signals", stats.get('today_signals', 0))

    st.markdown("---")

    # Two columns: Scanner Leaderboard and Top Signals
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Scanner Leaderboard (by Profit Factor)</div>', unsafe_allow_html=True)

        if leaderboard:
            # Create horizontal bar chart
            scanner_names = [s['scanner_name'].replace('_', ' ').title() for s in leaderboard]
            profit_factors = [float(s.get('profit_factor', 1.0)) for s in leaderboard]
            win_rates = [float(s.get('win_rate', 0)) for s in leaderboard]

            # Reverse for horizontal bar chart (top scanner at top)
            scanner_names = scanner_names[::-1]
            profit_factors = profit_factors[::-1]

            # Color gradient based on PF
            colors = ['#28a745' if pf >= 2.0 else '#17a2b8' if pf >= 1.5 else '#ffc107' if pf >= 1.0 else '#dc3545'
                      for pf in profit_factors]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=scanner_names,
                x=profit_factors,
                orientation='h',
                text=[f'{pf:.2f}' for pf in profit_factors],
                textposition='outside',
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Profit Factor: %{x:.2f}<extra></extra>"
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=60, t=20, b=20),
                xaxis_title="Profit Factor",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

            # Leaderboard details
            st.markdown("**Backtest Performance:**")
            for i, scanner in enumerate(leaderboard[:5]):
                name = scanner['scanner_name'].replace('_', ' ').title()
                pf = float(scanner.get('profit_factor', 1.0))
                wr = float(scanner.get('win_rate', 0))
                signals = scanner.get('total_signals', 0)

                pf_color = '#28a745' if pf >= 2.0 else '#17a2b8' if pf >= 1.5 else '#ffc107'
                st.markdown(f"{i+1}. **{name}** - PF: <span style='color:{pf_color}'>{pf:.2f}</span> | WR: {wr:.0f}% | Signals: {signals}", unsafe_allow_html=True)
        else:
            st.info("No scanner performance data available. Run backtests to generate metrics.")

    with col2:
        st.markdown('<div class="section-header">Top A+/A Signals</div>', unsafe_allow_html=True)

        if top_signals:
            for signal in top_signals:
                ticker = signal.get('ticker', 'N/A')
                sig_type = signal.get('signal_type', 'BUY')
                quality_tier = signal.get('quality_tier', 'B')
                score = signal.get('composite_score', 0)
                entry = signal.get('entry_price', 0)
                scanner = signal.get('scanner_name', '').replace('_', ' ').title()

                sig_class = "signal-card-buy" if sig_type == "BUY" else "signal-card-sell"
                sig_color = "#28a745" if sig_type == "BUY" else "#dc3545"
                tier_class = f"tier-{quality_tier.lower().replace('+', '-plus')}"

                # Claude analysis badge
                claude_grade = signal.get('claude_grade')
                claude_badge = ""
                if claude_grade:
                    grade_colors = {'A+': '#1e7e34', 'A': '#28a745', 'B': '#17a2b8', 'C': '#ffc107', 'D': '#6c757d'}
                    grade_color = grade_colors.get(claude_grade, '#6c757d')
                    claude_badge = f'<span style="background-color: {grade_color}; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; margin-left: 5px;">{claude_grade}</span>'

                st.markdown(f"""
                <div class="signal-card {sig_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><b>{ticker}</b> <span style="color: {sig_color};">{sig_type}</span>{claude_badge}</span>
                        <span class="tier-badge {tier_class}">{quality_tier}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #555;">
                        Entry: ${entry:.2f} | Score: {score}
                    </div>
                    <div style="font-size: 0.75rem; color: #888;">
                        {scanner}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-quality signals available.")

    st.markdown("---")

    # Bottom row: Quality Distribution and Scanner Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Quality Tier Distribution</div>', unsafe_allow_html=True)
        tier_dist = stats.get('tier_distribution', {})

        if tier_dist:
            tiers = ['A+', 'A', 'B', 'C', 'D']
            counts = [tier_dist.get(t, 0) for t in tiers]
            colors = ['#1e7e34', '#28a745', '#17a2b8', '#ffc107', '#6c757d']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tiers,
                y=counts,
                text=counts,
                textposition='outside',
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Signals: %{y}<extra></extra>"
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=40),
                yaxis_title="Number of Signals",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tier distribution data available.")

    with col2:
        st.markdown('<div class="section-header">Signals by Scanner</div>', unsafe_allow_html=True)
        scanner_dist = stats.get('scanner_distribution', {})

        if scanner_dist:
            scanner_names = [k.replace('_', ' ').title() for k in scanner_dist.keys()]
            scanner_counts = list(scanner_dist.values())

            fig = go.Figure(data=[go.Pie(
                labels=scanner_names,
                values=scanner_counts,
                hole=0.4,
                textinfo='label+value',
                marker_colors=px.colors.qualitative.Set2[:len(scanner_names)]
            )])
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scanner distribution data available.")
