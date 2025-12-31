"""
Unified Trading Analytics Dashboard

Slim orchestrator that combines modular tab components into a unified tabbed interface.
All business logic and data fetching is handled by dedicated services and tab components.

This file is approximately 150 lines (down from 6,578 lines).

Architecture:
- Tab components: streamlit/components/tabs/*.py
- Services: streamlit/services/*.py
- This file: Tab layout orchestration only
"""

import streamlit as st
from datetime import datetime

# Import cache warmer for faster first loads
from services.cache_warmer import run_cache_warmup_async

# Import tab components
from components.tabs import (
    render_smc_rejections_tab,
    render_ema_double_rejections_tab,
    render_alert_history_tab,
    render_overview_tab,
    render_strategy_performance_tab,
    render_trade_performance_tab,
    render_unfilled_orders_tab,
    render_settings_tab,
    render_market_intelligence_tab,
    render_breakeven_optimizer_tab,
    render_trade_analysis_tab,
    render_smc_config_tab,
)

# Configure page
st.set_page_config(
    page_title="Trading Analytics Hub",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for trading dashboard
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .profit-positive { color: #28a745; font-weight: bold; }
    .profit-negative { color: #dc3545; font-weight: bold; }
    .win-rate-high { color: #28a745; }
    .win-rate-medium { color: #ffc107; }
    .win-rate-low { color: #dc3545; }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-profitable { background-color: #28a745; }
    .status-losing { background-color: #dc3545; }
    .status-neutral { background-color: #6c757d; }
    .profit { color: #28a745; font-weight: bold; }
    .loss { color: #dc3545; font-weight: bold; }
    .pending { color: #ffc107; font-weight: bold; }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = '7_days'
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"


def main():
    """Main application entry point."""
    initialize_session_state()

    # Pre-warm caches in background (only runs once per session)
    run_cache_warmup_async()

    # App header
    st.title("Trading Analytics Hub")
    st.markdown("*Unified dashboard for comprehensive trading analysis*")

    # Tab navigation - Grouped into 5 main categories with sub-tabs
    tab_overview, tab_analysis, tab_performance, tab_rejections, tab_settings = st.tabs([
        "Overview",
        "Analysis",
        "Performance",
        "Rejections",
        "Settings"
    ])

    with tab_overview:
        render_overview_tab()

    with tab_analysis:
        # Sub-tabs for analysis features
        analysis_tabs = st.tabs([
            "Strategy",
            "Trade Analysis",
            "Market Intelligence",
            "Breakeven Optimizer"
        ])
        with analysis_tabs[0]:
            render_strategy_performance_tab()
        with analysis_tabs[1]:
            render_trade_analysis_tab()
        with analysis_tabs[2]:
            render_market_intelligence_tab()
        with analysis_tabs[3]:
            render_breakeven_optimizer_tab()

    with tab_performance:
        # Sub-tabs for performance tracking
        perf_tabs = st.tabs([
            "Trade Performance",
            "Alert History"
        ])
        with perf_tabs[0]:
            render_trade_performance_tab()
        with perf_tabs[1]:
            render_alert_history_tab()

    with tab_rejections:
        # Sub-tabs for rejection analysis
        rejection_tabs = st.tabs([
            "SMC Rejections",
            "EMA Rejections",
            "Unfilled Orders"
        ])
        with rejection_tabs[0]:
            render_smc_rejections_tab()
        with rejection_tabs[1]:
            render_ema_double_rejections_tab()
        with rejection_tabs[2]:
            render_unfilled_orders_tab()

    with tab_settings:
        # Sub-tabs for settings and configuration
        settings_tabs = st.tabs([
            "General",
            "SMC Config"
        ])
        with settings_tabs[0]:
            render_settings_tab()
        with settings_tabs[1]:
            render_smc_config_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 12px;'>
        Trading Analytics Hub v2.0 |
        Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")} |
        Status: Online
        </div>
        """,
        unsafe_allow_html=True
    )


# Run the dashboard
if __name__ == "__main__":
    main()
