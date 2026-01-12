"""
Unified Trading Analytics Dashboard

Slim orchestrator that combines modular tab components into a unified tabbed interface.
All business logic and data fetching is handled by dedicated services and tab components.

This file is approximately 150 lines (down from 6,578 lines).

Architecture:
- Tab components: streamlit/components/tabs/*.py
- Services: streamlit/services/*.py
- This file: Tab layout orchestration only

Performance optimization: Uses lazy loading to only render the active tab's content,
avoiding the Streamlit default behavior of rendering all tabs on every page load.
"""

import streamlit as st
from datetime import datetime
from typing import Callable

# Import cache warmer for faster first loads
from services.cache_warmer import run_cache_warmup_async


def lazy_import_tab(tab_name: str) -> Callable:
    """
    Lazy import tab render function to avoid loading all tabs at startup.
    Only imports the tab module when actually needed.
    """
    if tab_name == "overview":
        from components.tabs import render_overview_tab
        return render_overview_tab
    elif tab_name == "strategy_performance":
        from components.tabs import render_strategy_performance_tab
        return render_strategy_performance_tab
    elif tab_name == "trade_analysis":
        from components.tabs import render_trade_analysis_tab
        return render_trade_analysis_tab
    elif tab_name == "market_intelligence":
        from components.tabs import render_market_intelligence_tab
        return render_market_intelligence_tab
    elif tab_name == "performance_snapshot":
        from components.tabs import render_performance_snapshot_tab
        return render_performance_snapshot_tab
    elif tab_name == "breakeven_optimizer":
        from components.tabs import render_breakeven_optimizer_tab
        return render_breakeven_optimizer_tab
    elif tab_name == "trade_performance":
        from components.tabs import render_trade_performance_tab
        return render_trade_performance_tab
    elif tab_name == "alert_history":
        from components.tabs import render_alert_history_tab
        return render_alert_history_tab
    elif tab_name == "smc_rejections":
        from components.tabs import render_smc_rejections_tab
        return render_smc_rejections_tab
    elif tab_name == "ema_rejections":
        from components.tabs import render_ema_double_rejections_tab
        return render_ema_double_rejections_tab
    elif tab_name == "unfilled_orders":
        from components.tabs import render_unfilled_orders_tab
        return render_unfilled_orders_tab
    elif tab_name == "settings":
        from components.tabs import render_settings_tab
        return render_settings_tab
    elif tab_name == "smc_config":
        from components.tabs import render_smc_config_tab
        return render_smc_config_tab
    elif tab_name == "scanner_config":
        from components.tabs import render_scanner_config_tab
        return render_scanner_config_tab
    elif tab_name == "intelligence_config":
        from components.tabs import render_intelligence_config_tab
        return render_intelligence_config_tab
    elif tab_name == "backtest_results":
        from components.tabs import render_backtest_results_tab
        return render_backtest_results_tab
    elif tab_name == "backtest_config":
        from components.tabs import render_backtest_config_tab
        return render_backtest_config_tab
    elif tab_name == "htf_analysis":
        from components.tabs import render_htf_analysis_tab
        return render_htf_analysis_tab
    else:
        raise ValueError(f"Unknown tab: {tab_name}")


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
    if 'active_main_tab' not in st.session_state:
        st.session_state.active_main_tab = 0


def main():
    """Main application entry point."""
    initialize_session_state()

    # Pre-warm caches in background (only runs once per session)
    run_cache_warmup_async()

    # App header
    st.title("Trading Analytics Hub")
    st.markdown("*Unified dashboard for comprehensive trading analysis*")

    # Main tab selection using radio buttons in sidebar for better performance
    # This avoids the Streamlit behavior of rendering all tab contents
    main_tabs = ["Overview", "Analysis", "Performance", "Rejections", "Settings"]

    with st.sidebar:
        st.markdown("### Navigation")
        selected_main = st.radio(
            "Main Section",
            main_tabs,
            index=st.session_state.active_main_tab,
            key="main_tab_radio",
            label_visibility="collapsed"
        )
        # Update session state
        st.session_state.active_main_tab = main_tabs.index(selected_main)

    # Only render the selected main tab's content
    # This avoids the Streamlit default behavior of rendering ALL tabs on every page load
    if selected_main == "Overview":
        render_func = lazy_import_tab("overview")
        render_func()

    elif selected_main == "Analysis":
        # Use radio buttons for sub-navigation to avoid rendering all sub-tabs
        analysis_options = [
            "Strategy",
            "Trade Analysis",
            "HTF Analysis",
            "Market Intelligence",
            "Performance Snapshots",
            "Breakeven Optimizer",
            "Backtest Results",
            "Backtest Config"
        ]
        analysis_sub = st.radio(
            "Analysis Section",
            analysis_options,
            horizontal=True,
            key="analysis_sub_radio",
            label_visibility="collapsed"
        )

        if analysis_sub == "Strategy":
            render_func = lazy_import_tab("strategy_performance")
            render_func()
        elif analysis_sub == "Trade Analysis":
            render_func = lazy_import_tab("trade_analysis")
            render_func()
        elif analysis_sub == "HTF Analysis":
            render_func = lazy_import_tab("htf_analysis")
            render_func()
        elif analysis_sub == "Market Intelligence":
            render_func = lazy_import_tab("market_intelligence")
            render_func()
        elif analysis_sub == "Performance Snapshots":
            render_func = lazy_import_tab("performance_snapshot")
            render_func()
        elif analysis_sub == "Breakeven Optimizer":
            render_func = lazy_import_tab("breakeven_optimizer")
            render_func()
        elif analysis_sub == "Backtest Results":
            render_func = lazy_import_tab("backtest_results")
            render_func()
        elif analysis_sub == "Backtest Config":
            render_func = lazy_import_tab("backtest_config")
            render_func()

    elif selected_main == "Performance":
        # Use radio buttons for sub-navigation
        perf_options = ["Trade Performance", "Alert History"]
        perf_sub = st.radio(
            "Performance Section",
            perf_options,
            horizontal=True,
            key="perf_sub_radio",
            label_visibility="collapsed"
        )

        if perf_sub == "Trade Performance":
            render_func = lazy_import_tab("trade_performance")
            render_func()
        elif perf_sub == "Alert History":
            render_func = lazy_import_tab("alert_history")
            render_func()

    elif selected_main == "Rejections":
        # Use radio buttons for sub-navigation
        rejection_options = ["SMC Rejections", "EMA Rejections", "Unfilled Orders"]
        rej_sub = st.radio(
            "Rejections Section",
            rejection_options,
            horizontal=True,
            key="rej_sub_radio",
            label_visibility="collapsed"
        )

        if rej_sub == "SMC Rejections":
            render_func = lazy_import_tab("smc_rejections")
            render_func()
        elif rej_sub == "EMA Rejections":
            render_func = lazy_import_tab("ema_rejections")
            render_func()
        elif rej_sub == "Unfilled Orders":
            render_func = lazy_import_tab("unfilled_orders")
            render_func()

    elif selected_main == "Settings":
        # Use radio buttons for sub-navigation to avoid rendering all 4 config tabs
        settings_options = ["General", "SMC Config", "Scanner Config", "Intelligence Config"]
        settings_sub = st.radio(
            "Settings Section",
            settings_options,
            horizontal=True,
            key="settings_sub_radio",
            label_visibility="collapsed"
        )

        if settings_sub == "General":
            render_func = lazy_import_tab("settings")
            render_func()
        elif settings_sub == "SMC Config":
            render_func = lazy_import_tab("smc_config")
            render_func()
        elif settings_sub == "Scanner Config":
            render_func = lazy_import_tab("scanner_config")
            render_func()
        elif settings_sub == "Intelligence Config":
            render_func = lazy_import_tab("intelligence_config")
            render_func()

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
