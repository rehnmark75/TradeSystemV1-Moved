"""
Stock Scanner Dashboard

Slim orchestrator that combines modular tab components into a unified tabbed interface.
All business logic and data fetching is handled by dedicated services and tab components.

This file is approximately 150 lines (down from 3,192 lines).

Architecture:
- Tab components: streamlit/components/stock_tabs/*.py
- Services: streamlit/services/stock_analytics_service.py
- This file: Tab layout orchestration only
"""

import streamlit as st
from datetime import datetime

# Import stock analytics service
import sys
sys.path.insert(0, '/app')
from services.stock_analytics_service import get_stock_service

# Import tab components
from components.stock_tabs import (
    render_dashboard_tab,
    render_signals_tab,
    render_watchlists_tab,
    render_scanner_performance_tab,
    render_broker_stats_tab,
    render_chart_tab,
    render_market_context_tab,
    render_screener_builder_tab,
)

# Page configuration
st.set_page_config(
    page_title="Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header styles */
    .main-header {
        background: linear-gradient(90deg, #1a5f7a 0%, #57c5b6 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h2 { margin: 0; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1a5f7a;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a5f7a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }

    /* Quality tier badges (A+, A, B, C, D) */
    .tier-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .tier-a-plus { background: #1e7e34; color: white; }
    .tier-a { background: #28a745; color: white; }
    .tier-b { background: #17a2b8; color: white; }
    .tier-c { background: #ffc107; color: black; }
    .tier-d { background: #6c757d; color: white; }

    /* Scanner leaderboard */
    .leaderboard-item {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.3rem;
        background: #f8f9fa;
    }
    .leaderboard-rank {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1a5f7a;
        width: 30px;
    }
    .leaderboard-name {
        flex-grow: 1;
        font-weight: 500;
    }
    .leaderboard-pf {
        font-weight: bold;
        color: #28a745;
    }

    /* Signal comparison */
    .comparison-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    .comparison-selected {
        border-color: #1a5f7a;
        background: #e8f4f8;
    }

    /* Watchlist selector */
    .watchlist-info {
        background: #e8f4f8;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1a5f7a;
    }

    /* Signal styles */
    .signal-buy { color: #28a745; font-weight: bold; }
    .signal-sell { color: #dc3545; font-weight: bold; }
    .signal-card {
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .signal-card-buy {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
    }
    .signal-card-sell {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
    }

    /* Trend styles */
    .trend-up { color: #28a745; }
    .trend-down { color: #dc3545; }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e9ecef;
    }

    /* Score bar */
    .score-bar {
        height: 6px;
        background: #e9ecef;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.2rem;
    }
    .score-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Filter section */
    .filter-section {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    service = get_stock_service()

    # Create tabs - 8-tab structure with Custom Screener
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Dashboard",
        "📡 Signals",
        "📋 Watchlists",
        "🌍 Market Context",
        "🔍 Custom Screener",
        "📈 Scanner Performance",
        "💹 Broker Stats",
        "📉 Chart & Analysis"
    ])

    with tab1:
        render_dashboard_tab(service)

    with tab2:
        render_signals_tab(service)

    with tab3:
        render_watchlists_tab(service)

    with tab4:
        render_market_context_tab(service)

    with tab5:
        render_screener_builder_tab(service)

    with tab6:
        render_scanner_performance_tab(service)

    with tab7:
        render_broker_stats_tab()

    with tab8:
        render_chart_tab(service)


if __name__ == "__main__":
    main()
