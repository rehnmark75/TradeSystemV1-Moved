"""
Stock Scanner Tab Components

This module provides modular tab components for the Stock Scanner dashboard.
Each tab component handles its own rendering and uses the stock_analytics_service
for data fetching.

Architecture:
- dashboard_tab: Quick snapshot with scanner leaderboard, top signals, quality distribution
- signals_tab: Unified signal browser with filters, comparison, CSV export
- watchlists_tab: 5 predefined technical screens
- scanner_performance_tab: Per-scanner backtest metrics and drill-down
- broker_stats_tab: Real trading performance from RoboMarkets
- chart_tab: Interactive charting with EMAs, MACD, Volume, and Deep Dive analysis
- market_context_tab: Market regime, sector rotation, RS analysis
- position_calculator: Risk-based position sizing widget
"""

from .dashboard_tab import render_dashboard_tab
from .signals_tab import render_signals_tab
from .watchlists_tab import render_watchlists_tab
from .scanner_performance_tab import render_scanner_performance_tab
from .broker_stats_tab import render_broker_stats_tab
from .chart_tab import render_chart_tab
from .market_context_tab import render_market_context_tab
from .position_calculator import render_position_calculator, render_position_calculator_modal
from .screener_builder_tab import render_screener_builder_tab

__all__ = [
    'render_dashboard_tab',
    'render_signals_tab',
    'render_watchlists_tab',
    'render_scanner_performance_tab',
    'render_broker_stats_tab',
    'render_chart_tab',
    'render_market_context_tab',
    'render_position_calculator',
    'render_position_calculator_modal',
    'render_screener_builder_tab',
]
