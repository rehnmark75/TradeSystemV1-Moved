"""
Tab Components for Unified Analytics Dashboard

This module provides modular tab components that can be used in the main dashboard
or independently. Each tab component handles its own rendering and uses the
appropriate service for data fetching.
"""

from .smc_rejections_tab import render_smc_rejections_tab
from .ema_rejections_tab import render_ema_double_rejections_tab
from .alert_history_tab import render_alert_history_tab
from .overview_tab import render_overview_tab
from .strategy_performance_tab import render_strategy_performance_tab
from .trade_performance_tab import render_trade_performance_tab

__all__ = [
    'render_smc_rejections_tab',
    'render_ema_double_rejections_tab',
    'render_alert_history_tab',
    'render_overview_tab',
    'render_strategy_performance_tab',
    'render_trade_performance_tab',
]
