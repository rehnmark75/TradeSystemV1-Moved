"""
Cache Warmer Service

Pre-warms the Streamlit cache on app startup to reduce first-load latency.
Runs common queries in background threads to populate @st.cache_data caches.
"""

import streamlit as st
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def warm_rejection_caches():
    """Pre-warm SMC and EMA rejection caches with default parameters."""
    try:
        from services.rejection_analytics_service import RejectionAnalyticsService
        service = RejectionAnalyticsService()

        # Warm with default 3-day filter
        logger.info("Pre-warming SMC rejection cache (3 days)...")
        service.fetch_smc_rejections(days=3, stage_filter="All", pair_filter="All", session_filter="All")
        service.fetch_smc_rejection_stats(days=3)

        logger.info("Pre-warming EMA rejection cache (3 days)...")
        service.fetch_ema_double_rejections(days=3, stage_filter="All", pair_filter="All", session_filter="All")
        service.fetch_ema_double_rejection_stats(days=3)

        logger.info("Rejection caches pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-warm rejection caches: {e}")


def warm_overview_caches():
    """Pre-warm overview tab caches."""
    try:
        from services.trading_analytics_service import TradingAnalyticsService
        service = TradingAnalyticsService()

        logger.info("Pre-warming trading statistics cache (7 days)...")
        service.fetch_trading_statistics(days_back=7)
        service.fetch_trades_dataframe(days_back=7)

        logger.info("Overview caches pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-warm overview caches: {e}")


def warm_alert_history_cache():
    """Pre-warm alert history cache."""
    try:
        from services.alert_history_service import AlertHistoryService
        service = AlertHistoryService()

        logger.info("Pre-warming alert history cache (3 days)...")
        service.fetch_alert_history(days=3, status_filter="All", strategy_filter="All", pair_filter="All")

        logger.info("Alert history cache pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-warm alert history cache: {e}")


def warm_stock_scanner_caches():
    """Pre-warm stock scanner watchlist results and dashboard stats."""
    try:
        from services.stock_analytics_service import get_stock_service
        service = get_stock_service()

        logger.info("Pre-warming stock scanner dashboard stats...")
        service.get_dashboard_stats()
        service.get_scanner_leaderboard()

        watchlists = [
            'ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross',
            'gap_up_continuation', 'rsi_oversold_bounce'
        ]

        for wl in watchlists:
            logger.info(f"Pre-warming watchlist cache: {wl}...")
            # Warm with default limit 100
            service.get_watchlist_results(wl, limit=100)

        logger.info("Stock scanner caches pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-warm stock scanner caches: {e}")


def run_cache_warmup_async():
    """
    Run cache warmup in background threads.
    Call this once at app startup.
    """
    # Check if already warmed in this session
    if st.session_state.get('_cache_warmed', False):
        return

    st.session_state['_cache_warmed'] = True
    st.session_state['_cache_warm_started'] = datetime.now()

    logger.info("Starting async cache warmup...")

    # Run each warmup in its own thread
    threads = [
        threading.Thread(target=warm_rejection_caches, daemon=True),
        threading.Thread(target=warm_overview_caches, daemon=True),
        threading.Thread(target=warm_alert_history_cache, daemon=True),
        threading.Thread(target=warm_stock_scanner_caches, daemon=True),
    ]

    for t in threads:
        t.start()

    # Don't join - let them run in background while user interacts with app
    logger.info(f"Cache warmup threads started: {len(threads)}")


def run_cache_warmup_sync():
    """
    Run cache warmup synchronously (blocking).
    Use this if you want to ensure caches are warm before showing UI.
    """
    if st.session_state.get('_cache_warmed', False):
        return

    st.session_state['_cache_warmed'] = True

    with st.spinner("Loading data..."):
        warm_overview_caches()
        warm_rejection_caches()
        warm_alert_history_cache()
