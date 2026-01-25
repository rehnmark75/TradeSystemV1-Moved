"""
Stock Scanner Module

A stock market scanner for the TradeSystem platform.
Provides scanning, signal generation, and trading capabilities for US stocks.

Data Sources:
- yfinance: Historical OHLCV data
- RoboMarkets: Instrument list, real-time quotes, order execution

Usage:
    # CLI
    python -m stock_scanner.main status
    python -m stock_scanner.main fetch-watchlist --days 60
    python -m stock_scanner.main sync-tickers

    # Programmatic
    from stock_scanner.core.data_fetcher import StockDataFetcher
    from stock_scanner.core.trading.robomarkets_client import RoboMarketsClient
"""

__version__ = "0.1.0"
__author__ = "TradeSystem"

from . import config

__all__ = ["config", "__version__"]
