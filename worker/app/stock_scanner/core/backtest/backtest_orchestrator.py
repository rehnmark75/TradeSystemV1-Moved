"""
Stock Backtest Orchestrator

Coordinates the complete backtest workflow:
1. Create execution record
2. Iterate through date range
3. For each date, scan for signals
4. Simulate trades and log results
5. Calculate final statistics
"""

import logging
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Type
import pandas as pd

from ..database.async_database_manager import AsyncDatabaseManager
from .backtest_data_provider import BacktestDataProvider
from .trade_simulator import TradeSimulator
from .backtest_order_logger import BacktestOrderLogger
from ...strategies.ema_trend_pullback import EMATrendPullbackStrategy, PullbackSignal
from ...strategies.macd_momentum import MACDMomentumStrategy, MACDMomentumSignal
from ...strategies.zlma_crossover import ZLMACrossoverStrategy, ZLMASignal


# Strategy registry
STRATEGY_REGISTRY: Dict[str, Type] = {
    'EMA_PULLBACK': EMATrendPullbackStrategy,
    'EMA_TREND_PULLBACK': EMATrendPullbackStrategy,
    'MACD_MOMENTUM': MACDMomentumStrategy,
    'MACD': MACDMomentumStrategy,
    'ZLMA_CROSSOVER': ZLMACrossoverStrategy,
    'ZLMA': ZLMACrossoverStrategy,
}


class StockBacktestOrchestrator:
    """
    Orchestrates the complete stock backtesting workflow.

    Features:
    - Supports multiple strategies
    - Sector filtering
    - Strategy comparison mode
    - Progress tracking
    - CSV export
    """

    def __init__(
        self,
        db_manager: AsyncDatabaseManager,
        strategy_name: str = 'EMA_PULLBACK',
        timeframe: str = '1d',
        max_holding_days: int = 20
    ):
        self.db = db_manager
        self.strategy_name = strategy_name.upper()
        self.timeframe = timeframe
        self.max_holding_days = max_holding_days
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_provider = BacktestDataProvider(db_manager)
        self.trade_simulator = TradeSimulator(max_holding_days=max_holding_days)
        self.order_logger = BacktestOrderLogger(db_manager)

        # Initialize strategy
        self.strategy = self._create_strategy(strategy_name)

    def _create_strategy(self, strategy_name: str):
        """Create strategy instance from registry."""
        strategy_name = strategy_name.upper()
        if strategy_name not in STRATEGY_REGISTRY:
            available = ', '.join(STRATEGY_REGISTRY.keys())
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

        strategy_class = STRATEGY_REGISTRY[strategy_name]
        return strategy_class()

    async def run(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        days: int = 90,
        sector: Optional[str] = None,
        show_progress: bool = True
    ) -> int:
        """
        Run the backtest.

        Args:
            tickers: List of tickers to test (None = all tradeable)
            start_date: Start date (None = end_date - days)
            end_date: End date (None = today)
            days: Number of days if start_date not specified
            sector: Sector filter
            show_progress: Whether to print progress

        Returns:
            execution_id
        """
        # Determine date range
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Get tickers
        if tickers is None:
            tickers = await self.data_provider.get_tradeable_tickers(sector=sector)
            if not tickers:
                raise ValueError("No tradeable tickers found")

        self.logger.info(
            f"Starting backtest: {self.strategy_name} on {len(tickers)} tickers "
            f"from {start_date} to {end_date}"
        )

        # Reset logger stats
        self.order_logger.reset_stats()

        # Create execution record
        execution_id = await self.order_logger.create_execution(
            strategy_name=self.strategy_name,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            timeframe=self.timeframe,
            config=self.strategy.get_config() if hasattr(self.strategy, 'get_config') else None
        )

        try:
            # Process each ticker
            total_tickers = len(tickers)
            for idx, ticker in enumerate(tickers, 1):
                if show_progress and idx % 10 == 0:
                    print(f"  Processing {idx}/{total_tickers}: {ticker}")

                await self._process_ticker(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    execution_id=execution_id
                )

            # Finalize execution
            await self.order_logger.finalize_execution(execution_id, status='completed')

            self.logger.info(
                f"Backtest completed: execution_id={execution_id}, "
                f"signals={self.order_logger.signals_logged}, "
                f"winners={self.order_logger.winners}, losers={self.order_logger.losers}"
            )

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            await self.order_logger.finalize_execution(
                execution_id,
                status='failed',
                error_message=str(e)
            )
            raise

        return execution_id

    async def _process_ticker(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        execution_id: int
    ):
        """Process a single ticker for backtest."""
        # Get historical data with warmup
        df = await self.data_provider.get_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe=self.timeframe,
            include_warmup=True
        )

        # Require at least enough data for 200 EMA plus some buffer
        min_bars = 220  # 200 for EMA + 20 bar buffer
        if df.empty or len(df) < min_bars:
            self.logger.debug(f"{ticker}: Insufficient data ({len(df) if not df.empty else 0} bars, need {min_bars}), skipping")
            return

        # Get sector for context
        sector = await self.data_provider.get_ticker_sector(ticker)

        # Find start index (after warmup period)
        start_datetime = datetime.combine(start_date, datetime.min.time())
        warmup_end_idx = df[df['timestamp'] >= start_datetime].index.min()
        if pd.isna(warmup_end_idx):
            return

        # Iterate through each bar from start_date onwards
        for i in range(warmup_end_idx, len(df)):
            # Get data up to current bar (no future data leakage)
            current_data = df.iloc[:i + 1].copy()

            # Check for signal
            signal = self.strategy.scan(current_data, ticker, sector)

            if signal:
                # Get future data for trade simulation
                current_timestamp = current_data.iloc[-1]['timestamp']
                future_data = await self.data_provider.get_future_data(
                    ticker=ticker,
                    from_timestamp=current_timestamp,
                    bars=self.max_holding_days
                )

                # Simulate trade
                trade_result = None
                if not future_data.empty:
                    trade_result = self.trade_simulator.simulate_from_signal(
                        signal=signal,
                        future_data=future_data
                    )

                # Log signal and result
                await self.order_logger.log_signal(
                    execution_id=execution_id,
                    signal=signal,
                    trade_result=trade_result,
                    sector=sector
                )

    async def compare_strategies(
        self,
        strategies: List[str],
        tickers: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        days: int = 90,
        sector: Optional[str] = None
    ) -> List[int]:
        """
        Compare multiple strategies on the same data.

        Args:
            strategies: List of strategy names to compare
            Other args same as run()

        Returns:
            List of execution_ids
        """
        execution_ids = []

        for strategy_name in strategies:
            self.logger.info(f"Running comparison backtest for {strategy_name}")
            self.strategy_name = strategy_name.upper()
            self.strategy = self._create_strategy(strategy_name)

            execution_id = await self.run(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                days=days,
                sector=sector,
                show_progress=True
            )
            execution_ids.append(execution_id)

        return execution_ids

    async def get_comparison_results(self, execution_ids: List[int]) -> pd.DataFrame:
        """Get comparison table for multiple executions."""
        results = []

        for exec_id in execution_ids:
            summary = await self.order_logger.get_execution_summary(exec_id)
            if summary:
                results.append({
                    'execution_id': exec_id,
                    'strategy': summary.get('strategy_name'),
                    'signals': summary.get('total_signals', 0),
                    'trades': summary.get('total_trades', 0),
                    'winners': summary.get('winners', 0),
                    'losers': summary.get('losers', 0),
                    'win_rate': summary.get('win_rate', 0),
                    'total_pnl': summary.get('total_pnl_percent', 0),
                    'profit_factor': summary.get('profit_factor'),
                    'max_drawdown': summary.get('max_drawdown_percent', 0)
                })

        return pd.DataFrame(results)

    async def export_results(self, execution_id: int, filepath: str) -> bool:
        """Export execution results to CSV."""
        return await self.order_logger.export_to_csv(execution_id, filepath)

    async def get_execution_details(self, execution_id: int) -> Dict[str, Any]:
        """Get detailed execution information."""
        # Get execution summary
        summary = await self.order_logger.get_execution_summary(execution_id)
        if not summary:
            return {'error': 'Execution not found'}

        # Get signal breakdown
        query = """
            SELECT
                trade_result,
                exit_reason,
                COUNT(*) as count,
                AVG(pnl_percent) as avg_pnl,
                AVG(holding_days) as avg_holding_days
            FROM stock_backtest_signals
            WHERE execution_id = $1
            GROUP BY trade_result, exit_reason
            ORDER BY count DESC
        """
        breakdowns = await self.db.fetch(query, execution_id)

        # Get quality tier breakdown
        quality_query = """
            SELECT
                quality_tier,
                COUNT(*) as count,
                SUM(CASE WHEN trade_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                AVG(pnl_percent) as avg_pnl
            FROM stock_backtest_signals
            WHERE execution_id = $1
            GROUP BY quality_tier
            ORDER BY quality_tier
        """
        quality_breakdown = await self.db.fetch(quality_query, execution_id)

        # Get top performers
        top_query = """
            SELECT ticker, signal_timestamp, entry_price, exit_price,
                   trade_result, pnl_percent, exit_reason
            FROM stock_backtest_signals
            WHERE execution_id = $1
            ORDER BY pnl_percent DESC
            LIMIT 10
        """
        top_signals = await self.db.fetch(top_query, execution_id)

        # Get worst performers
        worst_query = """
            SELECT ticker, signal_timestamp, entry_price, exit_price,
                   trade_result, pnl_percent, exit_reason
            FROM stock_backtest_signals
            WHERE execution_id = $1
            ORDER BY pnl_percent ASC
            LIMIT 10
        """
        worst_signals = await self.db.fetch(worst_query, execution_id)

        return {
            'summary': dict(summary),
            'breakdown_by_result': [dict(r) for r in breakdowns],
            'breakdown_by_quality': [dict(r) for r in quality_breakdown],
            'top_performers': [dict(r) for r in top_signals],
            'worst_performers': [dict(r) for r in worst_signals]
        }
