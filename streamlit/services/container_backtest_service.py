"""
Container-Aware Backtest Service
Works within the Streamlit container using available database connections
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sqlalchemy import create_engine, text
import streamlit as st

# Use existing services
from .data import get_candle_data, get_trade_logs, get_epics


@dataclass
class BacktestConfig:
    """Configuration for a container backtest run"""
    strategy_name: str
    epic: str = "CS.D.EURUSD.MINI.IP"
    days: int = 7
    timeframe: str = "15m"
    show_signals: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Container backtest result"""
    strategy_name: str
    epic: str
    timeframe: str
    total_signals: int
    signals: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    chart_data: Optional[pd.DataFrame] = None


@dataclass
class StrategyInfo:
    """Information about an available strategy"""
    name: str
    display_name: str
    description: str
    parameters: Dict[str, Any]


class ContainerBacktestService:
    """Backtest service that works within container constraints"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database connection using environment variables"""
        try:
            database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
            self.engine = create_engine(database_url)
            self.logger.info("✅ Database connection initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")

    def get_available_strategies(self) -> Dict[str, StrategyInfo]:
        """Get available strategies (using historical data analysis)"""
        strategies = {
            'historical_signals': StrategyInfo(
                name='historical_signals',
                display_name='Historical Signal Analysis',
                description='Analyze historical trading signals from the database',
                parameters={
                    'epic': {
                        "type": "select",
                        "default": "CS.D.EURUSD.MINI.IP",
                        "description": "Trading pair to analyze"
                    },
                    'days': {
                        "type": "number",
                        "default": 7,
                        "min": 1,
                        "max": 30,
                        "description": "Number of days to analyze"
                    },
                    'timeframe': {
                        "type": "select",
                        "default": "15m",
                        "options": ["5m", "15m", "1h"],
                        "description": "Chart timeframe"
                    },
                    'min_confidence': {
                        "type": "number",
                        "default": 0.7,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "description": "Minimum confidence threshold"
                    }
                }
            ),
            'simple_ma_cross': StrategyInfo(
                name='simple_ma_cross',
                display_name='Simple Moving Average Crossover',
                description='Basic MA crossover strategy using price data',
                parameters={
                    'epic': {
                        "type": "select",
                        "default": "CS.D.EURUSD.MINI.IP",
                        "description": "Trading pair"
                    },
                    'fast_ma': {
                        "type": "number",
                        "default": 10,
                        "min": 5,
                        "max": 50,
                        "description": "Fast MA period"
                    },
                    'slow_ma': {
                        "type": "number",
                        "default": 20,
                        "min": 10,
                        "max": 100,
                        "description": "Slow MA period"
                    },
                    'days': {
                        "type": "number",
                        "default": 7,
                        "min": 1,
                        "max": 30,
                        "description": "Days to analyze"
                    }
                }
            ),
            'price_breakout': StrategyInfo(
                name='price_breakout',
                display_name='Price Breakout Strategy',
                description='Detect breakouts from recent high/low levels',
                parameters={
                    'epic': {
                        "type": "select",
                        "default": "CS.D.EURUSD.MINI.IP",
                        "description": "Trading pair"
                    },
                    'lookback_periods': {
                        "type": "number",
                        "default": 20,
                        "min": 10,
                        "max": 50,
                        "description": "Lookback periods for high/low"
                    },
                    'breakout_threshold': {
                        "type": "number",
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "description": "Breakout threshold in pips"
                    },
                    'days': {
                        "type": "number",
                        "default": 7,
                        "min": 1,
                        "max": 30,
                        "description": "Days to analyze"
                    }
                }
            )
        }

        return strategies

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Run a backtest using the specified configuration"""
        start_time = datetime.now()

        try:
            self.logger.info(f"🚀 Running container backtest: {config.strategy_name}")

            if config.strategy_name == 'historical_signals':
                result = self._run_historical_signals_analysis(config)
            elif config.strategy_name == 'simple_ma_cross':
                result = self._run_ma_crossover_backtest(config)
            elif config.strategy_name == 'price_breakout':
                result = self._run_breakout_backtest(config)
            else:
                raise ValueError(f"Unknown strategy: {config.strategy_name}")

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            self.logger.info(f"✅ Backtest complete: {result.total_signals} signals in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Backtest failed: {str(e)}")

            return BacktestResult(
                strategy_name=config.strategy_name,
                epic=config.epic,
                timeframe=config.timeframe,
                total_signals=0,
                signals=[],
                performance_metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    def _run_historical_signals_analysis(self, config: BacktestConfig) -> BacktestResult:
        """Analyze historical signals from the database"""
        try:
            # Get historical trade data
            min_time = datetime.now() - timedelta(days=config.days)
            trades_df = get_trade_logs(self.engine, config.epic, min_time)

            # Get chart data
            timeframe_map = {"5m": 5, "15m": 15, "1h": 60}
            tf_minutes = timeframe_map.get(config.timeframe, 15)
            chart_data = get_candle_data(self.engine, tf_minutes, config.epic, limit=config.days * 24 * 4)

            signals = []
            if not trades_df.empty:
                min_confidence = config.parameters.get('min_confidence', 0.7)

                for _, trade in trades_df.iterrows():
                    # Filter by confidence if available
                    confidence = trade.get('confidence_score', 0.8)
                    if confidence < min_confidence:
                        continue

                    # Calculate performance
                    profit_loss = trade.get('profit_loss', 0)
                    entry_price = trade.get('entry_price', 0)

                    # Estimate pips (simplified)
                    pip_value = 0.0001 if 'JPY' not in config.epic else 0.01
                    profit_pips = profit_loss / pip_value if entry_price > 0 else 0

                    signal = {
                        'timestamp': trade['timestamp'],
                        'epic': config.epic,
                        'signal_type': trade.get('signal_type', 'HISTORICAL'),
                        'direction': trade.get('direction', 'UNKNOWN'),
                        'entry_price': entry_price,
                        'confidence': confidence,
                        'strategy': trade.get('strategy', 'Historical'),
                        'timeframe': config.timeframe,
                        'spread_pips': 1.5,  # Estimated
                        'max_profit_pips': max(profit_pips, 0),
                        'max_loss_pips': max(-profit_pips, 0),
                        'profit_loss_ratio': abs(profit_pips) / 10 if profit_pips != 0 else 1,
                        'actual_profit_loss': profit_loss,
                        'status': trade.get('status', 'CLOSED')
                    }
                    signals.append(signal)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(signals)

            return BacktestResult(
                strategy_name=config.strategy_name,
                epic=config.epic,
                timeframe=config.timeframe,
                total_signals=len(signals),
                signals=signals,
                performance_metrics=performance_metrics,
                execution_time=0,
                success=True,
                chart_data=chart_data
            )

        except Exception as e:
            raise Exception(f"Historical analysis failed: {e}")

    def _run_ma_crossover_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Run a simple moving average crossover backtest"""
        try:
            # Get chart data
            timeframe_map = {"5m": 5, "15m": 15, "1h": 60}
            tf_minutes = timeframe_map.get(config.timeframe, 15)
            chart_data = get_candle_data(self.engine, tf_minutes, config.epic, limit=config.days * 24 * 4)

            if chart_data.empty:
                raise Exception("No chart data available")

            # Calculate moving averages
            fast_ma = config.parameters.get('fast_ma', 10)
            slow_ma = config.parameters.get('slow_ma', 20)

            chart_data['ma_fast'] = chart_data['close'].rolling(window=fast_ma).mean()
            chart_data['ma_slow'] = chart_data['close'].rolling(window=slow_ma).mean()

            # Detect crossovers
            chart_data['ma_cross_up'] = (
                (chart_data['ma_fast'] > chart_data['ma_slow']) &
                (chart_data['ma_fast'].shift(1) <= chart_data['ma_slow'].shift(1))
            )
            chart_data['ma_cross_down'] = (
                (chart_data['ma_fast'] < chart_data['ma_slow']) &
                (chart_data['ma_fast'].shift(1) >= chart_data['ma_slow'].shift(1))
            )

            signals = []
            pip_value = 0.0001 if 'JPY' not in config.epic else 0.01

            # Generate signals
            for i, row in chart_data.iterrows():
                if row['ma_cross_up'] or row['ma_cross_down']:
                    direction = 'BUY' if row['ma_cross_up'] else 'SELL'

                    # Look ahead for performance (simplified)
                    future_data = chart_data.iloc[i+1:i+25] if i+1 < len(chart_data) else pd.DataFrame()
                    if not future_data.empty:
                        if direction == 'BUY':
                            max_profit = (future_data['high'].max() - row['close']) / pip_value
                            max_loss = (row['close'] - future_data['low'].min()) / pip_value
                        else:
                            max_profit = (row['close'] - future_data['low'].min()) / pip_value
                            max_loss = (future_data['high'].max() - row['close']) / pip_value
                    else:
                        max_profit = max_loss = 0

                    signal = {
                        'timestamp': row['start_time'],
                        'epic': config.epic,
                        'signal_type': 'MA_CROSS',
                        'direction': direction,
                        'entry_price': row['close'],
                        'confidence': 0.75,  # Fixed confidence for MA cross
                        'strategy': f"MA({fast_ma},{slow_ma})",
                        'timeframe': config.timeframe,
                        'spread_pips': 1.5,
                        'max_profit_pips': max(max_profit, 0),
                        'max_loss_pips': max(max_loss, 0),
                        'profit_loss_ratio': max_profit / max(max_loss, 1)
                    }
                    signals.append(signal)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(signals)

            return BacktestResult(
                strategy_name=config.strategy_name,
                epic=config.epic,
                timeframe=config.timeframe,
                total_signals=len(signals),
                signals=signals,
                performance_metrics=performance_metrics,
                execution_time=0,
                success=True,
                chart_data=chart_data
            )

        except Exception as e:
            raise Exception(f"MA crossover backtest failed: {e}")

    def _run_breakout_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Run a price breakout backtest"""
        try:
            # Get chart data
            timeframe_map = {"5m": 5, "15m": 15, "1h": 60}
            tf_minutes = timeframe_map.get(config.timeframe, 15)
            chart_data = get_candle_data(self.engine, tf_minutes, config.epic, limit=config.days * 24 * 4)

            if chart_data.empty:
                raise Exception("No chart data available")

            # Calculate breakout levels
            lookback = config.parameters.get('lookback_periods', 20)
            threshold_pips = config.parameters.get('breakout_threshold', 0.5)
            pip_value = 0.0001 if 'JPY' not in config.epic else 0.01
            threshold_price = threshold_pips * pip_value

            chart_data['resistance'] = chart_data['high'].rolling(window=lookback).max()
            chart_data['support'] = chart_data['low'].rolling(window=lookback).min()

            # Detect breakouts
            chart_data['resistance_break'] = chart_data['high'] > (chart_data['resistance'].shift(1) + threshold_price)
            chart_data['support_break'] = chart_data['low'] < (chart_data['support'].shift(1) - threshold_price)

            signals = []

            # Generate signals
            for i, row in chart_data.iterrows():
                signal_direction = None
                signal_type = None

                if row['resistance_break']:
                    signal_direction = 'BUY'
                    signal_type = 'RESISTANCE_BREAK'
                elif row['support_break']:
                    signal_direction = 'SELL'
                    signal_type = 'SUPPORT_BREAK'

                if signal_direction:
                    # Look ahead for performance
                    future_data = chart_data.iloc[i+1:i+25] if i+1 < len(chart_data) else pd.DataFrame()
                    if not future_data.empty:
                        if signal_direction == 'BUY':
                            max_profit = (future_data['high'].max() - row['close']) / pip_value
                            max_loss = (row['close'] - future_data['low'].min()) / pip_value
                        else:
                            max_profit = (row['close'] - future_data['low'].min()) / pip_value
                            max_loss = (future_data['high'].max() - row['close']) / pip_value
                    else:
                        max_profit = max_loss = 0

                    signal = {
                        'timestamp': row['start_time'],
                        'epic': config.epic,
                        'signal_type': signal_type,
                        'direction': signal_direction,
                        'entry_price': row['close'],
                        'confidence': 0.8,  # Higher confidence for breakouts
                        'strategy': f"Breakout({lookback})",
                        'timeframe': config.timeframe,
                        'spread_pips': 1.5,
                        'max_profit_pips': max(max_profit, 0),
                        'max_loss_pips': max(max_loss, 0),
                        'profit_loss_ratio': max_profit / max(max_loss, 1),
                        'resistance_level': row['resistance'],
                        'support_level': row['support']
                    }
                    signals.append(signal)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(signals)

            return BacktestResult(
                strategy_name=config.strategy_name,
                epic=config.epic,
                timeframe=config.timeframe,
                total_signals=len(signals),
                signals=signals,
                performance_metrics=performance_metrics,
                execution_time=0,
                success=True,
                chart_data=chart_data
            )

        except Exception as e:
            raise Exception(f"Breakout backtest failed: {e}")

    def _calculate_performance_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for signals"""
        if not signals:
            return {}

        total_signals = len(signals)
        profitable_signals = [s for s in signals if s.get('max_profit_pips', 0) > s.get('max_loss_pips', 0)]
        win_rate = len(profitable_signals) / total_signals if total_signals > 0 else 0

        avg_confidence = sum(s.get('confidence', 0) for s in signals) / total_signals if total_signals > 0 else 0
        total_profit = sum(s.get('max_profit_pips', 0) for s in signals)
        total_loss = sum(s.get('max_loss_pips', 0) for s in signals)
        avg_profit = total_profit / total_signals if total_signals > 0 else 0
        avg_rr = sum(s.get('profit_loss_ratio', 0) for s in signals) / total_signals if total_signals > 0 else 0

        return {
            'total_signals': total_signals,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'avg_profit_pips': avg_profit,
            'total_profit_pips': total_profit,
            'total_loss_pips': total_loss,
            'avg_risk_reward': avg_rr,
            'profitable_signals': len(profitable_signals),
            'losing_signals': total_signals - len(profitable_signals)
        }


# Global instance
_container_backtest_service = None

def get_container_backtest_service() -> ContainerBacktestService:
    """Get the global container backtest service instance"""
    global _container_backtest_service
    if _container_backtest_service is None:
        _container_backtest_service = ContainerBacktestService()
    return _container_backtest_service