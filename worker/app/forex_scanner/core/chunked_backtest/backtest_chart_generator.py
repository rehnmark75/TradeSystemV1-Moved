"""
BacktestChartGenerator - Generate visual charts with backtest signals

Creates professional candlestick charts with signals plotted,
cumulative P&L curves, and performance statistics.
"""

import io
import base64
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Docker/async
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import mplfinance as mpf
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher


class BacktestChartGenerator:
    """
    Generate visual charts with backtest signals plotted on price data

    Features:
    - Candlestick price chart with signal markers
    - Buy/Sell markers with win/loss coloring
    - Cumulative P&L curve
    - Performance statistics header
    - Export to file or base64
    """

    # Chart configuration
    DPI = 150
    FIGURE_SIZE = (16, 10)

    # Colors
    COLOR_WIN = '#00C853'      # Green for wins
    COLOR_LOSS = '#FF1744'     # Red for losses
    COLOR_BUY = '#2196F3'      # Blue for buy markers
    COLOR_SELL = '#FF9800'     # Orange for sell markers
    COLOR_EQUITY_POS = '#00C853'  # Green for positive equity
    COLOR_EQUITY_NEG = '#FF1744'  # Red for negative equity
    COLOR_EQUITY_LINE = '#1976D2'  # Blue for equity line

    def __init__(self,
                 data_fetcher: Optional[DataFetcher] = None,
                 db_manager: Optional[DatabaseManager] = None,
                 logger: Optional[logging.Logger] = None):

        self.data_fetcher = data_fetcher
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available - chart generation disabled")

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes for database query"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return mapping.get(timeframe, 15)  # Default to 15m if unknown

    def generate_backtest_chart(self,
                                 epic: str,
                                 start_date: datetime,
                                 end_date: datetime,
                                 signals: List[Dict[str, Any]],
                                 strategy: str = 'SMC_SIMPLE',
                                 timeframe: str = '15m',
                                 output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate chart with signals plotted on price data

        Args:
            epic: Currency pair (e.g., 'CS.D.EURUSD.CEEM.IP')
            start_date: Start of backtest period
            end_date: End of backtest period
            signals: List of signal dicts with:
                     - timestamp: datetime
                     - type: 'BUY' or 'SELL'
                     - entry_price: float
                     - pips: float (pips gained/lost)
                     - result: 'win' or 'loss'
            strategy: Strategy name for title
            timeframe: Chart timeframe (default 15m)
            output_path: Save to file (optional)

        Returns:
            If output_path: file path
            Otherwise: base64 encoded PNG string
            Returns None if matplotlib unavailable or error
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Cannot generate chart: matplotlib not available")
            return None

        if not signals:
            self.logger.warning("No signals to plot")
            return None

        try:
            # 1. Fetch candle data
            candles_df = self._fetch_candles(epic, start_date, end_date, timeframe)

            if candles_df is None or candles_df.empty:
                self.logger.error("No candle data available for chart")
                return None

            # 2. Calculate metrics for header
            metrics = self._calculate_chart_metrics(signals)

            # 3. Create figure with subplots
            fig, (ax_price, ax_pnl) = plt.subplots(
                2, 1,
                figsize=self.FIGURE_SIZE,
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )

            # 4. Plot candlesticks
            self._plot_candlesticks(ax_price, candles_df)

            # 5. Plot signals
            self._plot_signals(ax_price, signals, candles_df)

            # 6. Plot cumulative P&L
            self._plot_pnl_curve(ax_pnl, signals)

            # 7. Add title and formatting
            self._add_chart_formatting(
                fig, ax_price, ax_pnl,
                epic, start_date, end_date, strategy, metrics
            )

            # 8. Export
            return self._export_chart(fig, output_path)

        except Exception as e:
            self.logger.error(f"Failed to generate chart: {e}")
            return None

        finally:
            plt.close('all')

    def _fetch_candles(self,
                        epic: str,
                        start_date: datetime,
                        end_date: datetime,
                        timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch candle data for the chart period"""
        try:
            if self.data_fetcher:
                # Use injected data fetcher
                candles_df = self.data_fetcher.fetch_candles(
                    epic, start_date, end_date, timeframe
                )
            elif self.db_manager:
                # Convert timeframe string to minutes (ig_candles uses integer minutes)
                timeframe_minutes = self._timeframe_to_minutes(timeframe)

                # Direct database query - use ig_candles_backtest for backtest charts
                # Uses :name style for SQLAlchemy text()
                query = """
                SELECT
                    start_time as timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM ig_candles_backtest
                WHERE epic = :epic
                  AND timeframe = :timeframe
                  AND start_time BETWEEN :start_date AND :end_date
                ORDER BY start_time
                """
                params = {
                    'epic': epic,
                    'timeframe': timeframe_minutes,
                    'start_date': start_date,
                    'end_date': end_date
                }
                candles_df = self.db_manager.execute_query(query, params)

                if not candles_df.empty:
                    candles_df.set_index('timestamp', inplace=True)
            else:
                self.logger.error("No data source available")
                return None

            return candles_df

        except Exception as e:
            self.logger.error(f"Failed to fetch candles: {e}")
            return None

    def _calculate_chart_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for chart header"""
        total = len(signals)
        wins = sum(1 for s in signals if s.get('pips', 0) > 0 or s.get('result') == 'win')
        losses = sum(1 for s in signals if s.get('pips', 0) < 0 or s.get('result') == 'loss')

        total_pips = sum(s.get('pips', 0) for s in signals)

        win_pips = sum(s.get('pips', 0) for s in signals if s.get('pips', 0) > 0)
        loss_pips = abs(sum(s.get('pips', 0) for s in signals if s.get('pips', 0) < 0))

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        profit_factor = win_pips / loss_pips if loss_pips > 0 else float('inf') if win_pips > 0 else 0

        return {
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'profit_factor': profit_factor
        }

    def _plot_candlesticks(self, ax: plt.Axes, candles_df: pd.DataFrame):
        """Plot candlesticks on the given axes"""
        # Prepare data for candlestick plotting
        dates = mdates.date2num(candles_df.index.to_pydatetime())

        # OHLC data
        opens = candles_df['open'].values
        highs = candles_df['high'].values
        lows = candles_df['low'].values
        closes = candles_df['close'].values

        # Candle width (in days)
        width = 0.0005  # Adjust based on timeframe

        # Plot each candle
        for i in range(len(dates)):
            if closes[i] >= opens[i]:
                # Bullish candle (green)
                color = '#26A69A'
                body_bottom = opens[i]
                body_height = closes[i] - opens[i]
            else:
                # Bearish candle (red)
                color = '#EF5350'
                body_bottom = closes[i]
                body_height = opens[i] - closes[i]

            # Draw wick
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]],
                   color=color, linewidth=0.5)

            # Draw body
            ax.add_patch(Rectangle(
                (dates[i] - width/2, body_bottom),
                width, body_height,
                facecolor=color, edgecolor=color
            ))

        ax.set_ylabel('Price', fontsize=10)

    def _plot_signals(self,
                       ax: plt.Axes,
                       signals: List[Dict[str, Any]],
                       candles_df: pd.DataFrame):
        """Plot buy/sell markers on the chart"""
        for signal in signals:
            timestamp = signal.get('timestamp')
            if timestamp is None:
                continue

            # Convert to matplotlib date
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            date_num = mdates.date2num(timestamp)
            price = signal.get('entry_price', 0)
            pips = signal.get('pips', 0)
            signal_type = signal.get('type', 'BUY')

            # Determine marker style
            if signal_type in ('BUY', 'BULL'):
                marker = '^'  # Up triangle
                marker_color = self.COLOR_WIN if pips > 0 else self.COLOR_LOSS
                y_offset = -10  # Below marker
            else:
                marker = 'v'  # Down triangle
                marker_color = self.COLOR_WIN if pips > 0 else self.COLOR_LOSS
                y_offset = 10  # Above marker

            # Plot marker
            ax.scatter(
                date_num, price,
                marker=marker, s=120,
                c=marker_color, edgecolors='black', linewidths=0.5,
                zorder=5
            )

            # Add pips annotation
            ax.annotate(
                f"{pips:+.1f}",
                (date_num, price),
                textcoords="offset points",
                xytext=(0, y_offset),
                fontsize=7,
                ha='center',
                color=marker_color,
                fontweight='bold'
            )

    def _plot_pnl_curve(self, ax: plt.Axes, signals: List[Dict[str, Any]]):
        """Plot cumulative P&L curve"""
        if not signals:
            return

        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.get('timestamp', datetime.min))

        # Calculate cumulative P&L
        timestamps = []
        cumulative_pips = []
        running_total = 0.0

        for signal in sorted_signals:
            ts = signal.get('timestamp')
            if ts is None:
                continue

            if isinstance(ts, str):
                ts = pd.to_datetime(ts)

            running_total += signal.get('pips', 0)
            timestamps.append(ts)
            cumulative_pips.append(running_total)

        if not timestamps:
            return

        # Convert to matplotlib dates
        date_nums = mdates.date2num(timestamps)

        # Create arrays for fill_between
        zeros = np.zeros(len(cumulative_pips))

        # Fill positive area (green)
        ax.fill_between(
            date_nums, cumulative_pips, zeros,
            where=[p >= 0 for p in cumulative_pips],
            color=self.COLOR_EQUITY_POS, alpha=0.3,
            interpolate=True
        )

        # Fill negative area (red)
        ax.fill_between(
            date_nums, cumulative_pips, zeros,
            where=[p < 0 for p in cumulative_pips],
            color=self.COLOR_EQUITY_NEG, alpha=0.3,
            interpolate=True
        )

        # Plot the equity line
        ax.plot(date_nums, cumulative_pips, color=self.COLOR_EQUITY_LINE, linewidth=1.5)

        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Labels
        ax.set_ylabel('Cumulative Pips', fontsize=10)

        # Final P&L annotation
        if cumulative_pips:
            final_pips = cumulative_pips[-1]
            color = self.COLOR_WIN if final_pips >= 0 else self.COLOR_LOSS
            ax.annotate(
                f"{final_pips:+.1f} pips",
                (date_nums[-1], final_pips),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=9,
                color=color,
                fontweight='bold'
            )

    def _add_chart_formatting(self,
                               fig: plt.Figure,
                               ax_price: plt.Axes,
                               ax_pnl: plt.Axes,
                               epic: str,
                               start_date: datetime,
                               end_date: datetime,
                               strategy: str,
                               metrics: Dict[str, Any]):
        """Add title, labels, and formatting"""
        # Extract pair name from epic
        pair_name = epic.split('.')[-2] if '.' in epic else epic

        # Title with metrics
        win_rate_pct = metrics['win_rate'] * 100
        total_pips = metrics['total_pips']
        pips_color = self.COLOR_WIN if total_pips >= 0 else self.COLOR_LOSS

        title = (
            f"{pair_name} Backtest Results: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
            f"Strategy: {strategy} | Signals: {metrics['total_signals']} | "
            f"Win Rate: {win_rate_pct:.1f}% | Total: {total_pips:+.1f} pips"
        )

        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

        # X-axis date formatting
        ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_pnl.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_pnl.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Grid
        ax_price.grid(True, alpha=0.3)
        ax_pnl.grid(True, alpha=0.3)

        # Legend for price chart
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=self.COLOR_WIN,
                   markersize=10, label='Buy (Win)', markeredgecolor='black'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=self.COLOR_LOSS,
                   markersize=10, label='Buy (Loss)', markeredgecolor='black'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=self.COLOR_WIN,
                   markersize=10, label='Sell (Win)', markeredgecolor='black'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=self.COLOR_LOSS,
                   markersize=10, label='Sell (Loss)', markeredgecolor='black'),
        ]
        ax_price.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # Tight layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _export_chart(self, fig: plt.Figure, output_path: Optional[str] = None) -> str:
        """Export chart to file or base64"""
        if output_path:
            # Save to file
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                output_path,
                format='png',
                dpi=self.DPI,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )

            self.logger.info(f"Chart saved to {output_path}")
            return output_path

        else:
            # Return base64
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format='png',
                dpi=self.DPI,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )

            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            self.logger.info(f"Chart generated as base64 ({len(image_base64)} bytes)")
            return image_base64

    def generate_quick_chart(self,
                              signals: List[Dict[str, Any]],
                              output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate a simple chart without fetching candle data

        Just plots the equity curve from signals - useful when candle
        data isn't available or for quick visualization.

        Args:
            signals: List of signal dicts
            output_path: Save to file (optional)

        Returns:
            File path or base64 string
        """
        if not MATPLOTLIB_AVAILABLE or not signals:
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Sort and calculate cumulative P&L
            sorted_signals = sorted(signals, key=lambda s: s.get('timestamp', datetime.min))

            timestamps = []
            cumulative_pips = []
            running_total = 0.0

            for signal in sorted_signals:
                ts = signal.get('timestamp')
                if ts is None:
                    continue

                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)

                running_total += signal.get('pips', 0)
                timestamps.append(ts)
                cumulative_pips.append(running_total)

            if not timestamps:
                return None

            # Plot
            date_nums = mdates.date2num(timestamps)
            zeros = np.zeros(len(cumulative_pips))

            ax.fill_between(date_nums, cumulative_pips, zeros,
                           where=[p >= 0 for p in cumulative_pips],
                           color=self.COLOR_EQUITY_POS, alpha=0.3)
            ax.fill_between(date_nums, cumulative_pips, zeros,
                           where=[p < 0 for p in cumulative_pips],
                           color=self.COLOR_EQUITY_NEG, alpha=0.3)
            ax.plot(date_nums, cumulative_pips, color=self.COLOR_EQUITY_LINE, linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

            # Markers for each trade
            for i, signal in enumerate(sorted_signals):
                ts = signal.get('timestamp')
                if ts is None:
                    continue
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)

                date_num = mdates.date2num(ts)
                pips = signal.get('pips', 0)
                color = self.COLOR_WIN if pips > 0 else self.COLOR_LOSS
                marker = '^' if signal.get('type') in ('BUY', 'BULL') else 'v'

                ax.scatter(date_num, cumulative_pips[i], marker=marker, s=50,
                          c=color, edgecolors='black', linewidths=0.3, zorder=5)

            # Formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Cumulative Pips', fontsize=11)
            ax.set_title(f'Equity Curve ({len(signals)} trades, {cumulative_pips[-1]:+.1f} pips)',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            fig.tight_layout()

            return self._export_chart(fig, output_path)

        except Exception as e:
            self.logger.error(f"Failed to generate quick chart: {e}")
            return None

        finally:
            plt.close('all')


# Factory function
def create_backtest_chart_generator(
        data_fetcher: Optional[DataFetcher] = None,
        db_manager: Optional[DatabaseManager] = None,
        logger: Optional[logging.Logger] = None) -> BacktestChartGenerator:
    """Create BacktestChartGenerator instance"""
    return BacktestChartGenerator(data_fetcher, db_manager, logger)
