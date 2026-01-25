"""
Stock Chart Generator

Generates professional candlestick charts for Claude AI vision analysis.
Uses mplfinance to create clean, informative charts with key levels marked.
"""

import io
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class StockChartGenerator:
    """
    Generates professional candlestick charts for Claude vision analysis.

    Features:
    - OHLC candlestick charts with volume
    - Moving average overlays (SMA 20, 50, 200)
    - Entry/Stop/Target level markers
    - Support/Resistance zones
    - SMC annotations (order blocks, BOS levels)
    - Clean, high-contrast style optimized for AI vision

    Usage:
        generator = StockChartGenerator()
        chart_base64 = await generator.generate_signal_chart(
            ticker='AAPL',
            candles=candle_data,
            signal=signal_data,
            smc_data=smc_data
        )
    """

    # Chart configuration
    DEFAULT_DAYS = 60  # Show 60 days of history
    CHART_SIZE = (12, 8)  # Width x Height in inches
    DPI = 150  # Resolution for clear details

    # Color scheme optimized for vision clarity
    COLORS = {
        'up_candle': '#26a69a',      # Green for up candles
        'down_candle': '#ef5350',    # Red for down candles
        'volume_up': '#26a69a80',    # Semi-transparent green
        'volume_down': '#ef535080',  # Semi-transparent red
        'sma_20': '#2196f3',         # Blue
        'sma_50': '#ff9800',         # Orange
        'sma_200': '#9c27b0',        # Purple
        'entry': '#00e676',          # Bright green
        'stop': '#ff1744',           # Bright red
        'target': '#00b0ff',         # Bright blue
        'support': '#4caf50',        # Green zone
        'resistance': '#f44336',     # Red zone
        'order_block': '#ffeb3b',    # Yellow
    }

    def __init__(self, db_manager=None):
        """
        Initialize chart generator.

        Args:
            db_manager: Optional database manager for fetching candle data
        """
        self.db = db_manager

        if not MPLFINANCE_AVAILABLE:
            logger.warning("mplfinance not installed - chart generation disabled")

    @property
    def is_available(self) -> bool:
        """Check if chart generation is available"""
        return MPLFINANCE_AVAILABLE

    async def generate_signal_chart(
        self,
        ticker: str,
        candles: List[Dict[str, Any]] = None,
        signal: Dict[str, Any] = None,
        smc_data: Dict[str, Any] = None,
        technical_data: Dict[str, Any] = None,
        days: int = None
    ) -> Optional[str]:
        """
        Generate a chart for a stock signal.

        Args:
            ticker: Stock ticker symbol
            candles: List of OHLCV candle dicts (or fetch from DB if None)
            signal: Signal data with entry, stop, targets
            smc_data: SMC analysis data (order blocks, BOS, etc.)
            technical_data: Technical indicators (SMAs, etc.)
            days: Number of days to show (default: 60)

        Returns:
            Base64-encoded PNG image string, or None if generation fails
        """
        if not self.is_available:
            logger.warning("Chart generation not available - mplfinance not installed")
            return None

        days = days or self.DEFAULT_DAYS

        # Fetch candles if not provided
        if candles is None and self.db is not None:
            candles = await self._fetch_candles(ticker, days)

        if not candles or len(candles) < 10:
            logger.warning(f"Insufficient candle data for {ticker}: {len(candles) if candles else 0} candles")
            return None

        try:
            # Convert to DataFrame
            df = self._prepare_dataframe(candles)

            # Build chart components
            addplots = []
            hlines = {}

            # Add moving averages if we have enough data
            if len(df) >= 20:
                df['SMA20'] = df['Close'].rolling(window=20).mean()
                addplots.append(mpf.make_addplot(
                    df['SMA20'], color=self.COLORS['sma_20'],
                    width=1.0, label='SMA 20'
                ))

            if len(df) >= 50:
                df['SMA50'] = df['Close'].rolling(window=50).mean()
                addplots.append(mpf.make_addplot(
                    df['SMA50'], color=self.COLORS['sma_50'],
                    width=1.0, label='SMA 50'
                ))

            if len(df) >= 200 and len(df) > 200:
                df['SMA200'] = df['Close'].rolling(window=200).mean()
                addplots.append(mpf.make_addplot(
                    df['SMA200'], color=self.COLORS['sma_200'],
                    width=1.5, label='SMA 200'
                ))

            # Add signal levels
            if signal:
                hlines = self._build_signal_hlines(signal, df)

            # Create custom style
            mc = mpf.make_marketcolors(
                up=self.COLORS['up_candle'],
                down=self.COLORS['down_candle'],
                edge={'up': self.COLORS['up_candle'], 'down': self.COLORS['down_candle']},
                wick={'up': self.COLORS['up_candle'], 'down': self.COLORS['down_candle']},
                volume={'up': self.COLORS['volume_up'], 'down': self.COLORS['volume_down']}
            )

            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#e0e0e0',
                facecolor='white',
                figcolor='white',
                rc={'font.size': 10}
            )

            # Generate chart to buffer
            buf = io.BytesIO()

            # Build title
            title = f"{ticker}"
            if signal:
                direction = signal.get('signal_type', 'BUY')
                scanner = signal.get('scanner_name', '')
                tier = signal.get('quality_tier', '')
                score = signal.get('composite_score', 0)
                title = f"{ticker} - {direction} Signal ({scanner}) | {tier} ({score}/100)"

            # Plot
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=style,
                volume=True,
                addplot=addplots if addplots else None,
                hlines=hlines if hlines else None,
                title=title,
                figsize=self.CHART_SIZE,
                returnfig=True,
                warn_too_much_data=1000
            )

            # Add legend for signal levels
            if signal:
                self._add_signal_legend(fig, axes[0], signal)

            # Add SMC annotations if available
            if smc_data:
                self._add_smc_annotations(axes[0], df, smc_data)

            # Save to buffer
            fig.savefig(buf, format='png', dpi=self.DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            # Encode to base64
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            logger.info(f"Generated chart for {ticker}: {len(image_base64)} bytes (base64)")

            return image_base64

        except Exception as e:
            logger.error(f"Error generating chart for {ticker}: {e}")
            return None

    def _prepare_dataframe(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert candle list to mplfinance-compatible DataFrame"""

        # Create DataFrame
        df = pd.DataFrame(candles)

        # Normalize column names
        column_map = {
            'timestamp': 'Date',
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Ensure we have required columns
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Set Date as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # Sort by date
        df = df.sort_index()

        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop any rows with NaN in OHLC
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        return df

    def _build_signal_hlines(
        self,
        signal: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build horizontal lines for signal levels"""

        hlines_dict = {'hlines': [], 'colors': [], 'linestyle': [], 'linewidths': []}

        # Get price range for filtering
        price_min = df['Low'].min()
        price_max = df['High'].max()
        price_range = price_max - price_min

        # Entry level
        entry = signal.get('entry_price')
        if entry and price_min - price_range * 0.1 <= entry <= price_max + price_range * 0.1:
            hlines_dict['hlines'].append(float(entry))
            hlines_dict['colors'].append(self.COLORS['entry'])
            hlines_dict['linestyle'].append('--')
            hlines_dict['linewidths'].append(2)

        # Stop loss
        stop = signal.get('stop_loss')
        if stop and price_min - price_range * 0.2 <= stop <= price_max + price_range * 0.2:
            hlines_dict['hlines'].append(float(stop))
            hlines_dict['colors'].append(self.COLORS['stop'])
            hlines_dict['linestyle'].append('--')
            hlines_dict['linewidths'].append(2)

        # Target 1
        target1 = signal.get('take_profit_1')
        if target1 and price_min - price_range * 0.2 <= target1 <= price_max + price_range * 0.3:
            hlines_dict['hlines'].append(float(target1))
            hlines_dict['colors'].append(self.COLORS['target'])
            hlines_dict['linestyle'].append('--')
            hlines_dict['linewidths'].append(1.5)

        # Target 2
        target2 = signal.get('take_profit_2')
        if target2 and price_min - price_range * 0.2 <= target2 <= price_max + price_range * 0.4:
            hlines_dict['hlines'].append(float(target2))
            hlines_dict['colors'].append(self.COLORS['target'])
            hlines_dict['linestyle'].append(':')
            hlines_dict['linewidths'].append(1.5)

        return hlines_dict if hlines_dict['hlines'] else None

    def _add_signal_legend(
        self,
        fig,
        ax,
        signal: Dict[str, Any]
    ) -> None:
        """Add legend showing signal levels"""

        from matplotlib.lines import Line2D

        legend_elements = []

        entry = signal.get('entry_price')
        if entry:
            legend_elements.append(
                Line2D([0], [0], color=self.COLORS['entry'], linestyle='--',
                      linewidth=2, label=f"Entry: ${entry:.2f}")
            )

        stop = signal.get('stop_loss')
        if stop:
            legend_elements.append(
                Line2D([0], [0], color=self.COLORS['stop'], linestyle='--',
                      linewidth=2, label=f"Stop: ${stop:.2f}")
            )

        target1 = signal.get('take_profit_1')
        if target1:
            legend_elements.append(
                Line2D([0], [0], color=self.COLORS['target'], linestyle='--',
                      linewidth=1.5, label=f"Target 1: ${target1:.2f}")
            )

        target2 = signal.get('take_profit_2')
        if target2:
            legend_elements.append(
                Line2D([0], [0], color=self.COLORS['target'], linestyle=':',
                      linewidth=1.5, label=f"Target 2: ${target2:.2f}")
            )

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                     framealpha=0.9, facecolor='white')

    def _add_smc_annotations(
        self,
        ax,
        df: pd.DataFrame,
        smc_data: Dict[str, Any]
    ) -> None:
        """Add SMC-related annotations to chart"""

        try:
            # Add swing high/low markers
            swing_high = smc_data.get('swing_high')
            swing_low = smc_data.get('swing_low')

            if swing_high:
                ax.axhline(y=swing_high, color='#9e9e9e', linestyle=':',
                          linewidth=1, alpha=0.7)
                ax.annotate(f'Swing H: ${swing_high:.2f}',
                           xy=(df.index[-1], swing_high),
                           fontsize=8, alpha=0.8)

            if swing_low:
                ax.axhline(y=swing_low, color='#9e9e9e', linestyle=':',
                          linewidth=1, alpha=0.7)
                ax.annotate(f'Swing L: ${swing_low:.2f}',
                           xy=(df.index[-1], swing_low),
                           fontsize=8, alpha=0.8)

            # Add order block zone if available
            ob_price = smc_data.get('nearest_ob_price')
            ob_type = smc_data.get('nearest_ob_type')
            if ob_price and ob_type:
                color = self.COLORS['order_block']
                ax.axhline(y=ob_price, color=color, linestyle='-.',
                          linewidth=1.5, alpha=0.6)
                ax.annotate(f'{ob_type} OB', xy=(df.index[-1], ob_price),
                           fontsize=8, color=color, alpha=0.8)

            # Add BOS level if recent
            bos_price = smc_data.get('last_bos_price')
            bos_type = smc_data.get('last_bos_type')
            if bos_price and bos_type:
                color = self.COLORS['up_candle'] if 'bull' in str(bos_type).lower() else self.COLORS['down_candle']
                ax.axhline(y=bos_price, color=color, linestyle='-',
                          linewidth=1, alpha=0.4)

        except Exception as e:
            logger.warning(f"Error adding SMC annotations: {e}")

    async def _fetch_candles(
        self,
        ticker: str,
        days: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch candle data from database"""

        if not self.db:
            return None

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles_synthesized
            WHERE ticker = $1 AND timeframe = '1d'
            ORDER BY timestamp DESC
            LIMIT $2
        """

        try:
            rows = await self.db.fetch(query, ticker, days)
            if rows:
                # Convert to list of dicts and reverse to chronological order
                candles = [dict(row) for row in rows]
                candles.reverse()
                return candles
        except Exception as e:
            logger.error(f"Error fetching candles for {ticker}: {e}")

        return None

    async def generate_comparison_chart(
        self,
        tickers: List[str],
        days: int = 60
    ) -> Optional[str]:
        """
        Generate a comparison chart showing multiple tickers normalized.
        Useful for sector rotation analysis.

        Args:
            tickers: List of ticker symbols to compare
            days: Number of days to show

        Returns:
            Base64-encoded PNG image
        """
        if not self.is_available or not self.db:
            return None

        if len(tickers) < 2:
            return None

        try:
            # Fetch data for all tickers
            all_data = {}
            for ticker in tickers[:5]:  # Limit to 5 for clarity
                candles = await self._fetch_candles(ticker, days)
                if candles and len(candles) >= 20:
                    df = self._prepare_dataframe(candles)
                    # Normalize to percentage change from start
                    df['Normalized'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                    all_data[ticker] = df['Normalized']

            if len(all_data) < 2:
                return None

            # Create comparison DataFrame
            comparison_df = pd.DataFrame(all_data)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))

            colors = ['#2196f3', '#f44336', '#4caf50', '#ff9800', '#9c27b0']
            for i, (ticker, data) in enumerate(comparison_df.items()):
                ax.plot(data.index, data.values, label=ticker,
                       color=colors[i % len(colors)], linewidth=1.5)

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_title('Price Performance Comparison (% Change)')
            ax.set_ylabel('% Change')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.DPI, bbox_inches='tight',
                       facecolor='white')
            plt.close(fig)

            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        except Exception as e:
            logger.error(f"Error generating comparison chart: {e}")
            return None
