"""
Forex Chart Generator

Generates professional multi-timeframe candlestick charts for Claude AI vision analysis.
Adapted from stock_chart_generator.py for forex-specific SMC analysis.

Features:
- Multi-timeframe panels (4H bias, 15m trigger, 5m entry)
- SMC-specific markers (swing levels, Fibonacci zones, entry/SL/TP)
- EMA 50 trend bias overlay
- Clean, high-contrast style optimized for AI vision
"""

import io
import base64
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

try:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ForexChartGenerator:
    """
    Generates professional forex candlestick charts for Claude vision analysis.

    Features:
    - Multi-timeframe analysis (4H/15m/5m)
    - SMC markers (swing levels, Fib zones, order blocks)
    - Entry/Stop/Target level markers
    - EMA 50 trend bias on 4H
    - Clean, high-contrast style optimized for AI vision

    Usage:
        generator = ForexChartGenerator()
        chart_base64 = generator.generate_signal_chart(
            epic='CS.D.EURUSD.MINI.IP',
            candles={'4h': df_4h, '15m': df_15m, '5m': df_5m},
            signal=signal_data,
            smc_data=smc_data
        )
    """

    # Chart configuration
    DEFAULT_BARS = {'4h': 100, '15m': 100, '5m': 60}
    CHART_SIZE = (16, 12)  # Width x Height in inches
    DPI = 150  # Resolution for clear details

    # Color scheme optimized for vision clarity
    COLORS = {
        'up_candle': '#26a69a',      # Green for up candles
        'down_candle': '#ef5350',    # Red for down candles
        'volume_up': '#26a69a80',    # Semi-transparent green
        'volume_down': '#ef535080',  # Semi-transparent red
        'ema_9': '#ff9800',          # Orange for EMA 9
        'ema_21': '#2196f3',         # Blue for EMA 21
        'ema_50': '#9c27b0',         # Purple for EMA 50
        'ema_200': '#673ab7',        # Deep purple for EMA 200
        'entry': '#00e676',          # Bright green
        'stop': '#ff1744',           # Bright red
        'target': '#00b0ff',         # Bright blue
        'swing_high': '#ff9800',     # Orange for swing high
        'swing_low': '#2196f3',      # Blue for swing low
        'fib_zone': '#ffeb3b40',     # Semi-transparent yellow for Fib zone
        'order_block': '#ffeb3b',    # Yellow for order blocks
        'bos_bull': '#00e676',       # Green for bullish BOS
        'bos_bear': '#ff1744',       # Red for bearish BOS
        'support': '#4caf50',        # Green for support levels
        'resistance': '#f44336',     # Red for resistance levels
        'entry_type_bg': '#000000cc', # Semi-transparent black for entry type label
    }

    def __init__(self, db_manager=None, data_fetcher=None):
        """
        Initialize chart generator.

        Args:
            db_manager: Optional database manager
            data_fetcher: Optional DataFetcher for getting candle data
        """
        self.db = db_manager
        self.data_fetcher = data_fetcher

        if not MPLFINANCE_AVAILABLE:
            logger.warning("mplfinance not installed - chart generation disabled")

    @property
    def is_available(self) -> bool:
        """Check if chart generation is available"""
        return MPLFINANCE_AVAILABLE

    def generate_signal_chart(
        self,
        epic: str,
        candles: Dict[str, pd.DataFrame],
        signal: Dict[str, Any],
        smc_data: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Generate a multi-timeframe chart for a forex signal.

        Args:
            epic: Currency pair epic (e.g., 'CS.D.EURUSD.MINI.IP')
            candles: Dict of DataFrames {'4h': df_4h, '15m': df_15m, '5m': df_5m}
            signal: Signal data with entry, stop, take_profit, etc.
            smc_data: SMC analysis data (swing levels, Fib zones, etc.)

        Returns:
            Base64-encoded PNG image string, or None if generation fails
        """
        if not self.is_available:
            logger.warning("Chart generation not available - mplfinance not installed")
            return None

        # Validate inputs
        if not candles:
            logger.warning(f"No candle data provided for {epic}")
            return None

        # Get available timeframes
        available_tfs = [tf for tf in ['4h', '15m', '5m'] if tf in candles and candles[tf] is not None and len(candles[tf]) >= 10]

        if not available_tfs:
            logger.warning(f"Insufficient candle data for {epic}")
            return None

        try:
            # Extract pair from epic
            pair = self._extract_pair(epic)

            # Create figure with subplots for each timeframe
            num_panels = len(available_tfs)
            fig, axes = plt.subplots(num_panels, 1, figsize=(self.CHART_SIZE[0], self.CHART_SIZE[1] * num_panels / 3))

            if num_panels == 1:
                axes = [axes]

            # Build title
            direction = signal.get('signal_type', signal.get('signal', 'UNKNOWN'))
            confidence = signal.get('confidence_score', 0)
            strategy = signal.get('strategy', 'SMC')
            rr_ratio = signal.get('rr_ratio', 0)

            fig.suptitle(
                f"{pair} - {direction} Signal | {strategy} | Confidence: {confidence:.1%} | R:R {rr_ratio:.1f}",
                fontsize=14, fontweight='bold', y=0.98
            )

            # Plot each timeframe
            for i, tf in enumerate(available_tfs):
                df = self._prepare_dataframe(candles[tf], tf)

                if df is None or len(df) < 10:
                    continue

                ax = axes[i]

                # Plot candlesticks manually (since we're using regular matplotlib)
                self._plot_candlesticks(ax, df, tf)

                # Extract strategy_indicators for enhanced charting
                strategy_indicators = signal.get('strategy_indicators', {}) if signal else {}

                # Add EMA 50 on 4H
                if tf == '4h' and len(df) >= 50:
                    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
                    ax.plot(range(len(df)), df['EMA50'].values, color=self.COLORS['ema_50'],
                           linewidth=1.5, label='EMA 50', alpha=0.9)

                # Add EMA 9/21 on 15m for swing break context and micro-structure
                if tf == '15m' and len(df) >= 21:
                    self._add_ema_stack(ax, df, strategy_indicators)

                # Add Support/Resistance levels on 15m (swing break timeframe)
                if tf == '15m' and strategy_indicators:
                    self._add_support_resistance_levels(ax, df, strategy_indicators)

                # Add signal levels on 15m (swing break timeframe) for context
                if tf == '15m' and signal:
                    self._add_signal_levels(ax, df, signal)

                # Add signal levels on entry timeframe (5m)
                if tf == '5m' and signal:
                    self._add_signal_levels(ax, df, signal)
                    # Add entry type annotation on 5m (entry timeframe)
                    self._add_entry_type_annotation(ax, df, signal, strategy_indicators)

                # Add SMC annotations
                if smc_data and tf in ['15m', '5m']:
                    self._add_smc_annotations(ax, df, signal, smc_data, tf)

                # Add Fibonacci zone on 5m
                if tf == '5m' and signal:
                    self._add_fib_zone(ax, df, signal)

                # Set labels and title
                ax.set_title(f"{tf.upper()} Timeframe", fontsize=11, fontweight='bold', loc='left')
                ax.set_ylabel('Price', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', labelsize=8)

                # Add legend
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            # Encode to base64
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            logger.info(f"Generated forex chart for {pair}: {len(image_base64)} bytes (base64)")

            return image_base64

        except Exception as e:
            logger.error(f"Error generating chart for {epic}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _extract_pair(self, epic: str) -> str:
        """Extract currency pair from epic"""
        try:
            # Format: CS.D.EURUSD.MINI.IP -> EURUSD
            parts = epic.split('.')
            if len(parts) >= 3:
                return parts[2]
            return epic
        except Exception:
            return epic

    def _prepare_dataframe(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Prepare DataFrame for charting"""
        try:
            if df is None or df.empty:
                return None

            # Make a copy
            df = df.copy()

            # Normalize column names
            column_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'volume': 'Volume', 'ltv': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'
            }

            for old_col, new_col in column_map.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]

            # Ensure required columns exist
            required = ['Open', 'High', 'Low', 'Close']
            for col in required:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in {timeframe} data")
                    return None

            # Ensure numeric types
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop NaN rows
            df = df.dropna(subset=required)

            # Limit data to most recent bars
            max_bars = self.DEFAULT_BARS.get(timeframe, 60)
            if len(df) > max_bars:
                df = df.tail(max_bars)

            # Reset index for plotting
            df = df.reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error preparing dataframe for {timeframe}: {e}")
            return None

    def _plot_candlesticks(self, ax, df: pd.DataFrame, timeframe: str) -> None:
        """Plot candlesticks manually on axis"""
        try:
            width = 0.6
            width2 = 0.1

            for i in range(len(df)):
                row = df.iloc[i]
                open_price = row['Open']
                close_price = row['Close']
                high_price = row['High']
                low_price = row['Low']

                # Determine color
                if close_price >= open_price:
                    color = self.COLORS['up_candle']
                    body_bottom = open_price
                    body_height = close_price - open_price
                else:
                    color = self.COLORS['down_candle']
                    body_bottom = close_price
                    body_height = open_price - close_price

                # Draw wick (high-low line)
                ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)

                # Draw body
                if body_height > 0:
                    ax.bar(i, body_height, bottom=body_bottom, width=width,
                          color=color, edgecolor=color, linewidth=0.5)
                else:
                    # Doji - draw a horizontal line
                    ax.plot([i - width/2, i + width/2], [close_price, close_price],
                           color=color, linewidth=1)

            # Set x-axis limits
            ax.set_xlim(-1, len(df))

        except Exception as e:
            logger.error(f"Error plotting candlesticks: {e}")

    def _add_signal_levels(self, ax, df: pd.DataFrame, signal: Dict[str, Any]) -> None:
        """Add entry, stop loss, and take profit levels"""
        try:
            price_range = df['High'].max() - df['Low'].min()
            x_max = len(df) - 1

            # Entry level
            entry = signal.get('entry_price')
            if entry:
                ax.axhline(y=entry, color=self.COLORS['entry'], linestyle='--',
                          linewidth=2, label=f"Entry: {entry:.5f}")

            # Stop loss
            stop = signal.get('stop_loss')
            if stop:
                ax.axhline(y=stop, color=self.COLORS['stop'], linestyle='--',
                          linewidth=2, label=f"SL: {stop:.5f}")

            # Take profit
            tp = signal.get('take_profit')
            if tp:
                ax.axhline(y=tp, color=self.COLORS['target'], linestyle='--',
                          linewidth=2, label=f"TP: {tp:.5f}")

        except Exception as e:
            logger.warning(f"Error adding signal levels: {e}")

    def _add_smc_annotations(
        self,
        ax,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        smc_data: Dict[str, Any],
        timeframe: str
    ) -> None:
        """Add SMC-related annotations (swing levels, BOS, etc.)"""
        try:
            x_max = len(df) - 1

            # Swing level (the level that was broken)
            swing_level = signal.get('swing_level')
            if swing_level:
                ax.axhline(y=swing_level, color=self.COLORS['swing_high'],
                          linestyle=':', linewidth=1.5, alpha=0.8)
                ax.annotate(f'Swing Break: {swing_level:.5f}',
                           xy=(x_max * 0.7, swing_level),
                           fontsize=8, color=self.COLORS['swing_high'],
                           fontweight='bold', alpha=0.9)

            # Opposite swing (for SL placement)
            opposite_swing = signal.get('opposite_swing')
            if opposite_swing and timeframe == '5m':
                ax.axhline(y=opposite_swing, color=self.COLORS['swing_low'],
                          linestyle=':', linewidth=1.5, alpha=0.8)
                ax.annotate(f'Opposite Swing: {opposite_swing:.5f}',
                           xy=(x_max * 0.7, opposite_swing),
                           fontsize=8, color=self.COLORS['swing_low'],
                           fontweight='bold', alpha=0.9)

            # EMA value on 4H (from signal data)
            ema_value = signal.get('ema_value')
            if ema_value and timeframe == '15m':
                ax.axhline(y=ema_value, color=self.COLORS['ema_50'],
                          linestyle='-', linewidth=1.5, alpha=0.6)
                ax.annotate(f'4H EMA50: {ema_value:.5f}',
                           xy=(x_max * 0.1, ema_value),
                           fontsize=8, color=self.COLORS['ema_50'],
                           fontweight='bold', alpha=0.9)

        except Exception as e:
            logger.warning(f"Error adding SMC annotations: {e}")

    def _add_ema_stack(self, ax, df: pd.DataFrame, strategy_indicators: Dict[str, Any]) -> None:
        """Add EMA 9/21 lines on 5m chart for micro-structure analysis"""
        try:
            # Try to get EMA values from strategy_indicators first
            dataframe_analysis = strategy_indicators.get('dataframe_analysis', {})
            ema_data = dataframe_analysis.get('ema_data', {})

            # Calculate EMAs from dataframe if not enough data in indicators
            if len(df) >= 9:
                df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
                ax.plot(range(len(df)), df['EMA9'].values, color=self.COLORS['ema_9'],
                       linewidth=1.2, label='EMA 9', alpha=0.8)

            if len(df) >= 21:
                df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
                ax.plot(range(len(df)), df['EMA21'].values, color=self.COLORS['ema_21'],
                       linewidth=1.2, label='EMA 21', alpha=0.8)

        except Exception as e:
            logger.warning(f"Error adding EMA stack: {e}")

    def _add_entry_type_annotation(
        self,
        ax,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        strategy_indicators: Dict[str, Any]
    ) -> None:
        """Add entry type (PULLBACK/MOMENTUM) annotation with pullback depth"""
        try:
            # Get entry type from signal or strategy_indicators
            tier3_entry = strategy_indicators.get('tier3_entry', {})
            entry_type = tier3_entry.get('entry_type') or signal.get('entry_type', 'UNKNOWN')
            pullback_depth = tier3_entry.get('pullback_depth', signal.get('pullback_depth', 0))
            in_optimal_zone = tier3_entry.get('in_optimal_zone', signal.get('in_optimal_zone', False))
            volume_confirmed = strategy_indicators.get('tier2_swing', {}).get('volume_confirmed', False)

            if not entry_type or entry_type == 'UNKNOWN':
                return

            # Position annotation in top-right corner
            x_pos = len(df) - 1
            y_range = df['High'].max() - df['Low'].min()
            y_pos = df['High'].max() - (y_range * 0.05)

            # Build annotation text
            depth_pct = abs(pullback_depth * 100)
            if entry_type == 'PULLBACK':
                zone_status = "✓ Optimal" if in_optimal_zone else "Outside Zone"
                annotation_text = f"⬇ {entry_type}\nDepth: {depth_pct:.1f}%\n{zone_status}"
                bg_color = '#1565c0'  # Blue for pullback
            else:  # MOMENTUM
                annotation_text = f"⚡ {entry_type}\nExtension: {depth_pct:.1f}%"
                bg_color = '#ff6f00'  # Orange for momentum

            # Add volume status
            vol_icon = "📊 Vol ✓" if volume_confirmed else "📊 Vol ✗"
            annotation_text += f"\n{vol_icon}"

            # Draw annotation box
            ax.annotate(
                annotation_text,
                xy=(x_pos, y_pos),
                fontsize=9,
                fontweight='bold',
                color='white',
                ha='right',
                va='top',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor=bg_color,
                    edgecolor='white',
                    alpha=0.9
                )
            )

        except Exception as e:
            logger.warning(f"Error adding entry type annotation: {e}")

    def _add_support_resistance_levels(
        self,
        ax,
        df: pd.DataFrame,
        strategy_indicators: Dict[str, Any]
    ) -> None:
        """Add support and resistance horizontal lines from strategy_indicators"""
        try:
            # Get S/R data from dataframe_analysis
            dataframe_analysis = strategy_indicators.get('dataframe_analysis', {})
            sr_data = dataframe_analysis.get('sr_data', {})

            if not sr_data:
                return

            x_max = len(df) - 1
            current_price = df['Close'].iloc[-1]

            # Support level
            support = sr_data.get('nearest_support')
            support_dist = sr_data.get('distance_to_support_pips', 0)
            if support and support < current_price:
                ax.axhline(y=support, color=self.COLORS['support'],
                          linestyle='-', linewidth=1.5, alpha=0.7)
                ax.annotate(
                    f"S: {support:.5f} ({support_dist:.1f}p)",
                    xy=(x_max * 0.02, support),
                    fontsize=8,
                    color=self.COLORS['support'],
                    fontweight='bold',
                    va='bottom',
                    alpha=0.9
                )

            # Resistance level
            resistance = sr_data.get('nearest_resistance')
            resistance_dist = sr_data.get('distance_to_resistance_pips', 0)
            if resistance and resistance > current_price:
                ax.axhline(y=resistance, color=self.COLORS['resistance'],
                          linestyle='-', linewidth=1.5, alpha=0.7)
                ax.annotate(
                    f"R: {resistance:.5f} ({resistance_dist:.1f}p)",
                    xy=(x_max * 0.02, resistance),
                    fontsize=8,
                    color=self.COLORS['resistance'],
                    fontweight='bold',
                    va='top',
                    alpha=0.9
                )

        except Exception as e:
            logger.warning(f"Error adding S/R levels: {e}")

    def _add_fib_zone(self, ax, df: pd.DataFrame, signal: Dict[str, Any]) -> None:
        """Add Fibonacci optimal zone shading (38.2% - 61.8%)"""
        try:
            swing_level = signal.get('swing_level')
            opposite_swing = signal.get('opposite_swing')

            if not swing_level or not opposite_swing:
                return

            # Calculate Fib levels
            swing_range = abs(swing_level - opposite_swing)
            direction = signal.get('signal_type', signal.get('signal', '')).upper()

            if direction in ['BULL', 'BUY']:
                # For bullish: Fib zone is between 38.2% and 61.8% retracement from high
                fib_382 = swing_level - (swing_range * 0.382)
                fib_618 = swing_level - (swing_range * 0.618)
                fib_low = min(fib_382, fib_618)
                fib_high = max(fib_382, fib_618)
            else:
                # For bearish: Fib zone is between 38.2% and 61.8% retracement from low
                fib_382 = swing_level + (swing_range * 0.382)
                fib_618 = swing_level + (swing_range * 0.618)
                fib_low = min(fib_382, fib_618)
                fib_high = max(fib_382, fib_618)

            # Draw shaded zone
            ax.axhspan(fib_low, fib_high, alpha=0.2, color='#ffeb3b',
                      label=f'Fib Zone (38.2%-61.8%)')

            # Add Fib level labels
            ax.axhline(y=fib_382, color='#ff9800', linestyle=':', linewidth=1, alpha=0.6)
            ax.axhline(y=fib_618, color='#ff9800', linestyle=':', linewidth=1, alpha=0.6)

        except Exception as e:
            logger.warning(f"Error adding Fib zone: {e}")

    def generate_single_timeframe_chart(
        self,
        epic: str,
        df: pd.DataFrame,
        timeframe: str,
        signal: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Generate a single timeframe chart (simpler version).

        Args:
            epic: Currency pair epic
            df: DataFrame with OHLC data
            timeframe: Timeframe string (e.g., '15m')
            signal: Optional signal data

        Returns:
            Base64-encoded PNG image string
        """
        if not self.is_available:
            return None

        try:
            df = self._prepare_dataframe(df, timeframe)
            if df is None or len(df) < 10:
                return None

            pair = self._extract_pair(epic)

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot candlesticks
            self._plot_candlesticks(ax, df, timeframe)

            # Add signal levels if provided
            if signal:
                self._add_signal_levels(ax, df, signal)

            # Title and labels
            direction = signal.get('signal_type', 'ANALYSIS') if signal else 'ANALYSIS'
            ax.set_title(f"{pair} - {timeframe.upper()} - {direction}", fontsize=12, fontweight='bold')
            ax.set_ylabel('Price', fontsize=10)
            ax.grid(True, alpha=0.3)

            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='upper left', fontsize=9)

            plt.tight_layout()

            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.DPI, bbox_inches='tight',
                       facecolor='white')
            plt.close(fig)

            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        except Exception as e:
            logger.error(f"Error generating single timeframe chart: {e}")
            return None


# Factory function
def create_forex_chart_generator(db_manager=None, data_fetcher=None) -> ForexChartGenerator:
    """Create a ForexChartGenerator instance"""
    return ForexChartGenerator(db_manager=db_manager, data_fetcher=data_fetcher)
