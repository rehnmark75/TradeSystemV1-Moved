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
        'order_block_bull': '#4caf5040',  # Semi-transparent green for bullish OB
        'order_block_bear': '#f4433640',  # Semi-transparent red for bearish OB
        'order_block_border_bull': '#4caf50',  # Green border for bullish OB
        'order_block_border_bear': '#f44336',  # Red border for bearish OB
        'fvg_bull': '#2196f330',      # Semi-transparent blue for bullish FVG
        'fvg_bear': '#ff980030',      # Semi-transparent orange for bearish FVG
        'fvg_border_bull': '#2196f3', # Blue border for bullish FVG
        'fvg_border_bear': '#ff9800', # Orange border for bearish FVG
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

                # Add EMA 9/21 on 15m and 5m for swing break context and micro-structure
                if tf in ['15m', '5m'] and len(df) >= 21:
                    self._add_ema_stack(ax, df, strategy_indicators)

                # Add Support/Resistance levels on 15m (swing break timeframe)
                if tf == '15m' and strategy_indicators:
                    self._add_support_resistance_levels(ax, df, strategy_indicators)

                # Add FVG zones on 15m and 5m (institutional imbalances)
                if tf in ['15m', '5m'] and signal:
                    self._add_fvg_zones(ax, df, signal, tf)

                # Add Order Block zones on 15m and 5m (institutional order areas)
                if tf in ['15m', '5m'] and signal:
                    self._add_order_block_zones(ax, df, signal, tf)

                # Add entry type annotation on 5m (entry timeframe)
                if tf == '5m' and signal:
                    self._add_entry_type_annotation(ax, df, signal, strategy_indicators)

                # [CHART_IMPROVE_V1] Add entry price and current price markers on 5m
                if tf == '5m' and signal:
                    self._add_entry_price_marker(ax, df, signal)
                    self._add_current_price_marker(ax, df)

                # [CHART_IMPROVE_V1] Add swing break level on 15m and 5m (always, not just when smc_data present)
                if tf in ['15m', '5m'] and signal:
                    self._add_swing_break_level(ax, df, signal, tf)

                # Add SMC annotations (swing levels, BOS)
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

            # [CHART_IMPROVE_V1] Preserve timestamp before resetting index
            # Check for timestamp in index or columns
            if df.index.dtype == 'datetime64[ns]' or 'datetime64' in str(df.index.dtype):
                df['_timestamp'] = df.index
            elif 'timestamp' in df.columns:
                df['_timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            elif 'time' in df.columns:
                df['_timestamp'] = pd.to_datetime(df['time'], errors='coerce')
            elif 'date' in df.columns:
                df['_timestamp'] = pd.to_datetime(df['date'], errors='coerce')

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

            # [CHART_IMPROVE_V1] Add time labels on x-axis if timestamps available
            self._add_time_axis_labels(ax, df, timeframe)

        except Exception as e:
            logger.error(f"Error plotting candlesticks: {e}")

    def _add_time_axis_labels(self, ax, df: pd.DataFrame, timeframe: str) -> None:
        """
        [CHART_IMPROVE_V1] Add time labels on x-axis for context.

        Shows timestamps at regular intervals so Claude can understand
        the time context of the price action.
        """
        try:
            if '_timestamp' not in df.columns:
                return

            n_bars = len(df)
            if n_bars < 5:
                return

            # Determine how many labels to show based on timeframe
            if timeframe == '4h':
                n_labels = 5  # Show ~5 labels for 4H
                date_format = '%m/%d %H:%M'
            elif timeframe == '15m':
                n_labels = 6  # Show ~6 labels for 15M
                date_format = '%H:%M'
            else:  # 5m
                n_labels = 8  # Show ~8 labels for 5M
                date_format = '%H:%M'

            # Calculate interval between labels
            interval = max(1, n_bars // n_labels)

            # Get positions and labels
            positions = []
            labels = []

            for i in range(0, n_bars, interval):
                ts = df.iloc[i].get('_timestamp')
                if pd.notna(ts):
                    positions.append(i)
                    if isinstance(ts, pd.Timestamp):
                        labels.append(ts.strftime(date_format))
                    else:
                        labels.append(str(ts)[-8:-3])  # Fallback: last part of string

            # Always include last bar
            if positions and positions[-1] != n_bars - 1:
                ts = df.iloc[-1].get('_timestamp')
                if pd.notna(ts):
                    positions.append(n_bars - 1)
                    if isinstance(ts, pd.Timestamp):
                        labels.append(ts.strftime(date_format))
                    else:
                        labels.append(str(ts)[-8:-3])

            if positions:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

        except Exception as e:
            logger.debug(f"Could not add time axis labels: {e}")

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

            # Position annotation in top-right area (inset from edge to avoid clipping)
            x_pos = len(df) - 5  # Move inward from edge
            y_range = df['High'].max() - df['Low'].min()
            y_pos = df['High'].max() - (y_range * 0.02)

            # Build annotation text - compact format
            depth_pct = abs(pullback_depth * 100)
            if entry_type == 'PULLBACK':
                zone_icon = "✓" if in_optimal_zone else "✗"
                annotation_text = f"⬇ PULLBACK {depth_pct:.0f}%\nZone: {zone_icon}"
                bg_color = '#1565c0'  # Blue for pullback
            else:  # MOMENTUM
                annotation_text = f"⚡ MOMENTUM {depth_pct:.0f}%"
                bg_color = '#ff6f00'  # Orange for momentum

            # Add volume status (compact)
            vol_icon = "Vol✓" if volume_confirmed else "Vol✗"
            annotation_text += f"\n{vol_icon}"

            # Draw annotation box
            ax.annotate(
                annotation_text,
                xy=(x_pos, y_pos),
                fontsize=8,
                fontweight='bold',
                color='white',
                ha='right',
                va='top',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=bg_color,
                    edgecolor='white',
                    alpha=0.9
                )
            )

        except Exception as e:
            logger.warning(f"Error adding entry type annotation: {e}")

    # ========================================================================
    # [CHART_IMPROVE_V1] New methods for improved Claude vision analysis
    # ========================================================================

    def _add_entry_price_marker(
        self,
        ax,
        df: pd.DataFrame,
        signal: Dict[str, Any]
    ) -> None:
        """
        [CHART_IMPROVE_V1] Add entry price horizontal line on chart.

        Shows where the trade entry is planned, helping Claude evaluate
        if the entry level makes sense relative to price action.
        """
        try:
            # Get entry price from signal (try multiple field names)
            entry_price = (
                signal.get('entry_price') or
                signal.get('price') or
                signal.get('limit_price') or
                signal.get('execution_price')
            )

            if not entry_price:
                return

            x_max = len(df) - 1

            # Draw entry price line
            ax.axhline(
                y=entry_price,
                color=self.COLORS['entry'],
                linestyle='--',
                linewidth=2,
                alpha=0.9,
                zorder=10  # Draw on top
            )

            # Add label on right side
            ax.annotate(
                f'ENTRY: {entry_price:.5f}',
                xy=(x_max, entry_price),
                xytext=(x_max + 1, entry_price),
                fontsize=9,
                fontweight='bold',
                color=self.COLORS['entry'],
                va='center',
                ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor=self.COLORS['entry'],
                    alpha=0.9
                )
            )

            logger.debug(f"Added entry price marker at {entry_price}")

        except Exception as e:
            logger.warning(f"Error adding entry price marker: {e}")

    def _add_current_price_marker(
        self,
        ax,
        df: pd.DataFrame
    ) -> None:
        """
        [CHART_IMPROVE_V1] Add current price marker (arrow/line at last close).

        Helps Claude see exactly where price is "now" vs the entry level.
        """
        try:
            if df is None or len(df) == 0:
                return

            current_price = df['Close'].iloc[-1]
            x_max = len(df) - 1

            # Draw small arrow pointing to current price
            ax.annotate(
                '',
                xy=(x_max, current_price),
                xytext=(x_max + 2, current_price),
                arrowprops=dict(
                    arrowstyle='->',
                    color='#ffffff',
                    lw=2
                )
            )

            # Add "NOW" label
            ax.annotate(
                f'NOW: {current_price:.5f}',
                xy=(x_max + 2, current_price),
                fontsize=8,
                fontweight='bold',
                color='white',
                va='center',
                ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='#424242',
                    edgecolor='white',
                    alpha=0.9
                )
            )

        except Exception as e:
            logger.warning(f"Error adding current price marker: {e}")

    def _add_swing_break_level(
        self,
        ax,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        timeframe: str
    ) -> None:
        """
        [CHART_IMPROVE_V1] Add swing break level horizontal line.

        The swing break is the core SMC concept - shows where structure
        was broken to confirm the trade direction.
        """
        try:
            # Get swing level from signal or strategy_indicators
            swing_level = signal.get('swing_level')

            # Also try strategy_indicators
            if not swing_level:
                strategy_indicators = signal.get('strategy_indicators', {})
                tier2_swing = strategy_indicators.get('tier2_swing', {})
                swing_level = tier2_swing.get('swing_level') or tier2_swing.get('swing_break_level')

            if not swing_level:
                return

            x_max = len(df) - 1
            direction = signal.get('signal_type', signal.get('signal', '')).upper()

            # Determine color based on direction
            if direction in ['BULL', 'BUY']:
                color = self.COLORS['swing_low']  # Blue for bullish swing break
                label = 'SB▲'  # Short label
            else:
                color = self.COLORS['swing_high']  # Orange for bearish swing break
                label = 'SB▼'  # Short label

            # Draw swing break line
            ax.axhline(
                y=swing_level,
                color=color,
                linestyle=':',
                linewidth=2,
                alpha=0.9,
                zorder=9
            )

            # [CHART_IMPROVE_V1] Simplified label - just "SB" on left side
            ax.annotate(
                label,
                xy=(2, swing_level),
                fontsize=9,
                fontweight='bold',
                color='white',
                va='center',
                ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color,
                    edgecolor='white',
                    alpha=0.9
                )
            )

            logger.debug(f"Added swing break level at {swing_level} on {timeframe}")

        except Exception as e:
            logger.warning(f"Error adding swing break level: {e}")

    # ========================================================================
    # End of [CHART_IMPROVE_V1] new methods
    # ========================================================================

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

    def _add_fvg_zones(
        self,
        ax,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        timeframe: str
    ) -> None:
        """
        Add Fair Value Gap (FVG) zones as rectangles on the chart.

        FVGs represent institutional imbalances where price moved too fast,
        leaving gaps that often get filled later.

        Args:
            ax: Matplotlib axis
            df: DataFrame with OHLC data
            signal: Signal data containing FVG information
            timeframe: Current timeframe being plotted
        """
        try:
            # Get FVG data from signal
            smc_data = signal.get('smc_data', {})
            fvg_data = smc_data.get('fvg_data', {})

            # Also check strategy_indicators for FVG data
            strategy_indicators = signal.get('strategy_indicators', {})
            dataframe_analysis = strategy_indicators.get('dataframe_analysis', {})

            # Get active FVGs list
            active_fvgs = fvg_data.get('active_fvgs', [])

            # Also check for FVG columns in the dataframe itself
            # FIXED: Deduplicate FVGs - only create one entry per unique high/low price pair
            if not active_fvgs and 'fvg_high' in df.columns:
                seen_fvgs = set()  # Track unique FVGs by (high, low, type) tuple
                for i in range(len(df)):
                    row = df.iloc[i]
                    if pd.notna(row.get('fvg_high')) and pd.notna(row.get('fvg_low')):
                        fvg_high = row['fvg_high']
                        fvg_low = row['fvg_low']
                        fvg_type = 'bullish' if row.get('fvg_bullish', False) else 'bearish'

                        # Create unique key for this FVG
                        fvg_key = (round(fvg_high, 5), round(fvg_low, 5), fvg_type)

                        if fvg_key not in seen_fvgs:
                            seen_fvgs.add(fvg_key)
                            active_fvgs.append({
                                'high': fvg_high,
                                'low': fvg_low,
                                'type': fvg_type,
                                'start_index': i,  # First occurrence = start of FVG
                                'significance': row.get('fvg_significance', 0.5)
                            })

            if not active_fvgs:
                return

            x_min = 0
            x_max = len(df) - 1

            fvg_count = 0
            max_fvgs = 5  # Limit to avoid chart clutter

            for fvg in active_fvgs:
                if fvg_count >= max_fvgs:
                    break

                fvg_high = fvg.get('high') or fvg.get('high_price')
                fvg_low = fvg.get('low') or fvg.get('low_price')
                fvg_type = fvg.get('type') or fvg.get('gap_type', 'bullish')

                if not fvg_high or not fvg_low:
                    continue

                # Determine if FVG is within visible price range
                price_min = df['Low'].min()
                price_max = df['High'].max()

                if fvg_low > price_max or fvg_high < price_min:
                    continue  # FVG outside visible range

                # [CHART_IMPROVE_V1] Improved FVG visualization
                # Get start index for the rectangle (or use recent portion of chart)
                start_idx = fvg.get('start_index', max(0, x_max - 30))

                # [CHART_IMPROVE_V1] Limit FVG width to max 20 bars
                fvg_width = min(20, x_max - start_idx)

                # [CHART_IMPROVE_V1] Enhanced colors - higher opacity for better visibility
                if 'bull' in str(fvg_type).lower():
                    fill_color = '#2196f350'  # 50% opacity (was 30%)
                    border_color = '#2196f3'
                    label = f'Bull FVG'
                else:
                    fill_color = '#ff980050'  # 50% opacity (was 30%)
                    border_color = '#ff9800'
                    label = f'Bear FVG'

                # [CHART_IMPROVE_V1] Draw FVG as limited-width rectangle
                rect = plt.Rectangle(
                    (start_idx, fvg_low),
                    fvg_width,  # Limited width
                    fvg_high - fvg_low,
                    facecolor=fill_color,
                    edgecolor=border_color,
                    linewidth=2,  # Thicker border
                    alpha=0.7,  # Higher overall opacity
                    label=label if fvg_count == 0 else None
                )
                ax.add_patch(rect)

                # [CHART_IMPROVE_V1] Add horizontal line to show FVG zone extends
                mid_price = (fvg_high + fvg_low) / 2
                if start_idx + fvg_width < x_max:
                    ax.axhline(
                        y=mid_price,
                        xmin=(start_idx + fvg_width) / x_max,
                        xmax=1.0,
                        color=border_color,
                        linestyle=':',
                        linewidth=1,
                        alpha=0.4
                    )

                # [CHART_IMPROVE_V1] Improved label with background
                ax.annotate(
                    'FVG',
                    xy=(x_max - 2, mid_price),
                    fontsize=8,
                    color=border_color,
                    fontweight='bold',
                    ha='right',
                    va='center',
                    alpha=0.95,
                    bbox=dict(
                        boxstyle='round,pad=0.15',
                        facecolor='white',
                        edgecolor=border_color,
                        linewidth=1,
                        alpha=0.85
                    )
                )

                fvg_count += 1

            if fvg_count > 0:
                logger.debug(f"Drew {fvg_count} FVG zones on {timeframe} chart")

        except Exception as e:
            logger.warning(f"Error adding FVG zones: {e}")

    def _add_order_block_zones(
        self,
        ax,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        timeframe: str
    ) -> None:
        """
        Add Order Block zones as shaded rectangles on the chart.

        Order Blocks represent areas where institutions placed large orders,
        often acting as strong support/resistance zones.

        Args:
            ax: Matplotlib axis
            df: DataFrame with OHLC data
            signal: Signal data containing Order Block information
            timeframe: Current timeframe being plotted
        """
        try:
            # Get Order Block data from signal
            smc_data = signal.get('smc_data', {})
            ob_data = smc_data.get('order_block_data', {})

            # Get active order blocks list
            active_obs = ob_data.get('active_order_blocks', [])
            logger.info(f"📊 [CHART] Drawing OBs - from smc_data: {len(active_obs)}, will check df columns: {'order_block_high' in df.columns}")

            # Also check for OB columns in the dataframe itself
            # FIXED: Deduplicate OBs - only create one entry per unique high/low price pair
            if not active_obs and 'order_block_high' in df.columns:
                seen_obs = set()  # Track unique OBs by (high, low, type) tuple
                for i in range(len(df)):
                    row = df.iloc[i]
                    if pd.notna(row.get('order_block_high')) and pd.notna(row.get('order_block_low')):
                        ob_high = row['order_block_high']
                        ob_low = row['order_block_low']
                        ob_type = 'bullish' if row.get('order_block_bullish', False) else 'bearish'

                        # Create unique key for this OB
                        ob_key = (round(ob_high, 5), round(ob_low, 5), ob_type)

                        if ob_key not in seen_obs:
                            seen_obs.add(ob_key)
                            active_obs.append({
                                'high': ob_high,
                                'low': ob_low,
                                'type': ob_type,
                                'start_index': i,  # First occurrence = start of OB
                                'strength': row.get('order_block_strength', 'medium'),
                                'confidence': row.get('order_block_confidence', 0.5)
                            })

            if not active_obs:
                return

            x_min = 0
            x_max = len(df) - 1

            ob_count = 0
            max_obs = 3  # Limit to avoid chart clutter (OBs are larger zones)

            for ob in active_obs:
                if ob_count >= max_obs:
                    break

                ob_high = ob.get('high') or ob.get('high_price')
                ob_low = ob.get('low') or ob.get('low_price')
                ob_type = ob.get('type') or ob.get('block_type', 'bullish')

                if not ob_high or not ob_low:
                    continue

                # Determine if OB is within visible price range
                price_min = df['Low'].min()
                price_max = df['High'].max()

                if ob_low > price_max or ob_high < price_min:
                    continue  # OB outside visible range

                # [CHART_IMPROVE_V1] Improved OB visualization
                # Get start index for the rectangle
                start_idx = ob.get('start_index', max(0, x_max - 40))

                # [CHART_IMPROVE_V1] Limit OB width to max 15 bars to reduce clutter
                # OBs are zones, not infinite extensions
                ob_width = min(15, x_max - start_idx)

                # Determine colors based on OB type
                if 'bull' in str(ob_type).lower():
                    fill_color = self.COLORS['order_block_bull']
                    border_color = self.COLORS['order_block_border_bull']
                    label = 'Bull OB'
                else:
                    fill_color = self.COLORS['order_block_bear']
                    border_color = self.COLORS['order_block_border_bear']
                    label = 'Bear OB'

                # [CHART_IMPROVE_V1] Draw Order Block as a limited-width rectangle
                # Also draw a horizontal line extending to show the zone level
                rect = plt.Rectangle(
                    (start_idx, ob_low),
                    ob_width,  # Limited width instead of x_max - start_idx
                    ob_high - ob_low,
                    facecolor=fill_color,
                    edgecolor=border_color,
                    linewidth=2,
                    linestyle='-',  # Solid border for better visibility
                    alpha=0.4,  # Slightly reduced opacity
                    label=label if ob_count == 0 else None
                )
                ax.add_patch(rect)

                # [CHART_IMPROVE_V1] Add horizontal dashed line showing OB zone extends
                mid_price = (ob_high + ob_low) / 2
                ax.axhline(
                    y=mid_price,
                    xmin=(start_idx + ob_width) / x_max,
                    xmax=1.0,
                    color=border_color,
                    linestyle=':',
                    linewidth=1,
                    alpha=0.5
                )

                # Add label on the left side of the OB
                ax.annotate(
                    'OB',
                    xy=(start_idx + 1, mid_price),
                    fontsize=9,
                    color=border_color,
                    fontweight='bold',
                    ha='left',
                    va='center',
                    alpha=0.95,
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        edgecolor=border_color,
                        linewidth=1.5,
                        alpha=0.9
                    )
                )

                ob_count += 1

            if ob_count > 0:
                logger.debug(f"Drew {ob_count} Order Block zones on {timeframe} chart")

        except Exception as e:
            logger.warning(f"Error adding Order Block zones: {e}")

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
