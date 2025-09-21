"""
Backtest Chart Service
Handles chart data preparation and rendering for backtest results
Integrates with TradingView Lightweight Charts
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pytz

from .backtest_service import BacktestResult


class BacktestChartService:
    """Service for preparing and rendering backtest charts"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def prepare_chart_data(self, result: BacktestResult, chart_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Prepare chart data for TradingView Lightweight Charts

        Args:
            result: Backtest result containing signals
            chart_data: Optional price data DataFrame

        Returns:
            Dictionary containing chart configuration and data
        """
        try:
            if chart_data is None or chart_data.empty:
                self.logger.warning("No chart data provided, generating minimal chart")
                return self._create_minimal_chart(result)

            # Clean and prepare price data
            cleaned_data = self._clean_price_data(chart_data)

            # Create candlestick data
            candles = self._create_candlestick_data(cleaned_data)

            # Create signal markers
            signal_markers = self._create_signal_markers(result.signals, cleaned_data)

            # Create chart series
            series = self._create_chart_series(candles, cleaned_data, result.epic)

            # Create chart configuration
            chart_config = self._create_chart_config(result.epic)

            return {
                'chart_config': chart_config,
                'series': series,
                'markers': signal_markers,
                'candle_count': len(candles),
                'signal_count': len(signal_markers),
                'timeframe': result.timeframe,
                'epic': result.epic
            }

        except Exception as e:
            self.logger.error(f"Error preparing chart data: {e}")
            return self._create_minimal_chart(result)

    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare price data"""
        try:
            df = df.copy()

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    raise ValueError(f"Missing required column: {col}")

            # Convert to numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN values
            df.dropna(subset=required_columns, inplace=True)

            # Ensure proper datetime index
            if 'start_time' in df.columns:
                df['start_time'] = pd.to_datetime(df['start_time'])
                if df['start_time'].dt.tz is None:
                    df['start_time'] = df['start_time'].dt.tz_localize('UTC')
                else:
                    df['start_time'] = df['start_time'].dt.tz_convert('UTC')
            elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)
                df.rename(columns={'timestamp': 'start_time'}, inplace=True)
                df['start_time'] = pd.to_datetime(df['start_time'])
                if df['start_time'].dt.tz is None:
                    df['start_time'] = df['start_time'].dt.tz_localize('UTC')
                else:
                    df['start_time'] = df['start_time'].dt.tz_convert('UTC')
            else:
                # Fallback: create timestamp column
                df['start_time'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15T', tz='UTC')

            # Sort by time
            df.sort_values('start_time', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Remove flat candles (where OHLC are the same)
            is_flat = (
                np.isclose(df['open'], df['high']) &
                np.isclose(df['high'], df['low']) &
                np.isclose(df['low'], df['close'])
            )
            df = df[~is_flat]

            self.logger.info(f"✅ Cleaned price data: {len(df)} candles")
            return df

        except Exception as e:
            self.logger.error(f"Error cleaning price data: {e}")
            raise

    def _create_candlestick_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create candlestick data for TradingView"""
        try:
            candles = []
            for _, row in df.iterrows():
                candle = {
                    "time": int(row.start_time.timestamp()),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close)
                }
                candles.append(candle)

            self.logger.info(f"✅ Created {len(candles)} candlestick data points")
            return candles

        except Exception as e:
            self.logger.error(f"Error creating candlestick data: {e}")
            return []

    def _create_signal_markers(self, signals: List[Dict[str, Any]], chart_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create signal markers for the chart"""
        markers = []

        try:
            if not signals:
                return markers

            # Create a mapping of times to candle times for alignment
            candle_times = chart_data['start_time'].dt.floor('min').unique()

            for signal in signals:
                try:
                    marker = self._create_single_marker(signal, candle_times)
                    if marker:
                        markers.append(marker)
                except Exception as e:
                    self.logger.warning(f"Error creating marker for signal: {e}")
                    continue

            self.logger.info(f"✅ Created {len(markers)} signal markers")
            return markers

        except Exception as e:
            self.logger.error(f"Error creating signal markers: {e}")
            return []

    def _create_single_marker(self, signal: Dict[str, Any], candle_times) -> Optional[Dict[str, Any]]:
        """Create a single signal marker"""
        try:
            # Parse timestamp
            timestamp = signal.get('timestamp')
            if not timestamp:
                return None

            if isinstance(timestamp, str):
                signal_time = pd.to_datetime(timestamp)
            else:
                signal_time = pd.to_datetime(timestamp)

            # Ensure UTC
            if signal_time.tz is None:
                signal_time = signal_time.tz_localize('UTC')
            else:
                signal_time = signal_time.tz_convert('UTC')

            # Find the nearest candle time
            nearest_time = min(candle_times, key=lambda x: abs((x - signal_time).total_seconds()))

            # Determine marker properties
            direction = signal.get('direction', '').upper()
            is_bull = direction == 'BUY' or direction == 'BULL'

            # Determine color based on profitability
            profit_pips = signal.get('max_profit_pips', 0)
            loss_pips = signal.get('max_loss_pips', 0)

            if profit_pips > loss_pips:
                color = "#26a69a"  # Green
                status = "Profitable"
            elif loss_pips > profit_pips:
                color = "#ef5350"  # Red
                status = "Loss"
            else:
                color = "#ffb74d"  # Orange
                status = "Breakeven"

            # Create marker text
            strategy = signal.get('strategy', 'Signal')
            confidence = signal.get('confidence', 0)

            text_parts = [strategy]
            if confidence > 0:
                text_parts.append(f"{confidence:.1%}")
            if profit_pips > 0 or loss_pips > 0:
                text_parts.append(f"{profit_pips - loss_pips:+.1f}p")

            marker = {
                "time": int(nearest_time.timestamp()),
                "position": "aboveBar" if is_bull else "belowBar",
                "color": color,
                "shape": "arrowUp" if is_bull else "arrowDown",
                "text": " ".join(text_parts),
                "size": 1
            }

            return marker

        except Exception as e:
            self.logger.warning(f"Error creating single marker: {e}")
            return None

    def _create_chart_series(self, candles: List[Dict], chart_data: pd.DataFrame, epic: str) -> List[Dict[str, Any]]:
        """Create chart series including price and indicators"""
        series = []

        try:
            # Determine price precision
            price_precision = 3 if "JPY" in epic else 5

            # Main candlestick series
            candlestick_series = {
                "type": "Candlestick",
                "data": candles,
                "options": {
                    "upColor": "#26a69a",
                    "downColor": "#ef5350",
                    "borderVisible": False,
                    "wickUpColor": "#26a69a",
                    "wickDownColor": "#ef5350",
                    "priceFormat": {
                        "type": "price",
                        "precision": price_precision,
                        "minMove": 1 / (10 ** price_precision)
                    }
                }
            }
            series.append(candlestick_series)

            # Add indicator series if available
            series.extend(self._create_indicator_series(chart_data))

            self.logger.info(f"✅ Created {len(series)} chart series")
            return series

        except Exception as e:
            self.logger.error(f"Error creating chart series: {e}")
            return [{"type": "Candlestick", "data": candles}]

    def _create_indicator_series(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create indicator series (EMA, MACD, etc.)"""
        series = []

        try:
            # EMA indicators
            ema_indicators = [
                ('ema21', "#4caf50", "EMA 21"),
                ('ema50', "#ff9800", "EMA 50"),
                ('ema200', "#2196f3", "EMA 200")
            ]

            for col, color, title in ema_indicators:
                if col in df.columns:
                    ema_data = [
                        {"time": int(row.start_time.timestamp()), "value": float(row[col])}
                        for _, row in df.iterrows() if not pd.isna(row[col])
                    ]

                    if ema_data:
                        series.append({
                            "type": "Line",
                            "data": ema_data,
                            "options": {
                                "color": color,
                                "lineWidth": 2,
                                "title": title
                            }
                        })

            # Zero Lag EMA
            if 'zlema' in df.columns:
                zlema_data = [
                    {"time": int(row.start_time.timestamp()), "value": float(row.zlema)}
                    for _, row in df.iterrows() if not pd.isna(row.zlema)
                ]

                if zlema_data:
                    series.append({
                        "type": "Line",
                        "data": zlema_data,
                        "options": {
                            "color": "#9c27b0",
                            "lineWidth": 2,
                            "title": "Zero Lag EMA"
                        }
                    })

            return series

        except Exception as e:
            self.logger.error(f"Error creating indicator series: {e}")
            return []

    def _create_chart_config(self, epic: str) -> Dict[str, Any]:
        """Create chart configuration"""
        return {
            "width": None,
            "height": 600,
            "layout": {
                "background": {"color": "#ffffff"},
                "textColor": "#333333"
            },
            "rightPriceScale": {
                "scaleMargins": {"top": 0.1, "bottom": 0.1},
                "borderVisible": True,
                "entireTextOnly": False,
                "visible": True
            },
            "crosshair": {
                "mode": 0,
                "vertLine": {
                    "width": 1,
                    "color": "#758696",
                    "style": 0
                },
                "horzLine": {
                    "width": 1,
                    "color": "#758696",
                    "style": 0
                }
            },
            "timeScale": {
                "timeVisible": True,
                "secondsVisible": False,
                "borderVisible": True
            },
            "grid": {
                "vertLines": {"color": "#e0e0e0", "style": 1},
                "horzLines": {"color": "#e0e0e0", "style": 1}
            },
            "trackingMode": {
                "exitOnScrollOrScale": False
            }
        }

    def _create_minimal_chart(self, result: BacktestResult) -> Dict[str, Any]:
        """Create a minimal chart when no data is available"""
        self.logger.warning("Creating minimal chart - no price data available")

        # Create dummy data
        dummy_candles = []
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for i in range(100):
            time_val = int((base_time.timestamp() + i * 900))  # 15-minute intervals
            price = 1.1000 + (i % 10) * 0.0001
            dummy_candles.append({
                "time": time_val,
                "open": price,
                "high": price + 0.0005,
                "low": price - 0.0005,
                "close": price + (0.0002 if i % 2 else -0.0002)
            })

        return {
            'chart_config': self._create_chart_config(result.epic),
            'series': [{"type": "Candlestick", "data": dummy_candles}],
            'markers': [],
            'candle_count': len(dummy_candles),
            'signal_count': 0,
            'timeframe': result.timeframe,
            'epic': result.epic,
            'is_minimal': True
        }

    def create_indicator_charts(self, chart_data: pd.DataFrame, strategy_name: str) -> List[Dict[str, Any]]:
        """Create additional indicator charts (MACD, oscillators, etc.)"""
        indicator_charts = []

        try:
            # MACD Chart
            if all(col in chart_data.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
                macd_chart = self._create_macd_chart(chart_data)
                if macd_chart:
                    indicator_charts.append(macd_chart)

            # Two-Pole Oscillator Chart
            if 'two_pole_osc' in chart_data.columns:
                two_pole_chart = self._create_two_pole_chart(chart_data)
                if two_pole_chart:
                    indicator_charts.append(two_pole_chart)

            self.logger.info(f"✅ Created {len(indicator_charts)} indicator charts")
            return indicator_charts

        except Exception as e:
            self.logger.error(f"Error creating indicator charts: {e}")
            return []

    def _create_macd_chart(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create MACD indicator chart"""
        try:
            macd_line_data = [
                {"time": int(row.start_time.timestamp()), "value": float(row.macd_line)}
                for _, row in df.iterrows() if not pd.isna(row.macd_line)
            ]

            macd_signal_data = [
                {"time": int(row.start_time.timestamp()), "value": float(row.macd_signal)}
                for _, row in df.iterrows() if not pd.isna(row.macd_signal)
            ]

            macd_histogram_data = [
                {
                    "time": int(row.start_time.timestamp()),
                    "value": float(row.macd_histogram),
                    "color": "#26a69a" if row.macd_histogram >= 0 else "#ef5350"
                }
                for _, row in df.iterrows() if not pd.isna(row.macd_histogram)
            ]

            if not macd_line_data:
                return None

            chart_config = {
                "width": None,
                "height": 200,
                "layout": {
                    "background": {"color": "#ffffff"},
                    "textColor": "#333333"
                },
                "rightPriceScale": {
                    "scaleMargins": {"top": 0.1, "bottom": 0.1},
                    "borderVisible": True
                },
                "timeScale": {
                    "timeVisible": False,
                    "borderVisible": False
                },
                "grid": {
                    "vertLines": {"color": "#e0e0e0", "style": 1},
                    "horzLines": {"color": "#e0e0e0", "style": 1}
                }
            }

            series = [
                {
                    "type": "Histogram",
                    "data": macd_histogram_data,
                    "options": {
                        "priceLineVisible": False,
                        "lastValueVisible": False
                    }
                },
                {
                    "type": "Line",
                    "data": macd_line_data,
                    "options": {
                        "color": "#2196f3",
                        "lineWidth": 2,
                        "title": "MACD"
                    }
                },
                {
                    "type": "Line",
                    "data": macd_signal_data,
                    "options": {
                        "color": "#ff9800",
                        "lineWidth": 2,
                        "title": "Signal"
                    }
                }
            ]

            return {
                "chart": chart_config,
                "series": series,
                "title": "MACD"
            }

        except Exception as e:
            self.logger.error(f"Error creating MACD chart: {e}")
            return None

    def _create_two_pole_chart(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create Two-Pole Oscillator chart"""
        try:
            two_pole_data = [
                {"time": int(row.start_time.timestamp()), "value": float(row.two_pole_osc)}
                for _, row in df.iterrows() if not pd.isna(row.two_pole_osc)
            ]

            if not two_pole_data:
                return None

            # Zero line
            zero_line_data = [
                {"time": t["time"], "value": 0}
                for t in two_pole_data
            ]

            chart_config = {
                "width": None,
                "height": 200,
                "layout": {
                    "background": {"color": "#ffffff"},
                    "textColor": "#333333"
                },
                "rightPriceScale": {
                    "scaleMargins": {"top": 0.1, "bottom": 0.1},
                    "borderVisible": True
                },
                "timeScale": {
                    "timeVisible": False,
                    "borderVisible": False
                },
                "grid": {
                    "vertLines": {"color": "#e0e0e0", "style": 1},
                    "horzLines": {"color": "#e0e0e0", "style": 1}
                }
            }

            series = [
                {
                    "type": "Line",
                    "data": two_pole_data,
                    "options": {
                        "color": "#9c27b0",
                        "lineWidth": 2,
                        "title": "Two-Pole Oscillator"
                    }
                },
                {
                    "type": "Line",
                    "data": zero_line_data,
                    "options": {
                        "color": "#666666",
                        "lineWidth": 1,
                        "lineStyle": 1,
                        "title": "Zero Line"
                    }
                }
            ]

            return {
                "chart": chart_config,
                "series": series,
                "title": "Two-Pole Oscillator"
            }

        except Exception as e:
            self.logger.error(f"Error creating Two-Pole chart: {e}")
            return None


# Global instance
_chart_service = None

def get_chart_service() -> BacktestChartService:
    """Get the global chart service instance"""
    global _chart_service
    if _chart_service is None:
        _chart_service = BacktestChartService()
    return _chart_service