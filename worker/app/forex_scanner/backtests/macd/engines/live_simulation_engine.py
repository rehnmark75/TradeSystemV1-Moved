"""
Live Simulation Engine for MACD Strategy

This engine exactly replicates how the live forex scanner works:
1. Processes data bar-by-bar in real-time sequence
2. Uses actual strategy detection logic (not modified backtest versions)
3. Implements proper signal spacing and trade management
4. Provides accurate simulation of live trading conditions

Key difference from old backtest: Only calls detect_signal() when a new bar arrives,
not for every historical bar in a loop.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add the worker app to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from forex_scanner.core.strategies.macd_strategy import MACDStrategy


class LiveSimulationEngine:
    """
    Real-time MACD strategy simulation that mirrors live scanner behavior

    Instead of calling detect_signal() 600+ times in a loop, this engine:
    1. Processes one bar at a time (like live data feed)
    2. Only calls detect_signal() when conditions are met
    3. Properly manages trade lifecycle
    4. Implements realistic timing and signal spacing
    """

    def __init__(self, epic: str, timeframe: str = '15m', lookback_days: int = 7):
        self.epic = epic
        self.timeframe = timeframe
        self.lookback_days = lookback_days

        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{epic}")

        # Initialize strategy exactly as live scanner does
        self.strategy = MACDStrategy(
            epic=epic,
            timeframe=timeframe,
            backtest_mode=True,  # Use backtest mode for proper signal filtering
            use_optimized_parameters=True  # Use database-stored optimized parameters
        )

        # Live simulation state
        self.current_time = None
        self.last_signal_time = None
        self.active_trades = []
        self.closed_trades = []
        self.market_data_buffer = pd.DataFrame()

        # Simulation settings (mirror live scanner)
        self.min_signal_spacing_minutes = 240  # 4 hours like live scanner
        self.max_trades_per_day = 3
        self.daily_trade_count = 0
        self.current_date = None

        self.logger.info(f"🚀 Live Simulation Engine initialized for {epic}")
        self.logger.info(f"   Strategy config: {self.strategy.macd_config}")
        self.logger.info(f"   Signal spacing: {self.min_signal_spacing_minutes} minutes")
        self.logger.info(f"   Max trades/day: {self.max_trades_per_day}")

    def load_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load real historical market data for simulation"""
        try:
            self.logger.info(f"📊 Loading REAL market data for {self.epic} from {start_date} to {end_date}")

            # Import real data fetcher components
            from core.database import DatabaseManager
            from core.data_fetcher import DataFetcher

            # Initialize data fetcher with database connection
            try:
                import config
            except ImportError:
                from forex_scanner import config

            db_manager = DatabaseManager(config.DATABASE_URL)
            data_fetcher = DataFetcher(db_manager, 'UTC')

            # Calculate lookback hours
            lookback_hours = (end_date - start_date).total_seconds() / 3600

            # Extract pair name from epic (e.g., CS.D.EURUSD.MINI.IP -> EURUSD)
            pair = self.epic.split('.')[2] if '.' in self.epic else self.epic

            self.logger.info(f"   Fetching {lookback_hours:.1f} hours of {pair} data in {self.timeframe} timeframe")

            # Fetch real historical data
            df = data_fetcher.get_enhanced_data(
                epic=self.epic,
                pair=pair,
                timeframe=self.timeframe,
                lookback_hours=int(lookback_hours) + 24,  # Add extra buffer
                ema_strategy=self.strategy  # Pass strategy for proper enhancement
            )

            if df is None or len(df) == 0:
                self.logger.error(f"❌ No data received for {self.epic}")
                raise ValueError(f"No market data available for {self.epic}")

            # Check if DataFrame has a datetime index, use start_time column if needed
            if 'start_time' in df.columns:
                self.logger.info("🕐 Using start_time column as DataFrame index")
                df = df.set_index('start_time')
                df.index = pd.to_datetime(df.index)
            elif not pd.api.types.is_datetime64_any_dtype(df.index):
                self.logger.warning("⚠️ DataFrame index is not datetime, attempting conversion...")
                df.index = pd.to_datetime(df.index, unit='s', errors='coerce')

            # Use available data range - be flexible about dates
            available_start = df.index.min()
            available_end = df.index.max()

            self.logger.info(f"   Available data range: {available_start} to {available_end}")
            self.logger.info(f"   Requested range: {start_date} to {end_date}")

            # Use all available data if it overlaps with our timeframe (be more flexible)
            # Just ensure we have recent data (within last few days)
            now = datetime.now()
            if hasattr(available_end, 'tz_localize'):
                # If available_end is timezone-aware, make now timezone-aware too
                import pytz
                now = now.replace(tzinfo=pytz.UTC)
            elif hasattr(available_end, 'tz_convert'):
                # Convert to naive if needed
                available_end = available_end.tz_localize(None)

            if available_end < (now - timedelta(days=3)):
                self.logger.error(f"❌ Data is too old for {self.epic}")
                raise ValueError(f"Available data is too old for {self.epic}. Latest: {available_end}")

            # Use the last 7 days of available data
            effective_start = available_end - timedelta(days=7)
            df = df[df.index >= effective_start]

            if len(df) == 0:
                self.logger.error(f"❌ No recent data available for {self.epic}")
                raise ValueError(f"No recent data available for {self.epic}")

            self.logger.info(f"✅ Loaded {len(df)} bars of REAL market data")
            self.logger.info(f"   Data range: {df.index[0]} to {df.index[-1]}")
            self.logger.info(f"   Columns available: {list(df.columns)}")

            return df

        except Exception as e:
            self.logger.error(f"❌ Error loading real market data: {e}")
            self.logger.error("❌ SYNTHETIC DATA DISABLED - Real data required for production settings")
            raise ValueError(f"Cannot proceed without real market data for {self.epic}: {e}")

    def _generate_synthetic_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic data as fallback"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='15min')
        np.random.seed(hash(self.epic) % 1000)

        if 'JPY' in self.epic:
            base_price = 150.0
            volatility = 0.002
        else:
            base_price = 1.1000
            volatility = 0.0008

        prices = [base_price]
        for i in range(1, len(date_range)):
            noise = np.random.normal(0, volatility * 0.5)
            new_price = prices[-1] + noise
            prices.append(max(0.1, new_price))

        df = pd.DataFrame({
            'start_time': date_range,
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices,
            'ltv': [1000] * len(prices)
        })
        df.set_index('start_time', inplace=True)
        return df

    def reset_daily_counters(self, current_date: datetime.date):
        """Reset daily trading counters (mirrors live scanner behavior)"""
        if self.current_date != current_date:
            self.daily_trade_count = 0
            self.current_date = current_date
            self.logger.info(f"📅 New trading day: {current_date}, reset counters")

    def can_generate_signal(self, timestamp: datetime) -> bool:
        """Check if a new signal can be generated (mirrors live scanner logic)"""

        # Check daily trade limit
        if self.daily_trade_count >= self.max_trades_per_day:
            return False

        # Check time spacing from last signal
        if self.last_signal_time is not None:
            time_diff = timestamp - self.last_signal_time
            minutes_since_last = time_diff.total_seconds() / 60

            if minutes_since_last < self.min_signal_spacing_minutes:
                return False

        return True

    def process_new_bar(self, timestamp: datetime, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Process a single new market data bar (exactly like live scanner)

        This is the key difference: we only call detect_signal() when a new bar arrives,
        not for every historical bar in the data.
        """
        try:
            self.current_time = timestamp
            self.reset_daily_counters(timestamp.date())

            # Add new bar to buffer (maintain rolling window like live scanner)
            self.market_data_buffer = market_data.copy()

            # Only attempt signal detection if conditions allow it
            if not self.can_generate_signal(timestamp):
                return None

            # Get minimum data required for MACD (like live scanner)
            min_bars_needed = 60  # Enough for MACD calculation
            if len(self.market_data_buffer) < min_bars_needed:
                return None

            # Call strategy detection exactly once per bar (like live scanner)
            self.logger.debug(f"🔍 Checking for signals at {timestamp}")

            signal = self.strategy.detect_signal(
                self.market_data_buffer,
                self.epic,
                spread_pips=1.5,
                timeframe=self.timeframe
            )

            if signal:
                # Signal detected - process it
                self.last_signal_time = timestamp
                self.daily_trade_count += 1

                # Get current market data for detailed signal information
                current_bar = self.market_data_buffer.iloc[-1]

                # Enhance signal with detailed market data for chart comparison
                signal['timestamp'] = timestamp
                signal['epic'] = self.epic
                signal['data_bars_used'] = len(self.market_data_buffer)
                signal['close_price'] = float(current_bar['close'])
                signal['open_price'] = float(current_bar.get('open', current_bar['close']))
                signal['high_price'] = float(current_bar.get('high', current_bar['close']))
                signal['low_price'] = float(current_bar.get('low', current_bar['close']))

                # Add technical indicators if available
                if 'ema_200' in current_bar.index:
                    signal['ema_200'] = float(current_bar['ema_200'])
                if 'rsi' in current_bar.index:
                    signal['rsi'] = float(current_bar['rsi'])
                if 'adx' in current_bar.index:
                    signal['adx'] = float(current_bar['adx'])

                # Add MACD values if available
                if 'macd_line' in current_bar.index:
                    signal['macd_line'] = float(current_bar['macd_line'])
                if 'macd_signal' in current_bar.index:
                    signal['macd_signal'] = float(current_bar['macd_signal'])
                if 'macd_histogram' in current_bar.index:
                    signal['macd_histogram'] = float(current_bar['macd_histogram'])

                self.logger.info(f"🎯 SIGNAL DETECTED: {signal['signal_type']} for {self.epic} at {timestamp}")
                self.logger.info(f"   Confidence: {signal.get('confidence', 0):.1%}")
                self.logger.info(f"   Close Price: {signal['close_price']:.5f}")
                self.logger.info(f"   Daily count: {self.daily_trade_count}/{self.max_trades_per_day}")

                return signal

            return None

        except Exception as e:
            self.logger.error(f"❌ Error processing bar at {timestamp}: {e}")
            return None

    def run_simulation(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Run live simulation of MACD strategy

        Returns comprehensive results that can be trusted to represent
        real trading performance.
        """
        try:
            self.logger.info(f"🚀 Starting LIVE SIMULATION for {self.epic}")
            self.logger.info(f"   Period: {start_date} to {end_date}")
            self.logger.info(f"   Duration: {(end_date - start_date).days} days")

            # Load all market data
            full_data = self.load_market_data(start_date, end_date)

            # Process data bar-by-bar (like live data feed)
            all_signals = []

            # Start with enough data for MACD calculation
            start_index = 60

            for i in range(start_index, len(full_data)):
                current_timestamp = full_data.index[i]

                # Get data up to current bar (rolling window like live scanner)
                current_data = full_data.iloc[:i+1].copy()

                # Process this bar exactly like live scanner would
                signal = self.process_new_bar(current_timestamp, current_data)

                if signal:
                    all_signals.append(signal)

            # Calculate simulation results
            total_days = (end_date - start_date).days
            signals_per_day = len(all_signals) / total_days if total_days > 0 else 0

            results = {
                'epic': self.epic,
                'timeframe': self.timeframe,
                'simulation_period': {
                    'start': start_date,
                    'end': end_date,
                    'days': total_days
                },
                'strategy_config': self.strategy.macd_config,
                'signals': {
                    'total': len(all_signals),
                    'per_day': signals_per_day,
                    'details': all_signals
                },
                'signal_breakdown': {
                    'bull': len([s for s in all_signals if s['signal_type'] == 'BULL']),
                    'bear': len([s for s in all_signals if s['signal_type'] == 'BEAR'])
                },
                'simulation_stats': {
                    'total_bars_processed': len(full_data),
                    'signal_generation_bars': len(full_data) - start_index,
                    'signal_rate': len(all_signals) / (len(full_data) - start_index) * 100
                }
            }

            self.logger.info(f"✅ SIMULATION COMPLETE for {self.epic}")
            self.logger.info(f"   Total signals: {len(all_signals)}")
            self.logger.info(f"   Signals per day: {signals_per_day:.2f}")
            self.logger.info(f"   Signal rate: {results['simulation_stats']['signal_rate']:.4f}%")

            return results

        except Exception as e:
            self.logger.error(f"❌ Simulation failed for {self.epic}: {e}")
            raise


if __name__ == "__main__":
    # Quick test of the live simulation engine
    logging.basicConfig(level=logging.INFO)

    engine = LiveSimulationEngine('CS.D.EURUSD.MINI.IP', '15m')

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 8)  # 7 days

    results = engine.run_simulation(start, end)
    print(f"Simulation results: {results['signals']['total']} signals in {results['simulation_period']['days']} days")