#!/usr/bin/env python3
"""
Signal Investigation Script
Diagnose why live signals don't appear in backtests
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    from core.signal_detector import SignalDetector
    from core.data_fetcher import DataFetcher
    from alerts.alert_history import AlertHistoryManager
    from backtests.backtest_ema import EMABacktest
    from core.database import DatabaseManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the forex_scanner directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalInvestigator:
    def __init__(self):
        # Initialize database manager first
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        
        # Initialize other components with proper parameters
        self.signal_detector = SignalDetector(self.db_manager, config.USER_TIMEZONE)
        self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
        self.alert_history = AlertHistoryManager(self.db_manager)
        # Note: Removed backtest_engine due to import issues
        
    def investigate_missing_signal(self, 
                                 epic='CS.D.EURUSD.MINI.IP', 
                                 alert_timestamp='2025-08-11 17:48:27',
                                 timeframe='15m'):
        """
        Investigate why a logged alert doesn't show up in backtests
        """
        print("🔍 SIGNAL INVESTIGATION REPORT")
        print("=" * 60)
        print(f"Epic: {epic}")
        print(f"Alert Timestamp: {alert_timestamp}")
        print(f"Timeframe: {timeframe}")
        print()
        
        # Step 1: Check database for the logged signal
        self._check_database_signal(epic, alert_timestamp)
        
        # Step 2: Get data around the alert time
        alert_dt = pd.to_datetime(alert_timestamp)
        self._analyze_data_around_alert(epic, alert_dt, timeframe)
        
        # Step 3: Compare live scanner vs backtest detection
        self._compare_detection_methods(epic, alert_dt, timeframe)
        
        # Step 4: Test strategy configuration differences
        self._check_strategy_configs()
        
        # Step 5: Recommendations
        self._provide_recommendations()
    
    def _check_database_signal(self, epic, alert_timestamp):
        """Check if the signal exists in the database"""
        print("📊 DATABASE SIGNAL CHECK")
        print("-" * 30)
        
        try:
            # Query alert_history for signals around this time
            query = """
            SELECT epic, signal_type, strategy, confidence_score, 
                   timestamp, signal_trigger, price
            FROM alert_history 
            WHERE epic = %s 
            AND timestamp BETWEEN %s::timestamp - INTERVAL '1 hour' 
            AND %s::timestamp + INTERVAL '1 hour'
            ORDER BY timestamp
            """
            
            results = self.alert_history.db_manager.execute_query(
                query, (epic, alert_timestamp, alert_timestamp)
            )
            
            if results:
                print(f"✅ Found {len(results)} signals in database around alert time:")
                for i, signal in enumerate(results, 1):
                    print(f"   {i}. {signal[5]} - {signal[1]} ({signal[3]:.1%}) at {signal[4]}")
                    if alert_timestamp in str(signal[4]):
                        print(f"      🎯 EXACT MATCH: This is likely your signal!")
            else:
                print("❌ No signals found in database around alert time")
                print("   This suggests the signal may not have been saved to DB")
                
        except Exception as e:
            print(f"❌ Database check failed: {e}")
    
    def _analyze_data_around_alert(self, epic, alert_dt, timeframe):
        """Analyze market data around the alert time"""
        print(f"\n📈 DATA ANALYSIS AROUND ALERT")
        print("-" * 30)
        
        try:
            # Get data for the alert day
            start_date = alert_dt.replace(hour=0, minute=0, second=0)
            end_date = start_date + timedelta(days=1)
            
            df = self.data_fetcher.get_historical_data(
                epic, timeframe, start_date, end_date
            )
            
            if df is None or len(df) == 0:
                print("❌ No data available for analysis")
                return
                
            print(f"✅ Retrieved {len(df)} candles for {alert_dt.date()}")
            
            # Find the closest candle to alert time
            alert_idx = None
            closest_time_diff = float('inf')
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                time_diff = abs((timestamp - alert_dt).total_seconds())
                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    alert_idx = i
            
            if alert_idx is not None:
                print(f"📍 Closest candle to alert: Index {alert_idx}")
                self._analyze_candle_context(df, alert_idx, alert_dt)
            else:
                print("❌ Could not find matching candle")
                
        except Exception as e:
            print(f"❌ Data analysis failed: {e}")
    
    def _analyze_candle_context(self, df, alert_idx, alert_dt):
        """Analyze the context around the alert candle"""
        print(f"\n🕯️ CANDLE CONTEXT ANALYSIS")
        print("-" * 30)
        
        # Show candles around the alert
        start_idx = max(0, alert_idx - 2)
        end_idx = min(len(df), alert_idx + 3)
        
        print("Timestamp            Close      EMA9       EMA21      EMA200     Notes")
        print("-" * 80)
        
        for i in range(start_idx, end_idx):
            candle = df.iloc[i]
            timestamp = df.index[i]
            
            close = candle.get('close', 0)
            ema9 = candle.get('ema_9', 0)
            ema21 = candle.get('ema_21', 0)
            ema200 = candle.get('ema_200', 0)
            
            marker = " ← ALERT" if i == alert_idx else ""
            
            print(f"{timestamp} {close:8.5f} {ema9:8.5f} {ema21:8.5f} {ema200:8.5f} {marker}")
        
        # Analyze price vs EMA relationships
        alert_candle = df.iloc[alert_idx]
        prev_candle = df.iloc[alert_idx - 1] if alert_idx > 0 else None
        
        if prev_candle is not None:
            self._detect_crossover_types(alert_candle, prev_candle)
    
    def _detect_crossover_types(self, current, previous):
        """Detect what type of crossovers occurred"""
        print(f"\n🔄 CROSSOVER DETECTION")
        print("-" * 30)
        
        # EMA crossovers
        ema9_curr, ema21_curr = current['ema_9'], current['ema_21']
        ema9_prev, ema21_prev = previous['ema_9'], previous['ema_21']
        
        # Price crossovers
        price_curr, price_prev = current['close'], previous['close']
        
        # Check EMA 9/21 crossover
        if ema9_prev <= ema21_prev and ema9_curr > ema21_curr:
            print("✅ EMA 9/21 BULLISH CROSSOVER detected")
        elif ema9_prev >= ema21_prev and ema9_curr < ema21_curr:
            print("✅ EMA 9/21 BEARISH CROSSOVER detected")
        
        # Check price/EMA crossovers
        if price_prev <= ema9_prev and price_curr > ema9_curr:
            print("✅ PRICE/EMA9 BULLISH CROSSOVER detected")
            print("   🎯 This could be the 'enhanced_ema_price_crossover' signal!")
        elif price_prev >= ema9_prev and price_curr < ema9_curr:
            print("✅ PRICE/EMA9 BEARISH CROSSOVER detected")
            print("   🎯 This could be the 'enhanced_ema_price_crossover' signal!")
        
        if price_prev <= ema21_prev and price_curr > ema21_curr:
            print("✅ PRICE/EMA21 BULLISH CROSSOVER detected")
        elif price_prev >= ema21_prev and price_curr < ema21_curr:
            print("✅ PRICE/EMA21 BEARISH CROSSOVER detected")
    
    def _compare_detection_methods(self, epic, alert_dt, timeframe):
        """Compare live scanner vs backtest detection"""
        print(f"\n⚖️ DETECTION METHOD COMPARISON")
        print("-" * 30)
        
        try:
            # Test live scanner detection
            print("🔴 LIVE SCANNER SIMULATION:")
            
            # Get recent data (as live scanner would)
            end_time = alert_dt + timedelta(minutes=15)  # Get a bit after alert
            start_time = alert_dt - timedelta(hours=24)  # Get 24h before
            
            live_df = self.data_fetcher.get_historical_data(
                epic, timeframe, start_time, end_time
            )
            
            if live_df is not None:
                # Run signal detection as live scanner would
                live_signals = self.signal_detector.detect_signals(epic, timeframe)
                print(f"   Live detection result: {len(live_signals) if live_signals else 0} signals")
                
                if live_signals:
                    for signal in live_signals:
                        print(f"   - {signal.get('signal_type', 'unknown')} ({signal.get('confidence_score', 0):.1%})")
            
            print("\n🔵 BACKTEST SIMULATION:")
            
            try:
                from backtests.backtest_ema import EMABacktest
                
                backtest_engine = EMABacktest()
                print(f"   Backtest detection result: Testing with EMA backtest engine...")
                
                # Try to run a simple backtest on the data
                try:
                    # Initialize the strategy within the backtest engine
                    backtest_engine.initialize_ema_strategy()
                    
                    # Try to detect signals using the strategy
                    if hasattr(backtest_engine, 'strategy') and backtest_engine.strategy:
                        test_signals = backtest_engine.strategy.detect_signal(
                            live_df, epic, 1.5, timeframe
                        )
                        if test_signals:
                            print(f"   - Found signals via strategy: {test_signals.get('signal_type', 'unknown')}")
                        else:
                            print(f"   - No signals found via strategy")
                    else:
                        print(f"   - Strategy not properly initialized")
                        
                except Exception as strategy_error:
                    print(f"   - Strategy test failed: {strategy_error}")
                    
            except ImportError as import_error:
                print(f"   - Could not import EMA backtest: {import_error}")
                backtest_signals = []
            
            # Compare results
            if live_signals and not any('signals' in str(locals()).lower()):
                print("\n🚨 DISCREPANCY FOUND:")
                print("   Live scanner finds signals but backtest analysis shows issues!")
                print("   Possible causes:")
                print("   - Different signal detection logic")
                print("   - Different data sources/timing")
                print("   - Different strategy configurations")
                print("   - Import/initialization problems")
                
        except Exception as e:
            print(f"❌ Detection comparison failed: {e}")
    
    def _check_strategy_configs(self):
        """Check for configuration differences"""
        print(f"\n⚙️ STRATEGY CONFIGURATION CHECK")
        print("-" * 30)
        
        # Check relevant config settings
        configs_to_check = [
            'SIMPLE_EMA_STRATEGY',
            'MACD_EMA_STRATEGY', 
            'MIN_CONFIDENCE',
            'USE_BID_ADJUSTMENT',
            'COMBINED_STRATEGY_MODE'
        ]
        
        print("Current Configuration:")
        for config_name in configs_to_check:
            value = getattr(config, config_name, 'NOT SET')
            print(f"   {config_name}: {value}")
            
        # Check if enhanced detection is enabled
        enhanced_enabled = getattr(config, 'ENHANCED_EMA_DETECTION', False)
        print(f"   ENHANCED_EMA_DETECTION: {enhanced_enabled}")
        
        if not enhanced_enabled:
            print("   ⚠️ Enhanced EMA detection may be disabled!")
    
    def _provide_recommendations(self):
        """Provide recommendations to fix the issue"""
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 30)
        
        print("1. 🔧 Check if enhanced_ema_price_crossover is enabled in backtest:")
        print("   - Verify backtest engine includes price crossover detection")
        print("   - Check if strategy configurations match between live and backtest")
        
        print("\n2. 📊 Data synchronization issues:")
        print("   - Live scanner may use real-time data")
        print("   - Backtest may use historical data with slight differences")
        print("   - Check timestamp alignment")
        
        print("\n3. 🔄 Strategy detection differences:")
        print("   - Live scanner: May detect price crossovers")
        print("   - Backtest engine: May only detect EMA crossovers")
        print("   - Ensure both use same detection logic")
        
        print("\n4. 🛠️ Debugging commands to run:")
        print("   python main.py debug --epic CS.D.EURUSD.MINI.IP")
        print("   python main.py debug-ema --epic CS.D.EURUSD.MINI.IP")
        print("   python main.py backtest --epic CS.D.EURUSD.MINI.IP --days 1 --show-signals")

def main():
    """Main investigation function"""
    if len(sys.argv) < 2:
        print("Usage: python signal_investigation.py <alert_timestamp> [epic] [timeframe]")
        print("Example: python signal_investigation.py '2025-08-11 17:48:27' CS.D.EURUSD.MINI.IP 15m")
        sys.exit(1)
    
    alert_timestamp = sys.argv[1]
    epic = sys.argv[2] if len(sys.argv) > 2 else 'CS.D.EURUSD.MINI.IP'
    timeframe = sys.argv[3] if len(sys.argv) > 3 else '15m'
    
    investigator = SignalInvestigator()
    investigator.investigate_missing_signal(epic, alert_timestamp, timeframe)

if __name__ == "__main__":
    main()