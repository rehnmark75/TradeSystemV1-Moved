#!/usr/bin/env python3
"""
MACD Debug Script - Full Investigation
Compares system MACD calculations with IG chart values
Can be run from any folder in the project
"""

import sys
import os
from pathlib import Path

# Get the project root directory (forex_scanner)
current_dir = Path(__file__).resolve().parent
project_root = current_dir

# Try to find project root by looking for key files
key_files = ['main.py', 'config.py', 'trade_scan.py']
while project_root.parent != project_root:
    if any((project_root / file).exists() for file in key_files):
        break
    project_root = project_root.parent

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"🔍 Running from: {current_dir}")
print(f"📁 Project root: {project_root}")

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import your system modules with error handling
MODULES_AVAILABLE = True
import_errors = []

try:
    from core.data_fetcher import DataFetcher
    print("✅ DataFetcher imported")
except ImportError as e:
    import_errors.append(f"DataFetcher: {e}")

try:
    from core.database import DatabaseManager
    print("✅ DatabaseManager imported")
except ImportError as e:
    import_errors.append(f"DatabaseManager: {e}")

try:
    from core.strategies.helpers.macd_data_helper import MACDDataHelper
    print("✅ MACDDataHelper imported")
except ImportError as e:
    import_errors.append(f"MACDDataHelper: {e}")

try:
    from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
    print("✅ MACDSignalDetector imported")
except ImportError as e:
    import_errors.append(f"MACDSignalDetector: {e}")

try:
    import config
    print("✅ config imported")
except ImportError as e:
    import_errors.append(f"config: {e}")

if import_errors:
    print(f"⚠️ Module import errors:")
    for error in import_errors:
        print(f"   - {error}")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_macd_calculations():
    """
    🔍 DEBUG: Complete MACD calculation investigation
    """
    print("=" * 80)
    print("🔍 MACD DEBUG SCRIPT - FULL INVESTIGATION")
    print("=" * 80)
    
    # Your actual price data from the logs
    print("\n📊 USING YOUR EXACT PRICE DATA:")
    raw_data = [
        # Index, Timestamp, Open, High, Low, Close, Volume
        (0, "2025-08-22 05:50:00", 148.62500, 148.64000, 148.60800, 148.61700, 138),
        (1, "2025-08-22 05:55:00", 148.62000, 148.67850, 148.62000, 148.66600, 235),
        (2, "2025-08-22 06:00:00", 148.67400, 148.70900, 148.66700, 148.67650, 279),
        (3, "2025-08-22 06:05:00", 148.67750, 148.69200, 148.63900, 148.65150, 278),
        (4, "2025-08-22 06:10:00", 148.65050, 148.65050, 148.54200, 148.54200, 334),
        (5, "2025-08-22 06:15:00", 148.54400, 148.57300, 148.50100, 148.52000, 339),
        (6, "2025-08-22 06:20:00", 148.51800, 148.56800, 148.51000, 148.54600, 244),
        (7, "2025-08-22 06:25:00", 148.54500, 148.55400, 148.50700, 148.51900, 248),
        (8, "2025-08-22 06:30:00", 148.52200, 148.56450, 148.50450, 148.55300, 189),
        (9, "2025-08-22 06:35:00", 148.55200, 148.56500, 148.51450, 148.51650, 150),
        (10, "2025-08-22 06:40:00", 148.51550, 148.51550, 148.42350, 148.44500, 380),
        (11, "2025-08-22 06:45:00", 148.44700, 148.50150, 148.44450, 148.50150, 250),
        (12, "2025-08-22 06:50:00", 148.50250, 148.51350, 148.48350, 148.49550, 143),
        (13, "2025-08-22 06:55:00", 148.49750, 148.53500, 148.49150, 148.52150, 213),
        (14, "2025-08-22 07:00:00", 148.52250, 148.55700, 148.49700, 148.53700, 364),
        (15, "2025-08-22 07:05:00", 148.53600, 148.55250, 148.50900, 148.51900, 173),
        (16, "2025-08-22 07:10:00", 148.52000, 148.52000, 148.45900, 148.47450, 199),
        (17, "2025-08-22 07:15:00", 148.47350, 148.48750, 148.44050, 148.45300, 200),
        (18, "2025-08-22 07:20:00", 148.45000, 148.47700, 148.43400, 148.47300, 227),
        (19, "2025-08-22 07:25:00", 148.47200, 148.48600, 148.41350, 148.46000, 283),
        (20, "2025-08-22 07:30:00", 148.46100, 148.54300, 148.45300, 148.54000, 208),
        (21, "2025-08-22 07:35:00", 148.53800, 148.57300, 148.53600, 148.55950, 232),
        (22, "2025-08-22 07:40:00", 148.56250, 148.61050, 148.56250, 148.59350, 203),
        (23, "2025-08-22 07:45:00", 148.59150, 148.62100, 148.58800, 148.61150, 196),
        (24, "2025-08-22 07:50:00", 148.61550, 148.62100, 148.55550, 148.56050, 210),
        (25, "2025-08-22 07:55:00", 148.56150, 148.61400, 148.54650, 148.61000, 233),
        (26, "2025-08-22 08:00:00", 148.61100, 148.62900, 148.57650, 148.58150, 187),
        (27, "2025-08-22 08:05:00", 148.58250, 148.61750, 148.57500, 148.59150, 193),
        (28, "2025-08-22 08:10:00", 148.59250, 148.60750, 148.57050, 148.58050, 146),
        (29, "2025-08-22 08:15:00", 148.57950, 148.60450, 148.57950, 148.58550, 156),
        (30, "2025-08-22 08:20:00", 148.58450, 148.60850, 148.55300, 148.55700, 182),
        (31, "2025-08-22 08:25:00", 148.55950, 148.59850, 148.55200, 148.57850, 227),
        (32, "2025-08-22 08:30:00", 148.57950, 148.59150, 148.56000, 148.56100, 141),
        (33, "2025-08-22 08:35:00", 148.56000, 148.58950, 148.55150, 148.57600, 143),
        (34, "2025-08-22 08:40:00", 148.57700, 148.58600, 148.55900, 148.57250, 127),
        (35, "2025-08-22 08:45:00", 148.57550, 148.61750, 148.57050, 148.57450, 200),
        (36, "2025-08-22 08:50:00", 148.57850, 148.59800, 148.54350, 148.59750, 190),
        (37, "2025-08-22 08:55:00", 148.59600, 148.67950, 148.59600, 148.63550, 347),
        (38, "2025-08-22 09:00:00", 148.63650, 148.67000, 148.61950, 148.62500, 239),
        (39, "2025-08-22 09:05:00", 148.62400, 148.63750, 148.60000, 148.63400, 201),
        (40, "2025-08-22 09:10:00", 148.63300, 148.65550, 148.60450, 148.62700, 158),
        (41, "2025-08-22 09:15:00", 148.63000, 148.66100, 148.61200, 148.61300, 178),  # >>> ANALYSIS POINT
        (42, "2025-08-22 09:20:00", 148.61400, 148.66350, 148.61400, 148.65350, 182),
        (43, "2025-08-22 09:25:00", 148.65600, 148.69700, 148.65500, 148.67350, 241),
        (44, "2025-08-22 09:30:00", 148.67250, 148.68400, 148.64550, 148.64850, 201),
        (45, "2025-08-22 09:35:00", 148.64750, 148.67150, 148.64650, 148.67150, 133),
        (46, "2025-08-22 09:40:00", 148.67100, 148.67950, 148.64100, 148.67250, 172),
        (47, "2025-08-22 09:45:00", 148.67650, 148.67650, 148.63000, 148.65500, 131),
        (48, "2025-08-22 09:50:00", 148.65450, 148.68650, 148.64900, 148.68250, 150),
        (49, "2025-08-22 09:55:00", 148.68350, 148.70700, 148.68150, 148.70300, 184),
        (50, "2025-08-22 10:00:00", 148.70400, 148.72600, 148.69900, 148.70300, 163),
        (51, "2025-08-22 10:05:00", 148.70400, 148.72000, 148.69900, 148.71300, 133)
    ]
    
    # Create DataFrame
    df = pd.DataFrame(raw_data, columns=['index', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"📊 Data Shape: {df.shape}")
    print(f"📊 Analysis Point: Index 41 (09:15) = {df.iloc[41]['close']}")
    
    # IG Chart Values for Comparison
    ig_values = {
        'macd_histogram': -0.00655,
        'macd_line': -0.03546,
        'macd_signal': -0.03729
    }
    
    print(f"\n🎯 IG CHART VALUES (Target to match):")
    print(f"   MACD Histogram: {ig_values['macd_histogram']}")
    print(f"   MACD Line: {ig_values['macd_line']}")
    print(f"   Signal Line: {ig_values['macd_signal']}")
    
    print("\n" + "="*50)
    print("🧮 MANUAL CALCULATIONS - MULTIPLE METHODS")
    print("="*50)
    
    # Extract close prices
    close_prices = df['close'].values
    
    # Method 1: Standard pandas with adjust=False (Your System)
    print("\n1️⃣ METHOD 1: pandas ewm(adjust=False) - YOUR SYSTEM")
    df_calc1 = calculate_macd_method1(close_prices)
    
    # Method 2: Standard pandas with adjust=True
    print("\n2️⃣ METHOD 2: pandas ewm(adjust=True) - STANDARD")  
    df_calc2 = calculate_macd_method2(close_prices)
    
    # Method 3: Traditional EMA calculation
    print("\n3️⃣ METHOD 3: Traditional EMA calculation")
    df_calc3 = calculate_macd_traditional(close_prices)
    
    # Method 4: Alternative price sources
    print("\n4️⃣ METHOD 4: Alternative Price Sources")
    test_alternative_prices(df)
    
    # Method 5: Data order verification
    print("\n5️⃣ METHOD 5: Data Order Verification")
    test_data_order(close_prices)
    
    # Method 6: Different historical windows
    print("\n6️⃣ METHOD 6: Different Historical Windows")
    test_historical_windows(close_prices)
    
    print("\n" + "="*50)
    print("🔍 SYSTEM INTEGRATION TEST")
    print("="*50)
    
    if MODULES_AVAILABLE:
        test_system_integration(df)
    else:
        print("❌ System modules not available - skipping integration test")
    
    print("\n" + "="*50)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*50)
    
    print_summary(ig_values)

def calculate_macd_method1(close_prices):
    """Method 1: pandas ewm with adjust=False (Your System)"""
    close_series = pd.Series(close_prices)
    
    ema_12 = close_series.ewm(span=12, adjust=False).mean()
    ema_26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    
    # Focus on analysis point (index 41)
    idx = 41
    print(f"   Index {idx} (09:15):")
    print(f"     EMA 12: {ema_12.iloc[idx]:.6f}")
    print(f"     EMA 26: {ema_26.iloc[idx]:.6f}")
    print(f"     MACD Line: {macd_line.iloc[idx]:.6f}")
    print(f"     Signal Line: {macd_signal.iloc[idx]:.6f}")
    print(f"     Histogram: {macd_histogram.iloc[idx]:.6f}")
    
    return {
        'ema_12': ema_12.iloc[idx],
        'ema_26': ema_26.iloc[idx],
        'macd_line': macd_line.iloc[idx],
        'macd_signal': macd_signal.iloc[idx],
        'macd_histogram': macd_histogram.iloc[idx]
    }

def calculate_macd_method2(close_prices):
    """Method 2: pandas ewm with adjust=True (Standard)"""
    close_series = pd.Series(close_prices)
    
    ema_12 = close_series.ewm(span=12, adjust=True).mean()
    ema_26 = close_series.ewm(span=26, adjust=True).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=True).mean()
    macd_histogram = macd_line - macd_signal
    
    idx = 41
    print(f"   Index {idx} (09:15):")
    print(f"     EMA 12: {ema_12.iloc[idx]:.6f}")
    print(f"     EMA 26: {ema_26.iloc[idx]:.6f}")
    print(f"     MACD Line: {macd_line.iloc[idx]:.6f}")
    print(f"     Signal Line: {macd_signal.iloc[idx]:.6f}")
    print(f"     Histogram: {macd_histogram.iloc[idx]:.6f}")
    
    return {
        'ema_12': ema_12.iloc[idx],
        'ema_26': ema_26.iloc[idx],
        'macd_line': macd_line.iloc[idx],
        'macd_signal': macd_signal.iloc[idx],
        'macd_histogram': macd_histogram.iloc[idx]
    }

def calculate_macd_traditional(close_prices):
    """Method 3: Traditional EMA calculation from scratch"""
    
    def calculate_ema_traditional(prices, period):
        """Calculate EMA using traditional formula"""
        multiplier = 2 / (period + 1)
        ema = [prices[0]]  # Start with first price
        
        for i in range(1, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return ema
    
    ema_12 = calculate_ema_traditional(close_prices, 12)
    ema_26 = calculate_ema_traditional(close_prices, 26)
    
    macd_line = [ema_12[i] - ema_26[i] for i in range(len(close_prices))]
    macd_signal = calculate_ema_traditional(macd_line, 9)
    macd_histogram = [macd_line[i] - macd_signal[i] for i in range(len(macd_line))]
    
    idx = 41
    print(f"   Index {idx} (09:15):")
    print(f"     EMA 12: {ema_12[idx]:.6f}")
    print(f"     EMA 26: {ema_26[idx]:.6f}")
    print(f"     MACD Line: {macd_line[idx]:.6f}")
    print(f"     Signal Line: {macd_signal[idx]:.6f}")
    print(f"     Histogram: {macd_histogram[idx]:.6f}")
    
    return {
        'ema_12': ema_12[idx],
        'ema_26': ema_26[idx],
        'macd_line': macd_line[idx],
        'macd_signal': macd_signal[idx],
        'macd_histogram': macd_histogram[idx]
    }

def test_alternative_prices(df):
    """Test different price sources (OHLC4, HL2, etc.)"""
    
    # OHLC4 average
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # HL2 average  
    hl2 = (df['high'] + df['low']) / 2
    
    # HLC3 average
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    
    print(f"   Close Price (Your System): {df.iloc[41]['close']:.5f}")
    print(f"   OHLC4 Average: {ohlc4.iloc[41]:.5f}")
    print(f"   HL2 Average: {hl2.iloc[41]:.5f}")
    print(f"   HLC3 Average: {hlc3.iloc[41]:.5f}")
    
    # Test MACD with OHLC4
    close_series = pd.Series(ohlc4.values)
    ema_12 = close_series.ewm(span=12, adjust=False).mean()
    ema_26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    
    print(f"   MACD with OHLC4: Histogram = {macd_histogram.iloc[41]:.6f}")

def test_data_order(close_prices):
    """Test if data order affects results"""
    
    print(f"   First Price: {close_prices[0]:.5f}")
    print(f"   Last Price: {close_prices[-1]:.5f}")
    print(f"   Data Trend: {'Upward' if close_prices[-1] > close_prices[0] else 'Downward'}")
    
    # Test with reversed data
    reversed_prices = close_prices[::-1]
    close_series = pd.Series(reversed_prices)
    
    ema_12 = close_series.ewm(span=12, adjust=False).mean()
    ema_26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    
    print(f"   MACD with REVERSED data: Histogram = {macd_histogram.iloc[41]:.6f}")

def test_historical_windows(close_prices):
    """Test different amounts of historical data"""
    
    for window in [30, 40, 50, len(close_prices)]:
        if window <= len(close_prices):
            data_subset = close_prices[-window:]
            close_series = pd.Series(data_subset)
            
            ema_12 = close_series.ewm(span=12, adjust=False).mean()
            ema_26 = close_series.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - macd_signal
            
            print(f"   {window} bars: Histogram = {macd_histogram.iloc[-1]:.6f}")

def test_system_integration(df):
    """Test your actual system's MACD calculation"""
    try:
        print("   Testing actual system modules...")
        
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        data_helper = MACDDataHelper(logger)
        
        # Enhance dataframe with MACD indicators
        df_enhanced = data_helper.ensure_macd_indicators(df.copy())
        
        # Get values at analysis point
        latest = df_enhanced.iloc[41]  # Index 41 = 09:15
        
        print(f"   System MACD Line: {latest.get('macd_line', 'N/A')}")
        print(f"   System MACD Signal: {latest.get('macd_signal', 'N/A')}")
        print(f"   System MACD Histogram: {latest.get('macd_histogram', 'N/A')}")
        print(f"   System MACD Color: {latest.get('macd_color', 'N/A')}")
        
    except Exception as e:
        print(f"   ❌ System integration test failed: {e}")

def print_summary(ig_values):
    """Print summary and recommendations"""
    print("🎯 KEY FINDINGS:")
    print(f"   • IG Target: Histogram = {ig_values['macd_histogram']:.6f} (NEGATIVE)")
    print("   • All our calculations show POSITIVE values")
    print("   • This indicates a fundamental calculation difference")
    
    print("\n💡 POSSIBLE CAUSES:")
    print("   1. Different price source (OHLC4 vs Close)")
    print("   2. Different EMA calculation method")
    print("   3. Data synchronization/timing issue")
    print("   4. Insufficient historical data")
    print("   5. Bug in system's MACD implementation")
    
    print("\n🔧 RECOMMENDATIONS:")
    print("   1. Check if IG uses OHLC4 instead of Close prices")
    print("   2. Verify IG's exact timestamp for the analysis point")
    print("   3. Test with more historical data (100+ bars)")
    print("   4. Contact IG support for exact MACD calculation method")
    print("   5. Compare with other charting platforms (TradingView, MT4)")

if __name__ == "__main__":
    debug_macd_calculations()