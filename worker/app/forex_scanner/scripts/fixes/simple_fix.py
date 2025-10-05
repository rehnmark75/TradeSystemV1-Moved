# debug_scanner_ema_issue.py
"""
Debug script to find why EMA indicators are not being added during live scanning
"""

import sys
import os
sys.path.append('.')

import config
from core.database import DatabaseManager
from core.data_fetcher import DataFetcher

def debug_ema_issue():
    """Debug why EMA indicators are not being added"""
    print("🔍 Debugging EMA Indicator Issue")
    print("=" * 50)
    
    # Check config
    print("📋 Configuration Check:")
    print(f"  SIMPLE_EMA_STRATEGY: {getattr(config, 'SIMPLE_EMA_STRATEGY', 'NOT_SET')}")
    print(f"  MACD_EMA_STRATEGY: {getattr(config, 'MACD_EMA_STRATEGY', 'NOT_SET')}")
    print(f"  EMA_PERIODS: {getattr(config, 'EMA_PERIODS', 'NOT_SET')}")
    
    if hasattr(config, 'EMA_STRATEGY_CONFIG'):
        active_config = getattr(config, 'ACTIVE_EMA_CONFIG', 'default')
        print(f"  ACTIVE_EMA_CONFIG: {active_config}")
        ema_config = config.EMA_STRATEGY_CONFIG.get(active_config, {})
        print(f"  EMA Config: {ema_config}")
    
    print()
    
    # Test data fetcher
    print("🔍 Testing Data Fetcher:")
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        data_fetcher = DataFetcher(db_manager, config.USER_TIMEZONE)
        
        epic = 'CS.D.EURUSD.CEEM.IP'
        pair = 'EURUSD'
        
        print(f"  Testing with {epic} ({pair})...")
        
        # Get enhanced data
        df = data_fetcher.get_enhanced_data(epic, pair, timeframe='5m')
        
        if df is not None:
            print(f"  ✅ Got {len(df)} bars of data")
            print(f"  📊 Columns: {list(df.columns)}")
            
            # Check for EMA indicators
            ema_columns = [col for col in df.columns if col.startswith('ema_')]
            print(f"  📈 EMA columns found: {ema_columns}")
            
            # Check for MACD indicators
            macd_columns = [col for col in df.columns if col.startswith('macd_')]
            print(f"  📊 MACD columns found: {macd_columns}")
            
            if not ema_columns:
                print("  ❌ NO EMA INDICATORS FOUND!")
                print("  💡 This is the problem - EMAs are not being added")
            else:
                print("  ✅ EMA indicators are present")
                
            if not macd_columns and getattr(config, 'MACD_EMA_STRATEGY', False):
                print("  ⚠️ MACD strategy enabled but no MACD indicators found")
            elif macd_columns:
                print("  ✅ MACD indicators are present")
                
        else:
            print("  ❌ Failed to get enhanced data")
            
    except Exception as e:
        print(f"  ❌ Error testing data fetcher: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    
    # Check if EMA_PERIODS is properly defined
    ema_periods = getattr(config, 'EMA_PERIODS', None)
    if not ema_periods:
        print("  1. ❌ EMA_PERIODS is not defined in config.py")
        print("     Add: EMA_PERIODS = [9, 21, 200]")
    else:
        print(f"  1. ✅ EMA_PERIODS is defined: {ema_periods}")
    
    # Check if technical analyzer is working
    if not ema_columns:
        print("  2. ❌ EMA indicators are not being added by technical_analyzer")
        print("     Check if technical_analyzer.add_ema_indicators() method is working")
        print("     Add logging to _enhance_with_analysis() method in data_fetcher.py")
    
    # Check strategy configuration
    simple_ema = getattr(config, 'SIMPLE_EMA_STRATEGY', False)
    if not simple_ema:
        print("  3. ⚠️ SIMPLE_EMA_STRATEGY is False - EMA strategy disabled")
        print("     Set SIMPLE_EMA_STRATEGY = True in config.py")
    
    print()
    print("🔧 QUICK FIX:")
    print("Add this logging to data_fetcher.py in _enhance_with_analysis():")
    print("""
    def _enhance_with_analysis(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        # Calculate EMAs and Bollinger Bands
        self.logger.info(f"🔄 Adding EMA indicators: {config.EMA_PERIODS}")
        df_enhanced = self.technical_analyzer.add_ema_indicators(df, config.EMA_PERIODS)
        
        # Add MACD indicators if MACD strategy is enabled
        if getattr(config, 'MACD_EMA_STRATEGY', False):
            self.logger.info(f"🔄 Adding MACD indicators (MACD strategy enabled)")
            df_enhanced = self.technical_analyzer.add_macd_indicators(...)
    """)

if __name__ == "__main__":
    debug_ema_issue()