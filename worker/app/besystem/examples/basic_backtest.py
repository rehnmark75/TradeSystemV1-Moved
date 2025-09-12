# examples/basic_backtest.py

from datetime import datetime, timedelta
from typing import List
import csv
from core.data_structures import Portfolio
from backtesting.engine import BacktestEngine

def run_simple_backtest(engine):
    """Run a basic backtest with default settings"""
    
    # Initialize the backtesting engine
    backtest_engine = BacktestEngine(
        engine=engine,
        initial_balance=10000,
        user_timezone='Europe/Stockholm'
    )
    
    # Define which currency pairs to test
    epic_list = [
        'CS.D.EURUSD.MINI.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP'
    ]
    
    # Set the testing period (last 2 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    print("🚀 Starting Simple Backtest...")
    
    try:
        # Run the backtest
        results = backtest_engine.run_backtest(
            epic_list=epic_list,
            start_date=start_date,
            end_date=end_date,
            timeframe=5  # 5-minute bars
        )
        
        # Print the results
        if 'report' in results:
            print(results['report'])
        
        # Export trades to CSV for analysis
        if 'portfolio' in results:
            export_trades_to_csv(results['portfolio'], 'simple_backtest_trades.csv')
        
        return results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_advanced_backtest(engine):
    """Run backtest with custom risk management settings"""
    
    # Initialize with custom settings
    backtest_engine = BacktestEngine(
        engine=engine,
        initial_balance=50000,  # Larger account
        user_timezone='Europe/Stockholm'
    )
    
    # Customize risk management
    backtest_engine.risk_manager.max_risk_per_trade = 0.015  # 1.5% per trade
    backtest_engine.risk_manager.max_daily_loss = 0.03      # 3% daily loss limit
    backtest_engine.risk_manager.max_open_trades = 5        # Max 5 concurrent trades
    backtest_engine.risk_manager.position_sizing_method = 'fixed_percent'  # Fixed percent sizing
    
    # Extended epic list including more pairs
    epic_list = [
        'CS.D.EURUSD.MINI.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.AUDUSD.MINI.IP',
        'CS.D.USDCAD.MINI.IP'
    ]
    
    # Longer testing period (3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    print("🔬 Starting Advanced Backtest...")
    
    try:
        results = backtest_engine.run_backtest(
            epic_list=epic_list,
            start_date=start_date,
            end_date=end_date,
            timeframe=5
        )
        
        # Generate detailed analysis
        if 'portfolio' in results and 'metrics' in results:
            portfolio = results['portfolio']
            metrics = results['metrics']
            
            print(f"\n📊 DETAILED RESULTS:")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"ROI: {metrics.get('roi_percent', 0):.2f}%")
            
            # Export detailed trade data
            export_trades_to_csv(portfolio, 'advanced_backtest_trades.csv')
        
        return results
        
    except Exception as e:
        print(f"❌ Error running advanced backtest: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def export_trades_to_csv(portfolio: Portfolio, filename: str = 'backtest_trades.csv'):
    """Export trades to CSV file"""
    if not portfolio.closed_trades:
        print("No trades to export")
        return
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'trade_id', 'epic', 'signal_type', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'position_size', 'pnl_pips',
                'pnl_currency', 'duration_minutes', 'status'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for trade in portfolio.closed_trades:
                writer.writerow({
                    'trade_id': trade.trade_id,
                    'epic': trade.signal.epic,
                    'signal_type': trade.signal.signal_type.value,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'position_size': trade.position_size,
                    'pnl_pips': trade.pnl_pips,
                    'pnl_currency': trade.pnl_currency,
                    'duration_minutes': trade.duration_minutes,
                    'status': trade.status.value
                })
        
        print(f"✅ {len(portfolio.closed_trades)} trades exported to {filename}")
        
    except Exception as e:
        print(f"❌ Error exporting trades: {e}")

def plot_equity_curve(portfolio: Portfolio):
    """Plot equity curve (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        from analytics.performance import PerformanceAnalytics
        
        equity_curve = PerformanceAnalytics._calculate_equity_curve(portfolio)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Account Balance')
        plt.grid(True)
        plt.show()
        
        print("📈 Equity curve plotted")
        
    except ImportError:
        print("⚠️ Matplotlib not available - cannot plot equity curve")
        print("Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error plotting equity curve: {e}")

def create_sample_epic_list() -> List[str]:
    """Create a sample list of epics for testing"""
    return [
        'CS.D.EURUSD.MINI.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.AUDUSD.MINI.IP',
        'CS.D.USDCAD.MINI.IP'
    ]

def test_system_components(engine):
    """Test individual system components"""
    print("🧪 Testing System Components...")
    
    try:
        # Test data manager
        from data.manager import DataManager
        data_manager = DataManager(engine)
        print("✅ DataManager created successfully")
        
        # Test technical analysis
        from analysis.technical import TechnicalAnalysis
        ta = TechnicalAnalysis()
        print("✅ TechnicalAnalysis created successfully")
        
        # Test signal detector
        from analysis.signals import SignalDetector
        signal_detector = SignalDetector(ta)
        print("✅ SignalDetector created successfully")
        
        # Test risk manager
        from trading.risk_manager import RiskManager
        risk_manager = RiskManager()
        print("✅ RiskManager created successfully")
        
        # Test trade executor
        from trading.executor import TradeExecutor
        executor = TradeExecutor(risk_manager)
        print("✅ TradeExecutor created successfully")
        
        # Test performance analytics
        from analytics.performance import PerformanceAnalytics
        analytics = PerformanceAnalytics()
        print("✅ PerformanceAnalytics created successfully")
        
        # Test backtest engine
        backtest_engine = BacktestEngine(engine)
        print("✅ BacktestEngine created successfully")
        
        print("\n🎉 All components tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_test(engine):
    """Run a quick test with minimal data"""
    print("⚡ Running Quick Test...")
    
    try:
        # Test with just one epic and short timeframe
        backtest_engine = BacktestEngine(
            engine=engine,
            initial_balance=10000
        )
        
        epic_list = ['CS.D.EURUSD.MINI.IP']  # Just one epic
        
        # Very short test period (1 week)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        results = backtest_engine.run_backtest(
            epic_list=epic_list,
            start_date=start_date,
            end_date=end_date,
            timeframe=5
        )
        
        if 'error' in results:
            print(f"❌ Quick test failed: {results['error']}")
            return False
        
        print("✅ Quick test completed successfully!")
        
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"Test Results: {metrics.get('total_trades', 0)} trades generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Example usage functions that can be called from main.py
def demo_basic_usage(engine):
    """Demonstrate basic usage of the system"""
    print("📚 Basic Usage Demo")
    print("=" * 50)
    
    # Test components first
    if not test_system_components(engine):
        print("❌ Component test failed - stopping demo")
        return
    
    # Run quick test
    if not run_quick_test(engine):
        print("❌ Quick test failed - stopping demo")
        return
    
    # Run simple backtest
    print("\n" + "="*50)
    results = run_simple_backtest(engine)
    
    if results and 'error' not in results:
        print("✅ Demo completed successfully!")
    else:
        print("❌ Demo failed")

if __name__ == "__main__":
    print("This file contains backtesting examples.")
    print("Import and use the functions in your main script.")