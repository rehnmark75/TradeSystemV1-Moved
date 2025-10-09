# main.py

import sys
import os
from sqlalchemy import create_engine, text

# --- Config ---
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

# Import what's actually available
try:
    from examples.basic_backtest import run_simple_backtest, run_advanced_backtest
    EXAMPLES_AVAILABLE = True
    print("✅ Examples imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import examples: {e}")
    EXAMPLES_AVAILABLE = False
    
    # Create fallback functions
    def run_simple_backtest(engine):
        return {"error": "examples not available"}
    
    def run_advanced_backtest(engine):
        return {"error": "examples not available"}

def setup_database_connection():
    """Set up database connection - REPLACE WITH YOUR ACTUAL CONNECTION"""
    
    # Option 1: PostgreSQL (replace with your credentials)
    # DATABASE_URL = "postgresql://username:password@localhost:5432/trading_db"
    
    # Option 2: SQLite (for testing)
    DATABASE_URL = "sqlite:///trading_data.db"
    
    # Option 3: Your actual IG database connection
    # DATABASE_URL = "your_actual_database_connection_string"
    
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            pass
        print(f"✅ Database connected successfully")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def check_system_setup():
    """Check if all required modules can be imported"""
    print("🔍 Checking system setup...")
    
    missing_modules = []
    
    try:
        from core.data_structures import Signal, Trade, Portfolio, SignalType, TradeStatus
        print("✅ Core data structures imported")
    except ImportError as e:
        missing_modules.append(f"core.data_structures: {e}")
    
    try:
        from core.config import EpicConfig
        print("✅ Core config imported")
    except ImportError as e:
        missing_modules.append(f"core.config: {e}")
    
    try:
        from backtesting.engine import BacktestEngine
        print("✅ Backtesting engine imported")
    except ImportError as e:
        missing_modules.append(f"backtesting.engine: {e}")
    
    try:
        from examples.basic_backtest import run_simple_backtest
        print("✅ Examples imported")
    except ImportError as e:
        missing_modules.append(f"examples.basic_backtest: {e}")
    
    if missing_modules:
        print("\n❌ Missing modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n📋 Please ensure all files are created and in the correct directories.")
        return False
    
    print("✅ All modules available!")
    return True

def run_system_test(engine):
    """Run a basic system test"""
    print("\n🧪 Running System Test...")
    
    try:
        from examples.basic_backtest import test_system_components, run_quick_test
        
        # Test individual components
        if not test_system_components(engine):
            return False
        
        # Run quick backtest
        if not run_quick_test(engine):
            return False
        
        print("✅ System test passed!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_menu(engine):
    """Display main menu and handle user choices"""
    
    while True:
        print("\n" + "="*60)
        print("🔧 TRADING BACKTESTING SYSTEM")
        print("="*60)
        print("1. Run System Test")
        print("2. Run Simple Backtest")
        print("3. Run Advanced Backtest") 
        print("4. Demo Basic Usage")
        print("5. Check System Setup")
        print("6. Exit")
        print("="*60)
        
        choice = input("Select option (1-6): ").strip()
        
        try:
            if choice == '1':
                print("\n🧪 Running System Test...")
                if EXAMPLES_AVAILABLE:
                    from examples.basic_backtest import test_system_components
                    test_system_components(engine)
                else:
                    print("❌ Examples not available for testing")
                
            elif choice == '2':
                print("\n🚀 Running Simple Backtest...")
                results = run_simple_backtest(engine)
                if results and 'error' not in results:
                    print("✅ Simple backtest completed!")
                
            elif choice == '3':
                print("\n🔬 Running Advanced Backtest...")
                results = run_advanced_backtest(engine)
                if results and 'error' not in results:
                    print("✅ Advanced backtest completed!")
                
            elif choice == '4':
                print("\n📚 Running Demo...")
                if EXAMPLES_AVAILABLE:
                    from examples.basic_backtest import demo_basic_usage
                    demo_basic_usage(engine)
                else:
                    print("❌ Demo not available - examples not loaded")
                
            elif choice == '5':
                check_system_setup()
                
            elif choice == '6':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("This might indicate missing files or import issues.")
            print("Try option 5 to check your system setup.")

def create_minimal_test():
    """Create a minimal test that doesn't require database"""
    print("\n🔬 Running Minimal Test (No Database Required)...")
    
    try:
        # Test core data structures
        from core.data_structures import Signal, SignalType, Trade, TradeStatus, Portfolio
        from datetime import datetime
        
        # Create test signal
        signal = Signal(
            signal_type=SignalType.BULL,
            epic="CS.D.EURUSD.MINI.IP",
            timestamp=datetime.now(),
            price=1.1000,
            confidence_score=0.75,
            ema_9=1.0995,
            ema_21=1.0990,
            ema_200=1.0980
        )
        
        print(f"✅ Created signal: {signal.signal_type.value} for {signal.epic}")
        
        # Create test portfolio
        portfolio = Portfolio(
            initial_balance=10000,
            current_balance=10000,
            equity=10000,
            margin_used=0,
            free_margin=10000
        )
        
        print(f"✅ Created portfolio with balance: ${portfolio.current_balance}")
        
        # Test enums
        print(f"✅ Signal type enum: {SignalType.BULL}")
        print(f"✅ Trade status enum: {TradeStatus.OPEN}")
        
        print("✅ Minimal test passed - core components working!")
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print("🚀 Starting Trading Backtesting System...")
    
    # First, check if we can import basic components
    if not create_minimal_test():
        print("\n❌ Basic component test failed.")
        print("Please check that all files are created correctly.")
        print("Run option 5 in the menu to diagnose issues.")
        return
    
    # Check full system setup
    if not check_system_setup():
        print("\n❌ System setup incomplete.")
        print("Please create all required files before proceeding.")
        return
    
    # Try to connect to database
    engine = setup_database_connection()
    
    if engine is None:
        print("\n⚠️ Database connection failed.")
        print("You can still test the system components without a database.")
        print("Update the database connection in setup_database_connection() function.")
        
        # Offer limited functionality without database
        choice = input("\nWould you like to test system components anyway? (y/n): ")
        if choice.lower() == 'y':
            print("\n🔧 Testing components without database...")
            engine = None  # Will be handled by individual functions
        else:
            print("👋 Please set up database connection and try again.")
            return
    
    # Run main menu
    try:
        main_menu(engine)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()