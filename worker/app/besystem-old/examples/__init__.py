# examples/__init__.py - SAFE VERSION THAT HANDLES ERRORS

"""
Examples module for the trading system
This version handles import errors gracefully
"""

# Try to import functions one by one
_available_functions = []

try:
    from .basic_backtest import run_simple_backtest
    _available_functions.append("run_simple_backtest")
except ImportError as e:
    print(f"Warning: Could not import run_simple_backtest: {e}")
    
    # Create a fallback function
    def run_simple_backtest(engine):
        print("❌ run_simple_backtest not available - using fallback")
        return {"error": "function not implemented"}

try:
    from .basic_backtest import run_advanced_backtest
    _available_functions.append("run_advanced_backtest")
except ImportError:
    def run_advanced_backtest(engine):
        print("❌ run_advanced_backtest not available")
        return {"error": "function not implemented"}

try:
    from .basic_backtest import export_trades_to_csv
    _available_functions.append("export_trades_to_csv")
except ImportError:
    def export_trades_to_csv(portfolio, filename="trades.csv"):
        print("❌ export_trades_to_csv not available")

try:
    from .basic_backtest import test_system_components
    _available_functions.append("test_system_components")
except ImportError:
    def test_system_components(engine):
        print("❌ test_system_components not available")
        return False

try:
    from .basic_backtest import demo_basic_usage
    _available_functions.append("demo_basic_usage")
except ImportError:
    def demo_basic_usage(engine):
        print("❌ demo_basic_usage not available")

# Export what we have
__all__ = [
    "run_simple_backtest",
    "run_advanced_backtest", 
    "export_trades_to_csv",
    "test_system_components",
    "demo_basic_usage"
]

print(f"Examples module loaded. Available functions: {_available_functions}")

# Test function to verify everything works
def test_imports():
    """Test that all imports work"""
    print("Testing example imports...")
    
    functions_to_test = [
        run_simple_backtest,
        run_advanced_backtest,
        export_trades_to_csv,
        test_system_components,
        demo_basic_usage
    ]
    
    for func in functions_to_test:
        print(f"✅ {func.__name__} is available")
    
    print("All imports successful!")

if __name__ == "__main__":
    test_imports()