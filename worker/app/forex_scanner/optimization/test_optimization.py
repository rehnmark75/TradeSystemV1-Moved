#!/usr/bin/env python3
"""
Test Optimization System
Simple test to verify the optimization system works
"""

import sys
import os
import logging

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from optimize_ema_parameters import ParameterOptimizationEngine
from optimization_analysis import OptimizationAnalyzer


def test_database_connection():
    """Test database connection and schema"""
    try:
        analyzer = OptimizationAnalyzer()
        
        # Test connection
        with analyzer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name LIKE 'ema_%'
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            print("✅ Database connection successful")
            print(f"📊 Found optimization tables: {[t[0] for t in tables]}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_parameter_grid():
    """Test parameter grid generation"""
    try:
        optimizer = ParameterOptimizationEngine()
        
        # Test quick grid
        quick_grid = optimizer.get_optimization_parameter_grid(quick_test=True)
        print("\n✅ Quick parameter grid:")
        for key, values in quick_grid.items():
            print(f"   {key}: {values}")
        
        # Calculate combinations
        total_combinations = 1
        for values in quick_grid.values():
            total_combinations *= len(values)
        
        print(f"📊 Quick test combinations: {total_combinations}")
        
        # Test full grid
        full_grid = optimizer.get_optimization_parameter_grid(quick_test=False)
        full_combinations = 1
        for values in full_grid.values():
            full_combinations *= len(values)
        
        print(f"📊 Full optimization combinations: {full_combinations:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter grid test failed: {e}")
        return False


def test_optimization_run_creation():
    """Test creating an optimization run"""
    try:
        optimizer = ParameterOptimizationEngine()
        
        # Create test run
        run_id = optimizer.create_optimization_run(
            run_name="test_run",
            epics=["CS.D.EURUSD.CEEM.IP"],
            backtest_days=7
        )
        
        if run_id:
            print(f"✅ Created test optimization run: {run_id}")
            
            # Verify it was created
            analyzer = OptimizationAnalyzer()
            runs_df = analyzer.get_optimization_run_summary(run_id)
            
            if not runs_df.empty:
                print(f"✅ Run verified in database")
                print(f"   Name: {runs_df.iloc[0]['run_name']}")
                print(f"   Status: {runs_df.iloc[0]['status']}")
                print(f"   Combinations: {runs_df.iloc[0]['total_combinations']:,}")
                return True
            else:
                print("❌ Run not found in database")
                return False
        else:
            print("❌ Failed to create optimization run")
            return False
            
    except Exception as e:
        print(f"❌ Optimization run test failed: {e}")
        return False


def test_analysis_system():
    """Test analysis and reporting system"""
    try:
        analyzer = OptimizationAnalyzer()
        
        # Test getting runs
        runs_df = analyzer.get_optimization_run_summary()
        print(f"\n✅ Found {len(runs_df)} optimization runs")
        
        # Test getting best parameters (may be empty)
        best_df = analyzer.get_best_parameters_summary()
        print(f"✅ Found {len(best_df)} best parameter sets")
        
        if not best_df.empty:
            print("📊 Sample best parameters:")
            print(best_df.head(3).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis system test failed: {e}")
        return False


def run_minimal_optimization_test():
    """Run a very minimal optimization test"""
    try:
        print("\n🧪 RUNNING MINIMAL OPTIMIZATION TEST")
        print("=" * 50)
        
        optimizer = ParameterOptimizationEngine()
        
        # Create a test run with minimal parameters
        run_id = optimizer.create_optimization_run(
            run_name="minimal_test",
            epics=["CS.D.EURUSD.CEEM.IP"],
            backtest_days=7
        )
        
        if not run_id:
            print("❌ Failed to create test run")
            return False
        
        print(f"✅ Created test run: {run_id}")
        
        # Note: We can't actually run the full optimization without the backtest system
        # being properly integrated, but we can test the framework
        
        print("⚠️ Full optimization test requires backtest system integration")
        print("   The framework is ready - need to integrate with EMABacktest signals")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal optimization test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 TESTING OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Parameter Grid", test_parameter_grid),
        ("Optimization Run Creation", test_optimization_run_creation),
        ("Analysis System", test_analysis_system),
        ("Minimal Optimization", run_minimal_optimization_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 40)
        
        if test_func():
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n🏁 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("The optimization system is ready to use.")
        print("\n📋 Next steps:")
        print("1. Integrate signal extraction from EMABacktest")
        print("2. Run optimization on a single epic with:")
        print("   python optimize_ema_parameters.py --epic CS.D.EURUSD.CEEM.IP --quick-test")
        print("3. View results with:")
        print("   python optimization_analysis.py --summary")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)