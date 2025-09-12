#!/usr/bin/env python3
"""
Simple Optimization System Test
Tests core optimization functionality without backtest dependencies
"""

import sys
import os
import logging

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.database import DatabaseManager
try:
    import config
except ImportError:
    from forex_scanner import config


def test_database_schema():
    """Test that optimization tables exist"""
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check optimization tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name LIKE 'ema_%'
                ORDER BY table_name
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['ema_optimization_runs', 'ema_optimization_results', 'ema_best_parameters']
            found_tables = [t for t in expected_tables if t in tables]
            
            print(f"✅ Database connection successful")
            print(f"📊 Expected tables: {expected_tables}")
            print(f"📊 Found tables: {found_tables}")
            
            if len(found_tables) == len(expected_tables):
                print("✅ All optimization tables exist")
                return True
            else:
                print(f"❌ Missing tables: {set(expected_tables) - set(found_tables)}")
                return False
                
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_parameter_grid_calculation():
    """Test parameter grid calculations"""
    try:
        # Simulate parameter grid
        grid = {
            'ema_configs': ['default', 'aggressive', 'conservative'],
            'confidence_levels': [0.45, 0.55, 0.65],
            'timeframes': ['15m', '1h'],
            'smart_money_options': [True, False],
            'stop_loss_levels': [10, 15],
            'take_profit_levels': [20, 30]
        }
        
        # Calculate total combinations
        total_combinations = 1
        for key, values in grid.items():
            total_combinations *= len(values)
            print(f"   {key}: {len(values)} options")
        
        print(f"✅ Parameter grid calculation successful")
        print(f"📊 Total combinations per epic: {total_combinations}")
        print(f"📊 For 9 epics: {total_combinations * 9:,} total tests")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter grid test failed: {e}")
        return False


def test_optimization_run_creation():
    """Test creating optimization run record"""
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create test run
            cursor.execute("""
                INSERT INTO ema_optimization_runs 
                (run_name, description, total_combinations, status)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (
                'simple_test',
                'Simple test of optimization system',
                100,
                'running'
            ))
            
            run_id = cursor.fetchone()[0]
            conn.commit()
            
            print(f"✅ Created optimization run: {run_id}")
            
            # Verify it exists
            cursor.execute("SELECT * FROM ema_optimization_runs WHERE id = %s", (run_id,))
            run_record = cursor.fetchone()
            
            if run_record:
                print(f"✅ Run verified in database")
                print(f"   ID: {run_record[0]}")
                print(f"   Name: {run_record[1]}")
                print(f"   Status: {run_record[6]}")  # status column
                return True
            else:
                print("❌ Run not found in database")
                return False
                
    except Exception as e:
        print(f"❌ Run creation test failed: {e}")
        return False


def test_sample_result_storage():
    """Test storing sample optimization result"""
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get the latest run ID
            cursor.execute("SELECT id FROM ema_optimization_runs ORDER BY id DESC LIMIT 1")
            run_result = cursor.fetchone()
            
            if not run_result:
                print("❌ No optimization run found")
                return False
            
            run_id = run_result[0]
            
            # Insert sample result
            cursor.execute("""
                INSERT INTO ema_optimization_results (
                    run_id, epic, ema_config, confidence_threshold, timeframe, smart_money_enabled,
                    stop_loss_pips, take_profit_pips, risk_reward_ratio,
                    total_signals, win_rate, profit_factor, net_pips, composite_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                run_id,
                'CS.D.EURUSD.CEEM.IP',
                'default',
                0.55,
                '15m',
                False,
                10.0,
                20.0,
                2.0,
                50,
                0.70,
                1.5,
                150.0,
                1.575  # 0.70 * 1.5 * (150/100)
            ))
            
            result_id = cursor.fetchone()[0]
            conn.commit()
            
            print(f"✅ Created sample result: {result_id}")
            
            # Test best parameters storage
            cursor.execute("""
                INSERT INTO ema_best_parameters (
                    epic, best_ema_config, best_confidence_threshold, best_timeframe,
                    optimal_stop_loss_pips, optimal_take_profit_pips,
                    best_win_rate, best_profit_factor, best_net_pips
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (epic) DO UPDATE SET
                    best_ema_config = EXCLUDED.best_ema_config,
                    best_confidence_threshold = EXCLUDED.best_confidence_threshold,
                    best_timeframe = EXCLUDED.best_timeframe,
                    optimal_stop_loss_pips = EXCLUDED.optimal_stop_loss_pips,
                    optimal_take_profit_pips = EXCLUDED.optimal_take_profit_pips,
                    best_win_rate = EXCLUDED.best_win_rate,
                    best_profit_factor = EXCLUDED.best_profit_factor,
                    best_net_pips = EXCLUDED.best_net_pips,
                    last_updated = NOW()
            """, (
                'CS.D.EURUSD.CEEM.IP',
                'default',
                0.55,
                '15m',
                10.0,
                20.0,
                0.70,
                1.5,
                150.0
            ))
            
            conn.commit()
            
            print(f"✅ Stored best parameters for EURUSD")
            
            return True
            
    except Exception as e:
        print(f"❌ Result storage test failed: {e}")
        return False


def test_analysis_queries():
    """Test basic analysis queries"""
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Test best parameters query
            cursor.execute("""
                SELECT epic, best_ema_config, best_win_rate, best_profit_factor, best_net_pips
                FROM ema_best_parameters 
                ORDER BY best_net_pips DESC
            """)
            
            best_params = cursor.fetchall()
            print(f"✅ Best parameters query: found {len(best_params)} records")
            
            for row in best_params:
                epic, config, win_rate, pf, net_pips = row
                print(f"   {epic}: {config}, WR={win_rate:.1%}, PF={pf:.2f}, Net={net_pips:.1f}")
            
            # Test results summary query
            cursor.execute("""
                SELECT 
                    epic,
                    COUNT(*) as test_count,
                    AVG(win_rate) as avg_win_rate,
                    MAX(profit_factor) as best_profit_factor
                FROM ema_optimization_results
                GROUP BY epic
                ORDER BY avg_win_rate DESC
            """)
            
            summaries = cursor.fetchall()
            print(f"✅ Results summary query: found {len(summaries)} epic summaries")
            
            for row in summaries:
                epic, count, avg_wr, best_pf = row
                print(f"   {epic}: {count} tests, Avg WR={avg_wr:.1%}, Best PF={best_pf:.2f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Analysis queries test failed: {e}")
        return False


def main():
    """Run all simple tests"""
    print("🧪 SIMPLE OPTIMIZATION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Database Schema", test_database_schema),
        ("Parameter Grid Calculation", test_parameter_grid_calculation),
        ("Optimization Run Creation", test_optimization_run_creation),
        ("Sample Result Storage", test_sample_result_storage),
        ("Analysis Queries", test_analysis_queries)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n🏁 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("The optimization database schema is working correctly.")
        print("\n📋 Next steps:")
        print("1. Integrate with EMA backtest system to extract signals")
        print("2. Run parameter optimization")
        print("3. Analyze results")
    else:
        print("\n⚠️ Some tests failed. Please check the issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)