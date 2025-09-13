#!/usr/bin/env python3
"""
Test script for TradingView Scripts database schema

Tests database creation, FTS5 search capabilities, and basic operations.
"""

import sys
import os
import sqlite3
import json
from pathlib import Path

# Add the mcp directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from tvscripts_server.db import DB

def test_database_creation():
    """Test database table creation and structure"""
    print("🔧 Testing database creation...")
    
    # Create test database
    test_db_path = "test_tv_scripts.db"
    
    # Remove existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # Initialize database
    db = DB(test_db_path)
    
    # Check if tables exist
    cursor = db.conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = ['scripts', 'script_bodies', 'scripts_fts', 'strategy_imports', 'imported_strategy_performance']
    
    print(f"📊 Created tables: {tables}")
    
    for table in expected_tables:
        if table in tables:
            print(f"✅ Table '{table}' created successfully")
        else:
            print(f"❌ Table '{table}' missing")
            return False
    
    db.close()
    
    # Clean up
    os.remove(test_db_path)
    return True

def test_script_operations():
    """Test script save and retrieval operations"""
    print("\n🔄 Testing script operations...")
    
    test_db_path = "test_tv_scripts.db"
    
    # Remove existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = DB(test_db_path)
    
    # Test script metadata
    test_script = {
        'slug': 'ema-crossover-strategy',
        'title': 'EMA Crossover Strategy',
        'author': 'TestAuthor',
        'tags': 'ema, crossover, strategy, trending',
        'open_source': True,
        'url': 'https://www.tradingview.com/script/test123/',
        'description': 'Simple EMA crossover strategy for trending markets',
        'likes_count': 150,
        'uses_count': 75,
        'script_type': 'strategy'
    }
    
    # Test Pine Script code
    test_code = """
//@version=5
strategy("EMA Crossover", overlay=true)

// Input parameters
fast_length = input.int(21, "Fast EMA Length", minval=1)
slow_length = input.int(50, "Slow EMA Length", minval=1)

// Calculate EMAs
fast_ema = ta.ema(close, fast_length)
slow_ema = ta.ema(close, slow_length)

// Entry conditions
bullish_cross = ta.crossover(fast_ema, slow_ema)
bearish_cross = ta.crossunder(fast_ema, slow_ema)

// Strategy entries
if bullish_cross
    strategy.entry("Long", strategy.long)
if bearish_cross
    strategy.entry("Short", strategy.short)

// Plot EMAs
plot(fast_ema, "Fast EMA", color=color.blue)
plot(slow_ema, "Slow EMA", color=color.red)
"""
    
    # Save script
    success = db.save_script(test_script, test_code)
    if success:
        print("✅ Script saved successfully")
    else:
        print("❌ Failed to save script")
        db.close()
        os.remove(test_db_path)
        return False
    
    # Retrieve script
    retrieved = db.get(test_script['slug'])
    if retrieved:
        print("✅ Script retrieved successfully")
        print(f"   Title: {retrieved['title']}")
        print(f"   Author: {retrieved['author']}")
        print(f"   Open Source: {retrieved['open_source']}")
        print(f"   Code Length: {len(retrieved['code']) if retrieved['code'] else 0} characters")
    else:
        print("❌ Failed to retrieve script")
        db.close()
        os.remove(test_db_path)
        return False
    
    db.close()
    os.remove(test_db_path)
    return True

def test_fts5_search():
    """Test FTS5 full-text search capabilities"""
    print("\n🔍 Testing FTS5 search...")
    
    test_db_path = "test_tv_scripts.db"
    
    # Remove existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = DB(test_db_path)
    
    # Add multiple test scripts
    test_scripts = [
        {
            'slug': 'ema-crossover-strategy',
            'title': 'EMA Crossover Strategy',
            'author': 'TestAuthor1',
            'tags': 'ema, crossover, strategy, trending',
            'open_source': True,
            'url': 'https://example.com/1',
            'description': 'Simple EMA crossover strategy for trending markets',
            'script_type': 'strategy'
        },
        {
            'slug': 'macd-divergence-indicator',
            'title': 'MACD Divergence Indicator',
            'author': 'TestAuthor2',
            'tags': 'macd, divergence, indicator, oscillator',
            'open_source': True,
            'url': 'https://example.com/2',
            'description': 'MACD divergence detection for reversal signals',
            'script_type': 'indicator'
        },
        {
            'slug': 'smart-money-concepts',
            'title': 'Smart Money Concepts',
            'author': 'TestAuthor3',
            'tags': 'smc, structure, institutional, orderblock',
            'open_source': True,
            'url': 'https://example.com/3',
            'description': 'Smart Money Concepts with BOS and CHoCH detection',
            'script_type': 'indicator'
        }
    ]
    
    # Save test scripts
    for script in test_scripts:
        success = db.save_script(script, f"// Test Pine Script for {script['title']}")
        if not success:
            print(f"❌ Failed to save script: {script['slug']}")
            db.close()
            os.remove(test_db_path)
            return False
    
    print("✅ Test scripts saved successfully")
    
    # Test search queries
    test_queries = [
        ("ema crossover", 1, "Should find EMA crossover strategy"),
        ("macd", 1, "Should find MACD indicator"),
        ("smart money", 1, "Should find SMC script"),
        ("strategy", 1, "Should find strategies"),
        ("indicator", 2, "Should find indicators"),
        ("nonexistent", 0, "Should find nothing")
    ]
    
    for query, expected_count, description in test_queries:
        results = db.search(query, limit=10)
        actual_count = len(results)
        
        if actual_count == expected_count:
            print(f"✅ Search '{query}': Found {actual_count} results ({description})")
        else:
            print(f"❌ Search '{query}': Expected {expected_count}, got {actual_count} ({description})")
    
    # Test search with filters
    results = db.search("", limit=10, filters={'open_source_only': True})
    if len(results) == 3:
        print("✅ Filter test: Open source filter working correctly")
    else:
        print(f"❌ Filter test: Expected 3 open source scripts, got {len(results)}")
    
    db.close()
    os.remove(test_db_path)
    return True

def test_database_stats():
    """Test database statistics functionality"""
    print("\n📈 Testing database statistics...")
    
    test_db_path = "test_tv_scripts.db"
    
    # Remove existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = DB(test_db_path)
    
    # Add test data
    test_scripts = [
        {'slug': 'test1', 'title': 'Test 1', 'author': 'A1', 'tags': '', 'open_source': True, 'url': '', 'script_type': 'strategy'},
        {'slug': 'test2', 'title': 'Test 2', 'author': 'A2', 'tags': '', 'open_source': False, 'url': '', 'script_type': 'indicator'},
        {'slug': 'test3', 'title': 'Test 3', 'author': 'A3', 'tags': '', 'open_source': True, 'url': '', 'script_type': 'indicator'}
    ]
    
    for script in test_scripts:
        db.save_script(script, "// Test code" if script['open_source'] else None)
    
    # Get statistics
    stats = db.get_stats()
    
    print(f"📊 Database Statistics:")
    print(f"   Total scripts: {stats.get('total_scripts', 0)}")
    print(f"   Open source: {stats.get('open_source_scripts', 0)}")
    print(f"   With code: {stats.get('scripts_with_code', 0)}")
    print(f"   Script types: {stats.get('script_types', {})}")
    
    # Verify stats
    if (stats.get('total_scripts') == 3 and 
        stats.get('open_source_scripts') == 2 and
        stats.get('scripts_with_code') == 2):
        print("✅ Database statistics working correctly")
        result = True
    else:
        print("❌ Database statistics incorrect")
        result = False
    
    db.close()
    os.remove(test_db_path)
    return result

def main():
    """Run all database tests"""
    print("🧪 TradingView Scripts Database Test Suite")
    print("=" * 50)
    
    tests = [
        ("Database Creation", test_database_creation),
        ("Script Operations", test_script_operations),
        ("FTS5 Search", test_fts5_search),
        ("Database Stats", test_database_stats)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database schema is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)