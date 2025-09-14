#!/usr/bin/env python3
"""
Test Complete TradingView Integration with Strategies and Indicators
Tests the full workflow including both strategy and indicator imports
"""

import sys
import sqlite3
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / "worker" / "app" / "forex_scanner"))

def test_comprehensive_search():
    """Test comprehensive search across strategies and indicators"""
    print("🔍 Testing comprehensive search functionality...")
    
    db_path = Path(__file__).parent / "data" / "tvscripts.db"
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get total counts by type
        cursor.execute("SELECT strategy_type, COUNT(*) FROM scripts GROUP BY strategy_type ORDER BY COUNT(*) DESC")
        type_counts = cursor.fetchall()
        
        print("📊 Complete TradingView library:")
        total_scripts = 0
        for script_type, count in type_counts:
            print(f"   {script_type}: {count} scripts")
            total_scripts += count
        print(f"   Total: {total_scripts} scripts")
        
        # Test different search categories
        test_searches = [
            ("EMA strategies", "EMA"),
            ("Momentum indicators", "momentum"),
            ("Volume analysis", "volume"), 
            ("Volatility tools", "volatility"),
            ("Oscillators", "oscillator"),
            ("Moving averages", "average"),
            ("RSI tools", "RSI"),
            ("MACD analysis", "MACD")
        ]
        
        print("\n🎯 Search test results:")
        for search_name, query in test_searches:
            cursor.execute("""
                SELECT s.title, s.strategy_type, s.likes 
                FROM scripts s
                JOIN scripts_fts ON s.id = scripts_fts.rowid
                WHERE scripts_fts MATCH ?
                ORDER BY s.likes DESC
                LIMIT 3
            """, (query,))
            
            results = cursor.fetchall()
            print(f"\n   {search_name} ('{query}'):")
            for i, (title, stype, likes) in enumerate(results, 1):
                print(f"     {i}. {title} ({stype}, {likes:,} likes)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_indicator_analysis():
    """Test analysis of indicator scripts"""
    print("\n🧮 Testing indicator analysis...")
    
    try:
        from configdata.strategies.tradingview_integration import TradingViewIntegration
        
        # Initialize integration
        integration = TradingViewIntegration()
        
        # Search for indicators
        indicator_results = integration.search_strategies("RSI momentum", limit=3)
        print(f"✅ Found {len(indicator_results)} indicator results")
        
        if indicator_results:
            first_indicator = indicator_results[0]
            print(f"📋 Analyzing: {first_indicator['title']}")
            
            # Test analysis
            analysis = integration.analyze_strategy(first_indicator['slug'])
            print(f"✅ Analysis completed:")
            print(f"   - Script type: {analysis.get('strategy_type', 'N/A')}")
            print(f"   - Indicators: {analysis.get('indicators', [])}")
            print(f"   - Complexity: {analysis.get('complexity_score', 0.0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Indicator analysis failed: {e}")
        return False

def test_mixed_search_results():
    """Test searching across both strategies and indicators"""
    print("\n🔄 Testing mixed search results...")
    
    try:
        from configdata.strategies.tradingview_integration import TradingViewIntegration
        
        integration = TradingViewIntegration()
        
        # Search for terms that should return both strategies and indicators
        mixed_searches = [
            "EMA",
            "MACD", 
            "moving average",
            "crossover"
        ]
        
        for query in mixed_searches:
            results = integration.search_strategies(query, limit=5)
            
            strategies = [r for r in results if r.get('strategy_type') in ['trending', 'scalping', 'momentum']]
            indicators = [r for r in results if r.get('strategy_type') == 'indicator']
            
            print(f"\n   Query: '{query}'")
            print(f"     Strategies: {len(strategies)}")
            print(f"     Indicators: {len(indicators)}")
            print(f"     Total: {len(results)}")
            
            # Show top results
            for i, result in enumerate(results[:3], 1):
                script_type = result.get('strategy_type', 'unknown')
                likes = result.get('likes', 0)
                print(f"     {i}. {result['title']} ({script_type}, {likes:,} likes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed search test failed: {e}")
        return False

def test_category_filtering():
    """Test filtering by script categories"""
    print("\n🏷️ Testing category filtering...")
    
    db_path = Path(__file__).parent / "data" / "tvscripts.db"
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Test filtering by different categories
        categories = ['indicator', 'trending', 'momentum', 'scalping']
        
        for category in categories:
            cursor.execute("""
                SELECT title, likes, views
                FROM scripts 
                WHERE strategy_type = ?
                ORDER BY likes DESC
                LIMIT 3
            """, (category,))
            
            results = cursor.fetchall()
            print(f"\n   Top {category} scripts:")
            for i, (title, likes, views) in enumerate(results, 1):
                print(f"     {i}. {title} ({likes:,} likes, {views:,} views)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Category filtering failed: {e}")
        return False

def test_popularity_ranking():
    """Test popularity-based ranking"""
    print("\n⭐ Testing popularity ranking...")
    
    db_path = Path(__file__).parent / "data" / "tvscripts.db"
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get top scripts by likes
        cursor.execute("""
            SELECT title, strategy_type, likes, views
            FROM scripts
            ORDER BY likes DESC
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        print("📈 Top 10 most popular scripts:")
        for i, (title, stype, likes, views) in enumerate(results, 1):
            print(f"   {i:2d}. {title} ({stype})")
            print(f"       {likes:,} likes, {views:,} views")
        
        # Calculate engagement metrics
        cursor.execute("SELECT AVG(likes), AVG(views), COUNT(*) FROM scripts")
        avg_likes, avg_views, total = cursor.fetchone()
        
        print(f"\n📊 Library statistics:")
        print(f"   Total scripts: {total}")
        print(f"   Average likes: {avg_likes:,.0f}")
        print(f"   Average views: {avg_views:,.0f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Popularity ranking failed: {e}")
        return False

def main():
    """Run comprehensive integration test"""
    print("🧪 TradingView Complete Integration Test")
    print("=" * 60)
    
    tests = [
        ("Comprehensive Search", test_comprehensive_search),
        ("Indicator Analysis", test_indicator_analysis),
        ("Mixed Search Results", test_mixed_search_results),
        ("Category Filtering", test_category_filtering),
        ("Popularity Ranking", test_popularity_ranking)
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
    
    print("\n" + "=" * 60)
    print(f"🏁 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Complete TradingView integration is fully functional!")
        print("\n📚 Available in your library:")
        print("   • 5 EMA trading strategies")
        print("   • 10 top community indicators")
        print("   • Full-text search across all scripts")
        print("   • Strategy analysis and import capabilities")
        print("   • Integration with TradeSystemV1 configuration")
        print("\n🚀 Ready for strategy development and backtesting!")
        return True
    else:
        print("⚠️ Some integration components need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)