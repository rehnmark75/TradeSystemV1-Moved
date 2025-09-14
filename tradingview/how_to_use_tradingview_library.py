#!/usr/bin/env python3
"""
How to Use Your TradingView Library - Practical Examples

This guide shows you exactly how to leverage the 15 TradingView scripts 
(5 strategies + 10 indicators) in your TradeSystemV1 trading system.
"""

import sys
import sqlite3
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / "worker" / "app" / "forex_scanner"))

def example_1_search_and_discover():
    """Example 1: Search and discover relevant scripts for your trading style"""
    print("🔍 Example 1: Search and Discover Scripts")
    print("=" * 50)
    
    try:
        from configdata.strategies.tradingview_integration import TradingViewIntegration
        
        integration = TradingViewIntegration()
        
        # Search examples for different trading styles
        search_examples = [
            ("Scalping setup", "scalping"),
            ("Momentum trading", "momentum RSI MACD"), 
            ("Trend following", "EMA trend following"),
            ("Volume analysis", "volume VWAP OBV"),
            ("Volatility trading", "volatility ATR Bollinger")
        ]
        
        for style, query in search_examples:
            print(f"\n🎯 {style}:")
            results = integration.search_strategies(query, limit=3)
            
            for i, result in enumerate(results, 1):
                title = result.get('title', 'N/A')
                script_type = result.get('strategy_type', 'unknown')
                likes = result.get('likes', 0)
                print(f"   {i}. {title} ({script_type}, {likes:,} likes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Search example failed: {e}")
        return False

def example_2_analyze_indicator_parameters():
    """Example 2: Extract parameters from top indicators for your strategies"""
    print("\n📊 Example 2: Extract Indicator Parameters")
    print("=" * 50)
    
    try:
        # Connect to database to get indicator code
        db_path = Path(__file__).parent / "data" / "tvscripts.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get top indicators with their parameters
        cursor.execute("""
            SELECT title, code FROM scripts 
            WHERE strategy_type = 'indicator' 
            ORDER BY likes DESC 
            LIMIT 3
        """)
        
        indicators = cursor.fetchall()
        
        print("🔧 Parameter extraction from top indicators:")
        
        for title, code in indicators:
            print(f"\n📈 {title}:")
            
            # Extract key parameters (simple regex patterns)
            import re
            
            # Find input parameters
            inputs = re.findall(r'input\.(int|float)\((\d+(?:\.\d+)?)', code)
            lengths = re.findall(r'length\s*=\s*input\.[a-z]+\((\d+)', code)
            
            if inputs:
                print("   Parameters found:")
                for param_type, value in inputs[:3]:  # Show first 3
                    print(f"     - {param_type}: {value}")
            
            if lengths:
                print(f"   Standard length: {lengths[0]}")
            
            # Extract what it measures
            if "momentum" in code.lower():
                print("   📊 Type: Momentum indicator")
            elif "volume" in code.lower():
                print("   📊 Type: Volume indicator")
            elif "volatility" in code.lower() or "deviation" in code.lower():
                print("   📊 Type: Volatility indicator")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Parameter extraction failed: {e}")
        return False

def example_3_import_strategy_to_existing_config():
    """Example 3: Import a TradingView strategy into your existing EMA config"""
    print("\n🔄 Example 3: Import Strategy to Your EMA Config")
    print("=" * 50)
    
    print("💡 Steps to import 'Triple EMA System' into your EMA strategy:")
    print()
    print("1️⃣ Analyze the strategy parameters:")
    print("   - EMA periods: 8, 21, 55")
    print("   - Entry: Close crosses above EMA1 in uptrend")
    print("   - Trend filter: EMA1 > EMA2 > EMA3")
    print()
    print("2️⃣ Map to your EMA config (worker/app/forex_scanner/configdata/strategies/config_ema_strategy.py):")
    print("   - short: 8 (fast EMA)")
    print("   - long: 21 (medium EMA)")  
    print("   - trend: 55 (slow EMA for trend filter)")
    print()
    print("3️⃣ Add new preset:")
    
    sample_preset = '''
    'triple_ema_tv': {
        'short': 8,
        'long': 21, 
        'trend': 55,
        'description': 'TradingView Triple EMA System - proven community strategy',
        'best_for': ['trending_markets', 'swing_trading'],
        'confidence_threshold': 0.65,
        'stop_loss_pips': 20,
        'take_profit_pips': 40,
        'provenance': {
            'source': 'tradingview',
            'url': 'https://www.tradingview.com/script/triple-ema-system/',
            'likes': 420,
            'imported_at': 'manual_import'
        }
    }'''
    
    print(sample_preset)
    print()
    print("4️⃣ Test with your scanner:")
    print("   docker exec forex_scanner python -m forex_scanner.main --strategy ema --preset triple_ema_tv")
    
    return True

def example_4_enhance_existing_strategies():
    """Example 4: Enhance your existing strategies with TradingView insights"""
    print("\n⚡ Example 4: Enhance Your Existing Strategies")
    print("=" * 50)
    
    enhancements = [
        {
            'your_strategy': 'EMA Strategy',
            'tv_insights': [
                'Add RSI confirmation (RSI < 70 for long, RSI > 30 for short)',
                'Use VWAP as trend filter (price above VWAP = uptrend)',
                'Add ATR-based position sizing (use 1.5x ATR for stop loss)'
            ]
        },
        {
            'your_strategy': 'MACD Strategy', 
            'tv_insights': [
                'Add Stochastic confirmation (avoid trades when Stoch > 80 or < 20)',
                'Use Bollinger Bands squeeze detection for entry timing',
                'Add volume confirmation with OBV trend alignment'
            ]
        },
        {
            'your_strategy': 'SMC Strategy',
            'tv_insights': [
                'Add VWAP levels as additional support/resistance',
                'Use ATR for dynamic stop loss placement',
                'Add RSI divergence detection for reversal signals'
            ]
        }
    ]
    
    for enhancement in enhancements:
        print(f"\n📈 {enhancement['your_strategy']} Enhancements:")
        for i, insight in enumerate(enhancement['tv_insights'], 1):
            print(f"   {i}. {insight}")
    
    print("\n💡 Implementation approach:")
    print("   - Add these as optional filters in your strategy configs")
    print("   - Create 'enhanced' presets with TradingView insights")
    print("   - Backtest performance comparison: original vs enhanced")
    
    return True

def example_5_create_hybrid_strategies():
    """Example 5: Create hybrid strategies combining multiple TradingView scripts"""
    print("\n🔬 Example 5: Create Hybrid Strategies")
    print("=" * 50)
    
    hybrid_strategies = [
        {
            'name': 'EMA-RSI-VWAP Hybrid',
            'components': [
                'EMA Crossover (trend direction)',
                'RSI (momentum confirmation)', 
                'VWAP (institutional level)'
            ],
            'logic': 'Enter long when: EMA cross up + RSI < 70 + Price above VWAP',
            'config_target': 'ema_strategy.py'
        },
        {
            'name': 'MACD-BB-Volume Hybrid',
            'components': [
                'MACD (trend momentum)',
                'Bollinger Bands (volatility)',
                'OBV (volume confirmation)'
            ],
            'logic': 'Enter when: MACD cross + BB squeeze release + OBV trending up',
            'config_target': 'macd_strategy.py'
        },
        {
            'name': 'Multi-Oscillator Scalp',
            'components': [
                'Stochastic (entry timing)',
                'Williams %R (overbought/oversold)',
                'ATR (volatility filter)'
            ],
            'logic': 'Quick scalps when all oscillators align + high ATR',
            'config_target': 'New scalping config'
        }
    ]
    
    for strategy in hybrid_strategies:
        print(f"\n🎯 {strategy['name']}:")
        print(f"   Components: {', '.join(strategy['components'])}")
        print(f"   Logic: {strategy['logic']}")
        print(f"   Target: {strategy['config_target']}")
    
    return True

def example_6_optimization_insights():
    """Example 6: Use TradingView data for parameter optimization"""
    print("\n🎯 Example 6: Parameter Optimization Insights")
    print("=" * 50)
    
    try:
        # Get parameter insights from database
        db_path = Path(__file__).parent / "data" / "tvscripts.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        print("📊 Popular parameter ranges from community scripts:")
        
        # Analyze EMA periods used
        cursor.execute("SELECT title, code FROM scripts WHERE title LIKE '%EMA%' OR code LIKE '%ema%'")
        ema_scripts = cursor.fetchall()
        
        print("\n📈 EMA Periods Analysis:")
        ema_periods = []
        for title, code in ema_scripts:
            import re
            periods = re.findall(r'ema\([^,]+,\s*(\d+)', code, re.IGNORECASE)
            periods.extend(re.findall(r'input\.int\((\d+)', code))
            ema_periods.extend([int(p) for p in periods if p.isdigit()])
        
        if ema_periods:
            unique_periods = sorted(set(ema_periods))
            print(f"   Common EMA periods: {unique_periods[:10]}")  # Top 10
            print(f"   Most popular range: {min(unique_periods)} - {max(unique_periods)}")
        
        # RSI analysis
        cursor.execute("SELECT code FROM scripts WHERE title LIKE '%RSI%'")
        rsi_scripts = cursor.fetchall()
        
        print("\n📊 RSI Settings Analysis:")
        for code_tuple in rsi_scripts[:1]:  # Just first one
            code = code_tuple[0]
            overbought = re.findall(r'overbought.*?(\d+)', code, re.IGNORECASE)
            oversold = re.findall(r'oversold.*?(\d+)', code, re.IGNORECASE)
            
            if overbought:
                print(f"   Overbought level: {overbought[0]}")
            if oversold:
                print(f"   Oversold level: {oversold[0]}")
        
        conn.close()
        
        print("\n💡 Optimization recommendations:")
        print("   - Test EMA periods in 8-21 range for short-term")
        print("   - Use 50-200 range for trend identification")
        print("   - RSI 70/30 levels are community standard")
        print("   - Consider dynamic ATR-based stop losses")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization insights failed: {e}")
        return False

def example_7_practical_implementation():
    """Example 7: Step-by-step implementation guide"""
    print("\n🛠️ Example 7: Practical Implementation Steps")
    print("=" * 50)
    
    print("🚀 Quick Start Guide:")
    print()
    print("1️⃣ IMMEDIATE USE (5 minutes):")
    print("   • Search for scripts: python3 test_full_integration.py")
    print("   • Browse by popularity to find proven parameters")
    print("   • Copy parameter values into your existing configs")
    print()
    print("2️⃣ ENHANCE EXISTING STRATEGIES (30 minutes):")
    print("   • Pick your best performing strategy")
    print("   • Find similar TradingView script")
    print("   • Add 1-2 additional filters from the script")
    print("   • Create new preset with 'tv_enhanced' suffix")
    print()
    print("3️⃣ CREATE NEW STRATEGIES (2 hours):")
    print("   • Use high-popularity indicators (>10k likes)")
    print("   • Extract exact parameters and logic")
    print("   • Implement in new config file")
    print("   • Backtest against your existing strategies")
    print()
    print("4️⃣ ADVANCED OPTIMIZATION (ongoing):")
    print("   • Use parameter ranges from popular scripts")
    print("   • A/B test community parameters vs your optimized ones")
    print("   • Combine multiple indicators from different scripts")
    print("   • Track performance: original vs TradingView-enhanced")
    
    print("\n📁 Files you can modify immediately:")
    config_files = [
        "worker/app/forex_scanner/configdata/strategies/config_ema_strategy.py",
        "worker/app/forex_scanner/configdata/strategies/config_macd_strategy.py", 
        "worker/app/forex_scanner/configdata/strategies/config_smc_strategy.py"
    ]
    
    for file in config_files:
        print(f"   • {file}")
    
    return True

def main():
    """Run all practical examples"""
    print("🎯 How to Use Your TradingView Library")
    print("=" * 80)
    print("You now have 15 TradingView scripts (5 strategies + 10 indicators)")
    print("Here's exactly how to leverage them in your trading system:")
    print()
    
    examples = [
        example_1_search_and_discover,
        example_2_analyze_indicator_parameters,
        example_3_import_strategy_to_existing_config,
        example_4_enhance_existing_strategies,
        example_5_create_hybrid_strategies,
        example_6_optimization_insights,
        example_7_practical_implementation
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 Your TradingView library is ready to enhance your trading system!")
    print("💡 Start with Example 3 - it's the fastest way to see immediate results")

if __name__ == "__main__":
    main()