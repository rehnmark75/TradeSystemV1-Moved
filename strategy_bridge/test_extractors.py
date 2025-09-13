#!/usr/bin/env python3
"""
Test script for Pine Script extractors and mappers

Tests the extraction of indicators, signals, and conversion to TradeSystemV1 configs.
"""

import sys
import json
from pathlib import Path

# Add strategy_bridge to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_pine import extract_inputs, extract_signals, analyze_pine_script
from map_to_python import to_config, generate_config_file_content

# Test Pine Script samples
TEST_SCRIPTS = {
    "ema_crossover": '''
//@version=5
strategy("EMA Crossover Strategy", overlay=true)

// Input parameters
fast_length = input.int(21, "Fast EMA Length", minval=1)
slow_length = input.int(50, "Slow EMA Length", minval=1)
trend_length = input.int(200, "Trend EMA Length", minval=1)

// Calculate EMAs
fast_ema = ta.ema(close, fast_length)
slow_ema = ta.ema(close, slow_length) 
trend_ema = ta.ema(close, trend_length)

// Entry conditions
bullish_cross = ta.crossover(fast_ema, slow_ema) and close > trend_ema
bearish_cross = ta.crossunder(fast_ema, slow_ema) and close < trend_ema

// Strategy entries
if bullish_cross
    strategy.entry("Long", strategy.long)
if bearish_cross
    strategy.entry("Short", strategy.short)

// Plot EMAs
plot(fast_ema, "Fast EMA", color=color.blue)
plot(slow_ema, "Slow EMA", color=color.red)
plot(trend_ema, "Trend EMA", color=color.orange)
''',

    "macd_rsi": '''
//@version=5
indicator("MACD RSI Combo", shorttitle="MACD-RSI")

// MACD Settings
fast_length = input.int(12, "MACD Fast Length")
slow_length = input.int(26, "MACD Slow Length")
signal_length = input.int(9, "MACD Signal Length")

// RSI Settings
rsi_length = input.int(14, "RSI Length")
rsi_overbought = input.float(70.0, "RSI Overbought Level")
rsi_oversold = input.float(30.0, "RSI Oversold Level")

// Calculate MACD
[macd_line, signal_line, hist] = ta.macd(close, fast_length, slow_length, signal_length)

// Calculate RSI
rsi_value = ta.rsi(close, rsi_length)

// Signals
bullish_macd = ta.crossover(macd_line, signal_line)
bearish_macd = ta.crossunder(macd_line, signal_line)
rsi_oversold_signal = rsi_value < rsi_oversold
rsi_overbought_signal = rsi_value > rsi_overbought

// Combined signals
buy_signal = bullish_macd and rsi_oversold_signal
sell_signal = bearish_macd and rsi_overbought_signal

// Plots
plot(macd_line, "MACD", color=color.blue)
plot(signal_line, "Signal", color=color.red)
plot(hist, "Histogram", color=color.gray, style=plot.style_histogram)
''',

    "smc_strategy": '''
//@version=5
strategy("Smart Money Concepts", overlay=true)

// SMC Settings
lookback = input.int(50, "Structure Lookback")
ob_threshold = input.float(0.5, "Order Block Threshold")

// Fair Value Gap detection
gap_threshold = input.float(0.1, "FVG Threshold %")

// Higher timeframe analysis  
htf_timeframe = input.timeframe("1H", "Higher Timeframe")

// Calculate structure breaks
high_break = high > ta.highest(high[1], lookback)
low_break = low < ta.lowest(low[1], lookback)

// BOS detection
bos_bullish = high_break and close > open
bos_bearish = low_break and close < open

// Change of Character (CHoCH)
choch_bull = low_break and close > open  // Failed low break becomes CHoCH
choch_bear = high_break and close < open // Failed high break becomes CHoCH

// Order Block detection
bullish_ob = low_break and (high - low) > (ta.atr(14) * ob_threshold)
bearish_ob = high_break and (high - low) > (ta.atr(14) * ob_threshold)

// Fair Value Gap detection
fvg_up = (low[2] > high) and (high[1] < low[2]) // Gap between bars
fvg_down = (high[2] < low) and (low[1] > high[2]) // Gap between bars

// HTF confirmation
htf_trend = request.security(syminfo.tickerid, htf_timeframe, ta.ema(close, 50))
htf_bullish = close > htf_trend
htf_bearish = close < htf_trend

// Entry conditions
long_entry = bos_bullish and bullish_ob and htf_bullish
short_entry = bos_bearish and bearish_ob and htf_bearish

if long_entry
    strategy.entry("Long", strategy.long)
if short_entry
    strategy.entry("Short", strategy.short)

// Visualization
plotshape(bos_bullish, "BOS Bull", shape.triangleup, location.belowbar, color.green)
plotshape(bos_bearish, "BOS Bear", shape.triangledown, location.abovebar, color.red)
''',

    "bollinger_scalping": '''
//@version=5
strategy("Bollinger Scalping", overlay=true)

// Bollinger Bands settings
bb_length = input.int(20, "BB Length")
bb_mult = input.float(2.0, "BB Multiplier")

// Scalping EMAs
ema_fast = input.int(5, "Fast EMA")
ema_slow = input.int(13, "Slow EMA")

// Calculate indicators
[bb_mid, bb_upper, bb_lower] = ta.bb(close, bb_length, bb_mult)
ema5 = ta.ema(close, ema_fast)
ema13 = ta.ema(close, ema_slow)

// Bollinger squeeze detection
squeeze = (bb_upper - bb_lower) < ta.atr(14) * 0.5

// Entry conditions
bb_long = close < bb_lower and ta.crossover(ema5, ema13)
bb_short = close > bb_upper and ta.crossunder(ema5, ema13)

// Scalping signals with tight stops
if bb_long and not squeeze
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", "Long", stop=close - ta.atr(14), limit=close + ta.atr(14) * 1.5)

if bb_short and not squeeze
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", "Short", stop=close + ta.atr(14), limit=close - ta.atr(14) * 1.5)

// Plots
plot(bb_upper, "BB Upper", color=color.blue)
plot(bb_mid, "BB Mid", color=color.orange)
plot(bb_lower, "BB Lower", color=color.blue)
plot(ema5, "EMA 5", color=color.green)
plot(ema13, "EMA 13", color=color.red)
'''
}

def test_input_extraction():
    """Test input parameter extraction"""
    print("🔧 Testing input parameter extraction...")
    
    for script_name, code in TEST_SCRIPTS.items():
        print(f"\n📝 Testing {script_name}:")
        inputs = extract_inputs(code)
        
        print(f"   Extracted {len(inputs)} inputs:")
        for inp in inputs:
            print(f"     - {inp['type']} {inp['label']}: {inp['default']}")
        
        # Validate extraction
        if script_name == "ema_crossover":
            expected_inputs = ["Fast EMA Length", "Slow EMA Length", "Trend EMA Length"]
            found_labels = [inp['label'] for inp in inputs]
            if all(exp in found_labels for exp in expected_inputs):
                print("   ✅ EMA inputs extracted correctly")
            else:
                print("   ❌ Missing expected EMA inputs")
        
        elif script_name == "macd_rsi":
            expected_labels = ["MACD Fast Length", "RSI Length"]
            found_labels = [inp['label'] for inp in inputs]
            if any(exp in found_labels for exp in expected_labels):
                print("   ✅ MACD/RSI inputs found")
            else:
                print("   ❌ Missing MACD/RSI inputs")

def test_signal_extraction():
    """Test signal and pattern extraction"""
    print("\n🔍 Testing signal extraction...")
    
    for script_name, code in TEST_SCRIPTS.items():
        print(f"\n📡 Testing {script_name}:")
        signals = extract_signals(code)
        
        print(f"   Strategy Type: {signals.get('strategy_type', 'unknown')}")
        print(f"   Complexity Score: {signals.get('complexity_score', 0.0)}")
        
        if signals.get('ema_periods'):
            print(f"   EMA Periods: {signals['ema_periods']}")
        
        if signals.get('macd'):
            macd = signals['macd']
            print(f"   MACD: {macd['fast']}/{macd['slow']}/{macd['signal']}")
        
        if signals.get('rsi_periods'):
            print(f"   RSI Periods: {signals['rsi_periods']}")
        
        if signals.get('bollinger_bands'):
            bb = signals['bollinger_bands']
            print(f"   Bollinger Bands: {bb['length']} period, {bb['multiplier']} std dev")
        
        print(f"   Crossovers: Up={signals.get('has_cross_up', False)}, Down={signals.get('has_cross_down', False)}")
        print(f"   SMC: {signals.get('mentions_smc', False)}")
        print(f"   FVG: {signals.get('mentions_fvg', False)}")
        print(f"   Volume: {signals.get('mentions_volume', False)}")
        
        # Validate specific patterns
        if script_name == "ema_crossover":
            if signals.get('ema_periods') and signals.get('has_cross_up'):
                print("   ✅ EMA crossover pattern detected correctly")
            else:
                print("   ❌ Failed to detect EMA crossover pattern")
        
        elif script_name == "smc_strategy":
            if signals.get('mentions_smc') and signals.get('mentions_fvg'):
                print("   ✅ SMC patterns detected correctly")
            else:
                print("   ❌ Failed to detect SMC patterns")

def test_config_generation():
    """Test configuration generation"""
    print("\n⚙️ Testing configuration generation...")
    
    for script_name, code in TEST_SCRIPTS.items():
        print(f"\n🔧 Testing {script_name}:")
        
        # Extract patterns
        inputs = extract_inputs(code)
        signals = extract_signals(code)
        
        # Generate configuration
        config = to_config(inputs, signals, f"TV_{script_name.title()}")
        
        print(f"   Generated config for: {config.get('name', 'Unknown')}")
        print(f"   Modules: {list(config.get('modules', {}).keys())}")
        print(f"   Presets: {list(config.get('presets', {}).keys())}")
        print(f"   Rules: {len(config.get('rules', []))}")
        
        # Validate configuration structure
        required_keys = ['name', 'provenance', 'modules', 'rules', 'presets']
        if all(key in config for key in required_keys):
            print("   ✅ Configuration structure valid")
        else:
            missing = [key for key in required_keys if key not in config]
            print(f"   ❌ Missing configuration keys: {missing}")
        
        # Test specific configurations
        if script_name == "ema_crossover":
            if 'ema' in config.get('modules', {}):
                ema_config = config['modules']['ema']
                if ema_config.get('periods') and len(ema_config['periods']) >= 2:
                    print("   ✅ EMA module configured correctly")
                else:
                    print("   ❌ EMA module configuration incomplete")
        
        elif script_name == "macd_rsi":
            modules = config.get('modules', {})
            if 'macd' in modules and 'rsi' in modules:
                print("   ✅ MACD and RSI modules configured")
            else:
                print("   ❌ Missing MACD or RSI module")

def test_config_file_generation():
    """Test configuration file content generation"""
    print("\n📄 Testing config file generation...")
    
    # Use EMA crossover as test case
    code = TEST_SCRIPTS["ema_crossover"]
    inputs = extract_inputs(code)
    signals = extract_signals(code)
    config = to_config(inputs, signals, "TV_EMA_Crossover")
    
    # Generate file content
    file_content = generate_config_file_content(config)
    
    print(f"   Generated file content: {len(file_content)} characters")
    
    # Check for key components
    required_patterns = [
        "TV_EMA_CROSSOVER_STRATEGY = True",
        "TV_EMA_CROSSOVER_STRATEGY_CONFIG",
        "def get_tv_ema_crossover_config_for_epic",
        "def validate_tv_ema_crossover_config"
    ]
    
    all_found = True
    for pattern in required_patterns:
        if pattern in file_content:
            print(f"   ✅ Found: {pattern}")
        else:
            print(f"   ❌ Missing: {pattern}")
            all_found = False
    
    if all_found:
        print("   ✅ Config file structure complete")
    else:
        print("   ❌ Config file structure incomplete")
    
    # Save test file for inspection
    test_file_path = Path(__file__).parent / "test_generated_config.py"
    with open(test_file_path, 'w') as f:
        f.write(file_content)
    print(f"   💾 Test config saved to: {test_file_path}")

def test_complete_analysis():
    """Test complete analysis pipeline"""
    print("\n🔬 Testing complete analysis pipeline...")
    
    code = TEST_SCRIPTS["smc_strategy"]
    analysis = analyze_pine_script(code)
    
    print(f"   Analysis complete: {analysis.get('analysis_complete', False)}")
    print(f"   Code stats: {analysis.get('code_stats', {})}")
    
    inputs = analysis.get('inputs', [])
    signals = analysis.get('signals', {})
    rules = analysis.get('rules', [])
    
    print(f"   Extracted: {len(inputs)} inputs, {len(signals)} signal types, {len(rules)} rules")
    
    if analysis.get('analysis_complete') and inputs and signals:
        print("   ✅ Complete analysis successful")
    else:
        print("   ❌ Complete analysis failed")

def main():
    """Run all extractor tests"""
    print("🧪 Pine Script Extractor Test Suite")
    print("=" * 60)
    
    tests = [
        ("Input Extraction", test_input_extraction),
        ("Signal Extraction", test_signal_extraction),
        ("Config Generation", test_config_generation),
        ("Config File Generation", test_config_file_generation),
        ("Complete Analysis", test_complete_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            test_func()
            print(f"✅ {test_name}: COMPLETED")
            passed += 1
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests completed")
    
    if passed == total:
        print("🎉 All extractor tests completed successfully!")
        return True
    else:
        print("⚠️  Some tests encountered issues. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)