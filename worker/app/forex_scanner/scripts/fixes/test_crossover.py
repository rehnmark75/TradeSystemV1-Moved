#!/usr/bin/env python3
"""
Test creating an alignment-based signal without crossover requirement
"""

import sys
sys.path.append('/app/forex_scanner')

def test_manual_signal_creation():
    """Test if we can create a signal manually based on alignment"""
    print("🧪 MANUAL SIGNAL CREATION TEST")
    print("=" * 60)
    
    try:
        from core.database import DatabaseManager
        from core.data_fetcher import DataFetcher
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        fetcher = DataFetcher(db, 'Europe/Stockholm')
        
        epic = 'CS.D.EURUSD.CEEM.IP'
        pair = 'EURUSD'
        
        # Get latest data
        df = fetcher.get_enhanced_data(epic, pair, timeframe='15m', lookback_hours=24)
        
        if df is None or len(df) < 2:
            print("❌ No data available")
            return
        
        # Get latest values
        latest = df.iloc[-1]
        current_price = latest['close']
        ema_9 = latest['ema_9']
        ema_21 = latest['ema_21']
        ema_200 = latest['ema_200']
        timestamp = latest['start_time']
        
        print(f"📊 Current Market State:")
        print(f"   Time: {timestamp}")
        print(f"   Price: {current_price:.5f}")
        print(f"   EMA 9: {ema_9:.5f}")
        print(f"   EMA 21: {ema_21:.5f}")
        print(f"   EMA 200: {ema_200:.5f}")
        
        # Check alignment conditions
        bull_conditions = {
            'price_above_ema9': current_price > ema_9,
            'ema9_above_ema21': ema_9 > ema_21,
            'ema9_above_ema200': ema_9 > ema_200,
            'ema21_above_ema200': ema_21 > ema_200
        }
        
        bear_conditions = {
            'price_below_ema9': current_price < ema_9,
            'ema21_above_ema9': ema_21 > ema_9,
            'ema200_above_ema9': ema_200 > ema_9,
            'ema200_above_ema21': ema_200 > ema_21
        }
        
        print(f"\n🐂 BULL CONDITIONS:")
        for condition, result in bull_conditions.items():
            status = "✅" if result else "❌"
            print(f"   {status} {condition}: {result}")
        
        print(f"\n🐻 BEAR CONDITIONS:")
        for condition, result in bear_conditions.items():
            status = "✅" if result else "❌"
            print(f"   {status} {condition}: {result}")
        
        # Determine signal
        if all(bull_conditions.values()):
            signal_type = 'BULL'
            print(f"\n✅ ALIGNMENT-BASED BULL SIGNAL DETECTED!")
        elif all(bear_conditions.values()):
            signal_type = 'BEAR'
            print(f"\n✅ ALIGNMENT-BASED BEAR SIGNAL DETECTED!")
        else:
            signal_type = None
            print(f"\n❌ No clear alignment detected")
        
        if signal_type:
            # Calculate simple confidence based on EMA separation
            ema_9_21_separation = abs(ema_9 - ema_21) * 10000  # in pips
            ema_9_200_separation = abs(ema_9 - ema_200) * 10000
            price_ema9_separation = abs(current_price - ema_9) * 10000
            
            # Base confidence + separation bonuses
            confidence = 0.5 + min(0.3, ema_9_21_separation * 0.02) + min(0.15, price_ema9_separation * 0.01)
            confidence = min(0.95, confidence)
            
            print(f"📊 Signal Details:")
            print(f"   Type: {signal_type}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   EMA 9-21 separation: {ema_9_21_separation:.1f} pips")
            print(f"   EMA 9-200 separation: {ema_9_200_separation:.1f} pips")
            print(f"   Price-EMA 9 separation: {price_ema9_separation:.1f} pips")
            
            # Create manual signal
            manual_signal = {
                'timestamp': timestamp,
                'epic': epic,
                'signal_type': signal_type,
                'strategy': 'alignment_based',
                'price': current_price,
                'confidence_score': confidence,
                'ema_9': ema_9,
                'ema_21': ema_21,
                'ema_200': ema_200,
                'trigger_reason': 'perfect_ema_alignment',
                'timeframe': '15m'
            }
            
            print(f"\n📋 MANUAL SIGNAL CREATED:")
            for key, value in manual_signal.items():
                if isinstance(value, float) and key != 'timestamp':
                    print(f"   {key}: {value:.5f}" if 'price' in key or 'ema' in key else f"   {key}: {value:.1%}" if 'confidence' in key else f"   {key}: {value}")
                else:
                    print(f"   {key}: {value}")
            
            print(f"\n💡 This proves alignment-based signals work!")
            print(f"   Your current strategy just requires crossovers in addition to alignment.")
            
            return manual_signal
        
    except Exception as e:
        print(f"❌ Manual signal test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def suggest_config_modifications():
    """Suggest specific config modifications to enable alignment signals"""
    print(f"\n🔧 CONFIGURATION MODIFICATIONS")
    print("=" * 60)
    
    print("Add these lines to your config.py to enable alignment-based signals:")
    print()
    print("```python")
    print("# Disable filters that require crossovers")
    print("ENABLE_BB_FILTER = False")
    print("ENABLE_BB_EXTREMES_FILTER = False")
    print()
    print("# Strategy settings for alignment-based signals")
    print("SIMPLE_EMA_STRATEGY = True")
    print("REQUIRE_NEW_CROSSOVER = False")
    print("REQUIRE_EMA_SEPARATION = False")
    print("REQUIRE_VOLUME_CONFIRMATION = False")
    print()
    print("# Lower confidence for testing")
    print("MIN_CONFIDENCE = 0.5")
    print("```")
    print()
    print("After adding these settings:")
    print("1. Save config.py")
    print("2. Run: python main.py debug --epic CS.D.EURUSD.CEEM.IP")
    print("3. You should see a BULL signal detected!")

def main():
    """Test manual signal creation"""
    print("Testing if alignment-based signals work without crossover requirement...\n")
    
    signal = test_manual_signal_creation()
    suggest_config_modifications()
    
    if signal:
        print(f"\n🎯 CONCLUSION:")
        print(f"✅ Perfect EMA alignment exists and can generate signals")
        print(f"✅ Manual signal creation works")
        print(f"❌ Current strategy requires crossover + alignment")
        print(f"💡 Solution: Modify config to allow alignment-based signals")
    else:
        print(f"\n🎯 CONCLUSION:")
        print(f"❌ No clear alignment detected at this moment")
        print(f"💡 Wait for better market alignment or check different timeframes")

if __name__ == "__main__":
    main()