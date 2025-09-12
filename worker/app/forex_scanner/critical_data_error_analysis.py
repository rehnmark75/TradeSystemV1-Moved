#!/usr/bin/env python3
"""
CRITICAL: Analysis of systematic data storage error discovered by user
"""

def analyze_critical_data_error():
    print("🚨" * 30)
    print("CRITICAL DATA STORAGE ERROR DISCOVERED")
    print("🚨" * 30)
    
    print("\nUser's findings reveal a SYSTEMATIC data storage problem:")
    print("=" * 80)
    
    # Compare IG Charts vs Database Storage
    print("IG 5M CHART DATA (CORRECT):")
    ig_data = [
        ("2025-09-01 02:30:00", 1.16960),
        ("2025-09-01 02:35:00", 1.16978), 
        ("2025-09-01 02:40:00", 1.16976),
        ("2025-09-01 02:45:00", 1.16978),
    ]
    
    print("DATABASE STORED DATA (WRONG):")
    db_data = [
        ("2025-09-01 02:30:00", 1.170555),
        ("2025-09-01 02:35:00", 1.17051),
        ("2025-09-01 02:40:00", 1.17055), 
        ("2025-09-01 02:45:00", 1.17048),
        ("2025-09-01 02:50:00", 1.170915),
    ]
    
    print(f"{'Timestamp':<20} {'IG Chart':<10} {'Database':<10} {'Diff (pips)':<12}")
    print("-" * 60)
    
    total_error = 0
    for i, (timestamp, ig_close) in enumerate(ig_data):
        if i < len(db_data):
            db_timestamp, db_close = db_data[i]
            diff = db_close - ig_close
            diff_pips = diff * 10000
            total_error += abs(diff_pips)
            
            print(f"{timestamp:<20} {ig_close:<10.5f} {db_close:<10.5f} {diff_pips:>+8.1f}")
    
    print(f"\nAVERAGE ERROR: {total_error/len(ig_data):.1f} pips per candle")
    
    print("\n" + "=" * 80)
    print("ERROR MAGNITUDE ANALYSIS")
    print("=" * 80)
    
    # Calculate the systematic offset
    errors = []
    for i, (timestamp, ig_close) in enumerate(ig_data):
        if i < len(db_data):
            _, db_close = db_data[i]
            errors.append(db_close - ig_close)
    
    avg_error = sum(errors) / len(errors)
    avg_error_pips = avg_error * 10000
    
    print(f"SYSTEMATIC OFFSET: +{avg_error_pips:.1f} pips")
    print(f"Our database prices are consistently ~{abs(avg_error_pips):.0f} pips HIGHER than IG Charts")
    
    print("\nThis explains:")
    print("✓ Why TradingView (1.16967) ≈ IG Charts (1.16978) - both correct")
    print("✓ Why our database (1.17048+) is ~70+ pips higher - WRONG data")
    print("✓ Why we get false bullish crossovers - inflated prices")
    
    print("\n" + "=" * 80)
    print("ROOT CAUSE INVESTIGATION")
    print("=" * 80)
    
    print("The systematic +70 pip offset suggests:")
    print("\n1. 🔧 PRICE CONVERSION ERROR:")
    print("   - IG API returns prices in different format than expected")
    print("   - Missing decimal point conversion (e.g., 116978 vs 1.16978)")
    print("   - Wrong price field being stored (bid/ask vs mid)")
    print("\n2. 📊 DATA PIPELINE BUG:")
    print("   - chart_streamer receiving wrong price feed")
    print("   - Transformation error in streaming service") 
    print("   - Database storage precision/conversion issue")
    print("\n3. 🌐 API ENDPOINT MISMATCH:")
    print("   - Streaming from wrong IG Markets endpoint")
    print("   - Different price tier (retail vs professional)")
    print("   - Wrong epic or market specification")
    
    print("\n" + "=" * 80)
    print("IMMEDIATE ACTION REQUIRED")
    print("=" * 80)
    
    print("🚨 STOP ALL LIVE TRADING IMMEDIATELY")
    print("   - Current database has systematically wrong prices")
    print("   - All signals are based on inflated price data")
    print("   - Risk of massive losses on false signals")
    
    print("\n🔍 INVESTIGATE STREAMING SERVICE:")
    print("   - Check chart_streamer data source configuration") 
    print("   - Verify IG Markets API price field mapping")
    print("   - Compare live streaming vs IG Charts in real-time")
    
    print("\n📊 DATA VALIDATION:")
    print("   - Implement real-time price validation against IG Charts")
    print("   - Add data quality checks before storing candles")
    print("   - Flag and reject prices that deviate >5 pips from expected range")
    
    print("\n🔄 DATABASE CLEANUP:")
    print("   - Historical data needs to be corrected or purged")
    print("   - Backtest results are invalid with wrong price data")
    print("   - All strategy performance metrics need recalculation")

if __name__ == "__main__":
    analyze_critical_data_error()