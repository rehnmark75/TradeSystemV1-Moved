#!/usr/bin/env python3
"""
Verify Trade 1162 - AUDUSD Break-Even Analysis

The user wants to verify trade 1162 specifically. This trade shows:
- ID: 1162
- Symbol: CS.D.AUDUSD.MINI.IP
- Entry: 0.66064 (BUY)
- Stop: 0.65814 (below entry = stop loss, not breakeven)
- IG minimum: 2 points
- moved_to_breakeven: false
- Status: expired

This trade NEVER moved to breakeven - it expired with stop loss below entry.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_trade_1162():
    """Analyze the specific trade 1162 that user mentioned"""

    print("🔍 TRADE 1162 ANALYSIS")
    print("=" * 50)

    # Trade 1162 data from database
    trade_data = {
        'id': 1162,
        'symbol': 'CS.D.AUDUSD.MINI.IP',
        'entry_price': 0.66064,
        'sl_price': 0.65814,
        'direction': 'BUY',
        'min_stop_distance_points': 2,
        'moved_to_breakeven': False,
        'status': 'expired'
    }

    print(f"Trade ID: {trade_data['id']}")
    print(f"Symbol: {trade_data['symbol']}")
    print(f"Direction: {trade_data['direction']}")
    print(f"Entry Price: {trade_data['entry_price']}")
    print(f"Stop Loss Price: {trade_data['sl_price']}")
    print(f"IG Minimum Distance: {trade_data['min_stop_distance_points']} points")
    print(f"Moved to Breakeven: {trade_data['moved_to_breakeven']}")
    print(f"Status: {trade_data['status']}")

    print("\n📊 ANALYSIS:")
    print("-" * 30)

    # Calculate stop distance
    point_value = 0.0001  # AUDUSD point value
    actual_stop_distance = abs(trade_data['entry_price'] - trade_data['sl_price']) / point_value

    print(f"Actual stop distance: {actual_stop_distance:.1f} points")

    # Determine if this was breakeven or initial stop
    if trade_data['direction'] == 'BUY':
        if trade_data['sl_price'] >= trade_data['entry_price']:
            breakeven_points = (trade_data['sl_price'] - trade_data['entry_price']) / point_value
            print(f"✅ MOVED TO BREAKEVEN: +{breakeven_points:.1f} points above entry")
        else:
            stop_loss_points = (trade_data['entry_price'] - trade_data['sl_price']) / point_value
            print(f"❌ INITIAL STOP LOSS: -{stop_loss_points:.1f} points below entry")

    print("\n🎯 CONCLUSION FOR TRADE 1162:")
    print("-" * 40)

    if not trade_data['moved_to_breakeven']:
        print("❌ This trade NEVER reached break-even trigger!")
        print("❌ The stop loss is BELOW entry price (initial protective stop)")
        print("❌ Status 'expired' means it never moved to breakeven")
        print("\n💡 This trade is NOT an example of breakeven behavior")
        print("💡 It's just showing the initial stop loss placement")
        print(f"💡 Initial stop was {actual_stop_distance:.1f} points below entry (standard risk management)")
    else:
        print("✅ This trade moved to breakeven")

    print("\n📈 TO SEE ACTUAL BREAKEVEN BEHAVIOR:")
    print("Look at these AUDUSD trades that actually moved to breakeven:")
    print("- Trade 820: moved 9 points to breakeven")
    print("- Trade 816: moved 1 point to breakeven")
    print("- Trade 815: moved 12 points to breakeven")
    print("- Trade 796: moved 1 point to breakeven")
    print("- Trade 783: moved 8 points to breakeven")


def show_comparison():
    """Show comparison between breakeven vs non-breakeven trades"""

    print("\n" + "=" * 60)
    print("🔄 BREAKEVEN vs NON-BREAKEVEN COMPARISON")
    print("=" * 60)

    print("\n✅ TRADE THAT MOVED TO BREAKEVEN (Trade 815):")
    print("Entry: 0.65111 (BUY)")
    print("Stop after breakeven: 0.65231 (ABOVE entry = +12 points)")
    print("moved_to_breakeven: true")
    print("Status: closed")
    print("➡️ This shows actual breakeven behavior")

    print("\n❌ TRADE THAT NEVER MOVED TO BREAKEVEN (Trade 1162):")
    print("Entry: 0.66064 (BUY)")
    print("Stop loss: 0.65814 (BELOW entry = -25 points)")
    print("moved_to_breakeven: false")
    print("Status: expired")
    print("➡️ This is just initial stop loss, not breakeven")

    print("\n🎯 KEY DIFFERENCE:")
    print("- Breakeven trades have stops ABOVE/AT entry (profit protection)")
    print("- Non-breakeven trades have stops BELOW entry (loss protection)")
    print("- Trade 1162 never triggered the breakeven logic")


if __name__ == "__main__":
    analyze_trade_1162()
    show_comparison()