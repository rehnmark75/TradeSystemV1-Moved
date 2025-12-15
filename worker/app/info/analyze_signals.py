#!/usr/bin/env python3
"""Analyze SMC backtest signals from database"""

import sys
sys.path.insert(0, "/app/forex_scanner")

import config
from core.database import DatabaseManager

# Connect to database
db = DatabaseManager(config.DATABASE_URL)

# Query all signals from the most recent backtest
query = """
SELECT
    signal_timestamp,
    epic,
    signal_type,
    entry_price,
    stop_loss_price,
    take_profit_price,
    exit_price,
    pips_gained,
    trade_result,
    confidence_score,
    holding_time_minutes,
    exit_reason,
    validation_passed
FROM backtest_signals
WHERE execution_id = (SELECT MAX(execution_id) FROM backtest_executions WHERE strategy_name = 'SMC_STRUCTURE')
ORDER BY signal_timestamp ASC
"""

signals_df = db.execute_query(query, {})

print("\n" + "="*80)
print("📊 SMC STRATEGY BACKTEST ANALYSIS")
print("="*80)

if len(signals_df) == 0:
    print("\n❌ No signals found in database")
    print("\nChecking recent executions:")
    exec_query = "SELECT execution_id, created_at, strategy_name FROM backtest_executions ORDER BY created_at DESC LIMIT 5"
    execs = db.execute_query(exec_query, {})
    print(execs)
    sys.exit(1)

print(f"\n✅ Found {len(signals_df)} signals\n")

# Overall statistics
total = len(signals_df)
completed = signals_df[signals_df['trade_result'].notna()]
wins = len(completed[completed['trade_result'] == 'win'])
losses = len(completed[completed['trade_result'] == 'loss'])
breakeven = len(completed[completed['trade_result'] == 'breakeven'])

win_rate = (wins / max(wins + losses, 1)) * 100 if (wins + losses) > 0 else 0

print("📈 OVERALL PERFORMANCE:")
print(f"   Total Signals: {total}")
print(f"   ✅ Wins: {wins} ({wins/total*100:.1f}%)")
print(f"   ❌ Losses: {losses} ({losses/total*100:.1f}%)")
print(f"   ⚖️  Breakeven: {breakeven} ({breakeven/total*100:.1f}%)")
print(f"   🎯 Win Rate: {win_rate:.1f}%")

# Analyze LOSERS
print("\n\n" + "="*80)
print("❌ ANALYZING LOSING TRADES")
print("="*80)

losers = completed[completed['trade_result'] == 'loss']
if len(losers) > 0:
    avg_pips_lost = losers['pips_gained'].mean()
    total_pips_lost = losers['pips_gained'].sum()
    avg_conf_losers = losers['confidence_score'].mean() * 100
    avg_time_losers = losers['holding_time_minutes'].mean()

    print(f"\nTotal losing trades: {len(losers)}")
    print(f"Average pips lost: {avg_pips_lost:.1f} pips")
    print(f"Total pips lost: {total_pips_lost:.1f} pips")
    print(f"Average confidence: {avg_conf_losers:.1f}%")
    print(f"Average holding time: {avg_time_losers:.0f} minutes")

    print("\nExit reasons for losses:")
    print(losers['exit_reason'].value_counts())

    print("\n📋 SAMPLE LOSING TRADES:")
    print("-" * 80)
    for idx, row in losers.head(10).iterrows():
        ts = row['signal_timestamp']
        entry = row['entry_price']
        exit_p = row['exit_price']
        pips = row['pips_gained']
        conf = row['confidence_score'] * 100
        reason = row['exit_reason']
        print(f"  {ts} | Entry: {entry:.5f} | Exit: {exit_p:.5f} | Pips: {pips:.1f} | Conf: {conf:.0f}% | Exit: {reason}")
else:
    print("\n✅ No losing trades!")

# Analyze WINNERS
print("\n\n" + "="*80)
print("✅ ANALYZING WINNING TRADES")
print("="*80)

winners = completed[completed['trade_result'] == 'win']
if len(winners) > 0:
    avg_pips_won = winners['pips_gained'].mean()
    total_pips_won = winners['pips_gained'].sum()
    avg_conf_winners = winners['confidence_score'].mean() * 100
    avg_time_winners = winners['holding_time_minutes'].mean()

    print(f"\nTotal winning trades: {len(winners)}")
    print(f"Average pips won: {avg_pips_won:.1f} pips")
    print(f"Total pips won: {total_pips_won:.1f} pips")
    print(f"Average confidence: {avg_conf_winners:.1f}%")
    print(f"Average holding time: {avg_time_winners:.0f} minutes")

    print("\nExit reasons for wins:")
    print(winners['exit_reason'].value_counts())

    print("\n📋 SAMPLE WINNING TRADES:")
    print("-" * 80)
    for idx, row in winners.head(10).iterrows():
        ts = row['signal_timestamp']
        entry = row['entry_price']
        exit_p = row['exit_price']
        pips = row['pips_gained']
        conf = row['confidence_score'] * 100
        reason = row['exit_reason']
        print(f"  {ts} | Entry: {entry:.5f} | Exit: {exit_p:.5f} | Pips: {pips:.1f} | Conf: {conf:.0f}% | Exit: {reason}")
else:
    print("\n❌ No winning trades!")

# Key insights
print("\n\n" + "="*80)
print("🔍 KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

if len(winners) > 0 and len(losers) > 0:
    conf_diff = (avg_conf_winners - avg_conf_losers)

    print(f"\n1. CONFIDENCE SCORE ANALYSIS:")
    print(f"   Winners avg confidence: {avg_conf_winners:.1f}%")
    print(f"   Losers avg confidence: {avg_conf_losers:.1f}%")
    print(f"   Difference: {conf_diff:.1f}%")
    if abs(conf_diff) < 5:
        print(f"   ⚠️  WARNING: Confidence barely differs! Strategy may not be filtering well.")

    print(f"\n2. HOLDING TIME ANALYSIS:")
    print(f"   Winners avg time: {avg_time_winners:.0f} minutes")
    print(f"   Losers avg time: {avg_time_losers:.0f} minutes")
    time_diff = avg_time_winners - avg_time_losers
    if time_diff > 0:
        print(f"   ✅ Winners hold {time_diff:.0f} min longer - good sign")
    else:
        print(f"   ⚠️  Losers hold {abs(time_diff):.0f} min longer - may need tighter stops")

    print(f"\n3. RISK/REWARD RATIO:")
    print(f"   Average win: +{avg_pips_won:.1f} pips")
    print(f"   Average loss: {avg_pips_lost:.1f} pips")
    rr_ratio = abs(avg_pips_won / avg_pips_lost)
    print(f"   Win/Loss ratio: {rr_ratio:.2f}:1")
    if rr_ratio < 1.5:
        print(f"   ⚠️  WARNING: Risk/Reward is poor! Need {rr_ratio:.2f}:1, target 2:1+")

    print(f"\n4. OVERALL ASSESSMENT:")
    if win_rate < 40:
        print(f"   ❌ Win rate {win_rate:.1f}% is TOO LOW")
        print(f"   📌 RECOMMENDATION: Strategy needs major improvements")
    elif win_rate < 50:
        print(f"   ⚠️  Win rate {win_rate:.1f}% is marginal")
        print(f"   📌 RECOMMENDATION: Improve entry filters or adjust R:R")
    else:
        print(f"   ✅ Win rate {win_rate:.1f}% is acceptable")

print("\n" + "="*80)
print("="*80 + "\n")
