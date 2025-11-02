#!/usr/bin/env python3
"""Parse all SMC signals from backtest output and analyze winners vs losers"""

import re
from collections import defaultdict

# Read the file
with open('/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/all_signals.txt', 'r') as f:
    content = f.read()

# Pattern to match signal blocks - each signal has Entry/Exit/Result info
signal_pattern = r'🎯 BACKTEST SIGNAL #(\d+): (CS\.D\.[A-Z]+\.[A-Z.]+) (BUY|SELL)\n.*?Entry: ([\d.]+).*?(?:Exit: ([\d.]+|None), Result: (win|loss|breakeven), Pips: ([-\d.]+))?'

signals = []
for match in re.finditer(signal_pattern, content, re.DOTALL):
    signal_num = match.group(1)
    epic = match.group(2)
    signal_type = match.group(3)
    entry = float(match.group(4))
    exit_price = match.group(5)
    result = match.group(6)
    pips = match.group(7)

    signal = {
        'num': int(signal_num),
        'epic': epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', ''),
        'type': signal_type,
        'entry': entry,
        'exit': float(exit_price) if exit_price and exit_price != 'None' else None,
        'result': result,
        'pips': float(pips) if pips else 0.0
    }
    signals.append(signal)

print("\n" + "="*80)
print(f"📊 PARSED {len(signals)} SIGNALS FROM LOG FILE")
print("="*80)

# Separate by result
wins = [s for s in signals if s['result'] == 'win']
losses = [s for s in signals if s['result'] == 'loss']
breakevens = [s for s in signals if s['result'] == 'breakeven']
pending = [s for s in signals if s['result'] is None]

print(f"\nTotal signals: {len(signals)}")
print(f"Winners: {len(wins)} ({len(wins)/len(signals)*100:.1f}%)")
print(f"Losers: {len(losses)} ({len(losses)/len(signals)*100:.1f}%)")
print(f"Breakeven: {len(breakevens)} ({len(breakevens)/len(signals)*100:.1f}%)")
print(f"Pending: {len(pending)}")

if len(wins) + len(losses) > 0:
    win_rate = len(wins) / (len(wins) + len(losses)) * 100
    print(f"\nWin Rate (excluding breakeven): {win_rate:.1f}%")

# Analyze LOSERS
print("\n" + "="*80)
print("❌ LOSING TRADES ANALYSIS")
print("="*80)

if len(losses) > 0:
    avg_loss = sum(s['pips'] for s in losses) / len(losses)
    print(f"\nTotal losers: {len(losses)}")
    print(f"Average loss: {avg_loss:.2f} pips")
    print(f"Total pips lost: {sum(s['pips'] for s in losses):.2f}")

    # Group by pair
    losses_by_pair = defaultdict(list)
    for s in losses:
        losses_by_pair[s['epic']].append(s)

    print("\nLosses by pair:")
    for pair in sorted(losses_by_pair.keys(), key=lambda x: len(losses_by_pair[x]), reverse=True):
        count = len(losses_by_pair[pair])
        avg_pips = sum(s['pips'] for s in losses_by_pair[pair]) / count
        print(f"  {pair}: {count} losses, avg {avg_pips:.1f} pips")

    # Group by signal type
    bull_losses = [s for s in losses if s['type'] == 'BUY']
    bear_losses = [s for s in losses if s['type'] == 'SELL']
    print(f"\nBUY losses: {len(bull_losses)}")
    print(f"SELL losses: {len(bear_losses)}")

    # Show sample of worst losses
    worst_losses = sorted(losses, key=lambda x: x['pips'])[:10]
    print("\nWorst 10 losing trades:")
    for s in worst_losses:
        print(f"  #{s['num']}: {s['epic']} {s['type']} | Entry: {s['entry']:.5f} Exit: {s['exit']:.5f} | {s['pips']:.1f} pips")

# Analyze WINNERS
print("\n" + "="*80)
print("✅ WINNING TRADES ANALYSIS")
print("="*80)

if len(wins) > 0:
    avg_win = sum(s['pips'] for s in wins) / len(wins)
    print(f"\nTotal winners: {len(wins)}")
    print(f"Average win: {avg_win:.2f} pips")
    print(f"Total pips won: {sum(s['pips'] for s in wins):.2f}")

    # Group by pair
    wins_by_pair = defaultdict(list)
    for s in wins:
        wins_by_pair[s['epic']].append(s)

    print("\nWins by pair:")
    for pair in sorted(wins_by_pair.keys(), key=lambda x: len(wins_by_pair[x]), reverse=True):
        count = len(wins_by_pair[pair])
        avg_pips = sum(s['pips'] for s in wins_by_pair[pair]) / count
        print(f"  {pair}: {count} wins, avg {avg_pips:.1f} pips")

    # Group by signal type
    bull_wins = [s for s in wins if s['type'] == 'BUY']
    bear_wins = [s for s in wins if s['type'] == 'SELL']
    print(f"\nBUY wins: {len(bull_wins)}")
    print(f"SELL wins: {len(bear_wins)}")

    # Show sample of best wins
    best_wins = sorted(wins, key=lambda x: x['pips'], reverse=True)[:10]
    print("\nBest 10 winning trades:")
    for s in best_wins:
        print(f"  #{s['num']}: {s['epic']} {s['type']} | Entry: {s['entry']:.5f} Exit: {s['exit']:.5f} | +{s['pips']:.1f} pips")

# KEY COMPARISONS
print("\n" + "="*80)
print("🔍 KEY PATTERNS - WINNERS VS LOSERS")
print("="*80)

if len(wins) > 0 and len(losses) > 0:
    print("\nR:R Ratio:")
    rr_ratio = abs(avg_win / avg_loss)
    print(f"  Average win: +{avg_win:.1f} pips")
    print(f"  Average loss: {avg_loss:.1f} pips")
    print(f"  Win/Loss ratio: {rr_ratio:.2f}:1")

    print("\nPair Performance:")
    all_pairs = set(list(wins_by_pair.keys()) + list(losses_by_pair.keys()))
    for pair in sorted(all_pairs):
        w = len(wins_by_pair.get(pair, []))
        l = len(losses_by_pair.get(pair, []))
        total = w + l
        if total > 0:
            pair_win_rate = w / total * 100
            print(f"  {pair}: {w}W / {l}L ({pair_win_rate:.0f}% win rate)")

    print("\nDirection Performance:")
    total_bull = len([s for s in signals if s['type'] == 'BUY'])
    total_bear = len([s for s in signals if s['type'] == 'SELL'])
    print(f"  BUY signals: {total_bull} total, {len(bull_wins)}W / {len(bull_losses)}L")
    print(f"  SELL signals: {total_bear} total, {len(bear_wins)}W / {len(bear_losses)}L")
    if total_bull > 0 and total_bear > 0:
        buy_wr = len(bull_wins) / (len(bull_wins) + len(bull_losses)) * 100 if (len(bull_wins) + len(bull_losses)) > 0 else 0
        sell_wr = len(bear_wins) / (len(bear_wins) + len(bear_losses)) * 100 if (len(bear_wins) + len(bear_losses)) > 0 else 0
        print(f"  BUY win rate: {buy_wr:.1f}%")
        print(f"  SELL win rate: {sell_wr:.1f}%")

print("\n" + "="*80)

# Save to CSV
import csv
with open('/home/hr/Projects/TradeSystemV1/signals_analysis.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['num', 'epic', 'type', 'entry', 'exit', 'result', 'pips'])
    writer.writeheader()
    writer.writerows(signals)

print(f"\n✅ Saved {len(signals)} signals to signals_analysis.csv")
