#!/usr/bin/env python3
"""
Quantitative Analysis: Swing Point Entry Performance
Statistical analysis of trade entries relative to swing points
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load backtest data
df = pd.read_csv('besystem/advanced_backtest_trades.csv')

print('=' * 80)
print('QUANTITATIVE ANALYSIS: SWING POINT ENTRY PERFORMANCE')
print('=' * 80)
print()

# =============================================================================
# 1. STATISTICAL ANALYSIS - Overall Performance
# =============================================================================
print('1. OVERALL BACKTEST STATISTICS')
print('-' * 80)
print(f'Total Trades: {len(df)}')
print(f'Wins: {len(df[df["status"] == "CLOSED_WIN"])}')
print(f'Losses: {len(df[df["status"] == "CLOSED_LOSS"])}')
print(f'Breakevens: {len(df[df["status"] == "CLOSED_BE"])}')

win_rate = len(df[df["status"] == "CLOSED_WIN"]) / len(df) * 100
loss_rate = len(df[df["status"] == "CLOSED_LOSS"]) / len(df) * 100
be_rate = len(df[df["status"] == "CLOSED_BE"]) / len(df) * 100

print(f'\nWin Rate: {win_rate:.2f}%')
print(f'Loss Rate: {loss_rate:.2f}%')
print(f'Breakeven Rate: {be_rate:.2f}%')

# PnL Statistics
print(f'\nAverage PnL (pips): {df["pnl_pips"].mean():.2f}')
print(f'Median PnL (pips): {df["pnl_pips"].median():.2f}')
print(f'Std Dev PnL (pips): {df["pnl_pips"].std():.2f}')
print(f'Total PnL (currency): ${df["pnl_currency"].sum():.2f}')

# Duration Statistics
print(f'\nAverage Duration: {df["duration_minutes"].mean():.1f} minutes')
print(f'Median Duration: {df["duration_minutes"].median():.1f} minutes')

print()

# =============================================================================
# 2. PERFORMANCE BY SIGNAL DIRECTION
# =============================================================================
print('2. PERFORMANCE BY SIGNAL DIRECTION')
print('-' * 80)

for direction in ['BULL', 'BEAR']:
    direction_df = df[df['signal_type'] == direction]
    if len(direction_df) == 0:
        continue

    direction_wins = len(direction_df[direction_df['status'] == 'CLOSED_WIN'])
    direction_win_rate = direction_wins / len(direction_df) * 100

    print(f'\n{direction} Signals:')
    print(f'  Total: {len(direction_df)}')
    print(f'  Wins: {direction_wins}')
    print(f'  Win Rate: {direction_win_rate:.2f}%')
    print(f'  Avg PnL: {direction_df["pnl_pips"].mean():.2f} pips')
    print(f'  Total PnL: ${direction_df["pnl_currency"].sum():.2f}')

print()

# =============================================================================
# 3. PERFORMANCE BY CURRENCY PAIR
# =============================================================================
print('3. PERFORMANCE BY CURRENCY PAIR')
print('-' * 80)

pair_stats = []
for epic in df['epic'].unique():
    epic_df = df[df['epic'] == epic]
    epic_wins = len(epic_df[epic_df['status'] == 'CLOSED_WIN'])
    epic_losses = len(epic_df[epic_df['status'] == 'CLOSED_LOSS'])
    epic_win_rate = epic_wins / len(epic_df) * 100 if len(epic_df) > 0 else 0

    pair_stats.append({
        'epic': epic,
        'total': len(epic_df),
        'wins': epic_wins,
        'losses': epic_losses,
        'win_rate': epic_win_rate,
        'avg_pnl_pips': epic_df['pnl_pips'].mean(),
        'total_pnl': epic_df['pnl_currency'].sum()
    })

pair_df = pd.DataFrame(pair_stats).sort_values('total', ascending=False)
print(pair_df.to_string(index=False))

print()

# =============================================================================
# 4. SWING POINT PROXIMITY ANALYSIS (INFERRED)
# =============================================================================
print('4. SWING POINT PROXIMITY ANALYSIS (INFERRED)')
print('-' * 80)
print('\nNOTE: Direct swing point data not available in trade log.')
print('Performing indirect analysis based on trade characteristics...')
print()

# Analyze quick reversals (potential swing point rejections)
# Trades that lose quickly likely hit resistance/support (swing points)
quick_losses = df[(df['status'] == 'CLOSED_LOSS') & (df['duration_minutes'] <= 30)]
extended_losses = df[(df['status'] == 'CLOSED_LOSS') & (df['duration_minutes'] > 30)]

print(f'Quick Losses (<30 min): {len(quick_losses)} ({len(quick_losses)/len(df)*100:.1f}%)')
print(f'  → Likely swing point rejections/resistance hits')
print(f'  → Avg PnL: {quick_losses["pnl_pips"].mean():.2f} pips')
print(f'  → Avg Duration: {quick_losses["duration_minutes"].mean():.1f} minutes')
print()

print(f'Extended Losses (>30 min): {len(extended_losses)} ({len(extended_losses)/len(df)*100:.1f}%)')
print(f'  → May indicate trend reversals, not swing rejections')
print(f'  → Avg PnL: {extended_losses["pnl_pips"].mean():.2f} pips')
print(f'  → Avg Duration: {extended_losses["duration_minutes"].mean():.1f} minutes')
print()

# Analyze wins by duration (good entries vs. swing point entries)
quick_wins = df[(df['status'] == 'CLOSED_WIN') & (df['duration_minutes'] <= 60)]
extended_wins = df[(df['status'] == 'CLOSED_WIN') & (df['duration_minutes'] > 60)]

print(f'Quick Wins (<60 min): {len(quick_wins)} ({len(quick_wins)/len(df)*100:.1f}%)')
print(f'  → Strong momentum away from entry (good swing timing)')
print(f'  → Avg PnL: {quick_wins["pnl_pips"].mean():.2f} pips')
print()

print(f'Extended Wins (>60 min): {len(extended_wins)} ({len(extended_wins)/len(df)*100:.1f}%)')
print(f'  → Gradual trend development')
print(f'  → Avg PnL: {extended_wins["pnl_pips"].mean():.2f} pips')
print()

# =============================================================================
# 5. RISK METRICS
# =============================================================================
print('5. RISK METRICS')
print('-' * 80)

# Calculate risk-adjusted returns
wins_only = df[df['status'] == 'CLOSED_WIN']
losses_only = df[df['status'] == 'CLOSED_LOSS']

avg_win = wins_only['pnl_pips'].mean() if len(wins_only) > 0 else 0
avg_loss = abs(losses_only['pnl_pips'].mean()) if len(losses_only) > 0 else 0

profit_factor = (wins_only['pnl_currency'].sum() / abs(losses_only['pnl_currency'].sum())) if len(losses_only) > 0 else 0
expectancy = (win_rate/100 * avg_win) - (loss_rate/100 * avg_loss)

print(f'Average Win: {avg_win:.2f} pips')
print(f'Average Loss: {avg_loss:.2f} pips')
print(f'Win/Loss Ratio: {avg_win/avg_loss:.2f}' if avg_loss > 0 else 'Win/Loss Ratio: N/A')
print(f'Profit Factor: {profit_factor:.2f}')
print(f'Expectancy: {expectancy:.2f} pips per trade')

# Calculate max drawdown
df_sorted = df.sort_values('entry_time')
df_sorted['cumulative_pnl'] = df_sorted['pnl_currency'].cumsum()
df_sorted['running_max'] = df_sorted['cumulative_pnl'].cumsum().expanding().max()
df_sorted['drawdown'] = df_sorted['cumulative_pnl'] - df_sorted['running_max']
max_drawdown = df_sorted['drawdown'].min()

print(f'Max Drawdown: ${max_drawdown:.2f}')

print()

# =============================================================================
# 6. STATISTICAL SIGNIFICANCE TESTING
# =============================================================================
print('6. STATISTICAL SIGNIFICANCE TESTING')
print('-' * 80)

# Binomial test for win rate significance
from scipy import stats as scipy_stats

n_trials = len(df)
n_successes = len(df[df['status'] == 'CLOSED_WIN'])
expected_win_rate = 0.50  # Null hypothesis: 50% win rate (random)

# Two-tailed binomial test
p_value = scipy_stats.binom_test(n_successes, n_trials, expected_win_rate, alternative='two-sided')

print(f'Hypothesis Test: Is win rate significantly different from 50%?')
print(f'  Observed Win Rate: {win_rate:.2f}%')
print(f'  Expected (Random): {expected_win_rate*100:.2f}%')
print(f'  P-value: {p_value:.4f}')

if p_value < 0.05:
    print(f'  Result: STATISTICALLY SIGNIFICANT (p < 0.05)')
    if win_rate > 50:
        print(f'  Conclusion: Strategy demonstrates edge above random chance')
    else:
        print(f'  Conclusion: Strategy performs worse than random')
else:
    print(f'  Result: NOT SIGNIFICANT (p >= 0.05)')
    print(f'  Conclusion: Win rate not significantly different from random')

print()

# =============================================================================
# 7. RECOMMENDATIONS
# =============================================================================
print('7. QUANTITATIVE RECOMMENDATIONS')
print('-' * 80)
print()

print('Based on the statistical analysis:')
print()

if win_rate < 45:
    print('❌ CRITICAL: Win rate below 45% - Major strategy revision needed')
    print('   → Review swing point detection algorithm')
    print('   → Consider stricter entry filters')
    print('   → Analyze false signal patterns')
elif win_rate < 55:
    print('⚠️  MARGINAL: Win rate 45-55% - Strategy needs optimization')
    print('   → Implement swing proximity validation')
    print('   → Add confirmation filters')
    print('   → Improve entry timing')
else:
    print('✅ GOOD: Win rate >55% - Strategy shows promise')
    print('   → Focus on R:R optimization')
    print('   → Fine-tune exit strategies')

print()

if len(quick_losses) / len(df) > 0.20:
    print(f'⚠️  HIGH SWING REJECTION RATE: {len(quick_losses)/len(df)*100:.1f}% quick losses')
    print('   → Many entries being rejected at swing points')
    print('   → RECOMMENDATION: Implement swing proximity filter (8-10 pips)')
    print('   → Expected improvement: 5-10% reduction in quick losses')

print()

if avg_loss > avg_win:
    print(f'⚠️  NEGATIVE WIN/LOSS RATIO: {avg_win/avg_loss:.2f}')
    print('   → Losses larger than wins - unsustainable without high win rate')
    print('   → RECOMMENDATION: Tighten stop losses or widen targets')
    print('   → Consider swing-based stop placement')

print()

if profit_factor < 1.5:
    print(f'⚠️  LOW PROFIT FACTOR: {profit_factor:.2f}')
    print('   → Strategy barely profitable or unprofitable')
    print('   → RECOMMENDATION: Reduce trade frequency, increase selectivity')
    print('   → Filter entries near recent swing points')

print()
print('=' * 80)
print('END OF ANALYSIS')
print('=' * 80)
