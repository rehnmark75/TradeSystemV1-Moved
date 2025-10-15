#!/usr/bin/env python3
"""Store backtest system documentation in MCP memory"""

import sys
import os

# Add forex_scanner to path
sys.path.insert(0, '/app/forex_scanner')
sys.path.insert(0, '/app')

try:
    from core.mcp_server import mcpServer

    memory_content = """Enhanced Backtest System Documentation

The TradeSystemV1 backtest system has two distinct modes:

1. RAW SIGNAL MODE (Default)
- Purpose: Tests strategy signal generation logic in isolation
- Command: python bt.py [PAIR] [DAYS] [STRATEGY] --show-signals
- Validation: Only basic strategy-level filters applied
- Use case: Strategy development, parameter tuning, signal quality assessment

2. PIPELINE MODE (--pipeline flag)
- Purpose: Tests complete signal processing pipeline matching live trading
- Command: python bt.py [PAIR] [DAYS] [STRATEGY] --pipeline
- Validation: Full trade_validator with S/R levels, position mgmt, risk mgmt
- Use case: Pre-production testing, realistic performance estimation

KEY DIFFERENCES:
| Aspect | Raw Signal Mode | Pipeline Mode |
|--------|-----------------|---------------|
| Purpose | Strategy development | Live trading simulation |
| Validation | Strategy filters only | Full trade_validator |
| S/R Checks | No | Yes (major filter) |
| Signal Count | Higher (permissive) | Lower (restrictive) |
| Accuracy | Shows signal potential | Shows realistic performance |

COMMAND STRUCTURE:

Basic Usage:
- python bt.py                              # 7 days, all pairs, EMA (raw)
- python bt.py EURUSD                       # 7 days, EUR/USD only (raw)
- python bt.py EURUSD 14                    # 14 days, EUR/USD (raw)
- python bt.py EURUSD 7 --show-signals      # With detailed signals (raw)
- python bt.py EURUSD 7 --pipeline          # Full pipeline validation (realistic)

Strategy Shortcuts:
- EMA → EMA
- MACD → MACD
- BB → BOLLINGER_SUPERTREND
- SMC → SMC_FAST
- MOMENTUM → MOMENTUM
- ICHIMOKU → ICHIMOKU
- KAMA → KAMA
- ZEROLAG → ZERO_LAG
- MEANREV → MEAN_REVERSION
- RANGING → RANGING_MARKET
- SCALPING → SCALPING
- VP/VOLUME_PROFILE → VOLUME_PROFILE

Pair Shortcuts:
- EURUSD → CS.D.EURUSD.CEEM.IP (special CEEM suffix)
- GBPUSD/USDJPY/etc → CS.D.{PAIR}.MINI.IP

RESULT SUMMARY FORMAT:

When a backtest completes:

📊 Total signals processed: X
✔️ Signals validated: Y
❌ Signals rejected: Z

✅ VALIDATED SIGNALS TABLE
[ID] [Timestamp] [Pair] [Direction] [Strategy] [Entry] [Confidence] [SL] [TP1-3] [R/R]

❌ REJECTED SIGNALS TABLE (with reasons)
[ID] [Timestamp] [Pair] [Direction] [Strategy] [Entry] [Confidence] [Rejection Reason]

📈 PERFORMANCE METRICS:
   📊 Total Signals: X
   🎯 Average Confidence: XX.X%
   📈 Bull Signals: X
   📉 Bear Signals: X
   💰 Average Profit: X.X pips
   📉 Average Loss: X.X pips
   🏆 Validation Rate: XX.X%

RECENT EXAMPLE (EMA Strategy Test):

Command:
docker exec -w /app task-worker python forex_scanner/bt.py --all 7 EMA --pipeline --timeframe 15m --show-signals

Results:
- 10 signals detected
- 3 validated (30% validation rate)
- 7 rejected by S/R level validator (support_cluster cluster risk)
- All signals: USDJPY BEAR at 95% confidence
- Avg profit: 3.7 pips, Avg loss: 4.1 pips

Key Insight: Without --pipeline, all 10 signals would be considered valid. Pipeline mode revealed 70% rejected due to S/R proximity.

COMMON REJECTION REASONS (--pipeline):
- S/R Level: support_cluster cluster risk - Near support cluster
- S/R Level: resistance_cluster cluster risk - Near resistance cluster
- Position Management: max positions reached - Too many open trades
- Risk Management: insufficient margin - Not enough margin
- Market Condition: high volatility - Volatility filter triggered

BEST PRACTICES:

1. Strategy Development Phase:
   - Use raw mode (no --pipeline) to tune parameters
   - Focus on signal quality and confidence scores
   - Iterate quickly with --show-signals

2. Pre-Production Testing:
   - Use --pipeline mode to simulate live trading
   - Evaluate realistic validation rates
   - Assess rejection reasons

3. Parameter Optimization:
   - Start with raw mode to find signal-generating parameters
   - Validate with pipeline mode to ensure signals survive validation
   - Balance signal frequency with validation rate

4. Performance Evaluation:
   - Raw mode validation rate = strategy signal quality
   - Pipeline mode validation rate = expected live performance
   - Gap between modes = impact of trade_validator filters

FILE LOCATIONS:
- bt.py wrapper: /app/forex_scanner/bt.py
- Main backtest CLI: /app/forex_scanner/backtest_cli.py
- Trade validator: /app/forex_scanner/core/trade_validator.py
- Strategy implementations: /app/forex_scanner/core/strategies/
"""

    result = mcpServer.write_memory('backtest_system_documentation', memory_content)
    print('✅ Backtest system documentation stored in MCP memory')
    print(result)

except Exception as e:
    print(f'❌ Error storing memory: {e}')
    sys.exit(1)
