🚀 New Chat Quick Start Guide
📋 Context for New Assistant:
"We just completed a successful refactoring of a Forex Scanner project. The system was working with 125 signals found in a 7-day backtest (vs 0 before). The original 800+ line signal_detector.py was split into 8 modular files. All systems are functional - live scanning, backtesting, and debugging. The current issue I need help with is: [describe your specific problem]"
📁 Essential Files to Upload:
Priority 1 - Core Files:
core/signal_detector.py (lightweight coordinator)
core/strategies/combined_strategy.py (main strategy logic)
main.py (updated with debug commands)
config.py (current configuration)
Priority 2 - If Needed:
core/strategies/ema_strategy.py
core/strategies/macd_strategy.py
core/backtest/backtest_engine.py
Recent error logs or output
🔧 Current Working State:
bash
# This should work:
python main.py backtest --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m
# Expected: ~125 signals with 92% confidence

# This should work:
python main.py debug-combined --epic CS.D.EURUSD.MINI.IP
# Expected: Shows EMA + MACD + Combined strategy results
⚡ Common Issues & Solutions:
If Backtesting Returns 0 Signals:
Check if using combined strategy (should be)
Verify MACD_EMA_STRATEGY = True in config
Run: python main.py debug-backtest --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m
If Missing Module Errors:
Check if all new folders exist: strategies/, detection/, backtest/
Verify __init__.py files are present
Check import paths in updated files
If Strategy Errors:
Test individual strategies: python main.py debug --epic CS.D.EURUSD.MINI.IP
Check MACD indicators: python main.py debug-macd --epic CS.D.EURUSD.MINI.IP
Verify combined mode: python main.py debug-combined --epic CS.D.EURUSD.MINI.IP
📊 Expected Performance:
Backtest: 100+ signals per week
Confidence: 85-95% average
Strategies: Mix of EMA, MACD, Combined
Win Rate: ~60%
🆘 Emergency Fallback:
If combined strategies fail, set in config.py:
python
MACD_EMA_STRATEGY = False  # Forces individual EMA only
🎯 Key Success Indicators:
✅ python main.py scan works without errors
✅ Backtesting finds 50+ signals per week
✅ Debug commands show strategy results
✅ Performance analysis shows realistic win rates
