When starting a new chat, lead with:

"I have a working Forex Scanner that was recently refactored. It was finding 125 signals in 7-day backtests. The system uses modular strategies (EMA, MACD, Combined). My current issue is [specific problem]. Here are the key files..."

4. Quick Health Check:
Before starting a new chat, run:
bashpython main.py scan --config-check
python main.py backtest --epic CS.D.EURUSD.MINI.IP --days 3