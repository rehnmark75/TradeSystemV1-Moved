import re
import csv
import sys

# Read all the backtest output (will be piped in)
log_content = sys.stdin.read()

# Extract the section with individual signals (🎯 BACKTEST SIGNAL #)
signal_pattern = r"(\d+:\d+:\d+) - INFO - 🎯 BACKTEST SIGNAL #(\d+): ([A-Z.]+) (BUY|SELL).*?Entry: ([\d.]+).*?SL: ([\d.]+|None).*?TP: ([\d.]+|None).*?Conf: ([\d.]+)%.*?(?:Result: (win|loss|breakeven), Pips: ([\d.-]+))?"

signals = []
for match in re.finditer(signal_pattern, log_content, re.DOTALL):
    time, signal_num, epic, signal_type, entry, sl, tp, conf, result, pips = match.groups()
    signals.append({
        'signal_num': signal_num,
        'epic': epic,
        'signal_type': signal_type,
        'entry_price': entry,
        'stop_loss': sl if sl \!= "None" else "",
        'take_profit': tp if tp \!= "None" else "",
        'confidence': conf,
        'result': result or "",
        'pips': pips or ""
    })

# Write to CSV
with open('/app/smc_signals_analysis.csv', 'w', newline='\) as f:
    if signals:
        writer = csv.DictWriter(f, fieldnames=signals[0].keys())
        writer.writeheader()
        writer.writerows(signals)
        print(f"✅ Extracted {len(signals)} signals to /app/smc_signals_analysis.csv")
    else:
        print("❌ No signals found in log output")
