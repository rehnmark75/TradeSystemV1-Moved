# 📘 Forex Scanner CLI - Command Reference

**`main.py`** is the unified entry point for all Forex Scanner operations, structured into modular command groups.

---

## 🔧 Global Options

These flags can be used with any command:

| Option             | Description                                      |
|--------------------|--------------------------------------------------|
| `--epic`           | Instrument epic to analyze                       |
| `--timestamp`      | Specific timestamp (format: `"YYYY-MM-DD HH:MM"`) |
| `--days`           | Number of days to backtest (default: from config) |
| `--timeframe`      | Timeframe: `5m`, `15m`, `1h`                      |
| `--ema-config`     | EMA config to use                                |
| `--ema-configs`    | List of EMA configs to compare                   |
| `--scalping-mode`  | `ultra_fast`, `aggressive`, `conservative`, `dual_ma` |
| `--show-signals`   | Show detailed signal list                        |
| `--bb-analysis`    | Include Bollinger Band analysis                  |
| `--no-future`      | Skip future data in analysis                     |
| `--max-analyses`   | Max batch size for Claude commands (default: 5)  |
| `--verbose` / `-v` | Verbose logging                                  |
| `--config-check`   | Show config and exit                             |

---

## 🟢 Scanner Commands

```bash
python main.py scan
python main.py live
```

---

## 📊 Backtest Commands

```bash
python main.py backtest --epic <EPIC> [--days <N>] [--show-signals] [--timeframe <5m|15m|1h>] [--bb-analysis] [--ema-config <NAME>]

python main.py compare-ema-configs --epic <EPIC> [--days <N>] [--timeframe <5m|15m|1h>] --ema-configs <CONFIG1> <CONFIG2> ...
```

---

## 🧪 Debug Commands

```bash
python main.py debug --epic <EPIC> [--timestamp "YYYY-MM-DD HH:MM"]
python main.py debug-macd --epic <EPIC>
python main.py debug-combined --epic <EPIC>
python main.py test-methods
```

---

## ⚡ Scalping Commands

```bash
python main.py scalp --epic <EPIC> [--scalping-mode ultra_fast|aggressive|conservative|dual_ma]
python main.py debug-scalping --epic <EPIC> [--scalping-mode ultra_fast|aggressive|conservative|dual_ma]
```

---

## 🤖 Claude Commands

```bash
python main.py test-claude

python main.py claude-timestamp --epic <EPIC> --timestamp "YYYY-MM-DD HH:MM" [--no-future]

python main.py claude-batch --epic <EPIC> [--days <N>] [--max-analyses <N>]
```

---

## 📈 Analysis Commands

```bash
python main.py test-bb [--epic <EPIC>]
python main.py compare-bb --epic <EPIC> [--days <N>]
python main.py list-ema-configs
```
