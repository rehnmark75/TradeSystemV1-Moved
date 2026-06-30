#!/usr/bin/env python3
"""
EXPERIMENTAL (Jun 30 2026) — generate & cache a strategy's raw ENTRY-SIGNAL
stream over long history, decoupled from exit simulation. Output feeds
exit_redesign_sim.py Phase 1b.

Drives BacktestScanner._scan_historical_timepoint() (raw detection, no exit sim,
no DB logging) and writes (timestamp, direction, price) rows to CSV.

Run inside task-worker:
  python /app/forex_scanner/extract_signals.py --epic CS.D.EURUSD.CEEM.IP \
      --strategy SMC_SIMPLE --start 2020-01-01 --end 2025-09-01 --timeframe 5m
"""
from __future__ import annotations
import sys, os, argparse, csv
from datetime import datetime, timezone

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

import logging
logging.basicConfig(level=logging.WARNING)

CACHE_DIR = '/app/forex_scanner/exit_sim_signals'


def _parse(dt: str) -> datetime:
    return datetime.fromisoformat(dt).replace(tzinfo=timezone.utc)


def _norm_dir(raw) -> str | None:
    s = str(raw).upper()
    if s in ('BULL', 'BUY', 'LONG'):
        return 'BUY'
    if s in ('BEAR', 'SELL', 'SHORT'):
        return 'SELL'
    return None


def extract(epic: str, strategy: str, start: str, end: str, timeframe: str) -> str:
    import config
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.backtest_scanner import BacktestScanner

    dbm = DatabaseManager(config.DATABASE_URL)
    cfg = {
        'start_date': _parse(start),
        'end_date': _parse(end),
        'execution_id': 999999,
        'strategy_name': strategy,
        'timeframe': timeframe,
        'epics': [epic],
        'skip_signal_logging': True,
        'pipeline_mode': False,   # raw strategy detection = native emitted stream
    }
    scanner = BacktestScanner(cfg, db_manager=dbm, use_historical_intelligence=False)
    scanner._ensure_backtest_candles_current()

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"{strategy}__{epic.replace('.', '_')}.csv")

    rows, n_steps, n_raw = [], 0, 0
    for t in scanner._create_time_iterator():
        n_steps += 1
        try:
            sigs = scanner._scan_historical_timepoint(t)
        except Exception as e:
            continue
        for sig in (sigs or []):
            if not sig:
                continue
            n_raw += 1
            ts = sig.get('timestamp') or sig.get('signal_timestamp') or t
            d = _norm_dir(sig.get('signal_type', ''))
            if not d:
                continue
            px = sig.get('current_price') or sig.get('price') or ''
            rows.append((ts.isoformat() if hasattr(ts, 'isoformat') else str(ts), d, px))

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'direction', 'price'])
        w.writerows(rows)
    print(f"✅ {epic} {strategy}: steps={n_steps} raw={n_raw} signals_written={len(rows)} -> {out_path}")
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epic', required=True)
    ap.add_argument('--strategy', default='SMC_SIMPLE')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--timeframe', default='5m')
    a = ap.parse_args()
    extract(a.epic, a.strategy, a.start, a.end, a.timeframe)
