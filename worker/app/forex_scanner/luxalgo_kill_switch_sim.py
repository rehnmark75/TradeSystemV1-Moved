#!/usr/bin/env python3
"""
EXPERIMENTAL (Jul 6 2026) — Test the user's proposal: don't demand 6y robustness;
instead trade a regime-conditional signal LIVE while a rolling monitor says it has
edge, and BLOCK it when it decays (shadow-track while blocked, re-enable on recovery).

Signal generator = the one salvageable piece of the two LuxAlgo showcases:
contrarian confirmation-fade + HyperWave filter (strategy #1), which was costed-
positive in the Mar-Jun 2026 regime (PF 1.27) but sign-flips across regimes.

The 6y history is used to validate the MONITORING POLICY, not the strategy:
  A) always-on (baseline — known ~PF 0.74-0.79 costed)
  B) rolling gate: trade only if net pips of last N closed trades > 0
     (shadow trades keep updating the window while blocked → auto re-enable)
  C) sign-adaptive: trade the rolling-favored direction (contrarian when window
     positive, inverted when negative beyond a dead-band, flat inside band)

No lookahead: gate decision for trade k uses only trades 0..k-1 (all closed before
k opens, since SAR trades are sequential). Costs charged only on trades taken live.

Run: docker exec task-worker python /app/forex_scanner/luxalgo_kill_switch_sim.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

from exit_redesign_sim import cost  # noqa: E402
from luxalgo_hyperwave_rep import load, entries, sim_sar, CONFS, HWS  # noqa: E402

EVAL_START = np.datetime64('2026-03-10')
Y2025 = np.datetime64('2025-01-01')


def gross_trades(epic, tf, cn, hn):
    df = load(epic, tf)
    if df is None:
        return None
    ents = entries(df, CONFS[cn], HWS[hn])
    return sim_sar(ents, df, epic, 0.0)   # gross pips (no cost)


def stats(rows):
    """rows = [(t, live_pips_or_None)] — only live trades counted."""
    x = np.array([p for _, p in rows if p is not None], float)
    if len(x) == 0:
        return "n=0"
    w = x[x > 0]; l = x[x <= 0]
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else float('inf')
    return (f"n={len(x):5d} WR={100*(x>0).mean():5.1f}% PF={pf:5.2f} "
            f"avg={x.mean():6.2f}p tot={x.sum():8.0f}p")


def run_policy(tr, cst, policy, N=30, band=0.0):
    """tr = [(t, gross_pips)]. Returns [(t, net_pips or None-if-blocked)].
    Rolling window tracks NET-if-contrarian-live pips of last N signals (shadow
    included, so a blocked strategy can recover)."""
    out = []
    roll = []   # net contrarian pips per signal (shadow view)
    for t, g in tr:
        net_c = g - cst
        s = sum(roll[-N:]) if len(roll) >= N else None
        if policy == 'A':
            out.append((t, net_c))
        elif policy == 'B':
            live = s is None or s > 0     # trade until proven bad; block while bad
            out.append((t, net_c if live else None))
        elif policy == 'C':
            if s is None or abs(s) <= band:
                out.append((t, None))
            elif s > 0:
                out.append((t, net_c))
            else:
                out.append((t, -g - cst))  # inverted trade
        roll.append(net_c)
    return out


def main():
    combos = [('CS.D.USDJPY.MINI.IP', 15, 'sHA', 'rsi'),
              ('CS.D.USDJPY.MINI.IP', 5, 'EMAX', 'rsi'),
              ('CS.D.EURJPY.MINI.IP', 15, 'sHA', 'rsi'),
              ('CS.D.EURUSD.CEEM.IP', 15, 'sHA', 'rsi')]
    for epic, tf, cn, hn in combos:
        tr = gross_trades(epic, tf, cn, hn)
        if not tr:
            continue
        cst = cost(epic)
        print(f"\n{'='*100}\n{epic} tf={tf} conf={cn} hw={hn}  (cost {cst}p RT, "
              f"{len(tr)} signals 2020-2026)\n{'='*100}")
        for label, policy, N, band in (
                ('A always-on          ', 'A', 0, 0),
                ('B block<0  N=20      ', 'B', 20, 0),
                ('B block<0  N=30      ', 'B', 30, 0),
                ('B block<0  N=50      ', 'B', 50, 0),
                ('C sign-adapt N=30 b0 ', 'C', 30, 0),
                ('C sign-adapt N=50 b25', 'C', 50, 25),
        ):
            rows = run_policy(tr, cst, policy, N, band)
            full = stats(rows)
            r25 = stats([r for r in rows if r[0] >= Y2025])
            rev = stats([r for r in rows if r[0] >= EVAL_START])
            n_live = sum(1 for _, p in rows if p is not None)
            expo = 100 * n_live / len(rows)
            print(f"  {label} expo={expo:5.1f}%  FULL[{full}]")
            print(f"  {'':21s}          2025+[{r25}]  eval[{rev}]")


if __name__ == '__main__':
    main()
