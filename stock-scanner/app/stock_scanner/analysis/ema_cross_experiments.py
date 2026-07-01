"""
EMA 9/21/50 Cross — experiment definitions
==========================================
Encodes the entry-funnel + exit-sweep plan from the strategy-analyst and the
online research (regime/breadth gate, RS-leadership, ATR/trailing exit; ADX as
inverse-ablation only; partials expected to hurt a fat-tail system).

Imported and driven by ema_cross_lab.py (`--experiments`). Every row prints
ALL / TRAIN / OOS so a config that only works in-sample is visibly a FAIL.
"""

import numpy as np
import pandas as pd

LIVE = {"sl": 2.5, "tp": 7.0}  # live auto-trader bracket (baseline exit)


def run2(lab, show, metrics, line):
    """Round 2: layer tighter entries on the ATR-trail exit; sweep arm; ADX test."""
    s = lab.signals
    c = pd.to_numeric(s["close"], errors="coerce")
    m_stack = s["stack_aligned"].fillna(False)
    m_breadth = s["breadth"].fillna(0) >= 0.50
    m_rs = s["rs_20d"].fillna(-999) > 0
    m_conf = (s["range_pos"].fillna(0) >= 0.5) & s["green"].fillna(False)
    m_pull = s["pullback_atr"].fillna(0) >= 0.1
    m_fresh = s["age50"].fillna(999) <= 10
    m_adx = (s["adx"].fillna(0) >= 20) & (s["adx"].fillna(0) <= 40)
    m_rsi = (s["rsi"].fillna(0) >= 45) & (s["rsi"].fillna(0) <= 70)

    base = m_stack & m_breadth & m_rs           # the round-1 winner entry
    TRAIL = {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": 3.0}  # robust exit

    print("\n" + "#" * 78)
    print("# ROUND 2 — tighter entries on ATR-trail(3.0x, arm2) exit")
    print("#" * 78)
    funnels = {
        "E3 base(stack+breadth+RS)": base,
        "E4 +confirm": base & m_conf,
        "E5 +confirm+pullback": base & m_conf & m_pull,
        "E6 +fresh-cross": base & m_fresh,
        "E7 +confirm+fresh": base & m_conf & m_fresh,
        "E8 +RSI[45,70]": base & m_rsi,
        "E9 +confirm+RSI": base & m_conf & m_rsi,
        "E10 +ADX[20,40]": base & m_adx,
        "E11 +ADX+confirm": base & m_adx & m_conf,
        "E12 all-in": base & m_conf & m_pull & m_fresh & m_rsi,
    }
    for name, m in funnels.items():
        show(lab, name, m, TRAIL)

    print("\n--- ADX inverse-ablation (does removing ADX hurt?) ---")
    show(lab, "kept ADX[20,40] on base", base & m_adx, TRAIL)
    show(lab, "rej  ADX (outside band)", base & ~m_adx, TRAIL)

    print("\n--- trail-arm sweep on E4 (confirm) ---")
    for arm in [1.5, 2.0, 2.5, 3.0]:
        cfg = {"sl_atr": 1.5, "trail_arm": arm, "trail_atr": 3.0}
        show(lab, f"E4 arm{arm}", base & m_conf, cfg)

    print("\n--- trail-mult sweep on E7 (confirm+fresh) plateau check ---")
    for mult in [2.0, 2.5, 3.0, 3.5, 4.0]:
        cfg = {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": mult}
        show(lab, f"E7 mult{mult}", base & m_conf & m_fresh, cfg)


def _z(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / (x.std(ddof=0) + 1e-9)


def _topk_mask(s, cand_mask, score, k):
    """Keep the top-k candidates per calendar day, ranked by score desc."""
    idx = s.index[cand_mask.fillna(False)]
    day = s.loc[idx, "timestamp"].dt.normalize()
    sc = pd.to_numeric(score.loc[idx], errors="coerce").fillna(-1e9)
    rank = sc.groupby(day).rank(method="first", ascending=False)
    keep = idx[rank.values <= k]
    m = pd.Series(False, index=s.index)
    m.loc[keep] = True
    return m


def run3(lab, show, metrics, line):
    """Round 3: per-day top-K conviction ranking (mirrors live top-pool execution)."""
    s = lab.signals
    m_stack = s["stack_aligned"].fillna(False)
    m_breadth = s["breadth"].fillna(0) >= 0.50
    m_conf = (s["range_pos"].fillna(0) >= 0.5) & s["green"].fillna(False)
    cand = m_stack & m_breadth                       # regime-gated quality pool
    cand_c = cand & m_conf                           # + cross conviction
    TRAIL = {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": 3.0}

    # pre-registered conviction scores (per-stock differentiators; breadth is
    # constant within a day so it is omitted from the ranker)
    scores = {
        "RS20": s["rs_20d"],
        "ret60": s["ret_60d"],
        "ema50slope": s["ema50_slope"],
        "composite": _z(s["rs_20d"]) + _z(s["ret_60d"]) + _z(s["ema50_slope"]) + _z(s["range_pos"]),
        "mom+conf": _z(s["rs_20d"]) + _z(s["ret_60d"]) + _z(s["range_pos"]),
    }

    print("\n" + "#" * 78)
    print("# ROUND 3 — per-day top-K ranking on stack+breadth pool, ATR-trail(3x) exit")
    print("#   (top-K mirrors live: rank the day's candidates, trade only the best)")
    print("#" * 78)
    for sname, score in scores.items():
        print(f"\n=== rank by {sname} ===")
        for k in [1, 2, 3, 5, 10]:
            show(lab, f"{sname} top{k}", _topk_mask(s, cand, score, k), TRAIL)

    print("\n" + "#" * 78)
    print("# ROUND 3b — same, on the +confirm pool (cand & confirm-candle)")
    print("#" * 78)
    for sname in ["composite", "mom+conf"]:
        score = scores[sname]
        print(f"\n=== rank by {sname} (confirm pool) ===")
        for k in [1, 2, 3, 5]:
            show(lab, f"{sname}+conf top{k}", _topk_mask(s, cand_c, score, k), TRAIL)


def _monthly(lab, mask, cfg, metrics, line):
    df = lab.evaluate(mask, cfg)
    if df is None or df.empty:
        print("  (no trades)"); return
    df = df.copy()
    df["month"] = pd.to_datetime(df["ts"]).dt.to_period("M").astype(str)
    for mth, g in df.groupby("month"):
        # trades/day within month for context
        print(line(mth, metrics(g, max(1, pd.to_datetime(g["ts"]).dt.normalize().nunique()))))


def run4(lab, show, metrics, line):
    """Round 4: finalize the per-day ranker + monthly + robustness."""
    s = lab.signals
    m_stack = s["stack_aligned"].fillna(False)
    m_breadth = s["breadth"].fillna(0) >= 0.50
    m_conf = (s["range_pos"].fillna(0) >= 0.5) & s["green"].fillna(False)
    pool = m_stack & m_breadth
    pool_c = pool & m_conf
    TRAIL = {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": 3.0}

    trend_comp = _z(s["ema50_slope"]) + _z(s["ret_60d"]) + _z(s["rs_20d"])
    rankers = {
        "ema50slope": s["ema50_slope"],
        "trend_comp": trend_comp,
        "composite": _z(s["rs_20d"]) + _z(s["ret_60d"]) + _z(s["ema50_slope"]) + _z(s["range_pos"]),
    }

    print("\n" + "#" * 78)
    print("# ROUND 4 — finalize ranker (pool=stack+breadth, ATR-trail 3x)")
    print("#" * 78)
    for rn, score in rankers.items():
        print(f"\n=== {rn} ===")
        for k in [3, 5, 7]:
            show(lab, f"{rn} top{k}", _topk_mask(s, pool, score, k), TRAIL)
        for k in [5, 7]:
            show(lab, f"{rn}+conf top{k}", _topk_mask(s, pool_c, score, k), TRAIL)

    print("\n" + "#" * 78)
    print("# FINALIST monthly breakdown: trend_comp top5 (pool=stack+breadth)")
    print("#" * 78)
    _monthly(lab, _topk_mask(s, pool, trend_comp, 5), TRAIL, metrics, line)


def run5(lab, show, metrics, line):
    """Round 5: least-invasive live exit — keep fixed 2.5% initial stop, add ATR trail."""
    s = lab.signals
    m_stack = s["stack_aligned"].fillna(False)
    m_breadth = s["breadth"].fillna(0) >= 0.50
    pool = m_stack & m_breadth
    ema50 = s["ema50_slope"]
    comp = _z(s["rs_20d"]) + _z(s["ret_60d"]) + _z(s["ema50_slope"]) + _z(s["range_pos"])

    variants = {
        "ATR-init 1.5x + trail3x (validated)": {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": 3.0},
        "FIXED 2.5% + ATR-trail3x": {"sl": 2.5, "trail_arm": 2.0, "trail_atr": 3.0},
        "FIXED 2.5% + ATR-trail2.5x": {"sl": 2.5, "trail_arm": 2.0, "trail_atr": 2.5},
        "FIXED 2.5% + ATR-trail3.5x": {"sl": 2.5, "trail_arm": 2.0, "trail_atr": 3.5},
        "FIXED 2.5% + keep TP7 (live now)": {"sl": 2.5, "tp": 7.0},
        "FIXED 2.5% + TP10": {"sl": 2.5, "tp": 10.0},
    }
    print("\n" + "#" * 78)
    print("# ROUND 5 — live-exit variants on composite top5 & ema50slope top5")
    print("#   (does keeping the FIXED 2.5% initial stop + only adding ATR trail work?)")
    print("#" * 78)
    for rname, score in [("composite", comp), ("ema50slope", ema50)]:
        m = _topk_mask(s, pool, score, 5)
        print(f"\n=== {rname} top5 ===")
        for vn, cfg in variants.items():
            show(lab, vn, m, cfg)


def run(lab, show, metrics, line):
    s = lab.signals
    T = pd.Series(True, index=s.index)

    # ---- reusable masks -------------------------------------------------
    m_stack = s["stack_aligned"].fillna(False)
    m_fresh = s["age50"].fillna(999) <= 10
    m_breadth = s["breadth"].fillna(0) >= 0.50
    m_breadth55 = s["breadth"].fillna(0) >= 0.55
    m_mkt = s["mkt_above_ema50"].fillna(False)
    m_rs = s["rs_20d"].fillna(-999) > 0
    m_conf = (s["range_pos"].fillna(0) >= 0.5) & s["green"].fillna(False)
    m_pull = s["pullback_atr"].fillna(0) >= 0.1
    m_adx = (s["adx"].fillna(0) >= 18) & (s["adx"].fillna(0) <= 35)

    print("\n" + "#" * 78)
    print("# PART A — ENTRY FUNNEL (exit = live bracket SL2.5/TP7)")
    print("#" * 78)
    show(lab, "A0 baseline ALL", T, LIVE)

    print("\n--- singles (inverse-ablation: compare kept vs rejected) ---")
    for name, m in [("stack 9>21>50", m_stack), ("fresh-cross<=10", m_fresh),
                    ("breadth>=.50", m_breadth), ("breadth>=.55", m_breadth55),
                    ("mkt>EMA50", m_mkt), ("RS_20d>0", m_rs),
                    ("confirm-candle", m_conf), ("pullback>=.1ATR", m_pull),
                    ("ADX[18,35]", m_adx)]:
        show(lab, f"A.kept  {name}", m, LIVE)
        show(lab, f"A.rej   {name}", ~m, LIVE)

    print("\n--- nested funnel ---")
    f1 = m_stack
    f2 = f1 & m_breadth
    f3 = f2 & m_rs
    f4 = f3 & m_conf
    f5 = f4 & m_fresh
    show(lab, "F1 stack", f1, LIVE)
    show(lab, "F2 stack+breadth", f2, LIVE)
    show(lab, "F3 stack+breadth+RS", f3, LIVE)
    show(lab, "F4 +confirm", f4, LIVE)
    show(lab, "F5 +fresh-cross", f5, LIVE)

    # regime as modulator (mkt + breadth strong)
    m_regime = m_mkt & m_breadth55
    show(lab, "G regime(mkt&breadth55)+stack+RS", m_stack & m_regime & m_rs, LIVE)

    print("\n" + "#" * 78)
    print("# PART B — EXIT SWEEP")
    print("#   on baseline-ALL and on the best entry stack (set below)")
    print("#" * 78)

    # choose a strong-but-still-frequent entry stack for exit testing
    best_entry = m_stack & m_breadth & m_rs
    print(f"\n[best_entry = stack & breadth>=.50 & RS>0]  n={int(best_entry.sum())}")

    exits = {
        "live SL2.5/TP7": LIVE,
        "TP10": {"sl": 2.5, "tp": 10.0},
        "TP5": {"sl": 2.5, "tp": 5.0},
        "BE-arm@3": {"sl": 2.5, "tp": 7.0, "be_arm": 3.0},
        "timestop d2": {"sl": 2.5, "tp": 7.0, "green_by_bar": 12, "green_by_pct": 0.0},
        "pct-trail arm3 give2.5": {"sl": 2.5, "trail_arm": 3.0, "trail_give": 2.5},
        "pct-trail arm2 give2": {"sl": 2.5, "trail_arm": 2.0, "trail_give": 2.0},
        "atr-trail 2.5x arm2": {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": 2.5},
        "atr-trail 3.0x arm2": {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": 3.0},
        "atr-trail 2.0x arm1.5": {"sl_atr": 1.2, "trail_arm": 1.5, "trail_atr": 2.0},
        "atr-stop1.5 TPatr4": {"sl_atr": 1.5, "tp_atr": 4.0},
        "partial tp1@4 .5 + trail2.5": {"sl": 2.5, "tp1": 4.0, "tp1_frac": 0.5,
                                        "trail_give": 2.5},
    }

    print("\n--- exits on baseline ALL ---")
    for name, cfg in exits.items():
        show(lab, f"X.all {name}", T, cfg)

    print("\n--- exits on best_entry ---")
    for name, cfg in exits.items():
        show(lab, f"X.be  {name}", best_entry, cfg)

    print("\n" + "#" * 78)
    print("# PART C — ATR-trail multiplier sensitivity (plateau check)")
    print("#" * 78)
    for mult in [2.0, 2.25, 2.5, 2.75, 3.0, 3.5]:
        cfg = {"sl_atr": 1.5, "trail_arm": 2.0, "trail_atr": mult}
        show(lab, f"C atr-trail {mult}x | best_entry", best_entry, cfg)
