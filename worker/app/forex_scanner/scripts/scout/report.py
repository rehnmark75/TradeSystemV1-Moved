"""Markdown + CSV emission for strategy scout output.

Three artifacts per run:
    characterization.md   - market profile for the epic
    strategy_scores.csv   - every (template, params) combo with metrics
    recommendations.md    - ranked top-3 with CIs, regime champions, warnings
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from forex_scanner.scripts.scout.characterize import Characterization


# ============================================================================
# characterization.md
# ============================================================================

def write_characterization(c: Characterization, out_path: str) -> None:
    lines: List[str] = []
    lines.append(f"# Market Characterization — {c.epic}")
    lines.append("")
    lines.append(f"Lookback: **{c.days} days** | Generated: {datetime.now().isoformat()[:19]}")
    lines.append(f"Candle counts: 5m={c.candle_count_5m:,}  15m={c.candle_count_15m:,}  1h={c.candle_count_1h:,}")
    lines.append(f"Pip value: {c.pip_value}")
    lines.append("")

    lines.append("## Regime Distribution")
    lines.append("")
    lines.append("| Regime | 15m % | 1h % |")
    lines.append("|---|---|---|")
    for regime in ("ranging", "low_volatility", "trending", "breakout"):
        r15 = c.regime_15m.get(regime, 0.0) * 100
        r1h = c.regime_1h.get(regime, 0.0) * 100
        lines.append(f"| {regime} | {r15:.1f}% | {r1h:.1f}% |")
    lines.append("")

    lines.append("## Volatility (15m, pips)")
    lines.append("")
    v = c.volatility_15m
    lines.append(f"- ATR percentiles: p05={v.get('atr_p05',0):.1f}  p25={v.get('atr_p25',0):.1f}  p50={v.get('atr_p50',0):.1f}  p75={v.get('atr_p75',0):.1f}  p95={v.get('atr_p95',0):.1f}")
    lines.append(f"- BB width percentiles: p25={v.get('bb_width_p25',0):.1f}  p50={v.get('bb_width_p50',0):.1f}  p75={v.get('bb_width_p75',0):.1f}")
    lines.append("")

    lines.append("## Mean-Reversion / Trending Signature (5m returns)")
    lines.append("")
    m = c.mr_signature_5m
    lines.append("**Autocorrelation** (negative = mean-reverting, positive = trending at that lag):")
    lines.append("")
    lines.append("| Lag | AC |")
    lines.append("|---|---|")
    for k in ("ac_lag1", "ac_lag5", "ac_lag20", "ac_lag60"):
        lines.append(f"| {k.replace('ac_lag','')} bars | {m.get(k, 0):+.3f} |")
    lines.append("")
    lines.append("**Variance Ratio** (<1 = mean-reverting, >1 = trending, ~1 = random walk):")
    lines.append("")
    lines.append("| k | VR(k) |")
    lines.append("|---|---|")
    for k in ("vr_k2", "vr_k4", "vr_k8", "vr_k16", "vr_k32"):
        lines.append(f"| {k.replace('vr_k','')} bars | {m.get(k, 1.0):.3f} |")
    lines.append("")
    hl = m.get("ou_half_life_bars", 0.0)
    if hl == 0.0:
        lines.append("**Ornstein-Uhlenbeck half-life**: series is *not* mean-reverting at tradeable horizons (b >= 0 or below noise floor).")
    elif hl >= 1000:
        lines.append(f"**Ornstein-Uhlenbeck half-life**: ≥ 1000 bars (capped) — weak mean reversion on this horizon.")
    else:
        lines.append(f"**Ornstein-Uhlenbeck half-life**: {hl:.0f} bars ({hl*5/60:.1f} hours at 5m cadence)")
    lines.append("")

    lines.append("## Structural Stats (15m)")
    lines.append("")
    s = c.structural_15m
    lines.append(f"- Swings per day: **{s.get('swings_per_day', 0):.1f}**")
    lines.append(f"- Avg swing size: **{s.get('avg_swing_size_atr', 0):.2f} × ATR**")
    lines.append(f"- EQH/EQL touches per day: **{s.get('eq_touch_rate_per_day', 0):.1f}**")
    lines.append(f"- Consolidation ranges detected: **{s.get('range_count', 0)}**")
    lines.append(f"- Range-to-breakout rate: **{s.get('range_breakout_rate', 0)*100:.1f}%** (of detected ranges that broke out within 10 bars)")
    lines.append("")

    lines.append("## Session × Regime Heatmap (15m, UTC hours)")
    lines.append("")
    if not c.session_heatmap.empty:
        h = c.session_heatmap.copy()
        h["ranging"] = (h["ranging"] * 100).round(1)
        h["low_volatility"] = (h["low_volatility"] * 100).round(1)
        h["trending"] = (h["trending"] * 100).round(1)
        h["breakout"] = (h["breakout"] * 100).round(1)
        lines.append("| Hour | n | ranging% | low_vol% | trending% | breakout% |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in h.iterrows():
            lines.append(f"| {int(row['hour_utc']):02d} | {int(row['n_bars'])} | "
                         f"{row['ranging']:.1f} | {row['low_volatility']:.1f} | "
                         f"{row['trending']:.1f} | {row['breakout']:.1f} |")
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# strategy_scores.csv
# ============================================================================

def write_strategy_scores(rows: List[Dict], out_path: str) -> None:
    """Write one CSV row per (template, params) combo.

    Columns flatten by_regime dict into regime-specific columns for easier
    pivoting externally (Excel/pandas).
    """
    if not rows:
        pd.DataFrame(columns=["template", "params"]).to_csv(out_path, index=False)
        return

    flat: List[Dict] = []
    for r in rows:
        out = {
            "template": r["template"],
            "params": str(r["params"]),
            "n_trades": r["n_trades"],
            "win_rate": r["win_rate"],
            "pf": r["pf"],
            "total_pips": r["total_pips"],
            "avg_win_pips": r["avg_win"],
            "avg_loss_pips": r["avg_loss"],
            "sharpe": r["sharpe"],
            "max_drawdown_pips": r["max_drawdown"],
            "sl_count": r["sl_count"],
            "tp_count": r["tp_count"],
            "timeout_count": r["timeout_count"],
            "pf_ci_lower": r.get("pf_ci_lower", 0.0),
            "pf_ci_upper": r.get("pf_ci_upper", 0.0),
        }
        by_regime = r.get("by_regime", {})
        for regime in ("trending", "ranging", "low_volatility", "breakout"):
            sub = by_regime.get(regime, {})
            out[f"n_trades_{regime}"] = sub.get("n_trades", 0)
            out[f"pf_{regime}"] = sub.get("pf", 0.0)
            out[f"wr_{regime}"] = sub.get("win_rate", 0.0)
        flat.append(out)

    pd.DataFrame(flat).to_csv(out_path, index=False)


# ============================================================================
# recommendations.md
# ============================================================================

def write_recommendations(
    c: Characterization,
    rows: List[Dict],
    out_path: str,
    min_trades_gate: int = 30,
    ci_lower_gate: float = 1.05,
) -> None:
    """Ranked recommendations with CIs, regime champions, and red flags."""
    lines: List[str] = []
    lines.append(f"# Strategy Recommendations — {c.epic}")
    lines.append("")
    lines.append(f"Lookback: **{c.days} days** | {c.candle_count_5m:,} 5m bars")
    lines.append("")

    if not rows:
        lines.append("No strategy results to rank.")
        _write(out_path, lines)
        return

    df = pd.DataFrame(rows)

    # Filter out degenerate rows (n < 3) before ranking — a single lucky trade
    # with no losers shows PF = inf (capped at 999), which poisons the top list.
    # These rows still appear in the full CSV but don't drive the narrative.
    df_rankable = df[df["n_trades"] >= 3].copy()
    if df_rankable.empty:
        df_rankable = df.copy()

    # Aggregate per-template: best param set by PF
    per_template = df_rankable.sort_values("pf", ascending=False).groupby("template", as_index=False).first()

    # Grid sensitivity: stddev of PF across param grid per template (on rankable rows)
    sens = df_rankable.groupby("template")["pf"].std().to_dict()

    # ==== Executive summary ====
    lines.append("## Executive Summary")
    lines.append("")
    dom_regime_15m = max(c.regime_15m, key=lambda k: c.regime_15m[k] if k != "n_bars" else -1)
    lines.append(f"**Market profile:** {dom_regime_15m} dominates 15m (" +
                 f"{c.regime_15m.get(dom_regime_15m,0)*100:.1f}%). "
                 f"Swings {c.structural_15m.get('swings_per_day',0):.0f}/day, "
                 f"ATR median {c.volatility_15m.get('atr_p50',0):.1f} pips.")
    lines.append("")

    # Top 3 by best-param PF
    top3 = per_template.sort_values("pf", ascending=False).head(3)
    lines.append("**Top-3 strategy classes by best-param PF:**")
    lines.append("")
    lines.append("| Rank | Template | Best PF | CI lower | n | WR | Params |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, (_, row) in enumerate(top3.iterrows(), start=1):
        pf_lower = row.get("pf_ci_lower", 0.0)
        lines.append(f"| {i} | {row['template']} | {row['pf']:.2f} | "
                     f"{pf_lower:.2f} | {int(row['n_trades'])} | "
                     f"{row['win_rate']*100:.1f}% | `{row['params']}` |")
    lines.append("")

    # Bottom 3 — avoid list
    bot3 = per_template.sort_values("pf", ascending=True).head(3)
    lines.append("**Bottom-3 (don't bother):**")
    lines.append("")
    lines.append("| Template | Best PF | n | Params |")
    lines.append("|---|---|---|---|")
    for _, row in bot3.iterrows():
        lines.append(f"| {row['template']} | {row['pf']:.2f} | {int(row['n_trades'])} | `{row['params']}` |")
    lines.append("")

    # ==== Per-regime champions ====
    lines.append("## Per-Regime Champions")
    lines.append("")
    lines.append("For each canonical regime, the template with the highest PF when that regime was active at trade entry.")
    lines.append("")
    lines.append("| Regime | Champion | PF | n trades | WR |")
    lines.append("|---|---|---|---|---|")
    for regime in ("trending", "ranging", "low_volatility", "breakout"):
        best = None
        best_pf = 0.0
        for _, row in df.iterrows():
            sub = row.get("by_regime", {}).get(regime, {})
            if sub.get("n_trades", 0) < 10:
                continue
            if sub["pf"] > best_pf:
                best_pf = sub["pf"]
                best = (row["template"], sub)
        if best is None:
            lines.append(f"| {regime} | — | — | — | — |")
        else:
            tmpl, sub = best
            lines.append(f"| {regime} | {tmpl} | {sub['pf']:.2f} | "
                         f"{sub['n_trades']} | {sub['win_rate']*100:.1f}% |")
    lines.append("")

    # ==== Warnings ====
    lines.append("## Warnings & Red Flags")
    lines.append("")
    warnings: List[str] = []

    # Sample size
    low_n = per_template[per_template["n_trades"] < min_trades_gate]
    for _, row in low_n.iterrows():
        warnings.append(f"- **{row['template']}**: only {int(row['n_trades'])} trades (< {min_trades_gate}). "
                       f"Stats are not reliable — rerun with longer window.")

    # CI lower below gate
    weak_ci = per_template[per_template.get("pf_ci_lower", pd.Series([0])) < ci_lower_gate]
    for _, row in weak_ci.iterrows():
        pf_lower = row.get("pf_ci_lower", 0.0)
        if row["pf"] > 1.2 and pf_lower < ci_lower_gate:
            warnings.append(f"- **{row['template']}**: PF {row['pf']:.2f} looks good, "
                           f"but 95% CI lower = {pf_lower:.2f} (< {ci_lower_gate}). "
                           f"Edge is not statistically robust.")

    # Grid sensitivity
    for template, stddev in sens.items():
        if pd.isna(stddev):
            continue
        if stddev > 0.5:
            warnings.append(f"- **{template}**: PF stddev across param grid = {stddev:.2f}. "
                           f"Strong grid sensitivity — risk of overfit to a lucky parameter.")

    if warnings:
        lines.extend(warnings)
    else:
        lines.append("_No major warnings._")
    lines.append("")

    # ==== Next steps ====
    lines.append("## Next Steps")
    lines.append("")

    # Qualified candidates: clear the n gate first. Among those, rank by PF.
    # This surfaces the real contenders, ignoring noisy high-PF small-n results.
    qualified = per_template[per_template["n_trades"] >= min_trades_gate].sort_values("pf", ascending=False)

    if qualified.empty:
        lines.append(f"1. **No template cleared the sample-size gate (n ≥ {min_trades_gate})** in this window.")
        lines.append(f"   Next: rerun with a longer window (e.g., 180 days) before trusting any rank.")
    else:
        best = qualified.iloc[0]
        pf_lower = best.get("pf_ci_lower", 0.0)
        if pf_lower >= ci_lower_gate:
            lines.append(f"1. **{best['template']}** is the qualified leader "
                         f"(PF {best['pf']:.2f}, CI lower {pf_lower:.2f}, n={int(best['n_trades'])}). "
                         f"Clears both the sample-size and CI gates.")
            lines.append(f"   Next: implement as a full strategy (with proper HTF context, exits, etc.) "
                         f"and run a richer bt.py backtest with the live-trading simulator.")
        elif pf_lower >= 0.85:
            lines.append(f"1. **{best['template']}** is the qualified leader "
                         f"(PF {best['pf']:.2f}, CI lower {pf_lower:.2f}, n={int(best['n_trades'])}) "
                         f"— passes sample-size but CI lower below {ci_lower_gate}. Edge is likely real but "
                         f"not yet statistically bulletproof.")
            lines.append(f"   Next: (a) rerun on longer window to tighten CI, or "
                         f"(b) run on a different pair to cross-validate, or "
                         f"(c) tune SL/TP per-template (uniform fixtures understate some edges).")
        else:
            lines.append(f"1. Qualified leader **{best['template']}** (n={int(best['n_trades'])}) has "
                         f"PF {best['pf']:.2f} but CI lower {pf_lower:.2f} — edge is weak.")
            lines.append(f"   Next: widen template library or test different instrument; this one doesn't "
                         f"reward any of the canonical hypotheses strongly.")

        # Also list any small-n templates with outlier-high PF (>=2) as "worth investigating with more data"
        interesting_small = per_template[
            (per_template["n_trades"] < min_trades_gate)
            & (per_template["pf"] >= 2.0)
        ].sort_values("pf", ascending=False)
        if not interesting_small.empty:
            lines.append("")
            lines.append(f"2. Promising small-n candidates (PF ≥ 2.0, n < {min_trades_gate} — rerun on longer window to confirm):")
            for _, row in interesting_small.iterrows():
                lines.append(f"   - {row['template']} PF {row['pf']:.2f} at n={int(row['n_trades'])}")

    lines.append("")
    lines.append("3. Hand this report to `trading-strategy-analyst` or `quantitative-researcher` for interpretation, "
                 "sanity-check against domain knowledge, and next-design decisions.")
    lines.append("")

    _write(out_path, lines)


def _write(out_path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
