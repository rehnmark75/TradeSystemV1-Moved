"""
Trade postmortem agent — calls Claude Sonnet to critique a closed trade.

Uses prompt caching on the static system prompt to keep per-trade cost low.
Only called for closed trades with confirmed P&L; never in the hot-path.
"""

import os
import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Static system prompt — cached by Anthropic (5-min TTL).
# Keep changes to this block minimal to maximise cache hits.
# Strategy-specific context is injected per-call in the user message.
_SYSTEM_PROMPT = """\
You are a forex trade analyst for an automated multi-strategy trading system on IG Markets.

## System Context
- Trailing stops: 4 progressive stages → Break-Even → Stage1 (profit lock) → Stage2 (profit lock) → Stage3 (ATR trail)
- Account currency: SEK  |  Instrument: FX majors/crosses

## Trailing Stage Triggers (typical values, pair-specific)
- Break-Even: SL moves to entry when profit ≥ trigger_pts
- Stage 1:    Locks partial pips once profit reaches stage1_trigger_pts
- Stage 2:    Locks more pips once profit reaches stage2_trigger_pts
- Stage 3:    ATR-based dynamic trail for the run-up

## Your task
Given structured trade data, write a concise post-mortem. Evaluate the entry against
the STRATEGY CONTEXT provided in the user message — not against a generic SMC rubric.
Focus on:
1. Was the entry setup strong FOR THIS STRATEGY?
2. Did the trade capture available profit (MFE efficiency)?
3. Did the trailing stop system behave appropriately?

Describe what happened for THIS trade. Do NOT recommend config/parameter changes —
a single trade is too small a sample to tune the system. Strategy-level tuning is
done separately from aggregate analysis, not from one post-mortem.

## Output format — JSON only, no other text
{
  "entry_verdict": "GOOD" | "AVERAGE" | "POOR",
  "entry_notes": "1-2 sentences",
  "exit_verdict": "OPTIMAL" | "GOOD" | "PREMATURE" | "REVERSAL" | "ADVERSE",
  "exit_notes": "1-2 sentences",
  "trailing_verdict": "EFFECTIVE" | "PREMATURE" | "MISSED_LOCK" | "NOT_TRIGGERED",
  "trailing_notes": "1-2 sentences",
  "key_lesson": "one sentence",
  "tags": [up to 3 from: "TEXTBOOK_TRADE","PROFITABLE_REVERSAL","FAILED_ENTRY","PREMATURE_BE",
           "MISSED_TP","GOOD_TRAILING","POOR_TIMING","HIGH_MFE_MAE","IMMEDIATE_ADVERSE","HELD_THROUGH_DRAWDOWN"]
}\
"""

# Per-strategy context injected into the user message so the agent
# evaluates entries against the correct rubric.
_STRATEGY_CONTEXT = {
    "SMC_SIMPLE": (
        "3-tier Smart Money Concepts strategy: "
        "TIER1 (1H EMA bias/direction) → TIER2 (5m swing break) → TIER3 (1m pullback entry). "
        "Good entry = strong HTF EMA alignment, clean swing break, pullback into optimal zone."
    ),
    "RANGE_FADE": (
        "Controlled mean-reversion fade of 5m Bollinger Band touches on FX majors, taken within the "
        "London/NY window (06-18 UTC). Triggers when price touches a BB edge (lower band = long, upper "
        "band = short), RSI is past a MODERATE threshold (~40 for longs / ~60 for shorts — NOT an "
        "extreme >70/<30), the 1H HTF EMA50+slope bias ALIGNS with the fade direction (required and "
        "strict — it fades the 5m stretch IN THE DIRECTION of the 1H trend, not against it), and ADX is "
        "low/ranging (an ADX CEILING, not a floor). This is NOT an impulse fade and has NO 20-22 UTC / "
        "late-US session requirement. Good entry = wick/stall at the band (not an expansion candle or "
        "band-walk), HTF bias aligned, low ADX. SL/TP are ATR/band-width based. Do NOT penalise a trade "
        "for session timing, for RSI not being at 70/30 extremes, or for lacking HTF EMA-tier structure."
    ),
    "SMC_MOMENTUM": (
        "Momentum-based SMC strategy using BOS/CHOCH with MACD confirmation. "
        "Good entry = clear BOS/CHOCH, MACD aligned, strong trend context."
    ),
    "XAU_GOLD": (
        "Gold 3-tier SMC, trending regime only: 4H EMA50/200 bias + market structure (HH/HL vs LH/LL) "
        "→ 1H BOS/CHOCH trigger → 15m OB/FVG (or 50% fib) pullback entry. Good entry = pullback into an "
        "order block / FVG, HTF-aligned, ADX in the ~25-35 trending zone. Ranging/expansion regimes and "
        "the 21-22 UTC rollover are blocked by the strategy. Large fixed stop (~80 pips) is by design for "
        "gold ATR — do NOT flag it. Entry quality is about OB/FVG confluence + trend structure, not RSI."
    ),
    "IMPULSE_FADE": (
        "Fades large impulsive 5m candle bodies (>=2.2x ATR14) during the late-US session (20-22 UTC) — a "
        "pure behavioural exhaustion edge. NO HTF alignment is required (missing/neutral HTF bias is "
        "normal — do NOT penalise it). Inverted R:R by design (TP~8 / SL~15) — do NOT flag R:R < 1. Good "
        "entry = larger body-vs-ATR ratio (stronger exhaustion). Ranging / low-ADX is fine, not a weakness."
    ),
    "MEAN_REVERSION": (
        "Bollinger Band + RSI-extreme reversion; low-ADX / ranging IS the target condition (do NOT "
        "penalise low ADX or ranging). Two variants — TOUCH entry (price touches the band + RSI extreme in "
        "a low-vol ATR regime, ~18-22 UTC) and REJECTION entry (prior bar breaches the band + RSI extreme, "
        "current bar closes back inside; hard ADX ceiling). Good entry = the variant's trigger met at a "
        "band extreme in a quiet/ranging regime. Elevated ATR or high ADX is the weakness."
    ),
    "DONCHIAN_TURTLE": (
        "20-bar Donchian channel breakout on 1H bars; long-only (SELL disabled). Trend-following — high "
        "ADX / trending regime IS the edge (do NOT treat a strong trend as 'overextended'). Good entry = a "
        "clean breakout (close clearly beyond the 20-bar high) with EMA50 > EMA200. Ranging / choppy "
        "(ADX < 20) is the weak condition. Entry quality is about breakout strength and trend context."
    ),
    "INSIDE_DAY": (
        "Daily inside-day breakout filtered by weekly directional bias. Waits for price to break the "
        "completed inside-day high/low on 5m; the stop sits beyond the opposite inside-day extreme plus a "
        "daily ATR buffer (by design). Good entry = breakout direction aligns with weekly_bias and the "
        "inside-day range is within the configured band (not a tiny-noise or over-wide range). Edge is "
        "daily compression + weekly bias, NOT intraday SMC swing quality."
    ),
    "FA_OR_ATR_TRAIL": (
        "Two variants — always read the MODEL field. FA (Failed Auction): price swept a prior extreme then "
        "closed back inside the value area (the rejection IS the signal). OR (Opening Range): break of the "
        "London/NY opening-range high/low after a lock period. ADX ~18-25 is the sweet spot (range-to-trend "
        "transition) — do NOT penalise mid ADX. Good entry = directional SLOPE (>=0.3), price well-anchored "
        "vs 4H EMA50 (HTF margin >= ~1 ATR), ATR-based stops by design. FA + ranging is a valid combination."
    ),
    "KAMA_V2": (
        "5m KAMA(10,2,30) adaptive moving-average crossover. Entry requires price crossing KAMA, the "
        "efficiency ratio (ER) above the configured threshold, EMA200 alignment, MACD-histogram sign "
        "confirmation, and RSI not at an exhaustion extreme. Good entry = high ER (strong directional "
        "efficiency) with EMA200 and MACD aligned. SL/TP ~10/15 (≈1.5R) is by design. ADX is not a default "
        "gate — do NOT reject on low ADX alone."
    ),
    "SMC_SIMPLE_V2": (
        "Stripped-down SMC_SIMPLE research scalp: 5m structure context + 1m rejection-break entry. The "
        "validated launch config is narrow (EURUSD, BUY-only, 07-12 UTC, fixed SL/TP ~5/6 pips, "
        "rejection-break model). Good entry = a clean 1m rejection-break in the direction of 5m structure. "
        "Confidence is coarse (~0.69) and is NOT a fine-grained quality ranker — do NOT apply SMC_SIMPLE's "
        "inverse-confidence bands here."
    ),
}


def _epic_to_pair(epic: str) -> str:
    m = re.search(r"CS\.D\.([A-Z]+)\.", epic or "")
    return m.group(1) if m else (epic or "UNKNOWN")


def _build_user_message(
    trade_data: Dict[str, Any],
    alert_data: Dict[str, Any],
    mfe_mae: Dict[str, Any],
    entry_quality: Dict[str, Any],
    exit_quality: Dict[str, Any],
    trailing_config: Dict[str, Any],
) -> str:
    pair = _epic_to_pair(trade_data.get("symbol", ""))
    direction = trade_data.get("direction", "?")
    pnl = trade_data.get("profit_loss", 0) or 0
    duration = trade_data.get("duration_minutes", 0) or 0
    dur_text = f"{duration}min" if duration < 60 else f"{duration // 60}h {duration % 60}m"

    be = "YES" if trade_data.get("moved_to_breakeven") else "NO"
    s1 = "YES" if trade_data.get("moved_to_stage1") else "NO"
    s2 = "YES" if trade_data.get("moved_to_stage2") else "NO"

    strategy = alert_data.get("strategy", "UNKNOWN")
    strategy_desc = _STRATEGY_CONTEXT.get(strategy, f"Strategy: {strategy} (no description available)")

    lines = [
        f"## Strategy Context",
        f"**Strategy:** {strategy} — {strategy_desc}",
        f"",
        f"## Trade: {pair} {direction}",
        f"",
        f"**Result:** P&L {pnl:+.2f} SEK | {exit_quality.get('actual_pips', 0):+.1f} pips | "
        f"Duration: {dur_text} | Exit: {exit_quality.get('exit_type', '?')}",
        f"",
        f"**MFE/MAE:**",
        f"- MFE: +{mfe_mae.get('mfe_pips', 0):.1f} pips "
        f"({mfe_mae.get('mfe_time_to_peak_minutes', 0)} min to peak, "
        f"{mfe_mae.get('percentage_of_tp', 0):.0f}% of TP distance)",
        f"- MAE: -{mfe_mae.get('mae_pips', 0):.1f} pips "
        f"({mfe_mae.get('percentage_of_sl', 0):.0f}% of SL distance)",
        f"- MFE/MAE ratio: {mfe_mae.get('mfe_mae_ratio', 0):.2f}",
        f"- Initial move: {mfe_mae.get('initial_move', '?')} | "
        f"Immediate reversal: {mfe_mae.get('immediate_reversal', False)}",
        f"",
        f"**Exit quality:**",
        f"- Efficiency: {exit_quality.get('exit_efficiency_pct', 0):.0f}% of MFE captured",
        f"- Missed profit: {exit_quality.get('missed_profit_pips', 0):.1f} pips",
        f"- Verdict (rule-based): {exit_quality.get('verdict', '?')}",
        f"",
        f"**Entry quality (rule-based score {entry_quality.get('score', 0):.0f}/100):**",
        f"- Verdict: {entry_quality.get('verdict', '?')} | HTF aligned: {entry_quality.get('htf_aligned', False)}",
        f"- Confluence count: {entry_quality.get('confluence_count', 0)}",
        f"",
        f"**Trailing progression:** BE={be}, Stage1={s1}, Stage2={s2}",
        f"**Trailing config:** "
        f"BE_trigger={trailing_config.get('break_even_trigger_points', '?')}pts, "
        f"Stage1_trigger={trailing_config.get('stage1_trigger_points', '?')}pts, "
        f"Stage2_trigger={trailing_config.get('stage2_trigger_points', '?')}pts",
    ]

    if alert_data:
        lines += [
            f"",
            f"**Signal context:** "
            f"confidence={alert_data.get('confidence_score', 0):.2f}, "
            f"session={alert_data.get('market_session', '?')}, "
            f"regime={alert_data.get('market_regime', '?')}",
        ]

    lines += ["", "Return JSON post-mortem:"]
    return "\n".join(lines)


async def generate_postmortem(
    trade_data: Dict[str, Any],
    alert_data: Dict[str, Any],
    mfe_mae: Dict[str, Any],
    entry_quality: Dict[str, Any],
    exit_quality: Dict[str, Any],
    trailing_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Call Claude Sonnet to generate a structured post-mortem for a closed trade.
    Returns the parsed dict (with _meta key for token usage) or None on failure.
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        logger.warning("CLAUDE_API_KEY not set — skipping postmortem")
        return None

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed — cannot generate postmortem")
        return None

    user_message = _build_user_message(
        trade_data, alert_data, mfe_mae, entry_quality, exit_quality, trailing_config
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        result = json.loads(raw_text)
        result["_meta"] = {
            "model": "claude-sonnet-4-6",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            "cache_write_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        }
        return result

    except json.JSONDecodeError as e:
        raw_preview = locals().get("raw_text", "")[:300]
        logger.error(f"Postmortem JSON parse error: {e} | raw: {raw_preview}")
        return None
    except Exception as e:
        logger.error(f"Postmortem agent call failed: {e}")
        return None
