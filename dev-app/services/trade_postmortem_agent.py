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
4. One concrete config suggestion (or "none" if nothing actionable).

## Output format — JSON only, no other text
{
  "entry_verdict": "GOOD" | "AVERAGE" | "POOR",
  "entry_notes": "1-2 sentences",
  "exit_verdict": "OPTIMAL" | "GOOD" | "PREMATURE" | "REVERSAL" | "ADVERSE",
  "exit_notes": "1-2 sentences",
  "trailing_verdict": "EFFECTIVE" | "PREMATURE" | "MISSED_LOCK" | "NOT_TRIGGERED",
  "trailing_notes": "1-2 sentences",
  "config_delta": {
    "suggestion": "none" | "tighten_be" | "loosen_be" | "tighten_stage1" | "loosen_stage1" | "add_partial_tp" | "other",
    "rationale": "one sentence",
    "pair": "e.g. EURUSD or 'all'",
    "suggested_value": null or number
  },
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
        "Fades large impulsive 5m candle bodies (≥2.2×ATR14) in late-US session (20-22 UTC). "
        "Good entry = price AT or just beyond Bollinger Band, RSI extreme (>70 or <30), "
        "fade direction aligned with HTF bias, low ADX (<25, ranging market). "
        "Entry quality is NOT based on trend-following or HTF EMA tiers."
    ),
    "SMC_MOMENTUM": (
        "Momentum-based SMC strategy using BOS/CHOCH with MACD confirmation. "
        "Good entry = clear BOS/CHOCH, MACD aligned, strong trend context."
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
