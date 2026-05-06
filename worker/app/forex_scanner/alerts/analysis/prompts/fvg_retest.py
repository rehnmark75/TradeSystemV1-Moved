"""Vision prompt for the FVG_RETEST strategy."""
import logging
from typing import Dict
from ._helpers import format_price, extract_pair, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """
    Build vision-enabled prompt for FVG Retest strategy analysis.
    Covers both Type A (Deep Value / FVG Tap) and Type B (Institutional Initiation).
    """
    try:
        epic = signal.get('epic', 'Unknown')
        pair = extract_pair(epic)
        direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
        confidence = signal.get('confidence_score', 0)
        entry_type = signal.get('entry_type', 'DEEP_VALUE')

        entry_price = signal.get('entry_price', signal.get('price', 0))
        risk_pips = signal.get('sl_pips', signal.get('risk_pips', 0))
        reward_pips = signal.get('tp_pips', signal.get('reward_pips', 0))
        rr_ratio = signal.get('rr_ratio', 0)

        if entry_type == 'DEEP_VALUE':
            fvg_zone = signal.get('fvg_zone', 'N/A')
            fvg_size = signal.get('fvg_size_pips', 0)
            fvg_significance = signal.get('fvg_significance', 0)
            setup_age = signal.get('setup_age_minutes', 0)
            entry_detail = f"""**TYPE A - DEEP VALUE (FVG Tap) Entry:**
- FVG Zone: {fvg_zone}
- FVG Size: {fvg_size:.1f} pips
- FVG Significance: {fvg_significance:.3f}
- Setup Age: {setup_age:.0f} minutes since BOS
- Swing Level (invalidation): {format_price(signal.get('swing_level', 0))}
- Entry Logic: Price retraced into an unfilled Fair Value Gap created during a Break of Structure"""
        else:
            displacement = signal.get('displacement_ratio', 0)
            follow_through = signal.get('follow_through', False)
            volume_spike = signal.get('volume_spike', False)
            entry_detail = f"""**TYPE B - INSTITUTIONAL INITIATION Entry:**
- Displacement: {displacement:.2f}x ATR (break candle body vs average)
- Follow-Through: {'✅ Confirmed' if follow_through else '❌ Not confirmed'}
- Volume Spike: {'✅ Above average' if volume_spike else '❌ Below average'}
- Swing Level (invalidation): {format_price(signal.get('swing_level', 0))}
- Entry Logic: High-velocity BOS with institutional displacement — immediate entry"""

        chart_instruction = ""
        if has_chart:
            chart_instruction = f"""
## CHART ANALYSIS (EXAMINE CAREFULLY)

The chart shows multi-timeframe analysis:

**Timeframes:**
- 1H: Shows 200 EMA trend bias (macro direction filter)
- 5m: Shows Break of Structure, FVG zones, and entry/SL/TP levels

**What to look for:**
- GREEN dashed line: Entry price
- RED dashed line: Stop loss
- BLUE dashed line: Take profit
- FVG zones (if visible): Semi-transparent shaded areas

**CHART ANALYSIS CHECKLIST:**
1. Is price clearly on the correct side of the 1H 200 EMA?
2. Is the Break of Structure clean (clear swing high/low violation)?
3. {'Is the FVG zone well-defined and price tapping into it cleanly?' if entry_type == 'DEEP_VALUE' else 'Are the displacement candles strong and impulsive?'}
4. Is there clean price structure (not choppy/ranging)?
5. Is stop loss behind a valid structural level?
6. Are there any reversal patterns at entry (engulfing, pin bars against direction)?
7. **SWING PROXIMITY CHECK:**
   - For BUY: Is entry dangerously close to a recent swing HIGH? (buying into resistance)
   - For SELL: Is entry dangerously close to a recent swing LOW? (selling into support)
   - Within 5-10 pips of opposing swing = major concern
"""

        return f"""You are a SENIOR FOREX TECHNICAL ANALYST specializing in Smart Money Concepts (SMC) and Fair Value Gap analysis.

**YOUR ROLE:** Validate this FVG Retest strategy signal. This strategy detects Breaks of Structure on the 5m chart with 1H 200 EMA macro confirmation, then enters via FVG retest (Type A) or institutional displacement (Type B).

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: FVG_RETEST
• Entry Mode: {entry_type}
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {format_price(entry_price)}
• Stop Loss: {risk_pips:.1f} pips
• Take Profit: {reward_pips:.1f} pips
• Risk:Reward Ratio: {rr_ratio:.2f}:1
{chart_instruction}
═══════════════════════════════════════════════════════════════
🔬 STRATEGY-SPECIFIC DATA
═══════════════════════════════════════════════════════════════

**MACRO FILTER (1H):**
- 200 EMA Direction: Price {'above' if direction in ('BUY', 'BULL') else 'below'} EMA → {direction} bias
- Last 1H candle confirmed direction alignment

**TRIGGER (5m):**
- Break of Structure detected on 5m chart
- BOS direction matches 1H macro bias

{entry_detail}

═══════════════════════════════════════════════════════════════
✅ THE SETUP IS THE TRIGGER — DO NOT REJECT FOR THESE
═══════════════════════════════════════════════════════════════
- A BOS followed by a retest into an FVG IS the entry mechanism — the retest is not a weakness
- Price pulling back counter to the BOS direction IS the FVG tap — that is the whole thesis
- Displacement candles that look "overextended" on the 5m ARE the institutional initiation signal

═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

**CRITICAL:** Begin with `SCORE:`. No preamble.

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | REJECT]
REASON: [≤40 words. If REJECT: name the specific criterion. If APPROVE_*: state primary positive factor.]

**SCORING RUBRIC (4 anchors):**
- 9–10 EXCEPTIONAL: Clean BOS, strong 1H trend, {'well-defined FVG with clean tap' if entry_type == 'DEEP_VALUE' else 'strong displacement with confirmed follow-through'}, no reversal signs, clear of swing opposition
- 7–8  STRONG: Good BOS and 1H trend alignment, minor concerns (slightly choppy structure or partial FVG fill)
- 5–6  ACCEPTABLE: BOS present, 1H trend aligned, entry quality acceptable — positive expectancy midband
- 3–4  MARGINAL: Weak trend, unclear BOS, or entry quality concerns — APPROVE unless criteria below are fully met
- 1–2  REJECT: one of the rejection criteria below is clearly true

**REJECTION CRITERIA (any one = REJECT):**
- Price on wrong side of 1H 200 EMA (counter-trend)
- BOS not clearly visible on 5m chart
- Reversal candlestick patterns at entry level
- Choppy, range-bound price action with no clear structure
- BUY entry within 5 pips of a recent multi-touch swing HIGH
- SELL entry within 5 pips of a recent multi-touch swing LOW

**NOT rejection criteria:**
- The FVG retest going deeper than expected (partial fill is normal)
- RSI or MACD opposing the entry — structural to this strategy
- Volume not confirmed on the retest candle

Be concise. Your four lines determine if real capital is risked."""

    except Exception as e:
        logger.error(f"Error building FVG Retest prompt: {e}")
        return build_fallback_prompt(signal)
