"""Vision prompt for the IMPULSE_FADE strategy."""
import logging
from typing import Dict
from ._helpers import format_price, extract_pair, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """
    Vision prompt for IMPULSE_FADE (large 5m candle body exhaustion fade).

    Edge: In the late-US session (20-22 UTC), large-body 5m candles (≥2.2×ATR14) tend
    to exhaust rather than continue. Entry fades the impulse at the close of that candle.
    No HTF alignment — purely behavioural (late-session stop-run exhaustion).
    Inverted R:R is allowed and compensated by the high WR thesis.
    """
    try:
        epic = signal.get('epic', 'Unknown')
        pair = extract_pair(epic)
        direction = signal.get('signal_type', signal.get('signal', 'Unknown')).upper()
        confidence = signal.get('confidence_score', signal.get('confidence', 0))
        entry_price = signal.get('entry_price', signal.get('price', 0))
        risk_pips = signal.get('risk_pips', 12)
        reward_pips = signal.get('reward_pips', 8)
        monitor_only = bool(signal.get('monitor_only', False))
        body_pips = signal.get('body_pips', 0)
        atr_pips = signal.get('atr_pips', 0)
        body_multiplier = signal.get('body_multiplier', 0)
        session_hour = signal.get('session_hour', 0)

        def fmt(v, prec=1, suffix=""):
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.{prec}f}{suffix}"
            except Exception:
                return str(v)

        if direction in ('BUY',):
            impulse_direction = "BEARISH impulse (large down-body candle)"
            fade_rationale = "price dropped sharply, now fading back up — exhaustion of sellers"
            reject_if = "the bearish candle was followed by another large bearish candle (continuation, not exhaustion)"
            approve_if = "the impulse candle has an upper wick or the next candle shows upward hesitation"
        else:
            impulse_direction = "BULLISH impulse (large up-body candle)"
            fade_rationale = "price spiked up sharply, now fading back down — exhaustion of buyers"
            reject_if = "the bullish candle was followed by another large bullish candle (continuation, not exhaustion)"
            approve_if = "the impulse candle has a lower wick or the next candle shows downward hesitation"

        chart_instruction = ""
        if has_chart:
            chart_instruction = f"""
## CHART ANALYSIS (EXHAUSTION-FADE LENS)

The chart shows the 5m timeframe — this is a single-timeframe strategy, so focus here only.

**What triggered this signal:**
- A {impulse_direction} whose body was {fmt(body_multiplier)}× the 14-period ATR ({fmt(body_pips)} pips body vs {fmt(atr_pips)} pips ATR)
- Signal fires at the CLOSE of that impulse candle; entry fades the move

**What to look for in the chart:**

✅ **APPROVE if:**
- The impulse candle stands out clearly — much larger body than surrounding candles
- {approve_if}
- Surrounding context shows choppy / ranging price before the spike (no established trend)
- The impulse candle ran into a visible S/R level, prior high/low, or round number
- Session context: late-US hours (20-22 UTC) — low-liquidity stop-run environment

❌ **REJECT if:**
- {reject_if}
- The impulse is part of a clear sustained trend with multiple prior large candles in the same direction
- Multiple large-body candles in a row (momentum, not a single exhaustion spike)
- The impulse broke through a major structural level cleanly (breakout, not exhaustion)
- Very wide-ranging market: if every candle is large, the 2.2× ATR threshold fires trivially
"""

        mo_note = (
            "\n⚠️ **MONITOR-ONLY / TEST MODE:** This is a demo-environment test signal. Score honestly; evaluate as if capital is at risk.\n"
            if monitor_only else ""
        )

        return f"""You are a SENIOR FOREX TECHNICAL ANALYST evaluating a Late-Session Impulse Fade signal.

**STRATEGY THESIS — READ FIRST (IMPULSE_FADE v1)**
IMPULSE_FADE fires when a 5m candle has an unusually large body (≥2.2× ATR14) during the late-US
session (20-22 UTC). The thesis: large impulsive moves in low-liquidity late-US hours frequently
exhaust rather than continue — they are often stop-hunts or overextensions that snap back.

Entry is a COUNTER-DIRECTION trade at the close of the impulse candle:
- Large UP candle → SHORT (fade the spike up)
- Large DOWN candle → LONG (fade the spike down)

**This is NOT a trend-following strategy.** No HTF alignment is used or expected.
The edge is purely behavioural: late-session exhaustion after an oversized single candle.

Critical rules for your analysis:
- ❌ Do NOT reject because the HTF trend opposes — no HTF filter exists in this strategy
- ❌ Do NOT reject because RSI is extreme — impulses fire at extremes by design
- ❌ Do NOT reject because R:R looks inverted (TP={fmt(reward_pips)} pips, SL={fmt(risk_pips)} pips) — this is by design; high WR compensates
- ✅ DO reject if price continued strongly in the impulse direction after the trigger candle
- ✅ DO reject if the impulse is clearly part of sustained momentum (multiple large candles in a row)
- ✅ DO reject if the candle looks like a genuine breakout through a major level (not exhaustion)
{mo_note}
═══════════════════════════════════════════════════════════════
✅ THE SETUP IS THE TRIGGER — DO NOT REJECT FOR THESE
═══════════════════════════════════════════════════════════════
These ARE the IMPULSE_FADE conditions — not red flags:
- HTF trend opposing the entry direction (no HTF filter — ignore this entirely)
- RSI at extreme values (impulse fades happen at extremes by definition)
- Inverted R:R of ~8 TP / 12 SL (this is the intended design — high WR compensates)
- Low tick volume (FX tick volume is not reliable for this strategy)
- The impulse candle looking "large" (that is literally the trigger — ≥2.2× ATR14)

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction} (fading the {impulse_direction})
• Strategy: IMPULSE_FADE v1
• Rationale: {fade_rationale}
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {format_price(entry_price)}
• Stop Loss: {fmt(risk_pips)} pips  (beyond the impulse candle extreme)
• Take Profit: {fmt(reward_pips)} pips  (fixed target)
• Risk:Reward: {float(reward_pips)/float(risk_pips) if float(risk_pips) > 0 else 0:.2f}:1  (inverted by design — ~80% WR compensates)

═══════════════════════════════════════════════════════════════
🔬 IMPULSE DATA (computed by strategy)
═══════════════════════════════════════════════════════════════
• Impulse Body: {fmt(body_pips)} pips  ({fmt(body_multiplier)}× ATR14)
• ATR14 at signal: {fmt(atr_pips)} pips
• Session Hour (UTC): {session_hour}:00  (valid window: 20-22 UTC)
{chart_instruction}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

**CRITICAL:** Begin with `SCORE:`. No preamble.

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | IF_CONTINUATION | IF_SUSTAINED_MOMENTUM | IF_BREAKOUT | IF_TRIVIAL_TRIGGER]
REASON: [≤40 words. Focus on: (a) whether impulse stands out as isolated spike vs. sustained momentum, (b) any exhaustion signals (wick, hesitation), (c) whether price context supports a snap-back. Do NOT mention HTF trend, RSI, MACD, or R:R ratio.]

**SCORING RUBRIC (4 anchors):**
- 9–10 EXCEPTIONAL: Isolated oversized spike with visible exhaustion (wick on impulse candle, next candle hesitates), no established momentum trend beforehand, clean late-US session context
- 7–8  STRONG: Large impulse but follow-through uncertain, or minor continuation before signal but impulse still stands out
- 5–6  ACCEPTABLE: Impulse is noticeably large, isolated enough, session timing valid — positive expectancy midband
- 3–4  MARGINAL: Impulse is part of a developing trend, or body multiplier marginal — APPROVE unless rejection criterion clearly met
- 1–2  REJECT: one of the rejection criteria below is clearly true

**REJECTION CRITERIA (any one = REJECT):**
- Clear follow-through: candle after the impulse is also a large-body candle in the same direction → IF_CONTINUATION
- Sustained momentum: 3+ large candles in the impulse direction without meaningful pullback → IF_SUSTAINED_MOMENTUM
- Clean structural breakout: price was pressing against a key level and this candle broke it with body close through → IF_BREAKOUT
- Market is broadly wide-ranging: most candles are large (trivial ATR threshold trigger) → IF_TRIVIAL_TRIGGER

Be concise. Your four lines determine if real capital is risked on this edge."""

    except Exception as e:
        logger.error(f"Error building IMPULSE_FADE prompt: {e}")
        return build_fallback_prompt(signal)
