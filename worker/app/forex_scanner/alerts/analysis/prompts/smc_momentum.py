"""Vision prompt for the SMC_MOMENTUM strategy (Liquidity Sweep + Rejection Wick)."""
import logging
from typing import Dict
from ._helpers import format_price, extract_pair, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """
    Vision prompt for SMC_MOMENTUM (Liquidity Sweep + Rejection Wick).

    Entry fires ON the sweep candle (or the bar immediately after).
    Mechanic: price runs a prior H/L to hunt stops, then closes back inside.
    HTF filter: 4H EMA50 confirms direction.
    """
    try:
        epic = signal.get('epic', 'Unknown')
        pair = extract_pair(epic)
        direction = signal.get('signal_type', signal.get('signal', 'Unknown')).upper()
        confidence = signal.get('confidence_score', signal.get('confidence', 0))
        entry_price = signal.get('entry_price', signal.get('price', 0))
        risk_pips = signal.get('risk_pips', 0)
        reward_pips = signal.get('reward_pips', 0)
        monitor_only = bool(signal.get('monitor_only', False))

        try:
            rr_ratio = float(reward_pips) / float(risk_pips) if float(risk_pips) > 0 else 0
        except Exception:
            rr_ratio = 0

        meta = signal.get('strategy_metadata', {}) or {}
        swept_pool_type = meta.get('swept_pool_type', 'unknown')
        pool_level = meta.get('pool_level', 0)
        excess_pips = meta.get('excess_pips', 0)
        wick_pct = meta.get('wick_pct', 0)
        is_double_sweep = bool(meta.get('is_double_sweep', False))
        htf_bias = meta.get('htf_bias', 'unknown')

        def fmt(v, prec=2, suffix=""):
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.{prec}f}{suffix}"
            except Exception:
                return str(v)

        pool_labels = {
            'PRIOR_DAY_HIGH': 'Prior Day High',
            'PRIOR_DAY_LOW': 'Prior Day Low',
            'PRIOR_LONDON_HIGH': 'Prior London Session High',
            'PRIOR_LONDON_LOW': 'Prior London Session Low',
            'PRIOR_NY_HIGH': 'Prior NY Session High',
            'PRIOR_NY_LOW': 'Prior NY Session Low',
            'RECENT_SWING_HIGH': 'Recent 15m Swing High',
            'RECENT_SWING_LOW': 'Recent 15m Swing Low',
        }
        pool_label = pool_labels.get(swept_pool_type.upper(), swept_pool_type.replace('_', ' ').title())

        if direction in ('BULL', 'BUY'):
            sweep_what = f"sweep of a prior LOW ({pool_label} at {format_price(pool_level)})"
            close_requirement = "candle wick poked BELOW the level then closed BACK ABOVE it"
            htf_check = "4H EMA50 is sloping UP (bullish) — confirms we are buying the dip"
            reject_if = "the 4H EMA50 is clearly bearish, or price closed BELOW the swept level (real breakdown, not a sweep)"
            candle_type = "hammer / bullish pin bar / bullish engulfing tail"
        else:
            sweep_what = f"sweep of a prior HIGH ({pool_label} at {format_price(pool_level)})"
            close_requirement = "candle wick poked ABOVE the level then closed BACK BELOW it"
            htf_check = "4H EMA50 is sloping DOWN (bearish) — confirms we are selling the spike"
            reject_if = "the 4H EMA50 is clearly bullish, or price closed ABOVE the swept level (real breakout, not a sweep)"
            candle_type = "shooting star / bearish pin bar / bearish engulfing tail"

        chart_instruction = ""
        if has_chart:
            chart_instruction = f"""
## CHART ANALYSIS (SWEEP-AND-REVERSE LENS)

The chart shows two timeframes:
- **4H**: Shows 50 EMA (purple) — THE critical filter. Determines which direction sweeps are valid.
- **15m**: Shows the sweep candle, the pool level (prior H/L), entry/SL/TP markers.

**What a valid sweep looks like on the 15m chart:**
- A candle whose WICK extends clearly beyond the pool level ({format_price(pool_level)})
- The BODY closes back on the correct side of that level
- The wick is prominent — at least half the total candle range
- The swept level ({pool_label}) should be a visible prior high or low

**Positive signs (APPROVE):**
- {candle_type} at or just beyond the pool level
- {htf_check}
- The pool level was clearly tested multiple times before (accumulation of stops)
- No prior consecutive closes beyond the level (single-candle puncture, not sustained pressure)

**Negative signs (REJECT):**
- {reject_if}
- The "wick" is barely visible — excess is within normal candle noise
- Multiple consecutive candle closes already through the level (trend break, not sweep)
- The entry is firing into obvious momentum (wide-range expansion candles)
"""

        mo_note = (
            "\n⚠️ **MONITOR-ONLY / TEST MODE:** This is a demo-environment test signal. Score honestly; evaluate as if capital is at risk.\n"
            if monitor_only else ""
        )

        return f"""You are a SENIOR FOREX TECHNICAL ANALYST evaluating a Liquidity Sweep + Rejection Wick signal.

**STRATEGY THESIS — READ FIRST (SMC_MOMENTUM v1)**
SMC_MOMENTUM fires when price makes a brief stop-hunt beyond a prior high or low (the "sweep"),
then immediately reverses back inside. This creates a rejection wick. Entry is WITH the 4H EMA50 trend:
- In a 4H BULLISH trend: take LONG entries on downside sweeps of prior lows
- In a 4H BEARISH trend: take SHORT entries on upside sweeps of prior highs

This is NOT a breakout trade and NOT a mean-reversion fade of the HTF trend. It is a WITH-TREND
reversal entry triggered by institutional stop-hunting behavior.

Critical rules for your analysis:
- ❌ Do NOT reject because MACD is against the entry — SMC_MOMENTUM does not use MACD
- ❌ Do NOT reject because RSI is "extended" — the sweep fires at extremes by design
- ❌ Do NOT reject because there is no volume spike — tick volume on FX 15m is unreliable
- ✅ DO reject if the 4H EMA50 clearly opposes the signal direction
- ✅ DO reject if price closed THROUGH the swept level (real breakout, not a sweep)
- ✅ DO reject if the "wick" is barely visible (excess within noise, not a genuine stop-hunt)
{mo_note}
═══════════════════════════════════════════════════════════════
✅ THE SETUP IS THE TRIGGER — DO NOT REJECT FOR THESE
═══════════════════════════════════════════════════════════════
These ARE the sweep-and-reverse setup conditions — not concerns:
- RSI at extreme levels (sweeps happen at distal price by definition)
- MACD opposing entry direction (sweep reversals are counter-MACD)
- Price "extended" from the 4H EMA (again — sweeps happen at distal price)
- No volume spike (FX tick volume is not reliable for this strategy)
- The entry appearing counter to a lower-TF micro-trend (it is a reversal entry)

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: SMC_MOMENTUM v1
• Trigger: LIQUIDITY_SWEEP → REJECTION_WICK
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {format_price(entry_price)}
• Stop Loss: {fmt(risk_pips, 1)} pips  (placed beyond swept pool level + buffer)
• Take Profit: {fmt(reward_pips, 1)} pips  (ATR-based target)
• Risk:Reward Ratio: {rr_ratio:.2f}:1

═══════════════════════════════════════════════════════════════
🔬 SWEEP DATA (computed by strategy)
═══════════════════════════════════════════════════════════════
• Pool Type: {pool_label}
• Pool Level: {format_price(pool_level)}  (the prior H/L that was swept)
• Excess Beyond Level: {fmt(excess_pips, 1)} pips  (how far wick poked past the level)
• Rejection Wick %: {fmt(wick_pct * 100, 0)}% of candle range  (minimum 50% required)
• Double Sweep: {'✅ Yes (+confidence)' if is_double_sweep else '❌ No'}
• HTF Bias (4H EMA50): {htf_bias}  ← must align with {direction} entry
{chart_instruction}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

**CRITICAL:** Begin with `SCORE:`. No preamble.

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | SM_HTF_OPPOSE | SM_REAL_BREAKOUT | SM_NO_WICK | SM_SUSTAINED_BREAK]
REASON: [≤40 words. Focus on: 4H EMA50 alignment, whether sweep candle shows genuine wick rejection, and pool level quality. Do NOT comment on MACD, RSI, or volume.]

**SCORING RUBRIC (4 anchors):**
- 9–10 EXCEPTIONAL: 4H trend clearly aligned, prominent wick rejection (>60% of range), meaningful pool level with multiple prior touches, clean close back inside
- 7–8  STRONG: 4H trend aligned but choppy, or wick present but modest (50-60%), pool level visible but less tested
- 5–6  ACCEPTABLE: 4H trend aligned, wick present, pool level identifiable — positive expectancy midband
- 3–4  MARGINAL: 4H trend unclear/ranging, wick marginal, or pool level looks like noise — APPROVE unless rejection criterion clearly met
- 1–2  REJECT: one of the rejection criteria below is clearly true

**REJECTION CRITERIA (any one = REJECT):**
- 4H EMA50 clearly slopes against the signal direction → SM_HTF_OPPOSE
- Price closed on the wrong side of the swept pool level (real breakout) → SM_REAL_BREAKOUT
- No visible wick — sweep candle has a normal body with no meaningful extension past the pool → SM_NO_WICK
- Multiple consecutive candle closes beyond the pool level before this signal → SM_SUSTAINED_BREAK

Be concise. Your four lines determine if real capital is risked on this mechanic."""

    except Exception as e:
        logger.error(f"Error building SMC_MOMENTUM prompt: {e}")
        return build_fallback_prompt(signal)
