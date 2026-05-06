"""Vision prompt for the RANGE_FADE strategy."""
import logging
from typing import Dict
from ._helpers import format_price, extract_pair, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """Vision prompt for EURUSD range-fade setups (HTF-aligned BB fade)."""
    try:
        epic = signal.get('epic', 'Unknown')
        pair = extract_pair(epic)
        strategy = signal.get('strategy', 'RANGE_FADE')
        direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
        confidence = signal.get('confidence_score', signal.get('confidence', 0))
        entry_price = signal.get('entry_price', signal.get('price', 0))
        risk_pips = signal.get('stop_loss_pips', signal.get('risk_pips', 0))
        reward_pips = signal.get('take_profit_pips', signal.get('reward_pips', 0))
        monitor_only = bool(signal.get('monitor_only', False))

        try:
            rr_ratio = float(reward_pips) / float(risk_pips) if float(risk_pips) > 0 else 0
        except Exception:
            rr_ratio = 0

        indicators = signal.get('strategy_indicators', {}) or {}
        rsi_val = indicators.get('rsi', signal.get('rsi'))
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        bb_mid = indicators.get('bb_mid')
        band_width = indicators.get('band_width_pips')
        htf_bias = indicators.get('htf_bias', 'unknown')
        range_high = indicators.get('range_high')
        range_low = indicators.get('range_low')
        dist_low = indicators.get('distance_to_low_pips')
        dist_high = indicators.get('distance_to_high_pips')

        def fmt(v, prec=2, suffix=""):
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.{prec}f}{suffix}"
            except Exception:
                return str(v)

        if direction in ('BUY', 'BULL'):
            setup_side = "lower Bollinger Band fade (HTF-aligned long)"
            desired_confirmation = "stalling downside momentum at/near the lower band while the 1h bias remains bullish"
            reject_if = "the 1h has just flipped bearish or price is expanding lower through the band with strong continuation candles"
        else:
            setup_side = "upper Bollinger Band fade (HTF-aligned short)"
            desired_confirmation = "stalling upside momentum at/near the upper band while the 1h bias remains bearish"
            reject_if = "the 1h has just flipped bullish or price is expanding higher through the band with strong continuation candles"

        chart_instruction = ""
        if has_chart:
            chart_instruction = f"""
## CHART ANALYSIS (HTF-ALIGNED BB FADE LENS)

This is a **controlled mean-reversion** trade. Price has touched a Bollinger Band edge while the 1h HTF bias is {htf_bias}. We are fading the 5m stretch IN THE DIRECTION of the 1h bias, not calling a reversal of the 1h trend.

Positive signs:
- Wick rejection or stall at the outer band rather than a sustained band-walk
- 1h bias is clearly {htf_bias} and the entry direction aligns with it
- Current 5m bar is not an expansion candle — body size looks typical
- No recent breakout structure on the 1h that would invalidate the fade

Reject if:
- 1h bias has visibly flipped or is in transition (HTF EMA being crossed right now)
- Multiple consecutive expansion candles pushing through the band (band-walking, not fading)
- News-driven spike or gap — this strategy has no macro filter
- Entry against an obvious intraday breakout on the 1h
"""

        mo_note = (
            "\n⚠️ **MONITOR-ONLY MODE:** This strategy is in live-signal validation. Score honestly; no capital is at risk.\n"
            if monitor_only else ""
        )

        return f"""You are a SENIOR FOREX TECHNICAL ANALYST evaluating a Bollinger Band fade with higher-timeframe alignment.

**STRATEGY THESIS — READ FIRST (RANGE_FADE v0.4.0 lean config)**
RANGE_FADE is a controlled mean-reversion trade on {pair} 5m. It triggers when:
1. Price touches a Bollinger Band edge (upper for shorts, lower for longs)
2. RSI is past 40 (for longs) or 60 (for shorts) — moderate directional bias, NOT an extreme
3. The 1h HTF bias (EMA50 + slope) aligns with the fade direction — strict, no neutral
4. Current hour is within London/NY window (06–18 UTC)

It is NOT a strict range-extreme fade and NOT a momentum continuation trade. It fades a 5m stretch IN THE DIRECTION of the 1h trend.

Important:
- ❌ Do NOT reject because RSI isn't at 20/80 — the lean config uses 40/60 moderate thresholds
- ❌ Do NOT demand price be within N pips of a horizontal range boundary — the proximity gate is disabled
- ❌ Do NOT apply a generic SMC "momentum continuation" or "wait for structure break" lens
- ✅ DO reject if the 1h bias is actively flipping against the entry
- ✅ DO reject if the current 5m bar is an expansion candle breaking clean out of the band
- ✅ DO approve if you see a normal pullback to the band in an HTF-aligned direction
{mo_note}
═══════════════════════════════════════════════════════════════
✅ THE SETUP IS THE TRIGGER — DO NOT REJECT FOR THESE
═══════════════════════════════════════════════════════════════
These ARE the RANGE_FADE setup conditions — not red flags:
- The 15m EMA9/21 crossing against the 1h bias: this IS the 5m stretch being faded
- 60+ bars of directional movement before the signal: this IS the exhaustion being faded
- RSI showing 40-60 moderate readings (not 20/80): this is the designed threshold
- Price at a recent high or low of the session: this is WHERE fades happen
- Momentum building toward the band: this is the TRIGGER condition

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: {strategy}
• Setup Type: {setup_side}
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {format_price(entry_price)}
• Stop Loss: {fmt(risk_pips, 1)} pips
• Take Profit: {fmt(reward_pips, 1)} pips
• Risk:Reward Ratio: {rr_ratio:.2f}:1

═══════════════════════════════════════════════════════════════
🔬 STRATEGY DATA
═══════════════════════════════════════════════════════════════
• RSI(14): {fmt(rsi_val, 1)}   (thresholds: 40 for longs / 60 for shorts — moderate, not extreme)
• Bollinger Bands:
    upper {format_price(bb_upper)} / mid {format_price(bb_mid)} / lower {format_price(bb_lower)}
• Band width: {fmt(band_width, 1)} pips   (informational — band-width gate is disabled)
• HTF bias (1h EMA50 + slope): {htf_bias}   ← **primary confluence, must align**
• Prior-range context: high {format_price(range_high)} / low {format_price(range_low)}
• Distance to range low: {fmt(dist_low, 1)} pips (context only; not a required proximity)
• Distance to range high: {fmt(dist_high, 1)} pips (context only; not a required proximity)
{chart_instruction}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

**CRITICAL:** Begin with `SCORE:`. No preamble.

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | RF_HTF_FLIP | RF_BAND_WALKING | RF_EXPANSION | RF_NEWS]
REASON: [≤40 words. Focus on: (a) whether 1h HTF bias truly aligns, (b) whether the 5m bar looks like a normal pullback vs. expansion through the band, (c) whether any regime change on 1h invalidates the fade.]

**SCORING RUBRIC (4 anchors):**
- 9–10 EXCEPTIONAL: 1h bias clearly {htf_bias}, band touch is a wick/stall (not expansion), no recent breakout structure
- 7–8  STRONG: 1h bias aligned, band touch with some uncertainty in current bar behavior
- 5–6  ACCEPTABLE: 1h aligned, band touched, current bar neutral — positive expectancy midband
- 3–4  MARGINAL: 1h bias uncertain or mild slope — APPROVE unless rejection criterion is clearly met
- 1–2  REJECT: one of the rejection criteria below is clearly true

**REJECTION CRITERIA (any one = REJECT):**
- 1h bias has visibly flipped against the entry direction (EMA crossing now) → RF_HTF_FLIP
- Multiple consecutive expansion candles pushing through the band (band-walking) → RF_BAND_WALKING
- Current 5m bar is an expansion candle breaking clean out of the band → RF_EXPANSION
- News-driven spike or gap is the visible cause → RF_NEWS

Reject if: {reject_if}.
Approve only if you see {desired_confirmation}."""

    except Exception as e:
        logger.error(f"Error building RANGE_FADE prompt: {e}")
        return build_fallback_prompt(signal)
