"""Vision prompt for the MEAN_REVERSION strategy."""
import logging
from typing import Dict
from ._helpers import format_price, extract_pair, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """
    Vision-enabled prompt for the MEAN_REVERSION strategy.

    Mean-reversion is the OPPOSITE thesis to SMC_SIMPLE / FVG_RETEST:
        • SMC / FVG: trade WITH momentum, respect HTF EMA, use MACD as filter.
        • MEAN_REVERSION: FADE the extreme, expect reversion to BB middle,
          overextension IS the setup, counter-MACD is the setup.
    """
    try:
        epic = signal.get('epic', 'Unknown')
        pair = extract_pair(epic)
        direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
        confidence = signal.get('confidence_score', signal.get('confidence', 0))

        entry_price = signal.get('entry_price', signal.get('price', 0))
        risk_pips = signal.get('stop_loss_pips', signal.get('risk_pips', 0))
        reward_pips = signal.get('take_profit_pips', signal.get('reward_pips', 0))
        try:
            rr_ratio = float(reward_pips) / float(risk_pips) if float(risk_pips) > 0 else 0
        except Exception:
            rr_ratio = 0

        monitor_only = bool(signal.get('monitor_only', False))

        indicators = signal.get('strategy_indicators', {}) or {}
        adx_15m = indicators.get('adx', signal.get('adx'))
        adx_1h = indicators.get('adx_htf', signal.get('adx_htf'))
        rsi_val = indicators.get('rsi', signal.get('rsi'))
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        bb_mid = indicators.get('bb_mid')
        bb_mult = indicators.get('bb_mult')
        extremity = indicators.get('extremity')

        def fmt(v, prec=2, suffix=""):
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.{prec}f}{suffix}"
            except Exception:
                return str(v)

        if direction in ('BUY', 'BULL'):
            setup_side = "OVERSOLD"
            fade_what = "fade the drop"
            expected_move = "back up toward the BB middle"
            reversal_patterns_wanted = "bullish engulfing, bullish pin bar, hammer, doji rejection, double-bottom"
        else:
            setup_side = "OVERBOUGHT"
            fade_what = "fade the rally"
            expected_move = "back down toward the BB middle"
            reversal_patterns_wanted = "bearish engulfing, shooting star, inverted hammer, doji rejection, double-top"

        chart_instruction = ""
        if has_chart:
            chart_instruction = f"""
## CHART ANALYSIS (EXAMINE CAREFULLY — MEAN-REVERSION LENS)

The chart spans multiple timeframes. For mean-reversion you look for different things than trend-continuation:

**What to look for (POSITIVE):**
- Price has *just touched or poked through* the Bollinger band (not riding it for many candles — that's a trend)
- Decelerating candles approaching the extreme: smaller bodies, longer wicks, volume tapering
- {reversal_patterns_wanted} at or just past the band
- 1h chart: price is inside a range, NOT in a strong directional run (flat or flattening EMAs)
- Prior BB touches on the same side that reverted cleanly to mid-band (validates the regime)

**What to look for (NEGATIVE — REJECT if any):**
- Price is *still impulsing* past the band with a wide body — this is a breakout, not exhaustion
- 1h chart shows clear directional structure (stair-step HH/HL or LH/LL) — fading a macro trend
- Price rode the BB band for many consecutive candles — band-walking confirms trend, not range
- Gap or news candle caused the extreme — fading news is catching knives

**Lines on chart:**
- GREEN dashed: Entry price
- RED dashed: Stop loss (structural — just beyond the band / recent wick)
- BLUE dashed: Take profit (aimed at BB middle, roughly)
"""

        mo_note = (
            "\n⚠️ **MONITOR-ONLY MODE:** This strategy is still in data-collection phase on this pair. "
            "Your score still matters for the historical dataset, but no capital is at risk on this alert.\n"
            if monitor_only else ""
        )

        return f"""You are a SENIOR FOREX TECHNICAL ANALYST evaluating a MEAN-REVERSION signal.

**STRATEGY THESIS — READ BEFORE ANY ANALYSIS:**
MEAN_REVERSION fires when price has pushed to a statistical extreme (outside Bollinger
Band 20/2σ) with a matching RSI(14) extreme (≤ 30 or ≥ 70), in a low-ADX regime
(15m ADX ≤ {fmt(22, 0)}, 1h ADX ≤ {fmt(25, 0)}). The trade is to **{fade_what}** and
expect price to revert **{expected_move}**.

This is the OPPOSITE of a trend-following trade. Therefore:
- ❌ DO NOT reject because MACD is against the entry — mean-reversion is counter-MACD by construction
- ❌ DO NOT reject because price is on the "wrong" side of the HTF EMA — we are fading, not continuing
- ❌ DO NOT apply the "overextended momentum often continues" rule — overextension IS the setup here
- ✅ DO reject if price is STILL IMPULSING with wide-range candles and rising volume
- ✅ DO reject if the 1h chart shows a clear directional trend (band-walking, stair-step structure)
- ✅ DO reject if the extreme was caused by a news spike or gap
{mo_note}
═══════════════════════════════════════════════════════════════
✅ THE SETUP IS THE TRIGGER — DO NOT REJECT FOR THESE
═══════════════════════════════════════════════════════════════
These conditions ARE the mean-reversion setup — rejecting for them negates the strategy:
- MACD opposing entry direction (counter-MACD is the whole point)
- Price on the "wrong" side of the HTF EMA (fade-of-trend inside a range IS the thesis)
- RSI at extreme values ≤30 or ≥70 (RSI extreme IS the entry trigger)
- Low R:R ratio of ~1.5:1 (structural to this strategy — high WR compensates)
- "Trend indicators" pointing against the fade (ranging regime IS required)

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}  ({setup_side} fade)
• Strategy: MEAN_REVERSION
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {format_price(entry_price)}
• Stop Loss: {fmt(risk_pips, 1)} pips  (structural — just beyond the extreme)
• Take Profit: {fmt(reward_pips, 1)} pips (target: BB middle)
• Risk:Reward Ratio: {rr_ratio:.2f}:1
  ℹ️  MR R:R is structurally ~1.5:1 — do NOT penalize the low R:R.

═══════════════════════════════════════════════════════════════
🔬 STRATEGY-SPECIFIC DATA (populated by the strategy at signal time)
═══════════════════════════════════════════════════════════════
• 15m ADX: {fmt(adx_15m, 1)}  (hard ceiling: 22 — signal already passed)
• 1h  ADX: {fmt(adx_1h, 1)}   (hard ceiling: 25 — signal already passed)
• RSI(14) 15m: {fmt(rsi_val, 1)}  (threshold for {setup_side}: ≤ 30 BUY / ≥ 70 SELL)
• Bollinger Bands (20, {fmt(bb_mult, 1)}σ):
    upper {format_price(bb_upper)} / mid {format_price(bb_mid)} / lower {format_price(bb_lower)}
• Extremity score (0-1, how far past threshold): {fmt(extremity, 2)}
{chart_instruction}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

**CRITICAL:** Begin with `SCORE:`. No preamble.

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | MR_STILL_IMPULSING | MR_BAND_WALKING | MR_TREND_CLEAR | MR_NEWS_SPIKE | MR_NO_EXHAUSTION]
REASON: [≤40 words. Focus on exhaustion quality at the band and 1h trend absence. Do NOT mention MACD or HTF EMA side.]

**SCORING RUBRIC (4 anchors):**
- 9–10 EXCEPTIONAL: Clean exhaustion (smaller bodies + long wicks into band), reversal pattern visible, 1h flat/range, prior reverts validate the regime
- 7–8  STRONG: Band touch with RSI extreme, some exhaustion signs visible, 1h slope mild
- 5–6  ACCEPTABLE: Band touch confirmed, RSI extreme met, no clear band-walking — positive expectancy
- 3–4  MARGINAL: Price at band but still impulsing, or 1h showing directional hints — APPROVE unless rejection criterion is clearly met
- 1–2  REJECT: one of the rejection criteria below is clearly true

**REJECTION CRITERIA (any one = REJECT):**
- Current candle is an impulsive wide-range expansion in the signal direction (catching a knife) → MR_STILL_IMPULSING
- 1h chart shows a clear, fresh directional structure (HH/HL for uptrend or LH/LL for downtrend) opposing the fade → MR_TREND_CLEAR
- Price has ridden the BB band on the same side for ≥ 5 consecutive candles (band-walking confirms trend) → MR_BAND_WALKING
- News / gap spike is the visible cause of the extreme (vertical candle, gap bar) → MR_NEWS_SPIKE
- No reversal structure whatsoever: current candle made the extreme with wide body and no wick → MR_NO_EXHAUSTION

Be concise but thorough. You are evaluating an *exhaustion fade*, not a trend entry."""

    except Exception as e:
        logger.error(f"Error building MEAN_REVERSION prompt: {e}")
        return build_fallback_prompt(signal)
