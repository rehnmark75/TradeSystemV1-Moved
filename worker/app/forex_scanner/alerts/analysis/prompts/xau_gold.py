"""Vision prompt for the XAU_GOLD strategy (Gold 3-tier BOS pullback + event playbooks)."""
import logging
from typing import Dict
from ._helpers import format_price, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """
    Vision prompt for XAU_GOLD.

    Two signal paths share the same prompt, distinguished by xau_event_layer flag:

    Path A — Strict BOS Pullback (xau_event_layer absent/False):
        4H EMA50/200 bias → 1H BOS with displacement → 15m fib-zone pullback (optional FVG).

    Path B — Event Playbook (xau_event_layer=True):
        5m scanner: range_break_2h, sweep_reversal_2h, ema21_pullback_5m, displacement_5m.
        HTF alignment is scored (bonus) not required (hard gate).

    Regime gate: TRENDING (ADX ≥ 25) required by strategy; RANGING/EXPANSION block before
    this prompt is reached. Neutral (ADX 20-25) can slip through — treat as borderline.
    Session: London 07-10 + NY 13-20 primary; Asian continuation acceptable; 21-22 blocked.
    """
    try:
        direction = signal.get('signal_type', signal.get('signal', 'Unknown')).upper()
        confidence = signal.get('confidence_score', signal.get('confidence', 0))
        entry_price = signal.get('entry_price', signal.get('price', 0))
        sl_pips = signal.get('stop_loss_pips', signal.get('risk_pips', 0))
        tp_pips = signal.get('take_profit_pips', signal.get('reward_pips', 0))
        monitor_only = bool(signal.get('monitor_only', False))

        market_regime = signal.get('market_regime', 'unknown')
        adx_value = signal.get('adx_value', 0)
        volatility_state = signal.get('volatility_state', 'normal')

        indicators = signal.get('strategy_indicators', {}) or {}
        is_event_layer = bool(indicators.get('xau_event_layer', False))

        def fmt(v, prec=1, suffix=""):
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.{prec}f}{suffix}"
            except Exception:
                return str(v)

        def yn(v):
            if v is None:
                return "n/a"
            return "YES" if bool(v) else "NO"

        # ── Direction-specific language ──────────────────────────────────────
        if direction in ('BUY', 'BULL'):
            htf_expect = "4H EMA50 sloping UP, price ABOVE EMA200, Higher Highs / Higher Lows structure"
            bos_expect = "1H swing HIGH broken upward with strong displacement body (bullish BOS)"
            pullback_expect = "15m price pulled back INTO the BOS leg (fib 0.38–0.62), bounced off demand OB or FVG"
            reject_counter = "4H EMA50 is clearly pointing DOWN or price is below EMA200"
        else:
            htf_expect = "4H EMA50 sloping DOWN, price BELOW EMA200, Lower Highs / Lower Lows structure"
            bos_expect = "1H swing LOW broken downward with strong displacement body (bearish BOS)"
            pullback_expect = "15m price pulled back INTO the BOS leg (fib 0.38–0.62), rejected from supply OB or FVG"
            reject_counter = "4H EMA50 is clearly pointing UP or price is above EMA200"

        # ── Path-specific data blocks ────────────────────────────────────────
        if is_event_layer:
            playbook = indicators.get('xau_playbook', 'unknown')
            htf_bias = indicators.get('htf_bias', 'unknown')
            htf_aligned = indicators.get('htf_aligned')
            trigger_level = indicators.get('trigger_level')
            event_strength = indicators.get('event_strength')
            body_atr = indicators.get('body_atr')
            rsi_14 = indicators.get('rsi_14')
            recent_range_pips = indicators.get('recent_range_pips')
            ema21_distance_pips = indicators.get('ema21_distance_pips')

            playbook_labels = {
                'range_break_2h':     'Range Breakout (2H)',
                'sweep_reversal_2h':  'Liquidity Sweep Reversal (2H)',
                'ema21_pullback_5m':  'EMA-21 Pullback (5m)',
                'displacement_5m':    'Displacement Break (5m)',
            }
            playbook_label = playbook_labels.get(playbook, playbook.replace('_', ' ').title())

            playbook_descriptions = {
                'range_break_2h': (
                    "Price broke above/below a 2H consolidation range with a strong-body 5m candle. "
                    "Expects momentum continuation. HTF alignment is a bonus, not a hard requirement."
                ),
                'sweep_reversal_2h': (
                    "Price swept above/below a 2H range extreme (stop-hunt), then closed back inside "
                    "with ≥45% rejection wick. A fading setup — HTF counter-trend is acceptable here."
                ),
                'ema21_pullback_5m': (
                    "5m EMA-21/50/200 fully stacked in trend direction. Price pulled back to EMA-21 "
                    "and bounced. HTF alignment strongly preferred."
                ),
                'displacement_5m': (
                    "Large-body 5m displacement candle breaks a recent micro swing while 5m EMAs "
                    "are fully stacked. Momentum entry. HTF alignment strongly preferred."
                ),
            }
            playbook_desc = playbook_descriptions.get(playbook, "Gold event-playbook entry.")

            data_block = f"""
═══════════════════════════════════════════════════════════════
🎯 EVENT PLAYBOOK SIGNAL DATA
═══════════════════════════════════════════════════════════════
• Playbook Setup:    {playbook_label}
• Setup Description: {playbook_desc}
• HTF Bias (4H):     {htf_bias}  (htf_aligned={yn(htf_aligned)})
• Trigger Level:     {fmt(trigger_level, 2)} (price level that was broken/swept)
• Event Strength:    {fmt(event_strength, 3)} (body_atr ratio or wick rejection %)
• Body/ATR Ratio:    {fmt(body_atr, 2)}
• RSI-14 at signal:  {fmt(rsi_14, 1)} (neutral 40–60, extremes risky)
• Recent Range:      {fmt(recent_range_pips, 0)} pips (2H prior range size)
• EMA-21 Distance:   {fmt(ema21_distance_pips, 0)} pips from close to 5m EMA-21"""

            chart_section = ""
            if has_chart:
                chart_section = f"""

## CHART ANALYSIS ({playbook_label} LENS)

The chart shows 3 timeframes: **4H** (EMA50/200 bias), **1H** (BOS/swing structure), **15m** (entry candle and trigger level).

**What triggered this signal:**
{playbook_desc}

**What to look for in the chart:**

✅ APPROVE if:
"""
                if playbook == 'range_break_2h':
                    chart_section += """- A clear consolidation box is visible before the breakout candle
- The breakout candle has a strong body, not just a wick spike through the level
- No major resistance (for BUY) or support (for SELL) immediately overhead/below
- ADX is visibly rising or already elevated (trending context)
"""
                elif playbook == 'sweep_reversal_2h':
                    chart_section += """- The sweep candle has a prominent wick that pierced the range extreme then closed back inside
- The rejection wick is clearly visible — price was rejected from beyond the level
- Surrounding context shows a prior range / consolidation (not a trending impulse)
- The current candle or next candle confirms the reversal direction
"""
                elif playbook == 'ema21_pullback_5m':
                    chart_section += """- 5m EMA lines are visibly stacked (EMA21 > EMA50 > EMA200 for BUY; reversed for SELL)
- The pullback touched or came close to EMA-21 before bouncing
- The bounce candle closes clearly in the signal direction
- No major S/R level directly opposing the entry
"""
                elif playbook == 'displacement_5m':
                    chart_section += """- 5m EMAs fully stacked in signal direction
- A large-body candle (noticeably bigger than neighbours) breaks recent micro swing
- The candle body (not just wick) closes through the trigger level
- No major S/R immediately ahead in the signal direction
"""
                chart_section += f"""
❌ REJECT if:
- For {playbook_label}: the triggering level was not clearly defined or the move through it looks weak (wick only, no body)
- Price is visibly approaching major structural resistance (BUY) or support (SELL) within the next 20–30 pips
- The chart looks chaotic / whippy with no clear structure — ADX likely weak
- HTF (4H) structure clearly opposes AND this is NOT a sweep_reversal_2h setup
"""

        else:
            # Strict BOS Pullback path
            bias = indicators.get('bias', 'unknown')
            bos_from = indicators.get('bos_from_price')
            bos_to = indicators.get('bos_to_price')
            bos_displacement_atr = indicators.get('bos_displacement_atr')
            fib_depth = indicators.get('fib_depth')
            entry_age_bars = indicators.get('entry_age_bars')
            fvg_confluence = indicators.get('fvg_confluence')
            rsi_14 = indicators.get('rsi_14')
            atr_pct = indicators.get('atr_pct')

            data_block = f"""
═══════════════════════════════════════════════════════════════
🔬 BOS PULLBACK SIGNAL DATA (4H/1H/15m)
═══════════════════════════════════════════════════════════════
• 4H Bias:           {bias}  (expected: aligned with {direction})
• BOS Level:         {fmt(bos_from, 2)} → {fmt(bos_to, 2)}  (1H swing break)
• BOS Displacement:  {fmt(bos_displacement_atr, 2)}× ATR  (>1.0 = strong momentum break)
• Fib Depth:         {fmt(fib_depth, 2)}  (0.38–0.62 = ideal pullback zone)
• Entry Age:         {fmt(entry_age_bars, 0)} bars on 15m since BOS
• FVG Confluence:    {yn(fvg_confluence)}  (YES = active FVG/OB near entry — stronger)
• RSI-14 at signal:  {fmt(rsi_14, 1)}  (neutral 40–60 ideal; extremes risky)
• ATR Percentile:    {fmt(atr_pct, 1)}%  (>80% = expansion / news zone — risky)"""

            chart_section = ""
            if has_chart:
                chart_section = f"""

## CHART ANALYSIS (BOS PULLBACK LENS)

The chart shows 3 timeframes: **4H** (EMA50/200 HTF bias), **1H** (BOS structure and swing break), **15m** (pullback entry).

**What triggered this signal:**
A {direction} 4H trend established via EMA50/200 alignment. A 1H swing was broken ({bos_expect}) with displacement. Price then pulled back into the fib zone of that BOS leg on the 15m, and the strategy entered.

**What to look for in the chart:**

✅ APPROVE if:
- **4H**: {htf_expect}
- **1H**: {bos_expect}. The break candle has a clear strong body (displacement, not a slow grind)
- **15m**: {pullback_expect}
- The pullback candle that triggered entry shows a reversal signal (pin bar, engulfing, rejection wick)
- Price has NOT exceeded the BOS origin level (if it has, the structure is broken)

❌ REJECT if:
- {reject_counter} on the 4H — counter-trend entries are not this strategy's edge
- The 1H BOS looks weak: price crept through the swing level without a strong displacement body
- The pullback went DEEPER than the BOS origin (structure invalidated — full retrace)
- 15m entry candle shows no reversal: large wick in the wrong direction, or bearish close on a BUY
- Price is visibly at a major higher-timeframe S/R level in opposition (e.g. BUY into prior week high)
- RSI at extreme (>70 for BUY / <30 for SELL) without visible divergence
"""

        # ── Regime block ─────────────────────────────────────────────────────
        regime_note = ""
        if market_regime == "trending":
            regime_note = "TRENDING (ADX strong) — ideal conditions for this strategy"
        elif market_regime == "neutral":
            regime_note = "NEUTRAL (ADX borderline ~20-25) — marginal regime; apply extra scrutiny to entry quality"
        elif market_regime == "expansion":
            regime_note = "⚠️ EXPANSION (ATR spike) — news-driven conditions; strategy normally blocks this. Proceed with caution."
        elif market_regime == "ranging":
            regime_note = "⚠️ RANGING (low ADX) — strategy normally blocks this. Apply maximum scrutiny."
        else:
            regime_note = f"{market_regime} — assess from chart"

        mo_note = (
            "\n⚠️ **MONITOR-ONLY / TEST MODE:** This is a demo-environment test signal. "
            "Score honestly; evaluate as if capital is at risk.\n"
            if monitor_only else ""
        )

        signal_type_label = "Event Playbook" if is_event_layer else "BOS Pullback"

        return f"""You are a SENIOR GOLD MARKET ANALYST evaluating an XAU/USD signal.

**STRATEGY THESIS — READ FIRST (XAU_GOLD)**
XAU_GOLD is a 3-tier Smart Money Concepts strategy for Gold (XAU/USD):
- Tier 1 (4H): EMA50/200 slope + HH/HL or LH/LL sequence establishes directional bias
- Tier 2 (1H): Break of Structure (BOS) aligned with 4H bias, with displacement (strong momentum body)
- Tier 3 (15m): Pullback into the fib zone of the BOS leg (0.38–0.62), optionally with FVG/OB confluence

REGIME GATE: strategy requires TRENDING (ADX ≥ 25). RANGING and EXPANSION are blocked upstream.
SESSIONS: London 07-10 UTC + NY 13-20 UTC are primary windows. Asian is continuation-only. 21-22 blocked.

This signal is from the **{signal_type_label}** path.
{mo_note}
═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Instrument:     XAU/USD (Gold)
• Direction:      {direction}
• Strategy Path:  XAU_GOLD — {signal_type_label}
• Entry Price:    {format_price(entry_price)}
• Stop Loss:      {fmt(sl_pips, 0)} pips  (1 pip = $0.10)
• Take Profit:    {fmt(tp_pips, 0)} pips
• Confidence:     {confidence:.1%}

═══════════════════════════════════════════════════════════════
📈 MARKET REGIME
═══════════════════════════════════════════════════════════════
• Regime:         {regime_note}
• ADX Value:      {fmt(adx_value, 1)}  (≥25 = trending, <20 = ranging)
• Volatility:     {volatility_state}  (expansion = news zone, normal = standard)
{data_block}
{chart_section}
═══════════════════════════════════════════════════════════════
⚖️ KEY SCORING RULES FOR XAU_GOLD
═══════════════════════════════════════════════════════════════
ALWAYS score based on the strategy's actual logic. Do NOT penalise for:
- ❌ "Inverted R:R" — SL/TP sizing is fixed by config, ignore R:R in scoring
- ❌ HTF opposing for sweep_reversal_2h — counter-HTF fades are the setup's design
- ❌ RSI at 55-65 range — RSI neutrality gate is 40-60; mild overextension is a minor flag only
- ❌ Low tick volume — FX/gold tick volume is not reliable

HARD CAPS:
- Score ≤ 4 (REJECT) if 4H EMA structure clearly opposes the signal AND this is NOT a sweep_reversal_2h
- Score ≤ 4 (REJECT) if the BOS looks like a slow grind (bos_displacement_atr < 0.5, no visible body strength on 1H)
- Score ≤ 3 (REJECT) if the pullback overshot the BOS origin (structure fully retraced — signal invalid)
- Score ≤ 3 (REJECT) if regime is RANGING or EXPANSION (these should be blocked upstream — flag it)

═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

**CRITICAL:** Begin with `SCORE:`. No preamble.

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | IF_COUNTER_HTF | IF_WEAK_BOS | IF_OVERSHOT_PULLBACK | IF_REGIME_MISMATCH | IF_POOR_ENTRY_CANDLE]
REASON: [≤45 words. Address: (a) 4H bias alignment, (b) BOS quality or event trigger clarity, (c) entry candle / pullback depth, (d) any S/R in opposition. No pip math commentary.]

**SCORING RUBRIC:**
- 9–10 EXCEPTIONAL: 4H trend clear + strong displaced BOS + clean fib pullback with FVG/OB + entry candle shows reversal + trending regime
- 7–8  STRONG: Solid setup but one minor issue (RSI slightly off, fib shallow, FVG absent)
- 5–6  ACCEPTABLE: Setup valid but entry candle or S/R context is suboptimal — positive expectancy
- 3–4  MARGINAL: HTF mixed or BOS weak — REJECT unless event_layer sweep_reversal_2h (counter-HTF acceptable there)
- 1–2  REJECT: clear counter-HTF (non-sweep), blown pullback, weak BOS, or regime mismatch

Be concise. Your four lines determine if real capital is risked on XAU/USD."""

    except Exception as e:
        logger.error(f"Error building XAU_GOLD prompt: {e}")
        return build_fallback_prompt(signal)
