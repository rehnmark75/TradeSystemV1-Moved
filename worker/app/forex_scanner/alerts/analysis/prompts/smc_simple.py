"""Vision prompt for the SMC_SIMPLE strategy (Smart Money Concepts v2.3.0)."""
import logging
from datetime import datetime as _datetime
from typing import Dict
from ._helpers import format_price, extract_pair, build_fallback_prompt

logger = logging.getLogger(__name__)


def build_prompt(signal: Dict, has_chart: bool = True) -> str:
    """
    Vision-enabled prompt for SMC Simple v2.3.0 (3-tier: HTF bias / swing break / entry).

    P0 improvements applied:
    - "SETUP IS THE TRIGGER" block prevents phantom rejections
    - Closed R1-R4 rejection list (only these four reasons produce REJECT)
    - 4-anchor scoring rubric (9-10 / 7-8 / 5-6 / 3-4)
    - REASON_CODE as 4th mandatory response line for closed-loop calibration
    """
    try:
        epic = signal.get('epic', 'Unknown')
        pair = extract_pair(epic)
        direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
        confidence = signal.get('confidence_score', 0)
        strategy = signal.get('strategy', 'SMC_SIMPLE')

        entry_price = signal.get('entry_price', signal.get('price', 0))
        stop_loss = signal.get('stop_loss', 0)
        take_profit = signal.get('take_profit', 0)
        risk_pips = signal.get('risk_pips', 0)
        reward_pips = signal.get('reward_pips', 0)
        rr_ratio = signal.get('rr_ratio', 0)
        monitor_only = bool(signal.get('monitor_only', False))

        strategy_indicators = signal.get('strategy_indicators', {})

        # Tier 1 — EMA Bias
        tier1_ema = strategy_indicators.get('tier1_ema', {})
        ema_value = tier1_ema.get('ema_value', signal.get('ema_value', 0))
        ema_distance = tier1_ema.get('distance_pips', signal.get('ema_distance_pips', 0))
        ema_direction = tier1_ema.get('direction', direction)
        strategy_htf = tier1_ema.get('timeframe', '4h') or '4h'
        if 'scalp_mode' in signal:
            is_scalp_mode = bool(signal.get('scalp_mode'))
        else:
            is_scalp_mode = strategy_htf != '4h'

        # Tier 2 — Swing Break
        tier2_swing = strategy_indicators.get('tier2_swing', {})
        swing_level = tier2_swing.get('swing_level', signal.get('swing_level', 0))
        body_close_confirmed = tier2_swing.get('body_close_confirmed', True)
        volume_confirmed = tier2_swing.get('volume_confirmed', signal.get('volume_confirmed', False))
        strategy_trigger = tier2_swing.get('timeframe', '15m') or '15m'
        volume_ratio_val = signal.get('volume_ratio')
        if volume_ratio_val is not None:
            try:
                _vr = float(volume_ratio_val)
                volume_spike_display = (
                    f"✅ Yes ({_vr:.2f}x avg)" if volume_confirmed
                    else f"❌ No ({_vr:.2f}x avg, threshold 1.30x)"
                )
            except (TypeError, ValueError):
                volume_spike_display = '✅ Yes' if volume_confirmed else '❌ No'
        else:
            volume_spike_display = '✅ Yes' if volume_confirmed else '❌ No'

        # Tier 3 — Entry
        tier3_entry = strategy_indicators.get('tier3_entry', {})
        strategy_entry = tier3_entry.get('timeframe', '5m') or '5m'
        entry_type = tier3_entry.get('entry_type', signal.get('entry_type', 'PULLBACK'))
        pullback_depth = tier3_entry.get('pullback_depth', signal.get('pullback_depth', 0))
        fib_zone = tier3_entry.get('fib_zone', 'Unknown')
        in_optimal_zone = tier3_entry.get('in_optimal_zone', signal.get('in_optimal_zone', False))
        order_type = tier3_entry.get('order_type', 'market')

        # Risk management override from strategy_indicators
        risk_mgmt = strategy_indicators.get('risk_management', {})
        if risk_mgmt:
            stop_loss = risk_mgmt.get('stop_loss', stop_loss)
            take_profit = risk_mgmt.get('take_profit', take_profit)
            risk_pips = risk_mgmt.get('risk_pips', risk_pips)
            reward_pips = risk_mgmt.get('reward_pips', reward_pips)
            rr_ratio = risk_mgmt.get('rr_ratio', rr_ratio)

        # Fixed SL/TP database override
        fixed_sl_note = ""
        try:
            from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
            smc_config = get_smc_simple_config()
            if smc_config.fixed_sl_tp_override_enabled:
                fixed_sl = smc_config.get_pair_fixed_stop_loss(epic)
                fixed_tp = smc_config.get_pair_fixed_take_profit(epic)
                if fixed_sl and fixed_tp:
                    risk_pips = fixed_sl
                    reward_pips = fixed_tp
                    rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                    fixed_sl_note = f"\n⚠️ **FIXED SL/TP MODE ACTIVE**: All trades use SL={fixed_sl} pips, TP={fixed_tp} pips (R:R={rr_ratio:.2f}:1) regardless of strategy calculation."
        except Exception:
            pass

        # Market intelligence
        market_intel = signal.get('market_intelligence', {})
        regime_analysis = market_intel.get('regime_analysis', {})
        market_regime = regime_analysis.get('dominant_regime', 'unknown')
        regime_confidence = regime_analysis.get('confidence', 0)
        current_session = market_intel.get('session_analysis', {}).get('current_session', 'unknown')
        volatility_level = market_intel.get('volatility_level', '')
        mi_confidence_modifier = signal.get('market_intelligence_confidence_modifier')

        # Technical indicators
        rsi_value = signal.get('rsi')
        rsi_zone = signal.get('rsi_zone', '')
        adx_value = signal.get('adx_value')
        adx_trend_strength = signal.get('adx_trend_strength', '')
        mtf_confluence = signal.get('mtf_confluence_score')
        all_tfs_aligned = signal.get('all_timeframes_aligned')
        atr_percentile = signal.get('atr_percentile')
        entry_quality = signal.get('entry_quality_score')

        # LPF data
        lpf_penalty = signal.get('lpf_penalty')
        lpf_would_block = signal.get('lpf_would_block', False)
        lpf_rules = signal.get('lpf_triggered_rules', [])

        # Day of week
        signal_ts = signal.get('timestamp') or signal.get('created_at')
        day_of_week = ''
        if signal_ts:
            try:
                if isinstance(signal_ts, str):
                    signal_ts = _datetime.fromisoformat(signal_ts)
                day_of_week = signal_ts.strftime('%A')
            except Exception:
                pass

        opposite_swing = signal.get('opposite_swing', 0)

        dataframe_analysis = strategy_indicators.get('dataframe_analysis', {})
        sr_data = dataframe_analysis.get('sr_data', {})
        ema_data = dataframe_analysis.get('ema_data', {})
        other_indicators = dataframe_analysis.get('other_indicators', {})
        confidence_breakdown = strategy_indicators.get('confidence_breakdown', {})

        try:
            _risk = float(risk_pips) if risk_pips else 0
        except (TypeError, ValueError):
            _risk = 0
        sr_buffer_pips = min(int(round(max(0.6 * _risk, 5))) if _risk > 0 else 10, 10)

        primary_htf_label = strategy_htf.upper() if is_scalp_mode else '4H'

        # --- Chart instruction ---
        chart_instruction = ""
        if has_chart:
            momentum_note = ""
            if entry_type == 'MOMENTUM':
                momentum_note = """
**⚡ MOMENTUM ENTRY NOTE:**
This is a MOMENTUM continuation trade (price beyond swing break).
- Entry is AFTER the swing break, riding momentum
- Higher risk but captures strong directional moves
- Look for: Clean breakout, no immediate reversal signs, volume confirmation
"""
            if is_scalp_mode:
                htf_upper = strategy_htf.upper()
                trigger_upper = strategy_trigger.upper()
                entry_upper = strategy_entry.upper()
                tf_display_section = f"""**Timeframes Displayed (Scalp Mode):**
- 4H timeframe: Macro trend context — 50 EMA bias (purple line)
- {htf_upper} timeframe: Strategy HTF bias — EMA used for trade direction decisions
- {trigger_upper} timeframe: PRIMARY ANALYSIS — swing break, EMAs 9/21, S/R levels, entry/SL/TP
- {entry_upper} timeframe: Entry precision — entry zone, Fibonacci levels, entry type annotation"""
                primary_markers_label = f"{trigger_upper} chart (PRIMARY)"
                entry_markers_label = f"{entry_upper} chart"
                checklist_items = f"""1. ✓ **{htf_upper} STRATEGY HTF (HIGHEST PRIORITY):** This signal uses {htf_upper} for directional bias — not 4H. Judge trend alignment against the {htf_upper} EMA/structure. Use 4H ONLY as macro backdrop: conflicting 4H is a −1 penalty, not a rejection.
2. ✓ **RESISTANCE/SUPPORT PROXIMITY:** For BUY — is entry near a recent swing HIGH, resistance zone, or round number? For SELL — near a swing LOW, support zone, or round number? Within {sr_buffer_pips} pips = scoring penalty (−1 to −2). Entry AT a MAJOR multi-touch S/R AGAINST {htf_upper} trend = score ≤4. Entry near S/R WITH {htf_upper} trend = penalty only, NOT a cap.
3. ✓ **POSITION IN RANGE:** For BUY: entry should be near the BOTTOM of a local range. For SELL: near the TOP. Entry at the TOP of a recovery in a downtrend = buying into supply. REJECT.
4. ✓ Is price clearly respecting the {htf_upper} EMA trend direction?
5. ✓ Is the swing break on the {trigger_upper} chart clean and confirmed (full candle close)?
6. ✓ Are EMA 9/21 aligned with the trade direction on the {trigger_upper} chart?
7. ✓ Is entry clear of nearby S/R obstacles shown on the {trigger_upper} chart?
8. ✓ For PULLBACK: Is entry within or near the optimal Fibonacci zone ({entry_upper})?
9. ✓ For MOMENTUM: Is breakout clean with strong directional candles?
10. ✓ Is stop loss placement below a valid structure low (for longs)?
11. ✓ Does the price action show clean trend structure?
12. ✓ Are there any concerning patterns (engulfing candles, dojis at entry)?
13. ✓ Does the entry type box ({entry_upper}) show favorable conditions?"""
            else:
                tf_display_section = """**Timeframes Displayed:**
- 4H timeframe: Shows 50 EMA trend bias (purple line)
- 15m timeframe: PRIMARY ANALYSIS - Shows swing break, EMAs 9/21, S/R levels, entry/SL/TP
- 5m timeframe: Shows entry zone with Fibonacci levels and entry type annotation"""
                primary_markers_label = "15m chart - PRIMARY"
                entry_markers_label = "5m chart"
                checklist_items = f"""1. ✓ **4H TREND STRUCTURE (HIGHEST PRIORITY):** Look at the 4H chart candles — is the trend making Higher Highs/Higher Lows (bullish) or Lower Highs/Lower Lows (bearish)? A BUY in a 4H downtrend (LH/LL) or SELL in a 4H uptrend (HH/HL) is COUNTER-TREND and must score ≤4.
2. ✓ **RESISTANCE/SUPPORT PROXIMITY:** For BUY — is entry near a recent swing HIGH, resistance zone, or round number? For SELL — near a recent swing LOW, support zone, or round number? Within {sr_buffer_pips} pips = scoring penalty (−1 to −2). Entry AT a MAJOR multi-touch S/R AGAINST the 4H trend = score ≤4. Entry near S/R WITH the trend = penalty only, NOT a cap.
3. ✓ **POSITION IN RANGE:** For BUY: entry near BOTTOM of local range (demand/support). For SELL: entry near TOP (supply/resistance). Entry at the TOP of a recovery in a downtrend = buying the worst location.
4. ✓ Is price clearly respecting the 4H EMA trend direction?
5. ✓ Is the swing break on 15m clean and confirmed (full candle close)?
6. ✓ Are EMA 9/21 aligned with the trade direction on 15m chart?
7. ✓ Is entry clear of nearby S/R obstacles shown on 15m?
8. ✓ For PULLBACK: Is entry within or near the optimal Fibonacci zone (5m)?
9. ✓ For MOMENTUM: Is breakout clean with strong directional candles?
10. ✓ Is stop loss placement below a valid structure low (for longs)?
11. ✓ Does the price action show clean trend structure?
12. ✓ Are there any concerning patterns (engulfing candles, dojis at entry)?
13. ✓ Does the entry type box (5m) show favorable conditions?"""

            macro_structure = signal.get('_macro_structure') or {}
            macro_auth_block = ""
            if macro_structure and macro_structure.get('trend_label', 'UNKNOWN') != 'UNKNOWN':
                seq_str = ' → '.join(macro_structure.get('swing_sequence', [])) or 'insufficient pivots'
                ema_str = macro_structure.get('ema50_relation', 'unknown')
                macro_auth_block = f"""
📊 [4H MACRO CONTEXT — PRE-COMPUTED FROM RAW PRICE DATA]
The following 4H structure values are accurate (computed, not inferred). Do not override these specific values with a visual impression of the 4H panel:
  • Trend: {macro_structure['trend_label']}
  • Recent swing sequence (oldest → newest): {seq_str}
  • Price vs 4H EMA50: {ema_str}

**How to use this data:** 4H is macro backdrop only — it is NOT the primary HTF for this signal. A 4H trend that opposes the signal is at most a −1 penalty when the primary HTF ({primary_htf_label}) aligns with the signal direction.
[END 4H MACRO CONTEXT]
"""

            chart_instruction = f"""
## CHART ANALYSIS (CRITICAL - EXAMINE CAREFULLY)
{macro_auth_block}
The attached chart shows multi-timeframe forex analysis with the following elements:

{tf_display_section}

**Key Visual Markers (on {primary_markers_label}):**
- GREEN dashed line: Entry price level
- RED dashed line: Stop loss level (below opposite swing)
- BLUE dashed line: Take profit target
- ORANGE line: EMA 9 (fast momentum)
- BLUE line: EMA 21 (trend confirmation)
- GREEN horizontal line: Support level with distance in pips
- RED horizontal line: Resistance level with distance in pips
- ORANGE horizontal lines: Swing high levels
- BLUE horizontal lines: Swing low levels

**Key Visual Markers (on {entry_markers_label}):**
- YELLOW shaded zone: Fibonacci optimal entry zone (38.2%-61.8%)
- Entry Type Box (top-right): Shows PULLBACK/MOMENTUM, depth %, zone status, volume ✓/✗
- LARGE ARROW MARKER (▲/▼ in circle): Points to the EXACT entry price level
- TRADE SUMMARY BOX (lower-left): Direction, Entry Type, SL pips, TP pips, R:R ratio, Confidence %
- GREEN dashed line with "ENTRY" label: Entry price level
- "NOW" marker: Shows where current price is relative to entry
{momentum_note}
**CRITICAL CHART ANALYSIS CHECKLIST (in priority order):**

{checklist_items}
"""

        # --- Entry type detail ---
        if entry_type == 'PULLBACK':
            zone_status = "✅ OPTIMAL (38.2%-61.8%)" if in_optimal_zone else "⚠️ OUTSIDE OPTIMAL"
            entry_type_detail = f"""
- Entry Style: PULLBACK (waiting for retracement)
- Pullback Depth: {pullback_depth:.1%} into swing range
- Fibonacci Zone: {zone_status}
- Risk Profile: Lower risk, better R:R potential"""
        else:
            entry_type_detail = f"""
- Entry Style: MOMENTUM (riding continuation)
- Position: {abs(pullback_depth):.1%} beyond swing break point
- Momentum Quality: {'Strong' if abs(pullback_depth) < 0.35 else 'Extended'}
- Risk Profile: Higher risk, captures strong moves"""

        # --- Confidence breakdown ---
        confidence_detail = ""
        if confidence_breakdown:
            confidence_detail = f"""
**CONFIDENCE SCORE BREAKDOWN ({confidence:.1%} total):**
- EMA Alignment: {confidence_breakdown.get('ema_alignment', 0)*100:.1f}%
- Volume Bonus: {confidence_breakdown.get('volume_bonus', 0)*100:.1f}%
- Pullback Quality: {confidence_breakdown.get('pullback_quality', 0)*100:.1f}%
- R:R Quality: {confidence_breakdown.get('rr_quality', 0)*100:.1f}%
- Fib Accuracy: {confidence_breakdown.get('fib_accuracy', 0)*100:.1f}%
"""

        # --- S/R context ---
        sr_context = ""
        if sr_data:
            nearest_support = sr_data.get('nearest_support')
            nearest_resistance = sr_data.get('nearest_resistance')
            dist_support = sr_data.get('distance_to_support_pips') or 0
            dist_resistance = sr_data.get('distance_to_resistance_pips') or 0
            support_str = f"{format_price(nearest_support)} ({dist_support:.1f} pips below)" if nearest_support else "N/A"
            resistance_str = f"{format_price(nearest_resistance)} ({dist_resistance:.1f} pips above)" if nearest_resistance else "N/A"

            if direction == 'BULL' and nearest_resistance and dist_resistance < reward_pips:
                path_note = '⚠️ Resistance in way'
            elif direction == 'BEAR' and nearest_support and dist_support < reward_pips:
                path_note = '⚠️ Support in way'
            else:
                path_note = '✅ Clear path'

            flip_context = ""
            flip_data = sr_data.get('flip_analysis', {})
            if flip_data:
                flip_lines = []
                for key, info in flip_data.items():
                    if key == 'warning':
                        continue
                    if isinstance(info, dict) and 'original_type' in info:
                        orig = info.get('original_type', 'unknown')
                        dist = info.get('distance_pips', 0)
                        recent = info.get('is_recent_flip', False)
                        strength = info.get('strength', 0)
                        price_str = key.rsplit('_', 1)[-1] if '_' in key else '?'
                        recency = "RECENT" if recent else "older"
                        flip_lines.append(
                            f"  - {price_str}: former {orig} → now flipped ({recency}, strength {strength:.2f}, {dist:.1f} pips away)"
                        )
                if flip_lines:
                    flip_context = "\n- **Level Flips (broken S/R):**\n" + "\n".join(flip_lines) + "\n  ⚠️ Flipped levels indicate a structural break — former resistance becomes support. Do NOT reject solely because price is near a level that has already been broken."

            sr_context = f"""
**SUPPORT/RESISTANCE CONTEXT:**
- Nearest Support: {support_str}
- Nearest Resistance: {resistance_str}
- Path to Target: {path_note}{flip_context}
"""

        # --- EMA stack context ---
        ema_stack_context = ""
        if ema_data:
            ema_9_val = ema_data.get('ema_9', 0)
            ema_21_val = ema_data.get('ema_21', 0)
            ema_50_val = ema_data.get('ema_50', 0)
            if ema_9_val and ema_21_val:
                ema_alignment = "Bullish" if ema_9_val > ema_21_val else "Bearish"
                ema_aligned_with_signal = (
                    (ema_alignment == "Bullish" and direction == "BULL") or
                    (ema_alignment == "Bearish" and direction == "BEAR")
                )
                ema_stack_context = f"""
**5M EMA MICRO-STRUCTURE:**
- EMA 9: {format_price(ema_9_val)}
- EMA 21: {format_price(ema_21_val)}
- EMA 50: {format_price(ema_50_val)}
- 5m Trend: {ema_alignment} {'✅ Aligned' if ema_aligned_with_signal else '⚠️ Conflict (scoring penalty only, NOT auto-reject — SMC reversal entries legitimately fire against 5m micro)'}
"""

        # --- Bollinger Band context ---
        bb_context = ""
        if other_indicators:
            bb_upper_val = other_indicators.get('bb_upper')
            bb_middle = other_indicators.get('bb_middle')
            bb_lower_val = other_indicators.get('bb_lower')
            if bb_upper_val and bb_lower_val and bb_middle is not None and entry_price:
                pip_scale = 100 if 'JPY' in pair else 10000
                bb_width = (bb_upper_val - bb_lower_val) * pip_scale
                price_in_bb = "Upper band" if entry_price > bb_middle else "Lower band"
                bb_context = f"""
**BOLLINGER BAND CONTEXT:**
- BB Width: {bb_width:.1f} pips ({'Wide/Volatile' if bb_width > 30 else 'Narrow/Consolidating'})
- Entry Position: {price_in_bb} region
"""

        # --- RSI / ADX warnings ---
        rsi_warning = ''
        if rsi_value:
            if direction == 'BULL' and rsi_value > 75:
                rsi_warning = ' ⚠️ Overbought for BUY (extended)'
            elif direction == 'BEAR' and rsi_value < 25:
                rsi_warning = ' ⚠️ Oversold for SELL (extended)'
            elif direction == 'BULL' and rsi_value > 65:
                rsi_warning = ' • Extended for BUY (score penalty only, not reject)'
            elif direction == 'BEAR' and rsi_value < 35:
                rsi_warning = ' • Extended for SELL (score penalty only, not reject)'

        adx_warning = ' ⚠️ Weak trend' if adx_value and adx_value < 20 else ''
        mtf_status = ''
        if all_tfs_aligned:
            mtf_status = ' ✅ All TFs aligned'
        elif mtf_confluence is not None:
            mtf_status = ' ⚠️ Partial alignment'

        market_context_section = f"""
═══════════════════════════════════════════════════════════════
🌍 MARKET CONTEXT
═══════════════════════════════════════════════════════════════
**Regime:** {market_regime.upper()} (confidence: {regime_confidence:.0%})
**Session:** {current_session.upper()}{f'  |  Day: {day_of_week}' if day_of_week else ''}
**Volatility:** {volatility_level.upper() if volatility_level else 'N/A'}  |  ATR Percentile: {f'{atr_percentile:.0f}%' if atr_percentile is not None else 'N/A'}

**Technical Indicators:**
- RSI(14): {f'{rsi_value:.1f} ({rsi_zone})' if rsi_value else 'N/A'}{rsi_warning}
- ADX: {f'{adx_value:.1f} ({adx_trend_strength})' if adx_value else 'N/A'}{adx_warning}
- MTF Confluence: {f'{mtf_confluence:.2f}' if mtf_confluence is not None else 'N/A'}{mtf_status}
- Entry Quality Score: {f'{entry_quality:.2f}' if entry_quality is not None else 'N/A'}
- MI Confidence Modifier: {f'{mi_confidence_modifier:+.1%}' if mi_confidence_modifier is not None else 'N/A'}
"""

        lpf_section = ''
        if lpf_penalty or lpf_rules:
            rules_str = ', '.join(lpf_rules) if lpf_rules else 'none'
            lpf_section = f"""
⚠️ **LOSS PREVENTION FILTER ALERT:**
- Penalty: {f'{lpf_penalty:.2f}' if lpf_penalty else '0'}  |  Would Block: {'YES' if lpf_would_block else 'NO'}
- Rules Triggered: {rules_str}
(LPF uses historical win-rate data to flag high-risk conditions — weight this heavily in your assessment)
"""

        smc_analysis = f"""
## SMC SIMPLE v2.3.0 STRATEGY DATA (3-TIER VALIDATION)

**TIER 1 - {primary_htf_label} Directional Bias:**
- 50 EMA Value: {format_price(ema_value)}
- Distance from EMA: {ema_distance:.1f} pips {'✅' if ema_distance >= 2.5 else '⚠️ Close to EMA'}
- Bias Direction: {ema_direction}

**TIER 2 - {strategy_trigger.upper()} Swing Break:**
- Swing Level Broken: {format_price(swing_level)}
- Opposite Swing (SL reference): {format_price(opposite_swing)}
- Body Close Confirmed: {'✅ Yes' if body_close_confirmed else '❌ No'}
- Volume Spike: {volume_spike_display} — normal volume is fine; this only flags above-average spikes

**TIER 3 - Entry Analysis:**
{entry_type_detail}
- Fib Zone: {fib_zone}
- Order Type: {order_type.upper()}
{confidence_detail}
{sr_context}
{ema_stack_context}
{bb_context}
"""

        mo_note = (
            "\n⚠️ **MONITOR-ONLY / TEST MODE:** This is a demo-environment test signal. Score honestly; evaluate as if capital is at risk.\n"
            if monitor_only else ""
        )

        eurusd_exception = ""
        if pair == "EURUSD":
            eurusd_exception = f"""
═══════════════════════════════════════════════════════════════
🔵 EURUSD-SPECIFIC RULE OVERRIDE
═══════════════════════════════════════════════════════════════
EURUSD frequently produces early pullback/reversal entries where the {primary_htf_label} HTF opposes the signal but price has already locally reversed with a clean path.

**For EURUSD pullback/reversal entries ONLY:**
- Do NOT hard-reject SC_HTF_OPPOSE if ALL of the following hold:
  (a) Entry type is PULLBACK or the entry candle shows a reversal structure
  (b) Immediate path to TP is clear (no S/R within 50% of reward distance)
  (c) Nearby adverse excursion risk is low (nearest opposing S/R ≥ {sr_buffer_pips} pips from entry)
- When the above conditions hold: treat {primary_htf_label} HTF conflict as a −2 score penalty instead of a rejection trigger.
- Still REJECT SC_HTF_OPPOSE if S/R genuinely blocks the path to target (overlapping resistance/support within reward distance AND {primary_htf_label} opposes).
"""

        return f"""You are a SENIOR FOREX TECHNICAL ANALYST with 20+ years of institutional trading experience specializing in Smart Money Concepts (SMC) analysis.

**STRATEGY THESIS — READ FIRST (SMC_SIMPLE v2.3.0)**
SMC Simple is a 3-tier WITH-TREND entry system:
1. **{primary_htf_label} bias**: 50 EMA slope determines the allowed direction
2. **{strategy_trigger.upper()} trigger**: swing break (BOS) in that direction
3. **Entry**: pullback into Fib zone (38.2-61.8%) OR momentum continuation beyond the break

Critical rules for your analysis:
- ❌ Do NOT reject because 5m/entry-TF EMA9 crossed against the signal — the pullback entry IS that EMA cross
- ❌ Do NOT reject because RSI is extended — pullback entries fire at extended levels by definition
- ❌ Do NOT reject because volume is unconfirmed — "volume_confirmed" is a spike bonus flag, not a requirement
- ❌ Do NOT reject because 4H macro opposes the signal when {primary_htf_label} aligns — 4H is backdrop only (−1 penalty max)
- ✅ DO reject if {primary_htf_label} clearly opposes the signal AND MTF Confluence < 0.6
- ✅ DO reject if entry is AT (within 2 pips) a MAJOR S/R level that ALSO opposes {primary_htf_label} bias
- ✅ DO reject if S/R blocks >75% of path to target AND {primary_htf_label} opposes the signal
{eurusd_exception}
{mo_note}
═══════════════════════════════════════════════════════════════
✅ THE SETUP IS THE TRIGGER — DO NOT REJECT FOR THESE
═══════════════════════════════════════════════════════════════
These ARE the SMC_SIMPLE setup conditions — not red flags:
- Entry-TF EMA9/21 cross against signal direction: this IS the pullback being faded
- RSI in extended territory (65-75 BUY, 25-35 SELL): pullbacks enter at momentum extremes
- Counter-to-5m micro-trend: SMC reversal entries fire against 5m micro by design
- Volume not confirmed: "volume_confirmed" is a bonus flag — normal volume = valid entry
- Price near {primary_htf_label} EMA: this is exactly where pullback entries target
- MICRO_PULLBACK label vs MOMENTUM label: internal classifier — evaluate the structure, not the label
- Counter-4H when {primary_htf_label} aligns: macro context only; penalty not rejection

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: {strategy} v2.3.0
• Entry Type: {entry_type}
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {format_price(entry_price)}
• Stop Loss: {risk_pips:.1f} pips  (beyond opposite swing)
• Take Profit: {reward_pips:.1f} pips
• Risk:Reward Ratio: {rr_ratio:.2f}:1{fixed_sl_note}
{chart_instruction}
{market_context_section}
{lpf_section}
{smc_analysis}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

**CRITICAL:** Begin with `SCORE:`. No preamble. No narrative headers.

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | SC_HTF_OPPOSE | SC_AT_RESISTANCE | SC_PATH_BLOCKED | SC_MOMENTUM_REVERSAL]
REASON: [≤40 words. Focus on: (a) {primary_htf_label} trend alignment vs signal direction, (b) entry location quality (range position, S/R proximity), (c) swing break cleanliness. Do NOT mention 5m EMA cross, volume confirmation, or RSI as rejection reasons.]

**SCORING RUBRIC (4 anchors):**
- 9–10 EXCEPTIONAL: {primary_htf_label} trend clearly aligned (HH/HL or LH/LL confirmed), clean swing break, entry at favorable range location, no S/R obstruction
- 7–8  STRONG: {primary_htf_label} trend aligned with minor concerns (e.g., momentum slightly extended, minor S/R nearby, entry slightly outside optimal Fib zone)
- 5–6  ACCEPTABLE: {primary_htf_label} trend aligned, swing break present, entry location acceptable — positive expectancy midband
- 3–4  MARGINAL: {primary_htf_label} trend unclear/choppy, or entry near (but not at) a meaningful S/R level — APPROVE unless a rejection criterion is clearly met
- 1–2  REJECT: one of the rejection criteria below is clearly met

**Penalty cap:** combined penalty from market context (RSI, ADX, regime, MTF) is capped at −2 total. Do NOT stack all context items into a cumulative −4 to −6.

**REJECTION CRITERIA (any one = REJECT, assign matching REASON_CODE):**
- **{primary_htf_label} trend opposes signal AND MTF Confluence < 0.6** (skip if MTF Confluence ≥ 0.6 or "All TFs aligned") → SC_HTF_OPPOSE{' — see EURUSD-SPECIFIC RULE OVERRIDE above: for EURUSD pullback entries with clear path and low adverse excursion risk, downgrade to −2 penalty instead of rejection' if pair == 'EURUSD' else ''}
- **Entry within 2 pips of a MAJOR multi-touch S/R level AND {primary_htf_label} opposes signal AND no reversal confirmation candle** (all three must be true) → SC_AT_RESISTANCE
- **S/R on trigger TF blocks >75% of path to target AND {primary_htf_label} opposes signal** → SC_PATH_BLOCKED
- **MOMENTUM entry: reversal candle on trigger TF (engulfing/pin bar against direction) AND {primary_htf_label} opposes signal** (reversal candle alone is −2 penalty) → SC_MOMENTUM_REVERSAL

**Score penalties (apply as deductions, never as rejection alone):**
- EMA 9/21 crossed against signal on trigger TF → −1
- 5m EMA micro-structure conflict → −1
- Counter-4H when primary {primary_htf_label} aligns → −1
- RSI 65–75 BUY or 25–35 SELL → −1
- ADX < 20 (weak trend) → −1
- Entry slightly outside optimal Fib zone (PULLBACK) → −1
- Entry at top of local recovery / bottom of local selloff (alone, without the 3-condition AND-gate) → −2

Be concise. Your four lines determine if real capital is risked on this edge."""

    except Exception as e:
        logger.error(f"Error building SMC_SIMPLE prompt: {e}")
        return build_fallback_prompt(signal)
