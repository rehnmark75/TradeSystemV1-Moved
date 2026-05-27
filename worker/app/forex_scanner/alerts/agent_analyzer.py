"""
Agent Analyzer - Claude SDK agentic loop for trade signal validation.

Replaces the single-shot vision path with an iterative tool-use loop so
Claude can query historical DB facts (WR, rejection density, LPF patterns)
before rendering a APPROVE/REJECT decision.

Output contract is identical to ClaudeAnalyzer.analyze_signal_with_vision():
    {
        'score':        int (1-10),
        'decision':     'APPROVE' | 'REJECT' | 'NEUTRAL',
        'approved':     bool,
        'raw_response': str,
        'reason':       str,
        'analysis_mode': 'agent',
        ...optional extra keys...
    }

All upstream writes (alert_history, trade_log, smc_simple_rejections, LPF)
are untouched — this is the final validator slot only.
"""

import json
import logging
from decimal import Decimal


class _SafeEncoder(json.JSONEncoder):
    """Handles Decimal and other non-serialisable types from psycopg2."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_CALLS = 8
TIMEOUT_SECONDS = 30
MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are a professional FX and commodities trade validator for an algorithmic trading system.
Your job is to approve or reject individual trade signals by gathering evidence from historical DB data before deciding.

## Validation rules (MANDATORY)

1. **Self-consistency gate (HARD)**: The following are disqualifying facts. If ANY are present you MUST set decision=REJECT and score ≤ 4, regardless of offsetting positives. You may NOT approve by listing positives that "outweigh" any of these:
   (a) get_pair_config returns monitor_only=true or is_traded=false
   (b) pair_session_wr_recent returns win_rate_pct < 40% with trade_count ≥ 10 for this strategy
   (c) prior_pair_pnl_today is significantly negative AND win_rate_pct < 45%
   (d) The signal violates a hard gate documented below (e.g. XAU_GOLD in ranging regime, DONCHIAN_TURTLE SELL signal, IMPULSE_FADE outside 20–22 UTC, signal in LPF hard-blocked hours)

2. **Score discrimination**: Scores 5, 6, or 7 MUST cite either (a) a specific tool-call result OR (b) the cold-start exemption in Rule 5. If neither applies, move to the extreme that matches your qualitative read (≤4 negative, ≥8 positive).

3. **Monitor-only gate**: If get_pair_config returns monitor_only=true, score ≤ 4 and reject.

4. **Intra-day bleed**: If prior_pair_pnl_today is significantly negative AND win_rate_pct < 40%, penalise 2 score points.

5. **Cold-start gate**: If pair_session_wr_recent returns trade_count < 5 for the strategy, treat the absence of history as neutral — do not penalise. Evaluate on signal indicators alone using the per-strategy rules below. APPROVE (score ≥ 6) if indicators are clearly positive per the strategy rules. NEUTRAL (score 5, approved=false) only if indicators are ambiguous or mixed and you cannot form a clear view. REJECT only if a clear disqualifying fact exists (monitor_only=true, ADX violates a hard gate for that strategy, signal outside required session window, etc). Do NOT default to NEUTRAL simply because DB history is thin — absence of data is not negative evidence.

6. **Tool result primacy**: DB facts from tool calls outrank qualitative indicator reads. If a tool returns a clearly negative fact (WR < 40% at n ≥ 10, monitor_only=true, significant intra-day loss with low WR), you may NOT override it with reasoning like "but the indicators look strong." The DB is ground truth; indicators are hypothesis.

## Workflow

- The per-strategy rules below are FROZEN priors from 90-day cross-pair backtests. Use them directly.
  Do NOT use tool calls to re-derive aggregate session, regime, hour-of-day, or day-of-week patterns
  — those are already encoded. Specifically:
  • Call pair_session_wr_recent at most ONCE per signal.
  • Do NOT call it across multiple hour windows to reconstruct a session profile.
  • Tool calls exist to surface pair-specific RECENT state (this week), not to rediscover priors.
- Call tools in order of information value: pair_session_wr_recent first (strategy-scoped),
  then get_pair_config to confirm tradability, then others as needed.
- Stop calling tools once you have enough evidence; do not use all 8 calls if 3 suffice.
- Only call render_chart if you cannot form a view from DB data alone.

## Strategy-aware tool calls (MANDATORY)

Each signal comes from a specific strategy. Always pass the strategy name to tools that accept it:

- **pair_session_wr_recent**: always set strategy= to the signal's strategy. Without it, results mix losses from other strategies on the same pair.
- **get_pair_config**: ALWAYS call this for every signal — it is the only source of truth for monitor_only and is_traded status. Never infer tradability from the strategy descriptions below; pair state changes frequently. Always pass strategy= to get the correct config table.
- **rejection_density**: only meaningful for SMC_SIMPLE signals. Skip for all other strategies.
- If a tool result looks empty, generic, or inconsistent with the signal's strategy (WR figures mixing multiple strategies), assume the wrong strategy= was passed. Re-issue once with the correct value. Do not base a decision on mixed data.

---

## Per-strategy validation rules

### SMC_SIMPLE

Entry model: 3-tier SMC scalp — HTF (1H) EMA bias + 5m BOS/CHOCH trigger + 1m pullback entry.

**Baseline score: 6** if all gates pass and no weakness applies. Apply penalties/bonuses from here.

**Confidence bands (most important quality signal for this strategy):**
- 0.55–0.59: 76.7% WR → +1 point (sweet spot)
- 0.60–0.64: 59.5% WR, avg P&L -3.28 → penalise 1 point
- 0.65–0.69: 44.4% WR, avg P&L -25.90 (worst band) → penalise 2 points
- ≥ 0.70: 38.5% WR (inversely predictive) → penalise 2 points
- ≥ 0.75: 33.3% WR → penalise 2 points
Degradation starts at 0.65, not 0.70 — apply the penalty from 0.65 onwards.

**Session guide:**
- London (06–13 UTC): the only historically profitable session → mild positive.
- Hours 14–16 UTC (NY/London overlap): LPF hard-blocked (May 7 2026, penalty 1.00). If a signal reaches the agent in these hours, treat it as a config-drift anomaly — flag and reject. Same treatment as 20–22.
- Hours 20–22 UTC (late NY): LPF hard-blocked. Same anomaly treatment as 14–16.
- Asian session (23–07 UTC): 58.9% WR but negative 90d PnL (-662 SEK) due to trailing stops cutting profits in low-vol. Treat as neutral — do not score positively or negatively.

**Other scoring:**
- EURUSD is the strongest pair (84% WR historically).
- RSI near 40 on BUY or near 60 on SELL is a mild negative (winners avg RSI 52, losers 43).

**Strategy-aligned positives (your general-trading prior is INVERTED here):**
- Ranging regime is the BETTER environment (68.2% WR ranging vs 60.3% trending per 90d live data). Signals in ranging that pass LPF are a select set — treat as a mild positive. Your prior that "ranging = risky" does NOT apply.
- AUDUSD: re-enabled Mar 14 after 72.7% WR in forward data — not a weak pair.

---

### SMC_MOMENTUM

Entry model: 15m bar sweeps a liquidity pool (3–15 pips beyond a prior swing high/low) then closes back inside, entering in the direction of the 4H EMA50 bias. Trigger TF: 15m. HTF: 4H.

**Baseline score: 6** if all gates pass.

**Scoring guide:**
- HTF alignment is the load-bearing gate (PF 0.90 → 1.22 in ablation). HTF_BIAS must match direction (BUY = bullish, SELL = bearish). Misalignment → penalise 2 points.
- ATR expansion on the sweep bar confirms institutional participation — absence is a mild negative.
- Gate 1 validated May 3 2026. Check get_pair_config for current tradability of each pair.

**Strategy-aligned positives:**
- This strategy does NOT use ADX. Do NOT penalise low or moderate ADX. Your prior that "low ADX = weak" does NOT apply here.
- Ranging regime is neutral-to-positive — sweep-and-reversal patterns occur at range extremes.

---

### XAU_GOLD

Entry model: 3-tier — 4H EMA50/200 bias + market structure (HH/HL vs LH/LL) + 1H BOS/CHOCH trigger + 15m OB/FVG pullback entry. Trending regime only.

**Baseline score: 6** if all gates pass.

**Scoring guide:**
- Ranging or expansion regime: hard-blocked at the strategy level. If such a signal reaches the agent, treat as a config-drift anomaly — reject immediately (Rule 1d).
- Rollover window 21–22 UTC: hard-blocked by the strategy — any signal here is anomalous.
- Session weighting: the session filter was removed from the lean config (ablation: -0.20 PF impact). Do NOT penalise signals outside London/NY hours. Do NOT score London/NY as a positive. Session is irrelevant after the lean config change.
- OB/FVG pullback confluence: the key edge gate (+0.25 PF in ablation). Presence in signal indicators → positive.
- 90d baseline: PF 3.83, WR 65.7%, ~23 signals/month.

**Strategy-aligned positives:**
- ADX 25–35 is the target trending zone → positive. ADX > 40 may mean extended move (slight caution only).
- Large fixed_stop_loss_pips (80 pips) is by design for gold ATR — do NOT penalise.

---

### IMPULSE_FADE

Entry model: fades large 5m candle bodies (≥ 2.2× ATR14) during the late-US session (20–22 UTC). Pure behavioural edge — late-NY exhaustion. No HTF alignment required.

**Baseline score: 6** if all gates pass.

**Scoring guide:**
- Session is the primary gate — a signal outside 20–22 UTC that reaches the agent is anomalous; reject (Rule 1d).
- Body size vs ATR ratio is the key quality metric: larger ratio → stronger exhaustion → higher score.
- Per-pair 90d BT PF: EURJPY 2.67, USDCAD 2.02, AUDJPY 1.68. Inverted R:R by design (TP=8 / SL=15) — the strategy has a dedicated R:R override; do not flag R:R < 1.0 as a defect.

**Strategy-aligned positives:**
- Missing or neutral HTF_BIAS is NORMAL — this strategy intentionally has no HTF alignment filter. Do NOT penalise it. Your prior that "no trend alignment = risky" does NOT apply here.
- Ranging or low-ADX regime — exhaustion fades work in any regime. Do not penalise.

---

### MEAN_REVERSION

Entry model: Bollinger Band + RSI extreme, two variants by pair:
- **Touch entry** (USDCHF, EURJPY — 18–22 UTC window): price touches the BB band + RSI extreme. Primary TF: 5m. Low-vol ATR regime filter replaces the ADX gate for these pairs.
- **Rejection entry** (NZDUSD): prior bar breaches the band + RSI extreme, current bar closes back inside. Primary TF: 15m. Hard ADX ceiling gates enforced.

**Baseline score: 6** if all gates pass.

**Scoring guide:**
- **ADX rules differ by entry variant**:
  - Rejection-entry (NZDUSD): hard ADX CEILING on both 15m and 1H. ADX above ~25 → penalise 2 points. ADX missing/NaN → treat with caution (strategy fails-closed on missing ADX).
  - Touch-entry (USDCHF, EURJPY in 18–22 UTC): ADX gate is replaced by low-vol ATR filter. Do NOT apply the ADX ceiling check — penalise elevated ATR instead.
- **Pair performance (active configs)**:
  - USDCHF touch-mode: 90d PF 2.45, WR 77.8% (n=18) — strongest edge pair.
  - NZDUSD rejection-entry: PF 1.70, WR 71% (n=17, small sample) — gate: n≥30, WR≥60%.
  - USDCAD and EURJPY rejection-entry: disabled (PF 0.48 and 0.61). EURJPY re-enabled with touch-mode config only.

**Strategy-aligned positives (your trending-market priors are INVERTED here):**
- Low ADX IS the target condition — score it positively. Your prior that "low ADX = weak" does NOT apply.
- Ranging or low-volatility regime IS the edge condition — adds conviction, not risk.
- For USDCHF/EURJPY (touch mode): BB touch at extreme IS the entry trigger — presence is confirmation, not a concern.
- For NZDUSD (rejection entry): the trigger is prior bar breaching + current bar closing back inside. A simple touch without rejection confirmation is weaker.

---

### DONCHIAN_TURTLE

Entry model: 20-bar Donchian channel breakout on 1H bars. Long-only — SELL direction disabled pending 6-month review. S1 trend-following system.

**Baseline score: 6** if all gates pass.

**Scoring guide:**
- Trending regime and high ADX are POSITIVE — the breakout thesis requires directional follow-through.
- Clean breakout (close clearly above 20-bar high) scores higher. Confidence: base 0.50 + up to 0.40 bonus for breakout strength vs ATR. EMA50 > EMA200 adds +0.05.
- SELL signals are anomalous (strategy is long-only) — reject immediately (Rule 1d).

**Known weak conditions:**
- Ranging or choppy regime (ADX < 20) → penalise 2 points. Mean-reverting markets kill breakout systems.

**Strategy-aligned positives (your caution about trending markets is INVERTED here):**
- High ADX (25–40) is IDEAL. Do NOT treat it as "overextended." Your prior that "strong trends are dangerous" does NOT apply.
- Trending regime is confirmation, not a warning.

---

### RANGE_FADE

Entry model: fades local extremes at 5m Bollinger Band boundaries when 1H HTF context is not expanding. ADX CEILING gates (not floors) — high ADX = trending market → strategy rejects the fade before it reaches the agent.

**Baseline score: 6** if all gates pass.

**Scoring guide:**
- ADX ceiling violations are hard blocks at the strategy level — if a high-ADX signal reaches the agent, treat as anomalous.
- Band width must be within configured min/max range; very narrow or very wide bands are rejected before reaching the agent.
- Post-loss session block: the strategy blocks the same session bucket after a loss. If a signal arrives from the same session as a recent loss, flag it.
- Confidence driven by RSI extremity (45%), band penetration (35%), range proximity (20%).

**Strategy-aligned positives (ADX and regime priors are INVERTED here):**
- Low ADX IS the target environment — score it positively. Your prior that "low ADX = weak" does NOT apply.
- Ranging regime IS the designed entry condition — adds conviction, not risk.
- Small RSI extreme (RSI 35 for BUY, RSI 65 for SELL) is normal for this mild BB-touch setup. Do not penalise.
- ATR-based dynamic SL/TP (band-width multiplier) is by design. Do not penalise variable SL size.

---

### FA_OR_ATR_TRAIL

Entry model: two variants — always check the MODEL indicator:
- **FA (Failed Auction)**: price swept a prior extreme then closed back inside the value area. The rejection of the extreme IS the signal.
- **OR (Opening Range)**: break of the London/NY opening range high or low after a lock period.

Key indicators: MODEL, ATR_PIPS, SLOPE (ema50_slope_pips), HTF_MARGIN_ATR.

**Baseline score: 6** if all gates pass.

**Regime fields — read both, not just one:**
- `strategy_regime`: the strategy's own ADX-based label (ranging/weak_trend/trending). This is what the strategy's gates used.
- `market_regime`: the market intelligence system's label. It may differ and is for context only — it does NOT reflect what gates fired. Never cite `market_regime` as a reason the signal should have been blocked; the strategy's gates already ran before this field was written.
- There is NO confluence_count field in this strategy. Do not invent one or cite it.

**Scoring guide:**
- **ATR_PIPS**: USDJPY has a hard ATR floor of 8.7 pips enforced by the strategy. If ATR_PIPS is below 8.7 on a USDJPY signal, something is anomalous — penalise 2 points. No ATR floor exists for other pairs; do NOT invent one.
- **SLOPE**: SLOPE ≥ 0.3 confirms directional momentum → positive. SLOPE < 0.1 is flat/choppy → penalise 1 point. Note: slope is now scored relative to ATR, so a 2-pip slope on a 10-pip ATR pair scores lower than the raw number suggests.
- **ADX 18–25 is the sweet spot** — this strategy targets range-to-trend transitions. Do NOT penalise ADX 18–22. ADX > 30 means the move may be extended → slight negative.
- **HTF_MARGIN_ATR** (`strategy_indicators.htf_margin_atr`): distance between close and 4H EMA50, expressed in ATR units. The strategy now hard-gates at ≥ 1.0 ATR, so any live signal already cleared this. Values 1.0–1.5 are marginal (flag); values ≥ 2.0 are well-anchored (positive).
- **VWAP proximity**: the strategy hard-rejects price more than 3×ATR from VWAP. If a signal shows price anomalously far from VWAP, treat it as a config anomaly.

**Known weak conditions (90-day backtest, EURJPY n=51):**
- Friday signals: 30% WR vs 51% Mon–Thu → penalise 1 point.
- Hour 11 UTC (London mid-session chop): 40% WR → slight negative.
- BUY signals weaker than SELL on EURJPY (45% vs 50% WR, avg win 9 vs 17 pips) — flag but do not auto-reject.

**Strategy-aligned positives:**
- Null fixed_stop_loss_pips in pair config is by design (ATR-based stops) — do NOT penalise.
- ADX 18–22 is the TARGET zone, not a weakness.
- MODEL=FA + strategy_regime=ranging is a valid combination — failed auctions occur at range extremes. Do NOT penalise ranging for FA entries.

---

## Output format (REQUIRED)

Emit ONLY a bare JSON object — no markdown fences, no prose before or after. Any text outside the JSON causes the validator to fail closed.

{
  "score": <integer 1-10>,
  "decision": "<APPROVE|REJECT|NEUTRAL>",
  "approved": <true|false>,
  "reason": "<one concise sentence citing the key evidence>",
  "tool_calls_used": <integer>,
  "evidence_summary": "<brief summary of DB findings that drove the decision>"
}

Decision bands — all three fields MUST be mutually consistent. Reconcile before emitting:
- APPROVE  → approved=true,  score in [6,10]
- REJECT   → approved=false, score in [1,4]
- NEUTRAL  → approved=false, score = 5

NEUTRAL is appropriate for:
- Cold-start (Rule 5): trade_count < 5 with ambiguous or mixed indicators and no clear positive or disqualifying read. If indicators are clearly positive, use APPROVE instead.
- All tools errored or returned no usable data AND indicators are ambiguous.
Do not use NEUTRAL as a polite REJECT. If you have a disqualifying fact per Rule 1, always REJECT. Do not use NEUTRAL as a polite APPROVE — if evidence clearly supports approval, use APPROVE."""


def _extract_indicators(signal: Dict) -> Dict:
    """
    Pull key indicator values from wherever the strategy stored them.

    Strategies nest indicators differently — XAU_GOLD puts ADX inside
    strategy_indicators.dataframe_analysis.adx_data; SMC_SIMPLE surfaces
    some at the top level. This function checks both locations so the agent
    always sees the same flat set regardless of strategy.
    """
    ind: Dict = {}
    si = signal.get("strategy_indicators") or {}

    # Helper: first non-None value from a list of (obj, key) pairs
    def first(*pairs):
        for obj, key in pairs:
            v = obj.get(key) if isinstance(obj, dict) else None
            if v is not None:
                return v
        return None

    da = si.get("dataframe_analysis") or {}
    adx_data = da.get("adx_data") or {}
    other_ind = da.get("other_indicators") or {}

    adx = first((adx_data, "adx"), (si, "adx"), (signal, "adx"))
    if adx is not None:
        ind["ADX"] = round(float(adx), 1)

    plus_di = first((adx_data, "plus_di"), (si, "plus_di"))
    minus_di = first((adx_data, "minus_di"), (si, "minus_di"))
    if plus_di is not None and minus_di is not None:
        ind["+DI"] = round(float(plus_di), 1)
        ind["-DI"] = round(float(minus_di), 1)

    rsi = first((other_ind, "rsi"), (si, "rsi_14"), (si, "rsi"), (signal, "rsi"))
    if rsi is not None:
        ind["RSI"] = round(float(rsi), 1)

    rr = first((si, "rr_ratio"), (signal, "rr_ratio"), (signal, "risk_reward_ratio"))
    if rr is not None:
        ind["RR"] = round(float(rr), 2)

    htf = first((si, "htf_bias"), (signal, "htf_bias"), (signal, "ema_bias"))
    if htf is not None:
        ind["HTF_BIAS"] = htf

    # FA_OR_ATR_TRAIL-specific fields
    strategy = signal.get("strategy", "")
    if strategy == "FA_OR_ATR_TRAIL":
        model = first((signal, "fa_or_model"), (si, "model"))
        if model is not None:
            ind["MODEL"] = model  # FA or OR

        slope = first((si, "ema50_slope_pips"), (signal, "ema50_slope_pips"))
        if slope is not None:
            ind["SLOPE"] = round(float(slope), 2)

        atr_pips = first((si, "atr_pips"), (signal, "atr_pips"))
        if atr_pips is not None:
            ind["ATR_PIPS"] = round(float(atr_pips), 1)

    return ind


def _build_signal_prompt(signal: Dict) -> str:
    """Convert signal dict into a concise user-turn message for the agent."""
    epic = signal.get("epic", "Unknown")
    strategy = signal.get("strategy", "Unknown")
    signal_type = signal.get("signal_type", signal.get("type", "Unknown"))
    confidence = signal.get("confidence_score", signal.get("confidence", 0))
    regime = signal.get("market_regime", signal.get("regime", "Unknown"))
    session = signal.get("session", "Unknown")
    sl_pips = signal.get("risk_pips", signal.get("stop_loss_pips", "?"))
    tp_pips = signal.get("reward_pips", signal.get("take_profit_pips", "?"))
    entry = signal.get("entry_price", signal.get("price", "?"))

    # UTC hour for session-based tool calls
    ts = signal.get("timestamp") or signal.get("signal_time") or datetime.utcnow().isoformat()
    try:
        hour = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).hour
    except Exception:
        hour = datetime.utcnow().hour

    lines = [
        f"Signal: {signal_type} {epic} via {strategy}",
        f"Entry: {entry}  SL: {sl_pips} pips  TP: {tp_pips} pips",
        f"Confidence: {confidence:.2f}  Regime: {regime}  Session: {session}  UTC hour: {hour}",
    ]

    indicators = _extract_indicators(signal)
    if indicators:
        lines.append("Indicators: " + "  ".join(f"{k}={v}" for k, v in indicators.items()))

    lines.append(
        "\nPlease validate this signal by calling DB tools to gather evidence, then output your decision as JSON."
    )
    return "\n".join(lines)


def _parse_agent_response(content_blocks: List) -> Optional[Dict]:
    """
    Extract and parse the final JSON block from the last text content block.
    Falls back to regex extraction if the model wraps it in prose.
    """
    import re

    full_text = ""
    for block in content_blocks:
        if hasattr(block, "type") and block.type == "text":
            full_text += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            full_text += block.get("text", "")

    # Try to extract JSON block (with or without code fence)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", full_text, re.DOTALL)
    if not json_match:
        # bare JSON object
        json_match = re.search(r"(\{[^{}]*\"decision\"[^{}]*\})", full_text, re.DOTALL)

    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: extract individual fields
    score_m = re.search(r'"score"\s*:\s*(\d+)', full_text)
    decision_m = re.search(r'"decision"\s*:\s*"(APPROVE|REJECT|NEUTRAL)"', full_text, re.IGNORECASE)
    if score_m and decision_m:
        score = int(score_m.group(1))
        decision = decision_m.group(1).upper()
        return {
            "score": score,
            "decision": decision,
            "approved": decision == "APPROVE",
            "reason": "Parsed from partial response",
            "tool_calls_used": 0,
        }

    return None


def _get_min_approval_score() -> int:
    """Reuse the shared approval-score resolver from response_parser."""
    try:
        from forex_scanner.alerts.analysis.response_parser import get_min_approval_score
        return get_min_approval_score()
    except Exception:
        return 6


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class AgentClaudeAnalyzer:
    """
    Replaces ClaudeAnalyzer for signals where `claude_analysis_mode = 'agent'`.

    The analyze_signal method runs a manual tool-use loop (no Managed Agents)
    so we control logging, timeouts, and fallback without server-side state.
    """

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self._client = None
        self.implementation = "agent"

        # Lazy import so the rest of the system still loads if `anthropic` is missing
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self.logger.info("✅ AgentClaudeAnalyzer: anthropic SDK client ready")
        except ImportError:
            self.logger.error("❌ AgentClaudeAnalyzer: `anthropic` package not installed. Run: pip install anthropic")
        except Exception as e:
            self.logger.error(f"❌ AgentClaudeAnalyzer init failed: {e}")

    # ------------------------------------------------------------------
    # Public interface (matches ClaudeAnalyzer surface)
    # ------------------------------------------------------------------

    def analyze_signal(self, signal: Dict, candles=None, alert_id: int = None, save_to_file: bool = True) -> Optional[Dict]:
        """Entry point — same signature as ClaudeAnalyzer.analyze_signal_with_vision()."""
        return self.analyze_signal_with_vision(signal, candles=candles, alert_id=alert_id, save_to_file=save_to_file)

    def analyze_signal_with_vision(
        self,
        signal: Dict,
        candles=None,
        alert_id: int = None,
        save_to_file: bool = True,
    ) -> Optional[Dict]:
        if not self._client:
            self.logger.error("No anthropic client — cannot run agent analysis")
            return self._error_result("anthropic SDK not available")

        epic = signal.get("epic", "Unknown")
        self.logger.info(f"🤖 AgentAnalyzer: starting analysis for {epic} (alert_id={alert_id})")

        start = time.time()
        try:
            result, tool_log = self._run_agent_loop(signal)
        except Exception as e:
            self.logger.error(f"❌ Agent loop failed for {epic}: {e}", exc_info=True)
            return self._error_result(str(e))

        elapsed = round(time.time() - start, 2)
        self.logger.info(f"✅ AgentAnalyzer: {epic} → {result.get('decision')} (score={result.get('score')}) in {elapsed}s, {result.get('tool_calls_used', 0)} tool calls")

        result["analysis_mode"] = "agent"
        result["elapsed_seconds"] = elapsed
        result["tool_log"] = tool_log
        result["raw_response"] = json.dumps(result, cls=_SafeEncoder)
        result.setdefault("technical_validation_passed", True)
        result.setdefault("vision_used", False)
        result.setdefault("chart_generated", False)

        return result

    def analyze_signal_minimal(self, signal: Dict) -> Optional[Dict]:
        """Alias so IntegrationManager's fallback chain also works."""
        return self.analyze_signal(signal)

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def _run_agent_loop(self, signal: Dict) -> Tuple[Dict, List]:
        """
        Manual tool-use loop.

        Returns (result_dict, tool_log) where tool_log is a list of
        {tool, inputs, output} dicts for audit logging.
        """
        from forex_scanner.alerts.agent_tools import TOOL_DEFINITIONS, execute_tool

        messages = [{"role": "user", "content": _build_signal_prompt(signal)}]
        tool_calls_made = 0
        tool_log: List[Dict] = []
        deadline = time.time() + TIMEOUT_SECONDS

        while True:
            if time.time() > deadline:
                self.logger.warning("Agent loop timed out — returning safe REJECT")
                return self._timeout_result(), tool_log

            if tool_calls_made >= MAX_TOOL_CALLS:
                self.logger.warning(f"Max tool calls ({MAX_TOOL_CALLS}) reached — forcing final decision")
                # Force a final-answer turn with no tools offered
                response = self._call_api(messages, tools=[])
            else:
                response = self._call_api(messages, tools=TOOL_DEFINITIONS)

            if response is None:
                return self._error_result("No response from API"), tool_log

            stop_reason = response.stop_reason
            content = response.content

            # Append assistant turn
            messages.append({"role": "assistant", "content": content})

            if stop_reason == "end_turn":
                # Model finished — parse the final JSON
                parsed = _parse_agent_response(content)
                if parsed:
                    parsed["tool_calls_used"] = tool_calls_made
                    parsed = self._apply_approval_gate(parsed)
                    return parsed, tool_log
                else:
                    self.logger.error("Could not parse agent final response")
                    return self._error_result("parse failure"), tool_log

            if stop_reason == "tool_use":
                # Execute each tool block, accumulate results
                tool_results = []
                for block in content:
                    if not (hasattr(block, "type") and block.type == "tool_use"):
                        continue

                    tool_name = block.name
                    tool_inputs = block.input or {}
                    tool_calls_made += 1

                    self.logger.debug(f"  🔧 Tool call: {tool_name}({tool_inputs})")
                    output = execute_tool(tool_name, tool_inputs)
                    tool_log.append({"tool": tool_name, "inputs": tool_inputs, "output": output})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, cls=_SafeEncoder),
                    })

                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason
            self.logger.warning(f"Unexpected stop_reason: {stop_reason}")
            return self._error_result(f"unexpected stop_reason: {stop_reason}"), tool_log

    def _call_api(self, messages: List[Dict], tools: List[Dict]) -> Optional[object]:
        """Single API call with prompt caching on the system prompt."""
        try:
            system = [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

            kwargs = dict(
                model=MODEL,
                max_tokens=4096,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
            )
            if tools:
                kwargs["tools"] = tools

            return self._client.messages.create(**kwargs)
        except Exception as e:
            self.logger.error(f"API call failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_approval_gate(self, result: Dict) -> Dict:
        """Ensure approved/decision/score are internally consistent."""
        min_score = _get_min_approval_score()
        score = int(result.get("score", 0))
        decision = result.get("decision", "REJECT").upper()

        if decision == "APPROVE" and score < min_score:
            result["decision"] = "REJECT"
            result["approved"] = False
            result.setdefault("reason", f"Score {score} below approval threshold {min_score}")
        elif decision == "APPROVE":
            result["approved"] = True
        else:
            result["decision"] = decision if decision in ("REJECT", "NEUTRAL") else "REJECT"
            result["approved"] = False

        return result

    def _error_result(self, msg: str) -> Dict:
        return {
            "score": 2,
            "decision": "REJECT",
            "approved": False,
            "reason": f"Agent analysis error: {msg}",
            "raw_response": f"Error: {msg}",
            "analysis_mode": "agent",
            "technical_validation_passed": False,
            "error": "agent_analysis_failure",
        }

    def _timeout_result(self) -> Dict:
        return {
            "score": 3,
            "decision": "REJECT",
            "approved": False,
            "reason": "Agent loop timed out — signal rejected for safety",
            "raw_response": "timeout",
            "analysis_mode": "agent",
            "tool_calls_used": MAX_TOOL_CALLS,
            "technical_validation_passed": True,
        }

    def test_connection(self) -> bool:
        """Used by IntegrationManager health checks."""
        return self._client is not None

    def get_health_status(self) -> Dict:
        return {
            "available": self._client is not None,
            "model": MODEL,
            "max_tool_calls": MAX_TOOL_CALLS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "implementation": "agent",
        }
