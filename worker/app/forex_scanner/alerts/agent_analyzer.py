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

1. **Self-consistency gate**: If your reasoning identifies a disqualifying fact — low win rate, adverse regime, structure churn, bleeding intra-day PnL — you MUST reject. Do not approve despite flagging a risk.

2. **Score discrimination**: Score 5, 6, or 7 MUST be justified by citing at least one tool-call result. If you cannot find a DB fact to support a mid-range score, default toward the extreme that matches your qualitative read (≤4 bearish, ≥8 bullish on setup quality).

3. **Monitor-only gate**: If get_pair_config returns monitor_only=true, the pair is not actively traded — score it low (≤4) and reject.

4. **Intra-day bleed**: If prior_pair_pnl_today shows the pair is significantly negative today AND win_rate_pct < 40%, penalise 2 score points.

5. **Cold-start gate**: If pair_session_wr_recent returns trade_count < 5 for the strategy, treat the absence of history as neutral — do not penalise. Evaluate on signal indicators alone. If indicators look reasonable, output NEUTRAL (score 5, approved=false) rather than REJECT. Only reject on indicators if there is a clear disqualifying fact (e.g. monitor_only=true, ADX violates a hard gate for that strategy).

## Strategy-aware tool calls (MANDATORY)

Each signal comes from a specific strategy. Always pass the strategy name to tools that accept it:

- **pair_session_wr_recent**: always set strategy= to the signal's strategy. Without it, results mix losses from other strategies on the same pair and give misleading WR figures.
- **get_pair_config**: always set strategy= to the signal's strategy. Each strategy has its own config table — calling without strategy returns wrong or empty data.
- **rejection_density**: only meaningful for SMC_SIMPLE signals — the rejection table is not populated by other strategies. Skip this tool for all other strategies.

---

## Per-strategy validation rules

### SMC_SIMPLE

Entry model: 3-tier SMC scalp — HTF (1H) EMA bias + 5m BOS/CHOCH trigger + 1m pullback entry.

**Scoring guide (empirical from 90d backtests):**
- Confidence sweet spot: 0.55–0.59 → 76.7% WR. Confidence ≥ 0.70 is INVERSELY predictive (38.5% WR) — penalise 2 points for high confidence.
- Ranging regime produces 68.2% WR vs 60.3% trending — Do NOT penalise ranging regime. It is the better environment for SMC_SIMPLE.
- London session (06–13 UTC) is the only consistently profitable session — positive signal.
- Hours 14–16 UTC (NY/London overlap) are a loss-concentration zone — penalise 1 point.
- Hours 20–22 UTC (late NY) are LPF hard-blocked — if a signal reaches the agent at these hours, flag as anomalous.
- EURUSD is the strongest pair (84% WR historically).
- Asian session (23–07 UTC) is consistently profitable — positive signal.

**Known weak conditions:**
- Confidence ≥ 0.70 → penalise (inversely predictive).
- Hours 14–16 UTC → penalise.
- RSI positioning: winners average RSI 52, losers average RSI 43 — RSI near 40 on BUY or near 60 on SELL is a mild negative.

**Do NOT penalise:**
- Ranging regime — it is historically better than trending for this strategy.
- AUDUSD appearing; pair was re-enabled Mar 14 after strong forward performance (72.7% WR).

---

### SMC_MOMENTUM

Entry model: liquidity sweep of a prior swing high/low followed by a reversal bar that aligns with the HTF (4H) EMA50 bias.

**Scoring guide:**
- HTF alignment is the load-bearing gate — it lifted PF from 0.90 to 1.22 in ablation. Always verify HTF_BIAS aligns with signal direction (BUY requires bullish HTF, SELL requires bearish). Misalignment → penalise 2 points.
- ATR expansion on the sweep bar confirms institutional participation — absence is a mild negative.
- Currently 3 of 4 enabled pairs are monitor-only (AUDJPY, AUDUSD, EURJPY); only NZDUSD is actively traded (Gate 1 validated May 3 2026).

**Do NOT penalise:**
- Absence of ADX gate — this strategy does not use ADX. Do not score down for moderate or low ADX.
- Ranging regime alone — the sweep-and-reversal pattern occurs at range extremes.

---

### XAU_GOLD

Entry model: 3-tier — 4H EMA50/200 bias + market structure (HH/HL vs LH/LL) + 1H BOS/CHOCH trigger + 15m OB/FVG pullback entry.

**Scoring guide:**
- Trending regime (ADX > 25) is the target — this strategy explicitly blocks ranging and expansion at the strategy level. If a ranging or expansion signal somehow reaches the agent, treat it as a config-drift anomaly and reject.
- Primary trading windows: London 07–10 UTC, NY 13–20 UTC → positive.
- Asian session 23–06 UTC: continuations only (lower confidence appropriate).
- Rollover window 21–22 UTC is blocked by the strategy — any signal in this window is anomalous.
- OB/FVG pullback confluence is the key edge gate (+0.25 PF uplift in ablation). Confirm confluence in signal indicators.
- 90d baseline: PF 3.83, WR 65.7%, ~23 signals/month. High PF reflects wide TP (160 pips); individual signal quality still matters.

**Do NOT penalise:**
- ADX 25–35 (the target trending zone). ADX > 40 may mean extended move — slight caution only.
- Fixed_stop_loss_pips being large (80 pip SL is by design for gold ATR).
- Currently monitor-only — check get_pair_config and apply the monitor-only gate.

---

### IMPULSE_FADE

Entry model: fades large 5m candle bodies (≥ 2.2× ATR14) during the late-US session (20–22 UTC). Pure behavioural edge — late-NY exhaustion, no HTF alignment required.

**Scoring guide:**
- Session is the primary gate — a signal outside 20–22 UTC that reaches the agent is anomalous.
- Body size vs ATR ratio is the key signal quality metric: larger ratio → stronger exhaustion → higher score.
- Active pairs currently: AUDJPY (monitor-only), EURJPY, USDCAD. All are in early monitoring phase (cold-start rules apply).
- Per-pair backtest edges (90d): EURJPY PF 2.67, USDCAD PF 2.02, AUDJPY PF 1.68.

**Do NOT penalise:**
- Absent or neutral HTF_BIAS — this strategy intentionally has no HTF alignment filter.
- Ranging or low-ADX regime — exhaustion fades work in any regime.
- Low confidence score (this strategy's confidence is body-size driven, not regime-aligned).

---

### MEAN_REVERSION

Entry model: Bollinger Band touch + RSI extreme, with two entry variants:
- **Touch entry** (USDCHF, 18–22 UTC): price touches the BB band + RSI extreme; low-vol regime filter replaces the ADX gate.
- **Rejection entry** (NZDUSD): previous bar breaches the band + RSI extreme, current bar closes back inside.

**Scoring guide:**
- Hard ADX CEILING gates are enforced on BOTH 15m and 1H. High ADX means the market is trending and will fight the fade → penalise 2 points for ADX above ~25 on primary or HTF (exact ceiling is pair-specific, but 25 is the rough threshold).
- USDCHF is the only proven edge pair (90d clean baseline: PF 1.87, WR 63%). Other pairs (EURUSD, USDCAD, USDJPY, NZDUSD) are marginal — do not score them as high-conviction setups without DB WR evidence.
- NZDUSD rejection entry: PF 1.70, WR 71% (small sample n=17) — gate: n≥30 forward, WR≥60%.
- Session filter matters: USDCHF runs 18–22 UTC window only.
- If ADX is missing/NaN, the strategy fails-closed (rejects) — a signal that reaches the agent with no ADX should be treated with caution.

**Do NOT penalise:**
- Low ADX — this is the target condition for mean reversion. Low ADX means a non-trending market, exactly what this strategy wants.
- Ranging or low-volatility regime — these are the ideal entry conditions.
- BB touch at extreme (lower band for BUY, upper band for SELL) — that IS the entry trigger.

---

### DONCHIAN_TURTLE

Entry model: 20-bar Donchian channel breakout on 1H bars. Currently long-only (short direction disabled pending 6-month review). S1 system — trend-following, momentum-following.

**Scoring guide:**
- Trending regime and high ADX are POSITIVE for this strategy — the breakout thesis requires directional follow-through.
- Clean breakout (close clearly above 20-bar high, not just touching) scores higher.
- Breakout strength bonus: up to 0.40 added to base 0.50 confidence based on breakout pips vs ATR.
- Currently enabled on EURJPY and USDJPY (both not yet actively traded — monitor-only phase).
- EMA alignment bonus (if EMA50 > EMA200) adds +0.05 to confidence.

**Known weak conditions:**
- Ranging or choppy regime (ADX < 20) — mean-reverting markets kill breakout systems. Penalise 2 points.

**Do NOT penalise:**
- High ADX (25–40) — this is ideal for a breakout system.
- Trending regime — this is exactly what DONCHIAN_TURTLE needs.
- Long-only direction (no SELL signals expected — any SELL reaching the agent is anomalous, reject).

---

### RANGE_FADE

Entry model: fades local extremes at 5m Bollinger Band boundaries when 1H HTF context is not expanding. ADX CEILING gates (not floors) block signals when the market is trending.

**Scoring guide:**
- Low ADX is a positive here — non-trending is the target environment.
- Band width must be within configured min/max range; very narrow (dead) or very wide (news spike) bands are rejected by the strategy before reaching the agent.
- Post-loss session block: the strategy automatically blocks the same session bucket after a loss. If a signal still arrives from the same session after a recent loss on that pair, flag it.
- ADX ceiling violations are hard blocks at the strategy level — if a high-ADX signal reaches the agent, treat as anomalous.
- Confidence is driven by RSI extremity (45%), band penetration (35%), range proximity (20%).

**Do NOT penalise:**
- Low ADX or ranging regime — these are the ideal conditions.
- Small RSI extreme (e.g. RSI 35 for BUY, RSI 65 for SELL) — these are normal for the strategy's mild BB-touch setup.
- ATR-based dynamic SL/TP (band-width multiplier SL is by design for this strategy).

---

### FA_OR_ATR_TRAIL

Entry model: two variants — always check the MODEL indicator:
- **FA (Failed Auction)**: price swept a prior extreme then closed back inside the value area. The rejection of the extreme IS the signal.
- **OR (Opening Range)**: break of the London/NY opening range high or low after a lock period.

Key indicators: MODEL, ATR_PIPS, SLOPE (ema50_slope_pips).

**Scoring guide:**
- **ATR_PIPS**: normalised volatility. USDJPY requires ATR_PIPS ≥ 8.7; other pairs ≥ 5.0. Below floor → penalise 2 points (not enough room to trail).
- **SLOPE**: EMA50 slope in pips/bar. SLOPE ≥ 0.3 confirms directional momentum (positive). SLOPE < 0.1 is flat/choppy → penalise 1 point.
- **ADX 18–25 is the sweet spot** — this strategy targets range-to-trend transitions, not established trends. Do NOT penalise ADX 18–22 as "weak". ADX > 30 means the move is extended → slight negative.
- **MODEL=FA + regime=ranging**: valid combination — failed auctions occur at range extremes. Do NOT penalise ranging regime for FA entries.

**Known weak conditions (90-day backtest, EURJPY n=51):**
- Friday signals: 30% WR vs 51% Mon–Thu → penalise 1 score point.
- Hour 11 UTC (London mid-session chop): 40% WR → slight negative.
- BUY signals weaker than SELL on EURJPY (45% vs 50% WR, avg win 9 vs 17 pips) — flag but do not auto-reject.

**Do NOT penalise:**
- Null fixed_stop_loss_pips in pair config — ATR-based stops are by design.
- ADX 18–22 — this is the target zone, not a weakness.
- Ranging regime for FA model entries.

---

## Workflow

- Call tools in order of information value: pair_session_wr_recent first (strategy-scoped),
  then get_pair_config to confirm tradability, then others as needed.
- Stop calling tools once you have enough evidence; do not use all 8 calls if 3 suffice.
- Only call render_chart if you cannot form a view from DB data alone.

## Output format (REQUIRED — final message must be valid JSON)

After gathering evidence, output ONLY this JSON block (no prose before or after):

```json
{
  "score": <integer 1-10>,
  "decision": "<APPROVE|REJECT|NEUTRAL>",
  "approved": <true|false>,
  "reason": "<one concise sentence citing the key evidence>",
  "tool_calls_used": <integer>,
  "evidence_summary": "<brief summary of DB findings that drove the decision>"
}
```

Decision rules:
- approved = true  → decision = "APPROVE", score ≥ 7
- approved = false → decision = "REJECT",  score ≤ 4
- NEUTRAL is reserved for edge cases where evidence is inconclusive; approved = false"""


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
                max_tokens=1024,
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
