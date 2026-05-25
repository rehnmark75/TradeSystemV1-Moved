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

3. **Regime gate for XAU**: Gold (epic contains CFEGOLD) in a 'ranging' or 'expansion' regime is a hard reject unless pair_session_wr_recent shows WR > 50% at n ≥ 10. Ranging gold has historically produced < 25% WR.

4. **Monitor-only gate**: If get_pair_config returns monitor_only=true, the pair is not actively traded — score it low and reject.

5. **Intra-day bleed**: If prior_pair_pnl_today shows the pair is significantly negative today AND win_rate_pct < 40%, penalise 2 score points.

## Strategy-aware tool calls (MANDATORY)

Each signal comes from a specific strategy (SMC_SIMPLE, XAU_GOLD, IMPULSE_FADE, DONCHIAN_TURTLE,
MEAN_REVERSION, SMC_MOMENTUM, RANGE_FADE, FA_OR_ATR_TRAIL). Always pass the strategy name to tools that accept it:

- **pair_session_wr_recent**: always set strategy= to the signal's strategy. Without it, results
  mix losses from other strategies on the same pair and give misleading WR figures.
- **get_pair_config**: always set strategy= to the signal's strategy. Each strategy has its own
  config table — calling without strategy, or with the wrong strategy, returns wrong or empty data.
  Note: FA_OR_ATR_TRAIL uses ATR-based stops so fixed_stop_loss_pips will be null — check monitor_only and is_enabled instead.
- **rejection_density**: only meaningful for SMC_SIMPLE signals — the rejection table is not
  populated by other strategies. Skip this tool for XAU_GOLD, IMPULSE_FADE, DONCHIAN_TURTLE,
  MEAN_REVERSION, RANGE_FADE, FA_OR_ATR_TRAIL signals.

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


def _build_signal_prompt(signal: Dict) -> str:
    """Convert signal dict into a concise user-turn message for the agent."""
    epic = signal.get("epic", "Unknown")
    strategy = signal.get("strategy", "Unknown")
    signal_type = signal.get("signal_type", signal.get("type", "Unknown"))
    confidence = signal.get("confidence", 0)
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

    # Include key indicator values if present
    for key in ("adx", "rsi", "macd_signal", "atr", "ema_bias", "htf_bias"):
        val = signal.get(key)
        if val is not None:
            lines.append(f"{key.upper()}: {val}")

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
