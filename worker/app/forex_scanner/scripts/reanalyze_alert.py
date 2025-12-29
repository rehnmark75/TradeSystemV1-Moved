#!/usr/bin/env python3
"""
Reanalyze an alert with the current prompt builder settings.
Usage: python reanalyze_alert.py <alert_id>
"""

import sys
import json
import os

# Add forex_scanner to path
sys.path.insert(0, '/app/forex_scanner')
sys.path.insert(0, '/app')

from sqlalchemy import create_engine, text
from alerts.analysis.prompt_builder import PromptBuilder
try:
    from alerts.claude_api import ClaudeAnalyzer
    CLAUDE_API_AVAILABLE = True
except ImportError:
    CLAUDE_API_AVAILABLE = False
    ClaudeAnalyzer = None

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")


def get_alert_data(alert_id: int) -> dict:
    """Fetch alert data from database."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                id, epic, signal_type, confidence_score, price,
                strategy, strategy_indicators, claude_decision, claude_score,
                claude_analysis
            FROM alert_history
            WHERE id = :alert_id
        """), {"alert_id": alert_id})
        row = result.fetchone()
        if not row:
            return None
        return dict(row._mapping)


def build_signal_from_alert(alert: dict) -> dict:
    """Convert alert data to signal format for prompt builder."""
    strategy_indicators = alert.get('strategy_indicators') or {}
    if isinstance(strategy_indicators, str):
        strategy_indicators = json.loads(strategy_indicators)

    # Extract risk management data
    risk_mgmt = strategy_indicators.get('risk_management', {})
    tier3 = strategy_indicators.get('tier3_entry', {})

    signal = {
        'epic': alert['epic'],
        'signal_type': alert['signal_type'],
        'price': float(alert['price']),
        'entry_price': tier3.get('entry_price', float(alert['price'])),
        'confidence_score': float(alert['confidence_score']),
        'strategy': alert['strategy'],
        'stop_loss': risk_mgmt.get('stop_loss', 0),
        'take_profit': risk_mgmt.get('take_profit', 0),
        'risk_pips': risk_mgmt.get('risk_pips', 0),
        'reward_pips': risk_mgmt.get('reward_pips', 0),
        'rr_ratio': risk_mgmt.get('rr_ratio', 0),
        'strategy_indicators': strategy_indicators,
        'entry_type': tier3.get('entry_type', 'PULLBACK'),
    }
    return signal


def reanalyze_alert(alert_id: int, call_claude: bool = False):
    """Reanalyze an alert with the current prompt builder."""
    print(f"\n{'='*80}")
    print(f"REANALYZING ALERT {alert_id}")
    print('='*80)

    # Get alert data
    alert = get_alert_data(alert_id)
    if not alert:
        print(f"❌ Alert {alert_id} not found")
        return

    print(f"\n📊 Original Analysis:")
    print(f"   Epic: {alert['epic']}")
    print(f"   Direction: {alert['signal_type']}")
    print(f"   Confidence: {float(alert['confidence_score']):.1%}")
    print(f"   Claude Decision: {alert['claude_decision']} (Score: {alert['claude_score']})")
    print(f"   Reason: {alert['claude_analysis']}")

    # Build signal and prompt
    signal = build_signal_from_alert(alert)
    builder = PromptBuilder()
    prompt = builder._build_smc_prompt(signal, has_chart=False)

    print(f"\n📝 Updated Prompt Preview:")
    print("-" * 40)

    # Show trade levels section
    if 'TRADE LEVELS' in prompt:
        start = prompt.find('TRADE LEVELS')
        end = prompt.find('##', start + 20) if '##' in prompt[start+20:start+500] else start + 400
        print(prompt[start:end].strip())

    # Check for fixed note
    if 'FIXED SL/TP MODE' in prompt:
        print("\n✅ Fixed SL/TP mode note is present")
    else:
        print("\n⚠️ Fixed SL/TP mode note NOT present")

    if call_claude:
        if not CLAUDE_API_AVAILABLE:
            print(f"\n❌ Claude API client not available")
        else:
            print(f"\n🤖 Calling Claude API for reanalysis...")
            try:
                import config
                import requests
                api_key = getattr(config, 'CLAUDE_API_KEY', None) or os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    print(f"❌ No Claude API key found")
                else:
                    # Call Claude API directly with the prompt
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    }
                    data = {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 150,
                        "messages": [{"role": "user", "content": prompt}]
                    }

                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=data,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        content = result.get('content', [])
                        if content:
                            raw_response = content[0].get('text', '')
                            print(f"\n📊 NEW Claude Analysis:")
                            print(f"   Raw Response: {raw_response}")

                            # Parse SCORE, DECISION, REASON
                            import re
                            score_match = re.search(r'SCORE:\s*(\d+)', raw_response)
                            decision_match = re.search(r'DECISION:\s*(APPROVE|REJECT)', raw_response)
                            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', raw_response, re.DOTALL)

                            print(f"\n   Parsed:")
                            print(f"   - Score: {score_match.group(1) if score_match else 'N/A'}")
                            print(f"   - Decision: {decision_match.group(1) if decision_match else 'N/A'}")
                            print(f"   - Reason: {reason_match.group(1).strip() if reason_match else 'N/A'}")
                    else:
                        print(f"❌ API Error: {response.status_code} - {response.text}")
            except Exception as e:
                import traceback
                print(f"❌ Claude API error: {e}")
                traceback.print_exc()
    else:
        print(f"\n💡 To call Claude API, run with --call-claude flag")

    return prompt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reanalyze_alert.py <alert_id> [--call-claude]")
        sys.exit(1)

    alert_id = int(sys.argv[1])
    call_claude = '--call-claude' in sys.argv

    reanalyze_alert(alert_id, call_claude)
