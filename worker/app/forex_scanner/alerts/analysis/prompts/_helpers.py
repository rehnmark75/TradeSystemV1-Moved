"""Shared helpers for strategy prompt modules."""
from typing import Dict


def format_price(price) -> str:
    try:
        return f"{float(price):.5f}" if price else "N/A"
    except (ValueError, TypeError):
        return str(price) if price else "N/A"


def extract_pair(epic: str) -> str:
    try:
        parts = epic.split('.')
        if len(parts) >= 3:
            return parts[2]
        return epic
    except Exception:
        return epic


def build_fallback_prompt(signal: Dict) -> str:
    try:
        epic = str(signal.get('epic', 'Unknown'))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        return f"""FOREX SIGNAL ANALYSIS - FALLBACK MODE

Signal: {epic} {signal_type}

Instructions: Provide only:
SCORE: [0-10]
DECISION: [APPROVE/REJECT]
REASON_CODE: [APPROVE_CLEAN | APPROVE_WITH_PENALTY | REJECT]
REASON: [brief reason]"""
    except Exception:
        return """FOREX SIGNAL ANALYSIS - ERROR MODE

SCORE: 5
DECISION: NEUTRAL
REASON_CODE: APPROVE_CLEAN
REASON: Analysis error - neutral assessment"""
