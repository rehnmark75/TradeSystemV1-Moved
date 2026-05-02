"""
Manual trigger router — demo-only endpoint that runs strategy evaluation
against live candles via docker exec on the task-worker container.

Endpoint: POST /api/manual-trigger/evaluate
"""

import json
import subprocess
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

TRADING_ENVIRONMENT = os.getenv("TRADING_ENVIRONMENT", "demo")

router = APIRouter(prefix="/api/manual-trigger", tags=["manual-trigger"])


class EvaluateRequest(BaseModel):
    epic: str
    strategy: str = "SMC_SIMPLE"
    config_override: Dict[str, Any] = {}
    spread_pips: float = 1.5


@router.post("/evaluate")
def manual_evaluate(body: EvaluateRequest):
    """
    Evaluate a strategy against current live candles.
    Returns the signal (if any) and a ready-made TradeRequest payload.
    Demo environment only.
    """
    if TRADING_ENVIRONMENT != "demo":
        raise HTTPException(
            status_code=403,
            detail="Manual trigger is only available in the demo environment.",
        )

    cmd = [
        "docker", "exec", "task-worker",
        "python", "/app/forex_scanner/manual_evaluate.py",
        "--epic", body.epic,
        "--strategy", body.strategy.upper(),
        "--config-override", json.dumps(body.config_override),
        "--spread-pips", str(body.spread_pips),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Strategy evaluation timed out (>90 s).")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="docker command not available on this host.")

    stdout = result.stdout.strip()

    if not stdout:
        stderr_preview = result.stderr.strip()[:400] if result.stderr else ""
        raise HTTPException(
            status_code=500,
            detail=f"No output from evaluator. stderr: {stderr_preview}",
        )

    # The script may emit warning lines before the final JSON; take the last
    # line that looks like a JSON object.
    json_line: Optional[str] = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            json_line = line
            break

    if json_line is None:
        raise HTTPException(
            status_code=500,
            detail=f"Could not locate JSON in evaluator output: {stdout[:400]}",
        )

    try:
        return json.loads(json_line)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"JSON decode error: {exc}")
