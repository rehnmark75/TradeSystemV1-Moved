"""Resume rule R1 (Phase 3).

Pure decision over a paused cell's shadow stats. Phase 3 is PROPOSE-ONLY: a
True result is LOGGED as a proposal, not acted on. R1:

    propose resume when
        >= resume_min_signals fresh reconstructed outcomes   AND
        >= resume_cooldown_days since pause                   AND
        shadow PF > resume_pf_threshold (hysteresis gap above the 0.8 trip)

The hysteresis gap (trip < 0.8, resume > 1.1) plus the cooldown and the
fresh-signal minimum are what prevented flip-flop in validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import AutoPauseParams
from .evaluator import PerfStats


@dataclass(frozen=True)
class ResumeProposal:
    should_propose: bool
    reason: str


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def evaluate_resume(
    paused_at: datetime,
    shadow_stats: PerfStats,
    n_resolved: int,
    params: AutoPauseParams,
    now: datetime,
) -> ResumeProposal:
    days_paused = (_naive(now) - _naive(paused_at)).total_seconds() / 86400.0

    if n_resolved < params.resume_min_signals:
        return ResumeProposal(
            False,
            f"insufficient shadow outcomes ({n_resolved}/{params.resume_min_signals})",
        )
    if days_paused < params.resume_cooldown_days:
        return ResumeProposal(
            False,
            f"cooldown not met ({days_paused:.1f}/{params.resume_cooldown_days}d)",
        )
    pf = shadow_stats.pf
    if pf is None or pf <= params.resume_pf_threshold:
        pf_txt = "n/a" if pf is None else f"{pf:.2f}"
        return ResumeProposal(
            False,
            f"shadow PF {pf_txt} <= resume threshold {params.resume_pf_threshold}",
        )
    return ResumeProposal(
        True,
        f"shadow PF {pf:.2f} > {params.resume_pf_threshold} over {n_resolved} "
        f"outcomes, {days_paused:.0f}d paused",
    )
