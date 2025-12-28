"""
Rejection Outcome Analysis Router

FastAPI router for SMC rejection outcome endpoints.
Provides data to the Unified Analytics dashboard for analyzing
what would have happened if rejected signals were executed.

Created: 2025-12-28
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
import logging

from services.db import get_db
from services.rejection_outcome_service import RejectionOutcomeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rejection-outcomes", tags=["Rejection Outcome Analysis"])


@router.get("/summary")
async def get_outcome_summary(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get high-level outcome summary statistics.

    Returns aggregated metrics showing:
    - Total rejections analyzed
    - Would-be winners and losers
    - Win rate, missed profit, avoided loss
    - MFE/MAE averages
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_outcome_summary(days)
    except Exception as e:
        logger.error(f"Error getting outcome summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/win-rate-by-stage")
async def get_win_rate_by_stage(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get win rate breakdown by rejection stage.

    Shows which rejection stages are:
    - Too aggressive (rejecting profitable signals)
    - Working correctly (filtering losing signals)
    - Neutral (balanced performance)
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_win_rate_by_stage(days)
    except Exception as e:
        logger.error(f"Error getting win rate by stage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/missed-profit")
async def get_missed_profit_analysis(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    pair: Optional[str] = Query(None, description="Filter by currency pair (e.g., EURUSD)"),
    db: Session = Depends(get_db)
):
    """
    Get missed profit analysis with breakdowns by pair and stage.

    Shows:
    - Total potential profit missed from rejected winners
    - Loss avoided from rejected losers
    - Net impact of rejection filters
    - Breakdown by pair and rejection stage
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_missed_profit_analysis(days, pair)
    except Exception as e:
        logger.error(f"Error getting missed profit analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-session")
async def get_outcome_by_session(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get outcome breakdown by market session.

    Shows win rate patterns across:
    - London session
    - New York session
    - Asian session
    - Overlap sessions
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_outcome_by_session(days)
    except Exception as e:
        logger.error(f"Error getting outcome by session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-hour")
async def get_outcome_by_hour(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get outcome breakdown by hour of day (UTC).

    Identifies which hours have:
    - Higher win rate for rejected signals
    - Lower win rate for rejected signals
    - Optimal trading windows
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_outcome_by_hour(days)
    except Exception as e:
        logger.error(f"Error getting outcome by hour: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-pair")
async def get_outcome_by_pair(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get outcome breakdown by currency pair (epic).

    Shows for each pair:
    - Win rate and total analyzed
    - Missed profit and avoided loss
    - Status (TOO_AGGRESSIVE, WORKING_WELL, NEUTRAL)
    - Pair-specific recommendations
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_outcome_by_pair(days)
    except Exception as e:
        logger.error(f"Error getting outcome by pair: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pair-stage-breakdown")
async def get_pair_stage_breakdown(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    pair: Optional[str] = Query(None, description="Filter by currency pair (e.g., EURUSD)"),
    db: Session = Depends(get_db)
):
    """
    Get detailed breakdown by pair and rejection stage.

    Useful for analyzing specific pairs to understand which stages
    are causing issues for particular currency pairs.
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_pair_stage_breakdown(days, pair)
    except Exception as e:
        logger.error(f"Error getting pair stage breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mfe-mae")
async def get_mfe_mae_analysis(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    stage: Optional[str] = Query(None, description="Filter by rejection stage"),
    db: Session = Depends(get_db)
):
    """
    Get Maximum Favorable/Adverse Excursion analysis.

    Shows for both winners and losers:
    - Average and max MFE (how far price moved favorably)
    - Average and max MAE (how far price moved adversely)
    - Time to MFE/MAE (how long to reach extremes)
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_mfe_mae_analysis(days, stage)
    except Exception as e:
        logger.error(f"Error getting MFE/MAE analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/parameter-suggestions")
async def get_parameter_suggestions(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get AI-ready parameter suggestions based on outcome analysis.

    Generates actionable recommendations:
    - Which filters are too aggressive
    - Which filters are working well
    - Session-based patterns
    - Overall assessment of filter effectiveness
    """
    try:
        service = RejectionOutcomeService(db)
        return service.get_parameter_suggestions(days)
    except Exception as e:
        logger.error(f"Error getting parameter suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def get_recent_outcomes(
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records"),
    stage: Optional[str] = Query(None, description="Filter by rejection stage"),
    outcome: Optional[str] = Query(None, description="Filter by outcome (HIT_TP, HIT_SL, etc.)"),
    pair: Optional[str] = Query(None, description="Filter by currency pair"),
    db: Session = Depends(get_db)
):
    """
    Get recent individual outcome records with optional filters.

    Returns detailed outcome data for individual rejections:
    - Entry, SL, TP prices
    - Outcome and outcome price
    - Time to outcome
    - MFE/MAE values
    """
    try:
        service = RejectionOutcomeService(db)
        filters = {}
        if stage:
            filters['stage'] = stage
        if outcome:
            filters['outcome'] = outcome
        if pair:
            filters['pair'] = pair
        return service.get_recent_outcomes(limit, filters if filters else None)
    except Exception as e:
        logger.error(f"Error getting recent outcomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for rejection outcome analysis service"""
    return {
        "status": "healthy",
        "service": "rejection-outcome-analysis",
        "version": "1.0",
        "description": "Analyzes SMC Simple rejected signals to determine if they would have been profitable"
    }
