from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from models import EconomicEvent, ScrapeLog, NewsImpactAnalysis, ImpactLevel, ScrapeStatus
from database.connection import db_manager
from scraper.scheduler import economic_scheduler
from config import config

logger = logging.getLogger(__name__)

# Create router
api_router = APIRouter()


def get_db():
    """Dependency to get database session"""
    with db_manager.get_session() as session:
        yield session


@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_health = db_manager.get_health_status()

        # Check scheduler status
        scheduler_status = economic_scheduler.get_job_status()

        # Determine overall health
        is_healthy = (
            db_health.get('status') == 'healthy' and
            scheduler_status.get('status') == 'running'
        )

        status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

        return JSONResponse(
            content={
                'status': 'healthy' if is_healthy else 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0',
                'service': 'economic-calendar',
                'database': db_health,
                'scheduler': scheduler_status
            },
            status_code=status_code
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@api_router.get("/status")
async def get_service_status(db: Session = Depends(get_db)):
    """Get detailed service status and metrics"""
    try:
        # Get latest scrape logs
        latest_scrapes = db.query(ScrapeLog).order_by(desc(ScrapeLog.scrape_date)).limit(5).all()

        # Get event counts by currency
        event_counts = db.query(EconomicEvent.currency, func.count(EconomicEvent.id)).group_by(
            EconomicEvent.currency
        ).all()

        # Get upcoming events count
        upcoming_events_count = db.query(EconomicEvent).filter(
            EconomicEvent.event_date >= datetime.utcnow()
        ).count()

        # Get next scheduled scrape
        next_scrape = economic_scheduler.get_next_scheduled_scrape()

        return {
            'service': 'economic-calendar',
            'status': 'operational',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'upcoming_events': upcoming_events_count,
                'events_by_currency': {currency: count for currency, count in event_counts},
                'recent_scrapes': len(latest_scrapes)
            },
            'scheduler': {
                'next_scrape': next_scrape.isoformat() if next_scrape else None,
                'jobs': economic_scheduler.get_job_status()
            },
            'recent_scrapes': [
                {
                    'id': log.id,
                    'date': log.scrape_date.isoformat(),
                    'status': log.status.value,
                    'events_found': log.events_found,
                    'duration': log.duration_seconds
                }
                for log in latest_scrapes
            ]
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/events")
async def get_economic_events(
    currency: Optional[str] = Query(None, description="Filter by currency (e.g., USD, EUR)"),
    impact: Optional[str] = Query(None, description="Filter by impact level (low, medium, high)"),
    from_date: Optional[datetime] = Query(None, description="Start date filter"),
    to_date: Optional[datetime] = Query(None, description="End date filter"),
    upcoming_only: bool = Query(False, description="Show only upcoming events"),
    limit: int = Query(100, le=1000, description="Maximum number of events to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """Get economic events with filtering options"""
    try:
        query = db.query(EconomicEvent)

        # Apply filters
        if currency:
            query = query.filter(EconomicEvent.currency == currency.upper())

        if impact:
            try:
                impact_level = ImpactLevel(impact.lower())
                query = query.filter(EconomicEvent.impact_level == impact_level)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid impact level: {impact}")

        if from_date:
            query = query.filter(EconomicEvent.event_date >= from_date)

        if to_date:
            query = query.filter(EconomicEvent.event_date <= to_date)

        if upcoming_only:
            query = query.filter(EconomicEvent.event_date >= datetime.utcnow())

        # Order by date
        query = query.order_by(EconomicEvent.event_date)

        # Apply pagination
        total_count = query.count()
        events = query.offset(offset).limit(limit).all()

        return {
            'events': [
                {
                    'id': event.id,
                    'event_name': event.event_name,
                    'currency': event.currency,
                    'country': event.country,
                    'event_date': event.event_date.isoformat(),
                    'event_time': event.event_time,
                    'impact_level': event.impact_level.value,
                    'previous_value': event.previous_value,
                    'forecast_value': event.forecast_value,
                    'actual_value': event.actual_value,
                    'category': event.category,
                    'source': event.source,
                    'market_moving': event.market_moving,
                    'created_at': event.created_at.isoformat()
                }
                for event in events
            ],
            'pagination': {
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total_count
            }
        }

    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/events/upcoming")
async def get_upcoming_events(
    hours: int = Query(24, description="Hours ahead to look for events"),
    impact_level: Optional[str] = Query(None, description="Minimum impact level"),
    currencies: Optional[str] = Query(None, description="Comma-separated list of currencies"),
    db: Session = Depends(get_db)
):
    """Get upcoming events within specified time window"""
    try:
        end_time = datetime.utcnow() + timedelta(hours=hours)

        query = db.query(EconomicEvent).filter(
            and_(
                EconomicEvent.event_date >= datetime.utcnow(),
                EconomicEvent.event_date <= end_time
            )
        )

        # Filter by impact level
        if impact_level:
            try:
                min_impact = ImpactLevel(impact_level.lower())
                if min_impact == ImpactLevel.HIGH:
                    query = query.filter(EconomicEvent.impact_level == ImpactLevel.HIGH)
                elif min_impact == ImpactLevel.MEDIUM:
                    query = query.filter(EconomicEvent.impact_level.in_([ImpactLevel.MEDIUM, ImpactLevel.HIGH]))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid impact level: {impact_level}")

        # Filter by currencies
        if currencies:
            currency_list = [c.strip().upper() for c in currencies.split(',')]
            query = query.filter(EconomicEvent.currency.in_(currency_list))

        events = query.order_by(EconomicEvent.event_date).all()

        return {
            'upcoming_events': [
                {
                    'id': event.id,
                    'event_name': event.event_name,
                    'currency': event.currency,
                    'event_date': event.event_date.isoformat(),
                    'impact_level': event.impact_level.value,
                    'forecast_value': event.forecast_value,
                    'previous_value': event.previous_value,
                    'time_until_event': str(event.event_date - datetime.utcnow()),
                    'market_moving': event.market_moving
                }
                for event in events
            ],
            'query_params': {
                'hours_ahead': hours,
                'impact_level': impact_level,
                'currencies': currencies
            },
            'count': len(events)
        }

    except Exception as e:
        logger.error(f"Failed to get upcoming events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/scrape/manual")
async def trigger_manual_scrape(
    week_offset: int = Query(0, description="Week offset (0=current, 1=next, -1=previous)")
):
    """Trigger manual scrape of economic calendar"""
    if not config.ENABLE_MANUAL_SCRAPE:
        raise HTTPException(status_code=403, detail="Manual scrape is disabled")

    try:
        result = await economic_scheduler.manual_scrape(week_offset=week_offset)

        status_code = status.HTTP_200_OK if result['success'] else status.HTTP_500_INTERNAL_SERVER_ERROR

        return JSONResponse(content=result, status_code=status_code)

    except Exception as e:
        logger.error(f"Manual scrape failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/scrape/logs")
async def get_scrape_logs(
    limit: int = Query(20, le=100, description="Maximum number of logs to return"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """Get scrape operation logs"""
    try:
        query = db.query(ScrapeLog)

        if status_filter:
            try:
                status_enum = ScrapeStatus(status_filter.lower())
                query = query.filter(ScrapeLog.status == status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status_filter}")

        logs = query.order_by(desc(ScrapeLog.scrape_date)).limit(limit).all()

        return {
            'scrape_logs': [
                {
                    'id': log.id,
                    'scrape_date': log.scrape_date.isoformat(),
                    'data_source': log.data_source,
                    'scrape_type': log.scrape_type,
                    'status': log.status.value,
                    'events_found': log.events_found,
                    'events_new': log.events_new,
                    'events_updated': log.events_updated,
                    'events_failed': log.events_failed,
                    'duration_seconds': log.duration_seconds,
                    'error_message': log.error_message,
                    'date_range': {
                        'from': log.date_from.isoformat() if log.date_from else None,
                        'to': log.date_to.isoformat() if log.date_to else None
                    }
                }
                for log in logs
            ],
            'count': len(logs)
        }

    except Exception as e:
        logger.error(f"Failed to get scrape logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/events/currency/{currency}")
async def get_events_by_currency(
    currency: str,
    days_ahead: int = Query(7, description="Days ahead to look for events"),
    impact_level: Optional[str] = Query(None, description="Minimum impact level"),
    db: Session = Depends(get_db)
):
    """Get events for a specific currency"""
    try:
        currency = currency.upper()
        if currency not in config.FOCUS_CURRENCIES:
            raise HTTPException(status_code=400, detail=f"Currency {currency} not supported")

        end_date = datetime.utcnow() + timedelta(days=days_ahead)

        query = db.query(EconomicEvent).filter(
            and_(
                EconomicEvent.currency == currency,
                EconomicEvent.event_date >= datetime.utcnow(),
                EconomicEvent.event_date <= end_date
            )
        )

        if impact_level:
            try:
                min_impact = ImpactLevel(impact_level.lower())
                if min_impact == ImpactLevel.HIGH:
                    query = query.filter(EconomicEvent.impact_level == ImpactLevel.HIGH)
                elif min_impact == ImpactLevel.MEDIUM:
                    query = query.filter(EconomicEvent.impact_level.in_([ImpactLevel.MEDIUM, ImpactLevel.HIGH]))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid impact level: {impact_level}")

        events = query.order_by(EconomicEvent.event_date).all()

        return {
            'currency': currency,
            'events': [
                {
                    'id': event.id,
                    'event_name': event.event_name,
                    'event_date': event.event_date.isoformat(),
                    'impact_level': event.impact_level.value,
                    'forecast_value': event.forecast_value,
                    'previous_value': event.previous_value,
                    'actual_value': event.actual_value,
                    'market_moving': event.market_moving
                }
                for event in events
            ],
            'query_params': {
                'days_ahead': days_ahead,
                'impact_level': impact_level
            },
            'count': len(events)
        }

    except Exception as e:
        logger.error(f"Failed to get events for currency {currency}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/scheduler/pause/{job_id}")
async def pause_scheduler_job(job_id: str):
    """Pause a scheduled job"""
    success = economic_scheduler.pause_job(job_id)

    if success:
        return {'message': f'Job {job_id} paused successfully', 'success': True}
    else:
        raise HTTPException(status_code=400, detail=f'Failed to pause job {job_id}')


@api_router.post("/scheduler/resume/{job_id}")
async def resume_scheduler_job(job_id: str):
    """Resume a scheduled job"""
    success = economic_scheduler.resume_job(job_id)

    if success:
        return {'message': f'Job {job_id} resumed successfully', 'success': True}
    else:
        raise HTTPException(status_code=400, detail=f'Failed to resume job {job_id}')


@api_router.get("/events/impact-analysis")
async def get_market_impact_analysis(
    currency_pair: Optional[str] = Query(None, description="Currency pair (e.g., EURUSD)"),
    days_back: int = Query(30, description="Days back to analyze"),
    db: Session = Depends(get_db)
):
    """Get market impact analysis for economic events"""
    if not config.ENABLE_NEWS_IMPACT_ANALYSIS:
        raise HTTPException(status_code=403, detail="News impact analysis is disabled")

    try:
        start_date = datetime.utcnow() - timedelta(days=days_back)

        query = db.query(NewsImpactAnalysis).filter(
            NewsImpactAnalysis.created_at >= start_date
        )

        if currency_pair:
            query = query.filter(NewsImpactAnalysis.currency_pair == currency_pair.upper())

        analyses = query.order_by(desc(NewsImpactAnalysis.created_at)).limit(100).all()

        return {
            'impact_analyses': [
                {
                    'id': analysis.id,
                    'economic_event_id': analysis.economic_event_id,
                    'currency_pair': analysis.currency_pair,
                    'impact_detected': analysis.impact_detected,
                    'volatility_increase': analysis.volatility_increase,
                    'price_movement_pips': analysis.price_movement_pips,
                    'impact_score': analysis.impact_score,
                    'surprise_factor': analysis.surprise_factor,
                    'created_at': analysis.created_at.isoformat()
                }
                for analysis in analyses
            ],
            'count': len(analyses),
            'query_params': {
                'currency_pair': currency_pair,
                'days_back': days_back
            }
        }

    except Exception as e:
        logger.error(f"Failed to get impact analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))