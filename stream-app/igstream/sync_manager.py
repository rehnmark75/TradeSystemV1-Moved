import asyncio
import time
from igstream.chart_streamer import stream_chart_candles
from igstream.ig_auth_prod import ig_login
from igstream.gap_detector import GapDetector
from services.keyvault import get_secret
from services.db import SessionLocal
from services.models import IGCandle
from config import ACTIVE_EPICS
from sqlalchemy import desc
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Constants
STREAM_REFRESH_INTERVAL = 60         # How often to check stream status
AUTH_REFRESH_INTERVAL = 55 * 60      # IG tokens expire hourly
RECONNECT_BACKOFF = 60               # Minimum wait before reconnecting

# ✅ Active streaming timeframes (matches chart_streamer.py configuration)
# v2.17.0: Only 1m candles are streamed - all higher TFs synthesized from 1m base
ACTIVE_TIMEFRAMES = [1]  # Only 1m candles streamed

# Staleness thresholds per timeframe (in seconds) - only for active timeframes
STALENESS_THRESHOLDS = {
    1: 120,      # 1m candles: stale after 2 minutes
}

# ✅ Define the epics you want to stream (imported from config)
EPICS = ACTIVE_EPICS

stream_tasks = {}
last_reconnect_attempt = {}
stream_status = {}
headers = {}
gap_detector = GapDetector(max_gap_hours=6)  # Initialize gap detector

def market_is_open() -> bool:
    """Check if forex market is open"""
    now = datetime.now(timezone.utc)
    # IG closes Friday 21:00 UTC and reopens Sunday 21:00 UTC
    if now.weekday() == 5:  # Saturday
        return False
    if now.weekday() == 6 and now.hour < 21:  # Sunday before 21:00 UTC
        return False
    if now.weekday() == 4 and now.hour >= 21:  # Friday after 21:00 UTC
        return False
    return True

async def check_data_freshness(epic: str, timeframe: int = 5) -> bool:
    """
    Check if we have received fresh data for an epic within the threshold
    Returns True if data is fresh, False if stale
    Only checks timeframes that are actively being streamed
    """
    try:
        # Skip check if this timeframe is not actively streamed
        if timeframe not in ACTIVE_TIMEFRAMES:
            logger.debug(f"Skipping staleness check for {epic} {timeframe}m (not actively streamed)")
            return True  # Consider non-streamed timeframes as "fresh" to avoid false warnings
            
        with SessionLocal() as session:
            # Get the most recent candle for this epic and timeframe
            latest_candle = session.query(IGCandle).filter(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe
            ).order_by(desc(IGCandle.start_time)).first()
            
            if not latest_candle:
                logger.warning(f"No candle data found for {epic} {timeframe}m")
                return False
            
            # Check how old the latest data is
            now = datetime.now(timezone.utc)
            
            # Ensure latest_candle timestamp is timezone-aware
            latest_time = latest_candle.start_time
            if latest_time.tzinfo is None:
                latest_time = latest_time.replace(tzinfo=timezone.utc)
            
            time_since_last_candle = (now - latest_time).total_seconds()
            
            # Get appropriate threshold for this timeframe
            staleness_threshold = STALENESS_THRESHOLDS.get(timeframe, 300)  # Default to 5 minutes
            is_fresh = time_since_last_candle < staleness_threshold
            
            if not is_fresh:
                logger.warning(f"Data for {epic} {timeframe}m is stale. Last update: {latest_time} "
                             f"({time_since_last_candle:.0f} seconds ago, threshold: {staleness_threshold}s)")
            else:
                logger.debug(f"Data for {epic} {timeframe}m is fresh. Last update: {latest_time} "
                           f"({time_since_last_candle:.0f} seconds ago)")
            
            return is_fresh
            
    except Exception as e:
        logger.error(f"Error checking data freshness for {epic}: {e}")
        return False

async def check_all_epics_data_freshness() -> dict:
    """Check data freshness for all epics - only for actively streamed timeframes"""
    freshness_status = {}
    
    for epic in EPICS:
        # Only check timeframes that are actively being streamed
        timeframe_status = {}
        
        for timeframe in ACTIVE_TIMEFRAMES:
            timeframe_status[f"{timeframe}m"] = await check_data_freshness(epic, timeframe)
        
        # Consider overall healthy if primary timeframe (1m) is fresh
        primary_tf_fresh = timeframe_status.get("1m", False)

        freshness_status[epic] = {
            **timeframe_status,
            "overall": primary_tf_fresh  # 1m is the primary indicator of active streaming
        }
    
    return freshness_status

async def get_database_stats():
    """Get some basic stats about recent database activity"""
    try:
        with SessionLocal() as session:
            # Count recent candles (last 30 minutes)
            thirty_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=30)
            
            recent_count = session.query(IGCandle).filter(
                IGCandle.start_time >= thirty_minutes_ago
            ).count()
            
            # Get latest timestamp
            latest_candle = session.query(IGCandle).order_by(desc(IGCandle.start_time)).first()
            latest_time = latest_candle.start_time if latest_candle else None
            
            return {
                "recent_candles_30m": recent_count,
                "latest_candle_time": latest_time
            }
            
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"recent_candles_30m": 0, "latest_candle_time": None}

async def watchdog():
    """Enhanced watchdog that checks streams, data freshness, and gaps"""
    global last_reconnect_attempt
    
    logger.info("⏱️ Enhanced watchdog loop with gap detection started...")
    
    # Track gap detection runs
    last_gap_check = 0
    gap_check_interval = 300  # Check for gaps every 5 minutes
    
    while True:
        if not market_is_open():
            logger.info("Market is closed. Skipping stream and data checks.")
            await asyncio.sleep(STREAM_REFRESH_INTERVAL)
            continue

        now = time.time()
        
        # Get database stats
        db_stats = await get_database_stats()
        logger.info(f"📊 Database stats: {db_stats['recent_candles_30m']} candles in last 30min, "
                   f"latest: {db_stats['latest_candle_time']}")
        
        # Check data freshness for all epics
        freshness_status = await check_all_epics_data_freshness()
        
        # Periodically check for gaps and log them
        if now - last_gap_check > gap_check_interval:
            try:
                logger.info("🔍 Running gap detection check...")
                gap_stats = gap_detector.get_gap_statistics(EPICS)
                
                if gap_stats["total_gaps"] > 0:
                    logger.warning(f"⚠️ Gap Detection: Found {gap_stats['total_gaps']} gaps "
                                 f"({gap_stats['total_missing_candles']} missing candles)")
                    logger.warning(f"   Recent gaps: {gap_stats['recent_gaps']}, "
                                 f"Largest gap: {gap_stats['largest_gap_minutes']} minutes")
                    
                    # Log gaps by epic
                    for epic, epic_gaps in gap_stats["gaps_by_epic"].items():
                        if epic_gaps["missing_candles"] > 0:
                            logger.warning(f"   {epic}: {epic_gaps.get('1m', 0)} gaps in 1m "
                                         f"({epic_gaps['missing_candles']} candles)")
                else:
                    logger.info("✅ No gaps detected in candle data")
                
                last_gap_check = now
                
            except Exception as e:
                logger.error(f"Error in gap detection check: {e}")
        
        for epic in EPICS:
            # Check stream task status
            status_info = stream_status.get(epic, {})
            task_status = status_info.get("last_status", "UNKNOWN")
            is_running = status_info.get("running", False)
            
            # Check data freshness
            data_fresh = freshness_status.get(epic, {}).get("overall", False)
            
            last_attempt = last_reconnect_attempt.get(epic, 0)
            time_since_last_attempt = now - last_attempt

            # Determine if reconnect is needed
            stream_issues = not is_running or task_status in ["DISCONNECTED", "STALLED", "UNKNOWN"]
            data_issues = not data_fresh
            
            reconnect_needed = (stream_issues or data_issues) and time_since_last_attempt >= RECONNECT_BACKOFF
            
            if reconnect_needed:
                reason = []
                if stream_issues:
                    reason.append(f"stream_status={task_status}")
                if data_issues:
                    reason.append("stale_data")
                
                reason_str = ", ".join(reason)
                logger.warning(f"⚠️ Stream {epic} needs restart ({reason_str}). Reconnecting...")
                last_reconnect_attempt[epic] = now

                try:
                    await stop_stream(epic)
                    await asyncio.sleep(2)
                    await start_stream(epic)
                    logger.info(f"✅ Stream {epic} restart triggered due to: {reason_str}")
                except Exception as e:
                    logger.exception(f"❌ Failed to restart stream {epic}: {e}")
            else:
                # Log status for healthy streams
                if is_running and data_fresh:
                    logger.debug(f"✅ Stream {epic} is healthy (running={is_running}, data_fresh={data_fresh})")

        await asyncio.sleep(STREAM_REFRESH_INTERVAL)

async def start_stream(epic: str):
    """Start streaming for a specific epic"""
    if epic in stream_tasks and not stream_tasks[epic].done():
        logger.info(f"Stream for {epic} is already running. Skipping.")
        return

    global headers
    logger.info(f"🚀 Starting chart stream for {epic}")
    coroutine = stream_chart_candles(epic, headers)
    task = asyncio.create_task(coroutine)
    stream_tasks[epic] = task
    last_reconnect_attempt[epic] = time.time()
    stream_status[epic] = {"running": True, "last_status": "CONNECTED"}

    def handle_task_result(t):
        try:
            t.result()
        except asyncio.CancelledError:
            logger.info(f"Stream for {epic} cancelled.")
            stream_status[epic] = {"running": False, "last_status": "CANCELLED"}
        except Exception as e:
            logger.error(f"Stream for {epic} crashed: {e}")
            stream_status[epic] = {"running": False, "last_status": "CRASHED"}

    task.add_done_callback(handle_task_result)

async def stop_stream(epic: str):
    """Stop streaming for a specific epic"""
    task = stream_tasks.get(epic)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info(f"Stopped stream for {epic}")
    stream_tasks.pop(epic, None)
    stream_status[epic] = {"running": False, "last_status": "STOPPED"}

async def refresh_auth():
    """Refresh IG authentication tokens"""
    global headers
    logger.info("🔑 Refreshing IG session...")
    try:
        api_key = get_secret("prodapikey")
        ig_pwd = get_secret("prodpwd")
        ig_usr = "rehnmarkh"

        auth = await ig_login(api_key, ig_pwd, ig_usr)
        headers = {
            "CST": auth["CST"],
            "X-SECURITY-TOKEN": auth["X-SECURITY-TOKEN"],
            "accountId": auth["ACCOUNT_ID"]
        }
        logger.info("✅ Auth refreshed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to refresh auth: {e}")
        raise

async def auth_refresher():
    """Background task to refresh authentication periodically"""
    while True:
        try:
            await refresh_auth()
        except Exception as e:
            logger.error(f"Auth refresh failed: {e}")
        await asyncio.sleep(AUTH_REFRESH_INTERVAL)

async def start_all_streams():
    """Initialize and start all streams"""
    logger.info("🚀 Starting all forex streams...")
    
    # Initial auth
    await refresh_auth()

    # Start all streams
    for epic in EPICS:
        await start_stream(epic)
        await asyncio.sleep(1)  # Small delay between starts

    logger.info("✅ All streams started.")

    # Start background tasks
    asyncio.create_task(watchdog())
    asyncio.create_task(auth_refresher())
    logger.info("🔄 Watchdog and auth refresher started.")

async def stop_all_streams():
    """Stop all streams gracefully"""
    logger.info("🛑 Stopping all streams...")
    
    # Cancel all stream tasks
    for epic, task in stream_tasks.items():
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Stream for {epic} cancelled.")
    
    await asyncio.sleep(0.2)
    logger.info("✅ All streams shut down.")

# Additional utility functions for monitoring
async def get_stream_health_report():
    """Get a comprehensive health report of all streams"""
    if not market_is_open():
        return {"status": "market_closed", "details": "Forex market is currently closed"}
    
    report = {
        "market_open": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "streams": {},
        "database": await get_database_stats()
    }
    
    freshness_status = await check_all_epics_data_freshness()
    
    for epic in EPICS:
        task = stream_tasks.get(epic)
        status_info = stream_status.get(epic, {})
        
        # Build report based on active timeframes only
        epic_report = {
            "task_running": task is not None and not task.done() if task else False,
            "status": status_info.get("last_status", "UNKNOWN"),
            "overall_healthy": (
                task is not None and not task.done() and 
                freshness_status.get(epic, {}).get("overall", False)
            ) if task else False
        }
        
        # Add data freshness for active timeframes only
        for timeframe in ACTIVE_TIMEFRAMES:
            epic_report[f"data_fresh_{timeframe}m"] = freshness_status.get(epic, {}).get(f"{timeframe}m", False)
        
        report["streams"][epic] = epic_report
    
    return report
