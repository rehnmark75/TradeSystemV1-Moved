#!/usr/bin/env python3
"""
Real-time Stream vs API Validation Service

Continuously validates streamed candle data against IG REST API to ensure data integrity.
Detects discrepancies immediately and provides confidence scoring for trading systems.
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from collections import deque
import json
import math

from services.db import SessionLocal
from services.models import IGCandle
from igstream.ig_auth_prod import ig_login
import config
from config import IG_API_BASE_URL

logger = logging.getLogger(__name__)

@dataclass
class ValidationRequest:
    """Request for candle validation against API"""
    epic: str
    timeframe: int
    timestamp: datetime
    streamed_ohlc: Dict[str, float]  # open, high, low, close from stream
    data_source: str
    request_time: datetime
    priority: int = 1  # 1=normal, 2=high priority

@dataclass 
class ValidationResult:
    """Result of stream vs API validation"""
    epic: str
    timeframe: int
    timestamp: datetime
    validation_time: datetime
    
    # Price comparisons
    stream_close: float
    api_close: float
    price_diff_pips: float
    
    # Validation outcome
    is_valid: bool
    confidence_score: float
    discrepancy_level: str  # 'NONE', 'MINOR', 'MODERATE', 'MAJOR', 'CRITICAL'
    
    # Additional context
    api_data_available: bool
    validation_error: Optional[str] = None

class StreamValidatorService:
    """Service to validate streamed candles against IG REST API"""
    
    def market_is_open(self) -> bool:
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
    
    def __init__(self, headers: Dict[str, str]):
        self.headers = headers
        self.api_base_url = IG_API_BASE_URL
        
        # Validation queue and processing
        self.validation_queue = deque()
        self.processing = False
        self.max_queue_size = 1000
        
        # Rate limiting
        self.requests_per_minute = 80  # Conservative limit
        self.request_timestamps = deque()
        
        # Configuration
        self.validation_delay = getattr(config, 'STREAM_VALIDATION_DELAY_SECONDS', 45)
        self.validation_enabled = getattr(config, 'ENABLE_STREAM_API_VALIDATION', True)
        self.validation_frequency = getattr(config, 'STREAM_VALIDATION_FREQUENCY', 5)  # Every 5th candle
        
        # Thresholds for discrepancy classification
        self.thresholds = {
            'MINOR': 1.0,      # 1 pip
            'MODERATE': 3.0,   # 3 pips  
            'MAJOR': 10.0,     # 10 pips
            'CRITICAL': 25.0   # 25 pips
        }
        
        # Statistics
        self.stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'discrepancies_found': 0,
            'api_errors': 0,
            'corrections_made': 0
        }
    
    async def start_validation_worker(self):
        """Start the async validation worker"""
        if not self.validation_enabled:
            logger.info("🔍 Stream validation disabled in configuration")
            return
            
        logger.info("🚀 Starting stream validation service...")
        logger.info(f"   Validation delay: {self.validation_delay}s")
        logger.info(f"   Validation frequency: every {self.validation_frequency} candles")
        logger.info(f"   Rate limit: {self.requests_per_minute} requests/minute")
        
        self.processing = True
        
        while self.processing:
            try:
                await self._process_validation_queue()
                await asyncio.sleep(1)  # Check queue every second
                
            except Exception as e:
                logger.error(f"❌ Error in validation worker: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def queue_candle_for_validation(self, 
                                  epic: str, 
                                  timeframe: int, 
                                  timestamp: datetime,
                                  streamed_ohlc: Dict[str, float],
                                  data_source: str = 'chart_streamer',
                                  priority: int = 1):
        """Add candle to validation queue"""
        
        if not self.validation_enabled:
            return
            
        # Skip validation when market is closed to reduce log noise
        if not self.market_is_open():
            logger.debug(f"Skipping validation for {epic} - market closed")
            return
            
        # Skip validation based on frequency (except high priority)
        if priority == 1 and self.stats['total_validations'] % self.validation_frequency != 0:
            return
            
        # Check queue size
        if len(self.validation_queue) >= self.max_queue_size:
            logger.warning(f"⚠️ Validation queue full ({self.max_queue_size}), dropping oldest request")
            self.validation_queue.popleft()
        
        request = ValidationRequest(
            epic=epic,
            timeframe=timeframe,
            timestamp=timestamp,
            streamed_ohlc=streamed_ohlc,
            data_source=data_source,
            request_time=datetime.utcnow(),
            priority=priority
        )
        
        self.validation_queue.append(request)
        logger.debug(f"🔍 Queued {epic} {timeframe}m candle for validation (queue size: {len(self.validation_queue)})")
    
    async def _process_validation_queue(self):
        """Process pending validation requests"""
        
        if not self.validation_queue:
            return
            
        # Get next request (prioritize high priority requests)
        request = None
        for i, req in enumerate(self.validation_queue):
            # Only validate completed candles - must be older than the timeframe period
            # For 60-min candles, wait at least 65 minutes; for 5-min candles, wait 10 minutes
            min_age_required = req.timeframe + 5  # Add 5 minute buffer
            candle_age_minutes = (datetime.utcnow() - req.timestamp).total_seconds() / 60
            candle_old_enough = candle_age_minutes >= min_age_required
            
            time_delay_met = (datetime.utcnow() - req.request_time).seconds >= self.validation_delay
            
            if candle_old_enough and (req.priority > 1 or time_delay_met):
                request = self.validation_queue[i]
                del self.validation_queue[i]
                break
        
        if not request:
            return  # No requests ready for processing
            
        # Check rate limit
        if not self._check_rate_limit():
            # Put request back in queue
            self.validation_queue.appendleft(request) 
            await asyncio.sleep(10)  # Wait before retrying
            return
        
        # Perform validation
        try:
            result = await self._validate_candle_against_api(request)
            await self._process_validation_result(result)
            
        except Exception as e:
            logger.error(f"❌ Validation failed for {request.epic}: {e}")
            self.stats['api_errors'] += 1
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within API rate limits"""
        now = datetime.utcnow()
        
        # Remove timestamps older than 1 minute
        while self.request_timestamps and (now - self.request_timestamps[0]).seconds >= 60:
            self.request_timestamps.popleft()
        
        # Check if we can make another request
        if len(self.request_timestamps) >= self.requests_per_minute:
            return False
            
        self.request_timestamps.append(now)
        return True
    
    async def _validate_candle_against_api(self, request: ValidationRequest) -> ValidationResult:
        """Fetch candle from API and compare with streamed data"""
        
        try:
            # Convert timeframe to IG resolution using correct API values
            resolution_map = {
                5: "MINUTE_5",
                15: "MINUTE_15", 
                60: "HOUR",
                240: "HOUR_4",
                1440: "DAY"
            }
            
            resolution = resolution_map.get(request.timeframe, "MINUTE_5")
            possible_resolutions = [resolution]
            
            # Calculate API request time range aligned to candle boundaries
            # For 5-minute candles: align to 00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
            # For 15-minute candles: align to 00, 15, 30, 45
            # etc.

            # Round down to nearest candle boundary
            minutes_since_hour = request.timestamp.minute
            aligned_minute = (minutes_since_hour // request.timeframe) * request.timeframe

            from_time = request.timestamp.replace(minute=aligned_minute, second=0, microsecond=0)
            to_time = from_time + timedelta(minutes=request.timeframe)
            
            async with httpx.AsyncClient(base_url=self.api_base_url, headers=self.headers, timeout=10.0) as client:
                url = f"/prices/{request.epic}"
                
                # Try each resolution format until one works
                data = None
                successful_resolution = None
                last_error = None
                
                for resolution_attempt in possible_resolutions:
                    try:
                        # Format dates as IG API expects (no timezone info)
                        # Ensure we work with naive datetime objects
                        from_dt = from_time.replace(tzinfo=None) if from_time.tzinfo else from_time
                        to_dt = to_time.replace(tzinfo=None) if to_time.tzinfo else to_time
                        
                        from_str = from_dt.strftime("%Y-%m-%dT%H:%M:%S")
                        to_str = to_dt.strftime("%Y-%m-%dT%H:%M:%S")
                        
                        # Skip if requesting future data
                        now = datetime.utcnow()
                        if from_dt > now or to_dt > now:
                            logger.debug(f"Skipping future data validation for {request.epic}: {from_str} to {to_str} (current: {now.strftime('%Y-%m-%dT%H:%M:%S')})")
                            continue
                        
                        params = {
                            "resolution": resolution_attempt,
                            "from": from_str,
                            "to": to_str, 
                            "max": 2  # Only need 1-2 candles
                        }
                        
                        response = await client.get(url, params=params)
                        response.raise_for_status()
                        
                        data = response.json()
                        successful_resolution = resolution_attempt
                        logger.debug(f"✅ API validation successful using resolution '{resolution_attempt}' for {request.epic}")
                        break  # Success, exit loop
                        
                    except httpx.HTTPStatusError as e:
                        last_error = e
                        logger.debug(f"Resolution '{resolution_attempt}' failed for {request.epic}: {e.response.status_code}")
                        continue  # Try next resolution

                if data is None:
                    # Handle 404 errors gracefully - they indicate no data available, not a validation failure
                    if last_error and hasattr(last_error, 'response') and last_error.response.status_code == 404:
                        logger.debug(f"No API data available for {request.epic} (404 - instrument may be inactive)")
                        return ValidationResult(
                            epic=request.epic,
                            timeframe=request.timeframe,
                            timestamp=request.timestamp,
                            validation_time=datetime.utcnow(),
                            stream_close=request.streamed_ohlc['close'],
                            api_close=0.0,
                            price_diff_pips=0.0,
                            is_valid=True,  # Valid - 404 means no data, not incorrect data
                            confidence_score=0.8,  # Reduced confidence but still valid
                            discrepancy_level='NONE',
                            api_data_available=False
                        )
                    # For other errors, raise them
                    raise last_error or Exception("All resolution formats failed")
                
                candles = data.get("prices", [])
                
                if not candles:
                    return ValidationResult(
                        epic=request.epic,
                        timeframe=request.timeframe, 
                        timestamp=request.timestamp,
                        validation_time=datetime.utcnow(),
                        stream_close=request.streamed_ohlc['close'],
                        api_close=0.0,
                        price_diff_pips=0.0,
                        is_valid=True,  # Assume valid if no API data
                        confidence_score=0.8,  # Reduced confidence
                        discrepancy_level='NONE',
                        api_data_available=False
                    )
                
                # Find matching candle by timestamp  
                api_candle = None
                for candle in candles:
                    candle_time = datetime.fromisoformat(candle['snapshotTime'].replace('Z', '+00:00'))
                    if abs((candle_time - request.timestamp).total_seconds()) < 300:  # Within 5 minutes
                        api_candle = candle
                        break
                
                if not api_candle:
                    logger.warning(f"⚠️ No matching API candle found for {request.epic} at {request.timestamp}")
                    return ValidationResult(
                        epic=request.epic,
                        timeframe=request.timeframe,
                        timestamp=request.timestamp, 
                        validation_time=datetime.utcnow(),
                        stream_close=request.streamed_ohlc['close'],
                        api_close=0.0,
                        price_diff_pips=0.0,
                        is_valid=True,
                        confidence_score=0.8,
                        discrepancy_level='NONE',
                        api_data_available=False
                    )
                
                # Calculate mid price from API data
                api_bid_close = float(api_candle['closePrice']['bid'])
                api_ask_close = float(api_candle['closePrice']['ask'])  
                api_mid_close = (api_bid_close + api_ask_close) / 2
                
                # Compare with streamed data
                stream_close = request.streamed_ohlc['close']
                price_diff = abs(api_mid_close - stream_close)
                
                # Convert to pips
                if request.epic.find('JPY') != -1:
                    price_diff_pips = price_diff * 100  # JPY pairs
                else:
                    price_diff_pips = price_diff * 10000  # Major pairs
                
                # Classify discrepancy
                discrepancy_level = 'NONE'
                for level, threshold in sorted(self.thresholds.items(), key=lambda x: x[1]):
                    if price_diff_pips >= threshold:
                        discrepancy_level = level
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(price_diff_pips, discrepancy_level)
                
                # Determine if valid
                is_valid = discrepancy_level in ['NONE', 'MINOR', 'MODERATE']
                
                return ValidationResult(
                    epic=request.epic,
                    timeframe=request.timeframe,
                    timestamp=request.timestamp,
                    validation_time=datetime.utcnow(),
                    stream_close=stream_close,
                    api_close=api_mid_close,
                    price_diff_pips=price_diff_pips,
                    is_valid=is_valid,
                    confidence_score=confidence_score, 
                    discrepancy_level=discrepancy_level,
                    api_data_available=True
                )
                
        except Exception as e:
            # Only log validation errors during market hours to reduce noise
            if self.market_is_open():
                logger.error(f"❌ API validation error for {request.epic}: {e}")
            else:
                logger.debug(f"API validation error for {request.epic} (market closed): {e}")
            return ValidationResult(
                epic=request.epic,
                timeframe=request.timeframe,
                timestamp=request.timestamp,
                validation_time=datetime.utcnow(),
                stream_close=request.streamed_ohlc.get('close', 0.0),
                api_close=0.0,
                price_diff_pips=0.0,
                is_valid=False,
                confidence_score=0.0,
                discrepancy_level='NONE',
                api_data_available=False,
                validation_error=str(e)
            )
    
    def _calculate_confidence_score(self, price_diff_pips: float, discrepancy_level: str) -> float:
        """Calculate confidence score based on price difference"""
        
        if discrepancy_level == 'NONE':
            return 1.0
        elif discrepancy_level == 'MINOR':
            return 0.95
        elif discrepancy_level == 'MODERATE': 
            return 0.8
        elif discrepancy_level == 'MAJOR':
            return 0.5
        elif discrepancy_level == 'CRITICAL':
            return 0.2
        else:
            return 0.7  # Default
    
    async def _process_validation_result(self, result: ValidationResult):
        """Process validation result and take appropriate actions"""
        
        self.stats['total_validations'] += 1
        
        if result.api_data_available:
            self.stats['successful_validations'] += 1
        
        # Log result based on discrepancy level
        if result.discrepancy_level == 'CRITICAL':
            logger.critical(
                f"🚨 CRITICAL PRICE DISCREPANCY: {result.epic} "
                f"Stream={result.stream_close:.5f} vs API={result.api_close:.5f} "
                f"({result.price_diff_pips:.1f} pips difference)"
            )
            self.stats['discrepancies_found'] += 1
            
        elif result.discrepancy_level == 'MAJOR':
            logger.error(
                f"❌ MAJOR PRICE DISCREPANCY: {result.epic} "
                f"Stream={result.stream_close:.5f} vs API={result.api_close:.5f} "
                f"({result.price_diff_pips:.1f} pips difference)"
            )
            self.stats['discrepancies_found'] += 1
            
        elif result.discrepancy_level == 'MODERATE':
            logger.warning(
                f"⚠️ MODERATE PRICE DISCREPANCY: {result.epic} "
                f"Stream={result.stream_close:.5f} vs API={result.api_close:.5f} "
                f"({result.price_diff_pips:.1f} pips difference)"
            )
            self.stats['discrepancies_found'] += 1
            
        elif result.discrepancy_level == 'MINOR':
            logger.info(
                f"ℹ️ Minor price variance: {result.epic} "
                f"({result.price_diff_pips:.1f} pips difference)"
            )
            
        else:
            logger.debug(f"✅ Validation passed: {result.epic} (no discrepancy)")
        
        # Store validation result in database
        await self._store_validation_result(result)
        
        # If significant discrepancy, consider correcting the data
        if result.discrepancy_level in ['MAJOR', 'CRITICAL'] and result.api_data_available:
            await self._consider_price_correction(result)
    
    async def _store_validation_result(self, result: ValidationResult):
        """Store validation result in price_validation_log table"""
        
        try:
            # Determine severity level
            severity_map = {
                'NONE': 'INFO',
                'MINOR': 'INFO', 
                'MODERATE': 'WARNING',
                'MAJOR': 'CRITICAL',
                'CRITICAL': 'CRITICAL'
            }
            
            severity = severity_map.get(result.discrepancy_level, 'INFO')
            
            # Create message
            if result.api_data_available:
                message = f"Stream vs API validation: {result.price_diff_pips:.2f} pips difference (confidence: {result.confidence_score:.2f})"
            else:
                message = "API data unavailable for validation"
                
            if result.validation_error:
                message += f" | Error: {result.validation_error}"
            
            # Store in database
            with SessionLocal() as session:
                from sqlalchemy import text
                session.execute(
                    text("""
                    INSERT INTO price_validation_log 
                    (epic, timeframe, candle_time, validation_type, severity, message,
                     old_value, new_value, price_difference_pips, data_source, resolution)
                    VALUES (:epic, :timeframe, :candle_time, :validation_type, :severity, :message,
                     :old_value, :new_value, :price_difference_pips, :data_source, :resolution)
                    """),
                    {
                        "epic": result.epic,
                        "timeframe": result.timeframe, 
                        "candle_time": result.timestamp,
                        "validation_type": 'STREAM_API_VALIDATION',
                        "severity": severity,
                        "message": message,
                        "old_value": result.stream_close,
                        "new_value": result.api_close,
                        "price_difference_pips": result.price_diff_pips,
                        "data_source": 'stream_validator',
                        "resolution": 'VALIDATED' if result.is_valid else 'DISCREPANCY'
                    }
                )
                session.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to store validation result: {e}")
    
    async def _consider_price_correction(self, result: ValidationResult):
        """Consider correcting streamed price based on API data"""
        
        # Only correct for critical discrepancies and when we have high confidence in API data
        if result.discrepancy_level != 'CRITICAL' or not result.api_data_available:
            return
            
        try:
            # Update the database record with corrected price and reduced quality score
            with SessionLocal() as session:
                from sqlalchemy import text
                update_result = session.execute(
                    text("""
                    UPDATE ig_candles 
                    SET close = :close,
                        quality_score = :quality_score,
                        validation_flags = :validation_flags,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE epic = :epic 
                      AND timeframe = :timeframe 
                      AND start_time = :start_time
                      AND data_source = 'chart_streamer'
                    """),
                    {
                        "close": result.api_close,
                        "quality_score": result.confidence_score,
                        "validation_flags": '{"PRICE_CORRECTED_FROM_API"}',  # PostgreSQL array literal
                        "epic": result.epic,
                        "timeframe": result.timeframe,
                        "start_time": result.timestamp
                    }
                )
                
                if update_result.rowcount > 0:
                    session.commit()
                    self.stats['corrections_made'] += 1
                    logger.warning(
                        f"🔧 CORRECTED PRICE: {result.epic} at {result.timestamp} "
                        f"from {result.stream_close:.5f} to {result.api_close:.5f} "
                        f"({result.price_diff_pips:.1f} pips correction)"
                    )
                    
        except Exception as e:
            logger.error(f"❌ Failed to correct price for {result.epic}: {e}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        
        success_rate = 0.0
        if self.stats['total_validations'] > 0:
            success_rate = (self.stats['successful_validations'] / self.stats['total_validations']) * 100
            
        return {
            **self.stats,
            'queue_size': len(self.validation_queue),
            'success_rate_pct': round(success_rate, 2),
            'processing': self.processing
        }
    
    def stop(self):
        """Stop the validation service"""
        logger.info("🛑 Stopping stream validation service...")
        self.processing = False