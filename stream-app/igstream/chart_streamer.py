import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener
from services.db import SessionLocal
from services.models import IGCandle
import logging
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import the stream validator
from .stream_validator import StreamValidatorService
from config import LIGHTSTREAMER_PROD_URL, DEFAULT_TEST_EPIC

logger = logging.getLogger(__name__)

@dataclass
class CandleData:
    """Structured candle data with validation"""
    timestamp: datetime
    epic: str
    timeframe: int
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ofr_open: float
    ofr_high: float
    ofr_low: float
    ofr_close: float
    ltv: Optional[int] = None
    is_complete: bool = False

    def validate(self) -> bool:
        """Validate OHLC relationships and NaN values"""
        try:
            # Check for NaN values
            values = [self.bid_open, self.bid_high, self.bid_low, self.bid_close,
                     self.ofr_open, self.ofr_high, self.ofr_low, self.ofr_close]
            
            if any(math.isnan(v) or math.isinf(v) for v in values):
                logger.warning(f"NaN/Inf values detected in {self.epic} candle")
                return False
            
            # Validate BID OHLC relationships
            if not (self.bid_low <= self.bid_open <= self.bid_high and 
                   self.bid_low <= self.bid_close <= self.bid_high):
                logger.warning(f"Invalid BID OHLC for {self.epic}: "
                             f"O={self.bid_open}, H={self.bid_high}, L={self.bid_low}, C={self.bid_close}")
                return False
            
            # Validate OFFER OHLC relationships  
            if not (self.ofr_low <= self.ofr_open <= self.ofr_high and 
                   self.ofr_low <= self.ofr_close <= self.ofr_high):
                logger.warning(f"Invalid OFR OHLC for {self.epic}: "
                             f"O={self.ofr_open}, H={self.ofr_high}, L={self.ofr_low}, C={self.ofr_close}")
                return False
            
            # Validate spread relationships (OFR >= BID)
            if not (self.ofr_open >= self.bid_open and self.ofr_high >= self.bid_high and
                   self.ofr_low >= self.bid_low and self.ofr_close >= self.bid_close):
                logger.warning(f"Invalid spread for {self.epic}: OFR < BID")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating candle data for {self.epic}: {e}")
            return False

    def get_mid_ohlc(self) -> tuple:
        """Calculate MID prices for analytics"""
        return (
            (self.bid_open + self.ofr_open) / 2,
            (self.bid_high + self.ofr_high) / 2, 
            (self.bid_low + self.ofr_low) / 2,
            (self.bid_close + self.ofr_close) / 2
        )

class CandleBuffer:
    """Buffer to handle partial updates and detect candle completion"""
    def __init__(self):
        self.buffer: Dict[str, Dict[int, Dict]] = {}  # epic -> timeframe -> field_data
        
    def update(self, epic: str, timeframe: int, field_updates: Dict[str, Any]) -> Optional[CandleData]:
        """Update buffer and return complete candle if available"""
        if epic not in self.buffer:
            self.buffer[epic] = {}
        if timeframe not in self.buffer[epic]:
            self.buffer[epic][timeframe] = {}
        
        # Update with new field data (only non-empty values)
        for field, value in field_updates.items():
            if value is not None and value != '':
                self.buffer[epic][timeframe][field] = value
        
        # Check if we have all required fields for a complete candle
        required_fields = [
            "UTM", "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",
            "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE"
        ]
        
        current_data = self.buffer[epic][timeframe]
        if all(field in current_data for field in required_fields):
            try:
                # Parse timestamp - convert to candle open time
                ts_ms = int(current_data["UTM"])
                raw_ts = datetime.utcfromtimestamp(ts_ms / 1000)
                
                # Calculate candle open time based on timeframe
                candle_open = self._calculate_candle_open(raw_ts, timeframe)
                
                # Parse prices with NaN handling
                def safe_float(value):
                    if isinstance(value, str) and value.upper() == 'NAN':
                        return float('nan')
                    return float(value)
                
                candle = CandleData(
                    timestamp=candle_open,
                    epic=epic,
                    timeframe=timeframe,
                    bid_open=safe_float(current_data["BID_OPEN"]),
                    bid_high=safe_float(current_data["BID_HIGH"]),
                    bid_low=safe_float(current_data["BID_LOW"]),
                    bid_close=safe_float(current_data["BID_CLOSE"]),
                    ofr_open=safe_float(current_data["OFR_OPEN"]),
                    ofr_high=safe_float(current_data["OFR_HIGH"]),
                    ofr_low=safe_float(current_data["OFR_LOW"]),
                    ofr_close=safe_float(current_data["OFR_CLOSE"]),
                    ltv=int(current_data.get("LTV", 0)) if current_data.get("LTV", '') != '' else None
                )
                
                return candle if candle.validate() else None
                
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing candle data for {epic}: {e}")
                return None
        
        return None
    
    def _calculate_candle_open(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """Calculate the candle open time from any timestamp within that candle"""
        # Round down to the nearest timeframe boundary
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        candle_start_minutes = (minutes_since_midnight // timeframe_minutes) * timeframe_minutes
        
        return timestamp.replace(
            hour=candle_start_minutes // 60,
            minute=candle_start_minutes % 60,
            second=0,
            microsecond=0
        )

class IGChartCandleListener(SubscriptionListener):
    def __init__(self, epic: str, timeframe: int, validator: Optional[StreamValidatorService] = None):
        self.epic = epic
        self.timeframe = timeframe
        self.candle_buffer = CandleBuffer()
        self.last_candle_time = None
        self.validator = validator
        
    def onSubscription(self):
        logger.info(f"✅ Successfully subscribed to {self.timeframe}m for {self.epic}")

    def onSubscriptionError(self, code, message):
        logger.error(f"❌ Subscription error for {self.epic} {self.timeframe}m: {code} - {message}")

    def onItemUpdate(self, item_update):
        try:
            # Collect all available field updates (including partial ones)
            field_updates = {}
            fields = ["UTM", "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE", 
                     "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE", "LTV"]
            
            for field in fields:
                value = item_update.getValue(field)
                if value is not None:  # Include empty strings - they're valid updates
                    field_updates[field] = value
            
            # Update buffer and check for complete candle
            complete_candle = self.candle_buffer.update(self.epic, self.timeframe, field_updates)
            
            if complete_candle:
                # Detect if this is a new candle (bar completion)
                is_new_candle = (self.last_candle_time is None or 
                               complete_candle.timestamp != self.last_candle_time)
                
                if is_new_candle:
                    complete_candle.is_complete = True
                    logger.debug(f"🕯️ New {self.timeframe}m candle detected for {self.epic} at {complete_candle.timestamp}")
                
                # Store candle in database
                self._store_candle(complete_candle)
                self.last_candle_time = complete_candle.timestamp
                
        except Exception as e:
            logger.error(f"❌ Error processing update for {self.epic}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _store_candle(self, candle: CandleData):
        """Store candle data with validation and audit trail"""
        try:
            # Calculate MID prices for analytics compatibility
            mid_open, mid_high, mid_low, mid_close = candle.get_mid_ohlc()

            # Note: IG Markets changed their data format in September 2025
            # EURUSD data now comes in correct format (1.176xx) and doesn't need scaling
            # Removed old scaling correction that was dividing by 10,000
            
            # Validate price data using database function
            with SessionLocal() as validation_session:
                from sqlalchemy import text
                validation_result = validation_session.execute(
                    text("SELECT * FROM validate_price_data(:epic, :timeframe, :timestamp, :close, :source, :threshold)"),
                    {
                        "epic": candle.epic,
                        "timeframe": candle.timeframe, 
                        "timestamp": candle.timestamp,
                        "close": mid_close,
                        "source": 'chart_streamer',
                        "threshold": 5.0
                    }
                ).fetchone()
                
                is_valid, quality_score, validation_flags, warning_message = validation_result
                
                # Debug logging for troubleshooting
                logger.debug(f"Validation result for {candle.epic}: "
                           f"is_valid={is_valid}, quality_score={quality_score}, "
                           f"validation_flags={validation_flags} (type: {type(validation_flags)}), "
                           f"warning_message={warning_message}")
            
            # Log validation warnings - only critical issues as warnings
            if validation_flags and validation_flags != '{}':
                if quality_score < 0.5:  # Only critical issues as warnings
                    logger.warning(f"⚠️ Critical validation issues for {candle.epic}: {warning_message}")
                else:  # Minor issues as debug
                    logger.debug(f"⚠️ Validation issues for {candle.epic}: {warning_message}")
            
            # Create IGCandle object with audit fields
            from datetime import datetime
            import json
            
            # Handle validation_flags from database function
            # The validation function returns a PostgreSQL text[] array
            pg_validation_flags = None
            try:
                if validation_flags:
                    logger.debug(f"Processing validation_flags: {validation_flags} (type: {type(validation_flags)})")
                    
                    if isinstance(validation_flags, (list, tuple)):
                        # Convert list to PostgreSQL array literal format
                        escaped_flags = [str(flag).replace('"', '\\"') for flag in validation_flags]
                        pg_validation_flags = '{' + ','.join(f'"{flag}"' for flag in escaped_flags) + '}'
                    elif isinstance(validation_flags, str):
                        # If it's already a string, check if it's a proper array format
                        if validation_flags.startswith('{') and validation_flags.endswith('}'):
                            pg_validation_flags = validation_flags  # Already in proper format
                        else:
                            # Single string, convert to array
                            escaped_flag = validation_flags.replace('"', '\\"')
                            pg_validation_flags = f'{{"{escaped_flag}"}}'
                    else:
                        # Convert other types to string
                        escaped_flag = str(validation_flags).replace('"', '\\"')
                        pg_validation_flags = f'{{"{escaped_flag}"}}'
                    
                    logger.debug(f"Converted to PostgreSQL array format: {pg_validation_flags}")
            except Exception as e:
                logger.error(f"Error processing validation_flags: {e}")
                pg_validation_flags = '{"VALIDATION_PROCESSING_ERROR"}'
            
            ig_candle = IGCandle(
                start_time=candle.timestamp,
                epic=candle.epic,
                timeframe=candle.timeframe,
                open=mid_open,
                high=mid_high,
                low=mid_low,
                close=mid_close,
                volume=0,  # No TTV field available in chart subscription
                ltv=candle.ltv,
                cons_tick_count=None,
                data_source='chart_streamer',
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                quality_score=float(quality_score),
                validation_flags=pg_validation_flags
            )
            
            # Use single session per update
            with SessionLocal() as session:
                session.merge(ig_candle)
                session.commit()
            
            # Queue candle for stream vs API validation if candle is complete
            if candle.is_complete and self.validator:
                try:
                    streamed_ohlc = {
                        'open': mid_open,
                        'high': mid_high, 
                        'low': mid_low,
                        'close': mid_close
                    }
                    
                    # Determine priority based on quality score
                    priority = 2 if quality_score < 0.7 else 1
                    
                    self.validator.queue_candle_for_validation(
                        epic=candle.epic,
                        timeframe=candle.timeframe,
                        timestamp=candle.timestamp,
                        streamed_ohlc=streamed_ohlc,
                        data_source='chart_streamer',
                        priority=priority
                    )
                    
                    logger.debug(f"🔍 Queued {candle.epic} for API validation (priority: {priority})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to queue candle for validation: {e}")
            
            # Log level based on candle completion and quality
            log_level = logging.WARNING if quality_score < 0.5 else (
                logging.INFO if candle.is_complete else logging.DEBUG)
            
            quality_indicator = "🔴" if quality_score < 0.5 else "🟡" if quality_score < 0.8 else "🟢"
            
            logger.log(log_level, 
                      f"{quality_indicator} {'🆕' if candle.is_complete else '🔄'} {candle.timeframe}m candle "
                      f"{'completed' if candle.is_complete else 'updated'} at {candle.timestamp} "
                      f"for {candle.epic} - MID OHLC: {mid_open:.5f}/{mid_high:.5f}/{mid_low:.5f}/{mid_close:.5f} "
                      f"(Quality: {quality_score:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error storing candle for {candle.epic}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

class StreamManager:
    """Manages streaming connection with reconnection logic"""
    def __init__(self, epic: str, headers: Dict[str, str]):
        self.epic = epic
        self.headers = headers
        self.client = None
        self.subscriptions = []
        self.listeners = []
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Initialize stream validator
        self.validator = StreamValidatorService(headers)
        self.validator_task = None
        
    async def connect(self):
        """Establish connection with authentication"""
        try:
            self.client = LightstreamerClient(LIGHTSTREAMER_PROD_URL, "DEFAULT")
            user = self.headers["accountId"]
            password = f"CST-{self.headers['CST']}|XST-{self.headers['X-SECURITY-TOKEN']}"
            
            self.client.connectionDetails.setUser(user)
            self.client.connectionDetails.setPassword(password)
            
            # Add connection listener
            class ConnectionStatusListener:
                def __init__(self, manager):
                    self.manager = manager
                    
                def onStatusChange(self, status):
                    logger.info(f"Lightstreamer status for {self.manager.epic}: {status}")
                    self.manager.is_connected = status in ["CONNECTED:STREAM-SENSING", "CONNECTED:WS-STREAMING"]
                    
                def onServerError(self, error_code, error_message):
                    logger.error(f"❌ Server error for {self.manager.epic}: {error_code} - {error_message}")
                    self.manager.is_connected = False
            
            self.client.addListener(ConnectionStatusListener(self))
            
            # Connect first, then subscribe
            logger.info(f"🔌 Connecting to Lightstreamer for {self.epic}...")
            self.client.connect()
            
            # Wait for connection to establish
            max_wait = 30
            wait_count = 0
            while not self.is_connected and wait_count < max_wait:
                await asyncio.sleep(1)
                wait_count += 1
            
            if not self.is_connected:
                raise ConnectionError(f"Failed to connect to Lightstreamer for {self.epic}")
            
            logger.info(f"✅ Connected to Lightstreamer for {self.epic}")
            self.reconnect_attempts = 0
            
        except Exception as e:
            logger.error(f"❌ Connection failed for {self.epic}: {e}")
            raise
    
    def setup_subscriptions(self):
        """Set up chart subscriptions for multiple timeframes"""
        # UPDATED: 5-minute + 1-minute for future migration - 60m synthesized from 5m data
        # 1-minute data will allow us to eventually synthesize ALL timeframes from 1m base
        # CORRECTED: IG uses "1MINUTE" format, not "MINUTE"
        timeframes = {5: "5MINUTE", 1: "1MINUTE"}
        
        for tf_minutes, tf_str in timeframes.items():
            try:
                item = f"CHART:{self.epic}:{tf_str}"
                fields = [
                    "UTM",           # Timestamp
                    "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE",  # Offer prices
                    "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",  # Bid prices
                    "LTV"            # Last traded volume
                ]
                
                subscription = Subscription(
                    mode="MERGE",
                    items=[item],
                    fields=fields
                )
                
                listener = IGChartCandleListener(self.epic, tf_minutes, self.validator)
                subscription.addListener(listener)
                
                logger.info(f"📊 Setting up subscription: {item}")
                self.client.subscribe(subscription)
                
                self.subscriptions.append(subscription)
                self.listeners.append(listener)
                
            except Exception as e:
                logger.error(f"❌ Failed to set up subscription for {tf_minutes}m: {e}")
    
    async def monitor_connection(self):
        """Monitor connection health and reconnect if needed"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.is_connected:
                    logger.warning(f"⚠️ Connection lost for {self.epic}, attempting reconnection...")
                    await self._reconnect()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in connection monitor for {self.epic}: {e}")
    
    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"❌ Max reconnection attempts reached for {self.epic}")
            return
        
        try:
            self.reconnect_attempts += 1
            wait_time = min(300, 2 ** self.reconnect_attempts)  # Exponential backoff, max 5 minutes
            
            logger.info(f"🔄 Reconnection attempt {self.reconnect_attempts} for {self.epic} in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
            # Disconnect old client
            if self.client:
                self.client.disconnect()
            
            # Reconnect
            await self.connect()
            self.setup_subscriptions()
            
            logger.info(f"✅ Reconnected successfully for {self.epic}")
            
        except Exception as e:
            logger.error(f"❌ Reconnection failed for {self.epic}: {e}")
    
    def disconnect(self):
        """Clean disconnect"""
        if self.client:
            logger.info(f"🔌 Disconnecting Lightstreamer for {self.epic}...")
            self.client.disconnect()
            self.is_connected = False

async def stream_chart_candles(epic: str, headers: Dict[str, str]):
    """Main streaming function with improved error handling and reconnection"""
    logger.info(f"🚀 Starting enhanced chart stream for {epic}")
    
    stream_manager = StreamManager(epic, headers)
    
    try:
        # Connect and set up subscriptions
        await stream_manager.connect()
        stream_manager.setup_subscriptions()
        
        logger.info(f"📡 All subscriptions active for {epic}")
        
        # Start validation service
        if stream_manager.validator:
            stream_manager.validator_task = asyncio.create_task(
                stream_manager.validator.start_validation_worker()
            )
            logger.info(f"🔍 Stream validation service started for {epic}")
        
        # Start connection monitoring
        monitor_task = asyncio.create_task(stream_manager.monitor_connection())
        
        # Keep streaming alive
        while True:
            await asyncio.sleep(60)  # Main loop heartbeat
            
    except asyncio.CancelledError:
        logger.info(f"Chart streaming cancelled for {epic}")
    except Exception as e:
        logger.error(f"❌ Critical error in stream_chart_candles for {epic}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        if 'monitor_task' in locals():
            monitor_task.cancel()
            
        if stream_manager.validator_task:
            stream_manager.validator.stop()
            stream_manager.validator_task.cancel()
            
            # Log final validation stats
            stats = stream_manager.validator.get_validation_stats()
            logger.info(f"📊 Validation stats for {epic}: {stats}")
        
        stream_manager.disconnect()
        logger.info(f"✅ Cleanup complete for {epic}")