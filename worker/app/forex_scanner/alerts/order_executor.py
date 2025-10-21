# /app/forex_scanner/alerts/order_executor.py
"""
Order Execution Module - ENHANCED WITH RETRY LOGIC
Integrates your existing send_order function with the forex scanner
Now includes performance tracking for dynamic EMA configuration
FIXED: Epic mapping now works with REVERSE_EPIC_MAP and auto-creates from config
FIXED: API request format matches working dev-app implementation
ENHANCED: Added robust retry logic with exponential backoff and circuit breaker
ENHANCED: Comprehensive timeout handling for HTTPConnectionPool errors
"""

import requests
import logging
import time
import random
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
try:
    import config
except ImportError:
    from forex_scanner import config


class RetryStrategy(Enum):
    """Retry strategy for different types of errors"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Timeout configurations
    connect_timeout: float = 10.0  # Connection timeout
    read_timeout: float = 45.0     # Read timeout (longer for processing)
    total_timeout: float = 60.0    # Total request timeout
    
    # Retry conditions
    retry_on_status_codes: List[int] = None
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True
    
    def __post_init__(self):
        if self.retry_on_status_codes is None:
            # Retry transient errors
            # 429 = Rate limit (retry with backoff)
            # 500 = Internal server error (could be transient)
            # 502/503/504 = Gateway/Service unavailable (transient)
            # 503 also used for position check failures (should retry)
            self.retry_on_status_codes = [429, 500, 502, 503, 504]


class CircuitBreaker:
    """Circuit breaker pattern for handling service failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call_allowed(self) -> bool:
        """Check if call is allowed through circuit breaker"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        return False
    
    def record_success(self):
        """Record successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class OrderExecutor:
    """Enhanced order executor with retry logic and comprehensive timeout handling"""
    
    def __init__(self, ema_strategy=None):
        self.logger = logging.getLogger(__name__)
        
        # Store reference to EMA strategy for performance tracking
        self.ema_strategy = ema_strategy
        
        # Load your API configuration from config
        self.order_api_url = getattr(config, 'ORDER_API_URL', None)
        self.api_subscription_key = getattr(config, 'API_SUBSCRIPTION_KEY', None)
        self.epic_map = getattr(config, 'EPIC_MAP', {})
        
        # FIXED: Load or create reverse epic mapping
        self.reverse_epic_map = getattr(config, 'REVERSE_EPIC_MAP', {})
        
        # If REVERSE_EPIC_MAP doesn't exist in config, create it from EPIC_MAP
        if not self.reverse_epic_map and self.epic_map:
            self.reverse_epic_map = {}
            for internal_epic, external_epic in self.epic_map.items():
                self.reverse_epic_map[external_epic] = internal_epic
            self.logger.info("🔄 Auto-created reverse epic mapping from EPIC_MAP")
        elif self.reverse_epic_map:
            self.logger.info("✅ Loaded REVERSE_EPIC_MAP from config")
        
        # Trading settings
        self.enabled = getattr(config, 'AUTO_TRADING_ENABLED', False)
        self.default_risk_reward = getattr(config, 'DEFAULT_RISK_REWARD', 2.0)
        self.default_stop_distance = getattr(config, 'DEFAULT_STOP_DISTANCE', 20)  # pips
        self.position_size = getattr(config, 'DEFAULT_POSITION_SIZE', None)
        
        # ENHANCED: Retry configuration
        self.retry_config = RetryConfig(
            max_retries=getattr(config, 'ORDER_MAX_RETRIES', 3),
            base_delay=getattr(config, 'ORDER_RETRY_BASE_DELAY', 2.0),
            max_delay=getattr(config, 'ORDER_RETRY_MAX_DELAY', 60.0),
            connect_timeout=getattr(config, 'ORDER_CONNECT_TIMEOUT', 10.0),
            read_timeout=getattr(config, 'ORDER_READ_TIMEOUT', 45.0),
            total_timeout=getattr(config, 'ORDER_TOTAL_TIMEOUT', 60.0),
        )
        
        # ENHANCED: Circuit breaker for service protection
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, 'ORDER_CIRCUIT_BREAKER_THRESHOLD', 5),
            recovery_timeout=getattr(config, 'ORDER_CIRCUIT_BREAKER_RECOVERY', 300.0)
        )
        
        # Performance tracking - store pending trades
        self.pending_trades = {}  # trade_id -> signal info
        self.completed_trades = []  # completed trade records
        
        # ENHANCED: Request statistics tracking
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'timeout_requests': 0,
            'retry_requests': 0,
            'circuit_breaker_blocks': 0
        }
        
        if self.enabled:
            self.logger.info("🤖 Enhanced Order Executor initialized - AUTO TRADING ENABLED")
            self.logger.info(f"   API URL: {self.order_api_url[:50]}..." if self.order_api_url else "   ❌ No API URL configured")
            self.logger.info(f"   Epic mappings: {len(self.epic_map)} forward, {len(self.reverse_epic_map)} reverse")
            self.logger.info(f"   Retry config: max_retries={self.retry_config.max_retries}, timeouts={self.retry_config.connect_timeout}/{self.retry_config.read_timeout}s")
            self.logger.info(f"   Circuit breaker: {self.circuit_breaker.failure_threshold} failures threshold")
            
            # Debug log some key mappings
            test_epics = ['USDCHF.1.MINI', 'EURUSD.1.MINI']
            for test_epic in test_epics:
                if test_epic in self.reverse_epic_map:
                    self.logger.info(f"✅ {test_epic} -> {self.reverse_epic_map[test_epic]}")
                else:
                    self.logger.warning(f"⚠️ {test_epic} not found in reverse mapping")
                    
        else:
            self.logger.info("📋 Enhanced Order Executor initialized - AUTO TRADING DISABLED")
        
        if self.ema_strategy:
            self.logger.info("📊 Performance tracking enabled for dynamic EMA configuration")
    
    def execute_signal_order(self, signal: Dict) -> Dict:
        """
        ENHANCED: Execute order based on forex scanner signal with retry logic and alert_id tracking
        
        Args:
            signal: Signal from forex scanner (now includes alert_id from database save)
            
        Returns:
            Order execution result with alert_id included
        """
        alert_id = signal.get('alert_id')
        internal_epic = signal.get('epic', '')
        signal_type = signal.get('signal_type', '').upper()
        confidence = signal.get('confidence_score', 0)
        
        self.logger.info(f"🚀 Enhanced order execution: {internal_epic} {signal_type} ({confidence:.1%})")
        if alert_id:
            self.logger.info(f"   Alert ID: {alert_id}")
        
        # Check circuit breaker first
        if not self.circuit_breaker.call_allowed():
            self.request_stats['circuit_breaker_blocks'] += 1
            self.logger.warning(f"🔒 Circuit breaker OPEN - blocking order execution for {internal_epic}")
            return {
                "status": "error",
                "message": "Circuit breaker open - service temporarily unavailable",
                "alert_id": alert_id,
                "circuit_breaker_state": self.circuit_breaker.state
            }
        
        try:
            # Validate required fields
            if not internal_epic or not signal_type:
                error_msg = "Missing required fields: epic or signal_type"
                return {
                    "status": "error", 
                    "message": error_msg,
                    "alert_id": alert_id
                }
            
            # FIXED: Convert internal epic to external epic format for API
            external_epic = self.epic_map.get(internal_epic)
            if not external_epic:
                error_msg = f"No epic mapping found for: {internal_epic}"
                self.logger.error(f"❌ {error_msg}")
                self.logger.error(f"Available mappings: {list(self.epic_map.keys())[:3]}...")
                return {
                    "status": "error", 
                    "message": error_msg,
                    "alert_id": alert_id
                }
            
            self.logger.info(f"🔄 Epic mapping: {internal_epic} -> {external_epic}")
            
            # Check if epic is blacklisted from trading
            trading_blacklist = getattr(config, 'TRADING_BLACKLIST', {})
            if internal_epic in trading_blacklist:
                reason = trading_blacklist[internal_epic]
                self.logger.warning(f"🚫 Trading blocked for {internal_epic}: {reason}")
                return {
                    "status": "blocked", 
                    "message": f"Trading blocked: {reason}",
                    "epic": internal_epic,
                    "alert_id": alert_id
                }
            
            # Convert signal type to trading direction
            direction = signal_type  # Assuming signal_type is already 'BUY' or 'SELL'
            if direction not in ['BUY', 'SELL']:
                error_msg = f"Invalid signal type: {signal_type}"
                return {
                    "status": "error", 
                    "message": error_msg,
                    "alert_id": alert_id
                }
            
            # Calculate order parameters
            stop_distance = signal.get('stop_distance', self.default_stop_distance)
            limit_distance = signal.get('limit_distance', int(stop_distance * self.default_risk_reward))

            # Debug: Log if using defaults vs strategy values
            if 'stop_distance' not in signal:
                self.logger.warning(f"⚠️ Signal missing 'stop_distance', using default: {stop_distance} pips")
                self.logger.debug(f"   Signal keys: {list(signal.keys())}")
            if 'limit_distance' not in signal:
                self.logger.warning(f"⚠️ Signal missing 'limit_distance', using default: {limit_distance} pips")

            # ✅ SAFETY VALIDATION: Ensure reasonable SL/TP values
            # For all pairs, stop_distance and limit_distance should be in reasonable pip/point range
            max_reasonable_sl = 100  # Maximum reasonable stop loss in pips/points
            max_reasonable_tp = 200  # Maximum reasonable take profit in pips/points

            if stop_distance > max_reasonable_sl:
                self.logger.error(
                    f"🚨 INVALID SL DETECTED for {internal_epic}: {stop_distance} pips exceeds max {max_reasonable_sl} pips. "
                    f"This likely indicates a calculation bug. Rejecting order."
                )
                return {
                    "status": "error",
                    "message": f"Invalid stop distance {stop_distance} pips (max: {max_reasonable_sl} pips)",
                    "epic": internal_epic,
                    "alert_id": alert_id
                }

            if limit_distance > max_reasonable_tp:
                self.logger.warning(
                    f"⚠️ LARGE TP DETECTED for {internal_epic}: {limit_distance} pips exceeds max {max_reasonable_tp} pips. "
                    f"Capping to {max_reasonable_tp} pips for safety."
                )
                limit_distance = max_reasonable_tp

            # Create custom label with alert_id reference if available
            if alert_id:
                custom_label = f"forex_scanner_alert_{alert_id}_{external_epic}_{direction}"
            else:
                custom_label = f"forex_scanner_{external_epic}_{direction}_{int(datetime.now().timestamp())}"
            
            self.logger.info(f"📊 Order parameters: Stop={stop_distance}pips, Limit={limit_distance}pips ({confidence:.1%})")
            
            # ENHANCED: Execute with retry logic
            result = self._execute_with_retry(signal, external_epic, direction, stop_distance, limit_distance, custom_label, alert_id)
            
            # Record success with circuit breaker
            self.circuit_breaker.record_success()
            self.request_stats['successful_requests'] += 1
            
            # Track pending trade for performance monitoring (use internal epic for consistency)
            if result.get('status') == 'success':
                trade_id = self._extract_trade_id(result)
                if trade_id:
                    self._track_pending_trade(trade_id, signal, direction, stop_distance, limit_distance, alert_id)
            
            return result
            
        except Exception as e:
            # Record failure with circuit breaker
            self.circuit_breaker.record_failure()
            self.request_stats['failed_requests'] += 1
            
            error_msg = f"Order execution failed after all retries: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            
            return {
                "status": "error",
                "message": error_msg,
                "alert_id": alert_id,
                "circuit_breaker_state": self.circuit_breaker.state
            }
    
    def _execute_with_retry(self, signal: Dict, external_epic: str, direction: str, 
                           stop_distance: int, limit_distance: int, custom_label: str, alert_id: int) -> Dict:
        """ENHANCED: Execute order with retry logic and exponential backoff"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                self.request_stats['total_requests'] += 1
                
                if attempt > 0:
                    self.request_stats['retry_requests'] += 1
                    delay = self._calculate_delay(attempt - 1)
                    self.logger.info(f"🔄 Retry attempt {attempt}/{self.retry_config.max_retries} for {external_epic} in {delay:.1f}s")
                    time.sleep(delay)
                
                # Execute the order using the existing send_order method
                result = self.send_order(
                    external_epic=external_epic,
                    direction=direction,
                    stop_distance=stop_distance,
                    limit_distance=limit_distance,
                    size=self.position_size,
                    custom_label=custom_label,
                    risk_reward=self.default_risk_reward,
                    alert_id=alert_id
                )
                
                if result and result.get('status') != 'error':
                    if attempt > 0:
                        self.logger.info(f"✅ Order succeeded on retry attempt {attempt} for {external_epic}")
                    return result
                else:
                    # Handle API-level errors
                    error_msg = result.get('message', 'Unknown API error') if result else 'No response'
                    raise requests.exceptions.RequestException(f"API Error: {error_msg}")
                    
            except (requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException) as e:
                
                last_exception = e
                
                # Handle specific error types
                if isinstance(e, requests.exceptions.Timeout):
                    self.request_stats['timeout_requests'] += 1
                    self.logger.warning(f"⏱️ Request timeout on attempt {attempt + 1} for {external_epic}: {str(e)}")
                elif isinstance(e, requests.exceptions.ConnectionError):
                    self.logger.warning(f"🔌 Connection error on attempt {attempt + 1} for {external_epic}: {str(e)}")
                else:
                    self.logger.warning(f"📡 Request error on attempt {attempt + 1} for {external_epic}: {str(e)}")
                
                # Don't retry on last attempt
                if attempt >= self.retry_config.max_retries:
                    break
                    
            except Exception as e:
                last_exception = e
                self.logger.error(f"💥 Unexpected error on attempt {attempt + 1} for {external_epic}: {str(e)}")
                
                # Don't retry unexpected errors
                break
        
        # All retries exhausted
        error_type = type(last_exception).__name__ if last_exception else "Unknown"
        error_msg = str(last_exception) if last_exception else "Unknown error"
        
        self.logger.error(f"❌ Order execution failed after {self.retry_config.max_retries + 1} attempts for {external_epic}")
        self.logger.error(f"   Last error ({error_type}): {error_msg}")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """ENHANCED: Calculate retry delay with exponential backoff and jitter"""
        if self.retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt)
        elif self.retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.retry_config.base_delay * (attempt + 1)
        else:  # FIXED_DELAY
            delay = self.retry_config.base_delay
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Clamp to max delay
        return min(delay, self.retry_config.max_delay)
    
    def send_order(self, external_epic: str, direction: str, stop_distance: int = None, 
               limit_distance: int = None, size: float = None, custom_label: str = None,
               risk_reward: float = None, alert_id: int = None):
        """
        ENHANCED: Send order to your trading platform via API using correct format with timeout handling
        
        Args:
            external_epic: Trading symbol (e.g., 'EURUSD')
            direction: 'BUY' or 'SELL'
            stop_distance: Stop loss distance in pips
            limit_distance: Take profit distance in pips
            size: Position size
            custom_label: Custom label for the trade
            risk_reward: Risk/reward ratio
            alert_id: Optional ID from alert_history table for tracking
        
        Returns:
            API response or None if failed
        """
        # BUGFIX: Initialize result variable at the beginning to prevent UnboundLocalError
        result = {
            "status": "error",
            "message": "Order execution failed - unknown error",
            "alert_id": alert_id
        }
        
        if not self.enabled:
            self.logger.info("💡 Auto-trading disabled - would execute order in paper mode")
            return {"status": "paper_mode", "alert_id": alert_id}
        
        if not self.order_api_url or not self.api_subscription_key:
            self.logger.error("❌ Order API not configured")
            return {"status": "error", "message": "API not configured", "alert_id": alert_id}
        
        # Prepare order data
        order_data = {
            "epic": external_epic,  # This should be the external epic format
            "direction": direction.upper(),
            "size": size or self.position_size or 1.0,
            "stop_distance": stop_distance or self.default_stop_distance,
            "limit_distance": limit_distance,
            "use_provided_sl_tp": True,  # ✅ NEW: Tell dev-app to use strategy-calculated values
            "custom_label": custom_label or f"forex_scanner_{int(datetime.now().timestamp())}",
            "risk_reward": risk_reward or self.default_risk_reward
        }

        # Include alert_id in the request if provided
        if alert_id is not None:
            order_data["alert_id"] = alert_id  # Add to request body
            self.logger.info(f"📊 Sending order with alert_id: {alert_id}")

        # Log SL/TP being sent
        self.logger.info(f"📤 Sending order with strategy SL/TP: {stop_distance}/{limit_distance} for {external_epic}")
        
        # FIXED: Use the correct headers format that works in dev-app
        headers = {
            "x-apim-gateway": "verified",  # This is required by the middleware
            "Content-Type": "application/json"
        }
        
        # FIXED: Use query parameter for subscription key (as in dev-app)
        params = {
            "subscription-key": self.api_subscription_key
        }
        
        # ENHANCED: Create timeout configuration
        timeout = (
            self.retry_config.connect_timeout,
            self.retry_config.read_timeout
        )
        
        self.logger.info(f"📤 Sending {direction} order: {external_epic}")
        self.logger.debug(f"   API URL: {self.order_api_url}")
        self.logger.debug(f"   Timeout: connect={timeout[0]}s, read={timeout[1]}s")
        self.logger.debug(f"   Size: {order_data['size']}")
        self.logger.debug(f"   Stop: {order_data['stop_distance']} pips")
        self.logger.debug(f"   Limit: {order_data['limit_distance']} pips")
        if alert_id:
            self.logger.debug(f"   Alert ID: {alert_id}")
        
        # ENHANCED: Send the request with enhanced timeout handling
        start_time = time.time()
        
        try:
            # FIXED: Send the order using POST with JSON body and query params
            response = requests.post(
                self.order_api_url,
                json=order_data,  # Use json parameter instead of data
                headers=headers,
                params=params,    # Add query parameters
                timeout=timeout   # ENHANCED: (connect_timeout, read_timeout)
            )
            
            response_time = time.time() - start_time
            self.logger.debug(f"📊 Request completed in {response_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                result["alert_id"] = alert_id
                result["response_time"] = response_time

                self.logger.info(f"✅ Order sent successfully: {external_epic} (took {response_time:.2f}s)")
                if alert_id:
                    self.logger.info(f"✅ Order linked to alert_id: {alert_id}")

                return result

            elif response.status_code == 409:
                # Position already open - not an error, just skip
                try:
                    detail = response.json().get("detail", {})
                    msg = detail.get("message", "Position already open")
                    reason = detail.get("reason", "duplicate_position")
                except:
                    msg = "Position already open"
                    reason = "duplicate_position"

                self.logger.info(f"ℹ️ {external_epic}: {msg} (this is expected behavior)")
                result = {
                    "status": "skipped",
                    "message": msg,
                    "alert_id": alert_id,
                    "status_code": 409,
                    "response_time": response_time,
                    "reason": reason
                }
                return result

            elif response.status_code == 429:
                # Rate limit / cooldown - log as info, allow retry
                try:
                    detail = response.json().get("detail", {})
                    msg = detail.get("message", "Rate limit or cooldown active")
                except:
                    msg = "Rate limit or cooldown active"

                self.logger.info(f"⏳ {external_epic}: {msg}")
                raise requests.exceptions.RequestException(f"Rate limit: {msg}")

            elif response.status_code == 403:
                # Forbidden (blacklisted) - log as warning, don't retry
                try:
                    detail = response.json().get("detail", {})
                    msg = detail.get("message", "Trading blocked")
                except:
                    msg = "Trading blocked for this epic"

                self.logger.warning(f"⚠️ {external_epic}: {msg}")
                result = {
                    "status": "blocked",
                    "message": msg,
                    "alert_id": alert_id,
                    "status_code": 403,
                    "response_time": response_time,
                    "reason": "blacklisted"
                }
                return result

            elif response.status_code == 503:
                # FIXED: Service unavailable (position check failed) - log as warning, allow retry
                try:
                    detail = response.json().get("detail", {})
                    msg = detail.get("message", "Position verification service temporarily unavailable")
                    reason = detail.get("reason", "position_check_failed")
                except:
                    msg = "Position verification service temporarily unavailable"
                    reason = "service_unavailable"

                self.logger.warning(f"⚠️ {external_epic}: {msg} (will retry)")
                self.logger.debug(f"   Reason: {reason}")
                # Raise exception to trigger retry logic
                raise requests.exceptions.RequestException(f"Service unavailable (503): {msg}")

            else:
                # Other errors - log appropriately with full details
                try:
                    error_detail = response.json()
                    error_msg = f"Order failed: HTTP {response.status_code} - {error_detail}"

                    # Special handling for 500 errors to provide more context
                    if response.status_code == 500:
                        detail_obj = error_detail if isinstance(error_detail, dict) else {}
                        error_desc = detail_obj.get("detail", str(error_detail))

                        # Check if this is actually a "position exists" error that was incorrectly returned as 500
                        if "already open" in str(error_desc).lower() or "duplicate" in str(error_desc).lower():
                            self.logger.warning(f"⚠️ Server returned 500 for duplicate position (should be 409): {error_desc}")
                            self.logger.info(f"ℹ️ Treating as skipped order for {external_epic}")
                            return {
                                "status": "skipped",
                                "message": f"Position already open (server error 500 converted)",
                                "alert_id": alert_id,
                                "status_code": 500,
                                "response_time": response_time,
                                "reason": "duplicate_position_500"
                            }
                except:
                    error_msg = f"Order failed: HTTP {response.status_code} - {response.text[:500]}"

                # Check if this status code should trigger retry (500, 502, 503, 504)
                if response.status_code in self.retry_config.retry_on_status_codes:
                    self.logger.warning(f"⚠️ Transient error (will retry): {error_msg}")
                    self.logger.debug(f"   Order data sent: {order_data}")
                    raise requests.exceptions.RequestException(error_msg)
                else:
                    # Real error - log as ERROR with full details
                    self.logger.error(f"❌ {error_msg}")
                    self.logger.error(f"   Order data sent: {order_data}")

                result = {
                    "status": "error",
                    "message": error_msg,
                    "alert_id": alert_id,
                    "status_code": response.status_code,
                    "response_time": response_time
                }
                return result
                
        except requests.exceptions.Timeout as e:
            response_time = time.time() - start_time
            timeout_type = "connection" if response_time < self.retry_config.connect_timeout else "read"
            error_msg = f"Request timeout ({timeout_type}) after {response_time:.1f}s: {str(e)}"
            self.logger.error(f"⏱️ {error_msg}")
            raise  # Re-raise to trigger retry
            
        except requests.exceptions.ConnectionError as e:
            response_time = time.time() - start_time
            error_msg = f"Connection error after {response_time:.1f}s: {str(e)}"
            self.logger.error(f"🔌 {error_msg}")
            raise  # Re-raise to trigger retry
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Unexpected error after {response_time:.1f}s: {str(e)}"
            self.logger.error(f"💥 {error_msg}")
            
            result = {
                "status": "error", 
                "message": error_msg, 
                "alert_id": alert_id,
                "response_time": response_time
            }
            return result
    
    def get_performance_stats(self) -> Dict:
        """ENHANCED: Get performance statistics including retry metrics"""
        total_requests = self.request_stats['total_requests']
        
        stats = {
            **self.request_stats,
            'success_rate': (
                (self.request_stats['successful_requests'] / total_requests * 100) 
                if total_requests > 0 else 0
            ),
            'timeout_rate': (
                (self.request_stats['timeout_requests'] / total_requests * 100) 
                if total_requests > 0 else 0
            ),
            'retry_rate': (
                (self.request_stats['retry_requests'] / total_requests * 100) 
                if total_requests > 0 else 0
            ),
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failure_count': self.circuit_breaker.failure_count
        }
        
        return stats
    
    def reset_circuit_breaker(self):
        """ENHANCED: Manually reset circuit breaker (for admin use)"""
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "CLOSED"
        self.logger.info("🔄 Circuit breaker manually reset")
    
    def _convert_to_integer_pips(self, distance_value, external_epic):
        """
        Convert price distance to integer pips based on pair type
        
        Args:
            distance_value: Price distance (could be in price units or already pips)
            external_epic: Epic like "USDCHF.1.MINI" to determine pair type
            
        Returns:
            Integer pip value
        """
        if distance_value is None:
            return None
        
        try:
            # Extract pair from external epic
            pair = external_epic.split('.')[0] if '.' in external_epic else external_epic
            
            # Determine pip multiplier based on pair type
            is_jpy_pair = 'JPY' in pair
            
            # Check if the value is already in pips (typically > 1 for stops)
            if distance_value >= 1:
                # Likely already in pips, just convert to integer
                return int(round(distance_value))
            
            # Convert price distance to pips
            if is_jpy_pair:
                # JPY pairs: 1 pip = 0.01
                pip_value = distance_value / 0.01
            else:
                # Non-JPY pairs: 1 pip = 0.0001
                pip_value = distance_value / 0.0001
            
            # Round to nearest integer pip
            pip_value_int = int(round(pip_value))
            
            # Ensure minimum pip value (avoid zero pips)
            if pip_value_int <= 0:
                pip_value_int = 5 if is_jpy_pair else 5  # Minimum 5 pips
                self.logger.warning(f"⚠️ Distance too small ({distance_value}), using minimum {pip_value_int} pips")
            
            # Log conversion for debugging
            self.logger.debug(f"📏 Pip conversion: {distance_value} -> {pip_value_int} pips ({pair})")
            
            return pip_value_int
            
        except Exception as e:
            self.logger.error(f"❌ Error converting distance to pips: {e}")
            # Fallback: assume already in pips and convert to int
            try:
                return int(round(float(distance_value)))
            except:
                return 20  # Default 20 pips as fallback

    def _try_create_mapping(self, external_epic: str) -> Optional[str]:
        """
        Try to create epic mapping on-the-fly for missing epics
        This makes the system future-proof when new epics are added to config
        """
        try:
            # Pattern: "USDCHF.1.MINI" -> "CS.D.USDCHF.MINI.IP"
            parts = external_epic.split('.')
            if len(parts) >= 2:
                base_pair = parts[0]  # "USDCHF"
                
                # Construct internal epic format
                constructed_internal = f"CS.D.{base_pair}.MINI.IP"
                
                # Check if this constructed epic exists in forward mapping
                if constructed_internal in self.epic_map:
                    # Add to reverse mapping for future use
                    self.reverse_epic_map[external_epic] = constructed_internal
                    self.logger.info(f"🔄 Auto-created mapping: {external_epic} -> {constructed_internal}")
                    return constructed_internal
                else:
                    # Try to find partial match in existing forward mappings
                    for internal_epic, mapped_external in self.epic_map.items():
                        if base_pair in internal_epic:
                            self.reverse_epic_map[external_epic] = internal_epic
                            self.logger.info(f"🔄 Found partial match: {external_epic} -> {internal_epic}")
                            return internal_epic
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating mapping for {external_epic}: {e}")
            return None
    
    def reload_epic_mappings(self):
        """
        Reload epic mappings from config - useful when config is updated
        Call this method after updating config.py with new epics
        """
        try:
            # Reload config module
            import importlib
            importlib.reload(config)
            
            # Update mappings
            self.epic_map = getattr(config, 'EPIC_MAP', {})
            new_reverse_map = getattr(config, 'REVERSE_EPIC_MAP', {})
            
            # If no reverse map in config, create from forward map
            if not new_reverse_map and self.epic_map:
                new_reverse_map = {}
                for internal_epic, external_epic in self.epic_map.items():
                    new_reverse_map[external_epic] = internal_epic
            
            self.reverse_epic_map = new_reverse_map
            
            self.logger.info(f"🔄 Reloaded epic mappings: {len(self.epic_map)} forward, {len(self.reverse_epic_map)} reverse")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reload epic mappings: {e}")
            return False
    
    def debug_epic_mappings(self):
        """Debug method to show all current epic mappings"""
        self.logger.info("📊 Current Epic Mappings:")
        self.logger.info(f"   Forward mappings ({len(self.epic_map)}):")
        for internal, external in list(self.epic_map.items())[:5]:
            self.logger.info(f"     {internal} -> {external}")
        if len(self.epic_map) > 5:
            self.logger.info(f"     ... and {len(self.epic_map) - 5} more")
            
        self.logger.info(f"   Reverse mappings ({len(self.reverse_epic_map)}):")
        for external, internal in list(self.reverse_epic_map.items())[:5]:
            self.logger.info(f"     {external} -> {internal}")
        if len(self.reverse_epic_map) > 5:
            self.logger.info(f"     ... and {len(self.reverse_epic_map) - 5} more")
    
    def validate_epic_mapping(self, internal_epic: str) -> bool:
        """
        Validate if an internal epic has a corresponding external mapping
        
        Args:
            internal_epic: Internal epic format (CS.D.USDJPY.MINI.IP)
            
        Returns:
            True if mapping exists, False otherwise
        """
        exists = internal_epic in self.epic_map
        if not exists:
            self.logger.warning(f"⚠️ No epic mapping found for: {internal_epic}")
            self.logger.info("📋 Available internal epics:")
            for i, epic in enumerate(list(self.epic_map.keys())[:5]):
                self.logger.info(f"   {i+1}. {epic}")
            if len(self.epic_map) > 5:
                self.logger.info(f"   ... and {len(self.epic_map) - 5} more")
        return exists
    
    def _extract_trade_id(self, order_result: Dict) -> Optional[str]:
        """Extract trade ID from order response"""
        try:
            # Adjust this based on your API response structure
            response_data = order_result.get('response', {})
            return response_data.get('trade_id') or response_data.get('deal_id') or response_data.get('order_id')
        except:
            return None
    
    def _track_pending_trade(self, trade_id: str, signal: Dict, direction: str, 
                       stop_distance: int, limit_distance: int, alert_id: int = None):
        """
        UPDATED: Track a pending trade for performance monitoring with alert_id linking
        
        Args:
            trade_id: Unique trade identifier
            signal: Original signal data
            direction: BUY or SELL
            stop_distance: Stop loss distance in pips
            limit_distance: Take profit distance in pips
            alert_id: Optional alert ID from alert_history table
        """
        
        self.pending_trades[trade_id] = {
            'signal': signal.copy(),
            'epic': signal.get('epic'),
            'direction': direction,
            'stop_distance': stop_distance,
            'limit_distance': limit_distance,
            'entry_time': datetime.now(),
            'expected_stop_pips': stop_distance,
            'expected_target_pips': limit_distance,
            'config_name': signal.get('config_name', 'unknown'),
            'config_mode': signal.get('config_mode', 'static'),
            'alert_id': alert_id
        }
        
        if alert_id:
            self.logger.info(f"📊 Tracking trade {trade_id} for performance analysis (Alert ID: {alert_id})")
        else:
            self.logger.info(f"📊 Tracking trade {trade_id} for performance analysis")
    
    def check_trade_completions(self):
        """
        Check for completed trades and update performance tracking
        Call this method periodically (e.g., every few minutes)
        """
        if not self.pending_trades:
            return
        
        self.logger.debug(f"🔍 Checking {len(self.pending_trades)} pending trades for completion")
        
        completed_trade_ids = []
        
        for trade_id, trade_info in self.pending_trades.items():
            try:
                # Check if trade has been completed
                # You'll need to implement this based on your API
                completion_result = self._check_single_trade_completion(trade_id, trade_info)
                
                if completion_result:
                    self._handle_trade_completion(trade_id, trade_info, completion_result)
                    completed_trade_ids.append(trade_id)
                    
            except Exception as e:
                self.logger.error(f"❌ Error checking trade {trade_id}: {e}")
        
        # Remove completed trades from pending list
        for trade_id in completed_trade_ids:
            del self.pending_trades[trade_id]
    
    def _check_single_trade_completion(self, trade_id: str, trade_info: Dict) -> Optional[Dict]:
        """
        Check if a single trade has been completed
        
        Returns:
            Completion result if completed, None if still pending
        """
        # TODO: Implement based on your API
        # This is a placeholder - you'll need to adapt this to your actual API
        
        # Option 1: Check via API call
        # try:
        #     response = requests.get(f"{self.order_api_url}/status/{trade_id}")
        #     if response.status_code == 200:
        #         data = response.json()
        #         if data.get('status') == 'CLOSED':
        #             return {
        #                 'status': 'completed',
        #                 'profit_loss': data.get('profit_loss'),
        #                 'exit_price': data.get('close_price'),
        #                 'close_time': data.get('close_time')
        #             }
        # except Exception as e:
        #     self.logger.error(f"Error checking trade status: {e}")
        
        # Option 2: Simple timeout-based simulation (for testing)
        entry_time = trade_info.get('entry_time')
        if datetime.now() - entry_time > timedelta(hours=4):  # Simulate 4-hour trades
            import random
            # Simulate random outcomes for testing
            profit_pips = random.uniform(-trade_info['stop_distance'], trade_info['limit_distance'])
            return {
                'status': 'completed',
                'profit_pips': profit_pips,
                'exit_price': 0,  # Would be actual exit price
                'close_time': datetime.now()
            }
        
        return None
    
    def _handle_trade_completion(self, trade_id: str, trade_info: Dict, completion_result: Dict):
        """Handle a completed trade and update performance tracking"""
        
        try:
            profit_pips = completion_result.get('profit_pips', 0)
            
            # Create performance result
            result = {
                'profitable': profit_pips > 0,
                'profit_pips': abs(profit_pips),
                'entry_price': 0,  # You can get this from your API
                'exit_price': completion_result.get('exit_price', 0),
                'signal_time': trade_info['entry_time'],
                'direction': trade_info['direction'],
                'confidence': trade_info['signal'].get('confidence_score', 0),
                'market_conditions': trade_info['signal'].get('market_conditions', {})
            }
            
            self.logger.info(f"📊 Trade {trade_id} completed:")
            self.logger.info(f"   Epic: {trade_info['epic']}")
            self.logger.info(f"   Profit: {profit_pips:.1f} pips")
            self.logger.info(f"   Config: {trade_info['config_name']} ({trade_info['config_mode']})")
            
            # Update performance tracking for dynamic EMA
            if self.ema_strategy:
                self.ema_strategy.update_signal_result(
                    trade_info['epic'],
                    trade_info['signal'],
                    result
                )
                self.logger.info("✅ Updated dynamic EMA performance tracking")
            
            # Store completed trade record
            self.completed_trades.append({
                'trade_id': trade_id,
                'completion_time': datetime.now(),
                'trade_info': trade_info,
                'result': result
            })
            
            # Keep only last 100 completed trades in memory
            if len(self.completed_trades) > 100:
                self.completed_trades = self.completed_trades[-100:]
                
        except Exception as e:
            self.logger.error(f"❌ Error handling trade completion for {trade_id}: {e}")
    
    def simulate_trade_completion(self, trade_id: str, profit_pips: float):
        """
        Manually simulate a trade completion (for testing)
        
        Args:
            trade_id: ID of the trade to complete
            profit_pips: Profit/loss in pips (positive = profit, negative = loss)
        """
        if trade_id not in self.pending_trades:
            self.logger.warning(f"Trade {trade_id} not found in pending trades")
            return
        
        trade_info = self.pending_trades[trade_id]
        completion_result = {
            'status': 'completed',
            'profit_pips': profit_pips,
            'exit_price': 0,
            'close_time': datetime.now()
        }
        
        self._handle_trade_completion(trade_id, trade_info, completion_result)
        del self.pending_trades[trade_id]
        
        self.logger.info(f"🧪 Simulated completion of trade {trade_id} with {profit_pips:.1f} pips")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for completed trades"""
        
        if not self.completed_trades:
            return {'total_trades': 0, 'message': 'No completed trades yet'}
        
        profitable_trades = [t for t in self.completed_trades if t['result']['profitable']]
        total_profit = sum(t['result']['profit_pips'] for t in self.completed_trades if t['result']['profitable'])
        total_loss = sum(t['result']['profit_pips'] for t in self.completed_trades if not t['result']['profitable'])
        
        return {
            'total_trades': len(self.completed_trades),
            'profitable_trades': len(profitable_trades),
            'win_rate': len(profitable_trades) / len(self.completed_trades),
            'total_profit_pips': total_profit,
            'total_loss_pips': total_loss,
            'net_profit_pips': total_profit - total_loss,
            'pending_trades': len(self.pending_trades)
        }
    
    def _calculate_stop_distance_enhanced(self, signal: Dict) -> int:
        """
        Enhanced stop distance calculation that returns integer pips
        """
        try:
            # Method 1: Use signal's stop_loss if available
            entry_price = signal.get('price', signal.get('entry_price'))
            stop_loss = signal.get('stop_loss')
            
            if entry_price and stop_loss:
                # Calculate distance in price terms
                price_distance = abs(float(entry_price) - float(stop_loss))
                
                # Convert to pips using the epic
                epic = signal.get('epic', '')
                return self._convert_to_integer_pips(price_distance, epic)
            
            # Method 2: Use configured dynamic stops
            if hasattr(config, 'DYNAMIC_STOPS') and config.DYNAMIC_STOPS:
                confidence = signal.get('confidence_score', 0.5)
                base_stop = self.default_stop_distance
                
                # Higher confidence = tighter stop
                if confidence > 0.8:
                    return int(base_stop * 0.8)  # 16 pips for high confidence
                elif confidence > 0.7:
                    return base_stop  # 20 pips for medium confidence
                else:
                    return int(base_stop * 1.2)  # 24 pips for lower confidence
            
            # Method 3: Use default stop distance
            return self.default_stop_distance
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating stop distance: {e}")
            return self.default_stop_distance  # Fallback
    
    def should_execute_signal(self, signal: Dict) -> bool:
        """
        Determine if a signal should trigger an order
        Add your risk management rules here
        """
        if not self.enabled:
            return False
        
        # Check minimum confidence
        confidence = signal.get('confidence_score', 0)
        min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.6)

        if confidence < min_confidence:
            self.logger.info(f"📊 Signal confidence {confidence:.1%} below threshold {min_confidence:.1%}")
            return False
        
        # Add more risk management rules here:
        # - Check if market is open
        # - Check daily/weekly loss limits
        # - Check maximum positions
        # - Check news events
        
        return True