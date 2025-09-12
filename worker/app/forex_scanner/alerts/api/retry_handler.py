"""
Retry Handler - API Retry Logic Module
Handles retry mechanisms, backoff strategies, and rate limiting
Extracted from claude_api.py for better modularity
"""

import time
import random
import logging
from typing import Callable, Any, Optional
from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_multiplier: float = 1.0


class RetryHandler:
    """
    Handles retry logic with exponential backoff and jitter
    """
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    def execute_with_retry(self, 
                          func: Callable, 
                          *args, 
                          retry_on_exceptions: tuple = None,
                          **kwargs) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            *args: Function arguments
            retry_on_exceptions: Tuple of exceptions to retry on
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises last exception
        """
        if retry_on_exceptions is None:
            retry_on_exceptions = (Exception,)
        
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"✅ Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except retry_on_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"⚠️ Attempt {attempt + 1}/{self.config.max_retries + 1} failed: {str(e)}"
                    )
                    self.logger.info(f"🔄 Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"❌ All {self.config.max_retries + 1} attempts failed. Last error: {str(e)}"
                    )
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and optional jitter
        """
        # Base exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Apply backoff multiplier
        delay *= self.config.backoff_multiplier
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Clamp to max delay
        return min(delay, self.config.max_delay)
    
    def create_http_retry_config(self) -> 'HTTPRetryConfig':
        """Create specialized config for HTTP retries"""
        return HTTPRetryConfig(
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base
        )


@dataclass
class HTTPRetryConfig(RetryConfig):
    """Specialized retry configuration for HTTP requests"""
    retry_on_status_codes: tuple = (429, 500, 502, 503, 504, 529)
    retry_on_timeouts: bool = True
    retry_on_connection_errors: bool = True
    rate_limit_delay: float = 60.0  # Extra delay for rate limits


class HTTPRetryHandler(RetryHandler):
    """
    Specialized retry handler for HTTP requests
    """
    
    def __init__(self, config: HTTPRetryConfig = None):
        self.http_config = config or HTTPRetryConfig()
        super().__init__(self.http_config)
    
    def execute_http_request(self, 
                           func: Callable, 
                           *args, 
                           **kwargs) -> Any:
        """
        Execute HTTP request with specialized retry logic
        """
        import requests
        
        retry_exceptions = (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException
        )
        
        def http_request_wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            
            # Check for HTTP status codes that should trigger retry
            if hasattr(response, 'status_code'):
                if response.status_code in self.http_config.retry_on_status_codes:
                    if response.status_code == 429:  # Rate limit
                        self.logger.warning(f"🚦 Rate limited (429), will retry with longer delay")
                        time.sleep(self.http_config.rate_limit_delay)
                    elif response.status_code == 529:  # Service overloaded
                        self.logger.warning(f"🔥 Service overloaded (529), will retry with backoff")
                    
                    # Raise an exception to trigger retry
                    raise requests.exceptions.RequestException(
                        f"HTTP {response.status_code}: {response.text[:100]}"
                    )
            
            return response
        
        return self.execute_with_retry(
            http_request_wrapper,
            *args,
            retry_on_exceptions=retry_exceptions,
            **kwargs
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        HTTP-specific delay calculation with longer delays for certain errors
        """
        base_delay = super()._calculate_delay(attempt)
        
        # Add extra delay for rate limits and server errors
        if attempt > 0:  # After first failure
            base_delay *= 1.5  # Increase delay for HTTP retries
        
        return min(base_delay, self.http_config.max_delay)


class RateLimiter:
    """
    Rate limiting to prevent API abuse
    """
    
    def __init__(self, 
                 calls_per_minute: int = 50, 
                 min_interval: float = 1.2):
        self.calls_per_minute = calls_per_minute
        self.min_interval = min_interval
        self.last_call_time = 0
        self.call_history = []
        self.logger = logging.getLogger(__name__)
    
    def wait_if_needed(self):
        """
        Wait if necessary to respect rate limits
        """
        current_time = time.time()
        
        # Check minimum interval between calls
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            self.logger.debug(f"⏱️ Rate limiting: sleeping {sleep_time:.1f}s (min interval)")
            time.sleep(sleep_time)
            current_time = time.time()
        
        # Check calls per minute limit
        minute_ago = current_time - 60
        self.call_history = [t for t in self.call_history if t > minute_ago]
        
        if len(self.call_history) >= self.calls_per_minute:
            # Calculate how long to wait
            oldest_call = min(self.call_history)
            wait_time = 60 - (current_time - oldest_call) + 1  # +1 for safety
            
            if wait_time > 0:
                self.logger.warning(
                    f"🚦 Rate limit reached ({self.calls_per_minute}/min), waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)
                current_time = time.time()
        
        # Record this call
        self.call_history.append(current_time)
        self.last_call_time = current_time
    
    def get_rate_limit_status(self) -> dict:
        """
        Get current rate limit status
        """
        current_time = time.time()
        minute_ago = current_time - 60
        recent_calls = [t for t in self.call_history if t > minute_ago]
        
        return {
            'calls_in_last_minute': len(recent_calls),
            'calls_per_minute_limit': self.calls_per_minute,
            'time_since_last_call': current_time - self.last_call_time,
            'min_interval': self.min_interval,
            'rate_limited': len(recent_calls) >= self.calls_per_minute
        }


# Factory functions for common configurations
def create_claude_retry_handler() -> HTTPRetryHandler:
    """Create retry handler optimized for Claude API"""
    config = HTTPRetryConfig(
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0,
        exponential_base=2.0,
        retry_on_status_codes=(429, 500, 502, 503, 504, 529),
        rate_limit_delay=60.0
    )
    return HTTPRetryHandler(config)


def create_claude_rate_limiter() -> RateLimiter:
    """Create rate limiter for Claude API (50 calls/minute)"""
    return RateLimiter(calls_per_minute=50, min_interval=1.2)


# Usage example and testing
if __name__ == "__main__":
    import requests
    
    # Example usage
    retry_handler = create_claude_retry_handler()
    rate_limiter = create_claude_rate_limiter()
    
    def test_api_call():
        """Simulate an API call that might fail"""
        rate_limiter.wait_if_needed()
        
        # This would be your actual API call
        response = requests.get("https://httpbin.org/status/200")
        return response
    
    try:
        result = retry_handler.execute_http_request(test_api_call)
        print(f"✅ API call succeeded: {result.status_code}")
        
        # Check rate limit status
        status = rate_limiter.get_rate_limit_status()
        print(f"📊 Rate limit status: {status}")
        
    except Exception as e:
        print(f"❌ API call failed after all retries: {e}")