"""
Claude API Client - Core Communication Module
Refactored to use official Anthropic SDK with vision support.

Features:
- Official Anthropic SDK for reliability
- Vision support for chart analysis
- Text-only fallback
- Retry logic with exponential backoff
- Rate limiting protection
- Token usage tracking
"""

import logging
import time
import os
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    anthropic = None

try:
    import config
except ImportError:
    from forex_scanner import config

logger = logging.getLogger(__name__)


class APIClient:
    """
    Claude API Client using official Anthropic SDK.

    Supports both text-only and vision (image) analysis.
    Uses fail-secure error handling for trade validation.
    """

    # Available models
    MODELS = {
        'haiku': 'claude-3-haiku-20240307',
        'sonnet': 'claude-sonnet-4-20250514',
        'sonnet-old': 'claude-3-5-sonnet-20241022',
        'opus': 'claude-opus-4-20250514',
    }

    # Default model for trade validation
    DEFAULT_MODEL = 'sonnet'

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 50
    MAX_REQUESTS_PER_DAY = 1000
    MIN_CALL_INTERVAL = 1.2  # seconds

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Claude API client.

        Args:
            api_key: Anthropic API key (or reads from config/env)
            model: Model to use ('haiku', 'sonnet', 'opus')
        """
        # Get API key from multiple sources
        self.api_key = api_key or getattr(config, 'CLAUDE_API_KEY', None) or os.getenv('ANTHROPIC_API_KEY')

        # Initialize SDK client
        self.client = None
        if ANTHROPIC_SDK_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("✅ Anthropic SDK client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Anthropic client: {e}")
        elif not ANTHROPIC_SDK_AVAILABLE:
            logger.error("❌ Anthropic SDK not installed. Run: pip install anthropic")
        else:
            logger.warning("⚠️ No Claude API key provided")

        # Model configuration
        self.model_key = model or getattr(config, 'CLAUDE_MODEL', self.DEFAULT_MODEL)
        self.model = self.MODELS.get(self.model_key, self.MODELS[self.DEFAULT_MODEL])

        # Default settings
        self.max_tokens = 300
        self.timeout = 60  # seconds

        # Retry configuration
        self.max_retries = 3
        self.base_delay = 2.0
        self.max_delay = 30.0

        # Rate limiting tracking
        self.last_api_call = 0
        self._request_times = []
        self._daily_request_count = 0
        self._daily_reset_date = datetime.now().date()

        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'vision_requests': 0,
            'text_requests': 0,
            'total_tokens': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
        }

        logger.info(f"📡 Claude API Client initialized - Model: {self.model_key} ({self.model})")

    @property
    def is_available(self) -> bool:
        """Check if the API client is available"""
        return self.client is not None and self.api_key is not None

    def call_api(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """
        Make a text-only API call to Claude.

        Args:
            prompt: Text prompt to send
            max_tokens: Maximum output tokens (default: 300)

        Returns:
            Response text or None on error
        """
        if not self.is_available:
            logger.warning("Claude API client not available")
            return None

        # Rate limiting
        self._enforce_rate_limit()

        self.stats['total_requests'] += 1
        self.stats['text_requests'] += 1

        for attempt in range(self.max_retries + 1):
            try:
                self.last_api_call = time.time()
                self._record_request()

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )

                # Extract response
                content = message.content[0].text if message.content else ''

                # Track tokens
                self._track_tokens(message.usage)

                self.stats['successful_requests'] += 1
                logger.debug(f"✅ Claude API call successful - {message.usage.output_tokens} output tokens")

                return content

            except anthropic.RateLimitError as e:
                logger.warning(f"⚠️ Rate limit error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    delay = min(60 + (attempt * 10), self.max_delay)
                    logger.info(f"🔄 Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

            except anthropic.APIStatusError as e:
                logger.error(f"❌ API status error: {e.status_code} - {e.message}")
                if e.status_code == 529:  # Overloaded
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)
                        logger.info(f"🔄 API overloaded, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                break

            except anthropic.APIConnectionError as e:
                logger.error(f"❌ API connection error: {e}")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"🔄 Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                break

            except Exception as e:
                logger.error(f"❌ Unexpected error calling Claude API: {e}")
                break

        self.stats['failed_requests'] += 1
        return None

    def call_api_with_image(
        self,
        prompt: str,
        image_base64: str,
        model: str = None,
        max_tokens: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make a vision API call to Claude with an image.

        Args:
            prompt: Text prompt to send
            image_base64: Base64-encoded PNG image
            model: Model to use (default: configured model)
            max_tokens: Maximum output tokens

        Returns:
            Dict with 'content' and 'tokens', or None on error
        """
        if not self.is_available:
            logger.warning("Claude API client not available")
            return None

        if not image_base64:
            logger.warning("No image provided, falling back to text-only")
            content = self.call_api(prompt, max_tokens)
            return {'content': content, 'tokens': 0} if content else None

        # Rate limiting
        self._enforce_rate_limit()

        self.stats['total_requests'] += 1
        self.stats['vision_requests'] += 1

        # Resolve model
        model_id = self.MODELS.get(model, self.model) if model else self.model

        for attempt in range(self.max_retries + 1):
            try:
                self.last_api_call = time.time()
                self._record_request()

                # Build multimodal message content
                message_content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]

                message = self.client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens or self.max_tokens,
                    messages=[{"role": "user", "content": message_content}]
                )

                # Extract response
                content = message.content[0].text if message.content else ''

                # Track tokens
                self._track_tokens(message.usage)

                tokens_used = message.usage.input_tokens + message.usage.output_tokens

                self.stats['successful_requests'] += 1
                logger.debug(f"✅ Claude Vision API call successful - {tokens_used} total tokens")

                return {
                    'content': content,
                    'tokens': tokens_used,
                    'input_tokens': message.usage.input_tokens,
                    'output_tokens': message.usage.output_tokens,
                    'model': model_id,
                }

            except anthropic.RateLimitError as e:
                logger.warning(f"⚠️ Rate limit error (vision, attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    delay = min(60 + (attempt * 10), self.max_delay)
                    time.sleep(delay)
                    continue

            except anthropic.APIStatusError as e:
                logger.error(f"❌ API status error (vision): {e.status_code} - {e.message}")
                if e.status_code == 529 and attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                break

            except anthropic.APIConnectionError as e:
                logger.error(f"❌ API connection error (vision): {e}")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                break

            except Exception as e:
                logger.error(f"❌ Unexpected error calling Claude Vision API: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                break

        self.stats['failed_requests'] += 1
        return None

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.MIN_CALL_INTERVAL:
            sleep_time = self.MIN_CALL_INTERVAL - time_since_last_call
            logger.debug(f"⏱️ Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Check daily limit
        today = datetime.now().date()
        if today != self._daily_reset_date:
            self._daily_request_count = 0
            self._daily_reset_date = today

        if self._daily_request_count >= self.MAX_REQUESTS_PER_DAY:
            logger.error(f"❌ Daily request limit reached ({self.MAX_REQUESTS_PER_DAY})")
            raise Exception("Daily API request limit reached")

        # Check per-minute limit
        minute_ago = current_time - 60
        self._request_times = [t for t in self._request_times if t > minute_ago]

        if len(self._request_times) >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                logger.warning(f"⚠️ Per-minute limit reached, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)

    def _record_request(self) -> None:
        """Record a request for rate limiting"""
        self._request_times.append(time.time())
        self._daily_request_count += 1

    def _track_tokens(self, usage) -> None:
        """Track token usage from API response"""
        if usage:
            self.stats['total_input_tokens'] += usage.input_tokens
            self.stats['total_output_tokens'] += usage.output_tokens
            self.stats['total_tokens'] += usage.input_tokens + usage.output_tokens

    # Class-level cache to avoid repeated connection tests during startup
    _connection_test_cache = {}
    _CONNECTION_TEST_CACHE_DURATION = 300  # 5 minutes

    def test_connection(self, force: bool = False) -> bool:
        """
        Test Claude API connection with caching to avoid redundant API calls.

        Args:
            force: If True, bypass cache and force a new connection test

        Returns:
            True if connection is working, False otherwise
        """
        if not self.is_available:
            return False

        # Check cache unless forced
        cache_key = self.api_key[:8] if self.api_key else 'no_key'
        now = datetime.now().timestamp()

        if not force and cache_key in APIClient._connection_test_cache:
            cached_result, cached_time = APIClient._connection_test_cache[cache_key]
            if now - cached_time < self._CONNECTION_TEST_CACHE_DURATION:
                logger.debug(f"✅ Using cached connection test result (age: {int(now - cached_time)}s)")
                return cached_result

        try:
            response = self.call_api("Respond with exactly: OK", max_tokens=10)
            success = response is not None and 'OK' in response.upper()

            if success:
                logger.info("✅ Claude API connection test successful")
            else:
                logger.warning("⚠️ Claude API connection test returned unexpected response")

            # Cache the result
            APIClient._connection_test_cache[cache_key] = (success, now)
            return success

        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            # Cache failure too (but for shorter duration)
            APIClient._connection_test_cache[cache_key] = (False, now)
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """
        Check Claude API health and return detailed status.

        Returns:
            Dict with status, reason, and recommendation
        """
        if not ANTHROPIC_SDK_AVAILABLE:
            return {
                'status': 'unavailable',
                'reason': 'Anthropic SDK not installed',
                'recommendation': 'Run: pip install anthropic'
            }

        if not self.api_key:
            return {
                'status': 'unavailable',
                'reason': 'No API key configured',
                'recommendation': 'Set CLAUDE_API_KEY in config or environment'
            }

        if not self.client:
            return {
                'status': 'error',
                'reason': 'Failed to initialize Anthropic client',
                'recommendation': 'Check API key validity'
            }

        try:
            test_success = self.test_connection()

            if test_success:
                return {
                    'status': 'healthy',
                    'reason': 'API responding normally',
                    'recommendation': 'Continue normal operations',
                    'model': self.model,
                    'stats': self.stats.copy()
                }
            else:
                return {
                    'status': 'degraded',
                    'reason': 'API responding with unexpected output',
                    'recommendation': 'Monitor closely'
                }

        except Exception as e:
            return {
                'status': 'error',
                'reason': f'Health check failed: {str(e)}',
                'recommendation': 'Use fallback analysis mode'
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            **self.stats,
            'model': self.model,
            'model_key': self.model_key,
            'daily_requests': self._daily_request_count,
            'daily_limit': self.MAX_REQUESTS_PER_DAY,
            'requests_remaining': self.MAX_REQUESTS_PER_DAY - self._daily_request_count,
        }


# Backward compatibility alias
ForexClaudeClient = APIClient


def create_api_client(api_key: str = None, model: str = None) -> APIClient:
    """Factory function to create API client"""
    return APIClient(api_key=api_key, model=model)
