"""
Stock Claude Analyzer

Main service for analyzing stock signals using Claude API.
Provides institutional-grade analysis with investment thesis,
risk assessment, and actionable recommendations.

Supports multimodal analysis with chart images for enhanced
pattern recognition and visual context.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .stock_prompt_builder import StockPromptBuilder
from .stock_response_parser import StockResponseParser, ClaudeAnalysis
from .stock_chart_generator import StockChartGenerator

logger = logging.getLogger(__name__)


class StockClaudeAnalyzer:
    """
    Claude API integration for stock signal analysis.

    Provides:
    - Single signal analysis with optional chart images
    - Multimodal vision analysis for enhanced pattern recognition
    - Batch analysis with rate limiting
    - Cost tracking and optimization
    - Fallback handling when API unavailable

    Usage:
        analyzer = StockClaudeAnalyzer(db_manager=db)
        result = await analyzer.analyze_signal(
            signal_data, technical_data, fundamental_data,
            include_chart=True  # Enable vision analysis
        )
        print(f"Grade: {result.grade}, Action: {result.action}")
    """

    # Model configurations - vision-capable models
    MODELS = {
        'haiku': 'claude-3-haiku-20240307',
        'sonnet': 'claude-sonnet-4-20250514',
        'opus': 'claude-opus-4-20250514',
    }

    # Default model for different analysis levels
    # Note: All these models support vision
    DEFAULT_MODELS = {
        'quick': 'haiku',
        'standard': 'sonnet',
        'comprehensive': 'sonnet',
    }

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 20
    MAX_REQUESTS_PER_DAY = 200

    # Token limits (increased for vision analysis)
    MAX_INPUT_TOKENS = 4000  # Increased for image tokens
    MAX_OUTPUT_TOKENS = {
        'quick': 150,
        'standard': 400,  # Increased for richer analysis with charts
        'comprehensive': 600,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = 'sonnet',
        auto_save: bool = False,
        db_manager=None,
        enable_charts: bool = True
    ):
        """
        Initialize the stock Claude analyzer.

        Args:
            api_key: Anthropic API key. If None, reads from CLAUDE_API_KEY env var
            default_model: Default model to use ('haiku', 'sonnet', 'opus')
            auto_save: Whether to auto-save analyses to files
            db_manager: Database manager for fetching candle data (required for charts)
            enable_charts: Whether to enable chart generation (default True)
        """
        self.api_key = api_key or os.environ.get('CLAUDE_API_KEY')
        self.default_model = default_model
        self.auto_save = auto_save
        self.db = db_manager
        self.enable_charts = enable_charts

        self.prompt_builder = StockPromptBuilder()
        self.response_parser = StockResponseParser()

        # Initialize chart generator if db available
        self.chart_generator = None
        if enable_charts:
            self.chart_generator = StockChartGenerator(db_manager=db_manager)
            if self.chart_generator.is_available:
                logger.info("Chart generation enabled for vision analysis")
            else:
                logger.warning("Chart generation not available - mplfinance not installed")
                self.chart_generator = None

        # Initialize client
        self.client = None
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("StockClaudeAnalyzer initialized with API key")
        else:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("anthropic package not installed")
            if not self.api_key:
                logger.warning("No Claude API key provided")

        # Rate limiting state
        self._request_times: List[float] = []
        self._daily_request_count = 0
        self._daily_reset_date = datetime.now().date()

        # Stats tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0,
            'charts_generated': 0,
            'vision_requests': 0,
        }

    @property
    def is_available(self) -> bool:
        """Check if Claude API is available"""
        return self.client is not None

    @property
    def charts_available(self) -> bool:
        """Check if chart generation is available"""
        return self.chart_generator is not None and self.chart_generator.is_available

    async def analyze_signal(
        self,
        signal: Dict[str, Any],
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        analysis_level: str = 'standard',
        model: Optional[str] = None,
        include_chart: bool = True,
        candles: Optional[List[Dict[str, Any]]] = None
    ) -> ClaudeAnalysis:
        """
        Analyze a single stock signal with Claude.

        Args:
            signal: Signal data including ticker, entry, stop, targets, score
            technical_data: Technical indicators (RSI, MACD, trends, etc.)
            fundamental_data: Fundamental metrics (P/E, growth, ownership, etc.)
            analysis_level: 'quick', 'standard', or 'comprehensive'
            model: Override model selection ('haiku', 'sonnet', 'opus')
            include_chart: Whether to include a chart image for vision analysis
            candles: Optional candle data for chart (fetched from DB if not provided)

        Returns:
            ClaudeAnalysis with grade, thesis, recommendations
        """
        if not self.is_available:
            logger.warning("Claude API not available, returning fallback analysis")
            return self._generate_fallback_analysis(signal, technical_data, fundamental_data)

        # Check rate limits
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, returning fallback analysis")
            return self.response_parser.get_default_analysis("Rate limit exceeded")

        # Prepare data
        technical = technical_data or {}
        fundamental = fundamental_data or {}

        # Merge signal data into technical if not already present
        for key in ['rsi_14', 'macd_histogram', 'relative_volume', 'trend_strength']:
            if key in signal and key not in technical:
                technical[key] = signal[key]

        # Generate chart image if enabled and available
        chart_base64 = None
        ticker = signal.get('ticker', 'UNKNOWN')

        # Generate chart for all analysis levels when enabled (charts always included by default)
        if include_chart and self.charts_available:
            try:
                smc_data = technical.get('smc', {})
                chart_base64 = await self.chart_generator.generate_signal_chart(
                    ticker=ticker,
                    candles=candles,
                    signal=signal,
                    smc_data=smc_data,
                    technical_data=technical
                )
                if chart_base64:
                    self.stats['charts_generated'] += 1
                    logger.debug(f"Generated chart for {ticker}")
            except Exception as e:
                logger.warning(f"Chart generation failed for {ticker}: {e}")

        # Build prompt (with chart context if available)
        prompt = self.prompt_builder.build_signal_analysis_prompt(
            signal=signal,
            technical_data=technical,
            fundamental_data=fundamental,
            analysis_level=analysis_level,
            has_chart=chart_base64 is not None
        )

        # Select model
        model_name = model or self.DEFAULT_MODELS.get(analysis_level, self.default_model)
        model_id = self.MODELS.get(model_name, self.MODELS['sonnet'])
        max_tokens = self.MAX_OUTPUT_TOKENS.get(analysis_level, 400)

        # Call API (with or without image)
        start_time = time.time()
        try:
            if chart_base64:
                response = await self._call_api_with_image(
                    prompt, chart_base64, model_id, max_tokens
                )
                self.stats['vision_requests'] += 1
            else:
                response = await self._call_api(prompt, model_id, max_tokens)

            latency_ms = int((time.time() - start_time) * 1000)

            if response:
                # Parse response
                analysis = self.response_parser.parse_response(
                    response_text=response['content'],
                    tokens_used=response['tokens'],
                    latency_ms=latency_ms,
                    model=model_id
                )

                # Update stats
                self.stats['successful_requests'] += 1
                self.stats['total_tokens'] += response['tokens']
                self.stats['total_latency_ms'] += latency_ms

                vision_tag = " [with chart]" if chart_base64 else ""
                logger.info(
                    f"Analyzed {ticker}{vision_tag}: "
                    f"Grade={analysis.grade}, Score={analysis.score}, "
                    f"Action={analysis.action} ({latency_ms}ms)"
                )

                return analysis
            else:
                self.stats['failed_requests'] += 1
                return self.response_parser.get_default_analysis("Empty API response")

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            self.stats['failed_requests'] += 1
            return self.response_parser.get_default_analysis(f"API error: {str(e)}")

    async def batch_analyze_signals(
        self,
        signals: List[Dict[str, Any]],
        technical_data_list: Optional[List[Dict[str, Any]]] = None,
        fundamental_data_list: Optional[List[Dict[str, Any]]] = None,
        analysis_level: str = 'standard',
        max_concurrent: int = 3,
        delay_between_requests: float = 0.5,
        include_chart: bool = True
    ) -> List[Tuple[Dict[str, Any], ClaudeAnalysis]]:
        """
        Analyze multiple signals in batch with rate limiting.

        Args:
            signals: List of signal dictionaries
            technical_data_list: Corresponding technical data (or None to extract from signals)
            fundamental_data_list: Corresponding fundamental data (or None)
            analysis_level: Analysis depth
            max_concurrent: Max concurrent API calls
            delay_between_requests: Delay between requests in seconds
            include_chart: Whether to include chart images (default True - always includes charts)

        Returns:
            List of (signal, analysis) tuples
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_one(
            idx: int,
            signal: Dict[str, Any]
        ) -> Tuple[Dict[str, Any], ClaudeAnalysis]:
            async with semaphore:
                # Get corresponding data
                technical = technical_data_list[idx] if technical_data_list and idx < len(technical_data_list) else None
                fundamental = fundamental_data_list[idx] if fundamental_data_list and idx < len(fundamental_data_list) else None

                # Add delay to avoid rate limits
                if idx > 0:
                    await asyncio.sleep(delay_between_requests)

                analysis = await self.analyze_signal(
                    signal=signal,
                    technical_data=technical,
                    fundamental_data=fundamental,
                    analysis_level=analysis_level,
                    include_chart=include_chart  # Pass through chart setting
                )
                return (signal, analysis)

        # Create tasks
        tasks = [
            analyze_one(i, signal)
            for i, signal in enumerate(signals)
        ]

        # Execute with progress logging
        logger.info(f"Starting batch analysis of {len(signals)} signals")

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            signal, analysis = await coro
            results.append((signal, analysis))
            logger.info(f"Completed {i+1}/{len(signals)}: {signal.get('ticker', '???')} = {analysis.grade}")

        logger.info(f"Batch analysis complete: {len(results)} signals analyzed")
        return results

    async def _call_api(
        self,
        prompt: str,
        model: str,
        max_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """
        Make API call to Claude.

        Args:
            prompt: The prompt to send
            model: Model ID to use
            max_tokens: Maximum output tokens

        Returns:
            Dict with 'content' and 'tokens' or None on error
        """
        if not self.client:
            return None

        self._record_request()
        self.stats['total_requests'] += 1

        try:
            # Run in thread pool since anthropic client is sync
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            )

            content = message.content[0].text if message.content else ''
            tokens = message.usage.input_tokens + message.usage.output_tokens

            return {
                'content': content,
                'tokens': tokens,
                'input_tokens': message.usage.input_tokens,
                'output_tokens': message.usage.output_tokens,
            }

        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            return None
        except anthropic.APIError as e:
            logger.error(f"API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Claude API: {e}")
            return None

    async def _call_api_with_image(
        self,
        prompt: str,
        image_base64: str,
        model: str,
        max_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """
        Make API call to Claude with an image for vision analysis.

        Args:
            prompt: The text prompt to send
            image_base64: Base64-encoded PNG image
            model: Model ID to use (must support vision)
            max_tokens: Maximum output tokens

        Returns:
            Dict with 'content' and 'tokens' or None on error
        """
        if not self.client:
            return None

        self._record_request()
        self.stats['total_requests'] += 1

        try:
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

            # Run in thread pool since anthropic client is sync
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ]
                )
            )

            content = message.content[0].text if message.content else ''
            tokens = message.usage.input_tokens + message.usage.output_tokens

            return {
                'content': content,
                'tokens': tokens,
                'input_tokens': message.usage.input_tokens,
                'output_tokens': message.usage.output_tokens,
            }

        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit error (vision): {e}")
            return None
        except anthropic.APIError as e:
            logger.error(f"API error (vision): {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Claude API (vision): {e}")
            return None

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()

        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self._daily_reset_date:
            self._daily_request_count = 0
            self._daily_reset_date = today

        # Check daily limit
        if self._daily_request_count >= self.MAX_REQUESTS_PER_DAY:
            logger.warning(f"Daily limit reached: {self._daily_request_count}")
            return False

        # Check per-minute limit
        minute_ago = now - 60
        recent_requests = [t for t in self._request_times if t > minute_ago]
        self._request_times = recent_requests  # Clean old entries

        if len(recent_requests) >= self.MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"Per-minute limit reached: {len(recent_requests)}")
            return False

        return True

    def _record_request(self) -> None:
        """Record a request for rate limiting"""
        self._request_times.append(time.time())
        self._daily_request_count += 1

    def _generate_fallback_analysis(
        self,
        signal: Dict[str, Any],
        technical: Optional[Dict[str, Any]],
        fundamental: Optional[Dict[str, Any]]
    ) -> ClaudeAnalysis:
        """
        Generate a rule-based fallback analysis when API unavailable.

        This uses the scanner's own scoring to generate a basic analysis.
        """
        score = signal.get('composite_score', 50)
        tier = signal.get('quality_tier', 'C')
        rr_ratio = signal.get('risk_reward_ratio', 1.5)

        # Map tier to grade
        grade_map = {'A+': 'A+', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
        grade = grade_map.get(tier, 'C')

        # Determine action based on score
        if score >= 80:
            action = 'STRONG BUY'
            conviction = 'HIGH'
            position = 'Full'
        elif score >= 70:
            action = 'BUY'
            conviction = 'HIGH'
            position = 'Half'
        elif score >= 60:
            action = 'BUY'
            conviction = 'MEDIUM'
            position = 'Quarter'
        elif score >= 50:
            action = 'HOLD'
            conviction = 'LOW'
            position = 'Skip'
        else:
            action = 'AVOID'
            conviction = 'LOW'
            position = 'Skip'

        # Build thesis from confluence factors
        factors = signal.get('confluence_factors', [])
        if factors:
            thesis = f"Signal has {len(factors)} confluence factors: {', '.join(factors[:3])}. "
        else:
            thesis = ""

        thesis += f"Scanner score {score}/100 ({tier} tier) with {rr_ratio:.1f}:1 risk/reward."

        # Extract strengths and risks from factors
        strengths = []
        risks = []

        for factor in factors:
            factor_lower = factor.lower()
            if any(word in factor_lower for word in ['bullish', 'above', 'high', 'strong', 'buy']):
                strengths.append(factor)
            elif any(word in factor_lower for word in ['bearish', 'below', 'low', 'weak', 'sell']):
                risks.append(factor)

        # Add R:R assessment
        if rr_ratio >= 2.5:
            strengths.append(f"Excellent R:R ratio ({rr_ratio:.1f}:1)")
        elif rr_ratio >= 2.0:
            strengths.append(f"Good R:R ratio ({rr_ratio:.1f}:1)")
        elif rr_ratio < 1.5:
            risks.append(f"Low R:R ratio ({rr_ratio:.1f}:1)")

        # Check fundamental risks
        if fundamental:
            days_to_earnings = fundamental.get('days_to_earnings')
            if days_to_earnings and days_to_earnings <= 7:
                risks.append(f"Earnings in {days_to_earnings} days")

            short_pct = fundamental.get('short_percent_float')
            if short_pct and short_pct >= 20:
                risks.append(f"High short interest ({short_pct:.0f}%)")

        return ClaudeAnalysis(
            grade=grade,
            score=max(1, min(10, score // 10)),
            conviction=conviction,
            action=action,
            key_strengths=strengths[:3],
            key_risks=risks[:3],
            thesis=thesis,
            position_recommendation=position,
            stop_adjustment='Keep',
            time_horizon='Swing',
            model='fallback',
            error='API unavailable - used rule-based fallback'
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            **self.stats,
            'daily_requests': self._daily_request_count,
            'api_available': self.is_available,
        }

    def test_connection(self) -> bool:
        """Test Claude API connection"""
        if not self.client:
            return False

        try:
            message = self.client.messages.create(
                model=self.MODELS['haiku'],
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return bool(message.content)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
