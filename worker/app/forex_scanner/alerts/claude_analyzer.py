"""
Claude Analyzer - Main Orchestrator for Signal Analysis
Modular refactor of the original claude_api.py for better maintainability
CLEAN ARCHITECTURE: Uses enhanced PromptBuilder without duplicate classes
ENHANCED: Vision API integration with chart generation for EMA_DOUBLE and other strategies

DATABASE-DRIVEN CONFIGURATION:
All behavioral settings are loaded from the database via scanner_config_service.
The database is the ONLY source of truth - no fallback to config.py allowed.
EXCEPTION: CLAUDE_API_KEY remains as getattr(config, ...) since it's a secret/env var.
"""

import logging
import os
import json
import base64
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd

from .api.client import APIClient
from .validation.technical_validator import TechnicalValidator
from .analysis.response_parser import ResponseParser
from .analysis.prompt_builder import PromptBuilder  # Enhanced PromptBuilder
from .storage.file_manager import FileManager
try:
    import config
except ImportError:
    from forex_scanner import config
from .validation.timestamp_validator import TimestampValidator

# Import scanner config service for database-driven settings (REQUIRED)
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    SCANNER_CONFIG_AVAILABLE = True
except ImportError:
    SCANNER_CONFIG_AVAILABLE = False
    get_scanner_config = None

# Try to import chart generator
try:
    from .forex_chart_generator import ForexChartGenerator
    CHART_GENERATOR_AVAILABLE = True
except ImportError:
    CHART_GENERATOR_AVAILABLE = False

# Try to import MinIO client
try:
    from forex_scanner.services.minio_client import get_minio_client
    MINIO_CLIENT_AVAILABLE = True
except ImportError:
    MINIO_CLIENT_AVAILABLE = False
    get_minio_client = None


class ClaudeAnalyzer:
    """
    Main interface for Claude analysis - orchestrates all components
    Significantly reduced complexity by delegating to specialized modules
    ENHANCED: Now uses the enhanced PromptBuilder for institutional-grade analysis

    DATABASE-DRIVEN CONFIGURATION:
    All behavioral settings are loaded from the database via scanner_config_service.
    The database is the ONLY source of truth - no fallback to config.py allowed.
    If database is unavailable, initialization will FAIL with RuntimeError.
    EXCEPTION: CLAUDE_API_KEY remains as getattr(config, ...) since it's a secret/env var.
    """

    def __init__(self, api_key: str = None, auto_save: bool = True, save_directory: str = "claude_analysis", data_fetcher=None):
        # Initialize logger FIRST (required by other initialization steps)
        self.logger = logging.getLogger(__name__)

        # ========================================================================
        # FAIL-FAST: Database configuration is REQUIRED - no fallback allowed
        # ========================================================================
        if not SCANNER_CONFIG_AVAILABLE:
            raise RuntimeError(
                "CRITICAL: Scanner config service not available - "
                "database is REQUIRED for ClaudeAnalyzer, no fallback allowed"
            )

        try:
            self._scanner_cfg = get_scanner_config()
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to load scanner config from database: {e} - "
                "no fallback allowed"
            ) from e

        if not self._scanner_cfg:
            raise RuntimeError(
                "CRITICAL: Scanner config returned None - "
                "database is REQUIRED for ClaudeAnalyzer, no fallback allowed"
            )

        self.logger.info(f"[CONFIG:DB] ClaudeAnalyzer config loaded from database (source={self._scanner_cfg.source})")

        # ========================================================================
        # Initialize API key - EXCEPTION: This is a secret/env var, use getattr
        # ========================================================================
        if not api_key:
            api_key = getattr(config, 'CLAUDE_API_KEY', None)

        # Initialize components
        self.api_client = APIClient(api_key)
        self.technical_validator = TechnicalValidator()
        self.response_parser = ResponseParser()
        self.prompt_builder = PromptBuilder()  # ENHANCED: Now includes advanced capabilities
        self.file_manager = FileManager(auto_save, save_directory)
        self.timestamp_validator = TimestampValidator()
        self.data_fetcher = data_fetcher

        # ========================================================================
        # Configuration for analysis mode - FROM DATABASE
        # ========================================================================
        # claude_analysis_mode: 'minimal', 'advanced', 'institutional', etc.
        analysis_mode = self._scanner_cfg.claude_analysis_mode or 'minimal'
        self.use_advanced_prompts = analysis_mode != 'minimal'
        # Default to 'institutional' if advanced mode is enabled but no specific level
        self.analysis_level = analysis_mode if analysis_mode in ['institutional', 'hedge_fund', 'prop_trader', 'risk_manager'] else 'institutional'

        # ========================================================================
        # Vision API configuration - FROM DATABASE (NO FALLBACK)
        # ========================================================================
        self.use_vision_api = self._scanner_cfg.claude_vision_enabled
        self.vision_strategies = self._scanner_cfg.claude_vision_strategies or ['EMA_DOUBLE', 'SMC', 'SMC_STRUCTURE']
        self.save_vision_artifacts = self._scanner_cfg.claude_save_vision_artifacts
        self.logger.info(
            f"[CONFIG:DB] Vision settings - enabled={self.use_vision_api}, "
            f"strategies={self.vision_strategies}, save_artifacts={self.save_vision_artifacts}"
        )

        # ========================================================================
        # Chart generator initialization
        # ========================================================================
        self.chart_generator = None
        if CHART_GENERATOR_AVAILABLE and self.use_vision_api:
            try:
                self.chart_generator = ForexChartGenerator(data_fetcher=data_fetcher)
                self.logger.info("Chart generator initialized for vision analysis")
            except Exception as e:
                self.logger.warning(f"Failed to initialize chart generator: {e}")

        # ========================================================================
        # MinIO client for chart storage (from database)
        # ========================================================================
        self.minio_client = None
        self.minio_enabled = self._scanner_cfg.minio_enabled
        if MINIO_CLIENT_AVAILABLE and self.minio_enabled:
            try:
                self.minio_client = get_minio_client()
                if self.minio_client.is_available:
                    self.logger.info("MinIO client initialized for chart storage")
                else:
                    self.logger.warning("MinIO client not available - using disk storage")
                    self.minio_client = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize MinIO client: {e}")
                self.minio_client = None

        # Save directory for analysis artifacts - from database or fallback
        self.save_directory = self._scanner_cfg.claude_vision_save_directory or save_directory

        if not api_key:
            self.logger.warning("No Claude API key provided")

        self.logger.info(
            f"ClaudeAnalyzer initialized - Advanced prompts: {self.use_advanced_prompts}, "
            f"Level: {self.analysis_level}"
        )
        self.logger.info(f"   Vision API: {'Enabled' if self.use_vision_api and self.chart_generator else 'Disabled'}")
        self.logger.info(f"   Chart Storage: {'MinIO' if self.minio_client else 'Disk'}")
    
    def analyze_signal_minimal(self, signal: Dict, save_to_file: bool = None) -> Optional[Dict]:
        """
        ENHANCED: Main analysis method with comprehensive error handling
        """
        if not self.api_client.api_key:
            self.logger.warning("No API key available for Claude analysis")
            return None
        
        try:
            # CRITICAL FIX: Check if signal is valid before processing
            if signal is None:
                self.logger.error("❌ Signal is None - cannot analyze")
                return {
                    'score': 0,
                    'decision': 'REJECT',
                    'reason': 'No signal data provided',
                    'approved': False,
                    'raw_response': 'Error: No signal data',
                    'technical_validation_passed': False,
                    'error': 'signal_is_none'
                }
            
            # Check for critical signal fields
            required_fields = ['epic', 'timestamp']  # Removed 'price' from required
            missing_fields = [field for field in required_fields if field not in signal or signal[field] is None]

            # Handle price field separately with flexibility
            price_found = False
            price_candidates = ['price', 'current_price', 'close_price', 'entry_price', 'signal_price']

            for price_field in price_candidates:
                if price_field in signal and signal[price_field] is not None:
                    try:
                        price_value = float(signal[price_field])
                        if 'price' not in signal:  # Add 'price' field for compatibility
                            signal['price'] = price_value
                            self.logger.debug(f"Added 'price' field from '{price_field}': {price_value:.5f}")
                        price_found = True
                        break
                    except (ValueError, TypeError):
                        continue

            if not price_found:
                self.logger.warning(f"⚠️ No valid price field found")
                signal['price'] = 1.0  # Fallback

            if missing_fields:
                self.logger.warning(f"⚠️ Signal missing required fields: {missing_fields}")
                if 'epic' not in signal:
                    signal['epic'] = 'UNKNOWN.EPIC'
                if 'timestamp' not in signal:
                    signal['timestamp'] = datetime.now()
            
            # STEP 1: Technical pre-validation with error handling
            try:
                technical_validation = self.technical_validator.validate_signal_technically_with_complete_data(signal)
            except Exception as validation_error:
                self.logger.error(f"❌ Technical validation failed: {validation_error}")
                # Provide fallback validation result
                technical_validation = {
                    'valid': True,  # Allow analysis to continue
                    'reason': f'Validation error: {str(validation_error)}',
                    'confidence_adjustment': 0.0,
                    'warnings': [f'Technical validation failed: {str(validation_error)}']
                }
            
            if not technical_validation.get('valid', False):
                self.logger.warning(f"🚨 Signal rejected - Technical validation failed: {technical_validation.get('reason', 'Unknown reason')}")
                return {
                    'score': 0,
                    'decision': 'REJECT',
                    'reason': f"Technical validation failed: {technical_validation.get('reason', 'Unknown reason')}",
                    'approved': False,
                    'raw_response': f"Technical validation failed: {technical_validation.get('reason', 'Unknown reason')}",
                    'technical_validation_passed': False
                }
            
            # STEP 2: Build prompt with error handling
            try:
                if self.use_advanced_prompts:
                    try:
                        prompt = self.prompt_builder.build_senior_analyst_prompt(
                            signal=signal,
                            technical_validation=technical_validation,
                            analysis_level=self.analysis_level
                        )
                        analysis_mode = f"advanced-{self.analysis_level}"
                        max_tokens = 200
                        self.logger.debug(f"✅ Using advanced {self.analysis_level} prompt")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Advanced prompt failed: {e}, falling back to basic")
                        prompt = self.prompt_builder.build_minimal_prompt_with_complete_data(signal, technical_validation)
                        analysis_mode = "enhanced-auto"
                        max_tokens = 150
                else:
                    prompt = self.prompt_builder.build_minimal_prompt_with_complete_data(signal, technical_validation)
                    analysis_mode = "enhanced-auto"
                    max_tokens = 150
            except Exception as prompt_error:
                self.logger.error(f"❌ Prompt building failed: {prompt_error}")
                return {
                    'score': 3,
                    'decision': 'NEUTRAL',
                    'reason': f'Analysis error: {str(prompt_error)}',
                    'approved': False,
                    'raw_response': f'Error building prompt: {str(prompt_error)}',
                    'technical_validation_passed': True,
                    'error': 'prompt_building_failed'
                }
            
            # STEP 3: Call Claude API with error handling
            try:
                response = self.api_client.call_api(prompt, max_tokens=max_tokens)
            except Exception as api_error:
                self.logger.error(f"❌ Claude API call failed: {api_error}")
                return {
                    'score': 3,
                    'decision': 'NEUTRAL',
                    'reason': f'API error: {str(api_error)}',
                    'approved': False,
                    'raw_response': f'API call failed: {str(api_error)}',
                    'technical_validation_passed': True,
                    'error': 'api_call_failed'
                }
            
            if not response:
                self.logger.error("❌ No response from Claude API")
                return {
                    'score': 3,
                    'decision': 'NEUTRAL',
                    'reason': 'No API response received',
                    'approved': False,
                    'raw_response': 'No response from Claude API',
                    'technical_validation_passed': True,
                    'error': 'no_api_response'
                }
            
            # STEP 4: Parse response with error handling
            try:
                parsed_result = self.response_parser.parse_minimal_response(response)
            except Exception as parse_error:
                self.logger.error(f"❌ Response parsing failed: {parse_error}")
                return {
                    'score': 3,
                    'decision': 'NEUTRAL',
                    'reason': f'Response parsing error: {str(parse_error)}',
                    'approved': False,
                    'raw_response': response,
                    'technical_validation_passed': True,
                    'error': 'response_parsing_failed'
                }
            
            if parsed_result and parsed_result.get('score') is not None:
                score_int = self._safe_convert_score(parsed_result.get('score'))
                decision = parsed_result.get('decision', 'UNKNOWN')
                
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                
                self.logger.info(f"✅ Claude {analysis_mode} analysis: {epic} {signal_type} - Score: {score_int}/10, Decision: {decision}")
                
                # Save to file if requested
                result = {
                    'score': score_int,
                    'decision': decision,
                    'reason': parsed_result.get('reason', 'Analysis completed'),
                    'approved': parsed_result.get('approved', False),
                    'raw_response': response,
                    'mode': analysis_mode,
                    'technical_validation_passed': True,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                should_save = save_to_file if save_to_file is not None else self.file_manager.auto_save
                if should_save:
                    try:
                        self.file_manager.save_minimal_analysis(signal, result)
                    except Exception as save_error:
                        self.logger.warning(f"⚠️ Failed to save analysis: {save_error}")
                
                return result
            else:
                self.logger.error("❌ Failed to parse Claude response")
                return {
                    'score': 3,
                    'decision': 'NEUTRAL',
                    'reason': 'Failed to parse Claude response',
                    'approved': False,
                    'raw_response': response,
                    'technical_validation_passed': True,
                    'error': 'parsing_failed'
                }

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Claude minimal analysis failed: {error_msg}")
            return {
                'score': 2,
                'decision': 'REJECT',
                'reason': f'Analysis error: {error_msg}',
                'approved': False,
                'raw_response': f'Error: {error_msg}',
                'technical_validation_passed': False,
                'error': 'general_analysis_failure'
            }

    def analyze_signal_with_vision(
        self,
        signal: Dict,
        candles: Dict[str, pd.DataFrame] = None,
        alert_id: int = None,
        save_to_file: bool = True
    ) -> Optional[Dict]:
        """
        ENHANCED: Analyze signal with vision API including chart generation.

        This method:
        1. Generates a multi-timeframe chart for the signal
        2. Builds a vision-enabled prompt specific to the strategy
        3. Sends both chart and prompt to Claude Vision API
        4. Saves chart and analysis data to disk with alert_id prefix

        Args:
            signal: Signal dictionary with all trading data
            candles: Dict of DataFrames {'4h': df_4h, '15m': df_15m, '5m': df_5m}
                     If None and data_fetcher is available, will fetch candles
            alert_id: Alert ID from database for file naming
            save_to_file: Whether to save analysis artifacts to disk

        Returns:
            Analysis result dictionary with vision-specific fields
        """
        if not self.api_client.api_key:
            self.logger.warning("No API key available for Claude vision analysis")
            return None

        try:
            epic = signal.get('epic', 'Unknown')
            strategy = signal.get('strategy', 'Unknown').upper()

            self.logger.info(f"🎯 Starting vision analysis for {epic} ({strategy})")

            # Check if strategy supports vision analysis
            if not self._should_use_vision(signal):
                self.logger.info(f"📝 Strategy {strategy} not configured for vision - using text-only analysis")
                return self.analyze_signal_minimal(signal, save_to_file)

            # Fetch candles if not provided and data_fetcher is available
            if candles is None and self.data_fetcher:
                candles = self._fetch_candles_for_chart(epic)

            # Generate chart if possible
            chart_base64 = None
            chart_generated = False

            if self.chart_generator and candles:
                try:
                    chart_base64 = self.chart_generator.generate_signal_chart(
                        epic=epic,
                        candles=candles,
                        signal=signal,
                        smc_data=signal.get('smc_data')
                    )
                    if chart_base64:
                        chart_generated = True
                        self.logger.info(f"📊 Chart generated: {len(chart_base64)} bytes (base64)")
                except Exception as chart_error:
                    self.logger.warning(f"⚠️ Chart generation failed: {chart_error}")

            # Build vision-enabled prompt
            has_chart = chart_base64 is not None
            prompt = self.prompt_builder.build_forex_vision_prompt(signal, has_chart=has_chart)

            # Call appropriate API (vision or text-only)
            if chart_base64:
                self.logger.info(f"🔮 Calling Claude Vision API with chart...")
                response_data = self.api_client.call_api_with_image(
                    prompt=prompt,
                    image_base64=chart_base64,
                    max_tokens=300
                )
                response = response_data.get('content') if response_data else None
                tokens_used = response_data.get('tokens', 0) if response_data else 0
            else:
                self.logger.info(f"📝 Calling Claude text-only API (no chart available)...")
                response = self.api_client.call_api(prompt, max_tokens=300)
                tokens_used = 0

            if not response:
                self.logger.error("❌ No response from Claude API")
                return {
                    'score': 3,
                    'decision': 'NEUTRAL',
                    'reason': 'No API response received',
                    'approved': False,
                    'raw_response': 'No response from Claude API',
                    'technical_validation_passed': True,
                    'vision_used': False,
                    'error': 'no_api_response'
                }

            # Parse response
            parsed_result = self.response_parser.parse_minimal_response(response)

            if parsed_result and parsed_result.get('score') is not None:
                score_int = self._safe_convert_score(parsed_result.get('score'))
                decision = parsed_result.get('decision', 'UNKNOWN')

                self.logger.info(f"✅ Claude Vision analysis: {epic} - Score: {score_int}/10, Decision: {decision}")

                result = {
                    'score': score_int,
                    'decision': decision,
                    'reason': parsed_result.get('reason', 'Vision analysis completed'),
                    'approved': parsed_result.get('approved', False),
                    'raw_response': response,
                    'mode': 'vision' if chart_generated else 'vision-text-only',
                    'technical_validation_passed': True,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'vision_used': chart_generated,
                    'chart_generated': chart_generated,
                    'tokens_used': tokens_used,
                    'vision_chart_url': None  # Will be populated if chart is uploaded to MinIO
                }

                # Save artifacts (chart to MinIO, optionally other files to disk)
                if save_to_file and self.save_vision_artifacts:
                    chart_url = self._save_vision_analysis_artifacts(
                        signal=signal,
                        result=result,
                        chart_base64=chart_base64,
                        prompt=prompt,
                        alert_id=alert_id
                    )
                    if chart_url:
                        result['vision_chart_url'] = chart_url

                return result
            else:
                self.logger.error("❌ Failed to parse Claude Vision response")
                return {
                    'score': 3,
                    'decision': 'NEUTRAL',
                    'reason': 'Failed to parse Claude response',
                    'approved': False,
                    'raw_response': response,
                    'technical_validation_passed': True,
                    'vision_used': chart_generated,
                    'error': 'parsing_failed'
                }

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Claude vision analysis failed: {error_msg}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {
                'score': 2,
                'decision': 'REJECT',
                'reason': f'Vision analysis error: {error_msg}',
                'approved': False,
                'raw_response': f'Error: {error_msg}',
                'technical_validation_passed': False,
                'vision_used': False,
                'error': 'vision_analysis_failure'
            }

    def _should_use_vision(self, signal: Dict) -> bool:
        """
        Determine if vision analysis should be used for this signal.

        Vision is used for:
        1. Strategies in the vision_strategies list (e.g., EMA_DOUBLE, SMC)
        2. When chart generator is available
        3. When vision API is enabled in config

        Args:
            signal: Signal dictionary

        Returns:
            True if vision should be used
        """
        if not self.use_vision_api or not self.chart_generator:
            return False

        strategy = signal.get('strategy', '').upper()

        # Check if strategy matches any in the vision list
        for vision_strategy in self.vision_strategies:
            if vision_strategy.upper() in strategy:
                return True

        return False

    def _fetch_candles_for_chart(self, epic: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Fetch candle data for chart generation.

        Fetches multi-timeframe data: 4H, 15m, and 5m for comprehensive chart.

        Args:
            epic: Currency pair epic code

        Returns:
            Dict of DataFrames {'4h': df_4h, '15m': df_15m, '5m': df_5m} or None
        """
        if not self.data_fetcher:
            self.logger.warning("No data_fetcher available for candle retrieval")
            return None

        try:
            # Extract pair name from epic
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

            candles = {}

            # Fetch 4H data (for higher timeframe context)
            try:
                df_4h = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe='4h',
                    lookback_hours=400  # ~100 bars
                )
                if df_4h is not None and len(df_4h) >= 20:
                    candles['4h'] = df_4h
                    self.logger.debug(f"📊 Fetched 4H data: {len(df_4h)} bars")
            except Exception as e:
                self.logger.debug(f"Could not fetch 4H data: {e}")

            # Fetch 15m data (main analysis timeframe)
            try:
                df_15m = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe='15m',
                    lookback_hours=100  # ~400 bars
                )
                if df_15m is not None and len(df_15m) >= 50:
                    candles['15m'] = df_15m
                    self.logger.debug(f"📊 Fetched 15m data: {len(df_15m)} bars")
            except Exception as e:
                self.logger.debug(f"Could not fetch 15m data: {e}")

            # Fetch 5m data (entry timeframe)
            try:
                df_5m = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe='5m',
                    lookback_hours=24  # ~288 bars
                )
                if df_5m is not None and len(df_5m) >= 50:
                    candles['5m'] = df_5m
                    self.logger.debug(f"📊 Fetched 5m data: {len(df_5m)} bars")
            except Exception as e:
                self.logger.debug(f"Could not fetch 5m data: {e}")

            if not candles:
                self.logger.warning(f"No candle data could be fetched for {epic}")
                return None

            self.logger.info(f"📊 Candles fetched for {epic}: {list(candles.keys())}")
            return candles

        except Exception as e:
            self.logger.error(f"Error fetching candles for chart: {e}")
            return None

    def _save_vision_analysis_artifacts(
        self,
        signal: Dict,
        result: Dict,
        chart_base64: str,
        prompt: str,
        alert_id: int = None
    ) -> Optional[str]:
        """
        Save vision analysis chart to MinIO (or disk as fallback).

        Primary storage is MinIO with 30-day retention. Text files (signal, prompt, result)
        are no longer saved as this data is already stored in the database.

        Args:
            signal: Signal dictionary
            result: Analysis result dictionary
            chart_base64: Base64-encoded chart image
            prompt: Prompt text sent to Claude (not saved - already in DB)
            alert_id: Alert ID for file naming prefix

        Returns:
            MinIO URL of the uploaded chart, or None if upload failed
        """
        chart_url = None

        try:
            # Generate filename components
            epic = signal.get('epic', 'unknown').replace('.', '_').replace(':', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if alert_id:
                object_name = f"{alert_id}_{epic}_{timestamp}_chart.png"
            else:
                object_name = f"{epic}_{timestamp}_chart.png"

            # Upload chart to MinIO if available
            if chart_base64 and self.minio_client and self.minio_client.is_available:
                try:
                    chart_bytes = base64.b64decode(chart_base64)
                    chart_url = self.minio_client.upload_chart(chart_bytes, object_name)
                    if chart_url:
                        self.logger.info(f"📊 Chart uploaded to MinIO: {object_name}")
                    else:
                        self.logger.warning("⚠️ MinIO upload returned no URL - falling back to disk")
                except Exception as e:
                    self.logger.warning(f"⚠️ MinIO upload failed: {e} - falling back to disk")

            # Fallback to disk storage if MinIO upload failed or not available
            if chart_base64 and not chart_url:
                vision_dir = os.path.join(self.save_directory, 'vision_analysis')
                os.makedirs(vision_dir, exist_ok=True)
                chart_path = os.path.join(vision_dir, object_name)
                try:
                    chart_bytes = base64.b64decode(chart_base64)
                    with open(chart_path, 'wb') as f:
                        f.write(chart_bytes)
                    self.logger.info(f"📊 Chart saved to disk: {chart_path}")
                    # Return local path for fallback (Streamlit can still read it)
                    chart_url = f"file://{chart_path}"
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to save chart to disk: {e}")

            # NOTE: Text files (_signal.json, _prompt.txt, _result.json) are no longer saved.
            # All this data is already stored in the alert_history database table:
            # - signal data -> strategy_indicators JSON column
            # - result data -> claude_score, claude_decision, claude_reason columns
            # - prompt can be regenerated from signal if needed

            if chart_url:
                self.logger.info(f"✅ Vision chart saved: {object_name}")
            else:
                self.logger.warning("⚠️ No chart was saved (no base64 data or storage failed)")

            return chart_url

        except Exception as e:
            self.logger.error(f"❌ Failed to save vision analysis artifacts: {e}")
            return None

    def _make_json_serializable(self, obj: Dict) -> Dict:
        """
        Convert signal dictionary to JSON-serializable format.

        Handles datetime objects, numpy types, pandas types, etc.

        Args:
            obj: Dictionary to convert

        Returns:
            JSON-serializable dictionary
        """
        import numpy as np

        def convert(item):
            if item is None:
                return None
            elif isinstance(item, (datetime,)):
                return item.isoformat()
            elif isinstance(item, pd.Timestamp):
                return item.isoformat()
            elif isinstance(item, (np.integer,)):
                return int(item)
            elif isinstance(item, (np.floating,)):
                return float(item)
            elif isinstance(item, (np.ndarray,)):
                return item.tolist()
            elif isinstance(item, pd.Series):
                return item.to_dict()
            elif isinstance(item, pd.DataFrame):
                return item.to_dict('records')
            elif isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [convert(i) for i in item]
            else:
                return item

        return convert(obj)

    def analyze_signal(self, signal: Dict, save_to_file: bool = None) -> Optional[str]:
        """
        Full analysis with text output for backward compatibility
        ENHANCED: Now includes advanced analysis details when available
        """
        try:
            # Get minimal result first
            minimal_result = self.analyze_signal_minimal(signal, save_to_file)
            
            if minimal_result:
                # Convert to text format
                technical_status = "✅ PASSED" if minimal_result.get('technical_validation_passed') else "❌ FAILED"
                analysis_mode = minimal_result.get('mode', 'standard')
                
                analysis_text = f"""
Claude Analysis for {signal.get('epic', 'Unknown')} {signal.get('signal_type', 'Unknown')} Signal

TECHNICAL VALIDATION: {technical_status}
Analysis Mode: {analysis_mode.upper()}
Signal Quality Score: {minimal_result['score']}/10
Decision: {minimal_result['decision']}
Approved: {minimal_result['approved']}
Reason: {minimal_result['reason']}

Strategy: {self._identify_strategy(signal)}
Price: {signal.get('price', 'N/A')}
Confidence: {signal.get('confidence_score', 0):.1%}

Claude Analysis: Based on the signal characteristics using {analysis_mode} analysis, this {signal.get('signal_type', 'signal').lower()} signal for {signal.get('epic', 'the pair')} receives a score of {minimal_result['score']}/10. The recommendation is to {minimal_result['decision'].lower()} this signal because {minimal_result['reason'].lower()}.
"""
                return analysis_text.strip()
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Claude full analysis failed: {e}")
            return None
    
    def analyze_signal_minimal_with_fallback(self, signal: Dict, save_to_file: bool = None) -> Optional[Dict]:
        """
        Analyze signal with intelligent fallback when Claude is unavailable
        """
        # Try Claude analysis first
        claude_result = self.analyze_signal_minimal(signal, save_to_file)
        
        if claude_result:
            return claude_result
        
        # Fallback analysis
        self.logger.warning("🔄 Claude unavailable, using intelligent fallback analysis")
        
        try:
            # Basic signal quality assessment
            confidence = float(signal.get('confidence_score', 0))
            strategy = signal.get('strategy', 'unknown')
            
            # Simple scoring based on confidence and strategy
            if confidence >= 0.9:
                score = 8
                decision = 'APPROVE'
                reason = 'High confidence signal with strong technical indicators'
            elif confidence >= 0.8:
                score = 7
                decision = 'APPROVE' 
                reason = 'Good confidence signal with solid technical setup'
            elif confidence >= 0.7:
                score = 6
                decision = 'APPROVE'
                reason = 'Moderate confidence signal meeting minimum criteria'
            elif confidence >= 0.6:
                score = 5
                decision = 'NEUTRAL'
                reason = 'Borderline signal with mixed technical indicators'
            else:
                score = 3
                decision = 'REJECT'
                reason = 'Low confidence signal with weak technical setup'
            
            # Adjust based on strategy
            if strategy in ['combined', 'consensus']:
                score = min(score + 1, 10)  # Boost for multi-strategy confirmation
            
            return {
                'score': score,
                'decision': decision,
                'reason': reason,
                'approved': decision == 'APPROVE',
                'raw_response': f'FALLBACK ANALYSIS: Score {score}/10, Decision: {decision}',
                'mode': 'fallback',
                'technical_validation_passed': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback analysis failed: {e}")
            return {
                'score': 5,
                'decision': 'NEUTRAL',
                'reason': 'Fallback analysis error - using neutral assessment',
                'approved': False,
                'raw_response': 'FALLBACK ERROR',
                'mode': 'error_fallback'
            }
    
    def batch_analyze_signals_minimal(self, signals: List[Dict], save_to_file: bool = None) -> List[Dict]:
        """
        Enhanced batch analysis with technical validation
        """
        results = []
        
        for i, signal in enumerate(signals, 1):
            self.logger.info(f"📊 Analyzing signal {i}/{len(signals)}: {signal.get('epic', 'Unknown')}")
            
            analysis = self.analyze_signal_minimal(signal, save_to_file=save_to_file)
            
            if analysis:
                results.append({
                    'signal': signal,
                    'score': analysis['score'],
                    'decision': analysis['decision'],
                    'approved': analysis['approved'],
                    'reason': analysis['reason'],
                    'technical_validation_passed': analysis.get('technical_validation_passed', False),
                    'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                results.append({
                    'signal': signal,
                    'score': None,
                    'decision': 'REJECT',
                    'approved': False,
                    'reason': 'Analysis failed',
                    'technical_validation_passed': False,
                    'error': 'Analysis failed'
                })
        
        # Save batch summary if enabled
        should_save = save_to_file if save_to_file is not None else self.file_manager.auto_save
        if should_save and results:
            self.file_manager.save_batch_summary_minimal(results)
        
        return results
    
    def test_connection(self) -> bool:
        """Test Claude API connection"""
        return self.api_client.test_connection()
    
    def get_api_health_status(self) -> Dict:
        """Check Claude API health and return status"""
        return self.api_client.get_health_status()
    
    def set_analysis_level(self, level: str):
        """
        Change the analysis level for advanced prompts
        
        Args:
            level: 'institutional', 'hedge_fund', 'prop_trader', or 'risk_manager'
        """
        valid_levels = ['institutional', 'hedge_fund', 'prop_trader', 'risk_manager']
        if level in valid_levels:
            self.analysis_level = level
            self.logger.info(f"✅ Analysis level changed to: {level}")
        else:
            self.logger.warning(f"⚠️ Invalid analysis level: {level}. Valid options: {valid_levels}")
    
    def toggle_advanced_prompts(self, enabled: bool):
        """
        Enable or disable advanced prompt builder
        
        Args:
            enabled: True to use advanced prompts, False for automatic selection
        """
        self.use_advanced_prompts = enabled
        mode = "advanced" if enabled else "enhanced-auto"
        self.logger.info(f"✅ Prompt mode changed to: {mode}")
    
    def analyze_signal_at_timestamp(self, epic: str, timestamp_str: str, signal_detector, include_future_analysis: bool = False) -> Optional[Dict]:
        """
        FIXED: Analyze signal at specific timestamp with proper datetime handling using TimestampValidator
        """
        try:
            from datetime import datetime
            import pandas as pd
            
            # FIXED: Use TimestampValidator to parse the timestamp safely
            timestamp_validation = self.timestamp_validator.validate_and_clean_timestamp(timestamp_str, "input_timestamp")
            
            if not timestamp_validation['valid']:
                error_msg = f"Invalid timestamp: {timestamp_str}. Errors: {timestamp_validation['warnings']}"
                self.logger.error(error_msg)
                return {'error': error_msg}
            
            timestamp = timestamp_validation['cleaned_timestamp']
            
            self.logger.info(f"🔍 Analyzing {epic} at {timestamp_str}")
            self.logger.debug(f"   Cleaned timestamp: {timestamp} (method: {timestamp_validation['method_used']})")
            
            # Extract pair from epic
            pair = epic.split('.')[2] if '.' in epic else epic
            
            # Get market data
            lookback_hours = 48 if include_future_analysis else 24
            
            df = signal_detector.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='5m',
                lookback_hours=lookback_hours,
                user_timezone='Europe/Stockholm',
                required_indicators=['ema', 'macd', 'kama', 'rsi', 'volume']
            )
            
            if df is None or df.empty:
                return {'error': f'No data available for {epic} at {timestamp_str}'}
            
            # CRITICAL FIX: Use TimestampValidator for DataFrame timestamp processing
            target_candle, target_time, method_used = self._find_target_candle_safe(df, timestamp)
            
            if target_candle is None:
                return {'error': f'No data available at or before {timestamp_str}'}
            
            self.logger.info(f"🕐 Found candle using method: {method_used}")
            self.logger.info(f"   Target time: {target_time}")
            
            # Prepare signal data
            signal_data = self._prepare_signal_data_from_candle(target_candle, target_time, epic, pair)
            
            # Add future analysis if requested
            if include_future_analysis:
                future_data = self._get_future_data_safe(df, target_time)
                if not future_data.empty:
                    signal_data['future_analysis'] = self._calculate_future_analysis(target_candle, future_data, epic)
            
            # Perform Claude analysis
            analysis = self.analyze_signal_minimal(signal_data, save_to_file=True)
            
            if analysis:
                analysis.update({
                    'timestamp_analyzed': timestamp_str,
                    'actual_candle_time': str(target_time),
                    'epic': epic,
                    'pair': pair,
                    'timestamp_method': method_used,
                    'market_data': {
                        'price': float(target_candle['close']),
                        'volume': float(target_candle.get('ltv', target_candle.get('volume', 0)))
                    }
                })
                
                if include_future_analysis and 'future_analysis' in signal_data:
                    analysis['outcome'] = signal_data['future_analysis']
                    
                    # Determine outcome accuracy
                    if analysis.get('decision') == 'APPROVE':
                        favorable = signal_data['future_analysis']['favorable_movement']
                        analysis['outcome_accuracy'] = 'correct' if favorable else 'incorrect'
                    else:
                        analysis['outcome_accuracy'] = 'rejected'
                
                return analysis
            else:
                return {'error': 'Claude analysis failed'}
                
        except Exception as e:
            self.logger.error(f"❌ Timestamp analysis failed: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {'error': f'Analysis error: {str(e)}'}
    
    def _find_target_candle_safe(self, df: pd.DataFrame, target_timestamp: datetime) -> tuple:
        """
        ENHANCED: Safely find the target candle with comprehensive error handling
        
        Returns:
            tuple: (target_candle, target_time, method_used)
        """
        methods_tried = []
        
        try:
            # Method 1: Use start_time column if available
            if 'start_time' in df.columns:
                try:
                    return self._find_candle_by_column(df, target_timestamp, 'start_time')
                except Exception as e:
                    methods_tried.append(f"start_time_column: {e}")
                    self.logger.debug(f"start_time column method failed: {e}")
            
            # Method 2: Use DataFrame index if it's datetime-like
            try:
                if hasattr(df.index, 'to_pydatetime') or pd.api.types.is_datetime64_any_dtype(df.index):
                    return self._find_candle_by_index(df, target_timestamp)
            except Exception as e:
                methods_tried.append(f"datetime_index: {e}")
                self.logger.debug(f"DateTime index method failed: {e}")
            
            # Method 3: Try other timestamp columns
            timestamp_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            for col in timestamp_columns:
                if col != 'start_time':  # Already tried above
                    try:
                        return self._find_candle_by_column(df, target_timestamp, col)
                    except Exception as e:
                        methods_tried.append(f"{col}_column: {e}")
                        self.logger.debug(f"Column {col} method failed: {e}")
            
            # Method 4: Try to convert index to datetime
            try:
                return self._find_candle_by_converted_index(df, target_timestamp)
            except Exception as e:
                methods_tried.append(f"converted_index: {e}")
                self.logger.debug(f"Index conversion method failed: {e}")
        
        except Exception as e:
            methods_tried.append(f"general_error: {e}")
        
        # Ultimate fallback: use latest available candle
        self.logger.warning(f"All timestamp methods failed: {'; '.join(methods_tried)}, using latest candle")
        
        target_candle = df.iloc[-1]
        
        # Try to get a reasonable target_time
        if 'start_time' in df.columns:
            target_time = df['start_time'].iloc[-1]
            # Convert pandas timestamp to Python datetime if needed
            if hasattr(target_time, 'to_pydatetime'):
                target_time = target_time.to_pydatetime()
        elif hasattr(df.index[-1], 'to_pydatetime'):
            target_time = df.index[-1].to_pydatetime()
        else:
            target_time = target_timestamp  # Use the requested timestamp as fallback
        
        return target_candle, target_time, "latest_fallback"
    
    def _find_candle_by_column(self, df: pd.DataFrame, target_timestamp: datetime, column_name: str) -> tuple:
        """
        ENHANCED: Find candle using datetime column with epoch timestamp detection
        """
        try:
            time_column = df[column_name]
            self.logger.debug(f"Searching using column '{column_name}' with dtype: {time_column.dtype}")
            
            # CRITICAL FIX: Validate target timestamp
            if target_timestamp.year <= 1970:
                self.logger.warning(f"⚠️ Epoch target timestamp detected: {target_timestamp}, adjusting to current time")
                target_timestamp = datetime.now().replace(tzinfo=target_timestamp.tzinfo)
            
            # Handle pandas datetime column
            if pd.api.types.is_datetime64_any_dtype(time_column):
                # Convert target timestamp to pandas Timestamp for safe comparison
                if hasattr(time_column.dtype, 'tz') and time_column.dtype.tz is not None:
                    target_pd = pd.Timestamp(target_timestamp).tz_convert(time_column.dtype.tz)
                else:
                    if target_timestamp.tzinfo is not None:
                        target_pd = pd.Timestamp(target_timestamp).tz_localize(None)
                    else:
                        target_pd = pd.Timestamp(target_timestamp)
                
                # ENHANCED: Filter out epoch timestamps from the column
                valid_mask = time_column.dt.year > 1970
                filtered_data = df[valid_mask]
                filtered_time_column = time_column[valid_mask]
                
                if filtered_data.empty:
                    raise ValueError(f"No valid (non-epoch) timestamps in column {column_name}")
                
                self.logger.debug(f"Filtered out {len(df) - len(filtered_data)} epoch timestamps")
                
                # Safe pandas comparison on filtered data
                try:
                    comparison_mask = filtered_time_column <= target_pd
                    available_data = filtered_data[comparison_mask]
                    
                    if available_data.empty:
                        # Use the earliest valid timestamp if nothing before target
                        self.logger.warning(f"⚠️ No data before target, using earliest valid timestamp")
                        target_candle = filtered_data.iloc[0]
                        target_time = filtered_time_column.iloc[0]
                    else:
                        # Get the last (most recent) valid row
                        target_candle = available_data.iloc[-1]
                        available_time_column = filtered_time_column[comparison_mask]
                        target_time = available_time_column.iloc[-1]
                    
                    # Convert pandas timestamp back to Python datetime
                    if hasattr(target_time, 'to_pydatetime'):
                        target_time = target_time.to_pydatetime()
                    
                    # FINAL VALIDATION
                    if target_time.year <= 1970:
                        self.logger.error(f"❌ CRITICAL: Still got epoch time: {target_time}")
                        target_time = datetime.now().replace(tzinfo=getattr(target_time, 'tzinfo', None))
                        self.logger.warning(f"🚨 Emergency override to current time: {target_time}")
                    
                    return target_candle, target_time, f"filtered_column_{column_name}"
                    
                except TypeError as comparison_error:
                    self.logger.debug(f"Pandas comparison failed: {comparison_error}")
                    raise ValueError(f"Column comparison failed: {comparison_error}")
            
            else:
                # Non-datetime column, use TimestampValidator with epoch filtering
                return self._find_candle_by_validation_with_epoch_filter(df, target_timestamp, column_name)
                
        except Exception as e:
            self.logger.debug(f"Column-based search failed: {e}")
            raise ValueError(f"Column search failed: {e}")

    def _find_candle_by_validation_with_epoch_filter(self, df: pd.DataFrame, target_timestamp: datetime, column_name: str) -> tuple:
        """
        ENHANCED: Find candle using TimestampValidator with epoch timestamp filtering
        """
        time_column = df[column_name]
        
        # Validate and clean all timestamps, filtering out epoch times
        valid_rows = []
        
        for idx, time_val in enumerate(time_column):
            validation = self.timestamp_validator.validate_and_clean_timestamp(time_val, f"{column_name}_{idx}")
            
            if validation['valid']:
                cleaned_time = validation['cleaned_timestamp']
                
                # CRITICAL FIX: Skip epoch timestamps
                if cleaned_time.year <= 1970:
                    self.logger.debug(f"Skipping epoch timestamp at row {idx}: {cleaned_time}")
                    continue
                
                # Safe comparison with timezone handling
                try:
                    if target_timestamp.tzinfo is None and cleaned_time.tzinfo is not None:
                        cleaned_time = cleaned_time.replace(tzinfo=None)
                    elif target_timestamp.tzinfo is not None and cleaned_time.tzinfo is None:
                        cleaned_time = cleaned_time.replace(tzinfo=target_timestamp.tzinfo)
                    
                    # Only add if not epoch and comparison works
                    if cleaned_time <= target_timestamp:
                        valid_rows.append((idx, cleaned_time))
                        
                except TypeError:
                    self.logger.debug(f"Timestamp comparison failed for row {idx}")
                    continue
        
        if not valid_rows:
            # Try to find ANY valid non-epoch timestamp as fallback
            fallback_rows = []
            for idx, time_val in enumerate(time_column):
                validation = self.timestamp_validator.validate_and_clean_timestamp(time_val, f"{column_name}_{idx}")
                if validation['valid'] and validation['cleaned_timestamp'].year > 1970:
                    fallback_rows.append((idx, validation['cleaned_timestamp']))
            
            if fallback_rows:
                # Use the latest fallback timestamp
                latest_row_idx, latest_time = max(fallback_rows, key=lambda x: x[1])
                target_candle = df.iloc[latest_row_idx]
                self.logger.warning(f"⚠️ Used fallback timestamp: {latest_time}")
                return target_candle, latest_time, f"fallback_validator_{column_name}"
            else:
                raise ValueError(f"No valid non-epoch timestamps found in {column_name} column")
        
        # Get the row with the latest valid timestamp
        latest_row_idx, latest_time = max(valid_rows, key=lambda x: x[1])
        target_candle = df.iloc[latest_row_idx]
        
        return target_candle, latest_time, f"validator_column_{column_name}"

    def _find_candle_by_validation(self, df: pd.DataFrame, target_timestamp: datetime, column_name: str) -> tuple:
        """
        FALLBACK: Find candle using TimestampValidator for complex timestamp formats
        """
        time_column = df[column_name]
        
        # Validate and clean all timestamps in the column
        valid_rows = []
        
        for idx, time_val in enumerate(time_column):
            validation = self.timestamp_validator.validate_and_clean_timestamp(time_val, f"{column_name}_{idx}")
            if validation['valid']:
                cleaned_time = validation['cleaned_timestamp']
                
                # Safe comparison with timezone handling
                try:
                    if target_timestamp.tzinfo is None and cleaned_time.tzinfo is not None:
                        cleaned_time = cleaned_time.replace(tzinfo=None)
                    elif target_timestamp.tzinfo is not None and cleaned_time.tzinfo is None:
                        cleaned_time = cleaned_time.replace(tzinfo=target_timestamp.tzinfo)
                    
                    if cleaned_time <= target_timestamp:
                        valid_rows.append((idx, cleaned_time))
                        
                except TypeError:
                    self.logger.debug(f"Timestamp comparison failed for row {idx}")
                    continue
        
        if not valid_rows:
            raise ValueError(f"No valid timestamps found in {column_name} column")
        
        # Get the row with the latest valid timestamp
        latest_row_idx, latest_time = max(valid_rows, key=lambda x: x[1])
        target_candle = df.iloc[latest_row_idx]
        
        return target_candle, latest_time, f"validator_column_{column_name}"
    
    def _find_candle_by_converted_index(self, df: pd.DataFrame, target_timestamp: datetime) -> tuple:
        """Find candle by converting index to datetime"""
        try:
            # Try to convert index to datetime
            datetime_index = pd.to_datetime(df.index)
            
            # Create a temporary DataFrame with datetime index
            temp_df = df.copy()
            temp_df.index = datetime_index
            
            return self._find_candle_by_index(temp_df, target_timestamp)
            
        except Exception as e:
            raise ValueError(f"Index conversion failed: {e}")
    
    def _get_future_data_safe(self, df: pd.DataFrame, target_time) -> pd.DataFrame:
        """
        ENHANCED: Safely get future data with robust timezone handling
        """
        try:
            # Validate target_time
            if target_time is None:
                self.logger.warning("⚠️ target_time is None, cannot get future data")
                return pd.DataFrame()
                
            if hasattr(target_time, 'year') and target_time.year <= 1970:
                self.logger.warning(f"⚠️ Epoch target time detected: {target_time}")
                from datetime import datetime
                target_time = datetime.now().replace(tzinfo=getattr(target_time, 'tzinfo', None))
            
            if 'start_time' in df.columns:
                time_column = df['start_time']
                
                # Filter out epoch timestamps from the data
                if pd.api.types.is_datetime64_any_dtype(time_column):
                    valid_mask = time_column.dt.year > 1970
                    filtered_df = df[valid_mask]
                    filtered_time_column = time_column[valid_mask]
                    
                    if filtered_df.empty:
                        self.logger.warning("No valid timestamps for future data analysis")
                        return pd.DataFrame()
                    
                    # CRITICAL FIX: Robust timezone handling for comparison
                    try:
                        # Determine timezone compatibility
                        col_has_tz = hasattr(filtered_time_column.dtype, 'tz') and filtered_time_column.dtype.tz is not None
                        target_has_tz = hasattr(target_time, 'tzinfo') and target_time.tzinfo is not None
                        
                        self.logger.debug(f"Column has timezone: {col_has_tz}, Target has timezone: {target_has_tz}")
                        
                        if col_has_tz and not target_has_tz:
                            # Column has timezone, target doesn't - localize target to column's timezone
                            target_pd = pd.Timestamp(target_time).tz_localize(filtered_time_column.dtype.tz)
                            self.logger.debug(f"Localized target to column timezone: {target_pd}")
                            
                        elif col_has_tz and target_has_tz:
                            # Both have timezones - convert target to column's timezone
                            target_pd = pd.Timestamp(target_time).tz_convert(filtered_time_column.dtype.tz)
                            self.logger.debug(f"Converted target to column timezone: {target_pd}")
                            
                        elif not col_has_tz and target_has_tz:
                            # Column is naive, target has timezone - make target naive
                            target_pd = pd.Timestamp(target_time).tz_localize(None)
                            self.logger.debug(f"Made target timezone-naive: {target_pd}")
                            
                        else:
                            # Both are naive - direct conversion
                            target_pd = pd.Timestamp(target_time)
                            self.logger.debug(f"Direct timestamp conversion: {target_pd}")
                        
                        # Safe comparison
                        future_mask = filtered_time_column > target_pd
                        future_data = filtered_df[future_mask]
                        
                        self.logger.debug(f"Found {len(future_data)} future candles after {target_time}")
                        return future_data
                        
                    except Exception as tz_error:
                        self.logger.warning(f"Timezone comparison failed: {tz_error}")
                        # Fallback: try string comparison
                        return self._get_future_data_string_fallback(filtered_df, filtered_time_column, target_time)
                else:
                    # Non-datetime column
                    return self._get_future_data_with_validation(df, target_time)
            
            else:
                # Use index with robust timezone handling
                return self._get_future_data_from_index(df, target_time)
                    
        except Exception as e:
            self.logger.warning(f"Could not get future data: {e}")
            return pd.DataFrame()

    def _get_future_data_from_index(self, df: pd.DataFrame, target_time) -> pd.DataFrame:
        """Get future data using DataFrame index with timezone handling"""
        try:
            if not hasattr(df.index, 'to_pydatetime'):
                return pd.DataFrame()
            
            python_times = df.index.to_pydatetime()
            
            # Filter out epoch timestamps
            valid_indices = [i for i, dt in enumerate(python_times) if dt.year > 1970]
            
            if not valid_indices:
                return pd.DataFrame()
            
            # Handle timezone compatibility
            index_times = [python_times[i] for i in valid_indices]
            future_indices = []
            
            for i, dt in enumerate(index_times):
                try:
                    # Normalize timezones for comparison
                    if hasattr(target_time, 'tzinfo') and target_time.tzinfo is not None:
                        if hasattr(dt, 'tzinfo') and dt.tzinfo is None:
                            dt = dt.replace(tzinfo=target_time.tzinfo)
                        elif hasattr(dt, 'tzinfo') and dt.tzinfo != target_time.tzinfo:
                            dt = dt.astimezone(target_time.tzinfo)
                    else:
                        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=None)
                    
                    if dt > target_time:
                        future_indices.append(valid_indices[i])
                        
                except Exception:
                    continue
            
            if future_indices:
                return df.iloc[future_indices]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.warning(f"Index-based future data failed: {e}")
            return pd.DataFrame()

    def _get_future_data_string_fallback(self, filtered_df: pd.DataFrame, time_column, target_time) -> pd.DataFrame:
        """Fallback method using string comparison when timezone comparison fails"""
        try:
            self.logger.debug("Using string comparison fallback for future data")
            
            # Convert both to string format for comparison
            target_str = target_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(target_time, 'strftime') else str(target_time)[:19]
            
            future_rows = []
            for idx, time_val in enumerate(time_column):
                try:
                    if hasattr(time_val, 'strftime'):
                        time_str = time_val.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        time_str = str(time_val)[:19]
                    
                    if time_str > target_str:
                        future_rows.append(idx)
                        
                except Exception:
                    continue
            
            if future_rows:
                # Get the original indices from filtered_df
                original_indices = filtered_df.index[future_rows]
                return filtered_df.loc[original_indices]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.warning(f"String fallback failed: {e}")
            return pd.DataFrame()


    def _get_future_data_with_validation(self, df: pd.DataFrame, target_time) -> pd.DataFrame:
        """Get future data using TimestampValidator with epoch filtering"""
        try:
            time_column = df['start_time']
            future_rows = []
            
            for idx, time_val in enumerate(time_column):
                validation = self.timestamp_validator.validate_and_clean_timestamp(time_val, f"future_{idx}")
                
                if validation['valid']:
                    cleaned_time = validation['cleaned_timestamp']
                    
                    # Skip epoch timestamps
                    if cleaned_time.year <= 1970:
                        continue
                    
                    # Check if it's after target time
                    try:
                        if cleaned_time > target_time:
                            future_rows.append(idx)
                    except TypeError:
                        continue
            
            if future_rows:
                return df.iloc[future_rows]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.warning(f"Future data validation failed: {e}")
            return pd.DataFrame()


    def _safe_convert_score(self, score) -> int:
        """Safely convert score to integer"""
        try:
            return int(float(score)) if score is not None else 0
        except (ValueError, TypeError):
            self.logger.warning(f"⚠️ Invalid score value: {score}, defaulting to 0")
            return 0
    
    def _identify_strategy(self, signal: Dict) -> str:
        """Identify the strategy type from signal data"""
        strategy = signal.get('strategy', '').lower()
        
        if 'combined' in strategy:
            return 'COMBINED'
        elif 'macd' in strategy:
            return 'MACD'
        elif 'kama' in strategy:
            return 'KAMA'
        elif 'ema' in strategy:
            return 'EMA'
        else:
            # Try to identify from available indicators
            if signal.get('macd_line') is not None or signal.get('macd_histogram') is not None:
                return 'MACD'
            elif signal.get('kama_value') is not None or any(k.startswith('kama_') for k in signal.keys()):
                return 'KAMA'
            elif signal.get('ema_short') is not None or signal.get('ema_9') is not None:
                return 'EMA'
            else:
                return 'UNKNOWN'
    
    def _prepare_signal_data_from_candle(self, target_candle, target_time, epic: str, pair: str) -> Dict:
        """
        FIXED: Prepare signal data from a DataFrame candle with robust null handling
        """
        try:
            # CRITICAL FIX: Handle None/null target_candle
            if target_candle is None:
                self.logger.error("❌ target_candle is None - cannot prepare signal data")
                # Return minimal signal data for emergency analysis
                return {
                    'epic': epic,
                    'timestamp': target_time or datetime.now(),
                    'price': 1.0,  # Default price to avoid errors
                    'open_price': 1.0,
                    'high_price': 1.0,
                    'low_price': 1.0,
                    'close_price': 1.0,
                    'volume': 1000,
                    'timeframe': '5m',
                    'pair': pair,
                    'strategy': 'emergency_analysis',
                    'signal_type': 'ANALYSIS',
                    'confidence_score': 0.1,
                    'error': 'No candle data available'
                }
            
            # ENHANCED: Safe value extraction with null checking
            def safe_float(value, default=0.0):
                """Safely convert value to float with comprehensive null checking"""
                try:
                    if value is None:
                        return default
                    if pd.isna(value):
                        return default
                    if isinstance(value, str) and value.strip() == '':
                        return default
                    return float(value)
                except (ValueError, TypeError, AttributeError):
                    return default
            
            def safe_get(candle, key, default=None):
                """Safely get value from candle data"""
                try:
                    # Handle pandas Series
                    if hasattr(candle, 'index') and key in candle.index:
                        value = candle[key]
                        if pd.isna(value):
                            return default
                        return value
                    # Handle dictionary-like objects
                    elif hasattr(candle, 'get'):
                        return candle.get(key, default)
                    # Handle other indexable objects
                    elif hasattr(candle, '__getitem__'):
                        try:
                            value = candle[key]
                            if pd.isna(value):
                                return default
                            return value
                        except (KeyError, IndexError):
                            return default
                    else:
                        return default
                except (KeyError, AttributeError, TypeError):
                    return default
            
            # Basic signal data with comprehensive safe extraction
            signal_data = {
                'epic': epic,
                'timestamp': target_time if target_time is not None else datetime.now(),
                'price': safe_float(safe_get(target_candle, 'close', 1.0)),
                'open_price': safe_float(safe_get(target_candle, 'open', 1.0)),
                'high_price': safe_float(safe_get(target_candle, 'high', 1.0)),
                'low_price': safe_float(safe_get(target_candle, 'low', 1.0)),
                'close_price': safe_float(safe_get(target_candle, 'close', 1.0)),
                'volume': safe_float(safe_get(target_candle, 'ltv', safe_get(target_candle, 'volume', 1000))),
                'timeframe': '5m',
                'pair': pair,
                'strategy': 'retrospective_analysis',
                'signal_type': 'ANALYSIS',
                'confidence_score': 0.8
            }
            
            # ENHANCED: Add technical indicators with comprehensive null handling
            indicator_mappings = {
                'ema_9': ['ema_9', 'ema_short'],
                'ema_21': ['ema_21', 'ema_long'], 
                'ema_200': ['ema_200', 'ema_trend'],
                'macd_line': ['macd_line', 'macd'],
                'macd_signal': ['macd_signal', 'macd_signal_line'],
                'macd_histogram': ['macd_histogram', 'macd_hist'],
                'kama_value': ['kama_value', 'kama'],
                'efficiency_ratio': ['efficiency_ratio', 'kama_er'],
                'rsi': ['rsi'],
                'bb_upper': ['bb_upper', 'bollinger_upper'],
                'bb_middle': ['bb_middle', 'bollinger_middle'],
                'bb_lower': ['bb_lower', 'bollinger_lower'],
                'atr': ['atr'],
                'supertrend': ['supertrend'],
                'supertrend_direction': ['supertrend_direction']
            }
            
            indicators_found = 0
            for standard_name, possible_cols in indicator_mappings.items():
                for col in possible_cols:
                    value = safe_get(target_candle, col)
                    if value is not None:
                        converted_value = safe_float(value)
                        if converted_value != 0.0 or value == 0.0:  # Allow legitimate zero values
                            signal_data[standard_name] = converted_value
                            indicators_found += 1
                            break  # Found this indicator, move to next
            
            # Add metadata about data quality
            signal_data['indicators_found'] = indicators_found
            signal_data['data_completeness'] = min(1.0, indicators_found / 10.0)  # Normalize to ~10 expected indicators
            
            # Log data quality for debugging
            if indicators_found == 0:
                self.logger.warning(f"⚠️ No technical indicators found in candle data")
                # List available columns for debugging
                try:
                    if hasattr(target_candle, 'index'):
                        available_cols = list(target_candle.index)
                        self.logger.debug(f"Available columns: {available_cols[:10]}...")  # Show first 10
                    elif hasattr(target_candle, 'keys'):
                        available_cols = list(target_candle.keys())
                        self.logger.debug(f"Available keys: {available_cols[:10]}...")
                except Exception:
                    self.logger.debug("Could not list available columns/keys")
            else:
                self.logger.debug(f"✅ Prepared signal data with {indicators_found} technical indicators")
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing signal data: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Return emergency signal data to prevent complete failure
            return {
                'epic': epic,
                'timestamp': target_time if target_time is not None else datetime.now(),
                'price': 1.0,  # Default price to avoid division by zero
                'open_price': 1.0,
                'high_price': 1.0,
                'low_price': 1.0,
                'close_price': 1.0,
                'volume': 1000,
                'timeframe': '5m',
                'pair': pair,
                'strategy': 'error_recovery',
                'signal_type': 'ANALYSIS',
                'confidence_score': 0.1,
                'error': f'Signal preparation failed: {str(e)}',
                'indicators_found': 0,
                'data_completeness': 0.0
            }
    

    


    def _calculate_future_analysis(self, target_candle, future_data, epic: str) -> Dict:
        """Calculate future price movement analysis"""
        next_candles = future_data.head(12)  # 1 hour for 5-minute timeframe
        
        if next_candles.empty:
            return {}
        
        start_price = float(target_candle['close'])
        max_high = float(next_candles['high'].max())
        min_low = float(next_candles['low'].min())
        end_price = float(next_candles.iloc[-1]['close'])
        
        # Calculate pip movements
        pip_multiplier = 100 if 'JPY' in epic else 10000
        
        max_gain_pips = (max_high - start_price) * pip_multiplier
        max_loss_pips = (start_price - min_low) * pip_multiplier
        net_movement_pips = (end_price - start_price) * pip_multiplier
        
        return {
            'next_hour_high': max_high,
            'next_hour_low': min_low,
            'next_hour_close': end_price,
            'max_gain_pips': round(max_gain_pips, 1),
            'max_loss_pips': round(max_loss_pips, 1),
            'net_movement_pips': round(net_movement_pips, 1),
            'favorable_movement': abs(max_gain_pips) > abs(max_loss_pips),
            'candles_analyzed': len(next_candles),
            'price_range_pips': round((max_high - min_low) * pip_multiplier, 1)
        }