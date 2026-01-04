# core/intelligence/intelligence_manager.py
"""
Intelligence Manager - Extracted from IntelligentForexScanner
Handles all intelligence filtering, scoring, and configuration management

CRITICAL: Database is the ONLY source of truth for configuration.
If database is unavailable, the system will FAIL (raise RuntimeError).
NO FALLBACK TO CONFIG FILES.
"""

import logging
from typing import Dict, List, Optional, Tuple

# Database-driven configuration (NO FALLBACK)
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    SCANNER_CONFIG_AVAILABLE = True
except ImportError:
    SCANNER_CONFIG_AVAILABLE = False


class IntelligenceManager:
    """
    Manages intelligence filtering and scoring for trading signals
    Extracted from IntelligentForexScanner to provide modular intelligence handling

    CRITICAL DESIGN: Database is the ONLY source of truth.
    - All settings come from scanner_global_config table
    - No fallback to config.py
    - If database unavailable, raises RuntimeError
    """

    def __init__(self,
                 intelligence_preset: str = None,
                 db_manager=None,
                 signal_detector=None,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.signal_detector = signal_detector
        self.logger = logger or logging.getLogger(__name__)

        # Fail-fast: Database configuration is REQUIRED
        if not SCANNER_CONFIG_AVAILABLE:
            raise RuntimeError(
                "IntelligenceManager requires scanner_config_service. "
                "Database configuration is mandatory - no fallback available."
            )

        # Load database configuration (fail-fast)
        try:
            self._scanner_cfg = get_scanner_config()
            self.logger.info(
                f"[CONFIG:DB] IntelligenceManager loaded config from database "
                f"(source: {self._scanner_cfg.source})"
            )
        except Exception as e:
            raise RuntimeError(
                f"IntelligenceManager failed to load database configuration: {e}. "
                f"Database is required - no fallback available."
            ) from e

        # Initialize intelligence configuration
        self.intelligence_config = None
        self.market_intelligence_engine = None
        self.intelligence_filters = None

        # Initialize components using database config
        self._initialize_intelligence_config(intelligence_preset)
        self._initialize_intelligence_filters()

        self.logger.info("IntelligenceManager initialized")
        if self.intelligence_config:
            status = self.get_status_summary()
            self.logger.info(f"   Preset: {status['preset']}")
            self.logger.info(f"   Threshold: {status['threshold']:.1%}")
            self.logger.info(f"   Components: {status['component_count']} enabled")

    def _refresh_config(self):
        """Refresh configuration from database"""
        try:
            self._scanner_cfg = get_scanner_config()
        except Exception as e:
            self.logger.warning(f"Failed to refresh config, using cached: {e}")

    def _initialize_intelligence_config(self, intelligence_preset: str = None):
        """Initialize configurable intelligence system using database config"""
        try:
            from core.intelligence.intelligence_config import IntelligenceConfigManager

            # Use provided preset, or get from database (NO config.py fallback)
            preset = intelligence_preset or self._scanner_cfg.intelligence_preset

            self.intelligence_config = IntelligenceConfigManager(preset)
            self.logger.info(f"Configurable intelligence initialized: {preset}")

            status = self.intelligence_config.get_status_summary()
            self.logger.debug(f"   Mode: {status['mode']}")
            self.logger.debug(f"   Threshold: {status['threshold']:.1%}")
            self.logger.debug(f"   Components: {', '.join(status['enabled_components'])}")

            # Initialize market intelligence engine if needed
            if status['uses_intelligence_engine']:
                try:
                    from core.intelligence import create_intelligence_engine
                    self.market_intelligence_engine = create_intelligence_engine(
                        data_fetcher=None  # Will be injected if needed
                    )
                    self.logger.info("MarketIntelligenceEngine initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize MarketIntelligenceEngine: {e}")
                    self.market_intelligence_engine = None
            else:
                self.market_intelligence_engine = None

            return True

        except ImportError as e:
            self.intelligence_config = None
            self.market_intelligence_engine = None
            self.logger.warning("Configurable intelligence not available, using legacy system")
            return False
        except Exception as e:
            self.intelligence_config = None
            self.market_intelligence_engine = None
            self.logger.error(f"Failed to initialize intelligence config: {e}")
            return False

    def _initialize_intelligence_filters(self):
        """Initialize intelligence filters"""
        try:
            from utils.scanner_utils import IntelligenceFilters

            if self.db_manager and self.signal_detector:
                self.intelligence_filters = IntelligenceFilters(
                    self.db_manager,
                    self.signal_detector,
                    self.logger
                )
                self.logger.debug("Intelligence filters initialized")
            else:
                self.logger.warning("Cannot initialize intelligence filters - missing dependencies")

        except ImportError:
            self.logger.warning("Intelligence filters not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize intelligence filters: {e}")

    def calculate_intelligence_score(self, signal: Dict) -> float:
        """Calculate intelligence score using configurable weights and components"""
        if not self.intelligence_config:
            return self._calculate_legacy_intelligence_score(signal)

        status = self.intelligence_config.get_status_summary()
        if status['is_disabled']:
            return 1.0

        component_scores = {}
        epic = signal.get('epic', 'Unknown')

        try:
            # Market regime analysis
            if self.intelligence_config.is_component_enabled('market_regime_filter'):
                component_scores['market_regime'] = self._get_market_regime_score(epic)

            # Volatility analysis
            if self.intelligence_config.is_component_enabled('volatility_filter'):
                component_scores['volatility'] = self._get_volatility_score(epic)

            # Volume analysis
            if self.intelligence_config.is_component_enabled('volume_filter'):
                component_scores['volume'] = self._get_volume_score(epic)

            # Time-based analysis
            if self.intelligence_config.is_component_enabled('time_filter'):
                component_scores['time'] = self._get_time_score()

            # Confidence analysis
            if self.intelligence_config.is_component_enabled('confidence_filter'):
                confidence = signal.get('confidence_score', 0)
                component_scores['confidence'] = min(1.0, confidence / 0.8)

            # Calculate weighted final score
            final_score = self.intelligence_config.calculate_weighted_score(component_scores)

            # Debug logging if enabled (from database config)
            if self._scanner_cfg.intelligence_debug_mode:
                self.logger.debug(f"Intelligence breakdown for {epic}:")
                for comp, score in component_scores.items():
                    weight = self.intelligence_config.get_component_weight(comp)
                    self.logger.debug(f"   {comp}: {score:.2f} (weight: {weight:.2f})")
                self.logger.debug(f"   Final score: {final_score:.2f}")

            return final_score

        except Exception as e:
            self.logger.error(f"Error calculating intelligence score for {epic}: {e}")
            return 0.5  # Safe fallback

    def _calculate_legacy_intelligence_score(self, signal: Dict) -> float:
        """Legacy intelligence score calculation"""
        if not self.intelligence_filters:
            return 0.7  # Default score when no filters available

        try:
            return self.intelligence_filters.calculate_intelligence_score(signal)
        except Exception as e:
            self.logger.error(f"Error in legacy intelligence calculation: {e}")
            return 0.7

    def _get_market_regime_score(self, epic: str) -> float:
        """Get market regime score"""
        try:
            if self.intelligence_filters:
                return self.intelligence_filters.get_market_regime_score(epic)
            else:
                return 0.5  # Neutral score
        except Exception as e:
            self.logger.debug(f"Market regime score error for {epic}: {e}")
            return 0.5

    def _get_volatility_score(self, epic: str) -> float:
        """Get volatility score"""
        try:
            if self.intelligence_filters:
                return self.intelligence_filters.get_volatility_score(epic)
            else:
                return 0.5  # Neutral score
        except Exception as e:
            self.logger.debug(f"Volatility score error for {epic}: {e}")
            return 0.5

    def _get_volume_score(self, epic: str) -> float:
        """Get volume score"""
        try:
            if self.intelligence_filters:
                return self.intelligence_filters.get_volume_score(epic)
            else:
                return 0.5  # Neutral score
        except Exception as e:
            self.logger.debug(f"Volume score error for {epic}: {e}")
            return 0.5

    def _get_time_score(self) -> float:
        """Get time-based score"""
        try:
            if self.intelligence_filters:
                return self.intelligence_filters.get_time_score()
            else:
                return 0.5  # Neutral score
        except Exception as e:
            self.logger.debug(f"Time score error: {e}")
            return 0.5

    def should_filter_signal(self, intelligence_score: float) -> bool:
        """Determine if signal should be filtered based on intelligence score"""
        if not self.intelligence_config:
            # Legacy mode - use fixed thresholds
            return intelligence_score < 0.7

        return self.intelligence_config.should_filter_signal(intelligence_score)

    def apply_intelligence_filtering(self, signals: List[Dict]) -> List[Dict]:
        """Apply intelligence filtering to a list of signals"""
        if not signals:
            return []

        filtered_signals = []

        for signal in signals:
            intelligence_score = self.calculate_intelligence_score(signal)
            should_filter = self.should_filter_signal(intelligence_score)

            if not should_filter:
                signal['intelligence_score'] = intelligence_score
                filtered_signals.append(signal)
                self._log_signal_decision(signal['epic'], intelligence_score, True)
            else:
                self._log_signal_decision(signal['epic'], intelligence_score, False)

        self.logger.info(f"Intelligence filtering: {len(filtered_signals)}/{len(signals)} signals passed")
        return filtered_signals

    def apply_backtest_intelligence_filtering(self, signals: List[Dict]) -> List[Dict]:
        """Apply backtest-consistent intelligence filters"""
        if not signals:
            return []

        filtered_signals = []

        for signal in signals:
            if self._passes_backtest_intelligence_filters(signal):
                filtered_signals.append(signal)
                self.logger.debug(f"Signal passed backtest intelligence: {signal['epic']}")
            else:
                self.logger.debug(f"Signal filtered by backtest intelligence: {signal['epic']}")

        self.logger.info(f"Backtest intelligence filtering: {len(filtered_signals)}/{len(signals)} signals passed")
        return filtered_signals

    def _passes_backtest_intelligence_filters(self, signal: Dict) -> bool:
        """Intelligence filters that should also be applied during backtesting"""
        if not self.intelligence_filters:
            return True  # No filters available, allow signal

        try:
            epic = signal['epic']

            # Market Hours filter
            if not self.intelligence_filters.is_optimal_trading_time():
                self.logger.debug(f"Signal filtered: {epic} - outside optimal trading hours")
                return False

            # Volume filter
            if not self.intelligence_filters.has_sufficient_volume(epic):
                self.logger.debug(f"Signal filtered: {epic} - insufficient volume")
                return False

            # Spread filter
            if not self.intelligence_filters.has_acceptable_spread(epic):
                self.logger.debug(f"Signal filtered: {epic} - unacceptable spread")
                return False

            # Frequency filter
            if self.intelligence_filters.too_many_recent_signals(epic):
                self.logger.debug(f"Signal filtered: {epic} - too many recent signals")
                return False

            # Multi-timeframe confluence filter (from database config)
            if self._scanner_cfg.enable_multi_timeframe_analysis:
                confluence_score = signal.get('confluence_score', 0)
                min_confluence = self._scanner_cfg.min_confluence_score
                if confluence_score < min_confluence:
                    self.logger.debug(
                        f"Signal filtered: {epic} - low confluence "
                        f"({confluence_score:.2f} < {min_confluence})"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error in backtest intelligence filtering: {e}")
            return True  # Default to allowing signal

    def _log_signal_decision(self, epic: str, intelligence_score: float, approved: bool):
        """Log intelligence decision for signal"""
        if self.intelligence_config:
            self.intelligence_config.log_signal_decision(epic, intelligence_score, approved)
        else:
            status = "APPROVED" if approved else "FILTERED"
            threshold = 0.7  # Legacy threshold
            self.logger.info(f"{status}: {epic} (score: {intelligence_score:.1%}, threshold: {threshold:.1%})")

    def change_intelligence_preset(self, preset_name: str) -> bool:
        """Change intelligence preset at runtime"""
        if not self.intelligence_config:
            self.logger.error("Configurable intelligence not available")
            return False

        try:
            old_preset = self.intelligence_config.preset
            self.intelligence_config.load_preset(preset_name)

            status = self.intelligence_config.get_status_summary()
            self.logger.info(f"Intelligence preset changed: {old_preset} -> {preset_name}")
            self.logger.info(f"   New threshold: {status['threshold']:.1%}")
            self.logger.info(f"   Components: {', '.join(status['enabled_components'])}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to change intelligence preset: {e}")
            return False

    def get_intelligence_status(self) -> Dict:
        """Get current intelligence configuration status"""
        if not self.intelligence_config:
            return {
                'available': False,
                'system': 'legacy',
                'warning': 'Legacy intelligence may cause zero alerts due to restrictive thresholds'
            }

        status = self.intelligence_config.get_status_summary()
        return {
            'available': True,
            'system': 'configurable',
            'preset': status['preset'],
            'mode': status['mode'],
            'threshold': status['threshold'],
            'enabled_components': status['enabled_components'],
            'component_count': status['component_count'],
            'description': status['description'],
            'is_disabled': status['is_disabled']
        }

    def get_status_summary(self) -> Dict:
        """Get status summary from intelligence config"""
        if self.intelligence_config:
            return self.intelligence_config.get_status_summary()
        else:
            return {
                'available': False,
                'system': 'legacy',
                'preset': 'legacy',
                'mode': 'legacy',
                'threshold': 0.7,
                'enabled_components': ['legacy_filters'],
                'component_count': 1,
                'description': 'Legacy intelligence system',
                'is_disabled': False
            }

    def test_intelligence_thresholds(self, epic: str, test_presets: List[str] = None) -> Dict:
        """Test how different intelligence presets would affect a signal"""
        if not self.intelligence_config:
            return {'error': 'Configurable intelligence not available'}

        if test_presets is None:
            test_presets = ['disabled', 'minimal', 'balanced', 'conservative']

        try:
            # Generate a test signal for this epic
            test_signal = self._generate_test_signal(epic)
            if not test_signal:
                return {'error': f'Could not generate test signal for {epic}'}

            results = {}
            current_preset = self.intelligence_config.preset

            for preset in test_presets:
                try:
                    self.intelligence_config.load_preset(preset)
                    score = self.calculate_intelligence_score(test_signal)
                    would_pass = not self.should_filter_signal(score)

                    threshold = self.intelligence_config.get_effective_threshold()
                    results[preset] = {
                        'score': score,
                        'threshold': threshold,
                        'would_pass': would_pass,
                        'status': 'PASS' if would_pass else 'FILTER'
                    }

                except Exception as e:
                    results[preset] = {'error': str(e)}

            # Restore original preset
            self.intelligence_config.load_preset(current_preset)

            return {
                'epic': epic,
                'signal_type': test_signal.get('signal_type'),
                'confidence': test_signal.get('confidence_score'),
                'strategy': test_signal.get('strategy'),
                'test_results': results,
                'recommendation': self._get_preset_recommendation(results)
            }

        except Exception as e:
            return {'error': f'Failed to test intelligence thresholds: {e}'}

    def _generate_test_signal(self, epic: str) -> Optional[Dict]:
        """Generate a test signal for testing intelligence thresholds"""
        try:
            # Try to get a real signal from signal detector
            if self.signal_detector:
                # Get pair_info from database config
                pair_info = self._scanner_cfg.pair_info.get(
                    epic, {'pair': 'EURUSD', 'pip_multiplier': 10000}
                )
                pair_name = pair_info.get('pair', 'EURUSD')

                if not pair_name or pair_name == 'EURUSD':
                    pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '')

                signal = self.signal_detector.detect_signals_mid_prices(
                    epic=epic,
                    pair=pair_name,
                    timeframe=self._scanner_cfg.default_timeframe
                )

                if signal:
                    return signal

            # Fallback: create a mock signal
            from datetime import datetime
            return {
                'epic': epic,
                'signal_type': 'BUY',
                'strategy': 'TEST',
                'confidence_score': 0.75,
                'price_mid': 1.1000,
                'timestamp': datetime.now().isoformat(),
                'pair': epic.replace('CS.D.', '').replace('.MINI.IP', ''),
                'timeframe': '5m'
            }

        except Exception as e:
            self.logger.error(f"Error generating test signal for {epic}: {e}")
            return None

    def _get_preset_recommendation(self, test_results: Dict) -> str:
        """Get recommendation based on intelligence test results"""
        passing_presets = [
            preset for preset, result in test_results.items()
            if isinstance(result, dict) and result.get('would_pass', False)
        ]

        if not passing_presets:
            return "All presets would filter this signal. Consider 'disabled' mode for maximum signals."
        elif 'conservative' in passing_presets:
            return "Signal passes even conservative filtering - high quality signal."
        elif 'balanced' in passing_presets:
            return "Signal passes balanced filtering - good quality signal."
        elif 'minimal' in passing_presets:
            return "Signal passes minimal filtering - acceptable signal."
        else:
            return "Signal only passes with disabled intelligence."

    def get_intelligence_modes(self) -> Dict:
        """Get available intelligence modes"""
        return {
            'disabled': {
                'description': 'No intelligence filtering - all strategy signals pass',
                'threshold': 0.0,
                'components': []
            },
            'backtest_consistent': {
                'description': 'Only filters that should also apply during backtesting',
                'threshold': 0.0,
                'components': ['market_hours', 'volume', 'spread', 'frequency']
            },
            'live_only': {
                'description': 'Live trading intelligence with moderate filtering',
                'threshold': 0.7,
                'components': ['all']
            },
            'enhanced': {
                'description': 'Advanced intelligence with strict filtering',
                'threshold': 0.8,
                'components': ['all']
            }
        }

    def set_intelligence_filters(self, intelligence_filters):
        """Set intelligence filters dependency (for dependency injection)"""
        self.intelligence_filters = intelligence_filters
        self.logger.debug("Intelligence filters injected")

    def set_signal_detector(self, signal_detector):
        """Set signal detector dependency (for dependency injection)"""
        self.signal_detector = signal_detector
        self.logger.debug("Signal detector injected")

    def get_component_status(self) -> Dict:
        """Get detailed status of all intelligence components"""
        if not self.intelligence_config:
            return {'error': 'Configurable intelligence not available'}

        try:
            status = {}
            components = ['market_regime_filter', 'volatility_filter', 'volume_filter',
                         'time_filter', 'confidence_filter']

            for component in components:
                enabled = self.intelligence_config.is_component_enabled(component)
                weight = self.intelligence_config.get_component_weight(component) if enabled else 0.0

                status[component] = {
                    'enabled': enabled,
                    'weight': weight,
                    'description': self._get_component_description(component)
                }

            return {
                'preset': self.intelligence_config.preset,
                'threshold': self.intelligence_config.get_effective_threshold(),
                'components': status,
                'total_components': len([c for c in status.values() if c['enabled']])
            }

        except Exception as e:
            return {'error': f'Failed to get component status: {e}'}

    def _get_component_description(self, component: str) -> str:
        """Get description for intelligence component"""
        descriptions = {
            'market_regime_filter': 'Analyzes current market regime (trending/ranging)',
            'volatility_filter': 'Evaluates market volatility levels',
            'volume_filter': 'Checks trading volume and liquidity',
            'time_filter': 'Considers optimal trading hours',
            'confidence_filter': 'Weighs signal confidence score'
        }
        return descriptions.get(component, 'Unknown component')

    def reset_intelligence_state(self):
        """Reset intelligence state (useful for testing)"""
        try:
            if self.intelligence_config:
                # Reset to preset from database (NO config.py fallback)
                default_preset = self._scanner_cfg.intelligence_preset
                self.intelligence_config.load_preset(default_preset)
                self.logger.info(f"Intelligence state reset to {default_preset}")

        except Exception as e:
            self.logger.error(f"Failed to reset intelligence state: {e}")

    def validate_intelligence_config(self) -> Tuple[bool, List[str]]:
        """Validate current intelligence configuration"""
        errors = []

        try:
            if not self.intelligence_config:
                errors.append("Configurable intelligence not available")
                return False, errors

            # Check if preset is valid
            status = self.intelligence_config.get_status_summary()
            if not status.get('preset'):
                errors.append("No valid preset loaded")

            # Check if threshold is reasonable
            threshold = status.get('threshold', 0)
            if threshold < 0 or threshold > 1:
                errors.append(f"Invalid threshold: {threshold}")

            # Check if at least one component is enabled (unless disabled mode)
            if not status.get('is_disabled') and status.get('component_count', 0) == 0:
                errors.append("No intelligence components enabled")

            # Check dependencies
            if self.intelligence_filters is None:
                errors.append("Intelligence filters not initialized")

            return len(errors) == 0, errors

        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors

    def get_config_source(self) -> Dict:
        """Get information about configuration source"""
        return {
            'source': self._scanner_cfg.source,
            'intelligence_preset': self._scanner_cfg.intelligence_preset,
            'intelligence_debug_mode': self._scanner_cfg.intelligence_debug_mode,
            'enable_multi_timeframe_analysis': self._scanner_cfg.enable_multi_timeframe_analysis,
            'min_confluence_score': self._scanner_cfg.min_confluence_score,
            'default_timeframe': self._scanner_cfg.default_timeframe,
        }
