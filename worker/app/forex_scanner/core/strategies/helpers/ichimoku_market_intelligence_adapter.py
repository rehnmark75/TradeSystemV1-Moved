# core/strategies/helpers/ichimoku_market_intelligence_adapter.py
"""
Ichimoku Market Intelligence Adapter
Bridges market intelligence analysis with RAG-enhanced parameter adaptation

🧠 INTELLIGENT ADAPTATION FEATURES:
- Real-time market regime detection integration
- RAG-powered parameter recommendations based on market conditions
- Dynamic Ichimoku configuration adaptation
- Session-aware parameter optimization
- Volatility-adaptive threshold adjustment

🎯 ADAPTATION STRATEGIES:
- Trending markets: Optimize for trend-following with wider stops
- Ranging markets: Tighten parameters for mean reversion signals
- Breakout markets: Enhance cloud breakout sensitivity
- High volatility: Increase confirmation requirements
- Low volatility: Reduce threshold requirements for signal generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config

# Import market intelligence
try:
    from ..intelligence.market_intelligence import MarketIntelligenceEngine
    MARKET_INTELLIGENCE_AVAILABLE = True
except ImportError:
    try:
        from ...intelligence.market_intelligence import MarketIntelligenceEngine
        MARKET_INTELLIGENCE_AVAILABLE = True
    except ImportError:
        MARKET_INTELLIGENCE_AVAILABLE = False
        class MarketIntelligenceEngine:
            def __init__(self, *args, **kwargs):
                pass


@dataclass
class MarketRegimeConfig:
    """Configuration for different market regimes"""
    regime_type: str
    confidence_threshold_modifier: float
    stop_loss_modifier: float
    take_profit_modifier: float
    cloud_thickness_modifier: float
    tk_cross_sensitivity: float
    chikou_clear_modifier: float
    mtf_requirement: bool
    additional_confirmations: int


class IchimokuMarketIntelligenceAdapter:
    """
    🧠 MARKET INTELLIGENCE BRIDGE

    Integrates market intelligence analysis with RAG-enhanced Ichimoku parameters:
    - Real-time regime detection and adaptation
    - RAG-powered configuration recommendations
    - Session-aware parameter optimization
    - Volatility-responsive threshold adjustment
    """

    def __init__(self, data_fetcher=None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
        self.market_intelligence = None

        # Initialize market intelligence if available
        if MARKET_INTELLIGENCE_AVAILABLE and data_fetcher:
            try:
                self.market_intelligence = MarketIntelligenceEngine(data_fetcher)
                self.logger.info("🧠 Market Intelligence Engine connected")
            except Exception as e:
                self.logger.warning(f"Market Intelligence Engine initialization failed: {e}")
                self.market_intelligence = None
        else:
            self.logger.warning("Market Intelligence Engine not available")

        # Market regime configurations
        self.regime_configs = self._initialize_regime_configs()

        self.logger.info("🧠 Ichimoku Market Intelligence Adapter initialized")

    def _initialize_regime_configs(self) -> Dict[str, MarketRegimeConfig]:
        """Initialize market regime-specific configurations"""
        return {
            'trending': MarketRegimeConfig(
                regime_type='trending',
                confidence_threshold_modifier=-0.05,  # Lower threshold for trending markets
                stop_loss_modifier=1.3,               # Wider stops to avoid whipsaws
                take_profit_modifier=1.8,             # Wider targets to capture trends
                cloud_thickness_modifier=1.2,        # Accept thicker clouds in trends
                tk_cross_sensitivity=0.8,             # Higher sensitivity for TK crosses
                chikou_clear_modifier=1.1,            # Slightly more strict Chikou requirement
                mtf_requirement=True,                 # Require MTF confirmation
                additional_confirmations=1            # One additional confirmation
            ),

            'ranging': MarketRegimeConfig(
                regime_type='ranging',
                confidence_threshold_modifier=0.08,   # Higher threshold for ranging markets
                stop_loss_modifier=0.8,               # Tighter stops for mean reversion
                take_profit_modifier=0.9,             # Tighter targets for quick exits
                cloud_thickness_modifier=0.7,        # Prefer thinner clouds
                tk_cross_sensitivity=1.2,             # Lower sensitivity, avoid false signals
                chikou_clear_modifier=0.9,            # Slightly relaxed Chikou requirement
                mtf_requirement=False,                # No MTF requirement in ranging
                additional_confirmations=2            # More confirmations for ranging
            ),

            'breakout': MarketRegimeConfig(
                regime_type='breakout',
                confidence_threshold_modifier=-0.03,  # Slightly lower threshold
                stop_loss_modifier=1.5,               # Wide stops for breakout moves
                take_profit_modifier=2.2,             # Wide targets for breakout momentum
                cloud_thickness_modifier=0.8,        # Prefer cleaner cloud breaks
                tk_cross_sensitivity=0.9,             # Good sensitivity for breakouts
                chikou_clear_modifier=1.3,            # Strong Chikou requirement
                mtf_requirement=True,                 # Require MTF confirmation
                additional_confirmations=0            # Fast entry on breakouts
            ),

            'high_volatility': MarketRegimeConfig(
                regime_type='high_volatility',
                confidence_threshold_modifier=0.12,   # Much higher threshold
                stop_loss_modifier=1.6,               # Wide stops for volatility
                take_profit_modifier=1.4,             # Moderate targets
                cloud_thickness_modifier=1.4,        # Accept thicker clouds
                tk_cross_sensitivity=1.5,             # Lower sensitivity, avoid noise
                chikou_clear_modifier=1.4,            # Strong Chikou requirement
                mtf_requirement=True,                 # Require MTF confirmation
                additional_confirmations=2            # Multiple confirmations
            ),

            'low_volatility': MarketRegimeConfig(
                regime_type='low_volatility',
                confidence_threshold_modifier=-0.08,  # Lower threshold for opportunities
                stop_loss_modifier=0.7,               # Tight stops in low vol
                take_profit_modifier=0.8,             # Tight targets
                cloud_thickness_modifier=0.6,        # Prefer very thin clouds
                tk_cross_sensitivity=0.7,             # High sensitivity for small moves
                chikou_clear_modifier=0.8,            # Relaxed Chikou requirement
                mtf_requirement=False,                # No MTF requirement
                additional_confirmations=0            # Quick entry in low vol
            )
        }

    def analyze_market_conditions(self, epic: str, timeframe: str = '15m') -> Dict:
        """
        🔍 ANALYZE MARKET CONDITIONS: Get comprehensive market analysis

        Args:
            epic: Currency pair to analyze
            timeframe: Analysis timeframe

        Returns:
            Market conditions analysis with regime, volatility, and session info
        """
        try:
            if not self.market_intelligence:
                return self._get_fallback_market_conditions(epic, timeframe)

            # Get market regime analysis
            epic_list = [epic]  # Analyze single pair for specific recommendations
            regime_analysis = self.market_intelligence.analyze_market_regime(
                epic_list=epic_list,
                lookback_hours=24
            )

            # Extract key market characteristics
            dominant_regime = regime_analysis.get('dominant_regime', 'trending')
            regime_confidence = regime_analysis.get('confidence', 0.5)
            regime_scores = regime_analysis.get('regime_scores', {})

            # Determine volatility level
            volatility_level = self._determine_volatility_level(regime_scores)

            # Get session information
            current_session = self._get_current_trading_session()

            # Compile market conditions
            market_conditions = {
                'regime': dominant_regime,
                'regime_confidence': regime_confidence,
                'regime_scores': regime_scores,
                'volatility_level': volatility_level,
                'trading_session': current_session,
                'timeframe': timeframe,
                'epic': epic,
                'analysis_timestamp': datetime.now().isoformat(),
                'source': 'market_intelligence_engine'
            }

            self.logger.info(f"🔍 Market analysis for {epic}: "
                           f"Regime={dominant_regime} ({regime_confidence:.1%}), "
                           f"Volatility={volatility_level}, Session={current_session}")

            return market_conditions

        except Exception as e:
            self.logger.error(f"Market conditions analysis failed: {e}")
            return self._get_fallback_market_conditions(epic, timeframe)

    def _determine_volatility_level(self, regime_scores: Dict) -> str:
        """Determine volatility level from regime scores"""
        try:
            high_vol_score = regime_scores.get('high_volatility', 0.3)
            low_vol_score = regime_scores.get('low_volatility', 0.3)

            if high_vol_score > 0.6:
                return 'high'
            elif low_vol_score > 0.6:
                return 'low'
            else:
                return 'medium'

        except Exception:
            return 'medium'

    def _get_current_trading_session(self) -> str:
        """Determine current trading session based on UTC time"""
        try:
            utc_hour = datetime.utcnow().hour

            if 0 <= utc_hour < 8:
                return 'asian'
            elif 8 <= utc_hour < 16:
                return 'london'
            elif 16 <= utc_hour < 22:
                return 'new_york'
            else:
                return 'asian'  # Late NY / Early Asian overlap

        except Exception:
            return 'london'  # Default to London session

    def _get_fallback_market_conditions(self, epic: str, timeframe: str) -> Dict:
        """Fallback market conditions when intelligence engine is unavailable"""
        return {
            'regime': 'trending',
            'regime_confidence': 0.5,
            'regime_scores': {
                'trending': 0.4,
                'ranging': 0.3,
                'breakout': 0.2,
                'high_volatility': 0.3,
                'low_volatility': 0.4
            },
            'volatility_level': 'medium',
            'trading_session': self._get_current_trading_session(),
            'timeframe': timeframe,
            'epic': epic,
            'analysis_timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }

    def adapt_ichimoku_parameters(
        self,
        base_config: Dict,
        market_conditions: Dict,
        rag_recommendations: Dict = None
    ) -> Dict:
        """
        🎯 ADAPT ICHIMOKU PARAMETERS: Apply market intelligence to parameter optimization

        Args:
            base_config: Base Ichimoku configuration
            market_conditions: Current market conditions
            rag_recommendations: Optional RAG-based recommendations

        Returns:
            Adapted configuration optimized for current market conditions
        """
        try:
            adapted_config = base_config.copy()

            # Get regime-specific configuration
            regime = market_conditions.get('regime', 'trending')
            volatility = market_conditions.get('volatility_level', 'medium')

            # Apply regime-based adaptations
            regime_config = self._get_regime_config(regime, volatility)

            # Apply base regime modifications
            adapted_config.update(self._apply_regime_modifications(
                adapted_config, regime_config, market_conditions
            ))

            # Apply RAG recommendations if available
            if rag_recommendations and not rag_recommendations.get('error'):
                adapted_config = self._integrate_rag_recommendations(
                    adapted_config, rag_recommendations, regime_config
                )

            # Apply session-specific adjustments
            adapted_config = self._apply_session_adjustments(
                adapted_config, market_conditions.get('trading_session', 'london')
            )

            # Log adaptation results
            self._log_adaptation_results(base_config, adapted_config, regime, volatility)

            return adapted_config

        except Exception as e:
            self.logger.error(f"Parameter adaptation failed: {e}")
            return base_config

    def _get_regime_config(self, regime: str, volatility: str) -> MarketRegimeConfig:
        """Get configuration for specific regime and volatility combination"""
        try:
            # Primary regime configuration
            if regime in self.regime_configs:
                primary_config = self.regime_configs[regime]
            else:
                primary_config = self.regime_configs['trending']

            # Overlay volatility adjustments
            if volatility == 'high' and regime != 'high_volatility':
                volatility_config = self.regime_configs['high_volatility']
                # Blend configurations (70% regime, 30% volatility)
                blended_config = self._blend_regime_configs(primary_config, volatility_config, 0.7)
                return blended_config
            elif volatility == 'low' and regime != 'low_volatility':
                volatility_config = self.regime_configs['low_volatility']
                # Blend configurations (70% regime, 30% volatility)
                blended_config = self._blend_regime_configs(primary_config, volatility_config, 0.7)
                return blended_config

            return primary_config

        except Exception:
            return self.regime_configs['trending']

    def _blend_regime_configs(
        self,
        primary: MarketRegimeConfig,
        secondary: MarketRegimeConfig,
        primary_weight: float = 0.7
    ) -> MarketRegimeConfig:
        """Blend two regime configurations with specified weights"""
        try:
            secondary_weight = 1.0 - primary_weight

            return MarketRegimeConfig(
                regime_type=f"{primary.regime_type}_{secondary.regime_type}",
                confidence_threshold_modifier=(
                    primary.confidence_threshold_modifier * primary_weight +
                    secondary.confidence_threshold_modifier * secondary_weight
                ),
                stop_loss_modifier=(
                    primary.stop_loss_modifier * primary_weight +
                    secondary.stop_loss_modifier * secondary_weight
                ),
                take_profit_modifier=(
                    primary.take_profit_modifier * primary_weight +
                    secondary.take_profit_modifier * secondary_weight
                ),
                cloud_thickness_modifier=(
                    primary.cloud_thickness_modifier * primary_weight +
                    secondary.cloud_thickness_modifier * secondary_weight
                ),
                tk_cross_sensitivity=(
                    primary.tk_cross_sensitivity * primary_weight +
                    secondary.tk_cross_sensitivity * secondary_weight
                ),
                chikou_clear_modifier=(
                    primary.chikou_clear_modifier * primary_weight +
                    secondary.chikou_clear_modifier * secondary_weight
                ),
                mtf_requirement=primary.mtf_requirement or secondary.mtf_requirement,
                additional_confirmations=max(
                    primary.additional_confirmations,
                    secondary.additional_confirmations
                )
            )

        except Exception:
            return primary

    def _apply_regime_modifications(
        self,
        config: Dict,
        regime_config: MarketRegimeConfig,
        market_conditions: Dict
    ) -> Dict:
        """Apply regime-specific modifications to configuration"""
        try:
            modifications = {}

            # Apply confidence threshold modification
            base_threshold = config.get('confidence_threshold', 0.55)
            modifications['adapted_confidence_threshold'] = max(
                0.3, min(0.9, base_threshold + regime_config.confidence_threshold_modifier)
            )

            # Apply parameter modifications
            modifications['regime_stop_loss_modifier'] = regime_config.stop_loss_modifier
            modifications['regime_take_profit_modifier'] = regime_config.take_profit_modifier

            # Apply cloud thickness modification
            base_thickness = config.get('cloud_thickness_threshold', 0.0001)
            modifications['adapted_cloud_thickness_threshold'] = (
                base_thickness * regime_config.cloud_thickness_modifier
            )

            # Apply TK cross sensitivity
            base_tk_threshold = config.get('tk_cross_strength_threshold', 0.5)
            modifications['adapted_tk_cross_threshold'] = (
                base_tk_threshold * regime_config.tk_cross_sensitivity
            )

            # Apply Chikou clear modification
            base_chikou_threshold = config.get('chikou_clear_threshold', 0.0002)
            modifications['adapted_chikou_clear_threshold'] = (
                base_chikou_threshold * regime_config.chikou_clear_modifier
            )

            # Apply MTF requirement
            modifications['regime_mtf_required'] = regime_config.mtf_requirement

            # Apply additional confirmations
            modifications['regime_additional_confirmations'] = regime_config.additional_confirmations

            # Store regime information
            modifications['applied_regime'] = regime_config.regime_type
            modifications['regime_confidence'] = market_conditions.get('regime_confidence', 0.5)

            return modifications

        except Exception as e:
            self.logger.error(f"Regime modifications failed: {e}")
            return {}

    def _integrate_rag_recommendations(
        self,
        config: Dict,
        rag_recommendations: Dict,
        regime_config: MarketRegimeConfig
    ) -> Dict:
        """Integrate RAG recommendations with regime-based adaptations"""
        try:
            # Get RAG parameter suggestions
            rag_params = rag_recommendations.get('parameters', {})
            rag_adjustments = rag_recommendations.get('recommended_adjustments', {})

            # Apply RAG parameter suggestions (if they align with regime requirements)
            if rag_params:
                for param, value in rag_params.items():
                    if param in ['tenkan_period', 'kijun_period', 'senkou_b_period', 'chikou_shift']:
                        config[f'rag_suggested_{param}'] = value

            # Apply RAG adjustments with regime constraints
            if rag_adjustments:
                rag_confidence_modifier = rag_adjustments.get('confidence_threshold', 0)

                # Blend RAG and regime confidence modifications (50/50)
                current_modifier = config.get('adapted_confidence_threshold', 0.55)
                base_confidence = 0.55  # Default base
                regime_modifier = current_modifier - base_confidence

                blended_modifier = (regime_modifier + rag_confidence_modifier) / 2
                config['rag_regime_confidence_threshold'] = max(
                    0.3, min(0.9, base_confidence + blended_modifier)
                )

                # Apply other RAG adjustments
                for adjustment, value in rag_adjustments.items():
                    if adjustment != 'confidence_threshold':
                        config[f'rag_{adjustment}'] = value

            # Store RAG integration info
            config['rag_integration_applied'] = True
            config['rag_technique_type'] = rag_recommendations.get('technique_type', 'unknown')

            return config

        except Exception as e:
            self.logger.error(f"RAG integration failed: {e}")
            return config

    def _apply_session_adjustments(self, config: Dict, session: str) -> Dict:
        """Apply trading session-specific adjustments"""
        try:
            session_adjustments = {
                'asian': {
                    'session_volatility_factor': 0.8,  # Lower volatility
                    'session_confidence_boost': 0.02,  # Slightly higher confidence needed
                    'session_preferred_pairs': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY']
                },
                'london': {
                    'session_volatility_factor': 1.2,  # Higher volatility
                    'session_confidence_boost': -0.03,  # Can accept lower confidence
                    'session_preferred_pairs': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURJPY']
                },
                'new_york': {
                    'session_volatility_factor': 1.0,   # Normal volatility
                    'session_confidence_boost': 0.0,   # No adjustment
                    'session_preferred_pairs': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']
                }
            }

            session_config = session_adjustments.get(session, session_adjustments['london'])

            # Apply session adjustments
            config['trading_session'] = session
            config['session_volatility_factor'] = session_config['session_volatility_factor']

            # Adjust confidence threshold for session
            current_threshold = config.get('rag_regime_confidence_threshold') or config.get('adapted_confidence_threshold', 0.55)
            config['final_confidence_threshold'] = max(
                0.3, min(0.9, current_threshold + session_config['session_confidence_boost'])
            )

            config['session_preferred_pairs'] = session_config['session_preferred_pairs']

            return config

        except Exception as e:
            self.logger.error(f"Session adjustments failed: {e}")
            return config

    def _log_adaptation_results(
        self,
        base_config: Dict,
        adapted_config: Dict,
        regime: str,
        volatility: str
    ):
        """Log the results of parameter adaptation"""
        try:
            base_confidence = base_config.get('confidence_threshold', 0.55)
            final_confidence = adapted_config.get('final_confidence_threshold', base_confidence)

            self.logger.info(f"🎯 Parameter adaptation complete:")
            self.logger.info(f"   Regime: {regime}, Volatility: {volatility}")
            self.logger.info(f"   Confidence: {base_confidence:.1%} → {final_confidence:.1%}")

            if 'regime_stop_loss_modifier' in adapted_config:
                self.logger.info(f"   Stop Loss Modifier: {adapted_config['regime_stop_loss_modifier']:.2f}")

            if 'regime_take_profit_modifier' in adapted_config:
                self.logger.info(f"   Take Profit Modifier: {adapted_config['regime_take_profit_modifier']:.2f}")

            if adapted_config.get('rag_integration_applied'):
                self.logger.info(f"   RAG Integration: Applied ({adapted_config.get('rag_technique_type', 'unknown')})")

        except Exception as e:
            self.logger.error(f"Adaptation logging failed: {e}")

    def get_adaptation_summary(self, epic: str, adapted_config: Dict) -> Dict:
        """Get summary of applied adaptations"""
        try:
            summary = {
                'epic': epic,
                'adaptation_timestamp': datetime.now().isoformat(),
                'applied_regime': adapted_config.get('applied_regime', 'unknown'),
                'regime_confidence': adapted_config.get('regime_confidence', 0.5),
                'trading_session': adapted_config.get('trading_session', 'unknown'),
                'final_confidence_threshold': adapted_config.get('final_confidence_threshold', 0.55),
                'rag_integration': adapted_config.get('rag_integration_applied', False),
                'adaptations_applied': []
            }

            # Collect applied adaptations
            adaptation_keys = [
                'regime_stop_loss_modifier',
                'regime_take_profit_modifier',
                'adapted_cloud_thickness_threshold',
                'adapted_tk_cross_threshold',
                'adapted_chikou_clear_threshold',
                'session_volatility_factor'
            ]

            for key in adaptation_keys:
                if key in adapted_config:
                    summary['adaptations_applied'].append({
                        'parameter': key,
                        'value': adapted_config[key]
                    })

            return summary

        except Exception as e:
            self.logger.error(f"Adaptation summary failed: {e}")
            return {'error': str(e)}

    def is_market_intelligence_available(self) -> bool:
        """Check if market intelligence is available"""
        return self.market_intelligence is not None

    def get_market_intelligence_status(self) -> Dict:
        """Get status of market intelligence components"""
        return {
            'market_intelligence_available': MARKET_INTELLIGENCE_AVAILABLE,
            'market_intelligence_initialized': self.market_intelligence is not None,
            'regime_configs_loaded': len(self.regime_configs),
            'supported_regimes': list(self.regime_configs.keys())
        }