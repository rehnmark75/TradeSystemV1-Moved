# configdata/strategies/config_rag_intelligence_strategy.py
"""
RAG Intelligence Strategy Configuration
=====================================

Configuration for the RAG-Enhanced Market Intelligence Strategy.
Includes market intelligence settings, RAG system parameters, and adaptive strategy configurations.
"""

from typing import Dict, List, Any
from datetime import timedelta
import logging


class RAGIntelligenceConfig:
    """Configuration class for RAG Intelligence Strategy"""

    # =============================================================================
    # STRATEGY CORE CONFIGURATION
    # =============================================================================

    # Strategy identification
    STRATEGY_NAME = "RAG Intelligence"
    STRATEGY_VERSION = "1.0"
    STRATEGY_TYPE = "adaptive_intelligence"

    # Basic parameters
    MIN_CONFIDENCE = 0.60  # Higher threshold for intelligence-based signals
    MIN_BARS = 100  # Need more data for intelligence analysis

    # =============================================================================
    # MARKET INTELLIGENCE CONFIGURATION
    # =============================================================================

    # Market analysis settings
    MARKET_ANALYSIS_HOURS = 24  # Hours of historical data to analyze
    INTELLIGENCE_CACHE_DURATION_MINUTES = 5  # Cache market intelligence for 5 minutes

    # Regime detection thresholds
    REGIME_CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for regime classification
    REGIME_STABILITY_MINUTES = 15  # Minutes to consider regime stable

    # Market regime definitions and weights
    MARKET_REGIMES = {
        'trending_up': {
            'description': 'Strong upward momentum',
            'weight': 1.0,
            'favorable_sessions': ['london', 'new_york', 'overlap'],
            'optimal_timeframes': ['15m', '1h'],
            'preferred_strategies': ['trend_following', 'momentum'],
            'confidence_boost': 0.1
        },
        'trending_down': {
            'description': 'Strong downward momentum',
            'weight': 1.0,
            'favorable_sessions': ['london', 'new_york', 'overlap'],
            'optimal_timeframes': ['15m', '1h'],
            'preferred_strategies': ['trend_following', 'momentum'],
            'confidence_boost': 0.1
        },
        'ranging': {
            'description': 'Sideways consolidation',
            'weight': 0.8,
            'favorable_sessions': ['asian', 'london', 'new_york'],
            'optimal_timeframes': ['5m', '15m'],
            'preferred_strategies': ['mean_reversion', 'range_trading'],
            'confidence_boost': 0.05
        },
        'breakout': {
            'description': 'Volatility expansion phase',
            'weight': 0.9,
            'favorable_sessions': ['london', 'overlap'],
            'optimal_timeframes': ['5m', '15m', '1h'],
            'preferred_strategies': ['breakout', 'volatility'],
            'confidence_boost': 0.08
        }
    }

    # Session analysis configuration
    TRADING_SESSIONS = {
        'asian': {
            'start_hour': 22,
            'end_hour': 8,
            'volatility': 'low',
            'characteristics': 'Range-bound trading, lower volume',
            'position_size_multiplier': 0.8,
            'confidence_adjustment': -0.05
        },
        'london': {
            'start_hour': 8,
            'end_hour': 13,
            'volatility': 'high',
            'characteristics': 'Trend establishment, high volume',
            'position_size_multiplier': 1.1,
            'confidence_adjustment': 0.05
        },
        'overlap': {
            'start_hour': 13,
            'end_hour': 16,
            'volatility': 'peak',
            'characteristics': 'Maximum liquidity and volatility',
            'position_size_multiplier': 1.2,
            'confidence_adjustment': 0.1
        },
        'new_york': {
            'start_hour': 16,
            'end_hour': 22,
            'volatility': 'high',
            'characteristics': 'Trend continuation, news impact',
            'position_size_multiplier': 1.0,
            'confidence_adjustment': 0.0
        }
    }

    # =============================================================================
    # RAG SYSTEM CONFIGURATION
    # =============================================================================

    # RAG service settings
    RAG_BASE_URL = "http://localhost:8090"  # RAG service endpoint
    RAG_TIMEOUT_SECONDS = 10  # Request timeout
    RAG_RETRY_ATTEMPTS = 2  # Retry failed requests
    RAG_CACHE_DURATION_MINUTES = 10  # Cache RAG responses

    # RAG query configurations by market regime
    RAG_QUERY_TEMPLATES = {
        'trending_up': {
            'strategy_description': 'momentum trend-following strategy for strong bullish markets',
            'indicator_query': 'trend-following momentum oscillators moving averages',
            'template_query': 'bullish trend continuation systems',
            'trading_style': 'swing_trading',
            'complexity_level': 'intermediate'
        },
        'trending_down': {
            'strategy_description': 'momentum trend-following strategy for strong bearish markets',
            'indicator_query': 'bearish momentum indicators trend reversal systems',
            'template_query': 'bearish trend continuation systems',
            'trading_style': 'swing_trading',
            'complexity_level': 'intermediate'
        },
        'ranging': {
            'strategy_description': 'mean reversion strategy for sideways consolidation markets',
            'indicator_query': 'mean reversion oscillators bollinger bands RSI',
            'template_query': 'range-bound trading systems oscillator strategies',
            'trading_style': 'range_trading',
            'complexity_level': 'beginner'
        },
        'breakout': {
            'strategy_description': 'volatility breakout strategy for explosive price movements',
            'indicator_query': 'volatility indicators ATR breakout systems channel breakouts',
            'template_query': 'breakout trading systems volatility strategies',
            'trading_style': 'day_trading',
            'complexity_level': 'advanced'
        }
    }

    # RAG response scoring weights
    RAG_SCORING_WEIGHTS = {
        'market_suitability': 0.4,  # How well it matches current market conditions
        'confidence_score': 0.3,    # RAG system's confidence in recommendation
        'historical_performance': 0.2,  # Historical backtest performance
        'complexity_match': 0.1     # Complexity level appropriateness
    }

    # =============================================================================
    # ADAPTIVE STRATEGY PARAMETERS
    # =============================================================================

    # Dynamic parameter ranges by regime
    ADAPTIVE_PARAMETERS = {
        'trending_up': {
            'fast_ema': [8, 12, 16],
            'slow_ema': [21, 26, 34],
            'trend_ema': [50, 100, 200],
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'atr_multiplier': 1.5,
            'bb_std_dev': 2.0,
            'min_volatility': 0.001
        },
        'trending_down': {
            'fast_ema': [8, 12, 16],
            'slow_ema': [21, 26, 34],
            'trend_ema': [50, 100, 200],
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'atr_multiplier': 1.5,
            'bb_std_dev': 2.0,
            'min_volatility': 0.001
        },
        'ranging': {
            'bb_period': [14, 20, 26],
            'bb_std_dev': [1.5, 2.0, 2.5],
            'rsi_period': [14, 21, 28],
            'rsi_oversold': 20,
            'rsi_overbought': 80,
            'mean_reversion_threshold': 0.02,
            'min_range_size': 0.005
        },
        'breakout': {
            'atr_period': [10, 14, 20],
            'atr_multiplier': [1.5, 2.0, 2.5],
            'breakout_lookback': [10, 20, 30],
            'volume_threshold': 1.2,
            'momentum_threshold': 0.001,
            'consolidation_period': 20
        }
    }

    # Risk management by regime
    RISK_MANAGEMENT = {
        'trending_up': {
            'stop_loss_pips': [15, 20, 25],
            'take_profit_pips': [30, 40, 50],
            'risk_reward_ratio': [1.5, 2.0, 2.5],
            'position_size_multiplier': [0.8, 1.0, 1.2],
            'trailing_stop_pips': 10
        },
        'trending_down': {
            'stop_loss_pips': [15, 20, 25],
            'take_profit_pips': [30, 40, 50],
            'risk_reward_ratio': [1.5, 2.0, 2.5],
            'position_size_multiplier': [0.8, 1.0, 1.2],
            'trailing_stop_pips': 10
        },
        'ranging': {
            'stop_loss_pips': [10, 15, 20],
            'take_profit_pips': [15, 20, 30],
            'risk_reward_ratio': [1.0, 1.5, 2.0],
            'position_size_multiplier': [0.6, 0.8, 1.0],
            'trailing_stop_pips': 8
        },
        'breakout': {
            'stop_loss_pips': [20, 25, 35],
            'take_profit_pips': [40, 60, 80],
            'risk_reward_ratio': [2.0, 2.5, 3.0],
            'position_size_multiplier': [1.0, 1.3, 1.5],
            'trailing_stop_pips': 15
        }
    }

    # =============================================================================
    # FILTERING AND VALIDATION
    # =============================================================================

    # Intelligence filtering thresholds
    INTELLIGENCE_FILTERS = {
        'min_regime_confidence': 0.6,
        'min_session_score': 0.4,
        'min_volatility_percentile': 0.2,
        'max_spread_cost_ratio': 0.3,  # Max spread as % of expected profit
        'min_volume_multiplier': 0.8,  # Minimum volume vs average
        'max_drawdown_threshold': 0.05  # Maximum acceptable drawdown risk
    }

    # Signal validation rules
    VALIDATION_RULES = {
        'require_regime_alignment': True,  # Signal must align with market regime
        'require_session_suitability': True,  # Must be favorable trading session
        'require_volatility_minimum': True,  # Minimum volatility required
        'require_rag_confidence': True,  # RAG must have reasonable confidence
        'allow_counter_trend': False,  # Allow counter-trend signals in ranging markets
        'max_signals_per_hour': 3,  # Limit signal frequency
        'min_signal_separation_minutes': 15  # Minimum time between signals
    }

    # =============================================================================
    # PERFORMANCE MONITORING
    # =============================================================================

    # Performance tracking settings
    PERFORMANCE_TRACKING = {
        'track_regime_performance': True,
        'track_rag_selection_success': True,
        'track_intelligence_accuracy': True,
        'track_session_performance': True,
        'performance_window_days': 30,
        'min_trades_for_analysis': 10
    }

    # Alert thresholds for performance monitoring
    PERFORMANCE_ALERTS = {
        'min_win_rate': 0.55,  # Alert if win rate falls below 55%
        'max_consecutive_losses': 5,  # Alert after 5 consecutive losses
        'min_profit_factor': 1.2,  # Alert if profit factor below 1.2
        'max_drawdown_percent': 8.0,  # Alert if drawdown exceeds 8%
        'min_regime_accuracy': 0.65  # Alert if regime detection accuracy falls
    }

    # =============================================================================
    # LOGGING AND DEBUGGING
    # =============================================================================

    # Logging configuration
    LOGGING_CONFIG = {
        'level': logging.INFO,
        'include_intelligence_details': True,
        'include_rag_responses': False,  # Can be verbose
        'include_performance_metrics': True,
        'log_regime_changes': True,
        'log_signal_filtering': True
    }

    # Debug settings
    DEBUG_SETTINGS = {
        'enable_debug_mode': False,
        'save_intelligence_data': False,
        'save_rag_responses': False,
        'detailed_signal_logging': False,
        'performance_profiling': False
    }

    # =============================================================================
    # FALLBACK CONFIGURATIONS
    # =============================================================================

    # Fallback settings when components are unavailable
    FALLBACK_CONFIG = {
        'no_intelligence_data': {
            'use_simple_ema_strategy': True,
            'ema_periods': [12, 26, 50],
            'confidence_reduction': 0.2,
            'position_size_reduction': 0.5
        },
        'no_rag_system': {
            'use_predefined_strategies': True,
            'default_strategy_type': 'ema_crossover',
            'confidence_reduction': 0.15,
            'strategy_rotation_hours': 6
        },
        'high_latency_mode': {
            'extend_cache_duration': True,
            'reduce_query_frequency': True,
            'simplify_analysis': True,
            'cache_duration_minutes': 15
        }
    }

    # Default strategy parameters for fallback mode
    DEFAULT_PARAMETERS = {
        'fast_ema': 12,
        'slow_ema': 26,
        'trend_ema': 50,
        'bb_period': 20,
        'bb_std_dev': 2.0,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'rsi_period': 14,
        'stop_loss_pips': 20,
        'take_profit_pips': 40,
        'risk_reward_ratio': 2.0
    }

    def __init__(self):
        """Initialize configuration with validation"""
        self.logger = logging.getLogger(__name__)
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate configuration settings"""
        try:
            # Validate regime configurations
            for regime, config in self.MARKET_REGIMES.items():
                if 'weight' not in config or not (0 < config['weight'] <= 1):
                    raise ValueError(f"Invalid weight for regime {regime}")

            # Validate session configurations
            for session, config in self.TRADING_SESSIONS.items():
                if 'start_hour' not in config or 'end_hour' not in config:
                    raise ValueError(f"Invalid hours for session {session}")

            # Validate risk management
            for regime, risk_config in self.RISK_MANAGEMENT.items():
                if 'stop_loss_pips' not in risk_config:
                    raise ValueError(f"Missing stop_loss_pips for regime {regime}")

            self.logger.info("✅ RAG Intelligence Strategy configuration validated successfully")

        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise

    def get_regime_config(self, regime: str) -> Dict[str, Any]:
        """Get configuration for specific market regime"""
        return self.MARKET_REGIMES.get(regime, self.MARKET_REGIMES['ranging'])

    def get_session_config(self, session: str) -> Dict[str, Any]:
        """Get configuration for specific trading session"""
        return self.TRADING_SESSIONS.get(session, self.TRADING_SESSIONS['new_york'])

    def get_adaptive_parameters(self, regime: str) -> Dict[str, Any]:
        """Get adaptive parameters for specific regime"""
        return self.ADAPTIVE_PARAMETERS.get(regime, self.DEFAULT_PARAMETERS)

    def get_risk_management(self, regime: str) -> Dict[str, Any]:
        """Get risk management settings for specific regime"""
        return self.RISK_MANAGEMENT.get(regime, self.RISK_MANAGEMENT['ranging'])

    def get_rag_query_template(self, regime: str) -> Dict[str, str]:
        """Get RAG query template for specific regime"""
        return self.RAG_QUERY_TEMPLATES.get(regime, self.RAG_QUERY_TEMPLATES['ranging'])