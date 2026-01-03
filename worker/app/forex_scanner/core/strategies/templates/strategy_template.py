#!/usr/bin/env python3
"""
TEMPLATE Strategy - [Brief Description]

VERSION: 1.0.0
DATE: YYYY-MM-DD
STATUS: Template - Copy and customize

INSTRUCTIONS:
1. Copy this file to core/strategies/your_strategy_name.py
2. Replace TEMPLATE with your strategy name (e.g., MY_MOMENTUM)
3. Replace all TODO comments with actual implementation
4. Create database migration using migrations/templates/strategy_config_template.sql
5. Enable in database (strategy_config.enabled_strategies table)

Strategy Architecture:
    - Tier 1: [Higher timeframe bias - e.g., 4H trend direction]
    - Tier 2: [Trigger timeframe - e.g., 1H confirmation signal]
    - Tier 3: [Entry timeframe - e.g., 15m entry timing]

Target Performance:
    - Win Rate: X%+
    - Profit Factor: X.X+
    - Average R:R: X.X:1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

# Strategy Registry - auto-registers on import
from .strategy_registry import register_strategy, StrategyInterface


# ==============================================================================
# CONFIGURATION DATACLASS
# ==============================================================================

@dataclass
class TemplateConfig:
    """
    Configuration for TEMPLATE strategy.

    Loaded from database (strategy_config.template_global_config).
    Falls back to defaults if database unavailable.
    """

    # Strategy identification
    strategy_name: str = "TEMPLATE"
    version: str = "1.0.0"

    # Timeframes
    htf_timeframe: str = "4h"      # Higher timeframe for bias
    trigger_timeframe: str = "1h"  # Trigger timeframe
    entry_timeframe: str = "15m"   # Entry timeframe

    # Core parameters - TODO: customize for your strategy
    ema_period: int = 50
    min_confidence: float = 0.60
    max_confidence: float = 0.90

    # Stop Loss / Take Profit
    fixed_stop_loss_pips: float = 15.0
    fixed_take_profit_pips: float = 25.0
    sl_buffer_pips: float = 3.0
    sl_atr_multiplier: float = 1.0

    # Filters
    volume_filter_enabled: bool = True
    volume_threshold: float = 1.5
    atr_filter_enabled: bool = True
    min_atr_pips: float = 5.0

    # Risk management
    min_risk_reward: float = 1.5
    max_position_size: float = 1.0

    # Cooldown settings
    signal_cooldown_minutes: int = 60

    # Enabled pairs (empty = all pairs enabled)
    enabled_pairs: List[str] = field(default_factory=list)

    @classmethod
    def from_database(cls, db_manager=None) -> 'TemplateConfig':
        """
        Load configuration from database.

        TODO: Replace 'template' with your strategy name in table references.
        """
        config = cls()

        if db_manager is None:
            return config

        try:
            # Try to load from strategy_config database
            query = """
                SELECT parameter_name, parameter_value, value_type
                FROM template_global_config  -- TODO: rename table
                WHERE is_active = TRUE
            """
            # TODO: Execute query and populate config fields
            # results = db_manager.execute_query(query)
            # for row in results:
            #     setattr(config, row['parameter_name'], row['parameter_value'])

        except Exception as e:
            logging.warning(f"Could not load TEMPLATE config from database: {e}")

        return config


# ==============================================================================
# CONFIG SERVICE (Singleton with caching)
# ==============================================================================

class TemplateConfigService:
    """
    Singleton service for loading and caching TEMPLATE configuration.

    Pattern: Based on smc_simple_config_service.py

    Usage:
        from .template_strategy import get_template_config
        config = get_template_config()
        sl = config.fixed_stop_loss_pips
    """

    _instance: Optional['TemplateConfigService'] = None
    _config: Optional[TemplateConfig] = None
    _last_refresh: Optional[datetime] = None
    _cache_ttl_seconds: int = 300  # 5 minute cache

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_manager = None
        self.logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> 'TemplateConfigService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_db_manager(self, db_manager) -> None:
        """Set database manager for config loading"""
        self._db_manager = db_manager
        self._config = None  # Force refresh on next access

    def get_config(self) -> TemplateConfig:
        """Get configuration with caching"""
        now = datetime.now()

        if (self._config is None or
            self._last_refresh is None or
            (now - self._last_refresh).seconds > self._cache_ttl_seconds):
            self._config = TemplateConfig.from_database(self._db_manager)
            self._last_refresh = now
            self.logger.debug("Refreshed TEMPLATE config from database")

        return self._config

    def refresh(self) -> TemplateConfig:
        """Force refresh configuration"""
        self._config = None
        return self.get_config()


def get_template_config() -> TemplateConfig:
    """Convenience function to get TEMPLATE configuration"""
    return TemplateConfigService.get_instance().get_config()


# ==============================================================================
# STRATEGY CLASS
# ==============================================================================

@register_strategy('TEMPLATE')  # Auto-registers with Strategy Registry
class TemplateStrategy(StrategyInterface):
    """
    TEMPLATE Strategy Implementation

    TODO: Customize the detect_signal method for your strategy logic.
    """

    def __init__(self, config=None, logger=None, db_manager=None):
        """
        Initialize TEMPLATE Strategy

        Args:
            config: Optional legacy config module (ignored, we use database)
            logger: Logger instance
            db_manager: Database manager for config loading
        """
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager

        # Load configuration from database
        if db_manager:
            TemplateConfigService.get_instance().set_db_manager(db_manager)
        self.config = get_template_config()

        # Cooldown tracking (per-pair)
        self._cooldowns: Dict[str, datetime] = {}

        # Rejection logging (for analysis)
        self._pending_rejections: List[Dict] = []

        self.logger.info(f"TEMPLATE Strategy v{self.config.version} initialized")

    @property
    def strategy_name(self) -> str:
        """Unique name for this strategy (required by StrategyInterface)"""
        return "TEMPLATE"

    def get_required_timeframes(self) -> List[str]:
        """
        Get list of timeframes this strategy requires.

        Returns:
            List of timeframe strings
        """
        return [
            self.config.htf_timeframe,
            self.config.trigger_timeframe,
            self.config.entry_timeframe
        ]

    def detect_signal(
        self,
        df_trigger: pd.DataFrame = None,
        df_4h: pd.DataFrame = None,
        epic: str = "",
        pair: str = "",
        df_entry: pd.DataFrame = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        Detect trading signal using TEMPLATE strategy logic.

        Args:
            df_trigger: Trigger timeframe data (e.g., 1H)
            df_4h: Higher timeframe data for bias (4H)
            epic: IG epic identifier
            pair: Currency pair name
            df_entry: Entry timeframe data (e.g., 15m)
            **kwargs: Additional parameters

        Returns:
            Signal dict if detected, None otherwise
        """
        # Validate input data
        if df_trigger is None or len(df_trigger) < 50:
            self.logger.debug(f"[TEMPLATE] Insufficient trigger data for {epic}")
            return None

        if df_4h is None or len(df_4h) < 20:
            self.logger.debug(f"[TEMPLATE] Insufficient 4H data for {epic}")
            return None

        # Check enabled pairs filter
        if self.config.enabled_pairs and epic not in self.config.enabled_pairs:
            self.logger.debug(f"[TEMPLATE] {epic} not in enabled pairs, skipping")
            return None

        # Check cooldown
        if not self._check_cooldown(epic):
            return None

        # =====================================================================
        # TODO: IMPLEMENT YOUR STRATEGY LOGIC HERE
        # =====================================================================

        # Example structure:

        # Step 1: Check HTF bias (4H)
        # htf_bias = self._check_htf_bias(df_4h)
        # if htf_bias is None:
        #     return None

        # Step 2: Check trigger signal
        # trigger_signal = self._check_trigger(df_trigger, htf_bias)
        # if trigger_signal is None:
        #     return None

        # Step 3: Check entry timing
        # entry_valid = self._check_entry(df_entry, trigger_signal)
        # if not entry_valid:
        #     return None

        # Step 4: Calculate confidence score
        # confidence = self._calculate_confidence(...)
        # if confidence < self.config.min_confidence:
        #     self._log_rejection(epic, "Low confidence", confidence)
        #     return None

        # Step 5: Calculate SL/TP
        # sl_pips, tp_pips = self._calculate_sl_tp(...)

        # Step 6: Build and return signal
        # signal = self._build_signal(epic, pair, ...)
        # self._set_cooldown(epic)
        # return signal

        # For template, return None (no signal)
        self.logger.debug(f"[TEMPLATE] Strategy not implemented for {epic}")
        return None

    # =========================================================================
    # HELPER METHODS - TODO: Implement these for your strategy
    # =========================================================================

    def _check_htf_bias(self, df_4h: pd.DataFrame) -> Optional[str]:
        """
        Check higher timeframe bias.

        Returns:
            'bullish', 'bearish', or None if no clear bias
        """
        # TODO: Implement HTF bias logic
        # Example: Check if price is above/below 50 EMA
        pass

    def _check_trigger(self, df_trigger: pd.DataFrame, bias: str) -> Optional[Dict]:
        """
        Check for trigger signal on trigger timeframe.

        Returns:
            Trigger dict with signal details, or None
        """
        # TODO: Implement trigger logic
        pass

    def _check_entry(self, df_entry: pd.DataFrame, trigger: Dict) -> bool:
        """
        Check if entry conditions are met on entry timeframe.

        Returns:
            True if entry is valid
        """
        # TODO: Implement entry timing logic
        pass

    def _calculate_confidence(self, **kwargs) -> float:
        """
        Calculate confidence score (0.0 to 1.0).

        Recommended components (each ~20% weight):
        1. Trend alignment
        2. Signal quality
        3. Entry timing
        4. Volume confirmation
        5. Risk:reward ratio
        """
        # TODO: Implement confidence calculation
        return 0.0

    def _calculate_sl_tp(self, df: pd.DataFrame, direction: str, epic: str) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit in pips.

        Returns:
            (sl_pips, tp_pips)
        """
        # Use fixed values from config as baseline
        sl_pips = self.config.fixed_stop_loss_pips
        tp_pips = self.config.fixed_take_profit_pips

        # TODO: Add ATR-based dynamic SL/TP if desired

        return sl_pips, tp_pips

    def _build_signal(
        self,
        epic: str,
        pair: str,
        direction: str,
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
        confidence: float,
        **kwargs
    ) -> Dict:
        """
        Build standardized signal dictionary.
        """
        now = datetime.now(timezone.utc)

        signal = {
            # Core signal data
            'signal': direction,  # 'BUY' or 'SELL'
            'signal_type': direction.lower(),
            'strategy': self.strategy_name,
            'epic': epic,
            'pair': pair,

            # Prices
            'entry_price': entry_price,
            'stop_loss_pips': sl_pips,
            'take_profit_pips': tp_pips,

            # Confidence
            'confidence_score': confidence,
            'confidence': confidence,

            # Metadata
            'signal_timestamp': now.isoformat(),
            'timestamp': now,
            'version': self.config.version,

            # Strategy-specific indicators (customize)
            'strategy_indicators': kwargs.get('indicators', {})
        }

        return signal

    # =========================================================================
    # COOLDOWN MANAGEMENT
    # =========================================================================

    def _check_cooldown(self, epic: str) -> bool:
        """Check if epic is in cooldown period"""
        if epic not in self._cooldowns:
            return True

        now = datetime.now(timezone.utc)
        cooldown_end = self._cooldowns[epic]

        if now >= cooldown_end:
            del self._cooldowns[epic]
            return True

        remaining = (cooldown_end - now).seconds // 60
        self.logger.debug(f"[TEMPLATE] {epic} in cooldown ({remaining}m remaining)")
        return False

    def _set_cooldown(self, epic: str) -> None:
        """Set cooldown for epic after signal"""
        cooldown_minutes = self.config.signal_cooldown_minutes
        self._cooldowns[epic] = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)

    def reset_cooldowns(self) -> None:
        """Reset all cooldowns (for backtesting)"""
        self._cooldowns.clear()
        self.logger.debug("[TEMPLATE] Cooldowns reset")

    # =========================================================================
    # REJECTION LOGGING
    # =========================================================================

    def _log_rejection(self, epic: str, reason: str, value: Any = None) -> None:
        """Log signal rejection for analysis"""
        self._pending_rejections.append({
            'epic': epic,
            'strategy': self.strategy_name,
            'reason': reason,
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def flush_rejections(self) -> None:
        """Flush pending rejections to database (optional)"""
        if not self._pending_rejections:
            return

        # TODO: Optionally write rejections to database for analysis
        # if self.db_manager:
        #     self.db_manager.bulk_insert('signal_rejections', self._pending_rejections)

        self._pending_rejections.clear()


# ==============================================================================
# FACTORY FUNCTION (for backward compatibility)
# ==============================================================================

def create_template_strategy(
    config=None,
    db_manager=None,
    logger=None
) -> TemplateStrategy:
    """
    Factory function to create TEMPLATE strategy instance.

    Usage:
        from forex_scanner.core.strategies.template_strategy import create_template_strategy
        strategy = create_template_strategy(db_manager=db_manager)
    """
    return TemplateStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager
    )
