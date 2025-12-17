# core/signal_detector.py
"""
Signal Detection Coordinator
Lightweight coordinator that delegates to specialized strategy modules
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional,Union, Tuple
from datetime import datetime, timedelta

try:
    from .database import DatabaseManager
    from .data_fetcher import DataFetcher
    from .strategies.ema_strategy import EMAStrategy
    from .strategies.macd_strategy import MACDStrategy
    # from .strategies.combined_strategy import CombinedStrategy  # Removed - strategy was disabled
    from .backtest.backtest_engine import BacktestEngine
    from .backtest.performance_analyzer import PerformanceAnalyzer
    from .backtest.signal_analyzer import SignalAnalyzer
    from .detection.price_adjuster import PriceAdjuster
    from .strategies.scalping_strategy import ScalpingStrategy
    from .strategies.bb_supertrend_strategy import BollingerSupertrendStrategy
    from .smart_money_integration import add_smart_money_to_signal, add_smart_money_to_signals
    from .strategies.zero_lag_strategy import ZeroLagStrategy
    from .strategies.mean_reversion_strategy import MeanReversionStrategy
    from .strategies.ranging_market_strategy import RangingMarketStrategy
    from .strategies.momentum_strategy import MomentumStrategy
    from .strategies.volume_profile_strategy import VolumeProfileStrategy
    from .detection.large_candle_filter import LargeCandleFilter
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.strategies.ema_strategy import EMAStrategy
    from forex_scanner.core.strategies.macd_strategy import MACDStrategy
    # from forex_scanner.core.strategies.combined_strategy import CombinedStrategy  # Removed - strategy was disabled
    from forex_scanner.core.backtest.backtest_engine import BacktestEngine
    from forex_scanner.core.backtest.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.core.backtest.signal_analyzer import SignalAnalyzer
    from forex_scanner.core.detection.price_adjuster import PriceAdjuster
    from forex_scanner.core.strategies.scalping_strategy import ScalpingStrategy
    from forex_scanner.core.strategies.bb_supertrend_strategy import BollingerSupertrendStrategy
    from forex_scanner.core.smart_money_integration import add_smart_money_to_signal, add_smart_money_to_signals
    from forex_scanner.core.strategies.zero_lag_strategy import ZeroLagStrategy
    from forex_scanner.core.strategies.mean_reversion_strategy import MeanReversionStrategy
    from forex_scanner.core.strategies.ranging_market_strategy import RangingMarketStrategy
    from forex_scanner.core.strategies.momentum_strategy import MomentumStrategy
    from forex_scanner.core.strategies.volume_profile_strategy import VolumeProfileStrategy
    from forex_scanner.core.detection.large_candle_filter import LargeCandleFilter

try:
    from configdata import config
    import config as system_config
except ImportError:
    from forex_scanner.configdata import config
    from forex_scanner import config as system_config


class SignalDetector:
    """Lightweight signal detection coordinator"""
    
    def __init__(self, db_manager: DatabaseManager, user_timezone: str = 'Europe/Stockholm'):
        self.db_manager = db_manager
        self.data_fetcher = DataFetcher(db_manager, user_timezone)
        self.price_adjuster = PriceAdjuster()
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self.ema_strategy = EMAStrategy(data_fetcher=self.data_fetcher)
        # MACD strategy will be created per-epic with optimized parameters
        self.macd_strategy = None  # Will be created when needed with epic parameter
        self.macd_strategies_cache = {}  # Cache epic-specific strategies
        # self.combined_strategy = CombinedStrategy(data_fetcher=self.data_fetcher)  # Removed - strategy was disabled
        # 🔥 Initialize Scalping strategy with mode from config (if enabled)
        if getattr(config, 'SCALPING_STRATEGY_ENABLED', False):
            try:
                from configdata.strategies.config_scalping_strategy import SCALPING_MODE
                scalping_mode = SCALPING_MODE
            except ImportError:
                try:
                    from forex_scanner.configdata.strategies.config_scalping_strategy import SCALPING_MODE
                    scalping_mode = SCALPING_MODE
                except ImportError:
                    scalping_mode = 'linda_raschke'  # Fallback

            self.scalping_strategy = ScalpingStrategy(scalping_mode=scalping_mode)
            self.logger.info(f"✅ Scalping strategy initialized with mode: {scalping_mode}")
        else:
            self.scalping_strategy = None
            self.logger.info("⚪ Scalping strategy disabled")

        self.large_candle_filter = LargeCandleFilter()
        self.logger.info("✅ Large candle filter initialized")
        
        # Initialize KAMA strategy if enabled
        if getattr(config, 'KAMA_STRATEGY', False):
            try:
                from .strategies.kama_strategy import KAMAStrategy
                kama_config = getattr(config, 'DEFAULT_KAMA_CONFIG', 'default')
                self.kama_strategy = KAMAStrategy()
                self.logger.info(f"✅ KAMA Strategy initialized with '{kama_config}' configuration")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import KAMA strategy: {e}")
                self.kama_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize KAMA strategy: {e}")
                self.kama_strategy = None
        else:
            self.kama_strategy = None
            self.logger.debug("⚠️ KAMA Strategy disabled in configuration")
        
        # Initialize BB+Supertrend strategy if enabled
        if getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False):
            try:
                bb_config = getattr(config, 'DEFAULT_BB_SUPERTREND_CONFIG', 'default')
                self.bb_supertrend_strategy = BollingerSupertrendStrategy(config_name=bb_config)
                self.logger.info(f"✅ BB+Supertrend Strategy initialized with '{bb_config}' configuration")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import BB+Supertrend strategy: {e}")
                self.bb_supertrend_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize BB+Supertrend strategy: {e}")
                self.bb_supertrend_strategy = None
        else:
            self.bb_supertrend_strategy = None
            self.logger.debug("⚠️ BB+Supertrend Strategy disabled in configuration")
        
        if getattr(config, 'ZERO_LAG_STRATEGY', False):
            self.zero_lag_strategy = ZeroLagStrategy()
            self.logger.info("✅ Zero Lag EMA strategy initialized")
        else:
            self.zero_lag_strategy = None


        # Initialize Advanced Momentum Strategy if enabled
        if getattr(config, 'MOMENTUM_STRATEGY', False):
            try:
                # Get the active momentum config name for proper logging
                try:
                    from configdata.strategies.config_momentum_strategy import ACTIVE_MOMENTUM_CONFIG
                    active_momentum_config = ACTIVE_MOMENTUM_CONFIG
                except ImportError:
                    active_momentum_config = 'default'

                self.momentum_strategy = MomentumStrategy(data_fetcher=self.data_fetcher)
                self.logger.info(f"✅ Advanced Momentum strategy initialized with '{active_momentum_config}' configuration")
                self.logger.info("🚀 Features: minimal lag oscillators, velocity momentum, volume confirmation")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import Advanced Momentum strategy: {e}")
                self.momentum_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Advanced Momentum strategy: {e}")
                self.momentum_strategy = None
        else:
            self.momentum_strategy = None
            self.logger.info("⚪ Advanced Momentum strategy disabled")

        # Initialize SMC Strategy if enabled
        if getattr(config, 'SMC_STRATEGY', False):
            try:
                from .strategies.smc_strategy_fast import SMCStrategyFast
                # Get the active SMC config name for proper logging
                try:
                    from configdata.strategies.config_smc_strategy import ACTIVE_SMC_CONFIG
                    active_smc_config = ACTIVE_SMC_CONFIG
                except ImportError:
                    active_smc_config = 'default'

                self.smc_strategy = SMCStrategyFast(
                    smc_config_name=active_smc_config,
                    data_fetcher=self.data_fetcher  # Pass data_fetcher for MTF analysis
                )
                self.logger.info("✅ SMC (Smart Money Concepts) strategy initialized")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import SMC strategy: {e}")
                self.smc_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize SMC strategy: {e}")
                self.smc_strategy = None
        else:
            self.smc_strategy = None
            self.logger.info("⚪ SMC strategy disabled")

        # Initialize SMC Structure Strategy if enabled (v2.4.0 - Profitable)
        if getattr(config, 'SMC_STRUCTURE_STRATEGY', False):
            try:
                # SMC Structure strategy uses lazy loading in detect_smc_structure_signals
                # Just set a flag to indicate it's enabled
                self.smc_structure_enabled = True
                self.smc_structure_strategy = None  # Will be lazy-loaded on first use
                self.logger.info("✅ SMC Structure strategy v2.4.0 enabled (lazy-load)")
            except Exception as e:
                self.logger.error(f"❌ Failed to enable SMC Structure strategy: {e}")
                self.smc_structure_enabled = False
                self.smc_structure_strategy = None
        else:
            self.smc_structure_enabled = False
            self.smc_structure_strategy = None
            self.logger.info("⚪ SMC Structure strategy disabled")

        # Initialize SMC Simple Strategy if enabled (v1.0.0 - 3-Tier EMA)
        # NOTE: Use system_config (main config.py) not configdata.config for strategy flags
        if getattr(system_config, 'SMC_SIMPLE_STRATEGY', False):
            try:
                # SMC Simple strategy uses lazy loading for consistency
                self.smc_simple_enabled = True
                self.smc_simple_strategy = None  # Will be lazy-loaded on first use
                self.logger.info("✅ SMC Simple strategy v1.0.0 enabled (3-tier EMA, lazy-load)")
            except Exception as e:
                self.logger.error(f"❌ Failed to enable SMC Simple strategy: {e}")
                self.smc_simple_enabled = False
                self.smc_simple_strategy = None
        else:
            self.smc_simple_enabled = False
            self.smc_simple_strategy = None
            self.logger.info("⚪ SMC Simple strategy disabled")

        # Initialize Ichimoku Strategy if enabled
        if getattr(config, 'ICHIMOKU_CLOUD_STRATEGY', False):
            try:
                from .strategies.ichimoku_strategy import IchimokuStrategy
                # Get the active Ichimoku config name for proper logging
                try:
                    from configdata.strategies.config_ichimoku_strategy import ACTIVE_ICHIMOKU_CONFIG
                    active_ichimoku_config = ACTIVE_ICHIMOKU_CONFIG
                except ImportError:
                    active_ichimoku_config = 'traditional'

                self.ichimoku_strategy = IchimokuStrategy(data_fetcher=self.data_fetcher)
                self.logger.info(f"✅ Ichimoku Cloud strategy initialized with '{active_ichimoku_config}' configuration")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import Ichimoku strategy: {e}")
                self.ichimoku_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Ichimoku strategy: {e}")
                self.ichimoku_strategy = None
        else:
            self.ichimoku_strategy = None
            self.logger.info("⚪ Ichimoku Cloud strategy disabled")

        # Initialize Mean Reversion Strategy if enabled
        if getattr(config, 'MEAN_REVERSION_STRATEGY', False):
            try:
                self.mean_reversion_strategy = MeanReversionStrategy()
                self.logger.info("✅ Mean Reversion strategy initialized")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import Mean Reversion strategy: {e}")
                self.mean_reversion_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Mean Reversion strategy: {e}")
                self.mean_reversion_strategy = None
        else:
            self.mean_reversion_strategy = None
            self.logger.info("⚪ Mean Reversion strategy disabled")

        # Initialize Ranging Market Strategy if enabled
        if getattr(config, 'RANGING_MARKET_STRATEGY', False):
            try:
                self.ranging_market_strategy = RangingMarketStrategy(data_fetcher=self.data_fetcher)
                self.logger.info("✅ Ranging Market strategy initialized")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import Ranging Market strategy: {e}")
                self.ranging_market_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Ranging Market strategy: {e}")
                self.ranging_market_strategy = None
        else:
            self.ranging_market_strategy = None
            self.logger.info("⚪ Ranging Market strategy disabled")

        # Initialize Volume Profile Strategy if enabled
        if getattr(config, 'VOLUME_PROFILE_STRATEGY', False):
            try:
                # Get the active Volume Profile config name for proper logging
                try:
                    from configdata.strategies.config_volume_profile_strategy import ACTIVE_VP_CONFIG
                    active_vp_config = ACTIVE_VP_CONFIG
                except ImportError:
                    active_vp_config = 'default'

                self.volume_profile_strategy = VolumeProfileStrategy(data_fetcher=self.data_fetcher)
                self.logger.info(f"✅ Volume Profile strategy initialized with '{active_vp_config}' configuration")
                self.logger.info("📊 Features: HVN/LVN zones, POC reversion, Value Area breakouts")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import Volume Profile strategy: {e}")
                self.volume_profile_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Volume Profile strategy: {e}")
                self.volume_profile_strategy = None
        else:
            self.volume_profile_strategy = None
            self.logger.info("⚪ Volume Profile strategy disabled")

        # Initialize EMA Double Confirmation Strategy if enabled
        if getattr(system_config, 'EMA_DOUBLE_CONFIRMATION_STRATEGY', False):
            try:
                from .strategies.ema_double_confirmation_strategy import EMADoubleConfirmationStrategy
                self.ema_double_confirmation_strategy = EMADoubleConfirmationStrategy(
                    data_fetcher=self.data_fetcher,
                    db_manager=self.db_manager  # v2.2.0: Pass db_manager for persistent crossover state
                )
                self.logger.info("✅ EMA Double Confirmation strategy initialized (9/21 crossover with DB persistence)")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import EMA Double Confirmation strategy: {e}")
                self.ema_double_confirmation_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize EMA Double Confirmation strategy: {e}")
                self.ema_double_confirmation_strategy = None
        else:
            self.ema_double_confirmation_strategy = None
            self.logger.info("⚪ EMA Double Confirmation strategy disabled")

        # Initialize ICT Silver Bullet Strategy if enabled
        if getattr(system_config, 'SILVER_BULLET_STRATEGY', False):
            try:
                from .strategies.silver_bullet_strategy import SilverBulletStrategy
                self.silver_bullet_strategy = SilverBulletStrategy(
                    config=config,
                    logger=self.logger
                )
                self.logger.info("✅ ICT Silver Bullet strategy initialized (time-based SMC)")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import Silver Bullet strategy: {e}")
                self.silver_bullet_strategy = None
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Silver Bullet strategy: {e}")
                self.silver_bullet_strategy = None
        else:
            self.silver_bullet_strategy = None
            self.logger.info("⚪ Silver Bullet strategy disabled")

        # Initialize analysis components
        self.backtest_engine = BacktestEngine(self.data_fetcher)
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()

    # =========================================================================
    # BACKTEST FORCE-INITIALIZATION METHODS
    # These methods allow backtests to initialize strategies regardless of config flags
    # =========================================================================

    def force_initialize_strategy(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Force-initialize a specific strategy for backtesting, regardless of config flags.

        This allows backtests to test any strategy without modifying config.py,
        which would otherwise affect the live system.

        Args:
            strategy_name: Strategy name (e.g., 'MACD', 'KAMA', 'MOMENTUM')

        Returns:
            Tuple of (success: bool, message: str)
        """
        strategy_name = strategy_name.upper()

        # Strategy initialization mapping
        init_map = {
            # EMA strategies are always available
            'EMA': lambda: (True, "EMA strategy always available"),
            'EMA_CROSSOVER': lambda: (True, "EMA strategy always available"),
            # MACD
            'MACD': self._force_init_macd,
            'MACD_EMA': self._force_init_macd,
            # KAMA
            'KAMA': self._force_init_kama,
            # Bollinger + Supertrend
            'BOLLINGER_SUPERTREND': self._force_init_bb_supertrend,
            'BB_SUPERTREND': self._force_init_bb_supertrend,
            'BB': self._force_init_bb_supertrend,
            # Zero Lag
            'ZERO_LAG': self._force_init_zero_lag,
            'ZEROLAG': self._force_init_zero_lag,
            'ZL': self._force_init_zero_lag,
            # Momentum
            'MOMENTUM': self._force_init_momentum,
            # SMC variants
            'SMC_FAST': self._force_init_smc_fast,
            'SMC': self._force_init_smc_fast,
            'SMC_STRUCTURE': self._force_init_smc_structure,
            'SMC_PURE': self._force_init_smc_structure,
            'SMC_SIMPLE': self._force_init_smc_simple,
            'SMC_EMA': self._force_init_smc_simple,
            # Ichimoku
            'ICHIMOKU': self._force_init_ichimoku,
            'ICHIMOKU_CLOUD': self._force_init_ichimoku,
            # Mean Reversion
            'MEAN_REVERSION': self._force_init_mean_reversion,
            'MEANREV': self._force_init_mean_reversion,
            # Ranging Market
            'RANGING_MARKET': self._force_init_ranging_market,
            'RANGING': self._force_init_ranging_market,
            # Scalping
            'SCALPING': self._force_init_scalping,
            'SCALP': self._force_init_scalping,
            # Volume Profile
            'VOLUME_PROFILE': self._force_init_volume_profile,
            'VP': self._force_init_volume_profile,
            # EMA Double Confirmation
            'EMA_DOUBLE_CONFIRMATION': self._force_init_ema_double_confirmation,
            'EMA_DOUBLE': self._force_init_ema_double_confirmation,
            'EDC': self._force_init_ema_double_confirmation,
            # ICT Silver Bullet
            'SILVER_BULLET': self._force_init_silver_bullet,
            'SB': self._force_init_silver_bullet,
            'ICT_SILVER_BULLET': self._force_init_silver_bullet,
        }

        if strategy_name not in init_map:
            return False, f"Unknown strategy: {strategy_name}"

        return init_map[strategy_name]()

    def _force_init_macd(self) -> Tuple[bool, str]:
        """Force-initialize MACD strategy for backtest"""
        try:
            # MACD uses lazy-loading per epic, just ensure cache is ready
            self.macd_strategy = None
            self.macd_strategies_cache = {}
            self.logger.info("🔧 Force-initialized MACD strategy (lazy-load per epic)")
            return True, "MACD strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init MACD: {e}"

    def _force_init_kama(self) -> Tuple[bool, str]:
        """Force-initialize KAMA strategy for backtest"""
        try:
            from .strategies.kama_strategy import KAMAStrategy
            self.kama_strategy = KAMAStrategy()
            self.logger.info("🔧 Force-initialized KAMA strategy")
            return True, "KAMA strategy force-initialized"
        except ImportError as e:
            return False, f"Failed to import KAMA strategy: {e}"
        except Exception as e:
            return False, f"Failed to force-init KAMA: {e}"

    def _force_init_bb_supertrend(self) -> Tuple[bool, str]:
        """Force-initialize Bollinger + Supertrend strategy for backtest"""
        try:
            bb_config = getattr(config, 'DEFAULT_BB_SUPERTREND_CONFIG', 'default')
            self.bb_supertrend_strategy = BollingerSupertrendStrategy(config_name=bb_config)
            self.logger.info(f"🔧 Force-initialized BB+Supertrend strategy (config: {bb_config})")
            return True, "BB+Supertrend strategy force-initialized"
        except ImportError as e:
            return False, f"Failed to import BB+Supertrend strategy: {e}"
        except Exception as e:
            return False, f"Failed to force-init BB+Supertrend: {e}"

    def _force_init_zero_lag(self) -> Tuple[bool, str]:
        """Force-initialize Zero Lag strategy for backtest"""
        try:
            self.zero_lag_strategy = ZeroLagStrategy()
            self.logger.info("🔧 Force-initialized Zero Lag EMA strategy")
            return True, "Zero Lag strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init Zero Lag: {e}"

    def _force_init_momentum(self) -> Tuple[bool, str]:
        """Force-initialize Momentum strategy for backtest"""
        try:
            self.momentum_strategy = MomentumStrategy(data_fetcher=self.data_fetcher)
            self.logger.info("🔧 Force-initialized Advanced Momentum strategy")
            return True, "Momentum strategy force-initialized"
        except ImportError as e:
            return False, f"Failed to import Momentum strategy: {e}"
        except Exception as e:
            return False, f"Failed to force-init Momentum: {e}"

    def _force_init_smc_fast(self) -> Tuple[bool, str]:
        """Force-initialize SMC Fast strategy for backtest"""
        try:
            from .strategies.smc_strategy_fast import SMCStrategyFast
            try:
                from configdata.strategies.config_smc_strategy import ACTIVE_SMC_CONFIG
                active_smc_config = ACTIVE_SMC_CONFIG
            except ImportError:
                active_smc_config = 'default'

            self.smc_strategy = SMCStrategyFast(
                smc_config_name=active_smc_config,
                data_fetcher=self.data_fetcher
            )
            self.logger.info(f"🔧 Force-initialized SMC Fast strategy (config: {active_smc_config})")
            return True, "SMC Fast strategy force-initialized"
        except ImportError as e:
            return False, f"Failed to import SMC Fast strategy: {e}"
        except Exception as e:
            return False, f"Failed to force-init SMC Fast: {e}"

    def _force_init_smc_structure(self) -> Tuple[bool, str]:
        """Force-initialize SMC Structure strategy for backtest"""
        try:
            # SMC Structure uses lazy loading, just enable the flag
            self.smc_structure_enabled = True
            self.smc_structure_strategy = None  # Will be lazy-loaded on first use
            self.logger.info("🔧 Force-initialized SMC Structure strategy (lazy-load)")
            return True, "SMC Structure strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init SMC Structure: {e}"

    def _force_init_smc_simple(self) -> Tuple[bool, str]:
        """Force-initialize SMC Simple strategy for backtest"""
        try:
            # SMC Simple uses lazy loading, just enable the flag
            self.smc_simple_enabled = True
            self.smc_simple_strategy = None  # Will be lazy-loaded on first use
            self.logger.info("🔧 Force-initialized SMC Simple strategy (lazy-load)")
            return True, "SMC Simple strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init SMC Simple: {e}"

    def _force_init_ichimoku(self) -> Tuple[bool, str]:
        """Force-initialize Ichimoku strategy for backtest"""
        try:
            from .strategies.ichimoku_strategy import IchimokuStrategy
            self.ichimoku_strategy = IchimokuStrategy(data_fetcher=self.data_fetcher)
            self.logger.info("🔧 Force-initialized Ichimoku Cloud strategy")
            return True, "Ichimoku strategy force-initialized"
        except ImportError as e:
            return False, f"Failed to import Ichimoku strategy: {e}"
        except Exception as e:
            return False, f"Failed to force-init Ichimoku: {e}"

    def _force_init_mean_reversion(self) -> Tuple[bool, str]:
        """Force-initialize Mean Reversion strategy for backtest"""
        try:
            self.mean_reversion_strategy = MeanReversionStrategy()
            self.logger.info("🔧 Force-initialized Mean Reversion strategy")
            return True, "Mean Reversion strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init Mean Reversion: {e}"

    def _force_init_ranging_market(self) -> Tuple[bool, str]:
        """Force-initialize Ranging Market strategy for backtest"""
        try:
            self.ranging_market_strategy = RangingMarketStrategy(data_fetcher=self.data_fetcher)
            self.logger.info("🔧 Force-initialized Ranging Market strategy")
            return True, "Ranging Market strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init Ranging Market: {e}"

    def _force_init_scalping(self) -> Tuple[bool, str]:
        """Force-initialize Scalping strategy for backtest"""
        try:
            try:
                from configdata.strategies.config_scalping_strategy import SCALPING_MODE
                scalping_mode = SCALPING_MODE
            except ImportError:
                try:
                    from forex_scanner.configdata.strategies.config_scalping_strategy import SCALPING_MODE
                    scalping_mode = SCALPING_MODE
                except ImportError:
                    scalping_mode = 'linda_raschke'

            self.scalping_strategy = ScalpingStrategy(scalping_mode=scalping_mode)
            self.logger.info(f"🔧 Force-initialized Scalping strategy (mode: {scalping_mode})")
            return True, "Scalping strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init Scalping: {e}"

    def _force_init_volume_profile(self) -> Tuple[bool, str]:
        """Force-initialize Volume Profile strategy for backtest"""
        try:
            self.volume_profile_strategy = VolumeProfileStrategy(data_fetcher=self.data_fetcher)
            self.logger.info("🔧 Force-initialized Volume Profile strategy")
            return True, "Volume Profile strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init Volume Profile: {e}"

    def _force_init_ema_double_confirmation(self) -> Tuple[bool, str]:
        """Force-initialize EMA Double Confirmation strategy for backtest"""
        try:
            from .strategies.ema_double_confirmation_strategy import EMADoubleConfirmationStrategy
            self.ema_double_confirmation_strategy = EMADoubleConfirmationStrategy(
                data_fetcher=self.data_fetcher,
                db_manager=self.db_manager  # v2.2.0: Pass db_manager for persistent crossover state
            )
            self.logger.info("🔧 Force-initialized EMA Double Confirmation strategy (with DB persistence)")
            return True, "EMA Double Confirmation strategy force-initialized"
        except ImportError as e:
            return False, f"Failed to import EMA Double Confirmation strategy: {e}"
        except Exception as e:
            return False, f"Failed to force-init EMA Double Confirmation: {e}"

    def _force_init_silver_bullet(self) -> Tuple[bool, str]:
        """Force-initialize ICT Silver Bullet strategy for backtest"""
        try:
            from .strategies.silver_bullet_strategy import SilverBulletStrategy
            self.silver_bullet_strategy = SilverBulletStrategy(
                config=config,
                logger=self.logger
            )
            self.logger.info("🔧 Force-initialized ICT Silver Bullet strategy")
            return True, "Silver Bullet strategy force-initialized"
        except ImportError as e:
            return False, f"Failed to import Silver Bullet strategy: {e}"
        except Exception as e:
            return False, f"Failed to force-init Silver Bullet: {e}"

    # =========================================================================
    # END BACKTEST FORCE-INITIALIZATION METHODS
    # =========================================================================

    def _get_default_timeframe(self, timeframe: str = None) -> str:
        """Get default timeframe from config if not specified"""
        if timeframe is None:
            return getattr(config, 'DEFAULT_TIMEFRAME', '15m')
        return timeframe
    
    def detect_signals_bid_adjusted(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect EMA signals with BID price adjustment to MID prices
        """
        # Use default timeframe if not specified
        timeframe = self._get_default_timeframe(timeframe)
            
        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe,ema_strategy=self.ema_strategy)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Use EMA strategy with BID adjustment
            signal = self.ema_strategy.detect_signal_auto(df, epic, spread_pips, timeframe)
            signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting BID-adjusted signals for {epic}: {e}")
            return None
    
    def _get_macd_strategy_for_epic(self, epic: str) -> 'MACDStrategy':
        """Get or create epic-specific MACD strategy with optimized parameters"""
        if epic not in self.macd_strategies_cache:
            # Detect if we're in backtest mode by checking data_fetcher type
            is_backtest = 'Backtest' in str(type(self.data_fetcher).__name__)

            try:
                # Create MACD strategy with epic-specific optimized parameters
                self.macd_strategies_cache[epic] = MACDStrategy(
                    data_fetcher=self.data_fetcher,
                    epic=epic,
                    use_optimized_parameters=True,
                    backtest_mode=is_backtest
                )
                self.logger.debug(f"✅ Created optimized MACD strategy for {epic} (backtest_mode={is_backtest})")
            except Exception as e:
                self.logger.warning(f"Failed to create optimized MACD strategy for {epic}: {e}")
                # Fallback to basic MACD strategy
                self.macd_strategies_cache[epic] = MACDStrategy(
                    data_fetcher=self.data_fetcher,
                    epic=epic,
                    use_optimized_parameters=False,
                    backtest_mode=is_backtest
                )
        
        return self.macd_strategies_cache[epic]
    
    def detect_signals_mid_prices(self, epic: str, pair: str, timeframe: str = None) -> Optional[Dict]:
        """
        Detect EMA signals using MID prices (no adjustment needed)
        """
        # Use default timeframe if not specified
        timeframe = self._get_default_timeframe(timeframe)
            
        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe,ema_strategy=self.ema_strategy)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Use EMA strategy without BID adjustment
            signal = self.ema_strategy.detect_signal_auto(df, epic, 0, timeframe)  # 0 spread = no adjustment

            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting MID-price signals for {epic}: {e}")
            return None

    def detect_macd_ema_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect MACD + EMA 200 signals with Multi-Timeframe Analysis

        🔒 HARD-CODED: MACD strategy ALWAYS uses 1H timeframe
        """
        # 🔒 CRITICAL FIX: Force 1H timeframe for MACD strategy (ignore scanner default)
        # The MACD strategy's internal timeframe lock at macd_strategy.py:76 was ineffective
        # because data was already fetched at the wrong timeframe. This is the correct place to fix it.
        timeframe = '1h'
        self.logger.info(f"🔒 [MACD] Forcing 1H timeframe for {epic} (overriding scanner default)")

        try:
            # Get enhanced data with MACD indicators pre-calculated
            df = self.data_fetcher.get_enhanced_data(
                epic, pair, timeframe=timeframe,
                ema_strategy=self.ema_strategy,
                required_indicators=['macd']  # Pre-calculate MACD indicators
            )

            if df is None or len(df) < config.MIN_BARS_FOR_MACD:
                return None

            # 📊 NEW: Use MTF-enhanced detection if available
            # Get epic-specific MACD strategy with optimized parameters
            macd_strategy = self._get_macd_strategy_for_epic(epic)
            
            if hasattr(macd_strategy, 'detect_signal_with_mtf') and getattr(config, 'MACD_FILTER_CONFIG', {}).get('multi_timeframe_analysis', False):
                signal = macd_strategy.detect_signal_with_mtf(df, epic, spread_pips, timeframe)
                self.logger.info(f"🔄 [MTF MACD] Used MTF-enhanced detection for {epic}")
            else:
                # Fallback to standard detection (strategy will log its own optimization status)
                signal = macd_strategy.detect_signal(df, epic, spread_pips, timeframe)
            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                
                # 🧠 FIXED: Smart Money Integration with proper import
                try:
                    from core.smart_money_integration import add_smart_money_to_signal
                    self.logger.info(f"🧠 [SMART MONEY] Starting analysis for {epic}")
                    enhanced_signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
                    if enhanced_signal:
                        self.logger.info(f"✅ [SMART MONEY] Enhanced signal for {epic}")
                        signal = enhanced_signal
                except ImportError as e:
                    self.logger.warning(f"⚠️ [SMART MONEY] Import failed: {e}")
                except Exception as e:
                    self.logger.error(f"❌ [SMART MONEY] Analysis failed: {e}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD+EMA signals for {epic}: {e}")
            return None
        
    def detect_kama_signals(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect KAMA (Kaufman's Adaptive Moving Average) signals
        """
        # Use default timeframe if not specified
        timeframe = self._get_default_timeframe(timeframe)
        
        if not self.kama_strategy:
            self.logger.debug("KAMA strategy not available")
            return None
            
        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe, ema_strategy=self.ema_strategy)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Use KAMA strategy
            signal = self.kama_strategy.detect_signal(df, epic, spread_pips, timeframe)
            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                # Add KAMA-specific context
                signal = self._add_kama_context(signal, df)

                signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting KAMA signals for {epic}: {e}")
            return None
    
    def detect_bb_supertrend_signals(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect Bollinger Bands + Supertrend signals
        """
        if not self.bb_supertrend_strategy:
            return None
            
        try:
            # Get enhanced data with required indicators
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Use BB+Supertrend strategy
            signal = self.bb_supertrend_strategy.detect_signal(df, epic, spread_pips, timeframe)
            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)

                signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
            
            return signal
                
        except Exception as e:
            self.logger.error(f"Error detecting BB+Supertrend signals for {epic}: {e}")
            return None

    def detect_zero_lag_signals(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect Zero-Lag EMA signals with enhanced market context
        """
        if not hasattr(self, 'zero_lag_strategy') or not self.zero_lag_strategy:
            self.logger.debug("Zero-Lag strategy not available")
            return None
            
        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe, ema_strategy=self.ema_strategy)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Use Zero-Lag strategy
            signal = self.zero_lag_strategy.detect_signal(df, epic, spread_pips, timeframe)
            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                
                # Add smart money context if available
                if hasattr(self, '_add_smart_money_context'):
                    try:
                        from core.processing.smart_money import add_smart_money_to_signal
                        signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
                    except ImportError:
                        self.logger.debug("Smart money module not available for zero-lag signals")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting Zero-Lag signals for {epic}: {e}")
            return None


    def detect_momentum_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """Detect Advanced Momentum signals with minimal lag indicators"""

        # Check if advanced momentum strategy is enabled and available
        if not getattr(config, 'MOMENTUM_STRATEGY', False):
            return None

        if not hasattr(self, 'momentum_strategy') or self.momentum_strategy is None:
            self.logger.debug("Advanced Momentum strategy not available or not initialized")
            return None

        try:
            # Use default timeframe if not specified
            timeframe = self._get_default_timeframe(timeframe)

            # Get enhanced data with momentum strategy support
            df = self.data_fetcher.get_enhanced_data(
                epic, pair, timeframe=timeframe
            )

            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None

            # Use Advanced Momentum strategy
            signal = self.momentum_strategy.detect_signal(df, epic, spread_pips, timeframe)

            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Advanced Momentum strategy error for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_smc_signals(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect Smart Money Concepts (SMC) signals
        """
        # Use default timeframe if not specified
        timeframe = self._get_default_timeframe(timeframe)
        
        if not self.smc_strategy:
            return None
            
        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Use SMC strategy
            signal = self.smc_strategy.detect_signal(df, epic, spread_pips, timeframe)
            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting SMC signals for {epic}: {e}")
            return None

    def detect_smc_structure_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect SMC Pure Structure signals (price action only)

        Uses multi-timeframe analysis:
        - Entry timeframe: From config SMC_ENTRY_TIMEFRAME (default 15m)
        - HTF timeframe: From config SMC_HTF_TIMEFRAME (default 4H)
        """
        # Load timeframes from SMC structure config
        try:
            from configdata.strategies import config_smc_structure as smc_config
            entry_tf = getattr(smc_config, 'SMC_ENTRY_TIMEFRAME', '15m')
            htf_tf = getattr(smc_config, 'SMC_HTF_TIMEFRAME', '4h')
        except ImportError:
            # Fallback to defaults if config not available
            entry_tf = '15m'
            htf_tf = '4h'

        self.logger.debug(f"🔍 [SMC_STRUCTURE] Using entry_tf={entry_tf}, htf_tf={htf_tf}")

        try:
            # Initialize strategy if not already done
            if not hasattr(self, 'smc_structure_strategy') or self.smc_structure_strategy is None:
                # Use relative import since signal_detector is in forex_scanner.core
                from .strategies import create_smc_structure_strategy

                self.smc_structure_strategy = create_smc_structure_strategy(logger=self.logger)
                self.logger.info(f"✅ SMC Structure strategy initialized (entry_tf={entry_tf}, htf_tf={htf_tf})")

            # CRITICAL FIX: Reset cooldowns at start of new backtest to prevent stale cooldowns
            # from previous runs blocking signals (same fix as SMC_SIMPLE)
            is_backtest = hasattr(self.data_fetcher, 'current_backtest_time') and self.data_fetcher.current_backtest_time is not None
            if is_backtest and hasattr(self.smc_structure_strategy, 'reset_cooldowns'):
                # Track backtest session ID to detect new backtests
                # Use id(data_fetcher) as a unique identifier for this backtest session
                current_backtest_id = id(self.data_fetcher)
                if not hasattr(self, '_smc_structure_backtest_id') or self._smc_structure_backtest_id != current_backtest_id:
                    self._smc_structure_backtest_id = current_backtest_id
                    self.smc_structure_strategy.reset_cooldowns()

            # Get entry timeframe data (15m by default, configured via SMC_ENTRY_TIMEFRAME)
            df_entry = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=entry_tf,
                lookback_hours=200
            )

            if df_entry is None or len(df_entry) < 50:
                self.logger.debug(f"Insufficient {entry_tf} data for {epic} (got {len(df_entry) if df_entry is not None else 0} bars)")
                return None

            # Get HTF data (4H by default, configured via SMC_HTF_TIMEFRAME)
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=htf_tf,
                lookback_hours=400  # 100 bars * 4 hours
            )

            if df_4h is None or len(df_4h) < 20:
                self.logger.debug(f"Insufficient {htf_tf} data for {epic} (got {len(df_4h) if df_4h is not None else 0} bars)")
                return None

            # df_15m is the entry timeframe data when entry_tf='15m'
            # For backwards compatibility, pass it as df_15m to detect_signal
            df_15m = df_entry if entry_tf == '15m' else None

            # If entry_tf is not 15m but we still need 15m for BOS/CHoCH, fetch it separately
            if entry_tf != '15m' and df_15m is None:
                try:
                    df_15m = self.data_fetcher.get_enhanced_data(
                        epic=epic,
                        pair=pair,
                        timeframe='15m',
                        lookback_hours=100
                    )
                    if df_15m is not None:
                        self.logger.debug(f"🔍 [SMC_STRUCTURE] Got 15m data for BOS/CHoCH detection: {len(df_15m)} bars")
                except Exception as e:
                    self.logger.debug(f"⚠️ [SMC_STRUCTURE] Could not fetch 15m data (non-critical): {e}")
                    df_15m = None

            self.logger.debug(f"🔍 [SMC_STRUCTURE] Analyzing {epic}: entry_tf={entry_tf} ({len(df_entry)} bars), htf_tf={htf_tf} ({len(df_4h)} bars)")

            # Detect signal - pass df_entry as df_1h for backwards compatibility with strategy interface
            signal = self.smc_structure_strategy.detect_signal(
                df_1h=df_entry,
                df_4h=df_4h,
                epic=epic,
                pair=pair,
                df_15m=df_15m
            )

            if signal:
                # Update signal timeframe to reflect actual entry timeframe used
                signal['timeframe'] = entry_tf
                self.logger.info(f"✅ [SMC_STRUCTURE] Signal detected for {epic}: {signal['signal']} @ {signal['entry_price']:.5f} (tf={entry_tf})")
                # Add market context if available
                signal = self._add_market_context(signal, df_entry)

            return signal

        except Exception as e:
            self.logger.error(f"❌ [SMC_STRUCTURE] Error detecting signals for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_smc_simple_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect SMC Simple 3-tier signals (50 EMA based)

        Uses multi-timeframe analysis:
        - Bias: 4H 50 EMA for directional bias
        - Trigger: 1H swing break with body-close confirmation
        - Entry: 15m pullback to Fibonacci zone
        """
        # Load timeframes from SMC simple config
        try:
            from configdata.strategies import config_smc_simple as smc_simple_config
            htf_tf = getattr(smc_simple_config, 'HTF_TIMEFRAME', '4h')
            trigger_tf = getattr(smc_simple_config, 'TRIGGER_TIMEFRAME', '1h')
            entry_tf = getattr(smc_simple_config, 'ENTRY_TIMEFRAME', '15m')
        except ImportError:
            htf_tf = '4h'
            trigger_tf = '1h'
            entry_tf = '15m'

        self.logger.debug(f"🔍 [SMC_SIMPLE] Using htf_tf={htf_tf}, trigger_tf={trigger_tf}, entry_tf={entry_tf}")

        try:
            # Initialize strategy if not already done (lazy loading)
            if not hasattr(self, 'smc_simple_strategy') or self.smc_simple_strategy is None:
                from .strategies.smc_simple_strategy import create_smc_simple_strategy

                self.smc_simple_strategy = create_smc_simple_strategy(config, logger=self.logger)
                self.logger.info(f"✅ SMC Simple strategy initialized (htf={htf_tf}, trigger={trigger_tf}, entry={entry_tf})")

            # Check if data_fetcher is in backtest mode (needed for lookback calculations)
            is_backtest = hasattr(self.data_fetcher, 'current_backtest_time') and self.data_fetcher.current_backtest_time is not None

            # CRITICAL FIX: Reset cooldowns at start of new backtest to prevent stale cooldowns
            # from previous runs blocking signals
            if is_backtest and hasattr(self.smc_simple_strategy, 'reset_cooldowns'):
                # Track backtest session ID to detect new backtests
                # Use id(data_fetcher) as a unique identifier for this backtest session
                current_backtest_id = id(self.data_fetcher)
                if not hasattr(self, '_smc_simple_backtest_id') or self._smc_simple_backtest_id != current_backtest_id:
                    self._smc_simple_backtest_id = current_backtest_id
                    self.smc_simple_strategy.reset_cooldowns()

            # Get 4H data for EMA bias
            # BACKTEST FIX: Use longer lookback in backtest mode to cover full historical period
            # 4H = 240 minutes, 4000 hours = ~1000 bars (covers ~166 days of backtest)
            htf_lookback = 4000 if is_backtest else 400  # ~100 bars of 4H for live, ~1000 for backtest
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=htf_tf,
                lookback_hours=htf_lookback
            )

            if df_4h is None or len(df_4h) < 60:  # Need 50+ bars for EMA
                self.logger.debug(f"Insufficient {htf_tf} data for {epic} (got {len(df_4h) if df_4h is not None else 0} bars)")
                return None

            # Get trigger timeframe data for swing break detection (dynamic: can be 1H or 15m)

            # Calculate lookback based on timeframe (need ~100 bars)
            if trigger_tf == '15m':
                trigger_lookback = 2000 if is_backtest else 30  # 15m: 30h = 120 bars, backtest needs more
            else:  # 1h
                trigger_lookback = 2000 if is_backtest else 100  # 1H: 100 bars

            df_trigger = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=trigger_tf,
                lookback_hours=trigger_lookback
            )

            if df_trigger is None or len(df_trigger) < 30:
                self.logger.debug(f"Insufficient {trigger_tf} data for {epic} (got {len(df_trigger) if df_trigger is not None else 0} bars)")
                return None

            # Get entry timeframe data for pullback entry (dynamic: can be 15m or 5m)
            df_entry = None
            if entry_tf in ['15m', '5m']:
                # For backtest mode, we need data going back far enough to cover historical periods
                if entry_tf == '5m':
                    entry_lookback = 2000 if is_backtest else 25  # 5m: 25h = 300 bars
                else:  # 15m
                    entry_lookback = 2000 if is_backtest else 50  # 15m: 50h = 200 bars

                df_entry = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe=entry_tf,
                    lookback_hours=entry_lookback
                )

            # Log the data being passed (INFO level for debugging)
            df_entry_len = len(df_entry) if df_entry is not None else 0
            self.logger.info(f"🔍 [SMC_SIMPLE] Passing to strategy: 4H({len(df_4h)} bars), {trigger_tf}({len(df_trigger)} bars), {entry_tf}({df_entry_len} bars)")

            # Detect signal - pass trigger TF data as df_trigger and entry TF data as df_entry
            signal = self.smc_simple_strategy.detect_signal(
                df_trigger=df_trigger,  # 15m data for swing break (or 1h in old config)
                df_4h=df_4h,
                epic=epic,
                pair=pair,
                df_entry=df_entry  # 5m data for pullback entry (or 15m in old config)
            )

            if signal:
                self.logger.info(f"✅ [SMC_SIMPLE] Signal detected for {epic}: {signal['signal']} @ {signal['entry_price']:.5f}")
                # Add market context if available
                signal = self._add_market_context(signal, df_trigger)

            return signal

        except Exception as e:
            self.logger.error(f"❌ [SMC_SIMPLE] Error detecting signals for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_ichimoku_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect Ichimoku Cloud signals using TK crosses, cloud breakouts, and Chikou confirmation
        """
        # Use default timeframe if not specified
        timeframe = self._get_default_timeframe(timeframe)

        if not self.ichimoku_strategy:
            return None

        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe)

            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return None

            # Detect Ichimoku signal
            signal = self.ichimoku_strategy.detect_signal(df, epic, spread_pips, timeframe)

            if signal:
                # Apply market context
                signal = self._add_market_context(signal, df)

            return signal

        except Exception as e:
            self.logger.error(f"Error detecting Ichimoku signals for {epic}: {e}")
            return None

    def detect_mean_reversion_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect Mean Reversion signals using multi-oscillator confluence
        """
        if not hasattr(self, 'mean_reversion_strategy') or not self.mean_reversion_strategy:
            self.logger.debug("Mean Reversion strategy not available")
            return None

        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe, ema_strategy=self.ema_strategy)

            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                self.logger.debug(f"Insufficient data for mean reversion analysis: {len(df) if df is not None else 0} bars")
                return None

            # Use mean reversion strategy
            signal = self.mean_reversion_strategy.detect_signal(df, epic, spread_pips, timeframe)

            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)

            return signal

        except Exception as e:
            self.logger.error(f"Error in mean reversion signal detection for {epic}: {e}")
            return None

    def detect_ranging_market_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect Ranging Market signals using multi-oscillator confluence approach
        """
        if not hasattr(self, 'ranging_market_strategy') or not self.ranging_market_strategy:
            self.logger.debug("Ranging Market strategy not available")
            return None

        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe, ema_strategy=self.ema_strategy)

            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                self.logger.debug(f"Insufficient data for ranging market analysis: {len(df) if df is not None else 0} bars")
                return None

            # Use ranging market strategy
            signal = self.ranging_market_strategy.detect_signal(df, epic, spread_pips, timeframe)

            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df)
                signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)

            return signal

        except Exception as e:
            self.logger.error(f"Error in ranging market signal detection for {epic}: {e}")
            return None

    # detect_combined_signals method removed - combined strategy was disabled and unused
    
    def detect_signals_all_strategies(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> List[Dict]:
        """
        ✅ OPTIMIZED: Detect signals using ALL enabled strategies individually
        🔧 FIXED: Collects individual results and passes to combined strategy to prevent duplicates
        """
        all_signals = []
        
        try:
            self.logger.debug(f"🔍 Running ALL individual strategies for {epic}")
            
            # 🔧 NEW: Collect individual results for combined strategy
            individual_results = {}
            
            # ========== RUN INDIVIDUAL STRATEGIES SEPARATELY ==========
            
            # 1. EMA Strategy
            if getattr(config, 'SIMPLE_EMA_STRATEGY', True):
                try:
                    self.logger.debug(f"🔍 [EMA STRATEGY] Starting detection for {epic}")
                    if system_config.USE_BID_ADJUSTMENT:
                        ema_signal = self.detect_signals_bid_adjusted(epic, pair, spread_pips, timeframe)
                    else:
                        ema_signal = self.detect_signals_mid_prices(epic, pair, timeframe)
                    
                    # 🔧 NEW: Store result for combined strategy
                    individual_results['ema'] = ema_signal
                    
                    if ema_signal:
                        all_signals.append(ema_signal)
                        self.logger.info(f"✅ [EMA STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [EMA STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [EMA STRATEGY] Error for {epic}: {e}")
                    individual_results['ema'] = None
            
            # 2. MACD Strategy
            macd_enabled = getattr(config, 'MACD_STRATEGY', False)
            if macd_enabled:
                try:
                    self.logger.debug(f"🔍 [MACD STRATEGY] Starting detection for {epic}")
                    macd_signal = self.detect_macd_ema_signals(epic, pair, spread_pips, timeframe)
                    
                    # 🔧 NEW: Store result for combined strategy
                    individual_results['macd'] = macd_signal
                    
                    if macd_signal:
                        all_signals.append(macd_signal)
                        self.logger.info(f"✅ [MACD STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [MACD STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [MACD STRATEGY] Error for {epic}: {e}")
                    individual_results['macd'] = None
            
            # 3. KAMA Strategy
            if getattr(config, 'KAMA_STRATEGY', True) and self.kama_strategy:
                try:
                    self.logger.debug(f"🔍 [KAMA STRATEGY] Starting detection for {epic}")
                    kama_signal = self.detect_kama_signals(epic, pair, spread_pips, timeframe)
                    
                    # 🔧 NEW: Store result for combined strategy
                    individual_results['kama'] = kama_signal
                    
                    if kama_signal:
                        all_signals.append(kama_signal)
                        self.logger.info(f"✅ [KAMA STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [KAMA STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [KAMA STRATEGY] Error for {epic}: {e}")
                    individual_results['kama'] = None
            
            # 4. Zero Lag Strategy
            if getattr(config, 'ZERO_LAG_STRATEGY', True) and self.zero_lag_strategy:
                try:
                    self.logger.debug(f"🔍 [ZERO LAG] Starting detection for {epic}")
                    zero_lag_signal = self.detect_zero_lag_signals(epic, pair, spread_pips, timeframe)
                    
                    # 🔧 NEW: Store result for combined strategy
                    individual_results['zero_lag'] = zero_lag_signal
                    
                    if zero_lag_signal:
                        all_signals.append(zero_lag_signal)
                        self.logger.info(f"✅ [ZERO LAG] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [ZERO LAG] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [ZERO LAG] Error for {epic}: {e}")
                    individual_results['zero_lag'] = None
            
            # 5. Bollinger Bands + SuperTrend Strategy
            if getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', True) and self.bb_supertrend_strategy:
                try:
                    self.logger.debug(f"🔍 [BB+SUPERTREND] Starting detection for {epic}")
                    bb_signal = self.detect_bb_supertrend_signals(epic, pair, spread_pips, timeframe)
                    
                    # 🔧 NEW: Store result for combined strategy
                    individual_results['bb_supertrend'] = bb_signal
                    
                    if bb_signal:
                        all_signals.append(bb_signal)
                        self.logger.info(f"✅ [BB+SUPERTREND] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [BB+SUPERTREND] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [BB+SUPERTREND] Error for {epic}: {e}")
                    individual_results['bb_supertrend'] = None
            
            # 6. Scalping Strategy (if enabled)
            if getattr(config, 'SCALPING_STRATEGY_ENABLED', False) and self.scalping_strategy is not None:
                try:
                    self.logger.info(f"🔍 [SCALPING] Starting detection for {epic}")
                    # 🔥 ALWAYS use 5m for Linda Raschke MACD 3-10-16 scalping, regardless of scanner timeframe
                    scalping_timeframe = '5m'
                    scalping_signal = self.detect_scalping_signals(epic, pair, spread_pips, scalping_timeframe)
                    
                    # 🔧 NEW: Store result for combined strategy
                    individual_results['scalping'] = scalping_signal
                    
                    if scalping_signal:
                        all_signals.append(scalping_signal)
                        self.logger.info(f"✅ [SCALPING] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [SCALPING] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [SCALPING] Error for {epic}: {e}")
                    individual_results['scalping'] = None
            

            # 8. Advanced Momentum Strategy (if enabled)
            if (getattr(config, 'MOMENTUM_STRATEGY', False) and
                hasattr(self, 'momentum_strategy') and self.momentum_strategy is not None):
                try:
                    self.logger.debug(f"🔍 [MOMENTUM STRATEGY] Starting detection for {epic}")
                    momentum_signal = self.detect_momentum_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['momentum'] = momentum_signal

                    if momentum_signal:
                        all_signals.append(momentum_signal)
                        self.logger.info(f"✅ [MOMENTUM STRATEGY] Signal detected for {epic} - {momentum_signal.get('signal_type')} with {momentum_signal.get('confidence_score', 0):.1%} confidence")
                    else:
                        self.logger.debug(f"📊 [MOMENTUM STRATEGY] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [MOMENTUM STRATEGY] Error for {epic}: {e}")
                    individual_results['momentum'] = None

            # 9. SMC Strategy (if enabled)
            if (getattr(config, 'SMC_STRATEGY', False) and
                hasattr(self, 'smc_strategy') and self.smc_strategy is not None):
                try:
                    self.logger.debug(f"🔍 [SMC STRATEGY] Starting detection for {epic}")
                    smc_signal = self.detect_smc_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['smc'] = smc_signal

                    if smc_signal:
                        all_signals.append(smc_signal)
                        self.logger.info(f"✅ [SMC STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [SMC STRATEGY] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [SMC STRATEGY] Error for {epic}: {e}")
                    individual_results['smc'] = None

            # 9. Ichimoku Strategy (if enabled)
            if (getattr(config, 'ICHIMOKU_CLOUD_STRATEGY', False) and
                hasattr(self, 'ichimoku_strategy') and self.ichimoku_strategy is not None):
                try:
                    self.logger.debug(f"🔍 [ICHIMOKU STRATEGY] Starting detection for {epic}")
                    ichimoku_signal = self.detect_ichimoku_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['ichimoku'] = ichimoku_signal

                    if ichimoku_signal:
                        all_signals.append(ichimoku_signal)
                        self.logger.info(f"✅ [ICHIMOKU STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [ICHIMOKU STRATEGY] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [ICHIMOKU STRATEGY] Error for {epic}: {e}")
                    individual_results['ichimoku'] = None

            # 10. Mean Reversion Strategy (if enabled)
            if (getattr(config, 'MEAN_REVERSION_STRATEGY', False) and
                hasattr(self, 'mean_reversion_strategy') and self.mean_reversion_strategy is not None):
                try:
                    self.logger.debug(f"🔍 [MEAN REVERSION STRATEGY] Starting detection for {epic}")
                    mean_reversion_signal = self.detect_mean_reversion_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['mean_reversion'] = mean_reversion_signal

                    if mean_reversion_signal:
                        all_signals.append(mean_reversion_signal)
                        self.logger.info(f"✅ [MEAN REVERSION STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [MEAN REVERSION STRATEGY] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [MEAN REVERSION STRATEGY] Error for {epic}: {e}")
                    individual_results['mean_reversion'] = None

            # 11. Ranging Market Strategy (if enabled)
            if (getattr(config, 'RANGING_MARKET_STRATEGY', False) and
                hasattr(self, 'ranging_market_strategy') and self.ranging_market_strategy is not None):
                try:
                    self.logger.debug(f"🔍 [RANGING MARKET STRATEGY] Starting detection for {epic}")
                    ranging_market_signal = self.detect_ranging_market_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['ranging_market'] = ranging_market_signal

                    if ranging_market_signal:
                        all_signals.append(ranging_market_signal)
                        self.logger.info(f"✅ [RANGING MARKET STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [RANGING MARKET STRATEGY] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [RANGING MARKET STRATEGY] Error for {epic}: {e}")
                    individual_results['ranging_market'] = None

            # 12. SMC Structure Strategy v2.4.0 (if enabled)
            if getattr(config, 'SMC_STRUCTURE_STRATEGY', False) and self.smc_structure_enabled:
                try:
                    self.logger.debug(f"🔍 [SMC STRUCTURE] Starting detection for {epic}")
                    smc_structure_signal = self.detect_smc_structure_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['smc_structure'] = smc_structure_signal

                    if smc_structure_signal:
                        all_signals.append(smc_structure_signal)
                        self.logger.info(f"✅ [SMC STRUCTURE] Signal detected for {epic}: {smc_structure_signal.get('signal')} @ {smc_structure_signal.get('entry_price', 0):.5f}")
                    else:
                        self.logger.debug(f"📊 [SMC STRUCTURE] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [SMC STRUCTURE] Error for {epic}: {e}")
                    individual_results['smc_structure'] = None

            # 13. SMC Simple Strategy v1.5.3 (if enabled) - 3-Tier EMA approach
            if getattr(system_config, 'SMC_SIMPLE_STRATEGY', False) and self.smc_simple_enabled:
                try:
                    self.logger.debug(f"🔍 [SMC SIMPLE] Starting detection for {epic}")
                    smc_simple_signal = self.detect_smc_simple_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['smc_simple'] = smc_simple_signal

                    if smc_simple_signal:
                        all_signals.append(smc_simple_signal)
                        self.logger.info(f"✅ [SMC SIMPLE] Signal detected for {epic}: {smc_simple_signal.get('signal')} @ {smc_simple_signal.get('entry_price', 0):.5f}")
                    else:
                        self.logger.debug(f"📊 [SMC SIMPLE] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [SMC SIMPLE] Error for {epic}: {e}")
                    individual_results['smc_simple'] = None

            # 14. EMA Double Confirmation Strategy (if enabled) - 21/50 crossover with 2-confirm
            if getattr(system_config, 'EMA_DOUBLE_CONFIRMATION_STRATEGY', False) and self.ema_double_confirmation_strategy:
                try:
                    self.logger.debug(f"🔍 [EMA_DOUBLE] Starting detection for {epic}")
                    ema_double_signal = self.detect_ema_double_confirmation_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['ema_double_confirmation'] = ema_double_signal

                    if ema_double_signal:
                        all_signals.append(ema_double_signal)
                        self.logger.info(f"✅ [EMA_DOUBLE] Signal detected for {epic}: {ema_double_signal.get('signal_type')} @ {ema_double_signal.get('price', 0):.5f}")
                    else:
                        self.logger.debug(f"📊 [EMA_DOUBLE] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [EMA_DOUBLE] Error for {epic}: {e}")
                    individual_results['ema_double_confirmation'] = None

            # 15. ICT Silver Bullet Strategy (if enabled) - Time-based SMC
            if getattr(system_config, 'SILVER_BULLET_STRATEGY', False) and self.silver_bullet_strategy:
                try:
                    self.logger.debug(f"🔍 [SILVER_BULLET] Starting detection for {epic}")
                    silver_bullet_signal = self.detect_silver_bullet_signals(epic, pair, spread_pips, timeframe)

                    # Store result for combined strategy
                    individual_results['silver_bullet'] = silver_bullet_signal

                    if silver_bullet_signal:
                        all_signals.append(silver_bullet_signal)
                        self.logger.info(f"✅ [SILVER_BULLET] Signal detected for {epic}: {silver_bullet_signal.get('signal')} @ {silver_bullet_signal.get('entry_price', 0):.5f}")
                    else:
                        self.logger.debug(f"📊 [SILVER_BULLET] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [SILVER_BULLET] Error for {epic}: {e}")
                    individual_results['silver_bullet'] = None

            # ========== COMBINED STRATEGY REMOVED ==========
            # Combined strategy was disabled and unused, removed to clean up codebase
            
            # ========== ADD SMART MONEY ANALYSIS TO ALL SIGNALS ==========
            
            # Add smart money analysis to all signals (if enabled)
            if all_signals:
                try:
                    all_signals = add_smart_money_to_signals(all_signals, self.data_fetcher, self.db_manager)
                    self.logger.debug(f"✅ Smart money analysis added to {len(all_signals)} signals")
                except Exception as e:
                    self.logger.warning(f"⚠️ Smart money analysis failed: {e}")
            
            # ========== RESULTS LOGGING ==========
            
            if all_signals:
                strategy_names = [s.get('strategy', 'unknown') for s in all_signals]
                self.logger.info(f"🎯 {epic}: {len(all_signals)} signals from strategies: {', '.join(strategy_names)}")
                
                # Log individual signal details
                for i, signal in enumerate(all_signals, 1):
                    strategy = signal.get('strategy', 'unknown')
                    signal_type = signal.get('signal_type', 'unknown')
                    confidence = signal.get('confidence_score', 0)
                    self.logger.info(f"   📈 Signal {i}: {strategy} - {signal_type} ({confidence:.1%})")
                    
                # 🔧 NEW: Log optimization status
                optimized_signals = [s for s in all_signals if s.get('optimized_execution', False)]
                if optimized_signals:
                    self.logger.debug(f"🚀 {len(optimized_signals)} signals used optimized execution (no duplicates)")
            else:
                self.logger.debug(f"📊 {epic}: No signals from any strategy")
            
            return all_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error in detect_signals_all_strategies for {epic}: {e}")
            return all_signals  # Return any signals collected before the error

    def detect_signals(self, epic: str, pair: str, spread_pips: float = 1.5, timeframe: str = None) -> List[Dict]:
        """Enhanced signal detection with large candle filtering and zero-lag strategy support"""
        # Use default timeframe if not specified
        timeframe = self._get_default_timeframe(timeframe)
        
        all_signals = []
        individual_results = {}
        
        try:
            self.logger.debug(f"🔍 Starting signal detection for {epic} ({timeframe})")
            # 🚨 ADD THIS DEBUG BLOCK
            self.logger.info(f"🔍 [MACD DEBUG] Starting signal detection for {epic}")
            self.logger.info(f"   DataFrame shape: {df.shape}")
            self.logger.info(f"   Columns available: {list(df.columns)}")
            
            # Check for required MACD columns
            required_columns = ['macd_line', 'macd_signal', 'macd_histogram', 'close', 'ema_200']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"❌ [MACD DEBUG] Missing required columns: {missing_columns}")
                return None
            
            # Check data quality
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            self.logger.info(f"   Latest MACD data:")
            self.logger.info(f"     macd_histogram: {latest.get('macd_histogram', 'N/A')}")
            self.logger.info(f"     macd_line: {latest.get('macd_line', 'N/A')}")
            self.logger.info(f"     macd_signal: {latest.get('macd_signal', 'N/A')}")
            self.logger.info(f"     close: {latest.get('close', 'N/A')}")
            self.logger.info(f"     ema_200: {latest.get('ema_200', 'N/A')}")
            
            # 🚨 END DEBUG BLOCK
            
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe)
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                self.logger.warning(f"❌ Insufficient data for {epic}: {len(df) if df is not None else 0} bars")
                return []
            
            # 1. EMA Strategy
            if getattr(config, 'SIMPLE_EMA_STRATEGY', True):
                try:
                    self.logger.debug(f"🔍 [EMA STRATEGY] Starting detection for {epic}")
                    if system_config.USE_BID_ADJUSTMENT:
                        ema_signal = self.detect_signals_bid_adjusted(epic, pair, spread_pips, timeframe)
                    else:
                        ema_signal = self.detect_signals_mid_prices(epic, pair, timeframe)
                    
                    individual_results['ema'] = ema_signal
                    
                    if ema_signal:
                        # 🆕 NEW: Apply large candle filter
                        should_block, block_reason = self._apply_large_candle_filter(
                            df, epic, ema_signal, timeframe
                        )
                        
                        if should_block:
                            self.logger.warning(f"🚫 [EMA STRATEGY] Signal blocked for {epic}: {block_reason}")
                            # Add rejection metadata to signal
                            ema_signal['filter_rejected'] = True
                            ema_signal['rejection_reason'] = f"Large candle filter: {block_reason}"
                            ema_signal['status'] = 'REJECTED'
                        else:
                            all_signals.append(ema_signal)
                            self.logger.info(f"✅ [EMA STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [EMA STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [EMA STRATEGY] Error for {epic}: {e}")
                    individual_results['ema'] = None
            
            # 2. MACD Strategy
            macd_enabled = getattr(config, 'MACD_STRATEGY', False)
            if macd_enabled:
                try:
                    self.logger.debug(f"🔍 [MACD STRATEGY] Starting detection for {epic}")
                    macd_signal = self.detect_macd_ema_signals(epic, pair, spread_pips, timeframe)
                    
                    individual_results['macd'] = macd_signal
                    
                    if macd_signal:
                        # 🆕 NEW: Apply large candle filter
                        should_block, block_reason = self._apply_large_candle_filter(
                            df, epic, macd_signal, timeframe
                        )
                        
                        if should_block:
                            self.logger.warning(f"🚫 [MACD STRATEGY] Signal blocked for {epic}: {block_reason}")
                            macd_signal['filter_rejected'] = True
                            macd_signal['rejection_reason'] = f"Large candle filter: {block_reason}"
                            macd_signal['status'] = 'REJECTED'
                        else:
                            all_signals.append(macd_signal)
                            self.logger.info(f"✅ [MACD STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [MACD STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [MACD STRATEGY] Error for {epic}: {e}")
                    individual_results['macd'] = None
            
            # 3. KAMA Strategy (if enabled)
            if getattr(config, 'KAMA_STRATEGY', False) and hasattr(self, 'kama_strategy') and self.kama_strategy:
                try:
                    self.logger.debug(f"🔍 [KAMA STRATEGY] Starting detection for {epic}")
                    kama_signal = self.detect_kama_signals(epic, pair, spread_pips, timeframe)
                    
                    individual_results['kama'] = kama_signal
                    
                    if kama_signal:
                        # 🆕 NEW: Apply large candle filter
                        should_block, block_reason = self._apply_large_candle_filter(
                            df, epic, kama_signal, timeframe
                        )
                        
                        if should_block:
                            self.logger.warning(f"🚫 [KAMA STRATEGY] Signal blocked for {epic}: {block_reason}")
                            kama_signal['filter_rejected'] = True
                            kama_signal['rejection_reason'] = f"Large candle filter: {block_reason}"
                            kama_signal['status'] = 'REJECTED'
                        else:
                            all_signals.append(kama_signal)
                            self.logger.info(f"✅ [KAMA STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [KAMA STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [KAMA STRATEGY] Error for {epic}: {e}")
                    individual_results['kama'] = None
            
            # 4. Zero-Lag Strategy (if enabled) - NEW ADDITION
            if getattr(config, 'ZERO_LAG_STRATEGY', False) and hasattr(self, 'zero_lag_strategy') and self.zero_lag_strategy:
                try:
                    self.logger.debug(f"🔍 [ZERO-LAG STRATEGY] Starting detection for {epic}")
                    zero_lag_signal = self.detect_zero_lag_signals(epic, pair, spread_pips, timeframe)
                    
                    individual_results['zero_lag'] = zero_lag_signal
                    
                    if zero_lag_signal:
                        # 🆕 NEW: Apply large candle filter
                        should_block, block_reason = self._apply_large_candle_filter(
                            df, epic, zero_lag_signal, timeframe
                        )
                        
                        if should_block:
                            self.logger.warning(f"🚫 [ZERO-LAG STRATEGY] Signal blocked for {epic}: {block_reason}")
                            zero_lag_signal['filter_rejected'] = True
                            zero_lag_signal['rejection_reason'] = f"Large candle filter: {block_reason}"
                            zero_lag_signal['status'] = 'REJECTED'
                        else:
                            all_signals.append(zero_lag_signal)
                            self.logger.info(f"✅ [ZERO-LAG STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [ZERO-LAG STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [ZERO-LAG STRATEGY] Error for {epic}: {e}")
                    individual_results['zero_lag'] = None

            # 5. SMC Strategy (if enabled)
            if (getattr(config, 'SMC_STRATEGY', False) and
                hasattr(self, 'smc_strategy') and self.smc_strategy is not None):
                try:
                    self.logger.debug(f"🔍 [SMC STRATEGY] Starting detection for {epic}")
                    smc_signal = self.detect_smc_signals(epic, pair, spread_pips, timeframe)
                    
                    individual_results['smc'] = smc_signal
                    
                    if smc_signal:
                        # Apply large candle filter
                        should_block, block_reason = self._apply_large_candle_filter(
                            df, epic, smc_signal, timeframe
                        )
                        
                        if should_block:
                            self.logger.warning(f"🚫 [SMC STRATEGY] Signal blocked for {epic}: {block_reason}")
                            smc_signal['filter_rejected'] = True
                            smc_signal['rejection_reason'] = f"Large candle filter: {block_reason}"
                            smc_signal['status'] = 'REJECTED'
                        else:
                            all_signals.append(smc_signal)
                            self.logger.info(f"✅ [SMC STRATEGY] Signal detected for {epic}")
                    else:
                        self.logger.debug(f"📊 [SMC STRATEGY] No signal for {epic}")
                        
                except Exception as e:
                    self.logger.error(f"❌ [SMC STRATEGY] Error for {epic}: {e}")
                    individual_results['smc'] = None
            
            # Combined Strategy removed - was disabled and unused
            
            # Log filter statistics periodically
            if hasattr(self, 'large_candle_filter') and self.large_candle_filter.filter_stats['total_signals_checked'] % 50 == 0:
                filter_stats = self.large_candle_filter.get_filter_statistics()
                self.logger.info(f"📊 Large Candle Filter Stats: {filter_stats['filter_rate']}")
            
            return all_signals
            
        except Exception as e:
            self.logger.error(f"❌ Signal detection failed for {epic}: {e}")
            return []

    def _apply_large_candle_filter(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        signal: Dict, 
        timeframe: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply large candle filter to a signal
        
        Returns:
            Tuple of (should_block: bool, reason: str or None)
        """
        try:
            # Check if filter is enabled
            if not getattr(config, 'ENABLE_LARGE_CANDLE_FILTER', True):
                return False, None
            
            signal_type = signal.get('signal_type', 'UNKNOWN')
            
            # Apply the filter
            should_block, reason = self.large_candle_filter.should_block_entry(
                df, epic, signal_type, timeframe
            )
            
            if should_block:
                # Add filter metadata to signal for debugging
                signal['large_candle_filter'] = {
                    'blocked': True,
                    'reason': reason,
                    'filter_timestamp': datetime.now().isoformat(),
                    'filter_config': {
                        'atr_multiplier': self.large_candle_filter.large_candle_multiplier,
                        'movement_threshold_pips': self.large_candle_filter.excessive_movement_threshold,
                        'lookback_periods': self.large_candle_filter.movement_lookback_periods
                    }
                }
                
                self.logger.info(f"🚫 Large candle filter blocked {signal_type} signal for {epic}: {reason}")
            else:
                # Add pass-through metadata
                signal['large_candle_filter'] = {
                    'blocked': False,
                    'reason': 'Signal passed large candle filter checks',
                    'filter_timestamp': datetime.now().isoformat()
                }
            
            return should_block, reason
            
        except Exception as e:
            self.logger.error(f"❌ Large candle filter application failed for {epic}: {e}")
            # In case of error, don't block the signal
            return False, f"Filter error: {str(e)}"
    
    def get_large_candle_filter_stats(self) -> Dict:
        """Get large candle filter statistics for monitoring"""
        return self.large_candle_filter.get_filter_statistics()
    
    def reset_large_candle_filter_stats(self):
        """Reset large candle filter statistics"""
        self.large_candle_filter.reset_statistics()
        self.logger.info("📊 Large candle filter statistics reset")

    def detect_signals_all_epics(self, intelligence_mode: str = 'live_only', min_confidence: float = 0.6) -> List[Dict]:
        """
        Detect signals for all configured epics
        
        Args:
            intelligence_mode: Intelligence filtering mode
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of all detected signals across all epics
        """
        try:
            # Get epic list from config
            epic_list = getattr(config, 'EPIC_LIST', [
                'CS.D.EURUSD.CEEM.IP',
                'CS.D.GBPUSD.MINI.IP',
                'CS.D.USDJPY.MINI.IP',
                'CS.D.AUDUSD.MINI.IP',
                'CS.D.USDCAD.MINI.IP',
                'CS.D.NZDUSD.MINI.IP',
                'CS.D.USDCHF.MINI.IP'
            ])
            
            all_signals = []
            
            for epic in epic_list:
                try:
                    signals = self.detect_signals_single_epic(epic, intelligence_mode, min_confidence)
                    if signals:
                        all_signals.extend(signals)
                        self.logger.debug(f"📈 {epic}: {len(signals)} signals detected")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error detecting signals for {epic}: {e}")
            
            self.logger.info(f"📊 Total signals detected across all epics: {len(all_signals)}")
            return all_signals
            
        except Exception as e:
            self.logger.error(f"Error detecting signals for all epics: {e}")
            return []

    def detect_signals_single_epic(self, epic: str, intelligence_mode: str = 'live_only', min_confidence: float = 0.6) -> List[Dict]:
        """
        Detect signals for a single epic
        
        Args:
            epic: Currency pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
            intelligence_mode: Intelligence filtering mode
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected signals for this epic
        """
        try:
            # Get pair info
            pair_info = getattr(config, 'PAIR_INFO', {}).get(epic, {'pair': 'EURUSD', 'pip_multiplier': 10000})
            pair_name = pair_info['pair']
            
            # Use the existing detection methods
            use_bid_adjustment = getattr(config, 'USE_BID_ADJUSTMENT', False)
            spread_pips = getattr(config, 'SPREAD_PIPS', 1.5)
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '5m')
            
            if use_bid_adjustment:
                signal = self.detect_signals_bid_adjusted(epic, pair_name, spread_pips, timeframe)
            else:
                signal = self.detect_signals_mid_prices(epic, pair_name, timeframe)
            
            # Return as list for consistency
            if signal:
                return [signal] if isinstance(signal, dict) else signal
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error detecting signals for {epic}: {e}")
            return []

    # Alternative method that takes a list of epics (if trading_orchestrator expects this)
    def detect_signals_batch(self, epic_list: List[str], intelligence_mode: str = 'live_only', min_confidence: float = 0.6) -> List[Dict]:
        """
        Detect signals for multiple epics
        
        Args:
            epic_list: List of epics to scan
            intelligence_mode: Intelligence filtering mode
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of all detected signals across all epics
        """
        all_signals = []
        
        for epic in epic_list:
            try:
                signals = self.detect_signals(epic, intelligence_mode, min_confidence)
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                self.logger.error(f"Error detecting signals for {epic}: {e}")
        
        return all_signals
    def _add_market_context(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """ENHANCED: Add comprehensive market context including complete technical indicators"""
        
        # First, add complete technical indicators
        signal = self._add_complete_technical_indicators(signal, df)
        
        # Then add existing market context
        if df is None or df.empty:
            return signal
        
        try:
            latest = df.iloc[-1]
            
            # Add available market context (existing functionality)
            context_fields = [
                'volume_ratio_20', 'distance_to_support_pips', 'distance_to_resistance_pips',
                'trend_alignment', 'consolidation_range_pips', 'bars_since_breakout',
                'rejection_wicks_count', 'consecutive_green_candles', 'consecutive_red_candles'
            ]
            
            for field in context_fields:
                if field in latest.index:
                    signal[field] = latest[field]
            
            # Add volume confirmation flag
            if 'volume' in signal and 'volume_sma_20' in latest.index:
                signal['volume_confirmation'] = signal['volume'] > latest['volume_sma_20'] * 1.2
            
            # Add recent price action summary (last 5 bars for context)
            if len(df) >= 5:
                recent_data = df.tail(5)
                signal['recent_price_action'] = {
                    'bars_count': len(recent_data),
                    'high_range': float(recent_data['high'].max()),
                    'low_range': float(recent_data['low'].min()),
                    'avg_volume': float(recent_data.get('ltv', recent_data.get('volume', pd.Series([0]))).mean()),
                    'price_trend': 'bullish' if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else 'bearish'
                }
            
            self.logger.debug(f"✅ Enhanced signal with complete market context for {signal.get('epic')}")
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error adding enhanced market context: {e}")
            return signal

    def _enhance_signal_with_metadata_fixed(self, signal: Dict, df: pd.DataFrame, epic: str) -> Dict:
        """
        FIXED: Enhanced signal with proper timestamp handling to prevent stale data warnings
        """
        try:
            # Use the latest candle timestamp from the DataFrame as the authoritative source
            if len(df) > 0:
                latest_candle = df.iloc[-1]
                latest_timestamp = df.index[-1]
                
                # Ensure we have a proper datetime for market_timestamp
                if hasattr(latest_timestamp, 'to_pydatetime'):
                    market_timestamp = latest_timestamp.to_pydatetime()
                elif hasattr(latest_timestamp, 'timestamp'):
                    market_timestamp = latest_timestamp
                else:
                    market_timestamp = pd.to_datetime(latest_timestamp)
                
                # Override any existing stale market_timestamp
                signal['market_timestamp'] = market_timestamp
                
                # Also set a clean timestamp for general use
                signal['timestamp'] = market_timestamp
                
                # Set data source for tracking
                signal['data_source'] = 'live_candles'
                signal['data_quality'] = 'fresh'
                
                self.logger.debug(f"✅ Enhanced signal with fresh timestamp: {market_timestamp}")
                
            else:
                # Fallback if no data available
                current_time = datetime.now()
                signal['market_timestamp'] = current_time
                signal['timestamp'] = current_time
                signal['data_source'] = 'fallback'
                signal['data_quality'] = 'fallback'
                
                self.logger.warning(f"⚠️ Used fallback timestamp for {epic}: {current_time}")
            
            # Add additional metadata for validation
            signal['timestamp_fix_applied'] = True
            signal['enhancement_version'] = 'v2_stale_fix'
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error enhancing signal metadata: {e}")
            # Ensure we have some timestamp even if enhancement fails
            if 'market_timestamp' not in signal or str(signal.get('market_timestamp', '')).startswith('1970'):
                signal['market_timestamp'] = datetime.now()
                signal['timestamp'] = datetime.now()
                signal['data_quality'] = 'emergency_fix'
                self.logger.warning(f"🚨 Applied emergency timestamp fix for {epic}")
            
            return signal

    def _add_kama_context(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Add KAMA-specific context to signal"""
        if len(df) == 0:
            return signal
            
        latest = df.iloc[-1]
        
        # Add KAMA-specific fields
        kama_fields = [
            'kama_10', 'kama_10_er', 'kama_10_trend', 'kama_10_signal',
            'kama_14', 'kama_14_er', 'kama_14_trend', 'kama_14_signal'
        ]
        
        for field in kama_fields:
            if field in latest:
                signal[f'context_{field}'] = latest[field]
        
        # Calculate market regime based on efficiency ratio
        if 'efficiency_ratio' in signal:
            er = signal['efficiency_ratio']
            if er > 0.7:
                signal['market_regime'] = 'trending_strong'
            elif er > 0.4:
                signal['market_regime'] = 'trending_moderate'
            elif er > 0.2:
                signal['market_regime'] = 'ranging_moderate'
            else:
                signal['market_regime'] = 'ranging_strong'
        
        return signal
    
    def _convert_timeframe_to_numeric(self, timeframe):
        """
        Convert timeframe string to numeric value - EMERGENCY FIX
        """
        if isinstance(timeframe, (int, float)):
            return timeframe
        
        if isinstance(timeframe, str):
            # Handle common timeframe formats
            if 'm' in timeframe:
                return int(timeframe.replace('m', ''))
            elif 'h' in timeframe:
                return int(timeframe.replace('h', '')) * 60
            elif 'd' in timeframe:
                return int(timeframe.replace('d', '')) * 1440
            else:
                # Try to convert directly
                try:
                    return float(timeframe)
                except ValueError:
                    self.logger.warning(f"⚠️ Could not convert timeframe {timeframe}, using 15")
                    return 15
        
        return 15  # Default fallback

    @staticmethod 
    def is_market_closed_timestamp(market_timestamp) -> bool:
        """Check if market timestamp indicates market is closed"""
        try:
            if market_timestamp is None:
                return True
            
            # Convert to string for checking
            if hasattr(market_timestamp, 'strftime'):
                timestamp_str = market_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(market_timestamp)
            
            # Check if timestamp starts with 1970
            return timestamp_str.startswith('1970')
            
        except Exception:
            return True

    # Backtesting methods (delegate to BacktestEngine)
    def backtest_signals(
        self,
        epic_list: List[str],
        lookback_days: int = 30,
        use_bid_adjustment: bool = True,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> List[Dict]:
        """Backtest signals on historical data"""
        return self.backtest_engine.backtest_signals(
            epic_list, lookback_days, use_bid_adjustment, spread_pips, timeframe
        )
    
    # Performance analysis methods (delegate to PerformanceAnalyzer)
    def analyze_performance(self, signals: List[Dict]) -> Dict:
        """Analyze performance of historical signals"""
        return self.performance_analyzer.analyze_performance(signals)
    
    # Signal analysis methods (delegate to SignalAnalyzer)
    def display_signal_list(self, signals: List[Dict], timezone_manager=None, max_signals: int = 50):
        """Display detailed list of signals with timestamps"""
        return self.signal_analyzer.display_signal_list(signals, timezone_manager, max_signals)
    
    def display_signal_summary_by_pair(self, signals: List[Dict]):
        """Display signal summary grouped by currency pair"""
        return self.signal_analyzer.display_signal_summary_by_pair(signals)
    
    # Debug methods
    def debug_signal_detection(self, epic: str, pair: str, spread_pips: float = 1.5) -> Dict:
        """Enhanced debug signal detection - shows why signals are accepted/rejected"""
        try:
            # Get enhanced data
            df_5m = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=self._get_default_timeframe(),ema_strategy=self.ema_strategy)

            if df_5m is None or len(df_5m) < config.MIN_BARS_FOR_SIGNAL:
                return {'error': 'Insufficient data', 'bars': len(df_5m) if df_5m is not None else 0}

            # Adjust BID prices to MID prices if needed
            if config.USE_BID_ADJUSTMENT:
                df_mid = self.price_adjuster.adjust_bid_to_mid_prices(df_5m, spread_pips)
            else:
                df_mid = df_5m.copy()

            # Get latest and previous values for analysis
            if len(df_mid) < 2:
                return {'error': 'Need at least 2 bars for signal detection'}

            latest = df_mid.iloc[-1]
            previous = df_mid.iloc[-2]

            # Basic price and EMA data
            current_price = latest['close']
            prev_price = previous['close']
            
            # Check if we have EMA indicators
            ema_cols = ['ema_9', 'ema_21', 'ema_200']
            if not all(col in latest for col in ema_cols):
                return {'error': f'Missing EMA indicators. Available columns: {latest.index.tolist()}'}

            ema_9_current = latest['ema_9']
            ema_21_current = latest['ema_21']
            ema_200_current = latest['ema_200']
            ema_9_prev = previous['ema_9']

            # Test signal detection with current strategy
            signal = self.ema_strategy.detect_signal_auto(df_mid, epic, spread_pips, '5m')

            debug_info = {
                'epic': epic,
                'timestamp': latest['start_time'],
                'current_price': current_price,
                'prev_price': prev_price,
                'ema_9_current': ema_9_current,
                'ema_21_current': ema_21_current,
                'ema_200_current': ema_200_current,
                'ema_9_prev': ema_9_prev,
                'signal_detected': signal is not None,
                'signal_type': signal.get('signal_type') if signal else None,
                'confidence_score': signal.get('confidence_score') if signal else 0,
                'strategy_used': signal.get('strategy') if signal else 'None',
                'spread_adjustment_pips': spread_pips,
                'original_close_price': df_5m.iloc[-1]['close'],
                'adjusted_close_price': current_price,
                'min_confidence_threshold': system_config.MIN_CONFIDENCE,
                'above_threshold': signal.get('confidence_score', 0) >= system_config.MIN_CONFIDENCE if signal else False
            }

            return debug_info

        except Exception as e:
            return {'error': str(e)}
    
    def debug_kama_signal_detection(self, epic: str, pair: str, spread_pips: float = 1.5) -> Dict:
        """Debug KAMA signal detection specifically"""
        if not self.kama_strategy:
            return {'error': 'KAMA strategy not enabled or not available'}
            
        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=self._get_default_timeframe(), ema_strategy=self.ema_strategy)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return {'error': 'Insufficient data', 'bars': len(df) if df is not None else 0}
            
            # Test KAMA signal detection
            signal = self.kama_strategy.detect_signal(df, epic, spread_pips, '5m')
            
            if len(df) < 2:
                return {'error': 'Need at least 2 bars for signal detection'}
            
            # Get latest and previous values for analysis
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            debug_info = {
                'epic': epic,
                'timestamp': latest['start_time'],
                'signal_detected': signal is not None,
                'signal_type': signal.get('signal_type') if signal else None,
                'confidence_score': signal.get('confidence_score') if signal else 0,
                'strategy': 'kama_adaptive',
                'current_price': latest['close'],
                'prev_price': previous['close'],
                'above_threshold': signal.get('confidence_score', 0) >= system_config.MIN_CONFIDENCE if signal else False,
                'min_confidence_threshold': config.MIN_CONFIDENCE
            }
            
            # Add KAMA specific data if available
            kama_config = getattr(config, 'DEFAULT_KAMA_CONFIG', 'default')
            kama_period = getattr(config, 'KAMA_STRATEGY_CONFIG', {}).get(kama_config, {}).get('period', 10)
            
            kama_fields = [
                f'kama_{kama_period}', f'kama_{kama_period}_er', 
                f'kama_{kama_period}_trend', f'kama_{kama_period}_signal',
                'ema_200'
            ]
            
            for field in kama_fields:
                if field in latest:
                    debug_info[f'{field}_current'] = latest[field]
                if field in previous:
                    debug_info[f'{field}_prev'] = previous[field]
            
            # Add KAMA signal specific fields if signal exists
            if signal:
                kama_specific_fields = [
                    'kama_value', 'kama_trend', 'efficiency_ratio', 'kama_slope',
                    'trigger_reason', 'kama_period', 'kama_config'
                ]
                for field in kama_specific_fields:
                    if field in signal:
                        debug_info[field] = signal[field]
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def debug_signal_at_timestamp(
        self, 
        epic: str, 
        pair: str, 
        timestamp_str: str, 
        spread_pips: float = 1.5
    ) -> Dict:
        """Debug signal detection at a specific timestamp"""
        try:
            from utils.timezone_utils import TimezoneManager
            tz_manager = TimezoneManager(system_config.USER_TIMEZONE)
            
            # Parse the timestamp (assume Stockholm time)
            try:
                target_time = pd.to_datetime(timestamp_str)
                if target_time.tz is None:
                    target_time = tz_manager.local_tz.localize(target_time)
                target_utc = tz_manager.local_to_utc(target_time)
            except Exception as e:
                return {'error': f'Invalid timestamp format: {timestamp_str}. Use format: "2025-06-26 14:30"'}
            
            # Get enhanced data with extra lookback
            df_5m = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=self._get_default_timeframe(), lookback_hours=168,ema_strategy=self.ema_strategy)
            
            if df_5m is None or len(df_5m) < config.MIN_BARS_FOR_SIGNAL:
                return {'error': 'Insufficient data', 'bars': len(df_5m) if df_5m is not None else 0}
            
            # Find the closest timestamp to our target
            df_5m['time_diff'] = abs(pd.to_datetime(df_5m['start_time']) - target_utc)
            closest_idx = df_5m['time_diff'].idxmin()
            closest_timestamp = df_5m.loc[closest_idx, 'start_time']
            
            # Check if we found a reasonable match (within 30 minutes)
            time_diff_minutes = df_5m.loc[closest_idx, 'time_diff'].total_seconds() / 60
            if time_diff_minutes > 30:
                return {
                    'error': f'No data found near {timestamp_str}. Closest available: {closest_timestamp} ({time_diff_minutes:.1f} minutes away)'
                }
            
            # Get the data around this timestamp (need previous candle too)
            if closest_idx == 0:
                return {'error': 'Target timestamp is at the beginning of data - need previous candle for analysis'}
            
            # Create mini DataFrame for analysis
            history_start = max(0, closest_idx - 50)  # Get some history for indicators
            analysis_df = df_5m.iloc[history_start:closest_idx+1].copy()
            
            # Test signal detection at this point
            signal = self.ema_strategy.detect_signal_auto(analysis_df, epic, spread_pips, '5m')
            
            # Get specific row data
            target_row = df_5m.loc[closest_idx]
            prev_row = df_5m.loc[closest_idx - 1] if closest_idx > 0 else target_row
            
            debug_info = {
                'epic': epic,
                'timestamp': target_row['start_time'],
                'closest_timestamp': tz_manager.utc_to_local(pd.to_datetime(closest_timestamp)).strftime('%Y-%m-%d %H:%M:%S %Z'),
                'time_diff_minutes': time_diff_minutes,
                'signal_detected': signal is not None,
                'signal_type': signal.get('signal_type') if signal else None,
                'confidence_score': signal.get('confidence_score') if signal else 0,
                'strategy_used': signal.get('strategy') if signal else 'None',
                'current_price': target_row['close'],
                'prev_price': prev_row['close'],
                'ema_9_current': target_row.get('ema_9', 'N/A'),
                'ema_21_current': target_row.get('ema_21', 'N/A'),
                'ema_200_current': target_row.get('ema_200', 'N/A'),
                'spread_adjustment_pips': spread_pips,
                'original_close_price': target_row['close'],
                'min_confidence_threshold': system_config.MIN_CONFIDENCE,
                'above_threshold': signal.get('confidence_score', 0) >= system_config.MIN_CONFIDENCE if signal else False
            }
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def debug_macd_ema_signal(self, epic: str, pair: str, spread_pips: float = 1.5) -> Dict:
        """Debug MACD + EMA 200 signal detection"""
        try:
            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=self._get_default_timeframe(),ema_strategy=self.ema_strategy)
            
            if df is None or len(df) < system_config.MIN_BARS_FOR_SIGNAL:
                return {'error': 'Insufficient data', 'bars': len(df) if df is not None else 0}
            
            # Test MACD signal detection
            macd_strategy = self._get_macd_strategy_for_epic(epic)
            signal = macd_strategy.detect_signal(df, epic, spread_pips, '5m')
            
            if len(df) < 2:
                return {'error': 'Need at least 2 bars for signal detection'}
            
            # Get latest and previous values for analysis
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            debug_info = {
                'epic': epic,
                'timestamp': latest['start_time'],
                'signal_detected': signal is not None,
                'signal_type': signal.get('signal_type') if signal else None,
                'confidence_score': signal.get('confidence_score') if signal else 0,
                'strategy': 'macd_ema200',
                'current_price': latest['close'],
                'prev_price': previous['close'],
                'above_threshold': signal.get('confidence_score', 0) >= system_config.MIN_CONFIDENCE if signal else False,
                'min_confidence_threshold': config.MIN_CONFIDENCE
            }
            
            # Add MACD specific data if available
            macd_fields = ['ema_200', 'macd_line', 'macd_signal', 'macd_histogram', 'macd_color']
            for field in macd_fields:
                if field in latest:
                    debug_info[f'{field}_current'] = latest[field]
                if field in previous:
                    debug_info[f'{field}_prev'] = previous[field]
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def debug_combined_strategy_setup(self, epic: str, pair: str) -> Dict:
        """Debug combined strategy setup including KAMA"""
        debug_info = {
            'config_check': {},
            'individual_signals': {},
            'combined_available': True,
            'error': None
        }
        
        try:
            # Check config settings including KAMA
            debug_info['config_check'] = {
                'SIMPLE_EMA_STRATEGY': getattr(config, 'SIMPLE_EMA_STRATEGY', 'NOT_SET'),
                'MACD_STRATEGY': getattr(config, 'MACD_STRATEGY', 'NOT_SET'),
                'KAMA_STRATEGY': getattr(config, 'KAMA_STRATEGY', 'NOT_SET'),
                # Combined strategy config removed - strategy was disabled and unused
                'STRATEGY_WEIGHT_EMA': getattr(config, 'STRATEGY_WEIGHT_EMA', 'NOT_SET'),
                'STRATEGY_WEIGHT_MACD': getattr(config, 'STRATEGY_WEIGHT_MACD', 'NOT_SET'),
                'STRATEGY_WEIGHT_KAMA': getattr(config, 'STRATEGY_WEIGHT_KAMA', 'NOT_SET'),
                'DEFAULT_KAMA_CONFIG': getattr(config, 'DEFAULT_KAMA_CONFIG', 'NOT_SET'),
                'KAMA_STRATEGY_AVAILABLE': self.kama_strategy is not None
            }
            
            # Test individual strategies
            self.logger.info("Testing EMA strategy...")
            ema_signal = None
            if getattr(config, 'SIMPLE_EMA_STRATEGY', True):
                if config.USE_BID_ADJUSTMENT:
                    ema_signal = self.detect_signals_bid_adjusted(epic, pair, system_config.SPREAD_PIPS, '5m')
                else:
                    ema_signal = self.detect_signals_mid_prices(epic, pair, '5m')
            
            debug_info['individual_signals']['ema'] = {
                'signal_detected': ema_signal is not None,
                'signal_type': ema_signal.get('signal_type') if ema_signal else None,
                'confidence': ema_signal.get('confidence_score') if ema_signal else None,
                'strategy': ema_signal.get('strategy') if ema_signal else None
            }
            
            self.logger.info("Testing MACD strategy...")
            macd_signal = None
            if getattr(config, 'MACD_STRATEGY', False):
                macd_signal = self.detect_macd_ema_signals(epic, pair, system_config.SPREAD_PIPS, '5m')
            
            debug_info['individual_signals']['macd'] = {
                'signal_detected': macd_signal is not None,
                'signal_type': macd_signal.get('signal_type') if macd_signal else None,
                'confidence': macd_signal.get('confidence_score') if macd_signal else None,
                'strategy': macd_signal.get('strategy') if macd_signal else None
            }
            
            # Test KAMA strategy
            self.logger.info("Testing KAMA strategy...")
            kama_signal = None
            if getattr(config, 'KAMA_STRATEGY', False) and self.kama_strategy:
                kama_signal = self.detect_kama_signals(epic, pair, system_config.SPREAD_PIPS, '5m')
            
            debug_info['individual_signals']['kama'] = {
                'signal_detected': kama_signal is not None,
                'signal_type': kama_signal.get('signal_type') if kama_signal else None,
                'confidence': kama_signal.get('confidence_score') if kama_signal else None,
                'strategy': kama_signal.get('strategy') if kama_signal else None,
                'efficiency_ratio': kama_signal.get('efficiency_ratio') if kama_signal else None,
                'market_regime': kama_signal.get('market_regime') if kama_signal else None
            }
            
            # Combined strategy testing removed - strategy was disabled and unused
            debug_info['combined_signal'] = {
                'signal_detected': False,
                'note': 'Combined strategy removed - was disabled and unused'
            }
            
        except Exception as e:
            debug_info['error'] = str(e)
            self.logger.error(f"Debug error: {e}")
        
        return debug_info
    
    def debug_bb_supertrend_signal_detection(self, epic: str, pair: str, spread_pips: float = 1.5) -> Dict:
        """Debug BB+Supertrend signal detection specifically"""
        if not self.bb_supertrend_strategy:
            return {'error': 'BB+Supertrend strategy not enabled or failed to initialize'}
        
        try:
            # Get data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe='15m')
            
            if df is None or len(df) < 30:
                return {'error': f'Insufficient data: {len(df) if df is not None else 0} bars'}
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Test signal detection
            signal = self.bb_supertrend_strategy.detect_signal(df, epic, spread_pips, '15m')
            
            debug_info = {
                'epic': epic,
                'timestamp': latest.get('start_time', 'N/A'),
                'current_price': latest['close'],
                'bb_upper': latest.get('bb_upper', 'N/A'),
                'bb_middle': latest.get('bb_middle', 'N/A'),
                'bb_lower': latest.get('bb_lower', 'N/A'),
                'supertrend': latest.get('supertrend', 'N/A'),
                'supertrend_direction': latest.get('supertrend_direction', 'N/A'),
                'atr': latest.get('atr', 'N/A'),
                'signal_detected': signal is not None,
                'signal_type': signal.get('signal_type') if signal else None,
                'confidence_score': signal.get('confidence_score') if signal else 0,
                'strategy_used': signal.get('strategy') if signal else 'None',
                'spread_adjustment_pips': spread_pips,
                'min_confidence_threshold': system_config.MIN_CONFIDENCE,
                'above_threshold': signal.get('confidence_score', 0) >= system_config.MIN_CONFIDENCE if signal else False,
                'bb_width': latest.get('bb_upper', 0) - latest.get('bb_lower', 0),
                'price_vs_bb_lower': latest['close'] - latest.get('bb_lower', latest['close']),
                'price_vs_bb_upper': latest['close'] - latest.get('bb_upper', latest['close'])
            }
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}

    def detect_scalping_signals(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect scalping signals using ultra-fast EMAs
        
        Args:
            epic: Epic code
            pair: Currency pair
            spread_pips: Spread in pips
            timeframe: Timeframe ('1m' or '5m' recommended)
            
        Returns:
            Scalping signal or None
        """
        try:
            # Get enhanced data with shorter lookback for scalping
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe, lookback_hours=24,ema_strategy=self.ema_strategy)
            
            if df is None or len(df) < 50:  # Need less data for scalping
                return None
            
            # Use scalping strategy
            signal = self.scalping_strategy.detect_signal(df, epic, spread_pips, timeframe)
            
            if signal:
                # Add scalping-specific market context
                signal = self._add_scalping_context(signal, df)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting scalping signals for {epic}: {e}")
            return None

    def _add_scalping_context(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Add scalping-specific market context"""
        latest = df.iloc[-1]
        
        # Add scalping-relevant context
        scalping_context = {
            'atr_5': df['close'].rolling(5).apply(lambda x: x.max() - x.min()).iloc[-1] * 10000,  # 5-bar ATR in pips
            'recent_range_pips': (df['high'].tail(10).max() - df['low'].tail(10).min()) * 10000,
            'current_session': self._get_current_session(),
            'bars_since_crossover': 1,  # Could be calculated more precisely
            'scalping_quality_score': self._calculate_scalping_quality(df)
        }
        
        signal.update(scalping_context)
        return signal

    def _get_current_session(self) -> str:
        """Get current trading session"""
        from datetime import datetime
        import pytz
        
        now = datetime.now(pytz.timezone(system_config.USER_TIMEZONE))
        hour = now.hour
        
        if 8 <= hour < 13:
            return 'london'
        elif 13 <= hour < 17:
            return 'london_new_york_overlap'  # Best for scalping
        elif 17 <= hour < 22:
            return 'new_york'
        else:
            return 'asian'

    def _calculate_scalping_quality(self, df: pd.DataFrame) -> float:
        """Calculate market quality score for scalping (0-1)"""
        if len(df) < 20:
            return 0.5
        
        recent_data = df.tail(20)
        
        # Volume consistency (prefer steady volume)
        if 'ltv' in recent_data.columns:
            volume_cv = recent_data['ltv'].std() / recent_data['ltv'].mean()
            volume_score = max(0, 1 - volume_cv)  # Lower CV = better
        else:
            volume_score = 0.5
        
        # Price range consistency (avoid too choppy or too quiet)
        ranges = recent_data['high'] - recent_data['low']
        avg_range = ranges.mean() * 10000  # in pips
        range_score = 1.0 if 2 <= avg_range <= 8 else 0.5  # Ideal 2-8 pip ranges
        
        # Trend consistency (mild trend preferred over ranging)
        price_changes = recent_data['close'].diff().dropna()
        trend_score = abs(price_changes.mean()) / price_changes.std() if price_changes.std() > 0 else 0
        trend_score = min(1.0, trend_score)
        
        return (volume_score + range_score + trend_score) / 3

    def detect_volume_profile_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect Volume Profile signals using institutional volume analysis

        Signals:
        - HVN Bounce: Price bounces at High Volume Nodes (support/resistance)
        - POC Reversion: Mean reversion to Point of Control
        - Value Area Breakout: Price breaks above VAH or below VAL
        - LVN Breakout: Price breaks through Low Volume Nodes

        Args:
            epic: Epic code
            pair: Currency pair
            spread_pips: Spread in pips
            timeframe: Timeframe (default: 15m)

        Returns:
            Volume Profile signal or None
        """
        # Check if Volume Profile strategy is enabled
        if not self.volume_profile_strategy:
            self.logger.debug("Volume Profile strategy not initialized")
            return None

        try:
            # Get required indicators from strategy
            required_indicators = self.volume_profile_strategy.get_required_indicators()

            # Get enhanced data with required indicators
            df = self.data_fetcher.get_enhanced_data(
                epic, pair,
                timeframe=timeframe,
                ema_strategy=self.ema_strategy,
                required_indicators=required_indicators
            )

            if df is None or len(df) < 70:  # Need sufficient data for volume profile
                self.logger.debug(f"Insufficient data for Volume Profile: {len(df) if df is not None else 0} bars")
                return None

            # Use Volume Profile strategy
            signal = self.volume_profile_strategy.detect_signal(df, epic, spread_pips, timeframe or '15m')

            if signal:
                # Add Volume Profile specific market context
                signal = self._add_volume_profile_context(signal, df)

            return signal

        except Exception as e:
            self.logger.error(f"Error detecting Volume Profile signals for {epic}: {e}")
            return None

    def _add_volume_profile_context(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Add Volume Profile specific market context and technical indicators"""
        try:
            latest = df.iloc[-1]

            # Add complete technical indicators (including EMA200 for validation)
            signal = self._add_complete_technical_indicators(signal, df)

            # Add VP-relevant context
            vp_context = {
                'hvn_price': signal.get('metadata', {}).get('hvn_price'),
                'poc': signal.get('metadata', {}).get('poc'),
                'vah': signal.get('metadata', {}).get('vah'),
                'val': signal.get('metadata', {}).get('val'),
                'signal_source': signal.get('signal_source', 'unknown'),
                'distance_to_poc_pips': signal.get('metadata', {}).get('distance_to_poc_pips'),
            }

            if 'market_context' not in signal:
                signal['market_context'] = {}

            signal['market_context']['volume_profile'] = vp_context

            return signal

        except Exception as e:
            self.logger.debug(f"Could not add Volume Profile context: {e}")
            return signal

    def detect_ema_double_confirmation_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = '15m'
    ) -> Optional[Dict]:
        """
        Detect EMA Double Confirmation signals.

        Strategy Logic:
        1. Detect EMA 21/50 crossovers
        2. Validate crossover "success" (price stays favorable for 4 candles)
        3. After 2 successful crossovers in same direction within 48h,
           take the 3rd crossover as entry signal

        Args:
            epic: Epic code
            pair: Currency pair
            spread_pips: Spread in pips
            timeframe: Timeframe (default: 15m)

        Returns:
            EMA Double Confirmation signal or None
        """
        # Check if strategy is enabled
        if not self.ema_double_confirmation_strategy:
            self.logger.debug("EMA Double Confirmation strategy not initialized")
            return None

        try:
            # Get enhanced data for the timeframe
            # Need 48h for crossover lookback + buffer for EMAs + buffer for 4H EMA 50 filter
            # 4H EMA 50 needs 50 * 8 = 400 bars of 15m (~100 hours)
            # Using 120 hours to provide crossover lookback (48h) + HTF EMA warmup (72h)
            df = self.data_fetcher.get_enhanced_data(
                epic, pair,
                timeframe=timeframe,
                lookback_hours=120,  # 48h crossover lookback + 72h HTF EMA warmup
                ema_strategy=self.ema_strategy if hasattr(self, 'ema_strategy') else None
            )

            if df is None or len(df) < 60:  # Need enough for EMA 50 + buffer
                self.logger.debug(f"Insufficient data for EMA Double Confirmation: {len(df) if df is not None else 0} bars")
                return None

            # Detect signal using strategy
            signal = self.ema_double_confirmation_strategy.detect_signal(
                df, epic, spread_pips, timeframe
            )

            if signal:
                # Add market context
                signal = self._add_market_context(signal, df)
                self.logger.info(f"✅ [EMA_DOUBLE] Signal for {epic}: {signal['signal_type']}")

            return signal

        except Exception as e:
            self.logger.error(f"Error detecting EMA Double Confirmation signals for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_silver_bullet_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = '5m'
    ) -> Optional[Dict]:
        """
        Detect ICT Silver Bullet signals.

        Strategy Logic:
        1. Check if in Silver Bullet time window (3-4AM, 10-11AM, 2-3PM NY)
        2. Detect liquidity sweep (BSL for shorts, SSL for longs)
        3. Confirm Market Structure Shift
        4. Find Fair Value Gap entry
        5. Generate signal with SL/TP

        Args:
            epic: Epic code
            pair: Currency pair
            spread_pips: Spread in pips
            timeframe: Timeframe (default: 5m for entry precision)

        Returns:
            Silver Bullet signal or None
        """
        # Check if strategy is initialized
        if not self.silver_bullet_strategy:
            self.logger.debug("Silver Bullet strategy not initialized")
            return None

        try:
            # Debug: Check if backtest time is set
            if hasattr(self.data_fetcher, 'current_backtest_time'):
                self.logger.debug(f"🔧 SB: Backtest time set to: {self.data_fetcher.current_backtest_time}")

            # Get entry timeframe data (5m)
            df_entry = self.data_fetcher.get_enhanced_data(
                epic, pair,
                timeframe='5m',
                lookback_hours=24,  # 24h for liquidity detection
                ema_strategy=self.ema_strategy if hasattr(self, 'ema_strategy') else None
            )

            if df_entry is None or len(df_entry) < 50:
                self.logger.debug(f"Insufficient 5m data for Silver Bullet: {len(df_entry) if df_entry is not None else 0} bars")
                return None

            # Debug: Log the actual timestamp from the data
            if 'start_time' in df_entry.columns:
                last_ts = df_entry['start_time'].iloc[-1]
                self.logger.debug(f"🔧 SB: Last 5m candle timestamp: {last_ts}")

            # Get HTF data for bias (1h)
            df_htf = self.data_fetcher.get_enhanced_data(
                epic, pair,
                timeframe='1h',
                lookback_hours=72,  # 72h for trend context
                ema_strategy=self.ema_strategy if hasattr(self, 'ema_strategy') else None
            )

            if df_htf is None or len(df_htf) < 20:
                self.logger.debug(f"Insufficient 1h data for Silver Bullet: {len(df_htf) if df_htf is not None else 0} bars")
                return None

            # Optional: Get trigger timeframe data (15m)
            df_trigger = self.data_fetcher.get_enhanced_data(
                epic, pair,
                timeframe='15m',
                lookback_hours=48,
                ema_strategy=self.ema_strategy if hasattr(self, 'ema_strategy') else None
            )

            # Detect signal using strategy
            signal = self.silver_bullet_strategy.detect_signal(
                df_entry=df_entry,
                df_htf=df_htf,
                epic=epic,
                pair=pair,
                df_trigger=df_trigger
            )

            if signal:
                # Add market context
                signal = self._add_market_context(signal, df_entry)
                self.logger.info(f"✅ [SILVER_BULLET] Signal for {epic}: {signal['signal_type']} during {signal.get('session', 'unknown')}")

            return signal

        except Exception as e:
            self.logger.error(f"Error detecting Silver Bullet signals for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_signals_multi_timeframe(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5,
        primary_timeframe: str = None
    ) -> Optional[Dict]:
        """
        Enhanced multi-timeframe signal detection with improved confluence scoring
        """
        try:
            self.logger.debug("Debugging MTF method step by step...")
            
            # Try to get multi-timeframe data
            from analysis.multi_timeframe import MultiTimeframeAnalyzer
            
            df_5m = self.data_fetcher.get_enhanced_data(epic, pair, self._get_default_timeframe(), lookback_hours=48, ema_strategy=self.ema_strategy)
            df_15m = self.data_fetcher.get_enhanced_data(epic, pair, '15m', lookback_hours=168, ema_strategy=self.ema_strategy)
            df_1h = self.data_fetcher.get_enhanced_data(epic, pair, '1h', lookback_hours=720, ema_strategy=self.ema_strategy)
            
            self.logger.debug(f"✅ Data fetched: 5m={len(df_5m) if df_5m is not None else 0}, "
                            f"15m={len(df_15m) if df_15m is not None else 0}, "
                            f"1h={len(df_1h) if df_1h is not None else 0}")
            
            # Check if we have enough data for multi-timeframe analysis
            min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)
            sufficient_data = not any(df is None or len(df) < min_bars for df in [df_5m, df_15m, df_1h])
            
            self.logger.debug(f"✅ Sufficient data check: {sufficient_data} (min_bars: {min_bars})")
            
            if not sufficient_data:
                # Fall back to existing single timeframe method
                self.logger.debug("Falling back to single timeframe detection")
                if config.USE_BID_ADJUSTMENT:
                    return self.detect_signals_bid_adjusted(epic, pair, spread_pips, primary_timeframe)
                else:
                    return self.detect_signals_mid_adjusted(epic, pair, primary_timeframe)
            
            # Get single timeframe signal first using existing method
            self.logger.debug("🔍 Testing base signal detection...")
            if config.USE_BID_ADJUSTMENT:
                base_signal = self.detect_signals_bid_adjusted(epic, pair, spread_pips, primary_timeframe)
                self.logger.debug(f"Base signal (bid_adjusted): {base_signal is not None}")
            else:
                base_signal = self.detect_signals_mid_prices(epic, pair, primary_timeframe)
                self.logger.debug(f"Base signal (mid_prices): {base_signal is not None}")
            
            if not base_signal:
                self.logger.debug("No base signal found")
                return None
            
            self.logger.debug(f"✅ Base signal found: {base_signal['signal_type']}, "
                            f"confidence: {base_signal.get('confidence_score', 0):.3f}")
            
            # Add multi-timeframe analysis if we have the data
            try:
                mt_analyzer = MultiTimeframeAnalyzer()
                
                # Perform confluence analysis across multiple strategies
                confluence_result = mt_analyzer.analyze_signal_confluence(
                    df_5m, pair, spread_pips, primary_timeframe
                )
                
                # Calculate enhanced confluence score
                # We'll combine the base signal quality with multi-strategy confluence
                base_confluence = mt_analyzer.get_confluence_score(base_signal)
                strategy_confluence = confluence_result.get('confluence_score', 0.0)
                
                # Weighted combination: 60% from base signal, 40% from strategy confluence
                final_confluence = (base_confluence * 0.6) + (strategy_confluence * 0.4)
                
                self.logger.debug(f"✅ Confluence score: {final_confluence:.3f}")
                
                # Enhance the signal
                enhanced_signal = base_signal.copy()
                enhanced_signal.update({
                    'confluence_score': final_confluence,
                    'multi_timeframe_analysis': True,
                    'timeframes_analyzed': ['5m', '15m', '1h'],
                    'base_confidence': base_signal.get('confidence_score', 0.6),
                    'confluence_bonus': min(0.15, final_confluence * 0.25),
                    'strategy_confluence': strategy_confluence,
                    'base_confluence': base_confluence,
                    'strategies_tested': confluence_result.get('strategies_tested', []),
                    'strategy_count': confluence_result.get('strategy_count', 0)
                })
                
                # Update final confidence
                final_confidence = min(0.95, 
                    enhanced_signal['base_confidence'] + enhanced_signal['confluence_bonus'])
                enhanced_signal['confidence_score'] = final_confidence
                
                # Check minimum confluence threshold
                min_confluence = getattr(config, 'MIN_CONFLUENCE_SCORE', 0.2)
                self.logger.debug(f"Min confluence required: {min_confluence}")
                self.logger.debug(f"Passes confluence test: {final_confluence >= min_confluence}")
                
                if final_confluence >= min_confluence:
                    return enhanced_signal
                else:
                    self.logger.debug(f"Low confluence signal filtered: {epic} "
                                    f"(score: {final_confluence:.3f} < {min_confluence})")
                    return None
                    
            except Exception as mt_error:
                self.logger.warning(f"Multi-timeframe enhancement failed for {epic}: {mt_error}")
                # Return the base signal if multi-timeframe enhancement fails
                return base_signal
                
        except Exception as e:
            self.logger.error(f"❌ Multi-timeframe analysis failed for {epic}: {e}")
            # Final fallback to existing method
            if config.USE_BID_ADJUSTMENT:
                return self.detect_signals_bid_adjusted(epic, pair, spread_pips, primary_timeframe)
            else:
                return self.detect_signals_mid_prices(epic, pair, primary_timeframe)

    def _detect_signal_on_timeframe(self, df: pd.DataFrame, epic: str, pair: str, spread_pips: float, timeframe: str) -> Optional[Dict]:
        """
        Helper method to detect signals on a specific timeframe
        Uses your existing signal detection logic
        """
        # Use your existing strategy detection methods
        strategies_to_try = []
        
        if getattr(config, 'SIMPLE_EMA_STRATEGY', True):
            strategies_to_try.append('ema')
        if getattr(config, 'MACD_STRATEGY', False):
            strategies_to_try.append('macd')
        if getattr(config, 'KAMA_STRATEGY', False) and self.kama_strategy:
            strategies_to_try.append('kama')
        
        for strategy in strategies_to_try:
            if strategy == 'ema':
                signal = self._detect_ema_signals(df, epic, pair, spread_pips, timeframe)
            elif strategy == 'macd':
                signal = self._detect_macd_signals(df, epic, pair, spread_pips, timeframe)
            elif strategy == 'kama':
                signal = self.kama_strategy.detect_signal(df, epic, spread_pips, timeframe)
            else:
                continue
                
            if signal:
                signal['primary_strategy'] = strategy
                signal['timeframe'] = timeframe
                return signal
        
        return None
    
    def analyze_signal_confluence(
        self,
        df: pd.DataFrame,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Dict:
        """
        Analyze signal confluence across multiple strategies including KAMA
        This method should be added to the SignalDetector class
        
        Args:
            df: DataFrame with market data
            pair: Currency pair
            spread_pips: Spread in pips  
            timeframe: Primary timeframe
            
        Returns:
            Dictionary with confluence analysis results
        """
        confluence_result = {
            'confluence_score': 0.0,
            'dominant_direction': 'NEUTRAL',
            'confidence_weighted_direction': 'NEUTRAL',
            'bull_signals': [],
            'bear_signals': [],
            'agreement_level': 'low',
            'strategy_count': 0,
            'strategies_tested': []
        }
        
        try:
            # Collect signals from all enabled strategies
            all_signals = []
            
            # Test EMA strategy
            if getattr(config, 'SIMPLE_EMA_STRATEGY', True):
                try:
                    signal = self.ema_strategy.detect_signal_auto(df, pair, spread_pips, timeframe)
                    if signal:
                        signal['strategy_name'] = 'ema'
                        all_signals.append(signal)
                        confluence_result['strategies_tested'].append('ema')
                except Exception as e:
                    self.logger.debug(f"EMA strategy failed: {e}")
            
            # Test MACD strategy
            if getattr(config, 'MACD_STRATEGY', False):
                try:
                    macd_strategy = self._get_macd_strategy_for_epic(epic)
                    signal = macd_strategy.detect_signal(df, pair, spread_pips, timeframe)
                    if signal:
                        signal['strategy_name'] = 'macd'
                        all_signals.append(signal)
                        confluence_result['strategies_tested'].append('macd')
                except Exception as e:
                    self.logger.debug(f"MACD strategy failed: {e}")
            
            # Test KAMA strategy
            if getattr(config, 'KAMA_STRATEGY', False) and self.kama_strategy:
                try:
                    signal = self.kama_strategy.detect_signal(df, pair, spread_pips, timeframe)
                    if signal:
                        signal['strategy_name'] = 'kama'
                        all_signals.append(signal)
                        confluence_result['strategies_tested'].append('kama')
                except Exception as e:
                    self.logger.debug(f"KAMA strategy failed: {e}")
            
            # Test BB+Supertrend strategy
            if getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False):
                try:
                    signal = self.bb_supertrend_strategy.detect_signal(df, pair, spread_pips, timeframe)
                    if signal:
                        signal['strategy_name'] = 'bb_supertrend'
                        all_signals.append(signal)
                        confluence_result['strategies_tested'].append('bb_supertrend')
                except Exception as e:
                    self.logger.debug(f"BB+Supertrend strategy failed: {e}")

            # Combined strategy removed - was disabled and unused
            
            confluence_result['strategy_count'] = len(all_signals)
            
            if not all_signals:
                self.logger.debug("No signals found for confluence analysis")
                return confluence_result
            
            # Categorize signals by direction
            for signal in all_signals:
                if signal['signal_type'] == 'BULL':
                    confluence_result['bull_signals'].append(signal)
                elif signal['signal_type'] == 'BEAR':
                    confluence_result['bear_signals'].append(signal)
            
            # Calculate confluence metrics
            bull_count = len(confluence_result['bull_signals'])
            bear_count = len(confluence_result['bear_signals'])
            total_signals = bull_count + bear_count
            
            if total_signals == 0:
                return confluence_result
            
            # Determine dominant direction by count
            if bull_count > bear_count:
                confluence_result['dominant_direction'] = 'BULL'
                confluence_result['confluence_score'] = bull_count / total_signals
            elif bear_count > bull_count:
                confluence_result['dominant_direction'] = 'BEAR'
                confluence_result['confluence_score'] = bear_count / total_signals
            else:
                confluence_result['dominant_direction'] = 'NEUTRAL'
                confluence_result['confluence_score'] = 0.5
            
            # Calculate confidence-weighted direction including KAMA
            bull_confidence_sum = sum(s.get('confidence_score', 0.5) for s in confluence_result['bull_signals'])
            bear_confidence_sum = sum(s.get('confidence_score', 0.5) for s in confluence_result['bear_signals'])
            total_confidence = bull_confidence_sum + bear_confidence_sum
            
            if total_confidence > 0:
                if bull_confidence_sum > bear_confidence_sum:
                    confluence_result['confidence_weighted_direction'] = 'BULL'
                elif bear_confidence_sum > bull_confidence_sum:
                    confluence_result['confidence_weighted_direction'] = 'BEAR'
                else:
                    confluence_result['confidence_weighted_direction'] = 'NEUTRAL'
                
                # Adjust confluence score based on confidence weights
                max_confidence = max(bull_confidence_sum, bear_confidence_sum)
                confidence_factor = max_confidence / total_confidence
                confluence_result['confluence_score'] = min(1.0, confluence_result['confluence_score'] * confidence_factor)
            
            # Determine agreement level
            if confluence_result['confluence_score'] >= 0.7:
                confluence_result['agreement_level'] = 'high'
            elif confluence_result['confluence_score'] >= 0.4:
                confluence_result['agreement_level'] = 'medium'
            else:
                confluence_result['agreement_level'] = 'low'
            
            self.logger.debug(f"Confluence analysis: {confluence_result['dominant_direction']} "
                            f"(Score: {confluence_result['confluence_score']:.2f}, "
                            f"Bull: {bull_count}, Bear: {bear_count})")
            
            return confluence_result
            
        except Exception as e:
            self.logger.error(f"Error in confluence analysis: {e}")
            return confluence_result

    def _detect_ema_signals(self, df: pd.DataFrame, epic: str, pair: str, spread_pips: float, timeframe: str) -> Optional[Dict]:
        """Helper method to detect EMA signals"""
        return self.ema_strategy.detect_signa_autol(df, epic, spread_pips, timeframe)
    
    def _detect_macd_signals(self, df: pd.DataFrame, epic: str, pair: str, spread_pips: float, timeframe: str) -> Optional[Dict]:
        """Helper method to detect MACD signals"""
        macd_strategy = self._get_macd_strategy_for_epic(epic)
        return macd_strategy.detect_signal(df, epic, spread_pips, timeframe)
    
    def _normalize_timestamp(self, timestamp: Union[str, pd.Timestamp, datetime]) -> datetime:
        """
        FIXED: Normalize different timestamp types to datetime for consistent sorting
        
        Args:
            timestamp: Timestamp in various formats (str, pd.Timestamp, datetime)
            
        Returns:
            datetime: Normalized datetime object
        """
        try:
            if isinstance(timestamp, str):
                # Handle string timestamps
                return pd.to_datetime(timestamp).to_pydatetime()
            elif isinstance(timestamp, pd.Timestamp):
                # Handle pandas Timestamp
                return timestamp.to_pydatetime()
            elif isinstance(timestamp, datetime):
                # Already datetime
                return timestamp
            else:
                # Fallback: try converting to pandas timestamp first
                return pd.to_datetime(timestamp).to_pydatetime()
        except Exception as e:
            # If all else fails, use current time as fallback
            self.logger.warning(f"⚠️ Could not normalize timestamp {timestamp}: {e}, using current time")
            return datetime.now()
    
    def _add_complete_technical_indicators(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """
        NEW METHOD: Add complete technical indicators from DataFrame to signal
        This ensures Claude gets ALL available technical data for comprehensive analysis
        """
        if df is None or df.empty:
            return signal
        
        try:
            latest = df.iloc[-1]
            
            # 1. PRICE DATA (always include)
            signal.update({
                'current_price': float(latest['close']),
                'open_price': float(latest['open']),
                'high_price': float(latest['high']),
                'low_price': float(latest['low']),
                'close_price': float(latest['close'])
            })
            
            # 2. EMA INDICATORS (all available periods)
            ema_indicators = {}
            for col in df.columns:
                if col.startswith('ema_') and col.replace('ema_', '').isdigit():
                    try:
                        period = int(col.replace('ema_', ''))
                        ema_indicators[col] = float(latest[col])
                        
                        # Map to standard names
                        if period == 9:
                            signal['ema_9'] = float(latest[col])
                            signal['ema_short'] = float(latest[col])
                        elif period == 21:
                            signal['ema_21'] = float(latest[col])
                            signal['ema_long'] = float(latest[col])
                        elif period == 200:
                            signal['ema_200'] = float(latest[col])
                            signal['ema_trend'] = float(latest[col])
                            
                    except (ValueError, KeyError):
                        continue
            
            # Add EMA data to signal
            if ema_indicators:
                signal.update(ema_indicators)
            
            # 3. MACD INDICATORS (critical for enhanced analysis)
            macd_indicators = {}
            macd_mappings = {
                'macd_line': ['macd_line', 'macd', 'macd_12_26_9'],  
                'macd_signal': ['macd_signal', 'macd_signal_line', 'macd_signal_12_26_9'],
                'macd_histogram': ['macd_histogram', 'macd_hist', 'macd_histogram_12_26_9']
            }
            
            for standard_name, possible_cols in macd_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            macd_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue
            
            # 4. KAMA INDICATORS (if available)
            kama_indicators = {}
            kama_mappings = {
                'kama_value': ['kama_value', 'kama', 'kama_10', 'kama_14'],
                'efficiency_ratio': ['efficiency_ratio', 'kama_er', 'kama_10_er', 'kama_14_er'],
                'kama_trend': ['kama_trend', 'kama_slope', 'kama_10_trend', 'kama_14_trend']
            }
            
            for standard_name, possible_cols in kama_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            kama_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue
            
            # 5. ADX AND DIRECTIONAL INDICATORS
            adx_indicators = {}
            adx_mappings = {
                'adx': ['adx', 'adx_14'],
                'plus_di': ['plus_di', 'di_plus', 'plus_di_14', '+di'],
                'minus_di': ['minus_di', 'di_minus', 'minus_di_14', '-di'],
                'dx': ['dx', 'dx_14']
            }

            for standard_name, possible_cols in adx_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            adx_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 6. OTHER TECHNICAL INDICATORS
            other_indicators = {}
            other_mappings = {
                'rsi': ['rsi', 'rsi_14'],
                'atr': ['atr', 'atr_14'],
                'bb_upper': ['bb_upper', 'bollinger_upper', 'bb_upper_20_2'],
                'bb_middle': ['bb_middle', 'bollinger_middle', 'bb_middle_20_2'],
                'bb_lower': ['bb_lower', 'bollinger_lower', 'bb_lower_20_2']
            }

            for standard_name, possible_cols in other_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            other_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 7. VOLUME DATA
            volume_fields = ['ltv', 'volume', 'volume_sma_20', 'volume_ratio_20']
            for field in volume_fields:
                if field in df.columns:
                    try:
                        if field == 'ltv' or field == 'volume':
                            signal['volume'] = float(latest[field])
                        else:
                            signal[field] = float(latest[field])
                    except (ValueError, KeyError):
                        continue

            # 8. SWING POINT DATA (from SMC/market structure analysis)
            swing_data = {}
            swing_fields = {
                'swing_high': ['swing_high', 'recent_swing_high', 'swing_high_price'],
                'swing_low': ['swing_low', 'recent_swing_low', 'swing_low_price'],
                'distance_to_swing_high_pips': ['distance_to_swing_high_pips', 'swing_high_distance_pips'],
                'distance_to_swing_low_pips': ['distance_to_swing_low_pips', 'swing_low_distance_pips'],
                'nearest_swing_type': ['nearest_swing_type', 'swing_type'],
                'swing_strength': ['swing_strength', 'swing_level_strength']
            }

            for standard_name, possible_cols in swing_fields.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            value = latest[col]
                            # Handle both numeric and string values
                            if isinstance(value, (int, float)):
                                swing_data[standard_name] = float(value)
                            else:
                                swing_data[standard_name] = str(value)
                            signal[standard_name] = swing_data[standard_name]
                            break
                        except (ValueError, KeyError):
                            continue

            # 9. SUPPORT/RESISTANCE DATA (enhanced)
            sr_data = {}
            sr_fields = {
                'nearest_support': ['nearest_support', 'support_level', 'support_price'],
                'nearest_resistance': ['nearest_resistance', 'resistance_level', 'resistance_price'],
                'distance_to_support_pips': ['distance_to_support_pips', 'support_distance_pips'],
                'distance_to_resistance_pips': ['distance_to_resistance_pips', 'resistance_distance_pips'],
                'support_strength': ['support_strength', 'support_level_strength'],
                'resistance_strength': ['resistance_strength', 'resistance_level_strength'],
                'level_flip_detected': ['level_flip_detected', 'sr_flip_detected'],
                'cluster_risk_level': ['cluster_risk_level', 'sr_cluster_risk']
            }

            for standard_name, possible_cols in sr_fields.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            value = latest[col]
                            # Handle both numeric and boolean/string values
                            if isinstance(value, bool):
                                sr_data[standard_name] = bool(value)
                            elif isinstance(value, (int, float)):
                                sr_data[standard_name] = float(value)
                            else:
                                sr_data[standard_name] = str(value)
                            signal[standard_name] = sr_data[standard_name]
                            break
                        except (ValueError, KeyError):
                            continue

            # 10. ADDITIONAL CONTEXT DATA
            context_fields = [
                'volume_confirmation', 'trend_alignment', 'market_session',
                'consolidation_range_pips', 'bars_since_breakout'
            ]
            for field in context_fields:
                if field in df.columns:
                    try:
                        signal[field] = latest[field]
                    except (ValueError, KeyError):
                        continue
            
            # 11. CREATE COMPREHENSIVE STRATEGY_INDICATORS JSON
            all_indicators = {}
            all_indicators.update(ema_indicators)
            all_indicators.update(macd_indicators)
            all_indicators.update(kama_indicators)
            all_indicators.update(adx_indicators)
            all_indicators.update(other_indicators)

            # Helper function to clean NaN values from dict (PostgreSQL JSON doesn't accept NaN)
            def clean_nan_values(obj):
                """Recursively replace NaN with None in dict/list structures"""
                if isinstance(obj, dict):
                    return {k: clean_nan_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan_values(item) for item in obj]
                elif isinstance(obj, float):
                    import math
                    return None if math.isnan(obj) or math.isinf(obj) else obj
                else:
                    return obj

            if all_indicators:
                # FIXED: Preserve existing strategy_indicators (e.g., from SMC strategy)
                # Instead of overwriting, merge the dataframe analysis with existing data
                existing_strategy_indicators = signal.get('strategy_indicators', {})

                # DEBUG: Log what we found
                strategy_name = signal.get('strategy', 'unknown')
                has_existing = bool(existing_strategy_indicators)
                if has_existing:
                    existing_keys = list(existing_strategy_indicators.keys())
                    self.logger.info(f"🔍 [{strategy_name}] Found existing strategy_indicators with keys: {existing_keys}")
                else:
                    self.logger.info(f"🔍 [{strategy_name}] No existing strategy_indicators found - will use dataframe analysis")

                # Create base dataframe analysis indicators
                dataframe_indicators = clean_nan_values({
                    'ema_data': ema_indicators,
                    'macd_data': macd_indicators,
                    'kama_data': kama_indicators,
                    'adx_data': adx_indicators,
                    'swing_data': swing_data,
                    'sr_data': sr_data,
                    'other_indicators': other_indicators,
                    'indicator_count': len(all_indicators),
                    'data_source': 'complete_dataframe_analysis'
                })

                # Merge: strategy-specific indicators take precedence, dataframe analysis fills gaps
                if existing_strategy_indicators:
                    # Strategy already provided comprehensive indicators (e.g., SMC structure data)
                    # Add dataframe analysis as supplementary data without overwriting
                    merged_indicators = existing_strategy_indicators.copy()
                    merged_indicators['dataframe_analysis'] = dataframe_indicators
                    signal['strategy_indicators'] = merged_indicators
                    self.logger.info(f"✅ [{strategy_name}] Preserved strategy indicators + added {len(all_indicators)} dataframe indicators")
                    self.logger.info(f"   Final keys: {list(merged_indicators.keys())}")
                else:
                    # No existing strategy indicators, use dataframe analysis as primary
                    signal['strategy_indicators'] = dataframe_indicators
                    self.logger.debug(f"📊 Enhanced signal with {len(all_indicators)} indicators + swing/SR data")

            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error adding complete technical indicators: {e}")
            return signal