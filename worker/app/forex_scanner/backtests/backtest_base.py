# ============================================================================
# backtests/backtest_base.py
# ============================================================================

import logging
import pandas as pd
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime

try:
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
    from core.backtest.performance_analyzer import PerformanceAnalyzer
    from core.backtest.signal_analyzer import SignalAnalyzer
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.backtest.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.core.backtest.signal_analyzer import SignalAnalyzer

# Import optimization service
try:
    from optimization.optimal_parameter_service import OptimalParameterService
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.optimization.optimal_parameter_service import OptimalParameterService
        OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        logging.getLogger(__name__).warning("Optimization service not available - using fallback parameters")

try:
    import config
except ImportError:
    from forex_scanner import config


class BacktestBase(ABC):
    """Base class for all strategy backtests with database optimization support"""
    
    def __init__(self, strategy_name: str, use_optimal_parameters: bool = True):
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"backtest_{strategy_name}")
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        
        # Database optimization integration
        self.use_optimal_parameters = use_optimal_parameters and OPTIMIZATION_AVAILABLE
        self.optimal_service = OptimalParameterService() if self.use_optimal_parameters else None
        
        if self.use_optimal_parameters:
            self.logger.info(f"🎯 Database optimization ENABLED for {strategy_name} backtests")
        else:
            if use_optimal_parameters and not OPTIMIZATION_AVAILABLE:
                self.logger.warning(f"⚠️ Database optimization requested but not available, using fallback parameters")
            else:
                self.logger.info(f"📊 Using static parameters for {strategy_name} backtests")
    
    @abstractmethod
    def initialize_strategy(self, epic: str = None):
        """Initialize the specific strategy with optional epic for optimal parameters"""
        pass
    
    @abstractmethod
    def run_strategy_backtest(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """Run backtest for this specific strategy"""
        pass
    
    def get_optimal_parameters(self, epic: str):
        """Get optimal parameters for an epic if optimization is enabled"""
        if not self.use_optimal_parameters or not self.optimal_service:
            return None
        
        try:
            params = self.optimal_service.get_epic_parameters(epic)
            self.logger.info(f"✅ Using optimal parameters for {epic}:")
            self.logger.info(f"   Config: {getattr(params, 'ema_config', 'N/A')}")
            self.logger.info(f"   Confidence: {params.confidence_threshold:.1%}")
            self.logger.info(f"   Timeframe: {params.timeframe}")
            self.logger.info(f"   SL/TP: {params.stop_loss_pips:.0f}/{params.take_profit_pips:.0f} pips")
            self.logger.info(f"   R:R: {params.risk_reward_ratio:.1f}")
            self.logger.info(f"   Performance: {params.performance_score:.3f}")
            return params
        except Exception as e:
            self.logger.warning(f"❌ Failed to get optimal parameters for {epic}: {e}")
            return None
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        **kwargs
    ) -> bool:
        """Main backtest execution with optimal parameter support"""
        
        epic_list = [epic] if epic else config.EPIC_LIST
        self.logger.info(f"🧪 Running {self.strategy_name} backtest")
        self.logger.info(f"   Epic(s): {epic_list}")
        self.logger.info(f"   Days: {days}, Timeframe: {timeframe}")
        self.logger.info(f"   Database optimization: {'✅ ENABLED' if self.use_optimal_parameters else '❌ DISABLED'}")
        
        try:
            all_signals = []
            
            for current_epic in epic_list:
                self.logger.info(f"\n📊 Processing {current_epic}")
                
                # Initialize strategy with epic for optimal parameters
                strategy = self.initialize_strategy(current_epic)
                
                # Get data - extract pair from epic (e.g., CS.D.EURUSD.MINI.IP -> EURUSD)
                pair = current_epic.split('.')[2] if '.' in current_epic else current_epic
                df = self.data_fetcher.get_enhanced_data(
                    epic=current_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=days * 24
                )
                
                if df is None or df.empty:
                    self.logger.warning(f"❌ No data for {current_epic}")
                    continue
                
                # Run strategy backtest
                signals = self.run_strategy_backtest(
                    df, current_epic, config.SPREAD_PIPS, timeframe
                )
                
                all_signals.extend(signals)
                self.logger.info(f"   Found {len(signals)} signals")
            
            # Analyze results
            if all_signals:
                self.logger.info(f"✅ Total signals found: {len(all_signals)}")
                
                if show_signals:
                    self.signal_analyzer.display_signal_list(all_signals)
                
                # Performance analysis
                metrics = self.performance_analyzer.analyze_performance(all_signals)
                self._display_performance(metrics)
                
                return True
            else:
                self.logger.warning("❌ No signals found")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            return False
    
    def _display_performance(self, metrics: Dict):
        """Display performance metrics"""
        self.logger.info(f"📈 {self.strategy_name} Performance:")
        self.logger.info(f"   Signals: {metrics.get('total_signals', 0)}")
        self.logger.info(f"   Avg Confidence: {metrics.get('avg_confidence', 0):.1%}")
        self.logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        self.logger.info(f"   Avg Profit: {metrics.get('avg_profit_pips', 0):.1f} pips")
