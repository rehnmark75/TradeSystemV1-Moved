# ============================================================================
# backtests/backtest_base.py
# ============================================================================

import logging
import pandas as pd
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.backtest.performance_analyzer import PerformanceAnalyzer
from core.backtest.signal_analyzer import SignalAnalyzer
try:
    import config
except ImportError:
    from forex_scanner import config


class BacktestBase(ABC):
    """Base class for all strategy backtests"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"backtest_{strategy_name}")
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
    
    @abstractmethod
    def initialize_strategy(self):
        """Initialize the specific strategy"""
        pass
    
    @abstractmethod
    def run_strategy_backtest(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """Run backtest for this specific strategy"""
        pass
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        **kwargs
    ) -> bool:
        """Main backtest execution"""
        
        epic_list = [epic] if epic else config.EPIC_LIST
        self.logger.info(f"🧪 Running {self.strategy_name} backtest")
        self.logger.info(f"   Epic(s): {epic_list}")
        self.logger.info(f"   Days: {days}, Timeframe: {timeframe}")
        
        try:
            # Initialize strategy
            strategy = self.initialize_strategy()
            
            all_signals = []
            
            for current_epic in epic_list:
                self.logger.info(f"📊 Processing {current_epic}")
                
                # Get data
                df = self.data_fetcher.fetch_enhanced_data(
                    epic=current_epic,
                    timeframe=timeframe,
                    lookback_hours=days * 24
                )
                
                if df.empty:
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
                    self.signal_analyzer.display_signals(all_signals)
                
                # Performance analysis
                metrics = self.performance_analyzer.analyze_signals(all_signals)
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
