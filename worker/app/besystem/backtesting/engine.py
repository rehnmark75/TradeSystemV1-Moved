# ================================
# 9. backtesting/engine.py
# ================================

from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from core.data_structures import Portfolio, Trade, TradeStatus, Signal, SignalType  # KEY: All needed
from core.config import EpicConfig
from data.manager import DataManager
from analysis.technical import TechnicalAnalysis
from analysis.signals import SignalDetector
from trading.risk_manager import RiskManager
from trading.executor import TradeExecutor
from analytics.performance import PerformanceAnalytics

# ================================
# 9. MAIN BACKTESTING ENGINE
# ================================

class BacktestEngine:
    """Main backtesting engine that orchestrates all components"""
    
    def __init__(self, engine, initial_balance: float = 10000, 
                 user_timezone: str = 'Europe/Stockholm'):
        self.db_engine = engine
        self.data_manager = DataManager(engine, user_timezone)
        self.technical_analysis = TechnicalAnalysis()
        self.signal_detector = SignalDetector(self.technical_analysis)
        self.risk_manager = RiskManager()
        self.trade_executor = TradeExecutor(self.risk_manager)
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_balance=initial_balance,
            current_balance=initial_balance,
            equity=initial_balance,
            margin_used=0,
            free_margin=initial_balance
        )
    
    def run_backtest(self, epic_list: List[str], 
                    start_date: datetime = None, 
                    end_date: datetime = None,
                    timeframe: int = 5) -> Dict:
        """Run complete backtest"""
        
        print(f"🚀 Starting Backtest for {len(epic_list)} epics")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"⏰ Timeframe: {timeframe}m")
        print(f"💰 Initial Balance: ${self.portfolio.initial_balance:,.2f}")
        print("=" * 60)
        
        # Fetch and enhance data for all epics
        enhanced_data = {}
        for epic in epic_list:
            try:
                print(f"📊 Loading data for {epic}...")
                df = self.data_manager.fetch_candle_data(epic, timeframe, lookback_hours=2000)
                
                if df is None or len(df) == 0:
                    print(f"❌ No data for {epic}")
                    continue
                
                # Filter by date range if specified
                if start_date:
                    df = df[df['start_time'] >= start_date]
                if end_date:
                    df = df[df['start_time'] <= end_date]
                
                if len(df) < 200:
                    print(f"❌ Insufficient data for {epic} ({len(df)} bars)")
                    continue
                
                # Enhance with technical analysis
                pair = EpicConfig.extract_pair_from_epic(epic)
                enhanced_df = self.technical_analysis.enhance_dataframe(df, pair)
                enhanced_data[epic] = enhanced_df
                
                print(f"✅ {epic}: {len(enhanced_df)} bars loaded and enhanced")
                
            except Exception as e:
                print(f"❌ Error loading {epic}: {e}")
                continue
        
        if not enhanced_data:
            return {"error": "No valid data loaded for any epic"}
        
        print(f"\n🔄 Running backtest simulation...")
        
        # Get all unique timestamps and sort them
        all_timestamps = set()
        for df in enhanced_data.values():
            all_timestamps.update(df['start_time'].tolist())
        
        sorted_timestamps = sorted(all_timestamps)
        
        # Reset daily P&L at start of each day
        current_date = None
        
        # Main backtest loop
        signals_generated = 0
        trades_executed = 0
        
        for i, timestamp in enumerate(sorted_timestamps):
            # Reset daily P&L at start of new day
            if current_date != timestamp.date():
                current_date = timestamp.date()
                self.portfolio.daily_pnl = 0
            
            # Get current market data for this timestamp
            current_data = {}
            for epic, df in enhanced_data.items():
                mask = df['start_time'] <= timestamp
                if mask.any():
                    current_data[epic] = df[mask]
            
            # Update existing trades
            self.trade_executor.update_open_trades(self.portfolio, current_data)
            
            # Check for new signals
            for epic, df in current_data.items():
                if len(df) < 200:  # Need sufficient data for EMAs
                    continue
                
                signal = self.signal_detector.detect_ema_signals(df, epic)
                
                if signal:
                    signals_generated += 1
                    print(f"🔔 {signal.signal_type.value} signal: {epic} at {timestamp} (Confidence: {signal.confidence_score:.1%})")
                    
                    # Execute trade
                    trade = self.trade_executor.execute_signal(signal, self.portfolio)
                    
                    if trade:
                        trades_executed += 1
                        print(f"✅ Trade executed: {trade.trade_id}")
                    else:
                        print(f"❌ Trade rejected by risk management")
            
            # Progress update
            if i % 1000 == 0:
                progress = (i / len(sorted_timestamps)) * 100
                print(f"Progress: {progress:.1f}% - Balance: ${self.portfolio.current_balance:.2f} - Open: {len(self.portfolio.open_trades)}")
        
        # Close any remaining open trades at the end
        print(f"\n🔒 Closing {len(self.portfolio.open_trades)} remaining open trades...")
        for trade in self.portfolio.open_trades[:]:
            final_data = enhanced_data.get(trade.signal.epic)
            if final_data is not None and len(final_data) > 0:
                final_bar = final_data.iloc[-1]
                trade.exit_price = final_bar['close']
                trade.status = TradeStatus.CLOSED_BE
                self.trade_executor._close_trade(trade, final_bar, self.portfolio)
        
        self.portfolio.open_trades.clear()
        
        print(f"\n✅ Backtest Complete!")
        print(f"Signals Generated: {signals_generated}")
        print(f"Trades Executed: {trades_executed}")
        print(f"Final Balance: ${self.portfolio.current_balance:.2f}")
        
        # Generate performance report
        report = PerformanceAnalytics.generate_report(self.portfolio, epic_list)
        metrics = PerformanceAnalytics.calculate_metrics(self.portfolio)
        
        return {
            'portfolio': self.portfolio,
            'metrics': metrics,
            'report': report,
            'signals_generated': signals_generated,
            'trades_executed': trades_executed,
            'enhanced_data': enhanced_data
        }
    
    def optimize_parameters(self, epic_list: List[str], 
                          parameter_ranges: Dict) -> Dict:
        """Run parameter optimization"""
        print("🔬 Starting Parameter Optimization...")
        
        best_result = None
        best_score = float('-inf')
        optimization_results = []
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Apply parameters
            if 'max_risk_per_trade' in params:
                self.risk_manager.max_risk_per_trade = params['max_risk_per_trade']
            if 'stop_loss_multiplier' in params:
                # Would need to implement this in risk manager
                pass
            
            # Run backtest with these parameters
            result = self.run_backtest(epic_list)
            
            if 'metrics' in result:
                # Score based on Sharpe ratio and profit factor
                score = (result['metrics'].get('sharpe_ratio', 0) * 
                        result['metrics'].get('profit_factor', 0))
                
                optimization_results.append({
                    'parameters': params,
                    'score': score,
                    'metrics': result['metrics']
                })
                
                if score > best_score:
                    best_score = score
                    best_result = result
            
            # Reset portfolio for next iteration
            self.portfolio = Portfolio(
                initial_balance=self.portfolio.initial_balance,
                current_balance=self.portfolio.initial_balance,
                equity=self.portfolio.initial_balance,
                margin_used=0,
                free_margin=self.portfolio.initial_balance
            )
        
        return {
            'best_result': best_result,
            'best_score': best_score,
            'all_results': optimization_results
        }