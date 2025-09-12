# core/backtest/performance_analyzer.py
"""
Performance Analyzer
Analyzes performance metrics of trading signals
"""

from typing import List, Dict
import logging


class PerformanceAnalyzer:
    """Analyzes performance of historical signals"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance(self, signals: List[Dict]) -> Dict:
        """Analyze performance of historical signals"""
        if not signals:
            return {}
        
        total_signals = len(signals)
        bull_signals = [s for s in signals if s['signal_type'] == 'BULL']
        bear_signals = [s for s in signals if s['signal_type'] == 'BEAR']
        
        # Calculate metrics
        avg_confidence = sum(s.get('confidence_score', 0) for s in signals) / total_signals
        
        # Performance metrics (if available)
        signals_with_performance = [s for s in signals if 'max_profit_pips' in s]
        
        if signals_with_performance:
            avg_profit = sum(s['max_profit_pips'] for s in signals_with_performance) / len(signals_with_performance)
            avg_loss = sum(s['max_loss_pips'] for s in signals_with_performance) / len(signals_with_performance)
            
            # Win rate (assuming 20 pip target, 10 pip stop)
            profit_target = 20
            stop_loss = 10
            
            winners = [s for s in signals_with_performance if s['max_profit_pips'] >= profit_target]
            losers = [s for s in signals_with_performance if s['max_loss_pips'] >= stop_loss]
            
            win_rate = len(winners) / len(signals_with_performance) if signals_with_performance else 0
        else:
            avg_profit = avg_loss = win_rate = 0
        
        # Analyze by strategy
        strategy_breakdown = {}
        for signal in signals:
            strategy = signal.get('strategy', 'unknown')
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = {'count': 0, 'bull': 0, 'bear': 0}
            strategy_breakdown[strategy]['count'] += 1
            if signal['signal_type'] == 'BULL':
                strategy_breakdown[strategy]['bull'] += 1
            else:
                strategy_breakdown[strategy]['bear'] += 1
        
        performance = {
            'total_signals': total_signals,
            'bull_signals': len(bull_signals),
            'bear_signals': len(bear_signals),
            'average_confidence': avg_confidence,
            'average_profit_pips': avg_profit,
            'average_loss_pips': avg_loss,
            'win_rate': win_rate,
            'signals_with_performance': len(signals_with_performance),
            'strategy_breakdown': strategy_breakdown
        }
        
        # Log performance summary
        self.logger.info("📈 Performance Analysis:")
        self.logger.info(f"  Total signals: {total_signals}")
        self.logger.info(f"  Bull/Bear: {len(bull_signals)}/{len(bear_signals)}")
        self.logger.info(f"  Avg confidence: {avg_confidence:.1%}")
        if signals_with_performance:
            self.logger.info(f"  Avg profit: {avg_profit:.1f} pips")
            self.logger.info(f"  Avg loss: {avg_loss:.1f} pips")
            self.logger.info(f"  Win rate: {win_rate:.1%}")
        
        # Log strategy breakdown
        self.logger.info("📊 Strategy Breakdown:")
        for strategy, stats in strategy_breakdown.items():
            self.logger.info(f"  {strategy}: {stats['count']} signals ({stats['bull']} bull, {stats['bear']} bear)")
        
        return performance
    
    def calculate_sharpe_ratio(self, signals: List[Dict], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for signal performance"""
        if not signals:
            return 0.0
        
        # Calculate returns for each signal
        returns = []
        for signal in signals:
            if 'max_profit_pips' in signal and 'max_loss_pips' in signal:
                # Simplified: assume we exit at target or stop
                profit_target = 20
                stop_loss = 10
                
                if signal['max_profit_pips'] >= profit_target:
                    returns.append(profit_target)
                elif signal['max_loss_pips'] >= stop_loss:
                    returns.append(-stop_loss)
                else:
                    # Exit at end of period
                    returns.append(signal['max_profit_pips'] - signal['max_loss_pips'])
        
        if not returns:
            return 0.0
        
        import numpy as np
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily signals)
        sharpe = (avg_return - risk_free_rate/252) / std_return * np.sqrt(252)
        
        return sharpe
    
    def calculate_maximum_drawdown(self, signals: List[Dict]) -> float:
        """Calculate maximum drawdown from signal sequence"""
        if not signals:
            return 0.0
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x['timestamp'])
        
        # Calculate cumulative returns
        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        
        for signal in sorted_signals:
            if 'max_profit_pips' in signal and 'max_loss_pips' in signal:
                # Simplified P&L calculation
                profit_target = 20
                stop_loss = 10
                
                if signal['max_profit_pips'] >= profit_target:
                    pnl = profit_target
                elif signal['max_loss_pips'] >= stop_loss:
                    pnl = -stop_loss
                else:
                    pnl = 0  # Breakeven
                
                cumulative_pnl += pnl
                
                # Update peak
                if cumulative_pnl > peak_pnl:
                    peak_pnl = cumulative_pnl
                
                # Calculate drawdown
                drawdown = peak_pnl - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        return max_drawdown