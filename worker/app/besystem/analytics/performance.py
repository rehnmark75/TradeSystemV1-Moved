# ================================
# 8. analytics/performance.py
# ================================

from typing import Dict, List, Tuple
from core.data_structures import Portfolio, Trade
import numpy as np
from core.data_structures import TradeStatus

# ================================
# 8. PERFORMANCE ANALYTICS
# ================================

class PerformanceAnalytics:
    """Comprehensive performance analysis and reporting"""
    
    @staticmethod
    def calculate_metrics(portfolio: Portfolio) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not portfolio.closed_trades:
            return {}
        
        trades = portfolio.closed_trades
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl_pips > 0])
        losing_trades = len([t for t in trades if t.pnl_pips < 0])
        breakeven_trades = total_trades - winning_trades - losing_trades
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl_pips = sum(t.pnl_pips for t in trades)
        total_pnl_currency = sum(t.pnl_currency for t in trades)
        
        winning_pnl = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
        losing_pnl = sum(t.pnl_pips for t in trades if t.pnl_pips < 0)
        
        avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
        avg_loss = abs(losing_pnl / losing_trades) if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Drawdown calculation
        equity_curve = PerformanceAnalytics._calculate_equity_curve(portfolio)
        max_drawdown, max_drawdown_pct = PerformanceAnalytics._calculate_max_drawdown(equity_curve)
        
        # Trade duration analysis
        durations = [t.duration_minutes for t in trades]
        avg_duration = np.mean(durations) if durations else 0
        
        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = PerformanceAnalytics._calculate_consecutive_trades(trades)
        
        # Return on investment
        roi = (portfolio.current_balance - portfolio.initial_balance) / portfolio.initial_balance * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,
            'total_pnl_pips': total_pnl_pips,
            'total_pnl_currency': total_pnl_currency,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_duration_minutes': avg_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'roi_percent': roi,
            'sharpe_ratio': PerformanceAnalytics._calculate_sharpe_ratio(trades),
            'calmar_ratio': roi / max_drawdown_pct if max_drawdown_pct > 0 else 0
        }
    
    @staticmethod
    def _calculate_equity_curve(portfolio: Portfolio) -> List[float]:
        """Calculate equity curve over time"""
        equity_curve = [portfolio.initial_balance]
        running_balance = portfolio.initial_balance
        
        for trade in sorted(portfolio.closed_trades, key=lambda t: t.exit_time or t.entry_time):
            running_balance += trade.pnl_currency
            equity_curve.append(running_balance)
        
        return equity_curve
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0, 0
        
        peak = equity_curve[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        return max_drawdown, max_drawdown_pct
    
    @staticmethod
    def _calculate_consecutive_trades(trades: List[Trade]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in sorted(trades, key=lambda t: t.exit_time or t.entry_time):
            if trade.pnl_pips > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.pnl_pips < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # Breakeven
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    @staticmethod
    def _calculate_sharpe_ratio(trades: List[Trade]) -> float:
        """Calculate Sharpe ratio"""
        if len(trades) < 2:
            return 0
        
        returns = [t.pnl_currency for t in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assuming risk-free rate of 0 for simplicity
        return avg_return / std_return

    @staticmethod
    def generate_report(portfolio: Portfolio, epic_list: List[str]) -> str:
        """Generate comprehensive performance report"""
        metrics = PerformanceAnalytics.calculate_metrics(portfolio)
        
        if not metrics:
            return "No trades completed - cannot generate report"
        
        report = f"""
📊 BACKTESTING PERFORMANCE REPORT
{'=' * 50}

💰 PORTFOLIO SUMMARY
Initial Balance: ${portfolio.initial_balance:,.2f}
Final Balance: ${portfolio.current_balance:,.2f}
Total P&L: ${metrics['total_pnl_currency']:,.2f}
ROI: {metrics['roi_percent']:.2f}%

📈 TRADE STATISTICS
Total Trades: {metrics['total_trades']}
Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']:.1%})
Losing Trades: {metrics['losing_trades']} ({(1-metrics['win_rate']):.1%})
Breakeven Trades: {metrics['breakeven_trades']}

💎 PERFORMANCE METRICS
Average Win: {metrics['avg_win_pips']:.1f} pips
Average Loss: {metrics['avg_loss_pips']:.1f} pips
Profit Factor: {metrics['profit_factor']:.2f}
Expectancy: {metrics['expectancy']:.1f} pips
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

⚠️ RISK METRICS
Maximum Drawdown: ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)
Calmar Ratio: {metrics['calmar_ratio']:.2f}
Max Consecutive Wins: {metrics['max_consecutive_wins']}
Max Consecutive Losses: {metrics['max_consecutive_losses']}

⏰ TRADE DURATION
Average Duration: {metrics['avg_duration_minutes']:.0f} minutes

📋 EPIC ANALYSIS
Analyzed Epics: {', '.join(epic_list)}
"""
        
        # Epic-specific performance
        epic_performance = PerformanceAnalytics._analyze_epic_performance(portfolio.closed_trades)
        if epic_performance:
            report += "\n🎯 EPIC PERFORMANCE BREAKDOWN\n"
            for epic, perf in epic_performance.items():
                report += f"{epic}: {perf['trades']} trades, {perf['win_rate']:.1%} win rate, {perf['total_pips']:.1f} pips\n"
        
        return report
    
    @staticmethod
    def _analyze_epic_performance(trades: List[Trade]) -> Dict:
        """Analyze performance by epic"""
        epic_stats = {}
        
        for trade in trades:
            epic = trade.signal.epic
            if epic not in epic_stats:
                epic_stats[epic] = {'trades': 0, 'wins': 0, 'total_pips': 0}
            
            epic_stats[epic]['trades'] += 1
            epic_stats[epic]['total_pips'] += trade.pnl_pips
            
            if trade.pnl_pips > 0:
                epic_stats[epic]['wins'] += 1
        
        # Calculate win rates
        for epic, stats in epic_stats.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        
        return epic_stats