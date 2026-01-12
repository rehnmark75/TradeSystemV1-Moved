#!/usr/bin/env python3
"""
Comprehensive Pair Performance Analysis Script

Analyzes trading performance for any epic to identify:
- Session timing patterns
- Confidence score effectiveness
- Day of week patterns
- Direction bias
- Indicator correlations (ADX, RSI, EMA distance, MACD, Bollinger Bands)
- HTF candle alignment
- Price action after entry
- Combined filter recommendations

Usage:
    # Inside docker container:
    python /app/forex_scanner/scripts/analyze_pair_performance.py CS.D.USDJPY.MINI.IP
    python /app/forex_scanner/scripts/analyze_pair_performance.py CS.D.EURUSD.CEEM.IP --days 60
    python /app/forex_scanner/scripts/analyze_pair_performance.py CS.D.GBPUSD.MINI.IP --days 90 --output json

    # From host:
    docker exec -it task-worker python /app/forex_scanner/scripts/analyze_pair_performance.py CS.D.USDJPY.MINI.IP
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    epic: str
    analysis_date: str
    lookback_days: int
    overall_stats: Dict
    session_analysis: Dict
    confidence_analysis: Dict
    day_of_week_analysis: Dict
    direction_analysis: Dict
    adx_analysis: Dict
    rsi_analysis: Dict
    ema_distance_analysis: Dict
    macd_analysis: Dict
    bollinger_analysis: Dict
    htf_alignment_analysis: Dict
    htf_transition_analysis: Dict
    volume_analysis: Dict
    claude_score_analysis: Dict
    combined_filter_analysis: Dict
    price_action_analysis: Dict
    worst_trades: List[Dict]
    best_trades: List[Dict]
    recommendations: List[str]
    proposed_filters: Dict


class PairPerformanceAnalyzer:
    """Analyzes trading performance for a specific epic"""

    def __init__(self, epic: str, lookback_days: int = 90):
        self.epic = epic
        self.lookback_days = lookback_days
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the forex database"""
        self.conn = psycopg2.connect(
            host="postgres",
            database="forex",
            user="postgres",
            password="postgres"
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def _execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results as list of dicts"""
        self.cursor.execute(query, params or ())
        return [dict(row) for row in self.cursor.fetchall()]

    def get_overall_stats(self) -> Dict:
        """Get overall trading statistics"""
        query = f"""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as filled,
            SUM(CASE WHEN status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            SUM(CASE WHEN status = 'closed' AND profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN status = 'closed' AND profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN status = 'closed' AND profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN status = 'closed' THEN profit_loss ELSE 0 END)::numeric, 2) as total_pnl,
            ROUND(AVG(CASE WHEN status = 'closed' AND profit_loss > 0 THEN profit_loss END)::numeric, 2) as avg_win,
            ROUND(AVG(CASE WHEN status = 'closed' AND profit_loss < 0 THEN profit_loss END)::numeric, 2) as avg_loss
        FROM trade_log
        WHERE symbol = %s
          AND timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        """
        result = self._execute_query(query, (self.epic,))
        return result[0] if result else {}

    def get_session_analysis(self) -> List[Dict]:
        """Analyze performance by trading session"""
        query = f"""
        SELECT
            CASE
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 0 AND 6 THEN 'Asian (00-07)'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 7 AND 11 THEN 'London AM (07-12)'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 12 AND 15 THEN 'NY Overlap (12-16)'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 16 AND 20 THEN 'NY PM (16-21)'
                ELSE 'Late NY (21-24)'
            END as session,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        GROUP BY
            CASE
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 0 AND 6 THEN 'Asian (00-07)'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 7 AND 11 THEN 'London AM (07-12)'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 12 AND 15 THEN 'NY Overlap (12-16)'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 16 AND 20 THEN 'NY PM (16-21)'
                ELSE 'Late NY (21-24)'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_hourly_analysis(self) -> List[Dict]:
        """Analyze performance by hour"""
        query = f"""
        SELECT
            EXTRACT(HOUR FROM t.timestamp) as hour_utc,
            COUNT(*) as total_trades,
            SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl,
            ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl
        FROM trade_log t
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        GROUP BY EXTRACT(HOUR FROM t.timestamp)
        ORDER BY hour_utc
        """
        return self._execute_query(query, (self.epic,))

    def get_confidence_analysis(self) -> List[Dict]:
        """Analyze performance by confidence score buckets"""
        query = f"""
        SELECT
            CASE
                WHEN a.confidence_score < 0.50 THEN '< 50%%'
                WHEN a.confidence_score < 0.55 THEN '50-55%%'
                WHEN a.confidence_score < 0.60 THEN '55-60%%'
                WHEN a.confidence_score < 0.65 THEN '60-65%%'
                ELSE '65%%+'
            END as confidence_bucket,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        GROUP BY
            CASE
                WHEN a.confidence_score < 0.50 THEN '< 50%%'
                WHEN a.confidence_score < 0.55 THEN '50-55%%'
                WHEN a.confidence_score < 0.60 THEN '55-60%%'
                WHEN a.confidence_score < 0.65 THEN '60-65%%'
                ELSE '65%%+'
            END
        ORDER BY confidence_bucket
        """
        return self._execute_query(query, (self.epic,))

    def get_day_of_week_analysis(self) -> List[Dict]:
        """Analyze performance by day of week"""
        query = f"""
        SELECT
            EXTRACT(DOW FROM t.timestamp) as dow,
            TO_CHAR(t.timestamp, 'Day') as day_name,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        GROUP BY EXTRACT(DOW FROM t.timestamp), TO_CHAR(t.timestamp, 'Day')
        ORDER BY dow
        """
        return self._execute_query(query, (self.epic,))

    def get_direction_analysis(self) -> List[Dict]:
        """Analyze performance by trade direction"""
        query = f"""
        SELECT
            t.direction,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        GROUP BY t.direction
        """
        return self._execute_query(query, (self.epic,))

    def get_adx_analysis(self) -> List[Dict]:
        """Analyze performance by ADX strength"""
        query = f"""
        SELECT
            CASE
                WHEN a.adx < 15 THEN 'Weak (<15)'
                WHEN a.adx < 20 THEN 'Developing (15-20)'
                WHEN a.adx < 25 THEN 'Moderate (20-25)'
                WHEN a.adx < 30 THEN 'Strong (25-30)'
                ELSE 'Very Strong (30+)'
            END as adx_strength,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.adx IS NOT NULL
        GROUP BY
            CASE
                WHEN a.adx < 15 THEN 'Weak (<15)'
                WHEN a.adx < 20 THEN 'Developing (15-20)'
                WHEN a.adx < 25 THEN 'Moderate (20-25)'
                WHEN a.adx < 30 THEN 'Strong (25-30)'
                ELSE 'Very Strong (30+)'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_rsi_analysis(self) -> List[Dict]:
        """Analyze performance by RSI zone"""
        query = f"""
        SELECT
            CASE
                WHEN a.rsi < 30 THEN 'Oversold (<30)'
                WHEN a.rsi < 40 THEN 'Weak (30-40)'
                WHEN a.rsi < 50 THEN 'Bearish (40-50)'
                WHEN a.rsi < 60 THEN 'Bullish (50-60)'
                WHEN a.rsi < 70 THEN 'Strong (60-70)'
                ELSE 'Overbought (70+)'
            END as rsi_zone,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.rsi IS NOT NULL
        GROUP BY
            CASE
                WHEN a.rsi < 30 THEN 'Oversold (<30)'
                WHEN a.rsi < 40 THEN 'Weak (30-40)'
                WHEN a.rsi < 50 THEN 'Bearish (40-50)'
                WHEN a.rsi < 60 THEN 'Bullish (50-60)'
                WHEN a.rsi < 70 THEN 'Strong (60-70)'
                ELSE 'Overbought (70+)'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_ema_distance_analysis(self) -> List[Dict]:
        """Analyze performance by EMA distance"""
        query = f"""
        SELECT
            CASE
                WHEN ABS((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric) < 20 THEN 'Near EMA (<20 pips)'
                WHEN ABS((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric) < 50 THEN 'Moderate (20-50 pips)'
                WHEN ABS((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric) < 100 THEN 'Extended (50-100 pips)'
                ELSE 'Far (100+ pips)'
            END as ema_distance,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.strategy_indicators->'tier1_ema'->>'distance_pips' IS NOT NULL
        GROUP BY
            CASE
                WHEN ABS((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric) < 20 THEN 'Near EMA (<20 pips)'
                WHEN ABS((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric) < 50 THEN 'Moderate (20-50 pips)'
                WHEN ABS((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric) < 100 THEN 'Extended (50-100 pips)'
                ELSE 'Far (100+ pips)'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_macd_analysis(self) -> List[Dict]:
        """Analyze performance by MACD alignment"""
        query = f"""
        SELECT
            CASE
                WHEN t.direction = 'BUY' AND a.macd_histogram > 0 THEN 'BUY + MACD bullish'
                WHEN t.direction = 'BUY' AND a.macd_histogram <= 0 THEN 'BUY + MACD bearish'
                WHEN t.direction = 'SELL' AND a.macd_histogram < 0 THEN 'SELL + MACD bearish'
                WHEN t.direction = 'SELL' AND a.macd_histogram >= 0 THEN 'SELL + MACD bullish'
            END as macd_alignment,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.macd_histogram IS NOT NULL
        GROUP BY
            CASE
                WHEN t.direction = 'BUY' AND a.macd_histogram > 0 THEN 'BUY + MACD bullish'
                WHEN t.direction = 'BUY' AND a.macd_histogram <= 0 THEN 'BUY + MACD bearish'
                WHEN t.direction = 'SELL' AND a.macd_histogram < 0 THEN 'SELL + MACD bearish'
                WHEN t.direction = 'SELL' AND a.macd_histogram >= 0 THEN 'SELL + MACD bullish'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_bollinger_analysis(self) -> List[Dict]:
        """Analyze performance by Bollinger Band position"""
        query = f"""
        SELECT
            a.price_vs_bb as bb_position,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.price_vs_bb IS NOT NULL
        GROUP BY a.price_vs_bb
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_htf_alignment_analysis(self) -> List[Dict]:
        """Analyze performance by HTF candle alignment"""
        query = f"""
        SELECT
            CASE
                WHEN t.direction = 'BUY' AND a.htf_candle_direction = 'BULLISH' THEN 'BUY + Bullish HTF (aligned)'
                WHEN t.direction = 'BUY' AND a.htf_candle_direction = 'BEARISH' THEN 'BUY + Bearish HTF (misaligned)'
                WHEN t.direction = 'SELL' AND a.htf_candle_direction = 'BEARISH' THEN 'SELL + Bearish HTF (aligned)'
                WHEN t.direction = 'SELL' AND a.htf_candle_direction = 'BULLISH' THEN 'SELL + Bullish HTF (misaligned)'
                ELSE 'Unknown'
            END as htf_alignment,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.htf_candle_direction IS NOT NULL
        GROUP BY
            CASE
                WHEN t.direction = 'BUY' AND a.htf_candle_direction = 'BULLISH' THEN 'BUY + Bullish HTF (aligned)'
                WHEN t.direction = 'BUY' AND a.htf_candle_direction = 'BEARISH' THEN 'BUY + Bearish HTF (misaligned)'
                WHEN t.direction = 'SELL' AND a.htf_candle_direction = 'BEARISH' THEN 'SELL + Bearish HTF (aligned)'
                WHEN t.direction = 'SELL' AND a.htf_candle_direction = 'BULLISH' THEN 'SELL + Bullish HTF (misaligned)'
                ELSE 'Unknown'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_htf_transition_analysis(self) -> List[Dict]:
        """Analyze performance by HTF candle transitions"""
        query = f"""
        SELECT
            a.htf_candle_direction_prev || ' -> ' || a.htf_candle_direction as htf_transition,
            t.direction as trade_dir,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.htf_candle_direction IS NOT NULL
          AND a.htf_candle_direction_prev IS NOT NULL
        GROUP BY a.htf_candle_direction_prev || ' -> ' || a.htf_candle_direction, t.direction
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_volume_analysis(self) -> List[Dict]:
        """Analyze performance by volume ratio"""
        query = f"""
        SELECT
            CASE
                WHEN a.volume_ratio < 0.8 THEN 'Low Volume (<0.8)'
                WHEN a.volume_ratio < 1.0 THEN 'Below Avg (0.8-1.0)'
                WHEN a.volume_ratio < 1.3 THEN 'Average (1.0-1.3)'
                WHEN a.volume_ratio < 1.5 THEN 'Above Avg (1.3-1.5)'
                ELSE 'High Volume (1.5+)'
            END as volume_bucket,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN t.status = 'limit_not_filled' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.volume_ratio IS NOT NULL
        GROUP BY
            CASE
                WHEN a.volume_ratio < 0.8 THEN 'Low Volume (<0.8)'
                WHEN a.volume_ratio < 1.0 THEN 'Below Avg (0.8-1.0)'
                WHEN a.volume_ratio < 1.3 THEN 'Average (1.0-1.3)'
                WHEN a.volume_ratio < 1.5 THEN 'Above Avg (1.3-1.5)'
                ELSE 'High Volume (1.5+)'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_claude_score_analysis(self) -> List[Dict]:
        """Analyze performance by Claude score"""
        query = f"""
        SELECT
            CASE
                WHEN a.claude_score < 5 THEN 'Low (<5)'
                WHEN a.claude_score < 7 THEN 'Medium (5-6)'
                WHEN a.claude_score < 8 THEN 'Good (7)'
                ELSE 'High (8+)'
            END as claude_score_bucket,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND a.claude_score IS NOT NULL
        GROUP BY
            CASE
                WHEN a.claude_score < 5 THEN 'Low (<5)'
                WHEN a.claude_score < 7 THEN 'Medium (5-6)'
                WHEN a.claude_score < 8 THEN 'Good (7)'
                ELSE 'High (8+)'
            END
        ORDER BY pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_combined_filter_analysis(self) -> List[Dict]:
        """Analyze performance by combined filters (session + confidence + direction)"""
        query = f"""
        SELECT
            CASE
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 0 AND 6 THEN 'Asian'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 7 AND 11 THEN 'London'
                ELSE 'NY'
            END as session,
            CASE WHEN a.confidence_score >= 0.60 THEN 'High Conf' ELSE 'Low Conf' END as conf_level,
            t.direction,
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        GROUP BY
            CASE
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 0 AND 6 THEN 'Asian'
                WHEN EXTRACT(HOUR FROM t.timestamp) BETWEEN 7 AND 11 THEN 'London'
                ELSE 'NY'
            END,
            CASE WHEN a.confidence_score >= 0.60 THEN 'High Conf' ELSE 'Low Conf' END,
            t.direction
        HAVING SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END) >= 2
        ORDER BY win_rate DESC NULLS LAST, pnl DESC
        """
        return self._execute_query(query, (self.epic,))

    def get_price_action_analysis(self) -> List[Dict]:
        """Analyze price action after entry for losing trades"""
        query = f"""
        WITH loss_trades AS (
            SELECT
                t.id,
                t.timestamp as entry_time,
                t.entry_price,
                t.direction,
                t.sl_price,
                t.profit_loss
            FROM trade_log t
            WHERE t.symbol = %s
              AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
              AND t.status = 'closed'
              AND t.profit_loss < 0
        )
        SELECT
            lt.id,
            lt.direction,
            ROUND(lt.entry_price::numeric, 5) as entry,
            ROUND(lt.sl_price::numeric, 5) as sl,
            ROUND((SELECT MAX(high) FROM ig_candles
                   WHERE epic = '{self.epic}'
                     AND timeframe = 1
                     AND start_time BETWEEN lt.entry_time AND lt.entry_time + INTERVAL '15 minutes')::numeric, 5) as max_high_15m,
            ROUND((SELECT MIN(low) FROM ig_candles
                   WHERE epic = '{self.epic}'
                     AND timeframe = 1
                     AND start_time BETWEEN lt.entry_time AND lt.entry_time + INTERVAL '15 minutes')::numeric, 5) as min_low_15m,
            CASE
                WHEN lt.direction = 'BUY' THEN
                    ROUND((((SELECT MAX(high) FROM ig_candles
                           WHERE epic = '{self.epic}'
                             AND timeframe = 1
                             AND start_time BETWEEN lt.entry_time AND lt.entry_time + INTERVAL '15 minutes') - lt.entry_price) * 100)::numeric, 1)
                ELSE
                    ROUND(((lt.entry_price - (SELECT MIN(low) FROM ig_candles
                           WHERE epic = '{self.epic}'
                             AND timeframe = 1
                             AND start_time BETWEEN lt.entry_time AND lt.entry_time + INTERVAL '15 minutes')) * 100)::numeric, 1)
            END as max_favorable_pips_15m,
            ROUND(lt.profit_loss::numeric, 2) as pnl
        FROM loss_trades lt
        ORDER BY lt.profit_loss ASC
        LIMIT 15
        """
        return self._execute_query(query, (self.epic,))

    def get_worst_trades(self) -> List[Dict]:
        """Get worst performing trades with details"""
        query = f"""
        SELECT
            t.id,
            t.timestamp::date as date,
            EXTRACT(HOUR FROM t.timestamp) as hour,
            t.direction,
            ROUND(a.confidence_score::numeric, 2) as conf,
            ROUND(a.adx::numeric, 1) as adx,
            ROUND(a.rsi::numeric, 1) as rsi,
            ROUND((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric, 1) as ema_dist,
            a.htf_candle_direction as htf_dir,
            a.htf_candle_direction_prev as htf_prev,
            ROUND(t.profit_loss::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND t.status = 'closed'
          AND t.profit_loss < 0
        ORDER BY t.profit_loss ASC
        LIMIT 15
        """
        return self._execute_query(query, (self.epic,))

    def get_best_trades(self) -> List[Dict]:
        """Get best performing trades with details"""
        query = f"""
        SELECT
            t.id,
            t.timestamp::date as date,
            EXTRACT(HOUR FROM t.timestamp) as hour,
            t.direction,
            ROUND(a.confidence_score::numeric, 2) as conf,
            ROUND(a.adx::numeric, 1) as adx,
            ROUND(a.rsi::numeric, 1) as rsi,
            ROUND((a.strategy_indicators->'tier1_ema'->>'distance_pips')::numeric, 1) as ema_dist,
            a.htf_candle_direction as htf_dir,
            a.htf_candle_direction_prev as htf_prev,
            ROUND(t.profit_loss::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND t.status = 'closed'
          AND t.profit_loss > 0
        ORDER BY t.profit_loss DESC
        LIMIT 15
        """
        return self._execute_query(query, (self.epic,))

    def get_proposed_filters(self) -> Dict:
        """Calculate proposed filter impact"""
        # Find best session
        session_data = self.get_session_analysis()
        best_session = max(session_data, key=lambda x: float(x.get('pnl') or -999999)) if session_data else None

        query = f"""
        SELECT
            'PROPOSED FILTER' as scenario,
            COUNT(*) as total_signals,
            SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END) as filled,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
          AND EXTRACT(HOUR FROM t.timestamp) BETWEEN 0 AND 7
          AND a.confidence_score >= 0.60

        UNION ALL

        SELECT
            'CURRENT (ALL)' as scenario,
            COUNT(*) as total_signals,
            SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END) as filled,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.status = 'closed' AND t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN t.status = 'closed' AND t.profit_loss > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0), 1) as win_rate,
            ROUND(SUM(CASE WHEN t.status = 'closed' THEN t.profit_loss ELSE 0 END)::numeric, 2) as pnl
        FROM trade_log t
        JOIN alert_history a ON t.alert_id = a.id
        WHERE t.symbol = %s
          AND t.timestamp >= NOW() - INTERVAL '{self.lookback_days} days'
        """
        results = self._execute_query(query, (self.epic, self.epic))
        return {r['scenario']: r for r in results}

    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Session recommendations
        session_data = results.get('session_analysis', [])
        if session_data:
            best_session = max(session_data, key=lambda x: float(x.get('pnl') or -999999))
            worst_session = min(session_data, key=lambda x: float(x.get('pnl') or 999999))

            if best_session.get('pnl') and float(best_session['pnl']) > 0:
                recommendations.append(
                    f"BEST SESSION: {best_session['session']} with {best_session['win_rate']}% win rate and +{best_session['pnl']} P&L"
                )
            if worst_session.get('pnl') and float(worst_session['pnl']) < -100:
                recommendations.append(
                    f"AVOID SESSION: {worst_session['session']} losing {worst_session['pnl']} - consider blocking"
                )

        # Confidence recommendations
        conf_data = results.get('confidence_analysis', [])
        for bucket in conf_data:
            if bucket.get('win_rate') and float(bucket['win_rate']) < 20 and bucket.get('total', 0) >= 5:
                recommendations.append(
                    f"LOW WIN RATE: {bucket['confidence_bucket']} confidence has only {bucket['win_rate']}% win rate - raise minimum confidence"
                )

        # Day of week recommendations
        dow_data = results.get('day_of_week_analysis', [])
        for day in dow_data:
            if day.get('win_rate') and float(day['win_rate']) < 25 and day.get('total', 0) >= 5:
                recommendations.append(
                    f"WEAK DAY: {day['day_name'].strip()} has {day['win_rate']}% win rate - consider blocking"
                )

        # Direction bias
        dir_data = results.get('direction_analysis', [])
        if len(dir_data) == 2:
            buy = next((d for d in dir_data if d['direction'] == 'BUY'), None)
            sell = next((d for d in dir_data if d['direction'] == 'SELL'), None)
            if buy and sell:
                buy_wr = float(buy.get('win_rate') or 0)
                sell_wr = float(sell.get('win_rate') or 0)
                if abs(buy_wr - sell_wr) > 15:
                    better = 'BUY' if buy_wr > sell_wr else 'SELL'
                    recommendations.append(
                        f"DIRECTION BIAS: {better} signals significantly outperform - consider direction filter"
                    )

        # Bollinger Band
        bb_data = results.get('bollinger_analysis', [])
        for bb in bb_data:
            if bb.get('bb_position') == 'in_band' and bb.get('win_rate') and float(bb['win_rate']) > 60:
                recommendations.append(
                    f"BB FILTER: Trades within Bollinger Bands have {bb['win_rate']}% win rate - filter to in_band only"
                )

        # Proposed filter impact
        proposed = results.get('proposed_filters', {})
        if 'PROPOSED FILTER' in proposed and 'CURRENT (ALL)' in proposed:
            current = proposed['CURRENT (ALL)']
            filtered = proposed['PROPOSED FILTER']
            if current.get('win_rate') and filtered.get('win_rate'):
                improvement = float(filtered['win_rate']) - float(current['win_rate'])
                if improvement > 10:
                    recommendations.append(
                        f"FILTER IMPACT: Asian session + 60%+ confidence would improve win rate from {current['win_rate']}% to {filtered['win_rate']}%"
                    )

        return recommendations

    def run_analysis(self) -> AnalysisResult:
        """Run complete analysis and return results"""
        self.connect()

        try:
            results = {
                'overall_stats': self.get_overall_stats(),
                'session_analysis': self.get_session_analysis(),
                'confidence_analysis': self.get_confidence_analysis(),
                'day_of_week_analysis': self.get_day_of_week_analysis(),
                'direction_analysis': self.get_direction_analysis(),
                'adx_analysis': self.get_adx_analysis(),
                'rsi_analysis': self.get_rsi_analysis(),
                'ema_distance_analysis': self.get_ema_distance_analysis(),
                'macd_analysis': self.get_macd_analysis(),
                'bollinger_analysis': self.get_bollinger_analysis(),
                'htf_alignment_analysis': self.get_htf_alignment_analysis(),
                'htf_transition_analysis': self.get_htf_transition_analysis(),
                'volume_analysis': self.get_volume_analysis(),
                'claude_score_analysis': self.get_claude_score_analysis(),
                'combined_filter_analysis': self.get_combined_filter_analysis(),
                'price_action_analysis': self.get_price_action_analysis(),
                'worst_trades': self.get_worst_trades(),
                'best_trades': self.get_best_trades(),
                'proposed_filters': self.get_proposed_filters(),
            }

            recommendations = self.generate_recommendations(results)

            return AnalysisResult(
                epic=self.epic,
                analysis_date=datetime.now().isoformat(),
                lookback_days=self.lookback_days,
                recommendations=recommendations,
                **results
            )
        finally:
            self.disconnect()


def format_table(data: List[Dict], columns: List[str], title: str = None) -> str:
    """Format data as a simple ASCII table"""
    if not data:
        return f"\n{title}\nNo data available\n" if title else "No data available\n"

    # Calculate column widths
    widths = {}
    for col in columns:
        max_width = len(col)
        for row in data:
            val = str(row.get(col, ''))
            max_width = max(max_width, len(val))
        widths[col] = max_width + 2

    # Build table
    lines = []
    if title:
        lines.append(f"\n{'=' * 60}")
        lines.append(f" {title}")
        lines.append('=' * 60)

    # Header
    header = '|'.join(col.center(widths[col]) for col in columns)
    separator = '+'.join('-' * widths[col] for col in columns)
    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    # Rows
    for row in data:
        line = '|'.join(str(row.get(col, '')).center(widths[col]) for col in columns)
        lines.append(line)

    lines.append(separator)
    return '\n'.join(lines)


def print_analysis(result: AnalysisResult):
    """Print analysis results in a readable format"""
    print("\n" + "=" * 80)
    print(f" PAIR PERFORMANCE ANALYSIS: {result.epic}")
    print(f" Analysis Date: {result.analysis_date}")
    print(f" Lookback Period: {result.lookback_days} days")
    print("=" * 80)

    # Overall Stats
    stats = result.overall_stats
    print("\n" + "-" * 40)
    print(" OVERALL STATISTICS")
    print("-" * 40)
    print(f" Total Trades:    {stats.get('total_trades', 0)}")
    print(f" Filled:          {stats.get('filled', 0)}")
    print(f" Expired:         {stats.get('expired', 0)}")
    print(f" Wins:            {stats.get('wins', 0)}")
    print(f" Losses:          {stats.get('losses', 0)}")
    print(f" Win Rate:        {stats.get('win_rate', 0)}%")
    print(f" Total P&L:       {stats.get('total_pnl', 0)}")
    print(f" Avg Win:         {stats.get('avg_win', 0)}")
    print(f" Avg Loss:        {stats.get('avg_loss', 0)}")

    # Session Analysis
    print(format_table(
        result.session_analysis,
        ['session', 'total', 'wins', 'losses', 'expired', 'win_rate', 'pnl'],
        'SESSION ANALYSIS'
    ))

    # Confidence Analysis
    print(format_table(
        result.confidence_analysis,
        ['confidence_bucket', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
        'CONFIDENCE SCORE ANALYSIS'
    ))

    # Day of Week
    print(format_table(
        result.day_of_week_analysis,
        ['day_name', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
        'DAY OF WEEK ANALYSIS'
    ))

    # Direction
    print(format_table(
        result.direction_analysis,
        ['direction', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
        'DIRECTION ANALYSIS'
    ))

    # ADX Analysis
    if result.adx_analysis:
        print(format_table(
            result.adx_analysis,
            ['adx_strength', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'ADX STRENGTH ANALYSIS'
        ))

    # RSI Analysis
    if result.rsi_analysis:
        print(format_table(
            result.rsi_analysis,
            ['rsi_zone', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'RSI ZONE ANALYSIS'
        ))

    # EMA Distance
    if result.ema_distance_analysis:
        print(format_table(
            result.ema_distance_analysis,
            ['ema_distance', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'EMA DISTANCE ANALYSIS'
        ))

    # MACD Analysis
    if result.macd_analysis:
        print(format_table(
            result.macd_analysis,
            ['macd_alignment', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'MACD ALIGNMENT ANALYSIS'
        ))

    # Bollinger Analysis
    if result.bollinger_analysis:
        print(format_table(
            result.bollinger_analysis,
            ['bb_position', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'BOLLINGER BAND ANALYSIS'
        ))

    # HTF Alignment
    if result.htf_alignment_analysis:
        print(format_table(
            result.htf_alignment_analysis,
            ['htf_alignment', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'HTF CANDLE ALIGNMENT'
        ))

    # HTF Transitions
    if result.htf_transition_analysis:
        print(format_table(
            result.htf_transition_analysis,
            ['htf_transition', 'trade_dir', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'HTF CANDLE TRANSITIONS'
        ))

    # Volume Analysis
    if result.volume_analysis:
        print(format_table(
            result.volume_analysis,
            ['volume_bucket', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'VOLUME RATIO ANALYSIS'
        ))

    # Claude Score
    if result.claude_score_analysis:
        print(format_table(
            result.claude_score_analysis,
            ['claude_score_bucket', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'CLAUDE SCORE ANALYSIS'
        ))

    # Combined Filter Analysis
    if result.combined_filter_analysis:
        print(format_table(
            result.combined_filter_analysis,
            ['session', 'conf_level', 'direction', 'total', 'wins', 'losses', 'win_rate', 'pnl'],
            'COMBINED FILTER ANALYSIS (Session + Confidence + Direction)'
        ))

    # Price Action Analysis
    if result.price_action_analysis:
        print(format_table(
            result.price_action_analysis,
            ['id', 'direction', 'entry', 'max_favorable_pips_15m', 'pnl'],
            'PRICE ACTION AFTER ENTRY (Losses)'
        ))

    # Worst Trades
    if result.worst_trades:
        print(format_table(
            result.worst_trades,
            ['id', 'date', 'hour', 'direction', 'conf', 'htf_dir', 'pnl'],
            'WORST TRADES'
        ))

    # Best Trades
    if result.best_trades:
        print(format_table(
            result.best_trades,
            ['id', 'date', 'hour', 'direction', 'conf', 'htf_dir', 'pnl'],
            'BEST TRADES'
        ))

    # Proposed Filters
    if result.proposed_filters:
        print("\n" + "=" * 60)
        print(" FILTER IMPACT SIMULATION")
        print("=" * 60)
        for scenario, data in result.proposed_filters.items():
            print(f"\n {scenario}:")
            print(f"   Trades: {data.get('total_signals', 0)} -> Filled: {data.get('filled', 0)}")
            print(f"   Win Rate: {data.get('win_rate', 0)}%")
            print(f"   P&L: {data.get('pnl', 0)}")

    # Recommendations
    print("\n" + "=" * 60)
    print(" RECOMMENDATIONS")
    print("=" * 60)
    for i, rec in enumerate(result.recommendations, 1):
        print(f" {i}. {rec}")

    if not result.recommendations:
        print(" No specific recommendations - review the data above for patterns")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze trading performance for a specific epic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_pair_performance.py CS.D.USDJPY.MINI.IP
    python analyze_pair_performance.py CS.D.EURUSD.CEEM.IP --days 60
    python analyze_pair_performance.py CS.D.GBPUSD.MINI.IP --output json
        """
    )
    parser.add_argument('epic', help='Epic to analyze (e.g., CS.D.USDJPY.MINI.IP)')
    parser.add_argument('--days', type=int, default=90, help='Lookback period in days (default: 90)')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='Output format')

    args = parser.parse_args()

    try:
        analyzer = PairPerformanceAnalyzer(args.epic, args.days)
        result = analyzer.run_analysis()

        if args.output == 'json':
            # Convert to dict for JSON output
            output = asdict(result)
            # Convert Decimal types to float for JSON serialization
            def convert_decimals(obj):
                if isinstance(obj, dict):
                    return {k: convert_decimals(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_decimals(i) for i in obj]
                elif hasattr(obj, '__float__'):
                    return float(obj)
                return obj

            print(json.dumps(convert_decimals(output), indent=2))
        else:
            print_analysis(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
