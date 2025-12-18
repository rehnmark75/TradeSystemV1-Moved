"""
Unified Trading Analytics Dashboard
Combines dashboard, strategy analysis, and trade performance into a single tabbed interface.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import os

# Import centralized database utilities for connection pooling
from services.db_utils import DatabaseContextManager, get_connection_string

# Configure page
st.set_page_config(
    page_title="Trading Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for trading dashboard
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .profit-positive { color: #28a745; font-weight: bold; }
    .profit-negative { color: #dc3545; font-weight: bold; }
    .win-rate-high { color: #28a745; }
    .win-rate-medium { color: #ffc107; }
    .win-rate-low { color: #dc3545; }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-profitable { background-color: #28a745; }
    .status-losing { background-color: #dc3545; }
    .status-neutral { background-color: #6c757d; }
    .profit { color: #28a745; font-weight: bold; }
    .loss { color: #dc3545; font-weight: bold; }
    .pending { color: #ffc107; font-weight: bold; }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class TradingStatistics:
    """Data class for trading statistics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    pending_trades: int
    total_profit_loss: float
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_win_duration: float
    avg_loss_duration: float
    total_volume: float
    active_pairs: List[str]
    best_pair: str
    worst_pair: str

class UnifiedTradingDashboard:
    """Unified Trading Analytics Dashboard"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '7_days'
        if 'selected_pairs' not in st.session_state:
            st.session_state.selected_pairs = []
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Overview"

    def get_database_context(self):
        """Get database connection context manager using pooled connections."""
        return DatabaseContextManager("trading")

    def get_database_connection(self):
        """
        Get database connection from the connection pool.

        Note: The returned connection should be closed when done to return it to the pool.
        Prefer using get_database_context() for automatic connection management.
        """
        try:
            from services.db_utils import get_psycopg2_connection
            return get_psycopg2_connection("trading")
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None

    def fetch_trading_statistics(self, days_back: int = 7, pairs_filter: List[str] = None) -> Optional[TradingStatistics]:
        """Fetch comprehensive trading statistics from trade_log table"""
        conn = self.get_database_connection()
        if not conn:
            return None

        try:
            with conn.cursor() as cursor:
                # Base query with enhanced filtering
                base_query = """
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losing_trades,
                    COUNT(CASE WHEN profit_loss IS NULL OR status = 'pending' THEN 1 END) as pending_trades,
                    COALESCE(SUM(profit_loss), 0) as total_profit_loss,
                    COALESCE(AVG(CASE WHEN profit_loss > 0 THEN profit_loss END), 0) as avg_profit,
                    COALESCE(AVG(CASE WHEN profit_loss < 0 THEN profit_loss END), 0) as avg_loss,
                    COALESCE(MAX(profit_loss), 0) as largest_win,
                    COALESCE(MIN(profit_loss), 0) as largest_loss,
                    COUNT(DISTINCT symbol) as unique_pairs
                FROM trade_log
                WHERE timestamp >= %s
                """

                params = [datetime.now() - timedelta(days=days_back)]

                if pairs_filter:
                    base_query += " AND symbol = ANY(%s)"
                    params.append(pairs_filter)

                cursor.execute(base_query, params)
                result = cursor.fetchone()

                if not result or result[0] == 0:
                    return TradingStatistics(
                        total_trades=0, winning_trades=0, losing_trades=0, pending_trades=0,
                        total_profit_loss=0.0, win_rate=0.0, avg_profit=0.0, avg_loss=0.0,
                        profit_factor=0.0, largest_win=0.0, largest_loss=0.0,
                        avg_win_duration=0.0, avg_loss_duration=0.0, total_volume=0.0,
                        active_pairs=[], best_pair="None", worst_pair="None"
                    )

                # Calculate additional metrics
                total_trades, winning_trades, losing_trades, pending_trades = result[0:4]
                total_profit_loss, avg_profit, avg_loss, largest_win, largest_loss = result[4:9]

                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                profit_factor = abs(avg_profit * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')

                # Get pair-specific statistics
                pair_stats_query = """
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    COALESCE(SUM(profit_loss), 0) as total_pnl,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins
                FROM trade_log
                WHERE timestamp >= %s
                """

                if pairs_filter:
                    pair_stats_query += " AND symbol = ANY(%s)"

                pair_stats_query += " GROUP BY symbol ORDER BY total_pnl DESC"

                cursor.execute(pair_stats_query, params)
                pair_results = cursor.fetchall()

                active_pairs = [row[0] for row in pair_results]
                best_pair = pair_results[0][0] if pair_results else "None"
                worst_pair = pair_results[-1][0] if pair_results else "None"

                return TradingStatistics(
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    pending_trades=pending_trades,
                    total_profit_loss=float(total_profit_loss),
                    win_rate=win_rate,
                    avg_profit=float(avg_profit),
                    avg_loss=float(avg_loss),
                    profit_factor=profit_factor,
                    largest_win=float(largest_win),
                    largest_loss=float(largest_loss),
                    avg_win_duration=0.0,
                    avg_loss_duration=0.0,
                    total_volume=0.0,
                    active_pairs=active_pairs,
                    best_pair=best_pair,
                    worst_pair=worst_pair
                )

        except Exception as e:
            st.error(f"Error fetching trading statistics: {e}")
            return None
        finally:
            conn.close()

    def fetch_trades_dataframe(self, days_back: int = 7, pairs_filter: List[str] = None) -> pd.DataFrame:
        """Fetch detailed trades data as DataFrame"""
        conn = self.get_database_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                t.id, t.symbol, t.entry_price, t.direction, t.timestamp, t.status,
                t.profit_loss, t.pnl_currency, t.deal_id, t.sl_price, t.tp_price,
                t.closed_at, t.alert_id, a.strategy
            FROM trade_log t
            LEFT JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            """

            params = [datetime.now() - timedelta(days=days_back)]

            if pairs_filter:
                query += " AND t.symbol = ANY(%s)"
                params.append(pairs_filter)

            query += " ORDER BY t.timestamp DESC"

            df = pd.read_sql_query(query, conn, params=params)

            # Enhance DataFrame
            if not df.empty:
                df['trade_result'] = df['profit_loss'].apply(
                    lambda x: 'WIN' if pd.notna(x) and x > 0 else 'LOSS' if pd.notna(x) and x < 0 else 'PENDING'
                )
                df['profit_loss_formatted'] = df['profit_loss'].apply(
                    lambda x: f"{x:.2f} {df.iloc[0]['pnl_currency']}" if pd.notna(x) else "Pending"
                )

            return df

        except Exception as e:
            st.error(f"Error fetching trades data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def fetch_strategy_performance(self, days_back: int = 30) -> pd.DataFrame:
        """Fetch strategy performance data"""
        conn = self.get_database_connection()
        if not conn:
            return pd.DataFrame()

        try:
            strategy_query = """
            SELECT
                a.strategy,
                COUNT(t.*) as total_trades,
                COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
                COALESCE(SUM(t.profit_loss), 0) as total_pnl,
                COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
                COALESCE(AVG(a.confidence_score), 0) as avg_confidence,
                COALESCE(MAX(t.profit_loss), 0) as best_trade,
                COALESCE(MIN(t.profit_loss), 0) as worst_trade,
                COUNT(DISTINCT t.symbol) as pairs_traded
            FROM trade_log t
            INNER JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            GROUP BY a.strategy
            ORDER BY total_pnl DESC
            """

            df = pd.read_sql_query(
                strategy_query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)
                df['profit_factor'] = df.apply(
                    lambda row: (row['wins'] * abs(row['avg_pnl'])) / (row['losses'] * abs(row['avg_pnl']))
                    if row['losses'] > 0 and row['avg_pnl'] < 0 else float('inf'), axis=1
                )

            return df

        except Exception as e:
            st.error(f"Error fetching strategy performance: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def fetch_latest_closed_trade_id(self) -> int:
        """Fetch the ID of the most recent trade entry that is closed"""
        conn = self.get_database_connection()
        if not conn:
            return 1  # Default fallback

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM trade_log
                    WHERE status = 'closed'
                    ORDER BY id DESC
                    LIMIT 1
                """)
                result = cursor.fetchone()
                return result[0] if result else 1
        except Exception as e:
            logging.warning(f"Error fetching latest closed trade: {e}")
            return 1  # Default fallback
        finally:
            conn.close()

    def render_overview_tab(self):
        """Render the overview tab with key metrics and charts"""
        st.header("📊 Trading Overview")

        # Time period selector
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            timeframe = st.selectbox(
                "📅 Analysis Period",
                options=['1_day', '7_days', '30_days', '90_days'],
                format_func=lambda x: {
                    '1_day': '24 Hours',
                    '7_days': '7 Days',
                    '30_days': '30 Days',
                    '90_days': '90 Days'
                }[x],
                key='overview_timeframe',
                index=1  # Default to 7 days instead of 1 day
            )

        with col2:
            if st.button("🔄 Refresh Data", key="overview_refresh"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        with col3:
            show_debug = st.checkbox("🔍 Debug Mode", key="overview_debug")

        # Map timeframe to days
        timeframe_map = {'1_day': 1, '7_days': 7, '30_days': 30, '90_days': 90}
        days_back = timeframe_map[timeframe]

        # Fetch data
        stats = self.fetch_trading_statistics(days_back)
        trades_df = self.fetch_trades_dataframe(days_back)

        if not stats:
            st.error("Unable to fetch trading statistics. Please check database connection.")
            return

        # Key metrics
        self.render_key_metrics(stats)

        # Performance charts
        if not trades_df.empty:
            st.subheader("📈 Performance Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Cumulative P&L Chart
                if 'timestamp' in trades_df.columns and 'profit_loss' in trades_df.columns:
                    trades_sorted = trades_df[trades_df['profit_loss'].notna()].sort_values('timestamp')
                    trades_sorted['cumulative_pnl'] = trades_sorted['profit_loss'].cumsum()

                    fig_cumulative = px.line(
                        trades_sorted,
                        x='timestamp',
                        y='cumulative_pnl',
                        title="Cumulative P&L Over Time",
                        labels={'cumulative_pnl': 'Cumulative P&L (SEK)', 'timestamp': 'Date'}
                    )
                    fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_cumulative, use_container_width=True)

            with col2:
                # Win/Loss Pie Chart
                win_loss_data = pd.DataFrame({
                    'Result': ['Wins', 'Losses', 'Pending'],
                    'Count': [stats.winning_trades, stats.losing_trades, stats.pending_trades]
                })

                fig_pie = px.pie(
                    win_loss_data,
                    values='Count',
                    names='Result',
                    title="Trade Outcomes",
                    color='Result',
                    color_discrete_map={'Wins': '#28a745', 'Losses': '#dc3545', 'Pending': '#6c757d'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Recent trades summary
        if not trades_df.empty:
            st.subheader("📋 Recent Trades Summary")
            recent_trades = trades_df.head(10)

            display_columns = ['timestamp', 'symbol', 'strategy', 'direction', 'profit_loss_formatted', 'trade_result']
            display_columns = [col for col in display_columns if col in recent_trades.columns]

            st.dataframe(
                recent_trades[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="MM/DD HH:mm"),
                    "symbol": "Pair",
                    "strategy": "Strategy",
                    "direction": "Direction",
                    "profit_loss_formatted": "P&L",
                    "trade_result": "Result"
                }
            )

    def render_strategy_analysis_tab(self):
        """Render the strategy analysis tab"""
        st.header("🎯 Strategy Performance Analysis")

        # Controls
        col1, col2 = st.columns([2, 1])

        with col1:
            days_back = st.selectbox(
                "📅 Analysis Period",
                [7, 30, 90],
                index=0,
                format_func=lambda x: f"{x} days",
                key="strategy_days"
            )

        with col2:
            if st.button("🔄 Refresh", key="strategy_refresh"):
                st.rerun()

        # Fetch strategy data
        strategy_df = self.fetch_strategy_performance(days_back)

        if strategy_df.empty:
            st.warning(f"No strategy data found for the last {days_back} days")
            return

        # Strategy performance table
        st.subheader("📊 Strategy Performance Summary")

        st.dataframe(
            strategy_df[['strategy', 'total_trades', 'wins', 'losses', 'win_rate', 'total_pnl', 'avg_pnl', 'avg_confidence', 'pairs_traded']],
            column_config={
                "strategy": "Strategy",
                "total_trades": "Total Trades",
                "wins": "Wins",
                "losses": "Losses",
                "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                "total_pnl": st.column_config.NumberColumn("Total P&L", format="%.2f"),
                "avg_pnl": st.column_config.NumberColumn("Avg P&L", format="%.2f"),
                "avg_confidence": st.column_config.NumberColumn("Avg Confidence", format="%.1%"),
                "pairs_traded": "Pairs Traded"
            },
            use_container_width=True,
            hide_index=True
        )

        # Strategy comparison charts
        st.subheader("📈 Strategy Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # P&L by strategy
            fig_pnl = px.bar(
                strategy_df,
                x='strategy',
                y='total_pnl',
                title="Total P&L by Strategy",
                color='total_pnl',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col2:
            # Win rate comparison
            fig_win = px.bar(
                strategy_df,
                x='strategy',
                y='win_rate',
                title="Win Rate by Strategy",
                color='win_rate',
                color_continuous_scale='RdYlGn'
            )
            fig_win.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Break Even")
            st.plotly_chart(fig_win, use_container_width=True)

        # Strategy details
        st.subheader("🔍 Strategy Details")

        for _, strategy in strategy_df.iterrows():
            with st.expander(f"📊 {strategy['strategy']} Strategy Details"):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Trades", strategy['total_trades'])
                    st.metric("Wins", strategy['wins'])

                with col2:
                    st.metric("Win Rate", f"{strategy['win_rate']:.1f}%")
                    st.metric("Losses", strategy['losses'])

                with col3:
                    st.metric("Total P&L", f"{strategy['total_pnl']:.2f}")
                    st.metric("Avg P&L", f"{strategy['avg_pnl']:.2f}")

                with col4:
                    st.metric("Avg Confidence", f"{strategy['avg_confidence']:.1%}")
                    st.metric("Pairs Traded", strategy['pairs_traded'])

                # Best and worst trades
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏆 Best Trade: +{strategy['best_trade']:.2f}")
                with col2:
                    st.error(f"📉 Worst Trade: {strategy['worst_trade']:.2f}")

    def render_trade_performance_tab(self):
        """Render the trade performance tab"""
        st.header("💰 Trade Performance Analysis")

        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            date_option = st.selectbox(
                "📅 Date Filter",
                ["7_days", "30_days", "90_days"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="trade_date_filter"
            )

        with col2:
            if st.button("🔄 Refresh", key="trade_refresh"):
                st.rerun()

        with col3:
            export_enabled = st.checkbox("📊 Enable Export", key="export_checkbox")

        # Map to days
        days_map = {"7_days": 7, "30_days": 30, "90_days": 90}
        days_back = days_map[date_option]

        # Fetch data
        trades_df = self.fetch_trades_dataframe(days_back)

        if trades_df.empty:
            st.warning(f"No trade data found for the last {days_back} days")
            return

        # Calculate metrics
        metrics = self.calculate_simple_metrics(trades_df)

        # Display metrics
        st.subheader("📊 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pnl = metrics.get('total_pnl', 0)
            pnl_class = 'profit' if pnl > 0 else 'loss' if pnl < 0 else ''
            st.markdown(f"""
            <div class="metric-box">
                <h3 class="{pnl_class}">{pnl:+.2f}</h3>
                <p>Total P&L</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            win_rate = metrics.get('win_rate', 0)
            st.markdown(f"""
            <div class="metric-box">
                <h3>{win_rate:.1f}%</h3>
                <p>Win Rate</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.metric("Completed Trades", metrics.get('completed_trades', 0))

        with col4:
            st.metric("Pending Trades", metrics.get('pending_trades', 0))

        # Detailed trades table
        st.subheader("📋 Trade Details")

        # Filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            result_filter = st.multiselect(
                "Filter by Result",
                options=['WIN', 'LOSS', 'PENDING'],
                default=['WIN', 'LOSS', 'PENDING'],
                key="result_filter"
            )

        with col2:
            direction_filter = st.multiselect(
                "Filter by Direction",
                options=trades_df['direction'].unique() if 'direction' in trades_df.columns else [],
                default=trades_df['direction'].unique() if 'direction' in trades_df.columns else [],
                key="direction_filter"
            )

        with col3:
            symbol_filter = st.multiselect(
                "Filter by Symbol",
                options=trades_df['symbol'].unique() if 'symbol' in trades_df.columns else [],
                default=trades_df['symbol'].unique() if 'symbol' in trades_df.columns else [],
                key="symbol_filter"
            )

        with col4:
            strategy_filter = st.multiselect(
                "Filter by Strategy",
                options=trades_df['strategy'].dropna().unique() if 'strategy' in trades_df.columns else [],
                default=trades_df['strategy'].dropna().unique() if 'strategy' in trades_df.columns else [],
                key="strategy_filter"
            )

        # Apply filters
        filtered_df = trades_df.copy()
        if result_filter and 'trade_result' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['trade_result'].isin(result_filter)]
        if direction_filter and 'direction' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['direction'].isin(direction_filter)]
        if symbol_filter and 'symbol' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]
        if strategy_filter and 'strategy' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['strategy'].isin(strategy_filter)]

        # Display table
        if not filtered_df.empty:
            display_columns = ['timestamp', 'symbol', 'strategy', 'direction', 'entry_price', 'profit_loss_formatted', 'trade_result', 'status']
            display_columns = [col for col in display_columns if col in filtered_df.columns]

            st.dataframe(
                filtered_df[display_columns].head(50),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Trade Time", format="DD/MM/YY HH:mm"),
                    "symbol": "Symbol",
                    "strategy": "Strategy",
                    "direction": "Direction",
                    "entry_price": st.column_config.NumberColumn("Entry Price", format="%.5f"),
                    "profit_loss_formatted": "P&L",
                    "trade_result": "Result",
                    "status": "Status"
                }
            )

            # Export functionality
            if export_enabled:
                st.subheader("📊 Export Data")
                if st.button("📥 Download CSV"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"trades_{date_option}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No trades match the selected filters.")

    def render_market_intelligence_tab(self):
        """Render the Market Intelligence analysis tab"""
        st.header("🧠 Market Intelligence Analysis")
        st.markdown("*Analyze market conditions and regime patterns from comprehensive market scans*")

        conn = self.get_database_connection()
        if not conn:
            st.error("❌ Database connection failed. Cannot load market intelligence data.")
            return

        try:
            # Data source selector and date range
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=7),
                    key="mi_start_date"
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    key="mi_end_date"
                )

            with col3:
                data_source = st.selectbox(
                    "Data Source",
                    options=["Comprehensive Scans", "Signal-Based", "Both"],
                    index=0,
                    help="Choose data source: Comprehensive = all scans, Signal-Based = only when signals generated"
                )

            with col4:
                if st.button("🔄 Refresh", key="mi_refresh"):
                    st.rerun()

            # Query market intelligence data based on selected source
            if data_source == "Comprehensive Scans":
                mi_data = self.get_comprehensive_market_intelligence_data(conn, start_date, end_date)
                scan_data = mi_data
                signal_data = pd.DataFrame()
            elif data_source == "Signal-Based":
                mi_data = self.get_market_intelligence_data(conn, start_date, end_date)
                scan_data = pd.DataFrame()
                signal_data = mi_data
            else:  # Both
                scan_data = self.get_comprehensive_market_intelligence_data(conn, start_date, end_date)
                signal_data = self.get_market_intelligence_data(conn, start_date, end_date)
                mi_data = scan_data  # Use comprehensive data as primary for overall metrics

            if mi_data.empty and scan_data.empty and signal_data.empty:
                st.warning("⚠️ No market intelligence data found for the selected period.")
                st.info("💡 Market intelligence data is captured automatically during forex scanner operations.")
                return

            # Market Intelligence Overview
            st.subheader("📊 Market Intelligence Overview")

            # Show data source information
            if data_source == "Comprehensive Scans":
                st.info(f"📈 Displaying comprehensive market intelligence from {len(mi_data)} scan cycles")
            elif data_source == "Signal-Based":
                st.info(f"🎯 Displaying market intelligence from {len(mi_data)} signals")
            else:
                st.info(f"📊 Displaying combined data: {len(scan_data)} scans + {len(signal_data)} signals")

            # Determine metrics based on data source
            if data_source == "Comprehensive Scans" or (data_source == "Both" and not scan_data.empty):
                # Comprehensive scan metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_scans = len(mi_data)
                    st.metric("Total Market Scans", total_scans)

                with col2:
                    avg_epics = mi_data['epic_count'].mean() if 'epic_count' in mi_data.columns else 0
                    st.metric("Avg Epics per Scan", f"{avg_epics:.1f}")

                with col3:
                    unique_regimes = mi_data['regime'].nunique() if 'regime' in mi_data.columns else 0
                    st.metric("Market Regimes Detected", unique_regimes)

                with col4:
                    avg_confidence = mi_data['regime_confidence'].mean() if 'regime_confidence' in mi_data.columns else 0
                    st.metric("Avg Regime Confidence", f"{avg_confidence:.1%}")

            else:
                # Signal-based metrics (original)
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_signals = len(mi_data)
                    st.metric("Total Signals with Intelligence", total_signals)

                with col2:
                    unique_strategies = mi_data['strategy'].nunique() if 'strategy' in mi_data.columns else 0
                    st.metric("Strategies Covered", unique_strategies)

                with col3:
                    unique_regimes = mi_data['regime'].nunique() if 'regime' in mi_data.columns else 0
                    st.metric("Market Regimes Detected", unique_regimes)

                with col4:
                    avg_confidence = mi_data['regime_confidence'].mean() if 'regime_confidence' in mi_data.columns else 0
                    st.metric("Avg Regime Confidence", f"{avg_confidence:.1%}")

            # Market Regime Analysis
            if 'regime' in mi_data.columns:
                st.subheader("📈 Market Regime Distribution")

                col1, col2 = st.columns(2)

                with col1:
                    # Regime distribution pie chart
                    regime_counts = mi_data['regime'].value_counts()
                    fig_regime = px.pie(
                        values=regime_counts.values,
                        names=regime_counts.index,
                        title="Market Regime Distribution"
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)

                with col2:
                    # Regime confidence by strategy/session
                    if 'regime_confidence' in mi_data.columns:
                        if 'strategy' in mi_data.columns:
                            # Signal-based data: show by strategy
                            regime_conf_by_strategy = mi_data.groupby(['strategy', 'regime'])['regime_confidence'].mean().reset_index()
                            fig_conf = px.bar(
                                regime_conf_by_strategy,
                                x='strategy',
                                y='regime_confidence',
                                color='regime',
                                title="Average Regime Confidence by Strategy",
                                labels={'regime_confidence': 'Confidence'}
                            )
                            fig_conf.update_layout(yaxis_tickformat='.1%')
                            st.plotly_chart(fig_conf, use_container_width=True)
                        elif 'session' in mi_data.columns:
                            # Comprehensive scan data: show by session
                            regime_conf_by_session = mi_data.groupby(['session', 'regime'])['regime_confidence'].mean().reset_index()
                            fig_conf = px.bar(
                                regime_conf_by_session,
                                x='session',
                                y='regime_confidence',
                                color='regime',
                                title="Average Regime Confidence by Session",
                                labels={'regime_confidence': 'Confidence'}
                            )
                            fig_conf.update_layout(yaxis_tickformat='.1%')
                            st.plotly_chart(fig_conf, use_container_width=True)

            # Session and Volatility Analysis
            col1, col2 = st.columns(2)

            with col1:
                if 'session' in mi_data.columns:
                    st.subheader("🕐 Trading Session Analysis")
                    session_counts = mi_data['session'].value_counts()
                    session_df = pd.DataFrame({
                        'session': session_counts.index,
                        'count': session_counts.values
                    })
                    fig_session = px.bar(
                        session_df,
                        x='session',
                        y='count',
                        title="Signals by Trading Session",
                        labels={'count': 'Signal Count', 'session': 'Session'}
                    )
                    st.plotly_chart(fig_session, use_container_width=True)

            with col2:
                if 'volatility_level' in mi_data.columns:
                    st.subheader("📊 Volatility Level Distribution")
                    vol_counts = mi_data['volatility_level'].value_counts()
                    vol_df = pd.DataFrame({
                        'volatility': vol_counts.index,
                        'count': vol_counts.values
                    })
                    fig_vol = px.bar(
                        vol_df,
                        x='volatility',
                        y='count',
                        title="Signals by Volatility Level",
                        labels={'count': 'Signal Count', 'volatility': 'Volatility Level'},
                        color='volatility',
                        color_discrete_map={'high': 'red', 'medium': 'orange', 'low': 'green'}
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

            # Strategy/Regime Performance Analysis
            if data_source != "Comprehensive Scans":
                st.subheader("🎯 Strategy Performance by Market Conditions")

                if 'confidence_score' in mi_data.columns and 'regime' in mi_data.columns and 'strategy' in mi_data.columns:
                    strategy_performance = mi_data.groupby(['strategy', 'regime']).agg({
                        'confidence_score': ['mean', 'count'],
                        'regime_confidence': 'mean' if 'regime_confidence' in mi_data.columns else lambda x: 0
                    }).round(3)

                    strategy_performance.columns = ['Avg_Signal_Confidence', 'Signal_Count', 'Avg_Regime_Confidence']
                    strategy_performance = strategy_performance.reset_index()

                    st.dataframe(strategy_performance, use_container_width=True)
            else:
                # For comprehensive scans, show regime performance by recommended strategy
                if 'recommended_strategy' in mi_data.columns and 'regime' in mi_data.columns:
                    st.subheader("🎯 Recommended Strategy by Market Conditions")

                    strategy_performance = mi_data.groupby(['recommended_strategy', 'regime']).agg({
                        'regime_confidence': ['mean', 'count']
                    }).round(3)

                    strategy_performance.columns = ['Avg_Regime_Confidence', 'Scan_Count']
                    strategy_performance = strategy_performance.reset_index()

                    st.dataframe(strategy_performance, use_container_width=True)

            # Intelligence Source Analysis
            if 'intelligence_source' in mi_data.columns:
                st.subheader("🔍 Intelligence Source Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    source_counts = mi_data['intelligence_source'].value_counts()
                    st.write("**Intelligence Sources:**")
                    for source, count in source_counts.items():
                        source_type = "🧠 Strategy-Specific" if "MarketIntelligenceEngine" in source else "🌐 Universal Capture"
                        st.write(f"{source_type}: {count} signals")

                with col2:
                    # Source distribution by strategy/session
                    if 'strategy' in mi_data.columns:
                        # Signal-based data: show by strategy
                        source_by_strategy = mi_data.groupby('strategy')['intelligence_source'].value_counts().reset_index()
                        fig_source = px.bar(
                            source_by_strategy,
                            x='strategy',
                            y='count',
                            color='intelligence_source',
                            title="Intelligence Source by Strategy",
                            labels={'count': 'Signal Count'}
                        )
                        st.plotly_chart(fig_source, use_container_width=True)
                    elif 'session' in mi_data.columns:
                        # Comprehensive scan data: show by session
                        source_by_session = mi_data.groupby('session')['intelligence_source'].value_counts().reset_index()
                        fig_source = px.bar(
                            source_by_session,
                            x='session',
                            y='count',
                            color='intelligence_source',
                            title="Intelligence Source by Session",
                            labels={'count': 'Scan Count'}
                        )
                        st.plotly_chart(fig_source, use_container_width=True)

            # Enhanced Comprehensive Market Intelligence Visualizations
            if data_source == "Comprehensive Scans" or (data_source == "Both" and not scan_data.empty):
                self.render_comprehensive_market_intelligence_charts(mi_data)

            # Market Intelligence Search
            st.subheader("🔍 Market Intelligence Search")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Only show strategy filter for signal-based data
                if 'strategy' in mi_data.columns:
                    search_strategy = st.selectbox(
                        "Filter by Strategy",
                        options=['All'] + sorted(mi_data['strategy'].unique().tolist()),
                        key="mi_search_strategy"
                    )
                else:
                    # For comprehensive scans, show recommended strategy filter instead
                    if 'recommended_strategy' in mi_data.columns:
                        search_strategy = st.selectbox(
                            "Filter by Recommended Strategy",
                            options=['All'] + sorted(mi_data['recommended_strategy'].unique().tolist()),
                            key="mi_search_strategy"
                        )
                    else:
                        search_strategy = 'All'
                        st.info("Strategy filtering not available for comprehensive scan data")

            with col2:
                search_regime = st.selectbox(
                    "Filter by Regime",
                    options=['All'] + sorted(mi_data['regime'].unique().tolist()) if 'regime' in mi_data.columns else ['All'],
                    key="mi_search_regime"
                )

            with col3:
                search_session = st.selectbox(
                    "Filter by Session",
                    options=['All'] + sorted(mi_data['session'].unique().tolist()) if 'session' in mi_data.columns else ['All'],
                    key="mi_search_session"
                )

            # Apply filters
            filtered_data = mi_data.copy()

            # Apply strategy filter based on data type
            if search_strategy != 'All':
                if 'strategy' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['strategy'] == search_strategy]
                elif 'recommended_strategy' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['recommended_strategy'] == search_strategy]

            if search_regime != 'All' and 'regime' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['regime'] == search_regime]
            if search_session != 'All' and 'session' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['session'] == search_session]

            # Update display text based on data source
            if data_source == "Comprehensive Scans":
                st.write(f"**Showing {len(filtered_data)} scan cycles matching filters:**")
            else:
                st.write(f"**Showing {len(filtered_data)} signals matching filters:**")

            # Display filtered results
            if not filtered_data.empty:
                # Determine display columns based on data source
                if data_source == "Comprehensive Scans" or (data_source == "Both" and 'scan_timestamp' in filtered_data.columns):
                    # Comprehensive scan data columns
                    display_columns = ['scan_timestamp', 'scan_cycle_id', 'epic_count']
                    if 'regime' in filtered_data.columns:
                        display_columns.extend(['regime', 'regime_confidence'])
                    if 'session' in filtered_data.columns:
                        display_columns.append('session')
                    if 'market_bias' in filtered_data.columns:
                        display_columns.append('market_bias')
                    if 'risk_sentiment' in filtered_data.columns:
                        display_columns.append('risk_sentiment')
                    if 'recommended_strategy' in filtered_data.columns:
                        display_columns.append('recommended_strategy')

                    sort_column = 'scan_timestamp'
                else:
                    # Signal-based data columns (original)
                    display_columns = ['alert_timestamp', 'epic', 'strategy', 'signal_type', 'confidence_score']
                    if 'regime' in filtered_data.columns:
                        display_columns.extend(['regime', 'regime_confidence'])
                    if 'session' in filtered_data.columns:
                        display_columns.append('session')
                    if 'volatility_level' in filtered_data.columns:
                        display_columns.append('volatility_level')

                    sort_column = 'alert_timestamp'

                available_columns = [col for col in display_columns if col in filtered_data.columns]

                # Sort and display
                if sort_column in filtered_data.columns:
                    sorted_data = filtered_data[available_columns].sort_values(sort_column, ascending=False)
                else:
                    sorted_data = filtered_data[available_columns]

                st.dataframe(sorted_data, use_container_width=True)

                # Export functionality
                if st.button("📥 Export Filtered Data as CSV"):
                    csv = filtered_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"market_intelligence_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No signals match the selected filters.")

        except Exception as e:
            st.error(f"❌ Error loading market intelligence data: {e}")

        finally:
            conn.close()

    def get_market_intelligence_data(self, conn, start_date, end_date):
        """Query market intelligence data from alert_history"""
        try:
            query = """
            SELECT
                a.id,
                a.alert_timestamp,
                a.epic,
                a.strategy,
                a.signal_type,
                a.confidence_score,
                a.strategy_metadata,
                (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'dominant_regime') as regime,
                (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'confidence')::float as regime_confidence,
                (a.strategy_metadata::json->'market_intelligence'->'session_analysis'->>'current_session') as session,
                (a.strategy_metadata::json->'market_intelligence'->>'volatility_level') as volatility_level,
                (a.strategy_metadata::json->'market_intelligence'->>'intelligence_source') as intelligence_source
            FROM alert_history a
            WHERE a.alert_timestamp >= %s
              AND a.alert_timestamp <= %s
              AND a.strategy_metadata IS NOT NULL
              AND (a.strategy_metadata::json->'market_intelligence') IS NOT NULL
            ORDER BY a.alert_timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            # Clean and convert data types
            if not df.empty:
                # Handle None values
                df = df.where(pd.notnull(df), None)

            return df

        except Exception as e:
            st.error(f"Database query error: {e}")
            return pd.DataFrame()

    def get_comprehensive_market_intelligence_data(self, conn, start_date, end_date):
        """Query comprehensive market intelligence data from market_intelligence_history table"""
        try:
            # First check which columns exist in the table
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'market_intelligence_history'
                """)
                existing_columns = [row[0] for row in cursor.fetchall()]

            # Base columns that should always exist
            base_columns = [
                "mih.id",
                "mih.scan_timestamp",
                "mih.scan_cycle_id",
                "mih.epic_list",
                "mih.epic_count",
                "mih.dominant_regime as regime",
                "mih.regime_confidence",
                "mih.current_session as session",
                "mih.session_volatility as volatility_level",
                "mih.market_bias",
                "mih.average_trend_strength",
                "mih.average_volatility",
                "mih.risk_sentiment",
                "mih.recommended_strategy",
                "mih.confidence_threshold",
                "mih.intelligence_source",
                "mih.regime_trending_score",
                "mih.regime_ranging_score",
                "mih.regime_breakout_score",
                "mih.regime_reversal_score",
                "mih.regime_high_vol_score",
                "mih.regime_low_vol_score"
            ]

            # Add new columns only if they exist
            optional_columns = []
            if 'individual_epic_regimes' in existing_columns:
                optional_columns.append("mih.individual_epic_regimes")
            if 'pair_analyses' in existing_columns:
                optional_columns.append("mih.pair_analyses")

            # Combine all columns
            all_columns = base_columns + optional_columns

            query = f"""
            SELECT
                {', '.join(all_columns)}
            FROM market_intelligence_history mih
            WHERE mih.scan_timestamp >= %s
              AND mih.scan_timestamp <= %s
            ORDER BY mih.scan_timestamp DESC
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            # Clean and convert data types
            if not df.empty:
                # Handle None values
                df = df.where(pd.notnull(df), None)

                # Convert timestamp to datetime if needed
                if 'scan_timestamp' in df.columns:
                    df['scan_timestamp'] = pd.to_datetime(df['scan_timestamp'])

            return df

        except Exception as e:
            st.error(f"Comprehensive market intelligence query error: {e}")
            return pd.DataFrame()

    def render_comprehensive_market_intelligence_charts(self, mi_data):
        """Render enhanced charts for comprehensive market intelligence data"""
        st.subheader("🧠 Enhanced Market Intelligence Analytics")
        st.markdown("*Advanced visualizations from comprehensive market scan data*")

        # Time series analysis
        if 'scan_timestamp' in mi_data.columns:
            st.subheader("📈 Market Regime Evolution Over Time")

            # Prepare time series data
            mi_data_time = mi_data.copy()
            mi_data_time['scan_timestamp'] = pd.to_datetime(mi_data_time['scan_timestamp'])
            mi_data_time = mi_data_time.sort_values('scan_timestamp')

            # Create time series plot
            fig_timeline = px.scatter(
                mi_data_time,
                x='scan_timestamp',
                y='regime_confidence',
                color='regime',
                size='epic_count',
                title="Market Regime Confidence Timeline",
                labels={
                    'scan_timestamp': 'Time',
                    'regime_confidence': 'Confidence',
                    'epic_count': 'Epics Analyzed'
                },
                hover_data=['session', 'market_bias']
            )
            fig_timeline.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Market bias and risk sentiment analysis
        col1, col2 = st.columns(2)

        with col1:
            if 'market_bias' in mi_data.columns:
                st.subheader("📊 Market Bias Distribution")
                bias_counts = mi_data['market_bias'].value_counts()
                fig_bias = px.pie(
                    values=bias_counts.values,
                    names=bias_counts.index,
                    title="Market Bias Distribution",
                    color_discrete_map={
                        'bullish': '#28a745',
                        'bearish': '#dc3545',
                        'neutral': '#6c757d'
                    }
                )
                st.plotly_chart(fig_bias, use_container_width=True)

        with col2:
            if 'risk_sentiment' in mi_data.columns:
                st.subheader("🎯 Risk Sentiment Analysis")
                risk_counts = mi_data['risk_sentiment'].value_counts()
                fig_risk = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title="Risk Sentiment Distribution",
                    labels={'x': 'Risk Sentiment', 'y': 'Count'},
                    color=risk_counts.index,
                    color_discrete_map={
                        'risk_on': '#28a745',
                        'risk_off': '#dc3545',
                        'neutral': '#6c757d'
                    }
                )
                st.plotly_chart(fig_risk, use_container_width=True)

        # Advanced regime analysis with individual scores
        regime_score_columns = [
            'regime_trending_score', 'regime_ranging_score', 'regime_breakout_score',
            'regime_reversal_score', 'regime_high_vol_score', 'regime_low_vol_score'
        ]

        available_score_columns = [col for col in regime_score_columns if col in mi_data.columns]

        if available_score_columns:
            st.subheader("📊 Detailed Regime Score Analysis")

            # Create regime scores heatmap data
            regime_scores_data = []
            for _, row in mi_data.iterrows():
                timestamp = row.get('scan_timestamp', 'Unknown')
                for score_col in available_score_columns:
                    if pd.notnull(row.get(score_col)):
                        regime_scores_data.append({
                            'timestamp': timestamp,
                            'regime_type': score_col.replace('regime_', '').replace('_score', ''),
                            'score': row[score_col],
                            'dominant_regime': row.get('regime', 'unknown')
                        })

            if regime_scores_data:
                scores_df = pd.DataFrame(regime_scores_data)

                # Create grouped bar chart
                fig_scores = px.box(
                    scores_df,
                    x='regime_type',
                    y='score',
                    color='dominant_regime',
                    title="Regime Score Distribution by Dominant Regime",
                    labels={'score': 'Score', 'regime_type': 'Regime Type'}
                )
                fig_scores.update_layout(yaxis_tickformat='.2f')
                st.plotly_chart(fig_scores, use_container_width=True)

        # Market strength indicators
        strength_columns = ['average_trend_strength', 'average_volatility']
        available_strength_columns = [col for col in strength_columns if col in mi_data.columns]

        if available_strength_columns:
            st.subheader("💪 Market Strength Indicators")

            col1, col2 = st.columns(2)

            with col1:
                if 'average_trend_strength' in mi_data.columns:
                    # Trend strength by session
                    if 'session' in mi_data.columns:
                        strength_by_session = mi_data.groupby('session')['average_trend_strength'].mean().reset_index()
                        fig_strength = px.bar(
                            strength_by_session,
                            x='session',
                            y='average_trend_strength',
                            title="Average Trend Strength by Session",
                            labels={'average_trend_strength': 'Trend Strength'}
                        )
                        fig_strength.update_layout(yaxis_tickformat='.2f')
                        st.plotly_chart(fig_strength, use_container_width=True)

            with col2:
                if 'average_volatility' in mi_data.columns:
                    # Volatility distribution
                    fig_vol_dist = px.histogram(
                        mi_data,
                        x='average_volatility',
                        nbins=20,
                        title="Market Volatility Distribution",
                        labels={'average_volatility': 'Volatility', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_vol_dist, use_container_width=True)

        # Session-based comprehensive analysis
        if 'session' in mi_data.columns and 'regime' in mi_data.columns:
            st.subheader("🕐 Session vs Regime Matrix")

            # Create session-regime cross-tabulation
            session_regime_matrix = pd.crosstab(mi_data['session'], mi_data['regime'])

            # Convert to percentage
            session_regime_pct = session_regime_matrix.div(session_regime_matrix.sum(axis=1), axis=0) * 100

            # Create heatmap
            fig_matrix = px.imshow(
                session_regime_pct.values,
                labels=dict(x="Market Regime", y="Trading Session", color="Percentage"),
                x=session_regime_pct.columns,
                y=session_regime_pct.index,
                title="Session vs Regime Distribution (%)",
                color_continuous_scale="RdYlGn"
            )
            fig_matrix.update_layout(width=700, height=400)
            st.plotly_chart(fig_matrix, use_container_width=True)

        # Individual Epic Regimes Analysis
        self.render_individual_epic_regimes(mi_data)

        # Summary statistics table
        if len(mi_data) > 0:
            st.subheader("📊 Comprehensive Market Intelligence Summary")

            # Calculate summary statistics
            summary_stats = {
                'Total Scan Cycles': len(mi_data),
                'Date Range': f"{mi_data['scan_timestamp'].min().date()} to {mi_data['scan_timestamp'].max().date()}" if 'scan_timestamp' in mi_data.columns else "N/A",
                'Most Common Regime': mi_data['regime'].mode().iloc[0] if 'regime' in mi_data.columns and len(mi_data['regime'].mode()) > 0 else "N/A",
                'Average Confidence': f"{mi_data['regime_confidence'].mean():.1%}" if 'regime_confidence' in mi_data.columns else "N/A",
                'Most Active Session': mi_data['session'].mode().iloc[0] if 'session' in mi_data.columns and len(mi_data['session'].mode()) > 0 else "N/A",
                'Average Epics per Scan': f"{mi_data['epic_count'].mean():.1f}" if 'epic_count' in mi_data.columns else "N/A"
            }

            # Add market strength stats if available
            if 'average_trend_strength' in mi_data.columns:
                summary_stats['Average Trend Strength'] = f"{mi_data['average_trend_strength'].mean():.3f}"
            if 'market_bias' in mi_data.columns:
                summary_stats['Most Common Bias'] = mi_data['market_bias'].mode().iloc[0] if len(mi_data['market_bias'].mode()) > 0 else "N/A"

            # Display as metrics
            col1, col2, col3 = st.columns(3)

            stats_items = list(summary_stats.items())
            for i, (key, value) in enumerate(stats_items):
                col = [col1, col2, col3][i % 3]
                with col:
                    st.metric(key, value)

    def render_individual_epic_regimes(self, mi_data):
        """Render individual epic regime analysis"""
        if mi_data.empty:
            return

        # Check if individual epic regimes data is available
        if 'individual_epic_regimes' not in mi_data.columns:
            st.info("💡 Individual epic regime analysis will be available after the next market intelligence scan with the enhanced system.")
            st.markdown("*This feature shows detailed regime analysis for each currency pair individually*")
            return

        st.subheader("🌍 Individual Epic Regime Analysis")
        st.markdown("*Detailed regime analysis for each currency pair across all market scans*")

        try:
            import json

            # Process individual epic regimes data
            all_epic_regimes = {}
            regime_timeline = []

            for idx, row in mi_data.iterrows():
                if pd.notna(row['individual_epic_regimes']):
                    try:
                        epic_regimes = json.loads(row['individual_epic_regimes']) if isinstance(row['individual_epic_regimes'], str) else row['individual_epic_regimes']
                        timestamp = row['scan_timestamp']

                        for epic, data in epic_regimes.items():
                            if epic not in all_epic_regimes:
                                all_epic_regimes[epic] = {'regimes': [], 'confidences': [], 'timestamps': []}

                            all_epic_regimes[epic]['regimes'].append(data.get('regime', 'unknown'))
                            all_epic_regimes[epic]['confidences'].append(data.get('confidence', 0.5))
                            all_epic_regimes[epic]['timestamps'].append(timestamp)

                            # Add to timeline data
                            regime_timeline.append({
                                'epic': epic,
                                'regime': data.get('regime', 'unknown'),
                                'confidence': data.get('confidence', 0.5),
                                'timestamp': timestamp,
                                'scan_id': row['id']
                            })
                    except (json.JSONDecodeError, TypeError):
                        continue

            if not all_epic_regimes:
                st.info("💡 Individual epic regime data not available for selected period.")
                return

            # Epic regime distribution overview
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Current Epic Regimes")

                # Get latest regime for each epic
                latest_regimes = {}
                for epic, data in all_epic_regimes.items():
                    if data['regimes']:
                        latest_regimes[epic] = {
                            'regime': data['regimes'][-1],
                            'confidence': data['confidences'][-1]
                        }

                # Display as colored badges
                for epic, regime_data in sorted(latest_regimes.items()):
                    regime = regime_data['regime']
                    confidence = regime_data['confidence']

                    # Color coding for regimes
                    color_map = {
                        'trending': '#28a745',      # Green
                        'ranging': '#007bff',       # Blue
                        'breakout': '#ff6b6b',      # Red
                        'reversal': '#ffc107',      # Yellow
                        'low_volatility': '#6c757d', # Gray
                        'high_volatility': '#e83e8c' # Pink
                    }
                    color = color_map.get(regime, '#6c757d')

                    st.markdown(f"""
                    <div style="
                        background: {color};
                        color: white;
                        padding: 8px 12px;
                        border-radius: 6px;
                        margin: 4px 0;
                        display: inline-block;
                        font-weight: bold;
                        min-width: 120px;
                        text-align: center;
                    ">
                        {epic}: {regime.upper()} ({confidence:.1%})
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Regime distribution pie chart
                if latest_regimes:
                    regime_counts = {}
                    for data in latest_regimes.values():
                        regime = data['regime']
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1

                    fig_pie = px.pie(
                        values=list(regime_counts.values()),
                        names=list(regime_counts.keys()),
                        title="Current Regime Distribution",
                        color_discrete_map={
                            'trending': '#28a745',
                            'ranging': '#007bff',
                            'breakout': '#ff6b6b',
                            'reversal': '#ffc107',
                            'low_volatility': '#6c757d',
                            'high_volatility': '#e83e8c'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            # Epic regime timeline
            if regime_timeline:
                st.subheader("📈 Epic Regime Evolution Timeline")

                # Convert to DataFrame for plotting
                timeline_df = pd.DataFrame(regime_timeline)
                timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])

                # Epic selector for focused view
                selected_epics = st.multiselect(
                    "Select Epics to View",
                    options=sorted(timeline_df['epic'].unique()),
                    default=sorted(timeline_df['epic'].unique())[:5],  # Show first 5 by default
                    help="Select which currency pairs to display in the timeline"
                )

                if selected_epics:
                    filtered_timeline = timeline_df[timeline_df['epic'].isin(selected_epics)]

                    # Create timeline scatter plot
                    fig_timeline = px.scatter(
                        filtered_timeline,
                        x='timestamp',
                        y='epic',
                        color='regime',
                        size='confidence',
                        title="Epic Regime Timeline",
                        labels={
                            'timestamp': 'Time',
                            'epic': 'Currency Pair',
                            'confidence': 'Confidence',
                            'regime': 'Market Regime'
                        },
                        hover_data=['confidence'],
                        color_discrete_map={
                            'trending': '#28a745',
                            'ranging': '#007bff',
                            'breakout': '#ff6b6b',
                            'reversal': '#ffc107',
                            'low_volatility': '#6c757d',
                            'high_volatility': '#e83e8c'
                        }
                    )
                    fig_timeline.update_layout(height=400)
                    st.plotly_chart(fig_timeline, use_container_width=True)

            # Epic-specific strategy recommendations
            st.subheader("🎯 Epic-Specific Strategy Recommendations")

            strategy_recommendations = {
                'trending': {
                    'strategy': 'Trend Following',
                    'description': 'Use momentum strategies, trend confirmations, and trailing stops',
                    'icon': '📈',
                    'color': '#28a745'
                },
                'ranging': {
                    'strategy': 'Range Trading',
                    'description': 'Trade support/resistance bounces, use mean reversion strategies',
                    'icon': '↔️',
                    'color': '#007bff'
                },
                'breakout': {
                    'strategy': 'Breakout Trading',
                    'description': 'Monitor for volume confirmation, use wider stops, expect volatility',
                    'icon': '🚀',
                    'color': '#ff6b6b'
                },
                'reversal': {
                    'strategy': 'Reversal Trading',
                    'description': 'Look for reversal patterns, use conservative position sizing',
                    'icon': '🔄',
                    'color': '#ffc107'
                },
                'low_volatility': {
                    'strategy': 'Conservative Trading',
                    'description': 'Use smaller position sizes, tighter stops, range strategies',
                    'icon': '🐌',
                    'color': '#6c757d'
                },
                'high_volatility': {
                    'strategy': 'Volatility Trading',
                    'description': 'Reduce position size, use wider stops, consider straddles',
                    'icon': '⚡',
                    'color': '#e83e8c'
                }
            }

            # Group epics by regime for recommendations
            regime_groups = {}
            for epic, regime_data in latest_regimes.items():
                regime = regime_data['regime']
                if regime not in regime_groups:
                    regime_groups[regime] = []
                regime_groups[regime].append(epic)

            for regime, epics in regime_groups.items():
                if regime in strategy_recommendations:
                    rec = strategy_recommendations[regime]

                    st.markdown(f"""
                    <div style="
                        border-left: 4px solid {rec['color']};
                        background: #f8f9fa;
                        padding: 1rem;
                        margin: 1rem 0;
                        border-radius: 4px;
                    ">
                        <h4 style="margin: 0; color: {rec['color']};">
                            {rec['icon']} {rec['strategy']} - {regime.upper()} Regime
                        </h4>
                        <p style="margin: 0.5rem 0; color: #6c757d;">
                            <strong>Pairs:</strong> {', '.join(epics)}
                        </p>
                        <p style="margin: 0; color: #333;">
                            {rec['description']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error rendering individual epic regimes: {e}")

    def render_trade_analysis_tab(self):
        """Render the Trade Analysis tab with sub-tabs for trailing stop and signal analysis"""
        st.header("🔍 Individual Trade Analysis")
        st.markdown("*Comprehensive analysis of trade execution and entry signals*")

        # Get latest closed trade ID as default
        default_trade_id = self.fetch_latest_closed_trade_id()

        # Input for trade ID (shared between sub-tabs)
        col1, col2 = st.columns([3, 1])

        with col1:
            trade_id = st.number_input("Enter Trade ID", min_value=1, value=default_trade_id, step=1, key="trade_id_input")

        with col2:
            analyze_btn = st.button("🔍 Analyze Trade", type="primary")

        # Sub-tabs for different analysis types
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "📊 Trailing Stop Analysis",
            "🎯 Signal Analysis",
            "📚 Outcome Analysis"
        ])

        with sub_tab1:
            self._render_trailing_stop_analysis(trade_id, analyze_btn)

        with sub_tab2:
            self._render_signal_analysis(trade_id, analyze_btn)

        with sub_tab3:
            self._render_outcome_analysis(trade_id, analyze_btn)

    def _render_trailing_stop_analysis(self, trade_id: int, analyze_btn: bool):
        """Render the trailing stop stage analysis sub-tab"""
        st.subheader("📊 Trailing Stop Stage Analysis")
        st.markdown("*Analyze break-even triggers and profit lock stages*")

        if analyze_btn or trade_id:
            try:
                import requests

                # Call the FastAPI endpoint
                headers = {
                    "X-APIM-Gateway": "verified",
                    "X-API-KEY": "436abe054a074894a0517e5172f0e5b6"
                }

                with st.spinner(f"Analyzing trade {trade_id}..."):
                    response = requests.get(
                        f"http://fastapi-dev:8000/api/trade-analysis/trade/{trade_id}",
                        headers=headers
                    )

                if response.status_code == 200:
                    data = response.json()

                    # Trade Details Section
                    st.subheader("📋 Trade Details")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Symbol", data['trade_details']['symbol'].replace('CS.D.', '').replace('.MINI.IP', ''))
                        st.metric("Direction", data['trade_details']['direction'])

                    with col2:
                        st.metric("Entry Price", f"{data['trade_details']['entry_price']:.5f}")
                        st.metric("Stop Loss", f"{data['trade_details']['sl_price']:.5f}")

                    with col3:
                        st.metric("Take Profit", f"{data['trade_details']['tp_price']:.5f}")
                        st.metric("Status", data['trade_details']['status'].upper())

                    with col4:
                        metrics = data['calculated_metrics']
                        sl_color = "🟢" if metrics['sl_above_entry'] else "🔴"
                        st.metric("SL Distance", f"{metrics['sl_distance_pts']:.1f} pts")
                        st.metric("Protection", f"{sl_color} {'+' if metrics['sl_above_entry'] else ''}{metrics['sl_distance_pts']:.1f}")

                    # Pair Configuration Section
                    st.subheader("⚙️ Pair-Specific Configuration")
                    cfg = data['pair_configuration']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🛡️ Break-Even</h4>
                            <p><strong>Trigger:</strong> {cfg['break_even_trigger_points']} points</p>
                            <p><strong>Lock:</strong> 0 points (entry)</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📊 Stage 1 (Profit Lock)</h4>
                            <p><strong>Trigger:</strong> {cfg['stage1_trigger_points']} points</p>
                            <p><strong>Lock:</strong> {cfg['stage1_lock_points']} points profit</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🎯 Stage 2 (Profit Lock)</h4>
                            <p><strong>Trigger:</strong> {cfg['stage2_trigger_points']} points</p>
                            <p><strong>Lock:</strong> {cfg['stage2_lock_points']} points profit</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🚀 Stage 3 (ATR Trailing)</h4>
                            <p><strong>Trigger:</strong> {cfg['stage3_trigger_points']} points</p>
                            <p><strong>ATR:</strong> {cfg['stage3_atr_multiplier']}x multiplier</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Stage Activation Analysis
                    st.subheader("📈 Stage Activation Analysis")
                    stages = data['stage_analysis']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        be_emoji = "✅" if stages['breakeven']['activated'] else "❌"
                        be_bg = '#d4edda' if stages['breakeven']['activated'] else '#f8d7da'
                        be_status = 'ACTIVATED' if stages['breakeven']['activated'] else 'NOT REACHED'

                        # Build conditional content
                        be_extra = ""
                        if stages['breakeven']['activated']:
                            be_extra = f"<p><strong>Time:</strong> {stages['breakeven']['activation_time']}</p><p><strong>Highest Profit:</strong> {stages['breakeven']['max_profit_reached']} pts</p><p><strong>Final Lock:</strong> +{stages['breakeven']['final_lock']} pts</p>"
                        else:
                            be_extra = f"<p><strong>Required:</strong> {stages['breakeven']['trigger_threshold']} pts</p>"

                        st.markdown(f"""
                        <div class="metric-card" style="background: {be_bg};">
                            <h4>{be_emoji} Break-Even</h4>
                            <p><strong>Status:</strong> {be_status}</p>
                            {be_extra}
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        stage1_emoji = "✅" if stages['stage1']['activated'] else "❌"
                        stage1_bg = '#d4edda' if stages['stage1']['activated'] else '#fff3cd'
                        stage1_status = 'ACTIVATED' if stages['stage1']['activated'] else 'NOT REACHED'

                        # Build conditional content
                        stage1_extra = ""
                        if stages['stage1']['activated']:
                            actual_lock = stages['stage1'].get('actual_lock', stages['stage1']['lock_amount'])
                            stage1_extra = f"<p><strong>Time:</strong> {stages['stage1']['activation_time']}</p><p><strong>Actual Lock:</strong> +{actual_lock} pts</p><p><strong>Expected:</strong> +{stages['stage1']['lock_amount']} pts</p>"
                        else:
                            stage1_extra = f"<p><strong>Required:</strong> {stages['stage1']['trigger_threshold']} pts</p><p><strong>Would Lock:</strong> +{stages['stage1']['lock_amount']} pts</p>"

                        st.markdown(f"""
                        <div class="metric-card" style="background: {stage1_bg};">
                            <h4>{stage1_emoji} Stage 1: Profit Lock</h4>
                            <p><strong>Status:</strong> {stage1_status}</p>
                            {stage1_extra}
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        stage2_emoji = "✅" if stages['stage2']['activated'] else "❌"
                        stage2_bg = '#d4edda' if stages['stage2']['activated'] else '#fff3cd'
                        stage2_status = 'ACTIVATED' if stages['stage2']['activated'] else 'NOT REACHED'

                        # Build conditional content
                        stage2_extra = ""
                        if stages['stage2']['activated']:
                            stage2_extra = f"<p><strong>Time:</strong> {stages['stage2']['activation_time']}</p>"
                        else:
                            stage2_extra = f"<p><strong>Required:</strong> {stages['stage2']['trigger_threshold']} pts</p><p><strong>Would Lock:</strong> +{stages['stage2']['lock_amount']} pts</p>"

                        st.markdown(f"""
                        <div class="metric-card" style="background: {stage2_bg};">
                            <h4>{stage2_emoji} Stage 2: Profit Lock</h4>
                            <p><strong>Status:</strong> {stage2_status}</p>
                            {stage2_extra}
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        stage3_emoji = "✅" if stages['stage3']['activated'] else "❌"
                        stage3_bg = '#d4edda' if stages['stage3']['activated'] else '#fff3cd'
                        stage3_status = 'ACTIVATED' if stages['stage3']['activated'] else 'NOT REACHED'

                        # Build conditional content
                        stage3_extra = ""
                        if stages['stage3']['activated']:
                            stage3_extra = f"<p><strong>Time:</strong> {stages['stage3']['activation_time']}</p>"
                        else:
                            stage3_extra = f"<p><strong>Required:</strong> {stages['stage3']['trigger_threshold']} pts</p>"

                        st.markdown(f"""
                        <div class="metric-card" style="background: {stage3_bg};">
                            <h4>{stage3_emoji} Stage 3: ATR Trailing</h4>
                            <p><strong>Status:</strong> {stage3_status}</p>
                            {stage3_extra}
                        </div>
                        """, unsafe_allow_html=True)

                    # Performance Summary
                    st.subheader("📊 Performance Summary")
                    summary = data['summary']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Stages Activated", f"{summary['stages_activated']}/3")

                    with col2:
                        st.metric("Highest Trigger Reached", f"{summary['max_profit_reached']} pts")

                    with col3:
                        profit_protected = summary['final_protection']
                        protection_emoji = "🟢" if profit_protected > 0 else "🔴"
                        st.metric("Final Protection", f"{protection_emoji} +{profit_protected:.1f} pts")

                    with col4:
                        fully_trailed = "Yes ✅" if summary['fully_trailed'] else "No ❌"
                        st.metric("Fully Trailed", fully_trailed)

                    # Profit Timeline Chart
                    if data['timeline']['profit_progression']:
                        st.subheader("📈 Profit Progression Timeline")

                        import plotly.graph_objects as go

                        profit_data = data['timeline']['profit_progression']
                        timestamps = [p['timestamp'] for p in profit_data]
                        profits = [p['profit_pts'] for p in profit_data]

                        fig = go.Figure()

                        # Add profit line
                        fig.add_trace(go.Scatter(
                            x=timestamps,
                            y=profits,
                            mode='lines+markers',
                            name='Profit (pts)',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6)
                        ))

                        # Add stage threshold lines
                        fig.add_hline(y=cfg['break_even_trigger_points'], line_dash="dash",
                                     line_color="green", annotation_text="Stage 1 Trigger")
                        fig.add_hline(y=cfg['stage2_trigger_points'], line_dash="dash",
                                     line_color="orange", annotation_text="Stage 2 Trigger")
                        fig.add_hline(y=cfg['stage3_trigger_points'], line_dash="dash",
                                     line_color="red", annotation_text="Stage 3 Trigger")

                        fig.update_layout(
                            title=f"Trade {trade_id} - Profit Evolution",
                            xaxis_title="Time",
                            yaxis_title="Profit (points)",
                            hovermode='x unified',
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Event Log
                    st.subheader("📝 Event Log")

                    # Break-even events
                    if data['timeline']['break_even_events']:
                        st.markdown("**🎯 Break-Even Triggers:**")
                        for event in data['timeline']['break_even_events']:
                            st.info(f"⏰ {event['timestamp']}: Profit {event['profit_pts']}pts ≥ Trigger {event['trigger_pts']}pts")

                    # Stop adjustments
                    if data['timeline']['stop_adjustments']:
                        st.markdown("**📤 Stop Adjustments:**")
                        for event in data['timeline']['stop_adjustments']:
                            st.success(f"⏰ {event['timestamp']}: Stop moved to {event['new_stop']:.5f}")

                    # Raw Data (Expandable)
                    with st.expander("🔍 View Raw Analysis Data"):
                        st.json(data)

                elif response.status_code == 404:
                    st.error(f"❌ Trade {trade_id} not found in database")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"❌ Error analyzing trade: {e}")
                import traceback
                st.code(traceback.format_exc())

    def _render_signal_analysis(self, trade_id: int, analyze_btn: bool):
        """Render the strategy signal analysis sub-tab"""
        st.subheader("🎯 Entry Signal Analysis")
        st.markdown("*Analyze the strategy signal that triggered this trade*")

        if analyze_btn or trade_id:
            try:
                import requests

                # Call the FastAPI signal analysis endpoint
                headers = {
                    "X-APIM-Gateway": "verified",
                    "X-API-KEY": "436abe054a074894a0517e5172f0e5b6"
                }

                with st.spinner(f"Analyzing signal for trade {trade_id}..."):
                    response = requests.get(
                        f"http://fastapi-dev:8000/api/trade-analysis/signal/{trade_id}",
                        headers=headers
                    )

                if response.status_code == 200:
                    data = response.json()

                    # Check if signal exists
                    if not data.get('has_signal', False):
                        st.warning(f"⚠️ {data.get('message', 'No signal data available for this trade')}")
                        if 'trade_details' in data:
                            td = data['trade_details']
                            st.info(f"Trade: {td.get('symbol')} | {td.get('direction')} | Entry: {td.get('entry_price'):.5f}")
                        return

                    # Signal Overview Section
                    st.subheader("📋 Signal Overview")
                    sig = data['signal_overview']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        pair_display = sig['pair'] or sig['epic'].replace('CS.D.', '').replace('.MINI.IP', '')
                        st.metric("Pair", pair_display)
                        direction_emoji = "🟢" if sig['direction'] == "BUY" else "🔴"
                        st.metric("Direction", f"{direction_emoji} {sig['direction']}")

                    with col2:
                        st.metric("Strategy", sig['strategy'] or "Unknown")
                        st.metric("Timeframe", sig['timeframe'] or "N/A")

                    with col3:
                        conf_pct = sig['confidence_score'] * 100 if sig['confidence_score'] < 1 else sig['confidence_score']
                        st.metric("Confidence", f"{conf_pct:.1f}%")
                        st.metric("Price at Signal", f"{sig['price_at_signal']:.5f}")

                    with col4:
                        st.metric("Spread", f"{sig['spread_pips']:.1f} pips" if sig['spread_pips'] else "N/A")
                        st.metric("Session", sig['market_session'] or "Unknown")

                    # Detect strategy type from signal overview
                    strategy_name = (sig.get('strategy') or '').upper()
                    is_smc_simple = 'SMC_SIMPLE' in strategy_name

                    # Get raw strategy indicators for SMC_SIMPLE display
                    raw_data = data.get('raw_data', {})
                    strategy_indicators = raw_data.get('strategy_indicators', {})

                    # Smart Money / Strategy Validation Section
                    if is_smc_simple:
                        st.subheader("📊 SMC Simple - 3-Tier Analysis")

                        # Extract tier data from raw strategy_indicators
                        tier1 = strategy_indicators.get('tier1_ema', {})
                        tier2 = strategy_indicators.get('tier2_swing', {})
                        tier3 = strategy_indicators.get('tier3_entry', {})

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # TIER 1: EMA Bias
                            ema_dir = tier1.get('direction', 'Unknown')
                            dir_emoji = "📈" if ema_dir == 'BULL' else ("📉" if ema_dir == 'BEAR' else "➡️")
                            dir_bg = '#d4edda' if ema_dir in ['BULL', 'BEAR'] else '#f8d7da'
                            ema_value = tier1.get('ema_value', 0)
                            ema_distance = tier1.get('distance_pips', 0)
                            st.markdown(f"""
                            <div class="metric-card" style="background: {dir_bg};">
                                <h4>{dir_emoji} TIER 1: 4H EMA Bias</h4>
                                <p><strong>Direction:</strong> {ema_dir}</p>
                                <p><strong>EMA {tier1.get('ema_period', 50)}:</strong> {ema_value:.5f}</p>
                                <p><strong>Distance:</strong> {ema_distance:.1f} pips</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            # TIER 2: Swing Break
                            swing_level = tier2.get('swing_level', 0)
                            body_confirmed = tier2.get('body_close_confirmed', False)
                            vol_confirmed = tier2.get('volume_confirmed', False)
                            body_emoji = "✅" if body_confirmed else "❌"
                            vol_emoji = "✅" if vol_confirmed else "⚪"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 TIER 2: Swing Break</h4>
                                <p><strong>Swing Level:</strong> {swing_level:.5f}</p>
                                <p><strong>Body Close:</strong> {body_emoji} {'Confirmed' if body_confirmed else 'Not confirmed'}</p>
                                <p><strong>Volume:</strong> {vol_emoji} {'Confirmed' if vol_confirmed else 'No spike'}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            # TIER 3: Entry Timing
                            entry_price = tier3.get('entry_price', 0)
                            pullback = tier3.get('pullback_depth', 0)
                            fib_zone = tier3.get('fib_zone', 'N/A')
                            in_optimal = tier3.get('in_optimal_zone', False)
                            optimal_emoji = "🎯" if in_optimal else "⚪"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🎯 TIER 3: Entry Timing</h4>
                                <p><strong>Entry:</strong> {entry_price:.5f}</p>
                                <p><strong>Pullback:</strong> {pullback*100:.1f}%</p>
                                <p><strong>Fib Zone:</strong> {fib_zone}</p>
                                <p><strong>Optimal:</strong> {optimal_emoji} {'Yes' if in_optimal else 'No'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Original SMC_STRUCTURE display
                        st.subheader("🧠 Smart Money Validation")
                        smc = data['smart_money_analysis']

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            validated = smc['validated']
                            val_emoji = "✅" if validated else "❌"
                            val_bg = '#d4edda' if validated else '#f8d7da'
                            st.markdown(f"""
                            <div class="metric-card" style="background: {val_bg};">
                                <h4>{val_emoji} SMC Validated</h4>
                                <p><strong>Type:</strong> {smc['type'] or 'Unknown'}</p>
                                <p><strong>Score:</strong> {smc['score']*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            ms = smc['market_structure']
                            structure = ms['current_structure'] or ms['structure_type'] or 'Unknown'
                            struct_emoji = "📈" if 'bullish' in structure.lower() else ("📉" if 'bearish' in structure.lower() else "➡️")
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{struct_emoji} Market Structure</h4>
                                <p><strong>Type:</strong> {structure.upper()}</p>
                                <p><strong>Trend Strength:</strong> {ms['trend_strength']*100:.0f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            # Structure breaks
                            breaks = ms.get('structure_breaks', [])
                            break_count = len(breaks) if isinstance(breaks, list) else 0
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 Structure Details</h4>
                                <p><strong>Swing High:</strong> {ms['swing_high']:.5f}</p>
                                <p><strong>Swing Low:</strong> {ms['swing_low']:.5f}</p>
                                <p><strong>Breaks:</strong> {break_count}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Confluence Factors Section
                    st.subheader("🎯 Confluence Factors")
                    conf = data['confluence_factors']

                    factors_present = conf['factors_present']
                    factors_total = conf['factors_total']
                    conf_score = conf['total_score']

                    # Progress bar for confluence
                    if factors_total > 0:
                        progress = factors_present / factors_total
                        st.progress(progress, text=f"Confluence: {factors_present}/{factors_total} factors present")

                    # Display individual factors
                    if conf['factors']:
                        cols = st.columns(min(len(conf['factors']), 6))
                        for i, factor in enumerate(conf['factors']):
                            with cols[i % len(cols)]:
                                emoji = "✅" if factor['present'] else "❌"
                                bg = '#d4edda' if factor['present'] else '#fff3cd'
                                st.markdown(f"""
                                <div style="background: {bg}; padding: 0.5rem; border-radius: 5px; margin: 0.25rem 0; text-align: center;">
                                    {emoji} {factor['name']}
                                </div>
                                """, unsafe_allow_html=True)

                    # Entry Timing Section
                    st.subheader("⏱️ Entry Timing Quality")
                    timing = data['entry_timing']

                    if is_smc_simple:
                        # SMC_SIMPLE: Display Fibonacci pullback and entry quality
                        tier3 = strategy_indicators.get('tier3_entry', {})
                        risk_mgmt = strategy_indicators.get('risk_management', {})

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            pullback = tier3.get('pullback_depth', 0)
                            # Determine pullback quality based on Fib levels
                            if 0.382 <= pullback <= 0.500:
                                pullback_quality = "Optimal (38.2-50%)"
                                pullback_color = '#28a745'
                            elif 0.236 <= pullback <= 0.618:
                                pullback_quality = "Good (Fib zone)"
                                pullback_color = '#ffc107'
                            else:
                                pullback_quality = "Extended"
                                pullback_color = '#dc3545'
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📐 Pullback Depth</h4>
                                <h3 style="color: {pullback_color};">{pullback*100:.1f}%</h3>
                                <small>{pullback_quality}</small>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            fib_zone = tier3.get('fib_zone', 'N/A')
                            in_optimal = tier3.get('in_optimal_zone', False)
                            zone_emoji = "🎯" if in_optimal else "⚪"
                            zone_bg = '#d4edda' if in_optimal else '#f0f0f0'
                            st.markdown(f"""
                            <div class="metric-card" style="background: {zone_bg};">
                                <h4>{zone_emoji} Fibonacci Zone</h4>
                                <p><strong>Range:</strong> {fib_zone}</p>
                                <p><strong>Optimal:</strong> {'Yes' if in_optimal else 'No'}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            rr_ratio = risk_mgmt.get('rr_ratio', 0)
                            rr_color = '#28a745' if rr_ratio >= 2 else ('#ffc107' if rr_ratio >= 1.5 else '#dc3545')
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📐 Risk:Reward</h4>
                                <h3 style="color: {rr_color};">{rr_ratio:.2f}</h3>
                                <small>{'Excellent' if rr_ratio >= 2 else ('Good' if rr_ratio >= 1.5 else 'Fair')}</small>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            entry_quality = timing.get('entry_quality_score', 0) * 100
                            quality_color = '#28a745' if entry_quality >= 70 else ('#ffc107' if entry_quality >= 50 else '#dc3545')
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>⭐ Entry Quality</h4>
                                <h3 style="color: {quality_color};">{entry_quality:.0f}%</h3>
                                <small>Overall score</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Original SMC_STRUCTURE: Premium/Discount zones
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            zone = timing['premium_discount_zone']
                            zone_emoji = "🟢" if zone == 'discount' else ("🔴" if zone == 'premium' else "🟡")
                            zone_quality = "Good for BUY" if zone == 'discount' else ("Good for SELL" if zone == 'premium' else "Neutral")
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📍 Price Zone</h4>
                                <h3>{zone_emoji} {zone.upper()}</h3>
                                <small>{zone_quality}</small>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            htf_aligned = timing['htf_aligned']
                            htf_emoji = "✅" if htf_aligned else "❌"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📈 HTF Alignment</h4>
                                <h3>{htf_emoji} {'Aligned' if htf_aligned else 'Not Aligned'}</h3>
                                <small>Structure: {timing['htf_structure']}</small>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            mtf_ratio = timing['mtf_alignment_ratio'] * 100
                            htf_strength = timing.get('htf_strength', 0) * 100
                            # Use HTF strength if MTF not available (SMC strategy uses HTF)
                            if mtf_ratio > 0:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>🔄 MTF Consensus</h4>
                                    <h3>{mtf_ratio:.0f}%</h3>
                                    <small>Timeframes aligned</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Show HTF strength for SMC trades
                                htf_color = '#28a745' if htf_strength >= 60 else ('#ffc107' if htf_strength >= 40 else '#dc3545')
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>📊 HTF Strength</h4>
                                    <h3 style="color: {htf_color};">{htf_strength:.0f}%</h3>
                                    <small>4H trend strength</small>
                                </div>
                                """, unsafe_allow_html=True)

                        with col4:
                            entry_quality = timing['entry_quality_score'] * 100
                            quality_color = '#28a745' if entry_quality >= 70 else ('#ffc107' if entry_quality >= 50 else '#dc3545')
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>⭐ Entry Quality</h4>
                                <h3 style="color: {quality_color};">{entry_quality:.0f}%</h3>
                                <small>Overall score</small>
                            </div>
                            """, unsafe_allow_html=True)

                    # Technical Context Section
                    st.subheader("📊 Technical Context at Entry")
                    tech = data['technical_context']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        # EMA data from new structure
                        ema_50 = tech.get('ema_50', 0)
                        ema_200 = tech.get('ema_200', 0)
                        price_pos = tech.get('price_vs_ema_50', 'unknown')
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📈 EMA Analysis</h4>
                            <p><strong>EMA 50:</strong> {ema_50:.5f}</p>
                            <p><strong>EMA 200:</strong> {ema_200:.5f}</p>
                            <p><strong>Price vs EMA50:</strong> {price_pos.upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        macd = tech.get('macd', {})
                        macd_dir = macd.get('direction', 'unknown')
                        macd_emoji = "📈" if macd_dir == 'bullish' else "📉"
                        hist_val = macd.get('histogram', 0)
                        hist_color = '#28a745' if hist_val > 0 else '#dc3545'
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{macd_emoji} MACD</h4>
                            <p><strong>Line:</strong> {macd.get('line', 0):.6f}</p>
                            <p><strong>Signal:</strong> {macd.get('signal', 0):.6f}</p>
                            <p style="color: {hist_color};"><strong>Histogram:</strong> {hist_val:.6f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        # Bollinger Bands
                        bb = tech.get('bollinger_bands', {})
                        if bb and bb.get('upper', 0) > 0:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 Bollinger Bands</h4>
                                <p><strong>Upper:</strong> {bb.get('upper', 0):.5f}</p>
                                <p><strong>Middle:</strong> {bb.get('middle', 0):.5f}</p>
                                <p><strong>Lower:</strong> {bb.get('lower', 0):.5f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            rsi = tech.get('rsi', 50)
                            rsi_zone = tech.get('rsi_zone', 'neutral')
                            rsi_color = '#dc3545' if rsi_zone == 'overbought' else ('#28a745' if rsi_zone == 'oversold' else '#6c757d')
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 RSI</h4>
                                <h3 style="color: {rsi_color};">{rsi:.1f}</h3>
                                <small>{rsi_zone.upper()}</small>
                            </div>
                            """, unsafe_allow_html=True)

                    with col4:
                        vol_ratio = tech.get('volume_ratio', 0)
                        vol_conf = tech.get('volume_confirmation', False)
                        vol_emoji = "✅" if vol_conf else "⚪"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📊 Volume & ATR</h4>
                            <p><strong>Ratio:</strong> {vol_ratio:.2f}x</p>
                            <p><strong>Confirmed:</strong> {vol_emoji}</p>
                            <p><strong>ATR:</strong> {tech.get('atr', 0):.5f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Pattern Analysis (if present)
                    pattern = data.get('pattern_analysis')
                    if pattern and pattern.get('pattern_type'):
                        st.subheader("🕯️ Price Pattern Detected")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            pattern_type = pattern.get('pattern_type', 'unknown')
                            pattern_emoji = "🟢" if 'bullish' in pattern_type.lower() else ("🔴" if 'bearish' in pattern_type.lower() else "🟡")
                            st.metric("Pattern", f"{pattern_emoji} {pattern_type.replace('_', ' ').title()}")

                        with col2:
                            strength = pattern.get('pattern_strength', 0) * 100
                            st.metric("Strength", f"{strength:.1f}%")

                        with col3:
                            st.metric("Rejection Level", f"{pattern.get('rejection_level', 0):.5f}")

                        with col4:
                            st.metric("Entry Price", f"{pattern.get('entry_price', 0):.5f}")

                    # Support/Resistance Analysis
                    sr = data.get('support_resistance', {})
                    if sr and sr.get('level_price', 0) > 0:
                        st.subheader("🎯 Support/Resistance Analysis")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            level_type = sr.get('level_type', 'unknown')
                            level_emoji = "🟢" if level_type == 'support' else "🔴"
                            st.metric("Key Level", f"{level_emoji} {level_type.upper()}")

                        with col2:
                            st.metric("Level Price", f"{sr.get('level_price', 0):.5f}")

                        with col3:
                            strength = sr.get('level_strength', 0) * 100
                            st.metric("Strength", f"{strength:.0f}%")

                        with col4:
                            st.metric("Touch Count", sr.get('touch_count', 0))

                        # S/R distances
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"📗 **Nearest Support:** {sr.get('nearest_support', 0):.5f} ({sr.get('distance_to_support_pips', 0):.1f} pips)")
                        with col2:
                            st.info(f"📕 **Nearest Resistance:** {sr.get('nearest_resistance', 0):.5f} ({sr.get('distance_to_resistance_pips', 0):.1f} pips)")

                    # Risk/Reward Section
                    st.subheader("💰 Risk/Reward Setup")
                    rr = data.get('risk_reward', {})

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        initial_rr = rr.get('initial_rr', 0)
                        rr_color = '#28a745' if initial_rr >= 2 else ('#ffc107' if initial_rr >= 1.5 else '#dc3545')
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📐 Initial R:R Ratio</h4>
                            <h2 style="color: {rr_color};">{initial_rr:.2f}</h2>
                            <small>{'Excellent' if initial_rr >= 2 else ('Good' if initial_rr >= 1.5 else 'Poor')}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📏 Risk/Reward Pips</h4>
                            <p><strong>Risk:</strong> {rr.get('risk_pips', 0):.1f} pips</p>
                            <p><strong>Reward:</strong> {rr.get('reward_pips', 0):.1f} pips</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📍 Trade Levels</h4>
                            <p><strong>Entry:</strong> {rr.get('entry_price', 0):.5f}</p>
                            <p><strong>SL:</strong> {rr.get('stop_loss', 0):.5f}</p>
                            <p><strong>TP:</strong> {rr.get('take_profit', 0):.5f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        partial_tp = rr.get('partial_tp', 0)
                        partial_pct = rr.get('partial_percent', 0)
                        if partial_tp > 0:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🎯 Partial Take Profit</h4>
                                <p><strong>Level:</strong> {partial_tp:.5f}</p>
                                <p><strong>Size:</strong> {partial_pct:.0f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 No Partial TP</h4>
                                <p>Full position to TP</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Confidence Breakdown (if present)
                    conf_breakdown = data.get('confidence_breakdown')
                    # Also check raw strategy indicators for SMC_SIMPLE breakdown
                    if not conf_breakdown and is_smc_simple:
                        conf_breakdown = strategy_indicators.get('confidence_breakdown', {})

                    if conf_breakdown:
                        st.subheader("📊 Confidence Breakdown")

                        if is_smc_simple:
                            # SMC_SIMPLE uses different breakdown: ema_alignment, volume_bonus, pullback_quality, rr_quality, fib_accuracy
                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                total = conf_breakdown.get('total', 0) * 100
                                total_color = '#28a745' if total >= 60 else ('#ffc107' if total >= 50 else '#dc3545')
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>📊 Total</h4>
                                    <h3 style="color: {total_color};">{total:.1f}%</h3>
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                ema = conf_breakdown.get('ema_alignment', 0) * 100
                                st.metric("EMA Alignment", f"{ema:.1f}%")

                            with col3:
                                vol = conf_breakdown.get('volume_bonus', 0) * 100
                                vol_emoji = "✅" if vol > 10 else "⚪"
                                st.metric("Volume Bonus", f"{vol_emoji} {vol:.1f}%")

                            with col4:
                                pullback = conf_breakdown.get('pullback_quality', 0) * 100
                                st.metric("Pullback Quality", f"{pullback:.1f}%")

                            with col5:
                                rr = conf_breakdown.get('rr_quality', 0) * 100
                                st.metric("R:R Quality", f"{rr:.1f}%")

                            # Show Fib accuracy in an additional row
                            fib_acc = conf_breakdown.get('fib_accuracy', 0) * 100
                            if fib_acc > 0:
                                st.info(f"📐 **Fibonacci Accuracy:** {fib_acc:.1f}% (how close to 38.2% optimal zone)")
                        else:
                            # Original SMC_STRUCTURE breakdown
                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                total = conf_breakdown.get('total', 0) * 100
                                st.metric("Total", f"{total:.1f}%")

                            with col2:
                                htf = conf_breakdown.get('htf_score', 0) * 100
                                st.metric("HTF Score", f"{htf:.1f}%")

                            with col3:
                                pattern_score = conf_breakdown.get('pattern_score', 0) * 100
                                st.metric("Pattern Score", f"{pattern_score:.1f}%")

                            with col4:
                                sr_score = conf_breakdown.get('sr_score', 0) * 100
                                st.metric("S/R Score", f"{sr_score:.1f}%")

                            with col5:
                                rr_score = conf_breakdown.get('rr_score', 0) * 100
                                st.metric("R:R Score", f"{rr_score:.1f}%")

                    # Order Block Details (if present) - not used by SMC_SIMPLE
                    if data.get('order_block_details') and not is_smc_simple:
                        st.subheader("🧱 Order Block Details")
                        ob = data['order_block_details']

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            ob_type = ob['type']
                            ob_emoji = "🟢" if 'bullish' in ob_type.lower() else "🔴"
                            st.metric("Type", f"{ob_emoji} {ob_type.upper()}")

                        with col2:
                            strength = ob['strength']
                            strength_colors = {'weak': '#dc3545', 'medium': '#ffc107', 'strong': '#28a745', 'very_strong': '#198754'}
                            st.metric("Strength", strength.upper())

                        with col3:
                            st.metric("Tested Count", ob['tested_count'])

                        with col4:
                            valid_emoji = "✅" if ob['still_valid'] else "❌"
                            st.metric("Still Valid", valid_emoji)

                    # FVG Details (if present) - not used by SMC_SIMPLE
                    if data.get('fair_value_gap_details') and not is_smc_simple:
                        st.subheader("📊 Fair Value Gap Details")
                        fvg = data['fair_value_gap_details']

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            fvg_type = fvg['type']
                            fvg_emoji = "🟢" if 'bullish' in fvg_type.lower() else "🔴"
                            st.metric("Type", f"{fvg_emoji} {fvg_type.upper()}")

                        with col2:
                            st.metric("Size", f"{fvg['size_pips']:.1f} pips")

                        with col3:
                            status = fvg['status']
                            status_emoji = "✅" if status == 'active' else ("⚠️" if status == 'filled' else "❌")
                            st.metric("Status", f"{status_emoji} {status.upper()}")

                        with col4:
                            st.metric("Confluence", f"{fvg['confluence_score']*100:.0f}%")

                    # Market Intelligence (if present)
                    market_intel = data.get('market_intelligence')
                    if market_intel:
                        st.subheader("🧠 Market Intelligence at Entry")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            regime = market_intel.get('regime', {})
                            regime_name = regime.get('dominant', 'unknown').replace('_', ' ').title()
                            regime_conf = regime.get('confidence', 0) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 Market Regime</h4>
                                <h3>{regime_name}</h3>
                                <small>Confidence: {regime_conf:.0f}%</small>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            session = market_intel.get('session', {})
                            session_name = session.get('current', 'unknown').replace('_', ' ').title()
                            volatility = session.get('volatility', 'unknown')
                            vol_emoji = "🔴" if volatility == 'high' else ("🟡" if volatility == 'medium' else "🟢")
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>⏰ Trading Session</h4>
                                <h3>{session_name}</h3>
                                <small>{vol_emoji} Volatility: {volatility.upper()}</small>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            strength = market_intel.get('market_strength', {})
                            trend_str = strength.get('trend_strength', 0) * 100
                            bias = strength.get('market_bias', 'neutral')
                            bias_emoji = "📈" if bias == 'bullish' else ("📉" if bias == 'bearish' else "➡️")
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{bias_emoji} Market Bias</h4>
                                <h3>{bias.upper()}</h3>
                                <small>Trend Strength: {trend_str:.0f}%</small>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            intel_applied = market_intel.get('intelligence_applied', False)
                            intel_emoji = "✅" if intel_applied else "❌"
                            risk_level = session.get('risk_level', 'unknown')
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{intel_emoji} Intelligence Applied</h4>
                                <p><strong>Risk Level:</strong> {risk_level.upper()}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Trade Outcome Correlation
                    st.subheader("📈 Trade Outcome Correlation")
                    outcome = data['trade_outcome']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        status = outcome['status']
                        status_emoji = "✅" if status == 'closed' else ("🔄" if status == 'tracking' else "⏳")
                        st.metric("Status", f"{status_emoji} {status.upper()}")

                    with col2:
                        is_winner = outcome['is_winner']
                        if is_winner is not None:
                            result_emoji = "🏆" if is_winner else "❌"
                            result_text = "WIN" if is_winner else "LOSS"
                            result_color = '#28a745' if is_winner else '#dc3545'
                            st.markdown(f"""
                            <div class="metric-card" style="background: {'#d4edda' if is_winner else '#f8d7da'};">
                                <h4>Result</h4>
                                <h2 style="color: {result_color};">{result_emoji} {result_text}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.metric("Result", "Pending")

                    with col3:
                        pnl = outcome['profit_loss']
                        pnl_color = '#28a745' if pnl > 0 else '#dc3545'
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>P&L</h4>
                            <h3 style="color: {pnl_color};">{'+' if pnl > 0 else ''}{pnl:.2f}</h3>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        pips = outcome['pips_gained']
                        pips_color = '#28a745' if pips > 0 else '#dc3545'
                        duration = outcome['duration_minutes']
                        duration_str = f"{duration // 60}h {duration % 60}m" if duration else "N/A"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Pips & Duration</h4>
                            <p style="color: {pips_color};"><strong>{'+' if pips > 0 else ''}{pips:.1f} pips</strong></p>
                            <small>Duration: {duration_str}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # Claude Analysis (if present)
                    if data.get('claude_analysis') and data['claude_analysis'].get('analysis_text'):
                        st.subheader("🤖 Claude AI Analysis")
                        claude = data['claude_analysis']

                        col1, col2, col3 = st.columns([1, 1, 2])

                        with col1:
                            if claude['score']:
                                st.metric("Score", f"{claude['score']}/100")

                        with col2:
                            if claude['decision']:
                                decision_emoji = "✅" if claude['approved'] else "❌"
                                st.metric("Decision", f"{decision_emoji} {claude['decision']}")

                        with col3:
                            if claude['reason']:
                                st.info(f"**Reason:** {claude['reason']}")

                        if claude['analysis_text']:
                            with st.expander("📝 Full Analysis"):
                                st.markdown(claude['analysis_text'])

                    # Raw Data Expander
                    with st.expander("🔍 View Raw Signal Data"):
                        st.json(data.get('raw_data', {}))

                elif response.status_code == 404:
                    st.error(f"❌ Trade {trade_id} not found in database")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"❌ Error analyzing signal: {e}")
                import traceback
                st.code(traceback.format_exc())

    def _render_outcome_analysis(self, trade_id: int, analyze_btn: bool):
        """Render the trade outcome analysis sub-tab - WHY the trade won or lost"""
        st.subheader("📚 Trade Outcome Analysis")
        st.markdown("*Understand WHY this trade won or lost - Learn from every trade*")

        if analyze_btn or trade_id:
            try:
                import requests

                # Call the FastAPI outcome analysis endpoint
                headers = {
                    "X-APIM-Gateway": "verified",
                    "X-API-KEY": "436abe054a074894a0517e5172f0e5b6"
                }

                with st.spinner(f"Analyzing outcome for trade {trade_id}..."):
                    response = requests.get(
                        f"http://fastapi-dev:8000/api/trade-analysis/outcome/{trade_id}",
                        headers=headers
                    )

                if response.status_code == 200:
                    data = response.json()

                    # Check if trade is still open
                    if data.get('status') == 'TRADE_STILL_OPEN':
                        st.warning(f"⚠️ {data.get('message', 'Trade is still open')}")
                        if 'trade_details' in data:
                            td = data['trade_details']
                            st.info(f"Trade: {td.get('symbol')} | {td.get('direction')} | Entry: {td.get('entry_price'):.5f} | Status: {td.get('status')}")
                        return

                    # ===== OUTCOME SUMMARY SECTION =====
                    st.subheader("📊 Outcome Summary")
                    summary = data['outcome_summary']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        result = summary['result']
                        if result == "WIN":
                            result_bg = '#d4edda'
                            result_emoji = "🏆"
                            result_color = '#28a745'
                        elif result == "LOSS":
                            result_bg = '#f8d7da'
                            result_emoji = "❌"
                            result_color = '#dc3545'
                        else:
                            result_bg = '#fff3cd'
                            result_emoji = "⚖️"
                            result_color = '#856404'
                        st.markdown(f"""
                        <div class="metric-card" style="background: {result_bg}; text-align: center;">
                            <h4>Result</h4>
                            <h2 style="color: {result_color};">{result_emoji} {result}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        pips = summary['pips_gained']
                        profit_loss = summary['profit_loss']
                        pnl_color = '#28a745' if profit_loss > 0 else '#dc3545'
                        # Show pips if available, otherwise show P&L prominently
                        if pips != 0:
                            st.markdown(f"""
                            <div class="metric-card" style="text-align: center;">
                                <h4>P&L</h4>
                                <h3 style="color: {pnl_color};">{'+' if pips > 0 else ''}{pips:.1f} pips</h3>
                                <small>{'+' if profit_loss > 0 else ''}{profit_loss:.2f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card" style="text-align: center;">
                                <h4>P&L</h4>
                                <h3 style="color: {pnl_color};">{'+' if profit_loss > 0 else ''}{profit_loss:.2f}</h3>
                                <small>pips not recorded</small>
                            </div>
                            """, unsafe_allow_html=True)

                    with col3:
                        r_mult = summary['r_multiple']
                        # If R-multiple is 0 but we have MFE/MAE, calculate approximate R
                        if r_mult == 0 and summary.get('mfe_pips', 0) > 0:
                            # Use MFE/MAE ratio as proxy for trade quality
                            mfe_mae_ratio = summary.get('mfe_mae_ratio', 0)
                            if result == "WIN":
                                r_display = f"~{mfe_mae_ratio:.1f}R*"
                                r_color = '#28a745'
                            elif result == "LOSS":
                                r_display = f"-1R*"
                                r_color = '#dc3545'
                            else:
                                r_display = "0R"
                                r_color = '#856404'
                            st.markdown(f"""
                            <div class="metric-card" style="text-align: center;">
                                <h4>R-Multiple</h4>
                                <h3 style="color: {r_color};">{r_display}</h3>
                                <small>*estimated</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            r_color = '#28a745' if r_mult > 0 else '#dc3545'
                            st.markdown(f"""
                            <div class="metric-card" style="text-align: center;">
                                <h4>R-Multiple</h4>
                                <h3 style="color: {r_color};">{'+' if r_mult > 0 else ''}{r_mult:.2f}R</h3>
                            </div>
                            """, unsafe_allow_html=True)

                    with col4:
                        exit_type = summary['exit_type']
                        exit_emoji = "🎯" if exit_type == "TP_HIT" else ("🛑" if exit_type == "SL_HIT" else "📊")
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Exit Type</h4>
                            <h3>{exit_emoji} {exit_type.replace('_', ' ')}</h3>
                            <small>Duration: {summary['duration_display']}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== MFE/MAE ANALYSIS SECTION =====
                    st.subheader("📈 MFE/MAE Analysis")
                    st.markdown("*Maximum Favorable Excursion (MFE) vs Maximum Adverse Excursion (MAE)*")

                    price_action = data['price_action_analysis']
                    mfe = price_action['mfe']
                    mae = price_action['mae']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        mfe_pips = mfe['pips']
                        st.markdown(f"""
                        <div class="metric-card" style="background: #d4edda;">
                            <h4>📈 MFE (Max Profit)</h4>
                            <h3 style="color: #28a745;">+{mfe_pips:.1f} pips</h3>
                            <small>Peak: {mfe['time_to_peak_minutes']}m</small>
                            <p><small>{mfe['percentage_of_tp']:.0f}% of TP distance</small></p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        mae_pips = mae['pips']
                        st.markdown(f"""
                        <div class="metric-card" style="background: #f8d7da;">
                            <h4>📉 MAE (Max Drawdown)</h4>
                            <h3 style="color: #dc3545;">-{mae_pips:.1f} pips</h3>
                            <small>Trough: {mae['time_to_trough_minutes']}m</small>
                            <p><small>{mae['percentage_of_sl']:.0f}% of SL distance</small></p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        ratio = price_action['mfe_mae_ratio']
                        ratio_color = '#28a745' if ratio > 2 else ('#ffc107' if ratio > 1 else '#dc3545')
                        ratio_verdict = "Excellent" if ratio > 3 else ("Good" if ratio > 2 else ("Fair" if ratio > 1 else "Poor"))
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>⚖️ MFE/MAE Ratio</h4>
                            <h3 style="color: {ratio_color};">{ratio:.2f}</h3>
                            <small>{ratio_verdict}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        initial = price_action['initial_move']
                        initial_emoji = "🟢" if initial == "FAVORABLE" else ("🔴" if initial == "ADVERSE" else "🟡")
                        reversal = price_action['immediate_reversal']
                        reversal_emoji = "⚠️" if reversal else "✅"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🚀 Initial Move</h4>
                            <h3>{initial_emoji} {initial}</h3>
                            <small>{reversal_emoji} {'Immediate Reversal' if reversal else 'No Quick Reversal'}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== ENTRY QUALITY SECTION =====
                    st.subheader("✅ Entry Quality Assessment")
                    entry_q = data['entry_quality_assessment']

                    score = entry_q['score']
                    verdict = entry_q['verdict']
                    verdict_color = '#28a745' if verdict == "GOOD_ENTRY" else ('#ffc107' if verdict in ["AVERAGE_ENTRY", "BELOW_AVERAGE_ENTRY"] else '#dc3545')
                    verdict_bg = '#d4edda' if verdict == "GOOD_ENTRY" else ('#fff3cd' if verdict in ["AVERAGE_ENTRY", "BELOW_AVERAGE_ENTRY"] else '#f8d7da')

                    # Show overall score
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card" style="background: {verdict_bg}; text-align: center;">
                            <h4>Entry Score</h4>
                            <h2 style="color: {verdict_color};">{score:.0f}/100</h2>
                            <p><strong>{verdict.replace('_', ' ')}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        # Show factor breakdown
                        factors = entry_q.get('factors', [])
                        if factors:
                            for factor in factors:
                                points = factor.get('points', 0)
                                max_pts = factor.get('max_points', 0)
                                pct = (points / max_pts * 100) if max_pts > 0 else 0
                                bar_color = '#28a745' if pct >= 70 else ('#ffc107' if pct >= 40 else '#dc3545')
                                st.markdown(f"""
                                <div style="margin-bottom: 0.5rem;">
                                    <div style="display: flex; justify-content: space-between;">
                                        <span>{factor['name']}: {factor['value']}</span>
                                        <span><strong>+{points:.0f}/{max_pts}</strong></span>
                                    </div>
                                    <div style="background: #e9ecef; border-radius: 4px; height: 8px;">
                                        <div style="background: {bar_color}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    # ===== EXIT QUALITY SECTION =====
                    st.subheader("🎯 Exit Quality Assessment")
                    exit_q = data['exit_quality_assessment']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        exit_verdict = exit_q['verdict']
                        exit_v_color = '#28a745' if exit_verdict in ["OPTIMAL_EXIT", "EXCELLENT_EXIT", "GOOD_EXIT"] else ('#ffc107' if exit_verdict == "ACCEPTABLE_EXIT" else '#dc3545')
                        exit_v_bg = '#d4edda' if exit_verdict in ["OPTIMAL_EXIT", "EXCELLENT_EXIT", "GOOD_EXIT"] else ('#fff3cd' if exit_verdict == "ACCEPTABLE_EXIT" else '#f8d7da')
                        st.markdown(f"""
                        <div class="metric-card" style="background: {exit_v_bg}; text-align: center;">
                            <h4>Exit Verdict</h4>
                            <h3 style="color: {exit_v_color};">{exit_verdict.replace('_', ' ')}</h3>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        efficiency = exit_q['exit_efficiency_pct']
                        eff_color = '#28a745' if efficiency >= 70 else ('#ffc107' if efficiency >= 40 else '#dc3545')
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Exit Efficiency</h4>
                            <h3 style="color: {eff_color};">{efficiency:.0f}%</h3>
                            <small>of MFE captured</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        actual_pips = exit_q['actual_pips']
                        # Show actual pips or P&L if pips is 0
                        if actual_pips != 0:
                            actual_display = f"{'+' if actual_pips > 0 else ''}{actual_pips:.1f} pips"
                        else:
                            actual_pnl = summary['profit_loss']
                            actual_display = f"{'+' if actual_pnl > 0 else ''}{actual_pnl:.2f}"
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Comparison</h4>
                            <p>MFE: <strong>+{exit_q['mfe_pips']:.1f}</strong> pips</p>
                            <p>Actual: <strong>{actual_display}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        missed = exit_q['missed_profit_pips']
                        missed_color = '#dc3545' if missed > 10 else ('#ffc107' if missed > 5 else '#28a745')
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Missed Profit</h4>
                            <h3 style="color: {missed_color};">{missed:.1f} pips</h3>
                            <small>{exit_q['verdict_details']}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== LEARNING INSIGHTS SECTION =====
                    st.subheader("🎓 Learning Insights")
                    insights = data['learning_insights']

                    result_color = '#28a745' if insights['trade_result'] == "WIN" else ('#dc3545' if insights['trade_result'] == "LOSS" else '#856404')
                    result_bg = '#d4edda' if insights['trade_result'] == "WIN" else ('#f8d7da' if insights['trade_result'] == "LOSS" else '#fff3cd')

                    # Primary Factor
                    st.markdown(f"""
                    <div class="metric-card" style="background: {result_bg}; border-left: 4px solid {result_color};">
                        <h4>🏆 PRIMARY FACTOR</h4>
                        <p style="font-size: 1.1rem;"><strong>{insights['primary_factor']}</strong></p>
                        <small>Pattern: <em>{insights.get('pattern_identified', 'Unknown').replace('_', ' ')}</em></small>
                    </div>
                    """, unsafe_allow_html=True)

                    # What went right/wrong
                    col1, col2 = st.columns(2)

                    with col1:
                        if insights.get('what_went_right'):
                            st.markdown("#### ✅ What Went Right")
                            for item in insights['what_went_right']:
                                st.success(f"• {item}")
                        elif insights['trade_result'] == "WIN":
                            st.markdown("#### ✅ What Went Right")
                            st.success("• Trade reached target as planned")

                    with col2:
                        if insights.get('what_went_wrong'):
                            st.markdown("#### ❌ What Went Wrong")
                            for item in insights['what_went_wrong']:
                                st.error(f"• {item}")

                    # Contributing Factors
                    if insights.get('contributing_factors'):
                        st.markdown("#### 📋 Contributing Factors")
                        for factor in insights['contributing_factors']:
                            st.info(f"• {factor}")

                    # Improvement Suggestions
                    if insights.get('improvement_suggestions'):
                        st.markdown("#### 💡 Improvement Suggestions")
                        for suggestion in insights['improvement_suggestions']:
                            st.warning(f"• {suggestion}")

                    # Key Takeaway
                    if insights.get('key_takeaway'):
                        st.markdown("---")
                        st.markdown(f"""
                        <div style="background: #e7f1ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0066cc;">
                            <h4>🔑 KEY TAKEAWAY</h4>
                            <p style="font-size: 1.1rem; margin: 0;"><em>{insights['key_takeaway']}</em></p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== TRADE DETAILS SECTION =====
                    with st.expander("📋 Trade Details"):
                        td = data['trade_details']
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Symbol", td['symbol'].replace('CS.D.', '').replace('.MINI.IP', ''))
                            st.metric("Direction", td['direction'])
                            st.metric("Deal ID", td.get('deal_id', 'N/A'))

                        with col2:
                            st.metric("Entry Price", f"{td['entry_price']:.5f}")
                            st.metric("Exit Price", f"{td['exit_price']:.5f}" if td.get('exit_price') else "N/A")
                            st.metric("Moved to BE", "Yes" if td['moved_to_breakeven'] else "No")

                        with col3:
                            st.metric("Stop Loss", f"{td['sl_price']:.5f}")
                            st.metric("Take Profit", f"{td['tp_price']:.5f}")
                            st.metric("Alert ID", td.get('alert_id', 'N/A'))

                        st.write(f"**Opened:** {td.get('opened_at', 'N/A')}")
                        st.write(f"**Closed:** {td.get('closed_at', 'N/A')}")

                    # ===== MARKET CONTEXT =====
                    market_ctx = data.get('market_context', {})
                    if market_ctx:
                        with st.expander("🌐 Market Context at Entry"):
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Market Regime", market_ctx.get('regime_at_entry', 'Unknown').replace('_', ' ').title())

                            with col2:
                                st.metric("Session", market_ctx.get('session_at_entry', 'Unknown').replace('_', ' ').title())

                            with col3:
                                htf = market_ctx.get('htf_aligned', False)
                                htf_emoji = "✅" if htf else "❌"
                                st.metric("HTF Aligned", f"{htf_emoji} {'Yes' if htf else 'No'}")

                            with col4:
                                vol = market_ctx.get('volatility_percentile', 0)
                                st.metric("Volatility", f"{vol*100:.0f}%")

                    # ===== CANDLE DATA (Optional Chart) =====
                    candle_data = data.get('candle_data', {})
                    if candle_data and candle_data.get('trade_candles'):
                        with st.expander("📊 Price Chart During Trade"):
                            import plotly.graph_objects as go

                            all_candles = (
                                candle_data.get('entry_context', []) +
                                candle_data.get('trade_candles', []) +
                                candle_data.get('exit_context', [])
                            )

                            if all_candles:
                                times = [c['timestamp'] for c in all_candles]
                                opens = [c['open'] for c in all_candles]
                                highs = [c['high'] for c in all_candles]
                                lows = [c['low'] for c in all_candles]
                                closes = [c['close'] for c in all_candles]

                                fig = go.Figure(data=[go.Candlestick(
                                    x=times,
                                    open=opens,
                                    high=highs,
                                    low=lows,
                                    close=closes,
                                    name='Price'
                                )])

                                # Add entry price line
                                td = data['trade_details']
                                fig.add_hline(y=td['entry_price'], line_dash="dash", line_color="blue",
                                             annotation_text="Entry")

                                # Add SL and TP lines
                                if td.get('sl_price'):
                                    fig.add_hline(y=td['sl_price'], line_dash="dash", line_color="red",
                                                 annotation_text="SL")
                                if td.get('tp_price'):
                                    fig.add_hline(y=td['tp_price'], line_dash="dash", line_color="green",
                                                 annotation_text="TP")

                                # Add MFE/MAE markers
                                if mfe.get('price') and mfe.get('timestamp'):
                                    fig.add_trace(go.Scatter(
                                        x=[mfe['timestamp']],
                                        y=[mfe['price']],
                                        mode='markers',
                                        marker=dict(size=12, color='green', symbol='triangle-up'),
                                        name=f"MFE (+{mfe['pips']:.1f} pips)"
                                    ))

                                if mae.get('price') and mae.get('timestamp'):
                                    fig.add_trace(go.Scatter(
                                        x=[mae['timestamp']],
                                        y=[mae['price']],
                                        mode='markers',
                                        marker=dict(size=12, color='red', symbol='triangle-down'),
                                        name=f"MAE (-{mae['pips']:.1f} pips)"
                                    ))

                                fig.update_layout(
                                    title=f"Trade {trade_id} - Price Action with MFE/MAE",
                                    xaxis_title="Time",
                                    yaxis_title="Price",
                                    height=500,
                                    showlegend=True
                                )

                                st.plotly_chart(fig, use_container_width=True)

                    # Raw Data Expander
                    with st.expander("🔍 View Raw Outcome Data"):
                        st.json(data)

                elif response.status_code == 404:
                    st.error(f"❌ Trade {trade_id} not found in database")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"❌ Error analyzing trade outcome: {e}")
                import traceback
                st.code(traceback.format_exc())

    def render_settings_debug_tab(self):
        """Render the settings and debug tab"""
        st.header("🔧 Settings & Debug")

        # Database connection status
        st.subheader("📡 Database Connection")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Test Database Connection"):
                conn = self.get_database_connection()
                if conn:
                    st.success("✅ Database connection successful!")
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT COUNT(*) FROM trade_log")
                            count = cursor.fetchone()[0]
                            st.info(f"Found {count} records in trade_log table")
                    except Exception as e:
                        st.warning(f"Connected but query failed: {e}")
                    finally:
                        conn.close()
                else:
                    st.error("❌ Database connection failed!")

        with col2:
            if st.button("🔄 Refresh Connection"):
                self.setup_database_connection()
                st.success("Connection settings refreshed!")

        # Connection debug information
        st.subheader("🔐 Connection Debug")

        if hasattr(st, 'secrets'):
            try:
                secrets_dict = st.secrets.to_dict()
                st.success(f"🟢 Secret sections: {list(secrets_dict.keys())}")

                if 'database' in secrets_dict:
                    db_keys = list(secrets_dict['database'].keys())
                    st.success(f"🟢 Database keys: {db_keys}")
                else:
                    st.warning("⚠️ No [database] section found")

            except Exception as e:
                st.error(f"❌ Error reading secrets: {e}")
        else:
            st.error("🔴 No secrets available")

        # Show current connection string (masked)
        if hasattr(self, 'conn_string') and self.conn_string:
            masked_conn = self.conn_string[:30] + "..." if len(self.conn_string) > 30 else self.conn_string
            st.success(f"✅ Connection string: {masked_conn}")
        else:
            st.error("❌ No connection string available")

        # Manual connection override
        st.subheader("⚙️ Manual Connection Override")

        col1, col2 = st.columns(2)

        with col1:
            manual_host = st.text_input("Host", value="postgres", key="manual_host")
            manual_port = st.number_input("Port", value=5432, min_value=1, max_value=65535, key="manual_port")
            manual_db = st.text_input("Database", value="forex", key="manual_db")

        with col2:
            manual_user = st.text_input("Username", value="postgres", key="manual_user")
            manual_pass = st.text_input("Password", value="postgres", type="password", key="manual_pass")

        if st.button("🧪 Test Manual Settings"):
            manual_conn_str = f"postgresql://{manual_user}:{manual_pass}@{manual_host}:{manual_port}/{manual_db}"
            try:
                test_conn = psycopg2.connect(manual_conn_str)
                test_conn.close()
                st.success("✅ Manual connection successful!")
                st.info("Consider updating your secrets.toml with these settings")
            except Exception as e:
                st.error(f"❌ Manual connection failed: {e}")

        # API Operations
        st.subheader("🔗 API Operations")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💰 Calculate Complete P&L", key="calc_pnl_btn"):
                try:
                    import requests
                    headers = {
                        "X-APIM-Gateway": "verified",
                        "X-API-KEY": "436abe054a074894a0517e5172f0e5b6",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "days_back": 7,
                        "update_trade_log": True,
                        "calculate_prices": True,
                        "include_detailed_results": False
                    }
                    response = requests.post("http://fastapi-dev:8000/api/trading/deals/calculate-complete-pnl", headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ P&L calculation completed!")
                        st.json(result)
                    else:
                        st.error(f"❌ API call failed: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ Error calling API: {e}")

        with col2:
            if st.button("🔄 Correlate Activities", key="correlate_btn"):
                try:
                    import requests
                    headers = {
                        "X-APIM-Gateway": "verified",
                        "X-API-KEY": "436abe054a074894a0517e5172f0e5b6",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "days_back": 7,
                        "update_trade_log": True,
                        "include_trade_lifecycles": False
                    }
                    response = requests.post("http://fastapi-dev:8000/api/trading/deals/correlate-activities", headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Activity correlation completed!")
                        st.json(result)
                    else:
                        st.error(f"❌ API call failed: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ Error calling API: {e}")

        # System information
        st.subheader("ℹ️ System Information")

        system_info = {
            "Streamlit Version": st.__version__,
            "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Last Refresh": st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M:%S"),
            "Session State Keys": len(st.session_state.keys())
        }

        for key, value in system_info.items():
            st.write(f"**{key}:** {value}")

        # Debug mode
        if st.checkbox("🐛 Show Debug Information"):
            st.subheader("🔍 Debug Information")
            st.write("**Session State:**")
            st.json(dict(st.session_state))

    def render_key_metrics(self, stats: TradingStatistics):
        """Render key trading metrics cards"""
        st.subheader("📊 Key Performance Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            profit_class = "profit-positive" if stats.total_profit_loss > 0 else "profit-negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Total P&L</h4>
                <h2 class="{profit_class}">{stats.total_profit_loss:.2f} SEK</h2>
                <small>{stats.total_trades} total trades</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            win_rate_class = "win-rate-high" if stats.win_rate >= 60 else "win-rate-medium" if stats.win_rate >= 40 else "win-rate-low"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Win Rate</h4>
                <h2 class="{win_rate_class}">{stats.win_rate:.1f}%</h2>
                <small>{stats.winning_trades}W / {stats.losing_trades}L</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Profit Factor</h4>
                <h2>{stats.profit_factor:.2f}</h2>
                <small>Avg Win: {stats.avg_profit:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏆 Best Trade</h4>
                <h2 class="profit-positive">+{stats.largest_win:.2f}</h2>
                <small>Largest single win</small>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📉 Worst Trade</h4>
                <h2 class="profit-negative">{stats.largest_loss:.2f}</h2>
                <small>Largest single loss</small>
            </div>
            """, unsafe_allow_html=True)

    def calculate_simple_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate simple performance metrics"""
        if df.empty:
            return {'has_data': False}

        # Basic counts
        total_trades = len(df)
        trades_with_pnl = df[df['profit_loss'].notna()]
        pending_trades = df[df['profit_loss'].isna()]

        if trades_with_pnl.empty:
            return {
                'total_trades': total_trades,
                'pending_trades': len(pending_trades),
                'completed_trades': 0,
                'has_data': False
            }

        # P&L analysis
        winning_trades = trades_with_pnl[trades_with_pnl['profit_loss'] > 0]
        losing_trades = trades_with_pnl[trades_with_pnl['profit_loss'] < 0]

        total_pnl = trades_with_pnl['profit_loss'].sum()
        win_rate = len(winning_trades) / len(trades_with_pnl) * 100 if len(trades_with_pnl) > 0 else 0

        return {
            'total_trades': total_trades,
            'completed_trades': len(trades_with_pnl),
            'pending_trades': len(pending_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'has_data': True
        }

    def fetch_alert_history(self, days: int, status_filter: str, strategy_filter: str, pair_filter: str) -> pd.DataFrame:
        """
        Fetch alert history with Claude analysis data from database.

        Args:
            days: Number of days to look back
            status_filter: 'All', 'Approved', or 'Rejected'
            strategy_filter: Strategy name or 'All'
            pair_filter: Currency pair or 'All'

        Returns:
            DataFrame with alert history data
        """
        conn = self.get_database_connection()
        if not conn:
            return pd.DataFrame()

        try:
            # Build base query
            query = """
            SELECT
                id,
                alert_timestamp,
                epic,
                pair,
                signal_type,
                strategy,
                price,
                market_session,
                claude_score,
                claude_decision,
                claude_approved,
                claude_reason,
                claude_mode,
                claude_raw_response,
                status,
                alert_level
            FROM alert_history
            WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
            """

            params = [days]

            # Add status filter
            if status_filter == "Approved":
                query += " AND (claude_approved = TRUE OR claude_decision = 'APPROVE')"
            elif status_filter == "Rejected":
                query += " AND (claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED')"

            # Add strategy filter
            if strategy_filter != "All":
                query += " AND strategy = %s"
                params.append(strategy_filter)

            # Add pair filter
            if pair_filter != "All":
                query += " AND (pair = %s OR epic LIKE %s)"
                params.append(pair_filter)
                params.append(f"%{pair_filter}%")

            query += " ORDER BY alert_timestamp DESC LIMIT 500"

            df = pd.read_sql_query(query, conn, params=params)
            return df

        except Exception as e:
            st.error(f"Error fetching alert history: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_vision_chart_path(self, epic: str, timestamp, alert_id: int = None) -> str:
        """
        Find the chart image path for an alert.

        Args:
            epic: Trading instrument epic
            timestamp: Alert timestamp
            alert_id: Optional alert ID

        Returns:
            Path to chart image or None if not found
        """
        import os
        import glob

        # Vision artifacts directory (check both local and Docker paths)
        vision_dirs = [
            "claude_analysis_enhanced/vision_analysis",
            "/app/claude_analysis_enhanced/vision_analysis",
            "../worker/app/claude_analysis_enhanced/vision_analysis"
        ]

        # Clean epic for filename matching
        epic_clean = epic.replace('.', '_') if epic else ""

        # Format timestamp for matching
        if isinstance(timestamp, str):
            try:
                ts = pd.to_datetime(timestamp)
                timestamp_str = ts.strftime('%Y%m%d_%H%M')
            except:
                timestamp_str = ""
        elif hasattr(timestamp, 'strftime'):
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M')
        else:
            timestamp_str = ""

        for vision_dir in vision_dirs:
            if not os.path.exists(vision_dir):
                continue

            # Try to find matching chart file
            patterns = [
                f"{vision_dir}/{alert_id}_{epic_clean}*_chart.png" if alert_id else None,
                f"{vision_dir}/{epic_clean}_{timestamp_str}*_chart.png",
                f"{vision_dir}/*{epic_clean}*_chart.png"
            ]

            for pattern in patterns:
                if pattern:
                    matches = glob.glob(pattern)
                    if matches:
                        # Return most recent match
                        return sorted(matches)[-1]

        return None

    # =========================================================================
    # SMC REJECTIONS TAB (v2.2.0)
    # =========================================================================

    def fetch_smc_rejections(self, days: int, stage_filter: str, pair_filter: str, session_filter: str) -> pd.DataFrame:
        """
        Fetch SMC Simple strategy rejection data from database.

        Args:
            days: Number of days to look back
            stage_filter: Rejection stage or 'All'
            pair_filter: Currency pair or 'All'
            session_filter: Market session or 'All'

        Returns:
            DataFrame with rejection data
        """
        conn = self.get_database_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                id,
                scan_timestamp,
                epic,
                pair,
                rejection_stage,
                rejection_reason,
                attempted_direction,
                current_price,
                market_hour,
                market_session,
                ema_4h_value,
                ema_distance_pips,
                price_position_vs_ema,
                atr_15m,
                atr_percentile,
                volume_ratio,
                swing_high_level,
                swing_low_level,
                pullback_depth,
                fib_zone,
                swing_range_pips,
                potential_entry,
                potential_stop_loss,
                potential_take_profit,
                potential_risk_pips,
                potential_reward_pips,
                potential_rr_ratio,
                confidence_score,
                strategy_version
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            """

            params = [days]

            # Add stage filter
            if stage_filter != "All":
                query += " AND rejection_stage = %s"
                params.append(stage_filter)

            # Add pair filter
            if pair_filter != "All":
                query += " AND (pair = %s OR epic LIKE %s)"
                params.append(pair_filter)
                params.append(f"%{pair_filter}%")

            # Add session filter
            if session_filter != "All":
                query += " AND market_session = %s"
                params.append(session_filter)

            query += " ORDER BY scan_timestamp DESC LIMIT 1000"

            df = pd.read_sql_query(query, conn, params=params)
            return df

        except Exception as e:
            # Table might not exist yet
            if "does not exist" in str(e):
                st.info("SMC Rejections table not yet created. Run the database migration first.")
            else:
                st.error(f"Error fetching SMC rejections: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def fetch_smc_rejection_stats(self, days: int) -> dict:
        """
        Fetch aggregated SMC rejection statistics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with aggregated statistics
        """
        conn = self.get_database_connection()
        if not conn:
            return {}

        try:
            # Total and by-stage counts
            query = """
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT epic) as unique_pairs,
                rejection_stage,
                COUNT(*) as stage_count
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY rejection_stage
            ORDER BY stage_count DESC
            """

            df = pd.read_sql_query(query, conn, params=[days])

            if df.empty:
                return {'total': 0, 'unique_pairs': 0, 'by_stage': {}}

            stats = {
                'total': df['total'].iloc[0] if not df.empty else 0,
                'unique_pairs': df['unique_pairs'].iloc[0] if not df.empty else 0,
                'by_stage': dict(zip(df['rejection_stage'], df['stage_count']))
            }

            # Near-miss count (high confidence rejects)
            near_miss_query = """
            SELECT COUNT(*) as near_misses
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            AND rejection_stage = 'CONFIDENCE'
            AND confidence_score >= 0.45
            """
            near_miss_df = pd.read_sql_query(near_miss_query, conn, params=[days])
            stats['near_misses'] = near_miss_df['near_misses'].iloc[0] if not near_miss_df.empty else 0

            # Most rejected pair
            pair_query = """
            SELECT pair, COUNT(*) as count
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY pair
            ORDER BY count DESC
            LIMIT 1
            """
            pair_df = pd.read_sql_query(pair_query, conn, params=[days])
            stats['most_rejected_pair'] = pair_df['pair'].iloc[0] if not pair_df.empty else 'N/A'

            return stats

        except Exception as e:
            if "does not exist" not in str(e):
                st.error(f"Error fetching SMC rejection stats: {e}")
            return {'total': 0, 'unique_pairs': 0, 'by_stage': {}, 'near_misses': 0, 'most_rejected_pair': 'N/A'}
        finally:
            conn.close()

    def render_smc_rejections_tab(self):
        """Render SMC Simple Rejection Analysis tab"""
        # Header with refresh button
        header_col1, header_col2 = st.columns([6, 1])
        with header_col1:
            st.header("🚫 SMC Simple Rejection Analysis")
        with header_col2:
            if st.button("🔄 Refresh", key="smc_rejections_refresh", help="Refresh rejection data"):
                st.rerun()

        st.markdown("Analyze why SMC Simple strategy signals were rejected to improve strategy parameters")

        # Check if table exists by trying to fetch data
        conn = self.get_database_connection()
        stages = ["All"]
        pairs = ["All"]
        sessions = ["All"]

        if conn:
            try:
                # Check table exists
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'smc_simple_rejections'
                    )
                """)
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    st.warning("⚠️ SMC Rejections table not yet created. Run the database migration:")
                    st.code("""
docker exec -it postgres psql -U postgres -d trading -f /path/to/create_smc_simple_rejections_table.sql
                    """)
                    conn.close()
                    return

                # Get unique stages
                stage_df = pd.read_sql_query(
                    "SELECT DISTINCT rejection_stage FROM smc_simple_rejections ORDER BY rejection_stage",
                    conn
                )
                stages.extend(stage_df['rejection_stage'].tolist())

                # Get unique pairs
                pair_df = pd.read_sql_query(
                    "SELECT DISTINCT pair FROM smc_simple_rejections WHERE pair IS NOT NULL ORDER BY pair",
                    conn
                )
                pairs.extend(pair_df['pair'].tolist())

                # Get unique sessions
                session_df = pd.read_sql_query(
                    "SELECT DISTINCT market_session FROM smc_simple_rejections WHERE market_session IS NOT NULL ORDER BY market_session",
                    conn
                )
                sessions.extend(session_df['market_session'].tolist())

            except Exception as e:
                if "does not exist" in str(e):
                    st.warning("⚠️ SMC Rejections table not yet created. Run the database migration first.")
                    return
                st.warning(f"Could not load filter options: {e}")
            finally:
                conn.close()

        # Filters row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            days_filter = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2, key="smc_rej_days")
        with col2:
            stage_filter = st.selectbox("Rejection Stage", stages, key="smc_rej_stage")
        with col3:
            pair_filter = st.selectbox("Pair", pairs, key="smc_rej_pair")
        with col4:
            session_filter = st.selectbox("Session", sessions, key="smc_rej_session")

        # Fetch statistics
        stats = self.fetch_smc_rejection_stats(days_filter)

        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Rejections", f"{stats.get('total', 0):,}")
        with col2:
            st.metric("Unique Pairs", stats.get('unique_pairs', 0))
        with col3:
            top_stage = max(stats.get('by_stage', {'N/A': 0}).items(), key=lambda x: x[1])[0] if stats.get('by_stage') else 'N/A'
            st.metric("Top Rejection Stage", top_stage)
        with col4:
            st.metric("Near-Misses", stats.get('near_misses', 0), help="Signals that reached confidence stage but were rejected")
        with col5:
            st.metric("Most Rejected Pair", stats.get('most_rejected_pair', 'N/A'))

        st.markdown("---")

        # Sub-tabs for different analysis views
        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "📊 Stage Breakdown", "⏰ Time Analysis", "💹 Market Context", "🎯 Near-Misses", "⚡ Scanner Efficiency"
        ])

        # Fetch data
        df = self.fetch_smc_rejections(days_filter, stage_filter, pair_filter, session_filter)

        if df.empty:
            st.info("No rejections found for the selected filters.")
            return

        with sub_tab1:
            self._render_stage_breakdown(df, stats)

        with sub_tab2:
            self._render_time_analysis(df)

        with sub_tab3:
            self._render_market_context(df)

        with sub_tab4:
            self._render_near_misses(df, days_filter)

        with sub_tab5:
            self._render_scanner_efficiency(days_filter)

    def _render_stage_breakdown(self, df: pd.DataFrame, stats: dict):
        """Render stage breakdown sub-tab"""
        st.subheader("Rejections by Stage")

        if df.empty:
            st.info("No rejection data available.")
            return

        # Pie chart for stage distribution
        stage_counts = df['rejection_stage'].value_counts().reset_index()
        stage_counts.columns = ['Stage', 'Count']

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_pie = px.pie(
                stage_counts,
                values='Count',
                names='Stage',
                title="Rejection Distribution by Stage",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart
            fig_bar = px.bar(
                stage_counts,
                x='Stage',
                y='Count',
                title="Rejection Counts by Stage",
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed reason breakdown
        st.subheader("Top Rejection Reasons")
        reason_counts = df.groupby(['rejection_stage', 'rejection_reason']).size().reset_index(name='Count')
        reason_counts = reason_counts.sort_values('Count', ascending=False).head(20)

        st.dataframe(
            reason_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rejection_stage": st.column_config.TextColumn("Stage"),
                "rejection_reason": st.column_config.TextColumn("Reason"),
                "Count": st.column_config.NumberColumn("Count", format="%d")
            }
        )

    def _render_time_analysis(self, df: pd.DataFrame):
        """Render time analysis sub-tab"""
        st.subheader("Rejection Patterns Over Time")

        if df.empty or 'market_hour' not in df.columns:
            st.info("No time data available.")
            return

        # Heatmap: Hour vs Stage
        if 'market_hour' in df.columns and df['market_hour'].notna().any():
            st.markdown("#### Rejections by Hour and Stage")
            pivot = df.groupby(['market_hour', 'rejection_stage']).size().unstack(fill_value=0)

            fig_heat = px.imshow(
                pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                labels=dict(x="Rejection Stage", y="Hour (UTC)", color="Count"),
                title="Rejection Heatmap: Hour vs Stage",
                color_continuous_scale='YlOrRd'
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Rejections by hour
        col1, col2 = st.columns(2)

        with col1:
            if 'market_hour' in df.columns:
                hour_counts = df['market_hour'].value_counts().sort_index().reset_index()
                hour_counts.columns = ['Hour', 'Count']

                fig_hour = px.bar(
                    hour_counts,
                    x='Hour',
                    y='Count',
                    title="Rejections by Hour (UTC)",
                    color='Count',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_hour, use_container_width=True)

        with col2:
            # By session
            if 'market_session' in df.columns and df['market_session'].notna().any():
                session_counts = df['market_session'].value_counts().reset_index()
                session_counts.columns = ['Session', 'Count']

                fig_session = px.bar(
                    session_counts,
                    x='Session',
                    y='Count',
                    title="Rejections by Market Session",
                    color='Session',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_session, use_container_width=True)

        # Rejections over time (daily trend)
        st.markdown("#### Daily Rejection Trend")
        if 'scan_timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['scan_timestamp']).dt.date
            daily_counts = df.groupby(['date', 'rejection_stage']).size().reset_index(name='Count')

            fig_trend = px.line(
                daily_counts,
                x='date',
                y='Count',
                color='rejection_stage',
                title="Daily Rejections by Stage",
                markers=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    def _render_market_context(self, df: pd.DataFrame):
        """Render market context sub-tab"""
        st.subheader("Market Context Analysis")

        if df.empty:
            st.info("No market context data available.")
            return

        col1, col2 = st.columns(2)

        with col1:
            # ATR distribution by stage
            if 'atr_15m' in df.columns and df['atr_15m'].notna().any():
                st.markdown("#### ATR at Rejection Points")
                fig_atr = px.box(
                    df[df['atr_15m'].notna()],
                    x='rejection_stage',
                    y='atr_15m',
                    title="ATR Distribution by Rejection Stage",
                    color='rejection_stage'
                )
                st.plotly_chart(fig_atr, use_container_width=True)

        with col2:
            # EMA distance distribution
            if 'ema_distance_pips' in df.columns and df['ema_distance_pips'].notna().any():
                st.markdown("#### EMA Distance at Rejection Points")
                fig_ema = px.histogram(
                    df[df['ema_distance_pips'].notna()],
                    x='ema_distance_pips',
                    color='rejection_stage',
                    title="EMA Distance Distribution (pips)",
                    nbins=30
                )
                st.plotly_chart(fig_ema, use_container_width=True)

        # Volume ratio analysis
        if 'volume_ratio' in df.columns and df['volume_ratio'].notna().any():
            st.markdown("#### Volume Ratio at Rejection Points")
            fig_vol = px.scatter(
                df[df['volume_ratio'].notna()],
                x='ema_distance_pips' if 'ema_distance_pips' in df.columns else 'market_hour',
                y='volume_ratio',
                color='rejection_stage',
                title="Volume Ratio vs EMA Distance",
                hover_data=['pair', 'rejection_reason']
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        # Pullback depth analysis
        if 'pullback_depth' in df.columns and df['pullback_depth'].notna().any():
            st.markdown("#### Pullback Depth Distribution")
            pullback_df = df[df['pullback_depth'].notna()].copy()
            pullback_df['pullback_pct'] = pullback_df['pullback_depth'] * 100

            fig_pb = px.histogram(
                pullback_df,
                x='pullback_pct',
                color='rejection_stage',
                title="Pullback Depth Distribution (%)",
                nbins=40,
                labels={'pullback_pct': 'Pullback Depth (%)'}
            )
            st.plotly_chart(fig_pb, use_container_width=True)

    def _render_near_misses(self, df: pd.DataFrame, days: int):
        """Render near-misses sub-tab"""
        st.subheader("Near-Miss Signals")
        st.caption("Signals that almost passed - reached the confidence stage but were rejected")

        # Filter to confidence stage rejections
        near_miss_df = df[df['rejection_stage'] == 'CONFIDENCE'].copy()

        if near_miss_df.empty:
            st.info("No near-miss signals (confidence-stage rejections) found for the selected period.")
            return

        # Sort by confidence score descending
        if 'confidence_score' in near_miss_df.columns:
            near_miss_df = near_miss_df.sort_values('confidence_score', ascending=False)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Near-Misses", len(near_miss_df))
        with col2:
            avg_conf = near_miss_df['confidence_score'].mean() * 100 if 'confidence_score' in near_miss_df.columns and near_miss_df['confidence_score'].notna().any() else 0
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        with col3:
            avg_rr = near_miss_df['potential_rr_ratio'].mean() if 'potential_rr_ratio' in near_miss_df.columns and near_miss_df['potential_rr_ratio'].notna().any() else 0
            st.metric("Avg R:R", f"{avg_rr:.2f}")
        with col4:
            bull_count = len(near_miss_df[near_miss_df['attempted_direction'] == 'BULL']) if 'attempted_direction' in near_miss_df.columns else 0
            bear_count = len(near_miss_df[near_miss_df['attempted_direction'] == 'BEAR']) if 'attempted_direction' in near_miss_df.columns else 0
            st.metric("Direction Split", f"{bull_count} Bull / {bear_count} Bear")

        st.markdown("---")

        # Display near-misses with expandable details
        for idx, row in near_miss_df.iterrows():
            timestamp = row.get('scan_timestamp', '')
            if isinstance(timestamp, pd.Timestamp):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
            else:
                timestamp_str = str(timestamp)[:16] if timestamp else 'N/A'

            pair = row.get('pair', 'N/A')
            direction = row.get('attempted_direction', 'N/A')
            direction_icon = "📈" if direction == 'BULL' else "📉" if direction == 'BEAR' else "⚪"
            confidence = row.get('confidence_score', 0)
            conf_str = f"{confidence*100:.0f}%" if confidence else 'N/A'
            rr = row.get('potential_rr_ratio', 0)
            rr_str = f"{rr:.2f}" if rr else 'N/A'
            session = row.get('market_session', 'N/A')
            reason = row.get('rejection_reason', 'N/A')

            expander_title = f"{direction_icon} {timestamp_str} | {pair} | {direction} | Conf: {conf_str} | R:R: {rr_str} | {session}"

            with st.expander(expander_title, expanded=False):
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.markdown("**Signal Details:**")
                    st.write(f"- **Pair:** {pair}")
                    st.write(f"- **Direction:** {direction}")
                    st.write(f"- **Confidence:** {conf_str}")
                    st.write(f"- **Session:** {session}")
                    entry = row.get('potential_entry', None)
                    if entry:
                        st.write(f"- **Entry:** {entry:.5f}")
                    sl = row.get('potential_stop_loss', None)
                    if sl:
                        st.write(f"- **Stop Loss:** {sl:.5f}")
                    tp = row.get('potential_take_profit', None)
                    if tp:
                        st.write(f"- **Take Profit:** {tp:.5f}")

                with detail_col2:
                    st.markdown("**Risk/Reward:**")
                    risk_pips = row.get('potential_risk_pips', None)
                    reward_pips = row.get('potential_reward_pips', None)
                    if risk_pips:
                        st.write(f"- **Risk:** {risk_pips:.1f} pips")
                    if reward_pips:
                        st.write(f"- **Reward:** {reward_pips:.1f} pips")
                    st.write(f"- **R:R Ratio:** {rr_str}")

                    st.markdown("**Market Context:**")
                    ema_dist = row.get('ema_distance_pips', None)
                    if ema_dist:
                        st.write(f"- **EMA Distance:** {ema_dist:.1f} pips")
                    pullback = row.get('pullback_depth', None)
                    if pullback:
                        st.write(f"- **Pullback Depth:** {pullback*100:.1f}%")
                    fib_zone = row.get('fib_zone', None)
                    if fib_zone:
                        st.write(f"- **Fib Zone:** {fib_zone}")

                # Rejection reason
                st.markdown("**Rejection Reason:**")
                st.warning(reason)

        # Export button
        st.markdown("---")
        csv = near_miss_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Near-Misses to CSV",
            data=csv,
            file_name=f"smc_near_misses_{days}days.csv",
            mime="text/csv"
        )

    def _render_scanner_efficiency(self, days: int):
        """Render scanner efficiency analysis sub-tab"""
        st.subheader("Scanner Efficiency Analysis")
        st.caption("Analyze scanner frequency vs candle timeframe to optimize scan intervals")

        conn = self.get_database_connection()
        if not conn:
            st.error("Database connection failed")
            return

        try:
            # Query for efficiency metrics
            efficiency_query = """
            SELECT
                scan_timestamp,
                created_at,
                EXTRACT(EPOCH FROM (created_at - scan_timestamp)) / 60 as latency_minutes,
                epic,
                pair,
                rejection_stage
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY scan_timestamp DESC
            """
            df = pd.read_sql_query(efficiency_query, conn, params=[days])

            if df.empty:
                st.info("No data available for scanner efficiency analysis.")
                return

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_latency = df['latency_minutes'].mean()
                st.metric("Avg Scan Latency", f"{avg_latency:.1f} min", help="Time between candle close and scan")

            with col2:
                # Scans per unique candle
                scans_per_candle = df.groupby(['scan_timestamp', 'epic']).size()
                avg_scans = scans_per_candle.mean()
                st.metric("Avg Scans/Candle", f"{avg_scans:.1f}", help="How many times each candle is scanned")

            with col3:
                unique_candles = df.groupby(['scan_timestamp', 'epic']).ngroups
                st.metric("Unique Candles", f"{unique_candles:,}")

            with col4:
                total_scans = len(df)
                redundant_pct = ((total_scans - unique_candles) / total_scans * 100) if total_scans > 0 else 0
                st.metric("Redundant Scans", f"{redundant_pct:.1f}%", help="% of scans that re-analyzed same candle")

            st.markdown("---")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                # Latency distribution
                st.markdown("#### Scan Latency Distribution")
                fig_latency = px.histogram(
                    df,
                    x='latency_minutes',
                    nbins=30,
                    title="Time from Candle Close to Scan (minutes)",
                    labels={'latency_minutes': 'Latency (minutes)'}
                )
                fig_latency.add_vline(x=avg_latency, line_dash="dash", line_color="red",
                                      annotation_text=f"Avg: {avg_latency:.1f} min")
                st.plotly_chart(fig_latency, use_container_width=True)

            with col2:
                # Scans per candle distribution
                st.markdown("#### Scans Per Candle Distribution")
                scans_df = scans_per_candle.reset_index(name='scan_count')
                fig_scans = px.histogram(
                    scans_df,
                    x='scan_count',
                    nbins=20,
                    title="How Many Times Each Candle Was Scanned",
                    labels={'scan_count': 'Scans per Candle'}
                )
                st.plotly_chart(fig_scans, use_container_width=True)

            # Detailed table: Most re-scanned candles
            st.markdown("#### Most Re-Scanned Candles")
            st.caption("Candles that were analyzed multiple times with the same rejection result")

            rescan_df = scans_per_candle.reset_index(name='scan_count')
            rescan_df = rescan_df.sort_values('scan_count', ascending=False).head(20)
            rescan_df.columns = ['Candle Time', 'Epic', 'Scan Count']

            st.dataframe(
                rescan_df,
                use_container_width=True,
                hide_index=True
            )

            # Recommendation
            st.markdown("---")
            st.markdown("#### Optimization Recommendations")

            if avg_scans > 5:
                st.warning(f"""
                **High redundancy detected**: Each candle is scanned ~{avg_scans:.0f} times on average.

                **Recommendation**: Consider:
                - Increasing scan interval from 2 min to 5-10 min
                - Adding candle-close detection to only scan on new candles
                - Caching rejection decisions for the same candle timestamp
                """)
            elif avg_scans > 2:
                st.info(f"""
                **Moderate redundancy**: Each candle is scanned ~{avg_scans:.0f} times on average.

                This is acceptable for a 15-min strategy with 2-min scanning, but there's room for optimization.
                """)
            else:
                st.success(f"""
                **Good efficiency**: Each candle is scanned ~{avg_scans:.0f} times on average.

                Scanner frequency is well-matched to the strategy timeframe.
                """)

        except Exception as e:
            if "does not exist" in str(e):
                st.info("Scanner efficiency data not yet available.")
            else:
                st.error(f"Error fetching scanner efficiency data: {e}")
        finally:
            conn.close()

    def render_alert_history_tab(self):
        """Render Alert History tab with Claude Vision analysis status"""
        # Header with refresh button
        header_col1, header_col2 = st.columns([6, 1])
        with header_col1:
            st.header("📋 Alert History")
        with header_col2:
            if st.button("🔄 Refresh", key="alert_history_refresh", help="Refresh alert data"):
                st.rerun()

        st.markdown("View all trading signals with Claude AI analysis status")

        # Get unique strategies and pairs for filters
        conn = self.get_database_connection()
        strategies = ["All"]
        pairs = ["All"]

        if conn:
            try:
                # Get unique strategies
                strat_df = pd.read_sql_query(
                    "SELECT DISTINCT strategy FROM alert_history WHERE strategy IS NOT NULL ORDER BY strategy",
                    conn
                )
                strategies.extend(strat_df['strategy'].tolist())

                # Get unique pairs
                pair_df = pd.read_sql_query(
                    "SELECT DISTINCT pair FROM alert_history WHERE pair IS NOT NULL ORDER BY pair",
                    conn
                )
                pairs.extend(pair_df['pair'].tolist())
            except Exception as e:
                st.warning(f"Could not load filter options: {e}")
            finally:
                conn.close()

        # Filters row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            days_filter = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=0, key="alert_history_days")
        with col2:
            status_filter = st.selectbox("Claude Status", ["All", "Approved", "Rejected"], key="alert_history_status")
        with col3:
            strategy_filter = st.selectbox("Strategy", strategies, key="alert_history_strategy")
        with col4:
            pair_filter = st.selectbox("Pair", pairs, key="alert_history_pair")

        # Fetch data
        df = self.fetch_alert_history(days_filter, status_filter, strategy_filter, pair_filter)

        if df.empty:
            st.info("No alerts found for the selected filters.")
            return

        # Summary metrics
        st.markdown("---")
        total_alerts = len(df)
        approved_count = len(df[df['claude_approved'] == True]) if 'claude_approved' in df.columns else 0
        rejected_count = len(df[(df['claude_approved'] == False) | (df['alert_level'] == 'REJECTED')]) if 'claude_approved' in df.columns else 0
        approval_rate = (approved_count / total_alerts * 100) if total_alerts > 0 else 0
        avg_score = df['claude_score'].mean() if 'claude_score' in df.columns and not df['claude_score'].isna().all() else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("Approved", approved_count, delta=None)
        with col3:
            st.metric("Rejected", rejected_count, delta=None)
        with col4:
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        with col5:
            st.metric("Avg Score", f"{avg_score:.1f}/10" if avg_score > 0 else "N/A")

        st.markdown("---")

        # Display alerts with expandable details
        for idx, row in df.iterrows():
            # Determine status icon and color
            is_approved = row.get('claude_approved', None)
            claude_decision = row.get('claude_decision', '')
            alert_level = row.get('alert_level', '')

            if is_approved == True or claude_decision == 'APPROVE':
                status_icon = "✅"
                status_text = "APPROVED"
                status_color = "green"
            elif is_approved == False or claude_decision == 'REJECT' or alert_level == 'REJECTED':
                status_icon = "❌"
                status_text = "REJECTED"
                status_color = "red"
            else:
                status_icon = "⚪"
                status_text = "PENDING"
                status_color = "gray"

            # Format timestamp
            timestamp = row.get('alert_timestamp', '')
            if isinstance(timestamp, pd.Timestamp):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
            else:
                timestamp_str = str(timestamp)[:16] if timestamp else 'N/A'

            # Get values with defaults
            pair = row.get('pair', row.get('epic', 'N/A'))
            if pair == 'N/A' or pd.isna(pair):
                epic = row.get('epic', '')
                if epic:
                    # Extract pair from epic like CS.D.EURUSD.CEEM.IP
                    parts = epic.split('.')
                    if len(parts) >= 3:
                        pair = parts[2][:6] if len(parts[2]) >= 6 else parts[2]

            strategy = row.get('strategy', 'N/A')
            signal_type = row.get('signal_type', 'N/A')
            price = row.get('price', 0)
            price_str = f"{price:.5f}" if price and not pd.isna(price) else 'N/A'
            session = row.get('market_session', 'N/A')
            if pd.isna(session):
                session = 'N/A'
            score = row.get('claude_score', 0)
            score_str = f"{int(score)}/10" if score and not pd.isna(score) else 'N/A'

            # Create expander for each row
            expander_title = f"{status_icon} {timestamp_str} | {pair} | {strategy} | {signal_type} | {price_str} | {session} | Score: {score_str}"

            with st.expander(expander_title, expanded=False):
                # Two columns: details and chart
                detail_col, chart_col = st.columns([1, 1])

                with detail_col:
                    st.markdown("**Signal Details:**")
                    st.write(f"- **Status:** {status_icon} {status_text}")
                    st.write(f"- **Pair:** {pair}")
                    st.write(f"- **Strategy:** {strategy}")
                    st.write(f"- **Signal:** {signal_type}")
                    st.write(f"- **Price:** {price_str}")
                    st.write(f"- **Session:** {session}")
                    st.write(f"- **Claude Score:** {score_str}")
                    st.write(f"- **Claude Mode:** {row.get('claude_mode', 'N/A')}")

                    # Claude reason
                    reason = row.get('claude_reason', '')
                    if reason and not pd.isna(reason):
                        st.markdown("**Claude Reason:**")
                        st.info(reason)

                with chart_col:
                    # Try to load chart image
                    alert_id = row.get('id', None)
                    chart_path = self.get_vision_chart_path(
                        row.get('epic', ''),
                        row.get('alert_timestamp'),
                        alert_id
                    )

                    if chart_path and os.path.exists(chart_path):
                        st.markdown("**Chart Image:**")
                        st.image(chart_path, caption="Vision Analysis Chart", use_container_width=True)
                    else:
                        st.markdown("**Chart Image:**")
                        st.info("No chart available (vision analysis not used or chart not saved)")

                # Full raw response in a separate section
                raw_response = row.get('claude_raw_response', '')
                if raw_response and not pd.isna(raw_response):
                    st.markdown("---")
                    st.markdown("**Full Claude Raw Response:**")
                    st.code(raw_response, language=None)

    def run(self):
        """Main application entry point"""
        # App header
        st.title("📊 Trading Analytics Hub")
        st.markdown("*Unified dashboard for comprehensive trading analysis*")

        # Tab navigation
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Overview", "🎯 Strategy Analysis", "💰 Trade Performance",
            "🧠 Market Intelligence", "🔍 Trade Analysis", "🔧 Settings & Debug",
            "📋 Alert History", "🚫 SMC Rejections"
        ])

        with tab1:
            self.render_overview_tab()

        with tab2:
            self.render_strategy_analysis_tab()

        with tab3:
            self.render_trade_performance_tab()

        with tab4:
            self.render_market_intelligence_tab()

        with tab5:
            self.render_trade_analysis_tab()

        with tab6:
            self.render_settings_debug_tab()

        with tab7:
            self.render_alert_history_tab()

        with tab8:
            self.render_smc_rejections_tab()

        # Footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; color: #666; font-size: 12px;'>
            Trading Analytics Hub v1.0 |
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")} |
            Status: 🟢 Online
            </div>
            """,
            unsafe_allow_html=True
        )

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = UnifiedTradingDashboard()
    dashboard.run()