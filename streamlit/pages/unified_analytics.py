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
        self.setup_database_connection()
        self.initialize_session_state()

    def setup_database_connection(self):
        """Setup database connection"""
        try:
            # Try Streamlit secrets first, then environment
            if hasattr(st, 'secrets') and 'database' in st.secrets:
                try:
                    self.conn_string = st.secrets.database.trading_connection_string
                except (AttributeError, KeyError):
                    try:
                        self.conn_string = st.secrets.database.config_connection_string
                    except (AttributeError, KeyError):
                        self.conn_string = st.secrets.database.connection_string
            else:
                self.conn_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
        except Exception as e:
            st.error(f"Database configuration error: {e}")
            self.conn_string = None

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

    def get_database_connection(self):
        """Get database connection"""
        if not self.conn_string:
            try:
                self.conn_string = st.secrets.database.trading_connection_string
            except (AttributeError, KeyError):
                try:
                    self.conn_string = st.secrets.database.config_connection_string
                except (AttributeError, KeyError):
                    self.conn_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")

        if not self.conn_string:
            return None
        try:
            return psycopg2.connect(self.conn_string)
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
            cursor = conn.cursor()
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'market_intelligence_history'
            """)
            existing_columns = [row[0] for row in cursor.fetchall()]
            cursor.close()

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
        """Render the Trade Analysis tab for detailed trailing stop analysis"""
        st.header("🔍 Individual Trade Analysis")
        st.markdown("*Analyze trailing stop stages and performance for specific trades*")

        # Input for trade ID
        col1, col2 = st.columns([3, 1])

        with col1:
            trade_id = st.number_input("Enter Trade ID", min_value=1, value=1273, step=1, key="trade_id_input")

        with col2:
            analyze_btn = st.button("🔍 Analyze Trade", type="primary")

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

    def run(self):
        """Main application entry point"""
        # App header
        st.title("📊 Trading Analytics Hub")
        st.markdown("*Unified dashboard for comprehensive trading analysis*")

        # Tab navigation
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🎯 Strategy Analysis", "💰 Trade Performance", "🧠 Market Intelligence", "🔍 Trade Analysis", "🔧 Settings & Debug"])

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

        # Footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; color: #666; font-size: 12px;'>
            Trading Analytics Hub v1.0 |
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")} |
            Status: {'🟢 Online' if self.conn_string else '🔴 Offline'}
            </div>
            """,
            unsafe_allow_html=True
        )

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = UnifiedTradingDashboard()
    dashboard.run()