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
                id, symbol, entry_price, direction, timestamp, status,
                profit_loss, pnl_currency, deal_id, sl_price, tp_price,
                closed_at, alert_id
            FROM trade_log
            WHERE timestamp >= %s
            """

            params = [datetime.now() - timedelta(days=days_back)]

            if pairs_filter:
                query += " AND symbol = ANY(%s)"
                params.append(pairs_filter)

            query += " ORDER BY timestamp DESC"

            df = pd.read_sql_query(query, conn, params=params)

            # Enhance DataFrame
            if not df.empty:
                df['trade_result'] = df['profit_loss'].apply(
                    lambda x: 'WIN' if x > 0 else 'LOSS' if x < 0 else 'PENDING'
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
                index=1
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

            display_columns = ['timestamp', 'symbol', 'direction', 'profit_loss_formatted', 'trade_result']
            display_columns = [col for col in display_columns if col in recent_trades.columns]

            st.dataframe(
                recent_trades[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="MM/DD HH:mm"),
                    "symbol": "Pair",
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
                index=1,
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
        col1, col2, col3 = st.columns(3)

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

        # Apply filters
        filtered_df = trades_df.copy()
        if result_filter and 'trade_result' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['trade_result'].isin(result_filter)]
        if direction_filter and 'direction' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['direction'].isin(direction_filter)]
        if symbol_filter and 'symbol' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]

        # Display table
        if not filtered_df.empty:
            display_columns = ['timestamp', 'symbol', 'direction', 'entry_price', 'profit_loss_formatted', 'trade_result', 'status']
            display_columns = [col for col in display_columns if col in filtered_df.columns]

            st.dataframe(
                filtered_df[display_columns].head(50),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Trade Time", format="DD/MM/YY HH:mm"),
                    "symbol": "Symbol",
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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Strategy Analysis", "💰 Trade Performance", "🔧 Settings & Debug"])

        with tab1:
            self.render_overview_tab()

        with tab2:
            self.render_strategy_analysis_tab()

        with tab3:
            self.render_trade_performance_tab()

        with tab4:
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