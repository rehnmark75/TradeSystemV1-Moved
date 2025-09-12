"""
Enhanced Streamlit Trading Statistics Dashboard
Integrates comprehensive P/L and performance data from trade_log table
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

# Configure page
st.set_page_config(
    page_title="Forex Trading Statistics",
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

class TradingDashboard:
    """Enhanced Trading Statistics Dashboard"""
    
    def __init__(self):
        self.setup_database_connection()
        self.initialize_session_state()
    
    def setup_database_connection(self):
        """Setup database connection"""
        try:
            # Try Streamlit secrets first, then environment
            if hasattr(st, 'secrets') and 'database' in st.secrets:
                self.conn_string = st.secrets.database.connection_string
            else:
                import os
                self.conn_string = os.getenv("DATABASE_URL", "postgresql://localhost:5432/forex")
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
    
    def get_database_connection(self):
        """Get database connection"""
        # If conn_string is not set in class, try to get it from secrets directly
        if not self.conn_string:
            try:
                self.conn_string = st.secrets.database.trading_connection_string
            except (AttributeError, KeyError):
                try:
                    self.conn_string = st.secrets.database.config_connection_string
                except (AttributeError, KeyError):
                    import os
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
            # Try one more time to get the connection string from secrets
            try:
                self.conn_string = st.secrets.database.trading_connection_string
                conn = self.get_database_connection()
                if not conn:
                    return None
            except (AttributeError, KeyError):
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
                    avg_win_duration=0.0,  # Could be calculated with more complex query
                    avg_loss_duration=0.0,  # Could be calculated with more complex query
                    total_volume=0.0,  # Would need volume data in trade_log
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
    
    def render_main_dashboard(self):
        """Render the main trading statistics dashboard"""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("📊 Forex Trading Statistics")
            st.caption("Comprehensive P&L and performance analysis from trade_log data")
        
        with col2:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col3:
            timeframe = st.selectbox(
                "📅 Timeframe",
                options=['1_day', '7_days', '30_days', '90_days', 'all_time'],
                format_func=lambda x: {
                    '1_day': '24 Hours',
                    '7_days': '7 Days', 
                    '30_days': '30 Days',
                    '90_days': '90 Days',
                    'all_time': 'All Time'
                }[x],
                key='selected_timeframe'
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Map timeframe to days
        timeframe_map = {
            '1_day': 1, '7_days': 7, '30_days': 30, 
            '90_days': 90, 'all_time': 3650
        }
        days_back = timeframe_map[timeframe]
        
        # Fetch statistics
        stats = self.fetch_trading_statistics(days_back)
        trades_df = self.fetch_trades_dataframe(days_back)
        
        if stats:
            # Render statistics sections
            self.render_key_metrics(stats)
            self.render_performance_charts(stats, trades_df)
            self.render_trades_table(trades_df)
            self.render_pair_analysis(trades_df)
        else:
            st.error("Unable to fetch trading statistics. Please check database connection.")
    
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
        
        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Pairs", len(stats.active_pairs))
        
        with col2:
            st.metric("Best Pair", stats.best_pair)
        
        with col3:
            st.metric("Pending Trades", stats.pending_trades)
        
        with col4:
            avg_trade = stats.total_profit_loss / stats.total_trades if stats.total_trades > 0 else 0
            st.metric("Avg P&L per Trade", f"{avg_trade:.2f}")
    
    def render_performance_charts(self, stats: TradingStatistics, trades_df: pd.DataFrame):
        """Render performance visualization charts"""
        st.subheader("📈 Performance Analysis")
        
        if trades_df.empty:
            st.info("No trade data available for the selected timeframe.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L Distribution Chart
            if 'profit_loss' in trades_df.columns:
                fig_pnl = px.histogram(
                    trades_df[trades_df['profit_loss'].notna()], 
                    x='profit_loss',
                    title="P&L Distribution",
                    labels={'profit_loss': 'Profit/Loss (SEK)', 'count': 'Number of Trades'},
                    color_discrete_sequence=['#17a2b8']
                )
                fig_pnl.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
                st.plotly_chart(fig_pnl, use_container_width=True)
        
        with col2:
            # Win/Loss Pie Chart
            win_loss_data = pd.DataFrame({
                'Result': ['Wins', 'Losses', 'Pending'],
                'Count': [stats.winning_trades, stats.losing_trades, stats.pending_trades],
                'Color': ['#28a745', '#dc3545', '#6c757d']
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
        
        # Cumulative P&L Chart
        if 'timestamp' in trades_df.columns and 'profit_loss' in trades_df.columns:
            # Calculate cumulative P&L
            trades_sorted = trades_df[trades_df['profit_loss'].notna()].sort_values('timestamp')
            trades_sorted['cumulative_pnl'] = trades_sorted['profit_loss'].cumsum()
            
            fig_cumulative = px.line(
                trades_sorted, 
                x='timestamp', 
                y='cumulative_pnl',
                title="Cumulative P&L Over Time",
                labels={'cumulative_pnl': 'Cumulative P&L (SEK)', 'timestamp': 'Date'},
                line_shape='linear'
            )
            fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
            st.plotly_chart(fig_cumulative, use_container_width=True)
    
    def render_trades_table(self, trades_df: pd.DataFrame):
        """Render detailed trades table"""
        st.subheader("📋 Recent Trades")
        
        if trades_df.empty:
            st.info("No trades found for the selected timeframe.")
            return
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            result_filter = st.multiselect(
                "Filter by Result", 
                options=['WIN', 'LOSS', 'PENDING'],
                default=['WIN', 'LOSS', 'PENDING']
            )
        
        with col2:
            direction_filter = st.multiselect(
                "Filter by Direction",
                options=trades_df['direction'].unique() if 'direction' in trades_df.columns else [],
                default=trades_df['direction'].unique() if 'direction' in trades_df.columns else []
            )
        
        with col3:
            symbol_filter = st.multiselect(
                "Filter by Symbol",
                options=trades_df['symbol'].unique() if 'symbol' in trades_df.columns else [],
                default=trades_df['symbol'].unique() if 'symbol' in trades_df.columns else []
            )
        
        # Apply filters
        filtered_df = trades_df.copy()
        if result_filter and 'trade_result' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['trade_result'].isin(result_filter)]
        if direction_filter and 'direction' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['direction'].isin(direction_filter)]
        if symbol_filter and 'symbol' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]
        
        # Display formatted table
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
        else:
            st.info("No trades match the selected filters.")
    
    def render_pair_analysis(self, trades_df: pd.DataFrame):
        """Render currency pair performance analysis"""
        st.subheader("🔍 Currency Pair Analysis")
        
        if trades_df.empty or 'symbol' not in trades_df.columns:
            st.info("No pair data available for analysis.")
            return
        
        # Calculate pair statistics
        pair_stats = trades_df.groupby('symbol').agg({
            'profit_loss': ['count', 'sum', 'mean'],
            'trade_result': lambda x: (x == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
        }).round(2)
        
        pair_stats.columns = ['Total Trades', 'Total P&L', 'Avg P&L', 'Win Rate %']
        pair_stats = pair_stats.sort_values('Total P&L', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pair performance bar chart
            fig_pairs = px.bar(
                x=pair_stats.index,
                y=pair_stats['Total P&L'],
                title="P&L by Currency Pair",
                labels={'x': 'Currency Pair', 'y': 'Total P&L (SEK)'},
                color=pair_stats['Total P&L'],
                color_continuous_scale=['red', 'gray', 'green']
            )
            st.plotly_chart(fig_pairs, use_container_width=True)
        
        with col2:
            # Pair statistics table
            st.dataframe(
                pair_stats,
                use_container_width=True,
                column_config={
                    "Total Trades": st.column_config.NumberColumn("Trades", format="%d"),
                    "Total P&L": st.column_config.NumberColumn("Total P&L", format="%.2f"),
                    "Avg P&L": st.column_config.NumberColumn("Avg P&L", format="%.2f"),
                    "Win Rate %": st.column_config.NumberColumn("Win Rate", format="%.1f%%")
                }
            )
    
    def render_sidebar(self):
        """Render enhanced sidebar with additional controls"""
        st.sidebar.title("📊 Trading Dashboard")
        
        # Debug info for connection
        st.sidebar.markdown("### 🔧 Connection Debug")
        
        # Debug: Show what secrets are available
        if hasattr(st, 'secrets'):
            # Show all available sections in secrets
            try:
                secrets_dict = st.secrets.to_dict()
                st.sidebar.success(f"🟢 Secret sections: {list(secrets_dict.keys())}")
                
                # Show database section contents if it exists
                if 'database' in secrets_dict:
                    db_keys = list(secrets_dict['database'].keys())
                    st.sidebar.success(f"🟢 Database keys: {db_keys}")
                    
                    # Show the actual connection string being used
                    try:
                        conn_str = st.secrets.database.trading_connection_string
                        st.sidebar.success("🟢 trading_connection_string found!")
                        st.sidebar.code(conn_str, language='text')
                    except (AttributeError, KeyError):
                        st.sidebar.error("❌ trading_connection_string not accessible")
                else:
                    st.sidebar.warning("⚠️ No [database] section found")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error reading secrets: {e}")
        else:
            st.sidebar.error("🔴 No secrets available")
        
        # Show current connection string being used
        if hasattr(self, 'conn_string') and self.conn_string:
            st.sidebar.success(f"✅ Class connection string set")
            st.sidebar.code(self.conn_string, language='text')
        else:
            st.sidebar.error("❌ No connection string in class")
            # Quick fix button
            if st.sidebar.button("🔧 Fix Connection String"):
                try:
                    self.conn_string = st.secrets.database.trading_connection_string
                    st.sidebar.success("✅ Connection string fixed!")
                    st.rerun()
                except (AttributeError, KeyError) as e:
                    st.sidebar.error(f"❌ Could not fix: {e}")
        
        # Database troubleshooting
        st.sidebar.markdown("### 🔍 Database Troubleshooting")
        
        # Test different connection strings
        if st.sidebar.button("🧪 Test Connection"):
            self._test_database_connections()
        
        # PostgreSQL status check
        st.sidebar.markdown("### 🐘 PostgreSQL Status")
        st.sidebar.markdown("""
        **Common solutions:**
        1. **Start PostgreSQL**: `brew services start postgresql` (macOS) or `sudo systemctl start postgresql` (Linux)
        2. **Check if running**: `pg_isready -h localhost -p 5432`
        3. **Check port**: Your connection uses port 5432
        4. **Docker PostgreSQL**: If using Docker, ensure container is running
        """)
        
        # Manual connection options
        st.sidebar.markdown("### ⚙️ Manual Override")
        manual_host = st.sidebar.text_input("Host", value="postgres")
        manual_port = st.sidebar.number_input("Port", value=5432, min_value=1, max_value=65535)
        manual_db = st.sidebar.text_input("Database", value="forex")
        manual_user = st.sidebar.text_input("Username", value="postgres")
        manual_pass = st.sidebar.text_input("Password", value="postgres", type="password")
        
        if st.sidebar.button("🧪 Test Manual Settings"):
            manual_conn_str = f"postgresql://{manual_user}:{manual_pass}@{manual_host}:{manual_port}/{manual_db}"
            try:
                import psycopg2
                test_conn = psycopg2.connect(manual_conn_str)
                test_conn.close()
                st.sidebar.success("✅ Manual connection successful!")
                st.sidebar.info("Consider updating your secrets.toml with these settings")
            except Exception as e:
                st.sidebar.error(f"❌ Manual connection failed: {e}")
        
        # Quick stats
        st.sidebar.markdown("### Quick Stats")
        
        # Last refresh
        st.sidebar.info(f"🕐 Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        # Navigation
        st.sidebar.markdown("### Navigation")
        page = st.sidebar.selectbox(
            "Select View",
            ["📊 Dashboard", "📈 Advanced Analytics", "⚙️ Settings", "📋 Export Data"]
        )
        
        if page == "📋 Export Data":
            st.sidebar.markdown("### Export Options")
            if st.sidebar.button("📥 Download CSV"):
                st.sidebar.success("Export feature coming soon!")
        
        # Auto-refresh
        st.sidebar.markdown("### Auto Refresh")
        auto_refresh = st.sidebar.checkbox("Enable auto-refresh")
        if auto_refresh:
            refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)
            st.sidebar.info(f"Auto-refreshing every {refresh_interval}s")
    
    def _test_database_connections(self):
        """Test various database connection scenarios"""
        st.sidebar.markdown("**Testing connections...**")
        
        # Test secrets connection
        try:
            conn_str = st.secrets.database.trading_connection_string
            import psycopg2
            test_conn = psycopg2.connect(conn_str)
            test_conn.close()
            st.sidebar.success("✅ Secrets connection works!")
        except Exception as e:
            st.sidebar.error(f"❌ Secrets connection failed: {str(e)[:100]}...")
        
        # Test class connection
        if self.conn_string:
            try:
                import psycopg2
                test_conn = psycopg2.connect(self.conn_string)
                test_conn.close()
                st.sidebar.success("✅ Class connection works!")
            except Exception as e:
                st.sidebar.error(f"❌ Class connection failed: {str(e)[:100]}...")
        
        # Test default connection
        try:
            default_conn = "postgresql://postgres:postgres@postgres:5432/forex"
            import psycopg2
            test_conn = psycopg2.connect(default_conn)
            test_conn.close()
            st.sidebar.success("✅ Default connection works!")
        except Exception as e:
            st.sidebar.error(f"❌ Default connection failed: {str(e)[:100]}...")
    
    def run(self):
        """Main application entry point"""
        self.render_sidebar()
        self.render_main_dashboard()

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()