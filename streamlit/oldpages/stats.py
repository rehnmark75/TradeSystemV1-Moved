"""
Trading Statistics Page - statistics.py
Dedicated page for comprehensive trading performance analysis from trade_log table
Place this file in: streamlit/pages/statistics.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

# Configure page
st.set_page_config(
    page_title="Trading Statistics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for statistics dashboard
st.markdown("""
<style>
    .stMetric > div > div > div > div {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .profit-card {
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .loss-card {
        background: linear-gradient(145deg, #f8d7da, #f5c6cb);
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .neutral-card {
        background: linear-gradient(145deg, #e2e3e5, #d6d8db);
        border: 1px solid #d6d8db;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .profit-text { color: #155724; }
    .loss-text { color: #721c24; }
    .neutral-text { color: #495057; }
</style>
""", unsafe_allow_html=True)

class TradingStatisticsAnalyzer:
    """Comprehensive trading statistics analyzer"""
    
    def __init__(self):
        self.conn_string = self._get_connection_string()
    
    def _get_connection_string(self) -> Optional[str]:
        """Get database connection string"""
        try:
            # Try Streamlit secrets first with multiple possible key names
            if hasattr(st, 'secrets'):
                if hasattr(st.secrets, 'trading_connection_string'):
                    return st.secrets.trading_connection_string
                elif hasattr(st.secrets, 'connection_string'):
                    return st.secrets.connection_string
                elif 'database' in st.secrets and hasattr(st.secrets.database, 'connection_string'):
                    return st.secrets.database.connection_string
                else:
                    # Fallback to environment
                    import os
                    return os.getenv("DATABASE_URL", "postgresql://localhost:5432/forex")
            else:
                import os
                return os.getenv("DATABASE_URL", "postgresql://localhost:5432/forex")
        except Exception as e:
            st.error(f"Database configuration error: {e}")
            return None
    
    def get_connection(self):
        """Get database connection"""
        if not self.conn_string:
            return None
        try:
            return psycopg2.connect(self.conn_string)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    
    def fetch_comprehensive_stats(self, start_date: datetime, end_date: datetime, 
                                pairs_filter: List[str] = None) -> Dict:
        """Fetch comprehensive trading statistics"""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor() as cursor:
                # Main statistics query
                base_query = """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                    COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses,
                    COUNT(CASE WHEN profit_loss IS NULL OR status = 'pending' THEN 1 END) as pending,
                    COALESCE(SUM(profit_loss), 0) as total_pnl,
                    COALESCE(SUM(CASE WHEN profit_loss > 0 THEN profit_loss END), 0) as total_profits,
                    COALESCE(SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) END), 0) as total_losses,
                    COALESCE(AVG(CASE WHEN profit_loss > 0 THEN profit_loss END), 0) as avg_win,
                    COALESCE(AVG(CASE WHEN profit_loss < 0 THEN profit_loss END), 0) as avg_loss,
                    COALESCE(MAX(profit_loss), 0) as best_trade,
                    COALESCE(MIN(profit_loss), 0) as worst_trade,
                    COUNT(DISTINCT symbol) as unique_pairs,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days
                FROM trade_log 
                WHERE timestamp BETWEEN %s AND %s
                """
                
                params = [start_date, end_date]
                
                if pairs_filter:
                    base_query += " AND symbol = ANY(%s)"
                    params.append(pairs_filter)
                
                cursor.execute(base_query, params)
                result = cursor.fetchone()
                
                if not result or result[0] == 0:
                    return self._empty_stats()
                
                # Calculate derived metrics
                total_trades, wins, losses, pending = result[0:4]
                total_pnl, total_profits, total_losses, avg_win, avg_loss = result[4:9]
                best_trade, worst_trade, unique_pairs, trading_days = result[9:13]
                
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                profit_factor = (total_profits / total_losses) if total_losses > 0 else float('inf')
                avg_daily_trades = total_trades / trading_days if trading_days > 0 else 0
                avg_daily_pnl = total_pnl / trading_days if trading_days > 0 else 0
                
                return {
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'pending': pending,
                    'total_pnl': float(total_pnl),
                    'total_profits': float(total_profits),
                    'total_losses': float(total_losses),
                    'win_rate': win_rate,
                    'avg_win': float(avg_win),
                    'avg_loss': float(avg_loss),
                    'best_trade': float(best_trade),
                    'worst_trade': float(worst_trade),
                    'profit_factor': profit_factor,
                    'unique_pairs': unique_pairs,
                    'trading_days': trading_days,
                    'avg_daily_trades': avg_daily_trades,
                    'avg_daily_pnl': float(avg_daily_pnl)
                }
                
        except Exception as e:
            st.error(f"Error fetching statistics: {e}")
            return self._empty_stats()
        finally:
            conn.close()
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics dictionary"""
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'pending': 0,
            'total_pnl': 0.0, 'total_profits': 0.0, 'total_losses': 0.0,
            'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'best_trade': 0.0, 'worst_trade': 0.0, 'profit_factor': 0.0,
            'unique_pairs': 0, 'trading_days': 0, 'avg_daily_trades': 0.0,
            'avg_daily_pnl': 0.0
        }
    
    def fetch_daily_performance(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch daily performance data"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                DATE(timestamp) as trade_date,
                COUNT(*) as daily_trades,
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as daily_wins,
                COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as daily_losses,
                COALESCE(SUM(profit_loss), 0) as daily_pnl,
                COALESCE(AVG(profit_loss), 0) as avg_trade_pnl
            FROM trade_log 
            WHERE timestamp BETWEEN %s AND %s
            GROUP BY DATE(timestamp)
            ORDER BY trade_date DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            
            if not df.empty:
                df['daily_win_rate'] = (df['daily_wins'] / df['daily_trades'] * 100).round(1)
                df['cumulative_pnl'] = df['daily_pnl'].cumsum()
                
            return df
            
        except Exception as e:
            st.error(f"Error fetching daily performance: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def fetch_pair_statistics(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch currency pair statistics"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                symbol,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses,
                COALESCE(SUM(profit_loss), 0) as total_pnl,
                COALESCE(AVG(profit_loss), 0) as avg_pnl,
                COALESCE(MAX(profit_loss), 0) as best_trade,
                COALESCE(MIN(profit_loss), 0) as worst_trade,
                COALESCE(STDDEV(profit_loss), 0) as pnl_volatility
            FROM trade_log 
            WHERE timestamp BETWEEN %s AND %s
            GROUP BY symbol
            ORDER BY total_pnl DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            
            if not df.empty:
                df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)
                df['profit_factor'] = df.apply(
                    lambda row: (row['wins'] * abs(row['avg_pnl'])) / (row['losses'] * abs(row['avg_pnl'])) 
                    if row['losses'] > 0 and row['avg_pnl'] < 0 else float('inf'), axis=1
                )
                
            return df
            
        except Exception as e:
            st.error(f"Error fetching pair statistics: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def fetch_hourly_distribution(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch trading performance by hour of day"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                EXTRACT(HOUR FROM timestamp) as hour_of_day,
                COUNT(*) as trades_count,
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                COALESCE(SUM(profit_loss), 0) as hourly_pnl,
                COALESCE(AVG(profit_loss), 0) as avg_pnl
            FROM trade_log 
            WHERE timestamp BETWEEN %s AND %s
            GROUP BY EXTRACT(HOUR FROM timestamp)
            ORDER BY hour_of_day
            """
            
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            
            if not df.empty:
                df['win_rate'] = (df['wins'] / df['trades_count'] * 100).round(1)
                
            return df
            
        except Exception as e:
            st.error(f"Error fetching hourly distribution: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

def main():
    """Main statistics dashboard"""
    analyzer = TradingStatisticsAnalyzer()
    
    # Header
    st.title("📊 Comprehensive Trading Statistics")
    st.markdown("---")
    
    # Sidebar controls
    render_sidebar_controls()
    
    # Date range from session state
    start_date = st.session_state.get('start_date', datetime.now() - timedelta(days=30))
    end_date = st.session_state.get('end_date', datetime.now())
    pairs_filter = st.session_state.get('pairs_filter', [])
    
    # Fetch data
    stats = analyzer.fetch_comprehensive_stats(start_date, end_date, pairs_filter)
    daily_df = analyzer.fetch_daily_performance(start_date, end_date)
    pair_df = analyzer.fetch_pair_statistics(start_date, end_date)
    hourly_df = analyzer.fetch_hourly_distribution(start_date, end_date)
    
    # Render sections
    if stats['total_trades'] > 0:
        render_overview_metrics(stats)
        render_performance_charts(stats, daily_df)
        render_pair_analysis(pair_df)
        render_time_analysis(hourly_df, daily_df)
        render_detailed_breakdowns(stats)
    else:
        st.info("No trading data found for the selected period and filters.")

def render_sidebar_controls():
    """Render sidebar control panel"""
    st.sidebar.title("📊 Statistics Controls")
    
    # Date range selector
    st.sidebar.subheader("📅 Date Range")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            key='start_date'
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.now(),
            key='end_date'
        )
    
    # Quick date presets
    st.sidebar.subheader("⚡ Quick Presets")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📅 Last 7 Days"):
            st.session_state.start_date = datetime.now() - timedelta(days=7)
            st.session_state.end_date = datetime.now()
            st.rerun()
        
        if st.button("📅 Last 30 Days"):
            st.session_state.start_date = datetime.now() - timedelta(days=30)
            st.session_state.end_date = datetime.now()
            st.rerun()
    
    with col2:
        if st.button("📅 Last 90 Days"):
            st.session_state.start_date = datetime.now() - timedelta(days=90)
            st.session_state.end_date = datetime.now()
            st.rerun()
        
        if st.button("📅 This Month"):
            now = datetime.now()
            st.session_state.start_date = now.replace(day=1)
            st.session_state.end_date = now
            st.rerun()
    
    # Pair filter
    st.sidebar.subheader("🔍 Filters")
    available_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "EURGBP"]
    pairs_filter = st.sidebar.multiselect(
        "Currency Pairs",
        options=available_pairs,
        default=[],
        key='pairs_filter'
    )
    
    # Export options
    st.sidebar.subheader("📤 Export")
    if st.sidebar.button("📥 Download Report"):
        st.sidebar.success("Export functionality coming soon!")
    
    # Refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

def render_overview_metrics(stats: Dict):
    """Render overview metrics cards"""
    st.subheader("📊 Performance Overview")
    
    # Top row metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pnl_color = "profit-text" if stats['total_pnl'] >= 0 else "loss-text"
        card_class = "profit-card" if stats['total_pnl'] >= 0 else "loss-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>💰 Total P&L</h4>
            <p class="big-number {pnl_color}">{stats['total_pnl']:.2f} SEK</p>
            <small>{stats['total_trades']} total trades</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        win_rate_color = "profit-text" if stats['win_rate'] >= 60 else "loss-text" if stats['win_rate'] < 40 else "neutral-text"
        card_class = "profit-card" if stats['win_rate'] >= 60 else "loss-card" if stats['win_rate'] < 40 else "neutral-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>🎯 Win Rate</h4>
            <p class="big-number {win_rate_color}">{stats['win_rate']:.1f}%</p>
            <small>{stats['wins']}W / {stats['losses']}L</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pf_color = "profit-text" if stats['profit_factor'] >= 1.5 else "neutral-text" if stats['profit_factor'] >= 1.0 else "loss-text"
        card_class = "profit-card" if stats['profit_factor'] >= 1.5 else "neutral-card" if stats['profit_factor'] >= 1.0 else "loss-card"
        
        pf_display = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "∞"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>📈 Profit Factor</h4>
            <p class="big-number {pf_color}">{pf_display}</p>
            <small>Profits/Losses ratio</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="profit-card">
            <h4>🏆 Best Trade</h4>
            <p class="big-number profit-text">+{stats['best_trade']:.2f}</p>
            <small>Largest single win</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="loss-card">
            <h4>📉 Worst Trade</h4>
            <p class="big-number loss-text">{stats['worst_trade']:.2f}</p>
            <small>Largest single loss</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Trading Days", stats['trading_days'])
    
    with col2:
        st.metric("📊 Avg Daily Trades", f"{stats['avg_daily_trades']:.1f}")
    
    with col3:
        st.metric("💰 Avg Daily P&L", f"{stats['avg_daily_pnl']:.2f} SEK")
    
    with col4:
        st.metric("🔄 Active Pairs", stats['unique_pairs'])

def render_performance_charts(stats: Dict, daily_df: pd.DataFrame):
    """Render performance visualization charts"""
    st.subheader("📈 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Win/Loss pie chart
        pie_data = pd.DataFrame({
            'Category': ['Wins', 'Losses', 'Pending'],
            'Count': [stats['wins'], stats['losses'], stats['pending']]
        })
        
        fig_pie = px.pie(
            pie_data,
            values='Count',
            names='Category',
            title="Trade Distribution",
            color_discrete_map={'Wins': '#28a745', 'Losses': '#dc3545', 'Pending': '#6c757d'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # P&L breakdown
        pnl_data = pd.DataFrame({
            'Type': ['Total Profits', 'Total Losses', 'Net P&L'],
            'Amount': [stats['total_profits'], -stats['total_losses'], stats['total_pnl']]
        })
        
        fig_bar = px.bar(
            pnl_data,
            x='Type',
            y='Amount',
            title="P&L Breakdown (SEK)",
            color='Amount',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Daily performance chart
    if not daily_df.empty:
        fig_daily = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily P&L', 'Cumulative P&L'),
            vertical_spacing=0.1
        )
        
        # Daily P&L bars
        fig_daily.add_trace(
            go.Bar(
                x=daily_df['trade_date'],
                y=daily_df['daily_pnl'],
                name='Daily P&L',
                marker_color=['green' if x >= 0 else 'red' for x in daily_df['daily_pnl']]
            ),
            row=1, col=1
        )
        
        # Cumulative P&L line
        fig_daily.add_trace(
            go.Scatter(
                x=daily_df['trade_date'],
                y=daily_df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig_daily.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig_daily.update_layout(height=600, title_text="Daily Performance Trends")
        
        st.plotly_chart(fig_daily, use_container_width=True)

def render_pair_analysis(pair_df: pd.DataFrame):
    """Render currency pair performance analysis"""
    st.subheader("🔍 Currency Pair Performance")
    
    if pair_df.empty:
        st.info("No pair data available for the selected period.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pair P&L chart
        fig_pairs = px.bar(
            pair_df.head(10),
            x='symbol',
            y='total_pnl',
            title="P&L by Currency Pair",
            color='total_pnl',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_pairs.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_pairs, use_container_width=True)
    
    with col2:
        # Win rate vs trades scatter
        fig_scatter = px.scatter(
            pair_df,
            x='total_trades',
            y='win_rate',
            size='total_pnl',
            color='total_pnl',
            hover_name='symbol',
            title="Win Rate vs Trade Volume",
            labels={'total_trades': 'Total Trades', 'win_rate': 'Win Rate (%)'},
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Win Rate")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detailed pair statistics table
    st.subheader("📋 Detailed Pair Statistics")
    
    display_columns = ['symbol', 'total_trades', 'win_rate', 'total_pnl', 'avg_pnl', 'best_trade', 'worst_trade']
    
    st.dataframe(
        pair_df[display_columns],
        column_config={
            "symbol": "Pair",
            "total_trades": "Trades",
            "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
            "total_pnl": st.column_config.NumberColumn("Total P&L", format="%.2f"),
            "avg_pnl": st.column_config.NumberColumn("Avg P&L", format="%.2f"),
            "best_trade": st.column_config.NumberColumn("Best Trade", format="%.2f"),
            "worst_trade": st.column_config.NumberColumn("Worst Trade", format="%.2f")
        },
        use_container_width=True,
        hide_index=True
    )

def render_time_analysis(hourly_df: pd.DataFrame, daily_df: pd.DataFrame):
    """Render time-based analysis"""
    st.subheader("⏰ Time-Based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly distribution
        if not hourly_df.empty:
            fig_hourly = px.bar(
                hourly_df,
                x='hour_of_day',
                y='hourly_pnl',
                title="P&L by Hour of Day",
                color='hourly_pnl',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig_hourly.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_hourly.update_xaxes(title="Hour (24h format)")
            fig_hourly.update_yaxes(title="P&L (SEK)")
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.info("No hourly data available.")
    
    with col2:
        # Daily win rate trend
        if not daily_df.empty:
            fig_winrate = px.line(
                daily_df,
                x='trade_date',
                y='daily_win_rate',
                title="Daily Win Rate Trend",
                markers=True
            )
            fig_winrate.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Threshold")
            fig_winrate.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="60% Target")
            fig_winrate.update_yaxes(title="Win Rate (%)")
            st.plotly_chart(fig_winrate, use_container_width=True)
        else:
            st.info("No daily trend data available.")

def render_detailed_breakdowns(stats: Dict):
    """Render detailed statistical breakdowns"""
    st.subheader("📊 Detailed Breakdowns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Profit/Loss Analysis")
        st.write(f"**Average Win:** {stats['avg_win']:.2f} SEK")
        st.write(f"**Average Loss:** {stats['avg_loss']:.2f} SEK")
        st.write(f"**Risk/Reward Ratio:** {abs(stats['avg_win'] / stats['avg_loss']):.2f}" if stats['avg_loss'] != 0 else "Risk/Reward Ratio: ∞")
        st.write(f"**Profit Factor:** {stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "**Profit Factor:** ∞")
        
        expectancy = (stats['win_rate'] / 100 * stats['avg_win']) + ((100 - stats['win_rate']) / 100 * stats['avg_loss'])
        st.write(f"**Expectancy:** {expectancy:.2f} SEK per trade")
    
    with col2:
        st.markdown("### 📈 Trading Activity")
        st.write(f"**Total Trades:** {stats['total_trades']}")
        st.write(f"**Winning Trades:** {stats['wins']}")
        st.write(f"**Losing Trades:** {stats['losses']}")
        st.write(f"**Pending Trades:** {stats['pending']}")
        st.write(f"**Trading Days:** {stats['trading_days']}")
        st.write(f"**Avg Trades/Day:** {stats['avg_daily_trades']:.1f}")

# Run the main application
if __name__ == "__main__":
    main()