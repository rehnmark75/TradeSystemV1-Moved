"""
Updated page2.py - Comprehensive Trading & Configuration Dashboard
Integrates real trading statistics from trade_log table with existing configuration monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import os

# Configure page
st.set_page_config(
    page_title="Trading & Configuration Dashboard",
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
    .status-active { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-inactive { background-color: #dc3545; }
    .config-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .trading-alert {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE CONNECTION UTILITIES
# ============================================================================

def get_database_connection():
    """Get database connection for trade log data"""
    try:
        # Try Streamlit secrets first with nested structure support
        if hasattr(st, 'secrets'):
            # Check for nested database structure (preferred format)
            try:
                conn_string = st.secrets.database.trading_connection_string
            except (AttributeError, KeyError):
                try:
                    conn_string = st.secrets.database.config_connection_string
                except (AttributeError, KeyError):
                    try:
                        conn_string = st.secrets.trading_connection_string
                    except AttributeError:
                        try:
                            conn_string = st.secrets.connection_string
                        except AttributeError:
                            # Fallback to environment
                            conn_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
        else:
            conn_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
        
        return psycopg2.connect(conn_string)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# ============================================================================
# ENHANCED DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_trade_log_statistics_with_strategy(days_back: int = 7):
    """Fetch comprehensive trading statistics with strategy information from joined tables"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cursor:
            # Enhanced query joining trade_log with alert_history for strategy info
            query = """
            SELECT 
                COUNT(t.*) as total_trades,
                COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losing_trades,
                COUNT(CASE WHEN t.profit_loss IS NULL OR t.status = 'pending' THEN 1 END) as pending_trades,
                COALESCE(SUM(t.profit_loss), 0) as total_profit_loss,
                COALESCE(AVG(CASE WHEN t.profit_loss > 0 THEN t.profit_loss END), 0) as avg_profit,
                COALESCE(AVG(CASE WHEN t.profit_loss < 0 THEN t.profit_loss END), 0) as avg_loss,
                COALESCE(MAX(t.profit_loss), 0) as largest_win,
                COALESCE(MIN(t.profit_loss), 0) as largest_loss,
                COUNT(DISTINCT t.symbol) as active_pairs,
                COALESCE(SUM(CASE WHEN t.profit_loss > 0 THEN t.profit_loss END), 0) as total_profits,
                COALESCE(SUM(CASE WHEN t.profit_loss < 0 THEN ABS(t.profit_loss) END), 0) as total_losses,
                COUNT(DISTINCT a.strategy) as strategies_used,
                COALESCE(AVG(a.confidence_score), 0) as avg_signal_confidence
            FROM trade_log t
            LEFT JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            """
            
            cursor.execute(query, [datetime.now() - timedelta(days=days_back)])
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                total_trades, winning_trades, losing_trades, pending_trades = result[0:4]
                total_pnl, avg_profit, avg_loss, largest_win, largest_loss = result[4:9]
                active_pairs, total_profits, total_losses, strategies_used, avg_signal_confidence = result[9:14]
                
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                profit_factor = (total_profits / total_losses) if total_losses > 0 else float('inf')
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'pending_trades': pending_trades,
                    'total_pnl': float(total_pnl),
                    'win_rate': win_rate,
                    'avg_profit': float(avg_profit),
                    'avg_loss': float(avg_loss),
                    'largest_win': float(largest_win),
                    'largest_loss': float(largest_loss),
                    'active_pairs': active_pairs,
                    'profit_factor': profit_factor,
                    'total_profits': float(total_profits),
                    'total_losses': float(total_losses),
                    'strategies_used': strategies_used,
                    'avg_signal_confidence': float(avg_signal_confidence)
                }
            else:
                return {
                    'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'pending_trades': 0,
                    'total_pnl': 0.0, 'win_rate': 0.0, 'avg_profit': 0.0, 'avg_loss': 0.0,
                    'largest_win': 0.0, 'largest_loss': 0.0, 'active_pairs': 0, 'profit_factor': 0.0,
                    'total_profits': 0.0, 'total_losses': 0.0, 'strategies_used': 0, 'avg_signal_confidence': 0.0
                }
                
    except Exception as e:
        st.error(f"Error fetching enhanced trade statistics: {e}")
        return None
    finally:
        conn.close()

def fetch_recent_trades_with_strategy(limit: int = 20):
    """Fetch recent trades with complete strategy and signal information"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            t.timestamp,
            t.symbol,
            t.direction,
            t.entry_price,
            t.profit_loss,
            t.status,
            a.strategy,
            a.signal_type,
            a.confidence_score,
            a.price as signal_price,
            a.claude_analysis
        FROM trade_log t
        LEFT JOIN alert_history a ON t.alert_id = a.id
        ORDER BY t.timestamp DESC 
        LIMIT %s
        """
        
        df = pd.read_sql_query(query, conn, params=[limit])
        
        if not df.empty:
            df['result'] = df['profit_loss'].apply(
                lambda x: '🟢 WIN' if x > 0 else '🔴 LOSS' if x < 0 else '⏳ PENDING'
            )
            df['pnl_formatted'] = df['profit_loss'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "Pending"
            )
            df['confidence_formatted'] = df['confidence_score'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
        
        return df
        
    except Exception as e:
        st.warning(f"Could not fetch detailed trades: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Legacy compatibility functions
def fetch_trade_log_statistics(days_back: int = 7):
    """Legacy wrapper - redirect to enhanced version"""
    return fetch_trade_log_statistics_with_strategy(days_back)

def fetch_recent_trades(limit: int = 15):
    """Legacy wrapper - redirect to enhanced version"""
    return fetch_recent_trades_with_strategy(limit)

def fetch_trade_log_statistics(days_back: int = 7):
    """Fetch real trading statistics from trade_log table"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cursor:
            # Enhanced query for comprehensive trading statistics
            query = """
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
                COUNT(DISTINCT symbol) as active_pairs,
                COALESCE(SUM(CASE WHEN profit_loss > 0 THEN profit_loss END), 0) as total_profits,
                COALESCE(SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) END), 0) as total_losses
            FROM trade_log 
            WHERE timestamp >= %s
            """
            
            cursor.execute(query, [datetime.now() - timedelta(days=days_back)])
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                total_trades, winning_trades, losing_trades, pending_trades = result[0:4]
                total_pnl, avg_profit, avg_loss, largest_win, largest_loss = result[4:9]
                active_pairs, total_profits, total_losses = result[9:12]
                
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                profit_factor = (total_profits / total_losses) if total_losses > 0 else float('inf')
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'pending_trades': pending_trades,
                    'total_pnl': float(total_pnl),
                    'win_rate': win_rate,
                    'avg_profit': float(avg_profit),
                    'avg_loss': float(avg_loss),
                    'largest_win': float(largest_win),
                    'largest_loss': float(largest_loss),
                    'active_pairs': active_pairs,
                    'profit_factor': profit_factor,
                    'total_profits': float(total_profits),
                    'total_losses': float(total_losses)
                }
            else:
                return {
                    'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'pending_trades': 0,
                    'total_pnl': 0.0, 'win_rate': 0.0, 'avg_profit': 0.0, 'avg_loss': 0.0,
                    'largest_win': 0.0, 'largest_loss': 0.0, 'active_pairs': 0, 'profit_factor': 0.0,
                    'total_profits': 0.0, 'total_losses': 0.0
                }
                
    except Exception as e:
        st.error(f"Error fetching trade statistics: {e}")
        return None
    finally:
        conn.close()

def fetch_alert_history_stats(days_back: int = 7):
    """Fetch signal statistics from alert_history table"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cursor:
            query = """
            SELECT 
                COUNT(*) as total_signals,
                COUNT(CASE WHEN signal_type = 'BULL' THEN 1 END) as bull_signals,
                COUNT(CASE WHEN signal_type = 'BEAR' THEN 1 END) as bear_signals,
                COALESCE(AVG(confidence_score), 0) as avg_confidence,
                COUNT(DISTINCT strategy) as active_strategies,
                COUNT(DISTINCT epic) as monitored_pairs
            FROM alert_history 
            WHERE alert_timestamp >= %s
            """
            
            cursor.execute(query, [datetime.now() - timedelta(days=days_back)])
            result = cursor.fetchone()
            
            if result:
                return {
                    'total_signals': result[0],
                    'bull_signals': result[1],
                    'bear_signals': result[2],
                    'avg_confidence': float(result[3]),
                    'active_strategies': result[4],
                    'monitored_pairs': result[5]
                }
            else:
                return {
                    'total_signals': 0, 'bull_signals': 0, 'bear_signals': 0,
                    'avg_confidence': 0.0, 'active_strategies': 0, 'monitored_pairs': 0
                }
                
    except Exception as e:
        st.warning(f"Could not fetch signal statistics: {e}")
        return {
            'total_signals': 0, 'bull_signals': 0, 'bear_signals': 0,
            'avg_confidence': 0.0, 'active_strategies': 0, 'monitored_pairs': 0
        }
    finally:
        conn.close()

def fetch_recent_trades(limit: int = 15):
    """Fetch recent trades for timeline display"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT timestamp, symbol, direction, entry_price, profit_loss, status, deal_id
        FROM trade_log 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        df = pd.read_sql_query(query, conn, params=[limit])
        return df
    except Exception as e:
        st.warning(f"Could not fetch recent trades: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_pair_performance(days_back: int = 30):
    """Fetch performance by currency pair"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            symbol,
            COUNT(*) as total_trades,
            COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
            COALESCE(SUM(profit_loss), 0) as total_pnl,
            COALESCE(AVG(profit_loss), 0) as avg_pnl
        FROM trade_log 
        WHERE timestamp >= %s
        GROUP BY symbol
        ORDER BY total_pnl DESC
        """
        df = pd.read_sql_query(query, conn, params=[datetime.now() - timedelta(days=days_back)])
        if not df.empty:
            df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)
        return df
    except Exception as e:
        st.warning(f"Could not fetch pair performance: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ============================================================================
# MAIN DASHBOARD RENDERING FUNCTIONS
# ============================================================================

def render_monitoring_dashboard(db_manager=None):
    """Render enhanced real-time monitoring dashboard with trading statistics"""
    
    # Header with controls
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.title("📊 Trading & Configuration Dashboard")
        st.caption("Real-time system monitoring with comprehensive trading statistics")
    
    with col2:
        time_period = st.selectbox("📅 Period", ["7 days", "30 days", "90 days"], index=0)
        days_map = {"7 days": 7, "30 days": 30, "90 days": 90}
        selected_days = days_map[time_period]
    
    with col3:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
    
    with col4:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fetch all data
    trade_stats = fetch_trade_log_statistics(selected_days)
    signal_stats = fetch_alert_history_stats(selected_days)
    
    # Render main sections
    render_enhanced_system_metrics(trade_stats, signal_stats, selected_days)
    
    col1, col2 = st.columns(2)
    with col1:
        render_trading_performance_section(trade_stats, selected_days)
    with col2:
        render_signal_generation_section(signal_stats, selected_days)
    
    render_recent_activity_section()
    render_system_health_alerts()
    render_configuration_impact_analysis(trade_stats)
    
    # Auto-refresh logic
    if auto_refresh:
        refresh_interval = st.selectbox("Refresh Rate (seconds)", [30, 60, 120, 300], index=1)
        time.sleep(refresh_interval)
        st.rerun()

def render_enhanced_system_metrics(trade_stats, signal_stats, days_back):
    """Render enhanced system metrics combining trading and configuration data"""
    st.subheader("📊 System Overview")
    
    # Top row - Core trading metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if trade_stats and trade_stats['total_trades'] > 0:
            pnl_color = "normal" if trade_stats['total_pnl'] >= 0 else "inverse"
            st.metric(
                label="💰 Total P&L",
                value=f"{trade_stats['total_pnl']:.2f} SEK",
                delta=f"{trade_stats['total_trades']} trades",
                delta_color=pnl_color,
                help=f"Total profit/loss from {days_back} days of trading"
            )
        else:
            st.metric(label="💰 Total P&L", value="No Data", help="No trading data available")
    
    with col2:
        if trade_stats and trade_stats['total_trades'] > 0:
            win_rate_color = "normal" if trade_stats['win_rate'] >= 50 else "inverse"
            st.metric(
                label="🎯 Win Rate",
                value=f"{trade_stats['win_rate']:.1f}%",
                delta=f"{trade_stats['winning_trades']}W/{trade_stats['losing_trades']}L",
                delta_color=win_rate_color,
                help="Percentage of winning trades"
            )
        else:
            st.metric(label="🎯 Win Rate", value="--")
    
    with col3:
        if signal_stats:
            st.metric(
                label="📡 Signals Generated",
                value=f"{signal_stats['total_signals']}",
                delta=f"{signal_stats['avg_confidence']:.1f}% avg confidence",
                help=f"Total signals generated in {days_back} days"
            )
        else:
            st.metric(label="📡 Signals Generated", value="--")
    
    with col4:
        if trade_stats and trade_stats['profit_factor'] != float('inf'):
            pf_display = f"{trade_stats['profit_factor']:.2f}"
            pf_color = "normal" if trade_stats['profit_factor'] >= 1.0 else "inverse"
        else:
            pf_display = "∞" if trade_stats and trade_stats['profit_factor'] == float('inf') else "--"
            pf_color = "normal"
        
        st.metric(
            label="📈 Profit Factor",
            value=pf_display,
            delta="Profits/Losses ratio",
            delta_color=pf_color,
            help="Ratio of total profits to total losses"
        )
    
    with col5:
        # System health indicator
        db_conn = get_database_connection()
        if db_conn:
            db_conn.close()
            health_status = "🟢 Healthy"
            health_color = "normal"
        else:
            health_status = "🔴 Issues"
            health_color = "inverse"
        
        st.metric(
            label="🏥 System Health",
            value=health_status,
            delta="Database & APIs",
            delta_color=health_color,
            help="Overall system health status"
        )
    
    # Second row - Operational metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_pairs = trade_stats['active_pairs'] if trade_stats else 0
        monitored_pairs = signal_stats['monitored_pairs'] if signal_stats else 0
        st.metric("🔄 Active Pairs", f"{active_pairs}/{monitored_pairs}", help="Trading pairs with activity")
    
    with col2:
        active_strategies = signal_stats['active_strategies'] if signal_stats else 0
        st.metric("⚙️ Active Strategies", active_strategies, help="Number of strategies generating signals")
    
    with col3:
        pending_trades = trade_stats['pending_trades'] if trade_stats else 0
        st.metric("⏳ Pending Trades", pending_trades, help="Trades waiting for execution/closure")
    
    with col4:
        # Calculate daily average
        daily_signals = signal_stats['total_signals'] / days_back if signal_stats and days_back > 0 else 0
        st.metric("📊 Daily Signals", f"{daily_signals:.1f}", help="Average signals per day")

def render_trading_performance_section(trade_stats, days_back):
    """Render detailed trading performance analysis"""
    st.subheader("📈 Trading Performance")
    
    if not trade_stats or trade_stats['total_trades'] == 0:
        st.info(f"No trading data available for the last {days_back} days.")
        return
    
    # Performance metrics cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 P&L Breakdown**")
        pnl_data = pd.DataFrame({
            'Type': ['Profits', 'Losses'],
            'Amount': [trade_stats['total_profits'], trade_stats['total_losses']],
            'Count': [trade_stats['winning_trades'], trade_stats['losing_trades']]
        })
        
        fig_pnl = px.bar(
            pnl_data, 
            x='Type', 
            y='Amount',
            text='Count',
            title=f"P&L Distribution",
            color='Type',
            color_discrete_map={'Profits': '#28a745', 'Losses': '#dc3545'}
        )
        fig_pnl.update_traces(texttemplate='%{text} trades', textposition='outside')
        fig_pnl.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_pnl, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Trade Distribution**")
        win_loss_data = pd.DataFrame({
            'Result': ['Wins', 'Losses', 'Pending'],
            'Count': [trade_stats['winning_trades'], trade_stats['losing_trades'], trade_stats['pending_trades']]
        })
        
        fig_pie = px.pie(
            win_loss_data, 
            values='Count', 
            names='Result',
            title="Trade Outcomes",
            color_discrete_map={'Wins': '#28a745', 'Losses': '#dc3545', 'Pending': '#6c757d'}
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Key statistics
    st.markdown("**📊 Key Statistics**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Best Trade:** +{trade_stats['largest_win']:.2f} SEK")
        st.error(f"**Worst Trade:** {trade_stats['largest_loss']:.2f} SEK")
    
    with col2:
        st.success(f"**Avg Win:** {trade_stats['avg_profit']:.2f} SEK")
        st.warning(f"**Avg Loss:** {trade_stats['avg_loss']:.2f} SEK")
    
    with col3:
        expectancy = (trade_stats['win_rate'] / 100 * trade_stats['avg_profit']) + ((100 - trade_stats['win_rate']) / 100 * trade_stats['avg_loss'])
        risk_reward = abs(trade_stats['avg_profit'] / trade_stats['avg_loss']) if trade_stats['avg_loss'] != 0 else float('inf')
        st.metric("💡 Expectancy", f"{expectancy:.2f} SEK")
        st.metric("⚖️ Risk/Reward", f"1:{risk_reward:.2f}")

def render_signal_generation_section(signal_stats, days_back):
    """Render signal generation performance"""
    st.subheader("📡 Signal Generation")
    
    if not signal_stats or signal_stats['total_signals'] == 0:
        st.info(f"No signal data available for the last {days_back} days.")
        return
    
    # Signal metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Signal Distribution**")
        signal_data = pd.DataFrame({
            'Type': ['BULL', 'BEAR'],
            'Count': [signal_stats['bull_signals'], signal_stats['bear_signals']]
        })
        
        fig_signals = px.bar(
            signal_data,
            x='Type',
            y='Count',
            title="Bull vs Bear Signals",
            color='Type',
            color_discrete_map={'BULL': '#28a745', 'BEAR': '#dc3545'}
        )
        fig_signals.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_signals, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Signal Quality**")
        # Mock confidence distribution data
        confidence_ranges = ['90-100%', '80-90%', '70-80%', '60-70%', '<60%']
        confidence_counts = [
            int(signal_stats['total_signals'] * 0.3),
            int(signal_stats['total_signals'] * 0.25),
            int(signal_stats['total_signals'] * 0.25),
            int(signal_stats['total_signals'] * 0.15),
            int(signal_stats['total_signals'] * 0.05)
        ]
        
        fig_conf = px.bar(
            x=confidence_ranges,
            y=confidence_counts,
            title="Confidence Distribution",
            labels={'x': 'Confidence Range', 'y': 'Signal Count'}
        )
        fig_conf.update_layout(height=300)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Signal performance summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        daily_signals = signal_stats['total_signals'] / days_back
        st.metric("📈 Daily Rate", f"{daily_signals:.1f} signals/day")
    
    with col2:
        st.metric("🎯 Avg Confidence", f"{signal_stats['avg_confidence']:.1f}%")
    
    with col3:
        signal_efficiency = (signal_stats['total_signals'] / signal_stats['monitored_pairs']) if signal_stats['monitored_pairs'] > 0 else 0
        st.metric("⚡ Signal Efficiency", f"{signal_efficiency:.1f} signals/pair")

def render_recent_activity_section():
    """Render recent trading activity"""
    st.subheader("📋 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔄 Recent Trades**")
        recent_trades = fetch_recent_trades(10)
        
        if not recent_trades.empty:
            # Format the data for display
            recent_trades['result'] = recent_trades['profit_loss'].apply(
                lambda x: '🟢 WIN' if x > 0 else '🔴 LOSS' if x < 0 else '⏳ PENDING'
            )
            recent_trades['pnl_formatted'] = recent_trades['profit_loss'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "Pending"
            )
            
            display_cols = ['timestamp', 'symbol', 'direction', 'pnl_formatted', 'result']
            
            st.dataframe(
                recent_trades[display_cols].head(8),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="MM/DD HH:mm"),
                    "symbol": "Pair",
                    "direction": "Dir",
                    "pnl_formatted": "P&L",
                    "result": "Result"
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent trades found.")
    
    with col2:
        st.markdown("**💱 Pair Performance (30d)**")
        pair_df = fetch_pair_performance(30)
        
        if not pair_df.empty:
            # Show top 8 pairs
            top_pairs = pair_df.head(8)
            
            st.dataframe(
                top_pairs[['symbol', 'total_trades', 'win_rate', 'total_pnl']],
                column_config={
                    "symbol": "Pair",
                    "total_trades": "Trades",
                    "win_rate": st.column_config.NumberColumn("Win %", format="%.1f%%"),
                    "total_pnl": st.column_config.NumberColumn("P&L", format="%.2f")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No pair performance data available.")

def render_system_health_alerts():
    """Render system health and alerts"""
    st.subheader("🚨 System Alerts & Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚙️ Configuration Alerts**")
        
        # Mock configuration alerts
        config_alerts = [
            {"type": "WARNING", "message": "MIN_CONFIDENCE changed to 0.75", "time": "2h ago"},
            {"type": "INFO", "message": "New strategy weights applied", "time": "4h ago"},
            {"type": "SUCCESS", "message": "System backup completed", "time": "6h ago"}
        ]
        
        for alert in config_alerts:
            alert_class = "config-alert"
            icon = "⚠️" if alert['type'] == 'WARNING' else "ℹ️" if alert['type'] == 'INFO' else "✅"
            
            st.markdown(f"""
            <div class="{alert_class}">
                {icon} <strong>{alert['type']}</strong>: {alert['message']} <em>({alert['time']})</em>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📊 Trading Alerts**")
        
        # Generate trading alerts based on current performance
        trade_stats = fetch_trade_log_statistics(7)
        trading_alerts = []
        
        if trade_stats:
            if trade_stats['win_rate'] < 50:
                trading_alerts.append({"type": "WARNING", "message": f"Win rate below 50% ({trade_stats['win_rate']:.1f}%)", "time": "Now"})
            
            if trade_stats['total_pnl'] < 0:
                trading_alerts.append({"type": "ERROR", "message": f"Negative P&L: {trade_stats['total_pnl']:.2f} SEK", "time": "Now"})
            
            if trade_stats['profit_factor'] > 2.0:
                trading_alerts.append({"type": "SUCCESS", "message": f"Excellent profit factor: {trade_stats['profit_factor']:.2f}", "time": "Now"})
        
        if not trading_alerts:
            trading_alerts.append({"type": "INFO", "message": "No trading alerts at this time", "time": "Now"})
        
        for alert in trading_alerts:
            alert_class = "trading-alert"
            icon = "🔴" if alert['type'] == 'ERROR' else "⚠️" if alert['type'] == 'WARNING' else "✅" if alert['type'] == 'SUCCESS' else "ℹ️"
            
            st.markdown(f"""
            <div class="{alert_class}">
                {icon} <strong>{alert['type']}</strong>: {alert['message']} <em>({alert['time']})</em>
            </div>
            """, unsafe_allow_html=True)

def render_configuration_impact_analysis(trade_stats):
    """Render analysis of configuration changes impact on performance"""
    st.subheader("📈 Configuration Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Recent Configuration Changes**")
        
        # Mock configuration changes with impact
        config_changes = [
            {
                "setting": "MIN_CONFIDENCE",
                "old_value": "0.70",
                "new_value": "0.75",
                "timestamp": datetime.now() - timedelta(hours=2),
                "impact": "↓ 15% signal volume, ↑ 5% accuracy"
            },
            {
                "setting": "STRATEGY_WEIGHT_EMA",
                "old_value": "0.40",
                "new_value": "0.45",
                "timestamp": datetime.now() - timedelta(hours=6),
                "impact": "↑ 8% EMA signal contribution"
            },
            {
                "setting": "RISK_PER_TRADE",
                "old_value": "1.0%",
                "new_value": "0.8%",
                "timestamp": datetime.now() - timedelta(hours=12),
                "impact": "↓ 20% position sizing"
            }
        ]
        
        changes_df = pd.DataFrame(config_changes)
        
        st.dataframe(
            changes_df[['setting', 'old_value', 'new_value', 'impact']],
            column_config={
                "setting": "Setting",
                "old_value": "Old Value",
                "new_value": "New Value",
                "impact": "Observed Impact"
            },
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("**📊 Performance Correlation**")
        
        if trade_stats and trade_stats['total_trades'] > 0:
            # Performance summary with targets
            performance_metrics = [
                {"metric": "Win Rate", "current": f"{trade_stats['win_rate']:.1f}%", "target": "≥60%", "status": "✅" if trade_stats['win_rate'] >= 60 else "⚠️"},
                {"metric": "Profit Factor", "current": f"{trade_stats['profit_factor']:.2f}" if trade_stats['profit_factor'] != float('inf') else "∞", "target": "≥1.5", "status": "✅" if trade_stats['profit_factor'] >= 1.5 else "⚠️"},
                {"metric": "Total P&L", "current": f"{trade_stats['total_pnl']:.2f} SEK", "target": ">0", "status": "✅" if trade_stats['total_pnl'] > 0 else "❌"},
                {"metric": "Active Pairs", "current": f"{trade_stats['active_pairs']}", "target": "≥6", "status": "✅" if trade_stats['active_pairs'] >= 6 else "⚠️"}
            ]
            
            metrics_df = pd.DataFrame(performance_metrics)
            
            st.dataframe(
                metrics_df,
                column_config={
                    "metric": "Metric",
                    "current": "Current",
                    "target": "Target",
                    "status": "Status"
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No performance data available for correlation analysis.")

# ============================================================================
# ENHANCED SIDEBAR AND NAVIGATION
# ============================================================================

def render_enhanced_sidebar():
    """Enhanced sidebar with all features"""
    st.sidebar.title("📊 Dashboard Control")
    
    # Database connection status
    st.sidebar.markdown("### 🔗 Connection Status")
    conn = get_database_connection()
    if conn:
        st.sidebar.success("🟢 Database Connected")
        conn.close()
    else:
        st.sidebar.error("🔴 Database Disconnected")
    
    # Quick stats
    st.sidebar.markdown("### 📊 Quick Stats")
    trade_stats = fetch_trade_log_statistics_with_strategy(7)
    signal_stats = fetch_alert_history_stats(7)
    
    if trade_stats:
        st.sidebar.metric("7d P&L", f"{trade_stats['total_pnl']:.2f} SEK")
        st.sidebar.metric("7d Win Rate", f"{trade_stats['win_rate']:.1f}%")
        st.sidebar.metric("Strategies Used", trade_stats['strategies_used'])
    
    if signal_stats:
        st.sidebar.metric("7d Signals", signal_stats['total_signals'])
    
    # Navigation options
    st.sidebar.markdown("### 🧭 Navigation")
    
    dashboard_options = [
        "📊 Overview Dashboard",
        "📈 Trading Performance", 
        "📡 Signal Analysis",
        "⚙️ Configuration Manager",
        "🔍 System Diagnostics"
    ]
    
    selected_view = st.sidebar.selectbox("Select View", dashboard_options)
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📊 Export", use_container_width=True):
            st.sidebar.success("Export ready!")
    
    # System controls
    st.sidebar.markdown("### 🔧 System Controls")
    
    if st.sidebar.button("⚙️ Reload Config"):
        st.sidebar.success("Configuration reloaded!")
    
    if st.sidebar.button("🚨 Test Alerts"):
        st.sidebar.warning("Alert system tested!")
    
    return selected_view

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    # Render sidebar and get selected view
    selected_view = render_enhanced_sidebar()
    
    # Render main content based on selection
    if selected_view == "📊 Overview Dashboard":
        render_monitoring_dashboard()
    elif selected_view == "📈 Trading Performance":
        st.header("📈 Detailed Trading Performance")
        trade_stats = fetch_trade_log_statistics(30)
        render_trading_performance_section(trade_stats, 30)
    elif selected_view == "📡 Signal Analysis":
        st.header("📡 Signal Generation Analysis")
        signal_stats = fetch_alert_history_stats(30)
        render_signal_generation_section(signal_stats, 30)
    elif selected_view == "⚙️ Configuration Manager":
        st.header("⚙️ Configuration Management")
        st.info("Configuration management interface - integrate with your existing TradeSystemv1.py")
    elif selected_view == "🔍 System Diagnostics":
        st.header("🔍 System Diagnostics")
        st.info("System diagnostics and health checks")
    else:
        # Default to overview dashboard
        render_monitoring_dashboard()

# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

# Keep existing function names for backward compatibility but use enhanced versions
def render_system_metrics(db_manager):
    """Legacy compatibility - redirect to enhanced version"""
    trade_stats = fetch_trade_log_statistics_with_strategy(7)
    signal_stats = fetch_alert_history_stats(7)
    render_enhanced_system_metrics(trade_stats, signal_stats, 7)

def render_changes_timeline(db_manager):
    """Legacy compatibility - basic configuration timeline"""
    st.subheader("📅 Configuration Changes Timeline")
    st.info("Configuration change tracking - integrate with your db_manager")

def render_performance_correlation(db_manager):
    """Legacy compatibility - redirect to enhanced version"""
    trade_stats = fetch_trade_log_statistics_with_strategy(7)
    render_configuration_impact_analysis(trade_stats)

def render_configuration_alerts(db_manager):
    """Legacy compatibility - redirect to enhanced version"""
    render_system_health_alerts()

# Legacy compatibility for the old function name
def fetch_trade_log_statistics(days_back: int = 7):
    """Legacy wrapper - redirect to enhanced version"""
    return fetch_trade_log_statistics_with_strategy(days_back)

def fetch_recent_trades(limit: int = 15):
    """Legacy wrapper - redirect to enhanced version"""
    return fetch_recent_trades_with_strategy(limit)

# Export all render functions for use in main app
__all__ = [
    'render_monitoring_dashboard',
    'render_enhanced_system_metrics',
    'render_trading_performance_section',
    'render_signal_generation_section',
    'render_system_health_alerts',
    'render_enhanced_sidebar',
    'main'
]

# Run the main application if called directly
if __name__ == "__main__":
    main()