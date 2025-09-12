"""
Simple Trade Performance Analysis - Streamlit Page

This page uses your EXACT query and keeps everything simple to avoid column errors.
Save as: streamlit/pages/simple_trade_analysis.py

Features:
- Your exact SQL query
- Basic P&L analysis
- Simple filtering
- No fancy charts that might break
"""

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Optional


# Configure page
st.set_page_config(
    page_title="Simple Trade Analysis",
    page_icon="💰",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)


def get_database_connection():
    """Get database connection"""
    try:
        # Try multiple secret locations
        if hasattr(st, 'secrets'):
            try:
                conn_string = st.secrets.database.trading_connection_string
                return psycopg2.connect(conn_string)
            except (AttributeError, KeyError):
                pass
            
            try:
                conn_string = st.secrets.trading_connection_string
                return psycopg2.connect(conn_string)
            except (AttributeError, KeyError):
                pass
            
            try:
                conn_string = st.secrets.DATABASE_URL
                return psycopg2.connect(conn_string)
            except (AttributeError, KeyError):
                pass
        
        # Try environment variable
        import os
        conn_string = os.getenv('DATABASE_URL')
        if conn_string:
            return psycopg2.connect(conn_string)
        
        return None
        
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def run_your_exact_query(date_filter: str = "2025-08-21"):
    """Run your exact original query"""
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        if date_filter == "today":
            date_condition = "DATE(tl.timestamp) = CURRENT_DATE"
        elif date_filter == "week":
            date_condition = "tl.timestamp >= CURRENT_DATE - INTERVAL '7 days'"
        elif date_filter == "month":
            date_condition = "tl.timestamp >= CURRENT_DATE - INTERVAL '30 days'"
        else:
            # Specific date
            date_condition = f"DATE(tl.timestamp) = '{date_filter}'"
        
        # YOUR EXACT QUERY with dynamic date
        query = f"""
            SELECT 
                tl.symbol, 
                tl.direction, 
                tl.timestamp as trade_timestamp,
                tl.alert_id, 
                tl.profit_loss,
                ah.epic,
                ah.signal_type,
                ah.confidence_score,
                ah.strategy,
                ah.strategy_indicators,
                ah.timeframe
            FROM trade_log tl
            INNER JOIN alert_history ah ON tl.alert_id = ah.id
            WHERE {date_condition}
            ORDER BY tl.alert_id DESC
        """
        
        df = pd.read_sql_query(query, conn)
        return df
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def calculate_simple_metrics(df: pd.DataFrame):
    """Calculate simple performance metrics"""
    if df.empty:
        return {}
    
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
    
    avg_profit = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
    
    best_trade = trades_with_pnl['profit_loss'].max()
    worst_trade = trades_with_pnl['profit_loss'].min()
    
    avg_confidence = df['confidence_score'].mean()
    
    return {
        'total_trades': total_trades,
        'completed_trades': len(trades_with_pnl),
        'pending_trades': len(pending_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'avg_confidence': avg_confidence,
        'has_data': True
    }


def display_metrics(metrics):
    """Display performance metrics"""
    if not metrics.get('has_data', False):
        st.warning("⚠️ No completed trades with P&L data found")
        if metrics.get('total_trades', 0) > 0:
            st.info(f"Found {metrics['total_trades']} total trades, but {metrics.get('pending_trades', 0)} are still pending")
        return
    
    # Top row metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pnl = metrics['total_pnl']
        pnl_class = 'profit' if pnl > 0 else 'loss' if pnl < 0 else ''
        st.markdown(f"""
        <div class="metric-box">
            <h3 class="{pnl_class}">{pnl:+.2f}</h3>
            <p>Total P&L</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        win_rate = metrics['win_rate']
        st.markdown(f"""
        <div class="metric-box">
            <h3>{win_rate:.1f}%</h3>
            <p>Win Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{metrics['completed_trades']}</h3>
            <p>Completed Trades</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{metrics['avg_confidence']:.3f}</h3>
            <p>Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Winning Trades", metrics['winning_trades'])
    
    with col2:
        st.metric("Losing Trades", metrics['losing_trades'])
    
    with col3:
        st.metric("Best Trade", f"{metrics['best_trade']:+.2f}")
    
    with col4:
        st.metric("Worst Trade", f"{metrics['worst_trade']:+.2f}")


def display_trade_table(df: pd.DataFrame):
    """Display trade data table"""
    if df.empty:
        st.info("No trades to display")
        return
    
    st.markdown("### 📋 Trade Details")
    
    # Format the data for display
    display_df = df.copy()
    
    # Format P&L column
    display_df['P&L'] = display_df['profit_loss'].apply(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "Pending"
    )
    
    # Format timestamp
    display_df['Trade Time'] = pd.to_datetime(display_df['trade_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Format confidence
    display_df['Confidence'] = display_df['confidence_score'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    
    # Select columns to show
    columns_to_show = [
        'alert_id', 'symbol', 'direction', 'strategy', 'signal_type',
        'P&L', 'Confidence', 'timeframe', 'Trade Time'
    ]
    
    # Color-code the dataframe
    def color_pnl(val):
        if 'Pending' in str(val):
            return 'background-color: #fff3cd'
        elif '+' in str(val):
            return 'background-color: #d4f6d4'
        elif '-' in str(val):
            return 'background-color: #f8d7da'
        return ''
    
    styled_df = display_df[columns_to_show].style.applymap(
        color_pnl, subset=['P&L']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Show raw data option
    if st.checkbox("Show Raw Data"):
        st.dataframe(df, use_container_width=True)


def main():
    """Main application"""
    st.title("💰 Simple Trade Performance Analysis")
    st.markdown("Analysis using your exact SQL query")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🔧 Controls")
        
        date_option = st.selectbox(
            "Date Filter",
            ["2025-08-21", "today", "week", "month"]
        )
        
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Test connection
    if st.sidebar.button("🔍 Test Connection"):
        conn = get_database_connection()
        if conn:
            st.sidebar.success("✅ Connected!")
            conn.close()
        else:
            st.sidebar.error("❌ Connection failed!")
    
    # Main content
    st.markdown(f"**Analyzing trades for:** {date_option}")
    
    # Get data
    with st.spinner("Loading trade data..."):
        df = run_your_exact_query(date_option)
    
    if df.empty:
        st.warning(f"⚠️ No trade data found for {date_option}")
        
        if date_option != "month":
            st.info("💡 Try selecting 'month' to see more data")
        
        # Show sample query
        st.markdown("### 🔍 Query Being Used:")
        sample_query = """
        SELECT 
            tl.symbol, tl.direction, tl.timestamp as trade_timestamp,
            tl.alert_id, tl.profit_loss,
            ah.epic, ah.signal_type, ah.confidence_score,
            ah.strategy, ah.strategy_indicators, ah.timeframe
        FROM trade_log tl
        INNER JOIN alert_history ah ON tl.alert_id = ah.id
        WHERE [DATE_CONDITION]
        ORDER BY tl.alert_id DESC
        """
        st.code(sample_query, language='sql')
        
    else:
        st.success(f"✅ Found {len(df)} trades!")
        
        # Calculate and display metrics
        metrics = calculate_simple_metrics(df)
        display_metrics(metrics)
        
        # Show strategy breakdown
        if not df.empty:
            st.markdown("### ⚡ Strategy Breakdown")
            strategy_counts = df['strategy'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(strategy_counts.reset_index(), use_container_width=True)
            
            with col2:
                # Simple bar chart
                st.bar_chart(strategy_counts)
        
        # Display trade table
        display_trade_table(df)
        
        # Export option
        if st.button("📊 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trades_{date_option}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()