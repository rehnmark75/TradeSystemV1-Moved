"""
Simple Strategy Dashboard - Direct Implementation
This replicates the exact queries that worked in the diagnostic
Save as: streamlit/pages/simple_strategy.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Strategy Performance",
    page_icon="🎯",
    layout="wide"
)

def get_database_connection():
    """Get database connection"""
    try:
        if hasattr(st, 'secrets'):
            try:
                conn_string = st.secrets.database.trading_connection_string
            except (AttributeError, KeyError):
                try:
                    conn_string = st.secrets.database.config_connection_string
                except (AttributeError, KeyError):
                    conn_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
        else:
            conn_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
        
        return psycopg2.connect(conn_string)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def main():
    st.title("🎯 Strategy Performance Dashboard")
    st.markdown("Direct implementation using the exact queries that work in diagnostics")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        days_back = st.selectbox(
            "📅 Analysis Period", 
            [7, 30, 90], 
            index=1,  # Default to 30 days
            format_func=lambda x: f"{x} days"
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    with col3:
        show_debug = st.checkbox("🔍 Show Debug", value=False)
    
    conn = get_database_connection()
    if not conn:
        st.error("❌ Cannot connect to database")
        return
    
    try:
        # 1. Strategy Performance Summary (exact query from diagnostic)
        st.subheader("📊 Strategy Performance Summary")
        
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
        
        strategy_df = pd.read_sql_query(
            strategy_query, 
            conn, 
            params=[datetime.now() - timedelta(days=days_back)]
        )
        
        if strategy_df.empty:
            st.warning(f"No strategy data found for the last {days_back} days")
            if show_debug:
                st.write("**Debug Query:**", strategy_query)
        else:
            # Calculate additional metrics
            strategy_df['win_rate'] = (strategy_df['wins'] / strategy_df['total_trades'] * 100).round(1)
            strategy_df['profit_factor'] = strategy_df.apply(
                lambda row: (row['wins'] * abs(row['avg_pnl'])) / (row['losses'] * abs(row['avg_pnl'])) 
                if row['losses'] > 0 and row['avg_pnl'] < 0 else float('inf'), axis=1
            )
            
            # Display summary table
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
            
            # 2. Strategy Performance Charts
            st.subheader("📈 Strategy Performance Visualization")
            
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
                fig_pnl.update_layout(height=400)
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
                fig_win.update_layout(height=400)
                st.plotly_chart(fig_win, use_container_width=True)
            
            # 3. Strategy Details
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
                    st.markdown("**Trade Performance:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"🏆 Best Trade: +{strategy['best_trade']:.2f}")
                    with col2:
                        st.error(f"📉 Worst Trade: {strategy['worst_trade']:.2f}")
        
        # 4. Signal to Trade Conversion Analysis
        st.subheader("📡 Signal to Trade Conversion")
        
        conversion_query = """
        SELECT 
            a.strategy,
            COUNT(a.*) as total_signals,
            COUNT(t.*) as executed_trades,
            COALESCE(AVG(a.confidence_score), 0) as avg_confidence,
            COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as profitable_trades,
            COUNT(CASE WHEN a.signal_type = 'BULL' THEN 1 END) as bull_signals,
            COUNT(CASE WHEN a.signal_type = 'BEAR' THEN 1 END) as bear_signals
        FROM alert_history a
        LEFT JOIN trade_log t ON a.id = t.alert_id
        WHERE a.alert_timestamp >= %s
        GROUP BY a.strategy
        ORDER BY total_signals DESC
        """
        
        conversion_df = pd.read_sql_query(
            conversion_query,
            conn,
            params=[datetime.now() - timedelta(days=days_back)]
        )
        
        if not conversion_df.empty:
            # Calculate conversion rates
            conversion_df['conversion_rate'] = (conversion_df['executed_trades'] / conversion_df['total_signals'] * 100).round(1)
            conversion_df['success_rate'] = (conversion_df['profitable_trades'] / conversion_df['executed_trades'] * 100).round(1)
            
            st.dataframe(
                conversion_df[['strategy', 'total_signals', 'executed_trades', 'conversion_rate', 'success_rate', 'avg_confidence', 'bull_signals', 'bear_signals']],
                column_config={
                    "strategy": "Strategy",
                    "total_signals": "Total Signals",
                    "executed_trades": "Executed Trades",
                    "conversion_rate": st.column_config.NumberColumn("Conversion Rate", format="%.1f%%"),
                    "success_rate": st.column_config.NumberColumn("Success Rate", format="%.1f%%"),
                    "avg_confidence": st.column_config.NumberColumn("Avg Confidence", format="%.1%"),
                    "bull_signals": "BULL Signals",
                    "bear_signals": "BEAR Signals"
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Conversion rate chart
            fig_conversion = px.bar(
                conversion_df,
                x='strategy',
                y='conversion_rate',
                title="Signal to Trade Conversion Rate",
                color='conversion_rate',
                color_continuous_scale='Blues'
            )
            fig_conversion.update_layout(height=300)
            st.plotly_chart(fig_conversion, use_container_width=True)
        
        # 5. Recent Trades with Strategy Info
        st.subheader("📋 Recent Trades with Strategy Context")
        
        recent_trades_query = """
        SELECT 
            t.timestamp,
            t.symbol,
            t.direction,
            t.entry_price,
            t.profit_loss,
            t.status,
            a.strategy,
            a.signal_type,
            a.confidence_score
        FROM trade_log t
        LEFT JOIN alert_history a ON t.alert_id = a.id
        WHERE t.timestamp >= %s
        ORDER BY t.timestamp DESC 
        LIMIT 20
        """
        
        recent_df = pd.read_sql_query(
            recent_trades_query,
            conn,
            params=[datetime.now() - timedelta(days=7)]  # Last 7 days for recent trades
        )
        
        if not recent_df.empty:
            # Format display columns
            recent_df['result'] = recent_df['profit_loss'].apply(
                lambda x: '🟢 WIN' if x > 0 else '🔴 LOSS' if x < 0 else '⏳ PENDING'
            )
            recent_df['pnl_formatted'] = recent_df['profit_loss'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "Pending"
            )
            recent_df['confidence_formatted'] = recent_df['confidence_score'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
            
            st.dataframe(
                recent_df[['timestamp', 'symbol', 'strategy', 'signal_type', 'direction', 'confidence_formatted', 'pnl_formatted', 'result']].head(15),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="MM/DD HH:mm"),
                    "symbol": "Pair",
                    "strategy": "Strategy",
                    "signal_type": "Signal",
                    "direction": "Direction",
                    "confidence_formatted": "Confidence",
                    "pnl_formatted": "P&L",
                    "result": "Result"
                },
                use_container_width=True,
                hide_index=True
            )
        
        # Debug information
        if show_debug:
            st.subheader("🔍 Debug Information")
            st.write("**Strategy Query:**")
            st.code(strategy_query, language='sql')
            st.write("**Conversion Query:**")
            st.code(conversion_query, language='sql')
            st.write("**Recent Trades Query:**")
            st.code(recent_trades_query, language='sql')
    
    except Exception as e:
        st.error(f"Error executing queries: {e}")
        if show_debug:
            import traceback
            st.code(traceback.format_exc())
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()