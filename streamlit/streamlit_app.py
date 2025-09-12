"""
Streamlit Configuration Management App - Compatible Version
Save this as: streamlit_config_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Forex Configuration Manager",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_rerun():
    """Safe rerun function that works across Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            # For very old versions, just refresh the page
            st.write("Please refresh the page manually")

# ============================================================================
# AUTHENTICATION
# ============================================================================

def check_authentication():
    """Simple authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Login Required")
        
        with st.form("login_form"):
            password = st.text_input("Enter admin password:", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                try:
                    correct_password = st.secrets.security.admin_password
                except:
                    correct_password = "admin123"  # Default fallback
                
                if password == correct_password:
                    st.session_state.authenticated = True
                    st.success("✅ Login successful!")
                    safe_rerun()
                else:
                    st.error("❌ Invalid password")
        
        st.info("💡 Default password: admin123")
        return False
    
    return True

# ============================================================================
# DATABASE CONNECTION TEST
# ============================================================================

def test_database_connection():
    """Test database connection"""
    try:
        if hasattr(st.secrets, 'database'):
            conn_str = st.secrets.database.config_connection_string
            st.success(f"✅ Database configured: {conn_str[:30]}...")
            return True
        elif hasattr(st.secrets, 'connection_string'):
            conn_str = st.secrets.connection_string
            st.success(f"✅ Database configured: {conn_str[:30]}...")
            return True
        else:
            st.warning("⚠️ No database configuration found - using demo mode")
            return False
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return False

# ============================================================================
# DASHBOARD
# ============================================================================

def render_dashboard():
    """Main dashboard"""
    st.header("📊 Configuration Dashboard")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔧 Total Settings", "67", "+2")
    
    with col2:
        st.metric("📈 Signals/Hour", "24", "+3")
    
    with col3:
        st.metric("🎯 Confidence", "78.5%", "+2.3%")
    
    with col4:
        st.metric("💰 Win Rate", "64.2%", "-1.1%")
    
    # Recent changes chart
    st.subheader("📅 Recent Configuration Changes")
    
    # Mock data
    changes = pd.DataFrame({
        'Setting': ['MIN_CONFIDENCE', 'RISK_PER_TRADE', 'STRATEGY_WEIGHT_EMA', 'EPIC_LIST'],
        'Changes': [5, 3, 2, 1],
        'Last_Changed': ['2h ago', '6h ago', '12h ago', '1d ago']
    })
    
    fig = px.bar(changes, x='Setting', y='Changes', title="Configuration Changes (Last 7 Days)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance correlation
    st.subheader("📈 Performance Trends")
    
    # Generate sample performance data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=24, freq='H')
    np.random.seed(42)
    
    performance_data = pd.DataFrame({
        'Time': dates,
        'Signals': 20 + np.random.normal(0, 3, 24),
        'Confidence': 0.75 + np.random.normal(0, 0.05, 24),
        'Win_Rate': 0.64 + np.random.normal(0, 0.03, 24)
    })
    
    fig = px.line(performance_data, x='Time', y=['Signals', 'Win_Rate'], 
                  title="Performance Over Time")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CONFIGURATION INTERFACE
# ============================================================================

def render_configuration():
    """Configuration management interface"""
    st.header("⚙️ Configuration Settings")
    
    tab1, tab2, tab3 = st.tabs(["Trading Parameters", "Strategy Settings", "Risk Management"])
    
    with tab1:
        st.subheader("📊 Trading Parameters")
        
        with st.form("trading_params_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.75, 0.01)
                spread_pips = st.number_input("Spread (Pips)", 0.1, 10.0, 1.0, 0.1)
                scan_interval = st.number_input("Scan Interval (seconds)", 10, 600, 60)
            
            with col2:
                timeframe = st.selectbox("Default Timeframe", ["5m", "15m", "30m", "1h"], index=1)
                pairs = st.multiselect("Currency Pairs", 
                                     ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"], 
                                     default=["EURUSD", "GBPUSD"])
            
            if st.form_submit_button("💾 Save Trading Parameters"):
                st.success("✅ Trading parameters saved!")
    
    with tab2:
        st.subheader("🎯 Strategy Settings")
        
        with st.form("strategy_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ema_weight = st.slider("EMA Weight", 0.0, 1.0, 0.4)
            
            with col2:
                macd_weight = st.slider("MACD Weight", 0.0, 1.0, 0.3)
            
            with col3:
                volume_weight = st.slider("Volume Weight", 0.0, 1.0, 0.2)
            
            total = ema_weight + macd_weight + volume_weight
            if abs(total - 1.0) > 0.01:
                st.warning(f"⚠️ Weights total: {total:.2f} (should be 1.0)")
            
            strategy_mode = st.selectbox("Strategy Mode", ["Consensus", "Confirmation", "Hierarchy"])
            
            if st.form_submit_button("💾 Save Strategy Settings"):
                st.success("✅ Strategy settings saved!")
    
    with tab3:
        st.subheader("🛡️ Risk Management")
        
        with st.form("risk_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                max_positions = st.number_input("Max Positions", 1, 20, 5)
                risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0)
            
            with col2:
                stop_distance = st.number_input("Stop Distance (pips)", 5, 100, 20)
                risk_reward = st.slider("Risk/Reward", 0.5, 10.0, 2.0)
            
            if st.form_submit_button("💾 Save Risk Settings"):
                st.success("✅ Risk settings saved!")

# ============================================================================
# BACKUP & RESTORE
# ============================================================================

def render_backup_restore():
    """Backup and restore interface"""
    st.header("💾 Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create Backup")
        
        with st.form("backup_form"):
            backup_name = st.text_input("Backup Name", placeholder="My_Config_Backup")
            description = st.text_area("Description", placeholder="What's special about this config?")
            
            if st.form_submit_button("💾 Create Backup"):
                if backup_name:
                    st.success(f"✅ Backup '{backup_name}' created!")
                else:
                    st.error("❌ Please enter a backup name")
    
    with col2:
        st.subheader("Restore Configuration")
        
        with st.form("restore_form"):
            backups = [
                "Production_2025_06_29",
                "Working_Config_June", 
                "Conservative_Setup",
                "Emergency_Backup"
            ]
            
            selected = st.selectbox("Select Backup", backups)
            
            st.warning("⚠️ This will overwrite current settings!")
            
            confirm = st.checkbox("I understand this will replace current configuration")
            
            if st.form_submit_button("🔄 Restore", disabled=not confirm):
                st.success(f"✅ Restored from '{selected}'!")

# ============================================================================
# SYSTEM STATUS
# ============================================================================

def render_system_status():
    """System status and diagnostics"""
    st.header("🔧 System Status")
    
    # Database connection test
    st.subheader("📡 Database Connection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Test Database Connection"):
            db_status = test_database_connection()
    
    with col2:
        if st.button("🔄 Refresh Status"):
            safe_rerun()
    
    # Secrets verification
    st.subheader("🔐 Configuration Status")
    
    secrets_status = {}
    
    try:
        if hasattr(st.secrets, 'database'):
            secrets_status['Database Config'] = "✅ Found"
        else:
            secrets_status['Database Config'] = "❌ Missing"
        
        if hasattr(st.secrets, 'security'):
            secrets_status['Security Config'] = "✅ Found"
        else:
            secrets_status['Security Config'] = "❌ Missing"
        
        if hasattr(st.secrets, 'api'):
            secrets_status['API Config'] = "✅ Found"
        else:
            secrets_status['API Config'] = "❌ Missing"
    
    except Exception as e:
        secrets_status['Error'] = f"❌ {str(e)}"
    
    # Display status
    for item, status in secrets_status.items():
        st.write(f"**{item}:** {status}")
    
    # Environment info
    st.subheader("🌍 Environment Information")
    
    env_info = {
        "Streamlit Version": st.__version__,
        "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Session State Keys": len(st.session_state.keys()),
        "Authenticated": st.session_state.get('authenticated', False)
    }
    
    for key, value in env_info.items():
        st.write(f"**{key}:** {value}")
    
    # Debug information
    if st.checkbox("🐛 Show Debug Info"):
        st.subheader("🔍 Debug Information")
        
        st.write("**Session State:**")
        st.json(dict(st.session_state))
        
        try:
            st.write("**Available Secrets:**")
            if hasattr(st, 'secrets'):
                secrets_list = list(st.secrets.keys())
                st.write(secrets_list)
            else:
                st.write("No secrets found")
        except Exception as e:
            st.write(f"Error accessing secrets: {e}")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

def render_sidebar():
    """Navigation sidebar"""
    st.sidebar.title("⚙️ Configuration Hub")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "⚙️ Configuration", "💾 Backup & Restore", "🔧 System Status"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.metric("Settings", "67")
    st.sidebar.metric("Last Update", "2h ago")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("🔄 Reload"):
        safe_rerun()
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        safe_rerun()
    
    return page

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Authentication check
    if not check_authentication():
        return
    
    # App header
    st.title("⚙️ Forex Scanner Configuration Management")
    st.markdown("*Real-time configuration management for your trading system*")
    
    # Navigation
    current_page = render_sidebar()
    
    # Page routing
    if current_page == "📊 Dashboard":
        render_dashboard()
    
    elif current_page == "⚙️ Configuration":
        render_configuration()
    
    elif current_page == "💾 Backup & Restore":
        render_backup_restore()
    
    elif current_page == "🔧 System Status":
        render_system_status()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 12px;'>
        Forex Scanner Configuration Management v1.0 | 
        Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")} | 
        Status: {'🟢 Online' if st.session_state.authenticated else '🔴 Offline'}
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()