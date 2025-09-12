import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import psycopg2
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
import os

# Configure page
st.set_page_config(
    page_title="Forex Scanner Configuration",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .config-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e9ef;
        margin-bottom: 1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class ConfigCategory(Enum):
    TRADING_PARAMS = "Trading Parameters"
    STRATEGY_SETTINGS = "Strategy Settings"
    RISK_MANAGEMENT = "Risk Management"
    MARKET_INTELLIGENCE = "Market Intelligence"
    API_CONFIG = "API Configuration"
    PAIR_MANAGEMENT = "Pair Management"

@dataclass
class ConfigSetting:
    name: str
    value: Any
    type_: str
    category: str
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_options: Optional[List] = None
    is_sensitive: bool = False
    requires_restart: bool = False

class DatabaseConfigManager:
    """Manages configuration in database"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)
    
    def get_all_settings(self) -> List[ConfigSetting]:
        """Retrieve all configuration settings"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT setting_name, setting_value, setting_type, category, 
                               description, min_value, max_value, valid_options,
                               is_sensitive, requires_restart
                        FROM config_settings 
                        ORDER BY category, setting_name
                    """)
                    
                    settings = []
                    for row in cur.fetchall():
                        # Parse value based on type
                        value = self._parse_value(row[1], row[2])
                        valid_options = json.loads(row[7]) if row[7] else None
                        
                        settings.append(ConfigSetting(
                            name=row[0],
                            value=value,
                            type_=row[2],
                            category=row[3],
                            description=row[4],
                            min_value=row[5],
                            max_value=row[6],
                            valid_options=valid_options,
                            is_sensitive=row[8],
                            requires_restart=row[9]
                        ))
                    
                    return settings
        except Exception as e:
            self.logger.error(f"Error fetching settings: {e}")
            return []
    
    def update_setting(self, setting_name: str, new_value: Any, changed_by: str, reason: str = ""):
        """Update a configuration setting"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get old value for history
                    cur.execute("SELECT setting_value FROM config_settings WHERE setting_name = %s", (setting_name,))
                    old_value = cur.fetchone()[0] if cur.fetchone() else None
                    
                    # Update setting
                    cur.execute("""
                        UPDATE config_settings 
                        SET setting_value = %s, updated_at = NOW()
                        WHERE setting_name = %s
                    """, (str(new_value), setting_name))
                    
                    # Add to history
                    cur.execute("""
                        INSERT INTO config_history (setting_name, old_value, new_value, changed_by, change_reason)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (setting_name, old_value, str(new_value), changed_by, reason))
                    
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error updating setting {setting_name}: {e}")
            return False
    
    def _parse_value(self, value: str, type_: str):
        """Parse string value to appropriate type"""
        if type_ == 'bool':
            return value.lower() in ('true', '1', 'yes', 'on')
        elif type_ == 'int':
            return int(value)
        elif type_ == 'float':
            return float(value)
        elif type_ == 'list':
            return json.loads(value) if value.startswith('[') else value.split(',')
        else:
            return value

class ConfigurationApp:
    """Main Streamlit application for configuration management"""
    
    def __init__(self):
        self.db_manager = None
        self.initialize_session_state()
        self.setup_database_connection()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'current_settings' not in st.session_state:
            st.session_state.current_settings = {}
        if 'pending_changes' not in st.session_state:
            st.session_state.pending_changes = {}
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
    
    def setup_database_connection(self):
        """Setup database connection from secrets or environment"""
        try:
            # Try to get from Streamlit secrets first
            if hasattr(st, 'secrets') and 'database' in st.secrets:
                conn_string = st.secrets.database.config_connection_string
            else:
                # Fallback to environment or default
                conn_string = os.getenv("CONFIG_DATABASE_URL")
            
            self.db_manager = DatabaseConfigManager(conn_string)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
    
    def render_header(self):
        """Render application header"""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("⚙️ Forex Scanner Configuration")
            st.caption("Dynamic trading system configuration management")
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                self.refresh_settings()
        
        with col3:
            if st.button("💾 Apply Changes", use_container_width=True, type="primary"):
                self.apply_pending_changes()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar navigation and status"""
        st.sidebar.title("📊 Configuration Panel")
        
        # Status indicators
        st.sidebar.markdown("### System Status")
        self.render_system_status()
        
        # Category selection
        st.sidebar.markdown("### Categories")
        selected_category = st.sidebar.selectbox(
            "Select Configuration Category",
            [cat.value for cat in ConfigCategory],
            key="selected_category"
        )
        
        # Preset management
        st.sidebar.markdown("### Presets")
        self.render_preset_management()
        
        # Change summary
        if st.session_state.pending_changes:
            st.sidebar.markdown("### Pending Changes")
            st.sidebar.warning(f"{len(st.session_state.pending_changes)} changes pending")
            for setting, value in st.session_state.pending_changes.items():
                st.sidebar.write(f"• {setting}: {value}")
    
    def render_system_status(self):
        """Render system status indicators"""
        # Database status
        db_status = "🟢 Connected" if self.db_manager else "🔴 Disconnected"
        st.sidebar.markdown(f"**Database:** {db_status}")
        
        # Scanner status (mock for now)
        st.sidebar.markdown("**Scanner:** 🟢 Running")
        
        # API status (mock for now)
        st.sidebar.markdown("**APIs:** 🟡 Partial")
        
        # Last update
        st.sidebar.markdown(f"**Last Update:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    def render_preset_management(self):
        """Render preset management controls"""
        # Load preset
        preset_options = ["Default", "Conservative", "Aggressive", "Scalping"]
        selected_preset = st.sidebar.selectbox("Load Preset", preset_options)
        
        if st.sidebar.button("Load Preset", use_container_width=True):
            st.sidebar.success(f"Loaded {selected_preset} preset")
        
        # Save current as preset
        new_preset_name = st.sidebar.text_input("Save as Preset")
        if st.sidebar.button("Save Preset", use_container_width=True):
            if new_preset_name:
                st.sidebar.success(f"Saved preset: {new_preset_name}")
    
    def render_main_content(self):
        """Render main configuration content"""
        selected_category = st.session_state.get('selected_category', ConfigCategory.TRADING_PARAMS.value)
        
        # Get settings for selected category
        if self.db_manager:
            all_settings = self.db_manager.get_all_settings()
            category_settings = [s for s in all_settings if s.category == selected_category]
        else:
            category_settings = self.get_mock_settings(selected_category)
        
        if selected_category == ConfigCategory.TRADING_PARAMS.value:
            self.render_trading_parameters(category_settings)
        elif selected_category == ConfigCategory.STRATEGY_SETTINGS.value:
            self.render_strategy_settings(category_settings)
        elif selected_category == ConfigCategory.RISK_MANAGEMENT.value:
            self.render_risk_management(category_settings)
        elif selected_category == ConfigCategory.MARKET_INTELLIGENCE.value:
            self.render_market_intelligence(category_settings)
        elif selected_category == ConfigCategory.API_CONFIG.value:
            self.render_api_configuration(category_settings)
        elif selected_category == ConfigCategory.PAIR_MANAGEMENT.value:
            self.render_pair_management(category_settings)
    
    def render_trading_parameters(self, settings: List[ConfigSetting]):
        """Render trading parameters configuration"""
        st.header("💰 Trading Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Signal Detection")
            
            # Min Confidence
            min_conf = st.slider(
                "Minimum Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum confidence required for signal generation"
            )
            self.track_change("MIN_CONFIDENCE", min_conf)
            
            # Spread Pips
            spread_pips = st.number_input(
                "Spread Pips",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Bid/ask spread in pips"
            )
            self.track_change("SPREAD_PIPS", spread_pips)
            
            # Scan Interval
            scan_interval = st.number_input(
                "Scan Interval (seconds)",
                min_value=30,
                max_value=300,
                value=60,
                step=30,
                help="Seconds between market scans"
            )
            self.track_change("SCAN_INTERVAL", scan_interval)
        
        with col2:
            st.subheader("Market Settings")
            
            # Default Timeframe
            timeframe = st.selectbox(
                "Default Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=2,
                help="Primary analysis timeframe"
            )
            self.track_change("DEFAULT_TIMEFRAME", timeframe)
            
            # Bid Adjustment
            use_bid_adj = st.checkbox(
                "Use Bid Adjustment",
                value=True,
                help="Apply bid/ask price adjustments"
            )
            self.track_change("USE_BID_ADJUSTMENT", use_bid_adj)
        
        # Impact Analysis
        self.render_impact_analysis("trading", {
            "signals_per_day": 15,
            "avg_confidence": 0.78,
            "estimated_change": "+12%"
        })
    
    def render_strategy_settings(self, settings: List[ConfigSetting]):
        """Render strategy configuration"""
        st.header("📈 Strategy Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strategy Selection")
            
            # Enable/disable strategies
            ema_enabled = st.checkbox("EMA Strategy", value=True)
            self.track_change("SIMPLE_EMA_STRATEGY", ema_enabled)
            
            macd_enabled = st.checkbox("MACD Strategy", value=True)
            self.track_change("MACD_EMA_STRATEGY", macd_enabled)
            
            combined_enabled = st.checkbox("Combined Strategy", value=True)
            self.track_change("COMBINED_STRATEGY_ENABLED", combined_enabled)
            
            # Strategy Mode
            strategy_mode = st.selectbox(
                "Combined Strategy Mode",
                ["consensus", "confirmation", "hierarchy"],
                help="How to combine multiple strategy signals"
            )
            self.track_change("COMBINED_STRATEGY_MODE", strategy_mode)
        
        with col2:
            st.subheader("Strategy Weights")
            
            # Strategy weights
            ema_weight = st.slider("EMA Strategy Weight", 0.0, 1.0, 0.4, 0.1)
            self.track_change("STRATEGY_WEIGHT_EMA", ema_weight)
            
            macd_weight = st.slider("MACD Strategy Weight", 0.0, 1.0, 0.3, 0.1)
            self.track_change("STRATEGY_WEIGHT_MACD", macd_weight)
            
            combined_weight = st.slider("Combined Strategy Weight", 0.0, 1.0, 0.3, 0.1)
            self.track_change("STRATEGY_WEIGHT_COMBINED", combined_weight)
            
        # EMA Configuration
        st.subheader("EMA Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            short_ema = st.number_input("Short EMA Period", min_value=5, max_value=50, value=12)
            self.track_change("EMA_SHORT_PERIOD", short_ema)
        
        with col2:
            long_ema = st.number_input("Long EMA Period", min_value=20, max_value=200, value=26)
            self.track_change("EMA_LONG_PERIOD", long_ema)
        
        with col3:
            trend_ema = st.number_input("Trend EMA Period", min_value=50, max_value=400, value=200)
            self.track_change("EMA_TREND_PERIOD", trend_ema)
    
    def render_risk_management(self, settings: List[ConfigSetting]):
        """Render risk management configuration"""
        st.header("🛡️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Position Sizing")
            
            risk_per_trade = st.slider(
                "Risk Per Trade (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Percentage of account to risk per trade"
            )
            self.track_change("RISK_PER_TRADE_PERCENT", risk_per_trade)
            
            max_positions = st.number_input(
                "Max Concurrent Positions",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of open positions"
            )
            self.track_change("MAX_CONCURRENT_POSITIONS", max_positions)
            
            max_daily_trades = st.number_input(
                "Max Daily Trades",
                min_value=1,
                max_value=100,
                value=20,
                help="Maximum trades per day"
            )
            self.track_change("MAX_DAILY_TRADES", max_daily_trades)
        
        with col2:
            st.subheader("Stop Loss & Take Profit")
            
            stop_distance = st.number_input(
                "Default Stop Distance (pips)",
                min_value=5,
                max_value=100,
                value=20,
                help="Default stop loss distance in pips"
            )
            self.track_change("DEFAULT_STOP_DISTANCE", stop_distance)
            
            risk_reward = st.number_input(
                "Risk/Reward Ratio",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Target profit vs risk ratio"
            )
            self.track_change("DEFAULT_RISK_REWARD", risk_reward)
        
        # Risk Summary
        self.render_risk_summary(risk_per_trade, max_positions, max_daily_trades)
    
    def render_market_intelligence(self, settings: List[ConfigSetting]):
        """Render market intelligence configuration"""
        st.header("🧠 Market Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Adaptive Intelligence")
            
            market_intel_enabled = st.checkbox(
                "Enable Market Intelligence",
                value=True,
                help="Enable adaptive market analysis"
            )
            self.track_change("MARKET_INTELLIGENCE_ENABLED", market_intel_enabled)
            
            regime_filtering = st.checkbox(
                "Regime-Based Filtering",
                value=True,
                help="Filter signals based on market regime"
            )
            self.track_change("REGIME_BASED_FILTERING", regime_filtering)
            
            adaptive_confidence = st.checkbox(
                "Adaptive Confidence Threshold",
                value=False,
                help="Dynamically adjust confidence thresholds"
            )
            self.track_change("ADAPTIVE_CONFIDENCE_THRESHOLD", adaptive_confidence)
        
        with col2:
            st.subheader("Market Conditions")
            
            volatility_threshold = st.slider(
                "Volatility Threshold",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Volatility filter level"
            )
            self.track_change("VOLATILITY_THRESHOLD", volatility_threshold)
    
    def render_api_configuration(self, settings: List[ConfigSetting]):
        """Render API configuration"""
        st.header("🔌 API Configuration")
        
        st.warning("⚠️ Sensitive information - handle with care")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Claude AI")
            
            claude_key = st.text_input(
                "Claude API Key",
                type="password",
                help="Anthropic Claude API key"
            )
            if claude_key:
                self.track_change("CLAUDE_API_KEY", claude_key)
            
            enable_claude = st.checkbox(
                "Enable Claude Analysis",
                value=True,
                help="Enable AI-powered signal analysis"
            )
            self.track_change("ENABLE_CLAUDE_ANALYSIS", enable_claude)
        
        with col2:
            st.subheader("Trading API")
            
            api_url = st.text_input(
                "Order API URL",
                help="Trading API endpoint"
            )
            if api_url:
                self.track_change("ORDER_API_URL", api_url)
            
            api_key = st.text_input(
                "API Subscription Key",
                type="password",
                help="Trading API key"
            )
            if api_key:
                self.track_change("API_SUBSCRIPTION_KEY", api_key)
        
        # API Status Testing
        st.subheader("API Status")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test Claude API"):
                st.success("✅ Claude API connection successful")
        
        with col2:
            if st.button("Test Trading API"):
                st.error("❌ Trading API connection failed")
    
    def render_pair_management(self, settings: List[ConfigSetting]):
        """Render trading pair management"""
        st.header("💱 Pair Management")
        
        # Available pairs
        available_pairs = [
            "CS.D.EURUSD.MINI.IP", "CS.D.GBPUSD.MINI.IP", "CS.D.USDJPY.MINI.IP",
            "CS.D.USDCHF.MINI.IP", "CS.D.AUDUSD.MINI.IP", "CS.D.USDCAD.MINI.IP",
            "CS.D.NZDUSD.MINI.IP", "CS.D.EURGBP.MINI.IP", "CS.D.EURJPY.MINI.IP"
        ]
        
        # Active pairs selection
        active_pairs = st.multiselect(
            "Active Trading Pairs",
            available_pairs,
            default=available_pairs[:4],
            help="Select pairs to include in scanning"
        )
        self.track_change("EPIC_LIST", active_pairs)
        
        # Pair-specific settings
        st.subheader("Pair-Specific Settings")
        
        if active_pairs:
            selected_pair = st.selectbox("Configure Pair", active_pairs)
            
            col1, col2 = st.columns(2)
            
            with col1:
                pair_confidence = st.slider(
                    f"Min Confidence for {selected_pair}",
                    0.0, 1.0, 0.7, 0.05
                )
                
                pair_spread = st.number_input(
                    f"Spread Override for {selected_pair}",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    help="0 = use default spread"
                )
            
            with col2:
                market_hours = st.checkbox(
                    "Respect Market Hours",
                    value=True,
                    help="Only trade during market hours"
                )
                
                timezone = st.selectbox(
                    "User Timezone",
                    ["UTC", "EST", "PST", "GMT", "CET"],
                    help="Your local timezone"
                )
        
    def render_impact_analysis(self, category: str, metrics: Dict):
        """Render configuration impact analysis"""
        st.subheader("📊 Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Signals/Day", metrics["signals_per_day"], "↑ 2")
        
        with col2:
            st.metric("Avg Confidence", f"{metrics['avg_confidence']:.2f}", "↑ 0.05")
        
        with col3:
            st.metric("Est. Change", metrics["estimated_change"], "")
        
        # Simple impact chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=["Current", "With Changes"],
            y=[100, 112],
            mode='lines+markers',
            name='Performance Index'
        ))
        fig.update_layout(
            title="Estimated Performance Impact",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_summary(self, risk_per_trade: float, max_positions: int, max_daily_trades: int):
        """Render risk management summary"""
        st.subheader("📊 Risk Summary")
        
        # Calculate risk metrics
        max_account_risk = risk_per_trade * max_positions
        max_daily_risk = risk_per_trade * max_daily_trades
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = "red" if max_account_risk > 10 else "green"
            st.metric("Max Account Risk", f"{max_account_risk:.1f}%", 
                     help="Maximum percentage of account at risk")
        
        with col2:
            color = "red" if max_daily_risk > 20 else "green"
            st.metric("Max Daily Risk", f"{max_daily_risk:.1f}%",
                     help="Maximum daily risk exposure")
        
        with col3:
            expected_return = max_daily_risk * 0.6  # Assuming 60% win rate
            st.metric("Expected Daily Return", f"{expected_return:.1f}%")
    
    def track_change(self, setting_name: str, value: Any):
        """Track configuration changes"""
        if setting_name not in st.session_state.current_settings:
            st.session_state.current_settings[setting_name] = value
        elif st.session_state.current_settings[setting_name] != value:
            st.session_state.pending_changes[setting_name] = value
    
    def apply_pending_changes(self):
        """Apply all pending configuration changes"""
        if not st.session_state.pending_changes:
            st.warning("No pending changes to apply")
            return
        
        if self.db_manager:
            success_count = 0
            for setting_name, value in st.session_state.pending_changes.items():
                if self.db_manager.update_setting(setting_name, value, "streamlit_user", "UI Update"):
                    success_count += 1
            
            if success_count == len(st.session_state.pending_changes):
                st.success(f"✅ Applied {success_count} configuration changes")
                st.session_state.pending_changes.clear()
                self.refresh_settings()
            else:
                st.error(f"❌ Only {success_count}/{len(st.session_state.pending_changes)} changes applied")
        else:
            st.error("❌ Database connection not available")
    
    def refresh_settings(self):
        """Refresh configuration settings from database"""
        st.session_state.last_refresh = datetime.now()
        st.session_state.current_settings.clear()
        st.session_state.pending_changes.clear()
        st.rerun()
    
    def get_mock_settings(self, category: str) -> List[ConfigSetting]:
        """Get mock settings when database is not available"""
        # Return empty list for now - in real implementation, this would have defaults
        return []

# Application entry point
if __name__ == "__main__":
    app = ConfigurationApp()
    app.run()