#!/usr/bin/env python3
"""
TradingView Strategy Importer - Streamlit Interface

Visual interface for searching, analyzing, and importing TradingView strategies
into TradeSystemV1. Provides an intuitive way to browse strategies and
configure imports with real-time previews.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "mcp"))
sys.path.insert(0, str(project_root / "strategy_bridge"))
sys.path.insert(0, str(project_root / "worker" / "app" / "forex_scanner"))

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error(f"Required packages not installed: {e}")
    st.info("Please install: streamlit, pandas, plotly")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="TradingView Strategy Importer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .strategy-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fafafa;
    }
    .success-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 12px;
    }
    .warning-badge {
        background-color: #ffc107;
        color: black;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 12px;
    }
    .error-badge {
        background-color: #dc3545;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'import_history' not in st.session_state:
        st.session_state.import_history = []
    if 'system_status' not in st.session_state:
        st.session_state.system_status = None

def check_system_status():
    """Check TradingView integration system status"""
    try:
        from configdata.strategies.tradingview_integration import validate_tradingview_integration
        status = validate_tradingview_integration()
        st.session_state.system_status = status
        return status
    except Exception as e:
        st.session_state.system_status = {"valid": False, "error": str(e)}
        return st.session_state.system_status

def search_strategies(query: str, limit: int = 20) -> List[Dict]:
    """Search TradingView strategies"""
    try:
        from client.mcp_client import search_scripts
        return search_scripts(query, limit)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def get_strategy_details(slug: str) -> Optional[Dict]:
    """Get detailed strategy information"""
    try:
        from client.mcp_client import get_script_by_slug
        return get_script_by_slug(slug)
    except Exception as e:
        st.error(f"Failed to get strategy details: {e}")
        return None

def analyze_strategy(slug: str) -> Optional[Dict]:
    """Analyze strategy patterns and indicators"""
    try:
        from client.mcp_client import TVScriptsClient
        with TVScriptsClient() as client:
            return client.analyze_script(slug=slug)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def import_strategy(slug: str, target_config: str, preset_name: str) -> Dict:
    """Import strategy into TradeSystemV1"""
    try:
        from configdata.strategies.tradingview_integration import import_tradingview_strategy
        return import_tradingview_strategy(slug, target_config, preset_name)
    except Exception as e:
        return {"success": False, "error": str(e)}

def render_sidebar():
    """Render sidebar with navigation and status"""
    st.sidebar.title("📈 TradingView Importer")
    
    # System status
    st.sidebar.subheader("🔧 System Status")
    status = st.session_state.system_status or check_system_status()
    
    if status.get("valid"):
        st.sidebar.success("✅ System Ready")
    else:
        st.sidebar.error("❌ System Issues")
        st.sidebar.caption(f"Error: {status.get('error', 'Unknown')}")
    
    # Status details
    with st.sidebar.expander("📊 Status Details"):
        st.write(f"**MCP Client:** {'✅' if status.get('mcp_client_available') else '❌'}")
        st.write(f"**Strategy Bridge:** {'✅' if status.get('strategy_bridge_available') else '❌'}")
        st.write(f"**Integrator:** {'✅' if status.get('integrator_ready') else '❌'}")
    
    # Navigation
    st.sidebar.subheader("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🔍 Search & Browse", "📊 Strategy Analysis", "⚙️ Import Center", "📋 Import History"]
    )
    
    return page

def render_search_page():
    """Render strategy search and browse page"""
    st.title("🔍 TradingView Strategy Search")
    st.markdown("Search and browse TradingView open-source strategies for import.")
    
    # Search interface
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., ema crossover, macd strategy, smart money concepts",
            help="Search for strategies by keywords, indicators, or concepts"
        )
    
    with col2:
        limit = st.selectbox("Results", [10, 20, 50], index=1)
    
    with col3:
        search_button = st.button("🔍 Search", type="primary")
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        col1, col2 = st.columns(2)
        with col1:
            script_type = st.selectbox(
                "Script Type",
                ["Any", "strategy", "indicator", "library"],
                help="Filter by TradingView script type"
            )
        with col2:
            min_likes = st.number_input(
                "Minimum Likes",
                min_value=0,
                value=0,
                help="Filter strategies by popularity"
            )
    
    # Search execution
    if search_button and query:
        with st.spinner("🔍 Searching TradingView strategies..."):
            results = search_strategies(query, limit)
            st.session_state.search_results = results
    
    # Display results
    if st.session_state.search_results:
        st.subheader(f"📋 Search Results ({len(st.session_state.search_results)})")
        
        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{result.get('title', 'Unknown Title')}**")
                    st.caption(f"by {result.get('author', 'Unknown')} • {result.get('tags', '')}")
                    if result.get('description'):
                        st.markdown(f"*{result['description'][:100]}...*")
                
                with col2:
                    if result.get('open_source'):
                        st.markdown('<span class="success-badge">Open Source</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="error-badge">Closed Source</span>', unsafe_allow_html=True)
                    
                    if result.get('likes_count'):
                        st.metric("👍 Likes", result['likes_count'])
                
                with col3:
                    if st.button(f"📊 Analyze", key=f"analyze_{i}"):
                        st.session_state.selected_strategy = result
                        st.experimental_rerun()
                    
                    if result.get('open_source') and st.button(f"⚙️ Import", key=f"import_{i}"):
                        st.session_state.selected_strategy = result
                        st.experimental_rerun()
                
                st.divider()

def render_analysis_page():
    """Render strategy analysis page"""
    st.title("📊 Strategy Analysis")
    
    if not st.session_state.selected_strategy:
        st.info("🔍 Please search and select a strategy to analyze.")
        return
    
    strategy = st.session_state.selected_strategy
    st.subheader(f"Analyzing: {strategy.get('title', 'Unknown')}")
    
    # Strategy overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Author", strategy.get('author', 'Unknown'))
    with col2:
        st.metric("👍 Likes", strategy.get('likes_count', 0))
    with col3:
        st.metric("📊 Uses", strategy.get('uses_count', 0))
    with col4:
        status = "✅ Open Source" if strategy.get('open_source') else "❌ Closed"
        st.metric("🔓 Status", status)
    
    # Analyze button
    if st.button("🔬 Run Analysis", type="primary"):
        with st.spinner("🔬 Analyzing Pine Script patterns..."):
            analysis = analyze_strategy(strategy['slug'])
            st.session_state.analysis_result = analysis
    
    # Display analysis results
    if st.session_state.analysis_result:
        analysis = st.session_state.analysis_result
        
        # Analysis overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            signals = analysis.get('signals', {})
            st.metric("🎯 Strategy Type", signals.get('strategy_type', 'unknown').title())
        
        with col2:
            st.metric("🧮 Complexity Score", f"{signals.get('complexity_score', 0):.2f}")
        
        with col3:
            st.metric("📝 Input Parameters", len(analysis.get('inputs', [])))
        
        # Indicators detected
        st.subheader("🔍 Detected Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if signals.get('ema_periods'):
                st.success(f"📈 EMA: {signals['ema_periods']}")
            
            if signals.get('macd'):
                macd = signals['macd']
                st.success(f"📊 MACD: {macd['fast']}/{macd['slow']}/{macd['signal']}")
            
            if signals.get('rsi_periods'):
                st.success(f"📉 RSI: {signals['rsi_periods']}")
            
            if signals.get('bollinger_bands'):
                bb = signals['bollinger_bands']
                st.success(f"📐 Bollinger Bands: {bb['length']}, {bb['multiplier']}")
        
        with col2:
            if signals.get('mentions_smc'):
                st.info("🧠 Smart Money Concepts")
            
            if signals.get('mentions_fvg'):
                st.info("⚡ Fair Value Gaps")
            
            if signals.get('has_cross_up') or signals.get('has_cross_down'):
                st.info("🔄 Crossover Signals")
            
            if signals.get('higher_tf'):
                st.info(f"⏰ Multi-Timeframe: {len(signals['higher_tf'])} TFs")
        
        # Input parameters
        if analysis.get('inputs'):
            st.subheader("⚙️ Input Parameters")
            inputs_df = pd.DataFrame(analysis['inputs'])
            st.dataframe(inputs_df, use_container_width=True)
        
        # Trading rules
        if analysis.get('rules'):
            st.subheader("📋 Trading Rules")
            for i, rule in enumerate(analysis['rules'], 1):
                st.markdown(f"{i}. **{rule.get('type', 'Unknown')}**: {rule.get('condition', 'No condition')}")

def render_import_page():
    """Render strategy import configuration page"""
    st.title("⚙️ Strategy Import Center")
    
    if not st.session_state.selected_strategy:
        st.info("🔍 Please search and select a strategy to import.")
        return
    
    strategy = st.session_state.selected_strategy
    
    if not strategy.get('open_source'):
        st.error("❌ This strategy is not open-source and cannot be imported.")
        return
    
    st.subheader(f"Importing: {strategy.get('title', 'Unknown')}")
    
    # Import configuration
    col1, col2 = st.columns(2)
    
    with col1:
        target_config = st.selectbox(
            "Target Configuration",
            ["Auto-detect", "ema_strategy", "macd_strategy", "smc_strategy", "zerolag_strategy"],
            help="Choose which strategy configuration to add this preset to"
        )
        
        preset_name = st.text_input(
            "Preset Name",
            value="tradingview_import",
            help="Name for the new configuration preset"
        )
    
    with col2:
        custom_name = st.text_input(
            "Custom Strategy Name",
            value="",
            help="Optional custom name for the strategy"
        )
        
        run_optimization = st.checkbox(
            "Run Parameter Optimization",
            value=False,
            help="Schedule parameter optimization after import"
        )
    
    # Preview configuration
    st.subheader("👀 Import Preview")
    
    if st.session_state.analysis_result:
        analysis = st.session_state.analysis_result
        signals = analysis.get('signals', {})
        
        # Predicted configuration
        predicted_config = "ema_strategy"  # Default
        if signals.get('mentions_smc'):
            predicted_config = "smc_strategy"
        elif signals.get('macd') and not signals.get('ema_periods'):
            predicted_config = "macd_strategy"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"🎯 Suggested Config: {predicted_config}")
        
        with col2:
            st.info(f"📊 Strategy Type: {signals.get('strategy_type', 'unknown').title()}")
        
        with col3:
            st.info(f"🧮 Complexity: {signals.get('complexity_score', 0):.2f}")
        
        # Configuration preview
        with st.expander("📋 Configuration Preview"):
            preview_config = {
                "preset_name": preset_name,
                "target_config": target_config if target_config != "Auto-detect" else predicted_config,
                "strategy_type": signals.get('strategy_type'),
                "indicators": {
                    "ema_periods": signals.get('ema_periods'),
                    "macd": signals.get('macd'),
                    "rsi_periods": signals.get('rsi_periods'),
                    "smc_features": signals.get('mentions_smc'),
                    "fvg_features": signals.get('mentions_fvg')
                },
                "metadata": {
                    "source": "tradingview",
                    "slug": strategy['slug'],
                    "title": strategy['title'],
                    "author": strategy['author']
                }
            }
            st.json(preview_config)
    
    # Import button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("📥 Import Strategy", type="primary", use_container_width=True):
            with st.spinner("📥 Importing strategy..."):
                final_target = target_config if target_config != "Auto-detect" else predicted_config
                
                result = import_strategy(
                    strategy['slug'],
                    final_target,
                    preset_name
                )
                
                if result.get('success'):
                    st.success("✅ Strategy imported successfully!")
                    st.balloons()
                    
                    # Add to import history
                    import_record = {
                        "timestamp": datetime.now().isoformat(),
                        "strategy": strategy['title'],
                        "author": strategy['author'],
                        "slug": strategy['slug'],
                        "target_config": final_target,
                        "preset_name": preset_name,
                        "success": True
                    }
                    st.session_state.import_history.append(import_record)
                    
                    st.info(f"Added preset '{preset_name}' to {final_target}")
                    
                else:
                    st.error(f"❌ Import failed: {result.get('error', 'Unknown error')}")

def render_history_page():
    """Render import history page"""
    st.title("📋 Import History")
    
    if not st.session_state.import_history:
        st.info("📭 No imports yet. Import some strategies to see them here!")
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Imports", len(st.session_state.import_history))
    
    with col2:
        successful = len([h for h in st.session_state.import_history if h.get('success')])
        st.metric("✅ Successful", successful)
    
    with col3:
        failed = len(st.session_state.import_history) - successful
        st.metric("❌ Failed", failed)
    
    with col4:
        unique_configs = len(set(h.get('target_config') for h in st.session_state.import_history))
        st.metric("🎯 Configs Used", unique_configs)
    
    # Import timeline
    st.subheader("⏰ Import Timeline")
    
    for import_record in reversed(st.session_state.import_history):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{import_record['strategy']}**")
                st.caption(f"by {import_record['author']}")
            
            with col2:
                st.text(import_record['target_config'])
            
            with col3:
                st.text(import_record['preset_name'])
            
            with col4:
                if import_record['success']:
                    st.success("✅ Success")
                else:
                    st.error("❌ Failed")
            
            st.caption(f"Imported: {import_record['timestamp']}")
            st.divider()

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Check system status on startup
    if st.session_state.system_status is None:
        check_system_status()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "🔍 Search & Browse":
        render_search_page()
    elif page == "📊 Strategy Analysis":
        render_analysis_page()
    elif page == "⚙️ Import Center":
        render_import_page()
    elif page == "📋 Import History":
        render_history_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("TradingView Strategy Importer v1.0")
    st.sidebar.caption("TradeSystemV1 Integration")

if __name__ == "__main__":
    main()