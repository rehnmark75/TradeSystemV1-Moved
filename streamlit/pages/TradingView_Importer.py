"""
TradingView Strategy Importer - Integrated with TradeSystemV1
Search, analyze, and import TradingView strategies into your trading system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import sys
import os
import requests
from typing import Dict, Any, List, Optional

# Page configuration
st.set_page_config(
    page_title="TradingView Strategy Importer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
TRADINGVIEW_API_BASE = "http://tradingview:8080"
REFRESH_INTERVAL = 30  # seconds

# Custom CSS to match your existing design
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #1f77b4;
    }
    .strategy-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fafafa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: bold;
    }
    .warning-badge {
        background-color: #ffc107;
        color: black;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: bold;
    }
    .error-badge {
        background-color: #dc3545;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: bold;
    }
    .info-badge {
        background-color: #17a2b8;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def safe_api_call(url: str, method: str = "GET", params: Dict = None, timeout: int = 10) -> Dict[str, Any]:
    """Safely make API call to TradingView service with error handling"""
    try:
        if method.upper() == "POST":
            response = requests.post(url, params=params, timeout=timeout)
        else:
            response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to TradingView service. Is it running?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_tradingview_service_health() -> Dict[str, Any]:
    """Check if TradingView service is available and healthy"""
    return safe_api_call(f"{TRADINGVIEW_API_BASE}/health")

def get_tradingview_stats() -> Dict[str, Any]:
    """Get TradingView library statistics"""
    return safe_api_call(f"{TRADINGVIEW_API_BASE}/api/tvscripts/stats")

def search_strategies(query: str, limit: int = 20) -> Dict[str, Any]:
    """Search TradingView strategies"""
    params = {"query": query, "limit": limit}
    return safe_api_call(f"{TRADINGVIEW_API_BASE}/api/tvscripts/search", method="POST", params=params)

def get_strategy_details(slug: str) -> Dict[str, Any]:
    """Get detailed strategy information"""
    return safe_api_call(f"{TRADINGVIEW_API_BASE}/api/tvscripts/script/{slug}")

def analyze_strategy(slug: str) -> Dict[str, Any]:
    """Analyze strategy patterns and indicators"""
    params = {"slug": slug}
    return safe_api_call(f"{TRADINGVIEW_API_BASE}/api/tvscripts/analyze", method="POST", params=params)

def initialize_session_state():
    """Initialize session state variables"""
    if 'tv_search_results' not in st.session_state:
        st.session_state.tv_search_results = []
    if 'tv_selected_strategy' not in st.session_state:
        st.session_state.tv_selected_strategy = None
    if 'tv_analysis_result' not in st.session_state:
        st.session_state.tv_analysis_result = None
    if 'tv_import_history' not in st.session_state:
        st.session_state.tv_import_history = []
    if 'tv_service_status' not in st.session_state:
        st.session_state.tv_service_status = None

def render_service_status():
    """Render TradingView service status"""
    st.subheader("🔧 TradingView Service Status")

    # Check service health
    health_result = check_tradingview_service_health()
    stats_result = get_tradingview_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if health_result["success"]:
            st.markdown('<div class="metric-card">✅ <strong>Service</strong><br>Healthy</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">❌ <strong>Service</strong><br>Unavailable</div>', unsafe_allow_html=True)

    if stats_result["success"]:
        stats = stats_result["data"]

        with col2:
            total_scripts = stats.get("total_scripts", 0)
            st.markdown(f'<div class="metric-card">📊 <strong>Scripts</strong><br>{total_scripts:,}</div>', unsafe_allow_html=True)

        with col3:
            strategies = stats.get("script_types", {}).get("strategy", 0)
            st.markdown(f'<div class="metric-card">🎯 <strong>Strategies</strong><br>{strategies:,}</div>', unsafe_allow_html=True)

        with col4:
            indicators = stats.get("script_types", {}).get("indicator", 0)
            st.markdown(f'<div class="metric-card">📈 <strong>Indicators</strong><br>{indicators:,}</div>', unsafe_allow_html=True)

        # Top categories
        if "categories" in stats:
            st.subheader("📋 Popular Categories")
            categories_df = pd.DataFrame([
                {"Category": k, "Scripts": v}
                for k, v in stats["categories"].items()
            ]).head(10)

            if not categories_df.empty:
                fig = px.bar(categories_df, x="Scripts", y="Category", orientation="h",
                           title="Most Popular Script Categories")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        with col2:
            st.markdown('<div class="metric-card">❌ <strong>Scripts</strong><br>N/A</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">❌ <strong>Strategies</strong><br>N/A</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">❌ <strong>Indicators</strong><br>N/A</div>', unsafe_allow_html=True)

    # Show error if service unavailable
    if not health_result["success"]:
        st.error(f"🚨 TradingView Service Error: {health_result['error']}")
        st.info("💡 Make sure the TradingView container is running: `./manage-tradingview.sh start`")

def render_search_page():
    """Render strategy search interface"""
    st.header("🔍 Strategy Search")
    st.markdown("Search and browse TradingView open-source strategies for import.")

    # Search interface
    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., ema crossover, macd strategy, smart money concepts",
            help="Search for strategies by keywords, indicators, or concepts"
        )

    with col2:
        limit = st.selectbox("Results", [5, 10, 20, 50], index=1)

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
            result = search_strategies(query, limit)

            if result["success"]:
                st.session_state.tv_search_results = result["data"].get("results", [])
                st.success(f"Found {len(st.session_state.tv_search_results)} strategies")
            else:
                st.error(f"Search failed: {result['error']}")
                st.session_state.tv_search_results = []

    # Display results
    if st.session_state.tv_search_results:
        st.subheader(f"📋 Search Results ({len(st.session_state.tv_search_results)})")

        for i, strategy in enumerate(st.session_state.tv_search_results):
            with st.container():
                st.markdown('<div class="strategy-card">', unsafe_allow_html=True)

                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{strategy.get('title', 'Unknown Title')}**")
                    st.caption(f"by {strategy.get('author', 'Unknown')} • {strategy.get('tags', '')}")
                    if strategy.get('description'):
                        desc = strategy['description'][:150]
                        st.markdown(f"*{desc}{'...' if len(strategy['description']) > 150 else ''}*")

                with col2:
                    # Strategy metadata
                    if strategy.get('strategy_type'):
                        st.markdown(f"📊 **Type:** {strategy['strategy_type'].title()}")

                    if strategy.get('likes'):
                        st.markdown(f"👍 **Likes:** {strategy['likes']:,}")

                    if strategy.get('open_source'):
                        st.markdown('<span class="success-badge">Open Source</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="error-badge">Closed Source</span>', unsafe_allow_html=True)

                with col3:
                    if st.button(f"📊 Analyze", key=f"analyze_{i}", help="Analyze strategy code"):
                        st.session_state.tv_selected_strategy = strategy
                        # Auto-analyze
                        with st.spinner("🔬 Analyzing strategy..."):
                            analysis_result = analyze_strategy(strategy.get('slug', ''))
                            if analysis_result["success"]:
                                st.session_state.tv_analysis_result = analysis_result["data"]
                                st.success("Analysis complete!")
                                st.rerun()
                            else:
                                st.error(f"Analysis failed: {analysis_result['error']}")

                    if strategy.get('open_source') and st.button(f"📥 Details", key=f"details_{i}", help="Get full strategy details"):
                        details_result = get_strategy_details(strategy.get('slug', ''))
                        if details_result["success"]:
                            st.json(details_result["data"])
                        else:
                            st.error(f"Failed to get details: {details_result['error']}")

                st.markdown('</div>', unsafe_allow_html=True)
                st.divider()

def render_analysis_results():
    """Render strategy analysis results"""
    if st.session_state.tv_selected_strategy and st.session_state.tv_analysis_result:
        strategy = st.session_state.tv_selected_strategy
        analysis = st.session_state.tv_analysis_result

        st.header("📊 Strategy Analysis Results")
        st.subheader(f"Analyzing: {strategy.get('title', 'Unknown')}")

        # Strategy overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("👤 Author", strategy.get('author', 'Unknown'))
        with col2:
            st.metric("👍 Likes", f"{strategy.get('likes', 0):,}")
        with col3:
            st.metric("📊 Uses", f"{strategy.get('uses', 0):,}")
        with col4:
            status = "✅ Open Source" if strategy.get('open_source') else "❌ Closed"
            st.metric("🔓 Status", status)

        # Analysis details
        if "signals" in analysis:
            signals = analysis["signals"]

            # Analysis metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                strategy_type = signals.get('strategy_type', 'unknown').title()
                st.metric("🎯 Strategy Type", strategy_type)

            with col2:
                complexity = signals.get('complexity_score', 0)
                st.metric("🧮 Complexity Score", f"{complexity:.2f}")

            with col3:
                inputs_count = len(analysis.get('inputs', []))
                st.metric("📝 Input Parameters", inputs_count)

            # Detected indicators
            st.subheader("🔍 Detected Indicators")

            col1, col2 = st.columns(2)

            with col1:
                indicators_found = []

                if signals.get('ema_periods'):
                    indicators_found.append(f"📈 **EMA**: {signals['ema_periods']}")

                if signals.get('macd'):
                    macd = signals['macd']
                    indicators_found.append(f"📊 **MACD**: {macd.get('fast', 'N/A')}/{macd.get('slow', 'N/A')}/{macd.get('signal', 'N/A')}")

                if signals.get('rsi_periods'):
                    indicators_found.append(f"📉 **RSI**: {signals['rsi_periods']}")

                if signals.get('bollinger_bands'):
                    bb = signals['bollinger_bands']
                    indicators_found.append(f"📐 **Bollinger Bands**: {bb.get('length', 'N/A')}, {bb.get('multiplier', 'N/A')}")

                for indicator in indicators_found:
                    st.markdown(indicator)

                if not indicators_found:
                    st.info("No standard indicators detected")

            with col2:
                features_found = []

                if signals.get('mentions_smc'):
                    features_found.append("🧠 **Smart Money Concepts**")

                if signals.get('mentions_fvg'):
                    features_found.append("⚡ **Fair Value Gaps**")

                if signals.get('has_cross_up') or signals.get('has_cross_down'):
                    features_found.append("🔄 **Crossover Signals**")

                if signals.get('higher_tf'):
                    tf_count = len(signals['higher_tf'])
                    features_found.append(f"⏰ **Multi-Timeframe**: {tf_count} TFs")

                for feature in features_found:
                    st.markdown(feature)

                if not features_found:
                    st.info("No special features detected")

            # Input parameters table
            if analysis.get('inputs'):
                st.subheader("⚙️ Input Parameters")
                inputs_df = pd.DataFrame(analysis['inputs'])
                st.dataframe(inputs_df, use_container_width=True)

            # Trading rules
            if analysis.get('rules'):
                st.subheader("📋 Trading Rules")
                for i, rule in enumerate(analysis['rules'], 1):
                    rule_type = rule.get('type', 'Unknown').title()
                    condition = rule.get('condition', 'No condition specified')
                    st.markdown(f"{i}. **{rule_type}**: {condition}")

        # Import suggestion
        if strategy.get('open_source'):
            st.subheader("📥 Import Suggestion")

            # Suggest target configuration based on analysis
            suggested_config = "ema_strategy"  # Default
            if analysis.get("signals", {}).get('mentions_smc'):
                suggested_config = "smc_strategy"
            elif analysis.get("signals", {}).get('macd') and not analysis.get("signals", {}).get('ema_periods'):
                suggested_config = "macd_strategy"

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🎯 **Suggested Config**: `{suggested_config}`")
            with col2:
                complexity_score = analysis.get("signals", {}).get('complexity_score', 0)
                if complexity_score > 0.7:
                    st.warning("⚠️ **High Complexity** - Manual review recommended")
                else:
                    st.success("✅ **Ready for Import**")

            st.markdown("---")
            st.markdown("**Next Steps:**")
            st.markdown("1. Review the analysis results above")
            st.markdown("2. Manual integration into your strategy configurations")
            st.markdown("3. Test the imported parameters in your backtesting system")
            st.markdown("4. Deploy to live trading if results are satisfactory")
        else:
            st.warning("⚠️ This strategy is not open-source and cannot be imported.")

def main():
    """Main application"""
    initialize_session_state()

    # Title and description
    st.title("📈 TradingView Strategy Importer")
    st.markdown("Search, analyze, and import TradingView community strategies into TradeSystemV1")

    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Service Status", "🔍 Search Strategies", "📊 Analysis Results"])

    with tab1:
        render_service_status()

    with tab2:
        render_search_page()

    with tab3:
        if st.session_state.tv_selected_strategy or st.session_state.tv_analysis_result:
            render_analysis_results()
        else:
            st.info("🔍 Search and analyze a strategy to see results here.")

    # Footer
    st.markdown("---")
    st.caption("TradingView Strategy Importer - Integrated with TradeSystemV1")

    # Auto-refresh service status
    if st.checkbox("Auto-refresh status", value=False):
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()