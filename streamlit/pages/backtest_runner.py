"""
Unified Backtest Runner
Streamlit page for running and analyzing backtests across all strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback

# Configure page
st.set_page_config(
    page_title="Strategy Backtest Runner",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import backtest services
from services.worker_backtest_service import (
    get_worker_backtest_service, BacktestConfig, BacktestResult
)
from services.container_backtest_service import get_container_backtest_service
from services.backtest_chart_service import get_chart_service
from services.data import get_epics

# Import chart rendering
try:
    from streamlit_lightweight_charts_ntf import renderLightweightCharts
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    st.error("⚠️ Chart rendering library not available. Charts will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)


# Initialize session state
if 'backtest_result' not in st.session_state:
    st.session_state.backtest_result = None
if 'selected_signals' not in st.session_state:
    st.session_state.selected_signals = []
if 'chart_data_cache' not in st.session_state:
    st.session_state.chart_data_cache = {}


def initialize_services():
    """Initialize backtest service with fallback"""
    try:
        # Try worker service first for existing strategies
        worker_service = get_worker_backtest_service()
        worker_health = worker_service.check_worker_health()

        if worker_health.get('status') == 'healthy':
            st.sidebar.success("🔗 Connected to worker container")
            return worker_service, 'worker'
        else:
            st.sidebar.warning("⚠️ Worker container unavailable, using container service")
            container_service = get_container_backtest_service()
            return container_service, 'container'

    except Exception as e:
        st.sidebar.warning(f"⚠️ Worker service error: {e}")
        try:
            container_service = get_container_backtest_service()
            st.sidebar.info("🔄 Using container-aware service")
            return container_service, 'container'
        except Exception as fallback_error:
            st.error(f"❌ Error initializing services: {fallback_error}")
            return None, None


def render_sidebar():
    """Render the sidebar with strategy selection and parameters"""
    st.sidebar.title("🧪 Backtest Configuration")

    # Initialize services
    service_result = initialize_services()
    if not service_result or service_result[0] is None:
        return None, None, None

    service, service_type = service_result

    # Strategy selection
    st.sidebar.subheader("📊 Strategy Selection")

    available_strategies = service.get_available_strategies()
    if not available_strategies:
        st.sidebar.error("❌ No strategies available")
        return None, None, None

    # Create display names
    strategy_display_map = {}
    for strategy_name, strategy_info in available_strategies.items():
        strategy_display_map[strategy_info.display_name] = strategy_name

    selected_display = st.sidebar.selectbox(
        "Choose Strategy",
        list(strategy_display_map.keys()),
        help="Select a trading strategy to backtest"
    )

    if not selected_display:
        return None, None, None

    selected_strategy = strategy_display_map[selected_display]
    strategy_info = available_strategies[selected_strategy]

    # Display strategy info
    if strategy_info:
        st.sidebar.info(f"📋 **{strategy_info.display_name}**\n\n{strategy_info.description}")

    # Basic parameters
    st.sidebar.subheader("⚙️ Basic Parameters")

    # Epic selection - get from database
    try:
        # Use database connection to get available epics
        import os
        from sqlalchemy import create_engine
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
        engine = create_engine(database_url)
        epic_options = get_epics(engine)
        if not epic_options:
            epic_options = [
                "CS.D.EURUSD.MINI.IP",
                "CS.D.GBPUSD.MINI.IP",
                "CS.D.USDJPY.MINI.IP",
                "CS.D.AUDUSD.MINI.IP"
            ]
    except Exception as e:
        st.sidebar.warning(f"Could not load epics from database: {e}")
        epic_options = [
            "CS.D.EURUSD.MINI.IP",
            "CS.D.GBPUSD.MINI.IP",
            "CS.D.USDJPY.MINI.IP",
            "CS.D.AUDUSD.MINI.IP"
        ]

    # Add "ALL EPICS" option
    epic_options_with_all = ["🌍 ALL EPICS"] + epic_options
    selected_epic_display = st.sidebar.selectbox("Trading Pair", epic_options_with_all)

    # Convert back to actual epic value (None for ALL EPICS)
    epic = None if selected_epic_display == "🌍 ALL EPICS" else selected_epic_display

    # Time parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        days = st.sidebar.number_input("Days", min_value=1, max_value=30, value=7)
    with col2:
        timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h"], index=1)

    # Strategy-specific parameters
    parameters = {}

    if strategy_info.parameters:
        st.sidebar.subheader("🎛️ Strategy Parameters")

        for param_name, param_config in strategy_info.parameters.items():
            if param_name in ['epic', 'days', 'timeframe']:
                continue  # Skip basic parameters

            param_type = param_config.get('type', 'text')
            default_value = param_config.get('default')
            description = param_config.get('description', '')

            if param_type == 'boolean':
                parameters[param_name] = st.sidebar.checkbox(
                    param_name.replace('_', ' ').title(),
                    value=default_value if default_value is not None else False,
                    help=description
                )
            elif param_type == 'number':
                min_val = param_config.get('min', 0.0)
                max_val = param_config.get('max', 100.0)
                step = param_config.get('step', 0.1)
                parameters[param_name] = st.sidebar.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default_value if default_value is not None else min_val,
                    step=step,
                    help=description
                )
            elif param_type == 'select':
                options = param_config.get('options', [])
                if options:
                    # Handle None option
                    display_options = [str(opt) if opt is not None else "Default" for opt in options]
                    selected_display = st.sidebar.selectbox(
                        param_name.replace('_', ' ').title(),
                        display_options,
                        help=description
                    )
                    # Map back to actual value
                    selected_index = display_options.index(selected_display)
                    parameters[param_name] = options[selected_index]
                else:
                    parameters[param_name] = st.sidebar.text_input(
                        param_name.replace('_', ' ').title(),
                        value=str(default_value) if default_value is not None else "",
                        help=description
                    )

    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        show_signals = st.checkbox("Show Individual Signals", value=True)
        parameters['show_signals'] = show_signals

        export_results = st.checkbox("Enable Export", value=False)

        clear_cache = st.button("🗑️ Clear Cache")
        if clear_cache:
            st.session_state.chart_data_cache = {}
            st.success("Cache cleared!")

    # Create backtest config
    config = BacktestConfig(
        strategy_name=selected_strategy,
        epic=epic,
        days=days,
        timeframe=timeframe,
        parameters=parameters
    )

    return config, export_results, service_type


def run_backtest_with_progress(config: BacktestConfig, service_type: str):
    """Run backtest with progress updates"""

    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(percent: int, message: str):
        progress_bar.progress(percent / 100)
        status_text.text(f"🚀 {message}")

    try:
        # Get appropriate service
        if service_type == 'worker':
            service = get_worker_backtest_service()
            status_text.text("🔗 Executing via worker container...")
        else:
            service = get_container_backtest_service()
            status_text.text("🔄 Executing via container service...")

        # Run backtest
        result = service.run_backtest(config)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        return result

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Backtest execution failed: {str(e)}")
        st.error(traceback.format_exc())
        return None


def render_results_summary(result: BacktestResult):
    """Render enhanced backtest results summary with market intelligence"""
    if not result.success:
        st.error(f"❌ Backtest failed: {result.error_message}")
        return

    # Handle execution time formatting safely
    exec_time = result.execution_time if result.execution_time is not None else 0
    st.success(f"✅ Backtest completed in {exec_time:.2f} seconds")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Strategy", result.strategy_name.upper())
    with col2:
        st.metric("Total Signals", result.total_signals)
    with col3:
        # Handle None epic for "All Epics" runs
        if result.epic is None:
            epic_display = "ALL EPICS"
        elif '.' in result.epic:
            epic_display = result.epic.split('.')[-3]
        else:
            epic_display = result.epic
        st.metric("Epic", epic_display)
    with col4:
        st.metric("Timeframe", result.timeframe)
    with col5:
        # Check if this is an enhanced result (from StandardBacktestResult)
        enhanced = "✅ Enhanced" if hasattr(result, 'market_intelligence_summary') else "⚠️ Legacy"
        st.metric("Framework", enhanced)

    # Enhanced feature detection
    enhanced_features = []
    if hasattr(result, 'market_intelligence_summary'):
        enhanced_features.append("🧠 Market Intelligence")
    if result.signals and any(signal.get('smart_money_analysis') for signal in result.signals[:5]):
        enhanced_features.append("💰 Smart Money Analysis")
    if result.signals and any(signal.get('market_conditions') for signal in result.signals[:5]):
        enhanced_features.append("🌐 Market Conditions")

    if enhanced_features:
        st.info(f"**Enhanced Features Active:** {', '.join(enhanced_features)}")

    # Performance metrics
    if result.performance_metrics:
        st.subheader("📊 Performance Metrics")

        metrics = result.performance_metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            win_rate = metrics.get('win_rate', 0)
            if win_rate > 1:
                win_rate = win_rate / 100  # Handle percentage vs decimal
            st.metric("Win Rate", f"{win_rate:.1%}")

        with col2:
            avg_confidence = metrics.get('avg_confidence', 0)
            if avg_confidence > 1:
                avg_confidence = avg_confidence / 100
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

        with col3:
            avg_profit = metrics.get('avg_profit_pips', 0)
            st.metric("Avg Profit", f"{avg_profit:.1f} pips")

        with col4:
            avg_rr = metrics.get('risk_reward_ratio', metrics.get('avg_risk_reward', 0))
            # Handle None values for risk/reward ratio
            if avg_rr is None or avg_rr == 0:
                st.metric("Risk/Reward", "N/A")
            else:
                st.metric("Risk/Reward", f"{avg_rr:.2f}")

        with col5:
            # Show market regime distribution if available
            if hasattr(result, 'market_intelligence_summary') and result.market_intelligence_summary:
                market_summary = result.market_intelligence_summary
                regime_analysis = market_summary.get('regime_analysis', {})
                most_common_regime = regime_analysis.get('most_common_regime', 'unknown')
                st.metric("Market Regime", most_common_regime.replace('_', ' ').title())

    # Enhanced Market Intelligence Section
    if hasattr(result, 'market_intelligence_summary') and result.market_intelligence_summary:
        render_market_intelligence_summary(result.market_intelligence_summary)

    # All Epics Results Breakdown
    if result.epic == "ALL_EPICS" and hasattr(result, 'epic_results'):
        render_all_epics_breakdown(result)


def render_market_intelligence_summary(market_summary: dict):
    """Render market intelligence analysis"""
    st.subheader("🧠 Market Intelligence Analysis")

    if not market_summary.get('market_conditions_detected', False):
        st.info("ℹ️ Market intelligence not available for this backtest")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Market Regime Analysis**")
        regime_analysis = market_summary.get('regime_analysis', {})
        if regime_analysis:
            regime_dist = regime_analysis.get('distribution', {})
            if regime_dist:
                # Create a simple visualization of regime distribution
                regime_df = pd.DataFrame([
                    {'Regime': k.replace('_', ' ').title(), 'Count': v}
                    for k, v in regime_dist.items()
                ])
                st.dataframe(regime_df, hide_index=True)

                most_common = regime_analysis.get('most_common_regime', 'unknown')
                st.info(f"🎯 **Dominant Regime:** {most_common.replace('_', ' ').title()}")

    with col2:
        st.markdown("**Volatility & Session Analysis**")
        vol_analysis = market_summary.get('volatility_analysis', {})
        session_analysis = market_summary.get('session_analysis', {})

        if vol_analysis:
            avg_vol = vol_analysis.get('average_volatility_percentile', 0.5)
            st.metric("Avg Volatility Percentile", f"{avg_vol:.1%}")

        if session_analysis:
            session_dist = session_analysis.get('distribution', {})
            most_active = session_analysis.get('most_active_session', 'unknown')
            st.info(f"📅 **Most Active Session:** {most_active.replace('_', ' ').title()}")


def render_all_epics_breakdown(result: BacktestResult):
    """Render breakdown of All Epics results"""
    if not hasattr(result, 'epic_results'):
        return

    st.subheader("🌍 All Epics Results Breakdown")

    epic_results = getattr(result, 'epic_results', {})
    if not epic_results:
        st.warning("No epic-specific results available")
        return

    # Create summary table
    summary_data = []
    for epic, epic_result in epic_results.items():
        summary_data.append({
            'Epic': epic.split('.')[-3] if '.' in epic else epic,
            'Signals': len(epic_result.signals) if hasattr(epic_result, 'signals') else 0,
            'Status': '✅ Success' if not getattr(epic_result, 'error_message', None) else '❌ Failed',
            'Avg Confidence': f"{getattr(epic_result, 'performance_metrics', {}).get('avg_confidence', 0):.1%}",
            'Market Regime': getattr(epic_result, 'market_conditions_summary', {}).get('regime', 'unknown') if hasattr(epic_result, 'market_conditions_summary') else 'N/A'
        })

    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, hide_index=True, use_container_width=True)

        # Epic selection for detailed view
        st.subheader("📊 Epic-Specific Analysis")
        selected_epic_display = st.selectbox(
            "Select Epic for Detailed Analysis",
            [row['Epic'] for row in summary_data],
            key="epic_detail_selector"
        )

        # Find the full epic name
        selected_full_epic = None
        for epic in epic_results.keys():
            if epic.split('.')[-3] == selected_epic_display or epic == selected_epic_display:
                selected_full_epic = epic
                break

        if selected_full_epic and selected_full_epic in epic_results:
            epic_detail = epic_results[selected_full_epic]
            render_epic_detail(selected_full_epic, epic_detail)


def render_epic_detail(epic: str, epic_result):
    """Render detailed analysis for a specific epic"""
    st.markdown(f"### 📈 {epic}")

    if hasattr(epic_result, 'error_message') and epic_result.error_message:
        st.error(f"❌ Error: {epic_result.error_message}")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        signal_count = len(epic_result.signals) if hasattr(epic_result, 'signals') else 0
        st.metric("Signals", signal_count)

    with col2:
        if hasattr(epic_result, 'performance_metrics') and epic_result.performance_metrics:
            avg_conf = epic_result.performance_metrics.get('avg_confidence', 0)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

    with col3:
        if hasattr(epic_result, 'market_conditions_summary') and epic_result.market_conditions_summary:
            regime = getattr(epic_result.market_conditions_summary, 'regime', 'unknown')
            if hasattr(regime, 'value'):
                regime = regime.value
            st.metric("Market Regime", regime.replace('_', ' ').title())

    # Show signals for this epic if available
    if hasattr(epic_result, 'signals') and epic_result.signals:
        st.markdown("**Recent Signals:**")
        signals_to_show = epic_result.signals[:5]  # Show latest 5 signals

        signal_data = []
        for signal in signals_to_show:
            if hasattr(signal, 'to_dict'):
                signal_dict = signal.to_dict()
            else:
                signal_dict = signal

            signal_data.append({
                'Type': signal_dict.get('signal_type', 'N/A'),
                'Price': f"{signal_dict.get('price', 0):.5f}",
                'Confidence': f"{signal_dict.get('confidence', 0):.1%}",
                'Timestamp': signal_dict.get('timestamp', 'N/A')[:19] if signal_dict.get('timestamp') else 'N/A'
            })

        if signal_data:
            signals_df = pd.DataFrame(signal_data)
            st.dataframe(signals_df, hide_index=True, use_container_width=True)


def render_chart(result: BacktestResult):
    """Render the chart with signals"""
    if not CHARTS_AVAILABLE:
        st.warning("⚠️ Chart rendering not available")
        return

    if not result.success:
        return

    st.subheader("📈 Chart Analysis")

    try:
        # Get chart service
        chart_service = get_chart_service()

        # Prepare chart data
        chart_data_dict = chart_service.prepare_chart_data(result, result.chart_data)

        if chart_data_dict.get('is_minimal'):
            st.warning("⚠️ No price data available - showing minimal chart")

        # Create charts list for rendering
        charts_to_render = []

        # Main price chart with markers
        main_series = chart_data_dict['series'].copy()
        if chart_data_dict['markers']:
            # Add markers to the candlestick series
            for series in main_series:
                if series.get('type') == 'Candlestick':
                    series['markers'] = chart_data_dict['markers']
                    break

        main_chart = {
            "chart": chart_data_dict['chart_config'],
            "series": main_series
        }
        charts_to_render.append(main_chart)

        # Add indicator charts
        if result.chart_data is not None and not result.chart_data.empty:
            indicator_charts = chart_service.create_indicator_charts(
                result.chart_data,
                result.strategy_name
            )
            charts_to_render.extend(indicator_charts)

        # Render charts
        renderLightweightCharts(charts_to_render, f"backtest-chart-{result.strategy_name}")

        # Chart info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Candles", chart_data_dict['candle_count'])
        with col2:
            st.metric("Signal Markers", chart_data_dict['signal_count'])
        with col3:
            st.metric("Series", len(main_series))

    except Exception as e:
        st.error(f"❌ Error rendering chart: {str(e)}")
        st.error(traceback.format_exc())


def render_signals_table(result: BacktestResult):
    """Render interactive signals table"""
    if not result.success or not result.signals:
        return

    st.subheader("🎯 Signals Analysis")

    # Convert signals to DataFrame
    df = pd.DataFrame(result.signals)

    if df.empty:
        st.info("No signals found in this backtest")
        return

    # Format DataFrame for display
    display_df = df.copy()

    # Format numeric columns
    numeric_columns = ['entry_price', 'confidence', 'max_profit_pips', 'max_loss_pips', 'profit_loss_ratio']
    for col in numeric_columns:
        if col in display_df.columns:
            if col == 'confidence':
                display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
            elif 'pips' in col:
                display_df[col] = display_df[col].round(1).astype(str) + ' pips'
            elif col == 'entry_price':
                display_df[col] = display_df[col].round(5)
            else:
                display_df[col] = display_df[col].round(2)

    # Format timestamp
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Reorder columns for better display
    column_order = ['timestamp', 'signal_type', 'direction', 'entry_price', 'confidence',
                   'max_profit_pips', 'max_loss_pips', 'profit_loss_ratio']
    available_columns = [col for col in column_order if col in display_df.columns]
    other_columns = [col for col in display_df.columns if col not in available_columns]
    display_columns = available_columns + other_columns

    display_df = display_df[display_columns]

    # Add filters
    col1, col2, col3 = st.columns(3)

    with col1:
        direction_filter = st.selectbox(
            "Filter by Direction",
            ['All'] + list(df['direction'].unique()) if 'direction' in df.columns else ['All']
        )

    with col2:
        signal_type_filter = st.selectbox(
            "Filter by Signal Type",
            ['All'] + list(df['signal_type'].unique()) if 'signal_type' in df.columns else ['All']
        )

    with col3:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )

    # Apply filters
    filtered_df = display_df.copy()

    if direction_filter != 'All' and 'direction' in df.columns:
        filtered_df = filtered_df[df['direction'] == direction_filter]

    if signal_type_filter != 'All' and 'signal_type' in df.columns:
        filtered_df = filtered_df[df['signal_type'] == signal_type_filter]

    if 'confidence' in df.columns:
        filtered_df = filtered_df[df['confidence'] >= min_confidence]

    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

    # Table summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered Signals", len(filtered_df))
    with col2:
        st.metric("Total Signals", len(display_df))
    with col3:
        if len(filtered_df) > 0 and 'max_profit_pips' in df.columns:
            avg_profit = df.loc[filtered_df.index, 'max_profit_pips'].mean()
            st.metric("Avg Profit", f"{avg_profit:.1f} pips")


def render_export_options(result: BacktestResult):
    """Render export options for results"""
    if not result.success:
        return

    st.subheader("📁 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Signals CSV"):
            if result.signals:
                signals_df = pd.DataFrame(result.signals)
                csv = signals_df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{result.strategy_name}_signals_{timestamp}.csv"

                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )

    with col2:
        if st.button("📈 Export Performance JSON"):
            performance_data = {
                'strategy': result.strategy_name,
                'epic': result.epic,
                'timeframe': result.timeframe,
                'execution_time': result.execution_time,
                'total_signals': result.total_signals,
                'performance_metrics': result.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }

            json_str = json.dumps(performance_data, indent=2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.strategy_name}_performance_{timestamp}.json"

            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )

    with col3:
        if st.button("📋 Copy Config"):
            config_data = {
                'strategy_name': result.strategy_name,
                'epic': result.epic,
                'timeframe': result.timeframe
            }

            st.code(json.dumps(config_data, indent=2), language='json')


def main():
    """Main application function"""
    st.title("🧪 Strategy Backtest Runner")
    st.markdown("**Unified backtesting platform for all trading strategies**")

    # Render sidebar and get configuration
    sidebar_result = render_sidebar()

    if not sidebar_result or sidebar_result[0] is None:
        st.warning("⚠️ Please configure backtest parameters in the sidebar")
        return

    config, enable_export, service_type = sidebar_result

    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            result = run_backtest_with_progress(config, service_type)

            if result:
                st.session_state.backtest_result = result
                st.rerun()

    # Display results if available
    if st.session_state.backtest_result:
        result = st.session_state.backtest_result

        # Results summary
        render_results_summary(result)

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Chart", "🎯 Signals", "📊 Analysis"])

        with tab1:
            render_chart(result)

        with tab2:
            render_signals_table(result)

        with tab3:
            # Additional analysis tab
            st.subheader("📊 Detailed Analysis")

            if result.performance_metrics:
                # Performance breakdown
                st.write("**Performance Breakdown:**")
                st.json(result.performance_metrics)

            # Strategy-specific metrics
            if hasattr(result, 'strategy_specific_metrics'):
                st.write("**Strategy-Specific Metrics:**")
                st.json(result.strategy_specific_metrics)

        # Export options
        if enable_export:
            render_export_options(result)


if __name__ == "__main__":
    main()