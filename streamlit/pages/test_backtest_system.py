"""
Test Backtest System
Test page for the container-aware backtest system
"""

import streamlit as st
import traceback
from datetime import datetime

st.set_page_config(
    page_title="Test Backtest System",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Test Backtest System")
st.markdown("**Testing the container-aware backtest system**")

# Test 1: Import Test
st.subheader("1️⃣ Import Test")
try:
    from services.container_backtest_service import get_container_backtest_service, BacktestConfig
    from services.worker_backtest_service import get_worker_backtest_service
    st.success("✅ Successfully imported both container and worker backtest services")
except Exception as e:
    st.error(f"❌ Import failed: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Test 2: Service Initialization
st.subheader("2️⃣ Service Initialization")

# Test Container Service
try:
    container_service = get_container_backtest_service()
    st.success(f"✅ Container service initialized: {type(container_service)}")
except Exception as e:
    st.error(f"❌ Container service initialization failed: {e}")
    st.code(traceback.format_exc())

# Test Worker Service
try:
    worker_service = get_worker_backtest_service()
    st.success(f"✅ Worker service initialized: {type(worker_service)}")

    # Test worker health
    worker_health = worker_service.check_worker_health()
    if worker_health.get('status') == 'healthy':
        st.success("🔗 Worker container is healthy and available")
        st.write("Worker Info:")
        st.json(worker_service.get_worker_info())
    else:
        st.warning(f"⚠️ Worker container status: {worker_health.get('status', 'unknown')}")
        st.json(worker_health)

except Exception as e:
    st.warning(f"⚠️ Worker service initialization failed: {e}")
    st.code(traceback.format_exc())

# Test 3: Strategy Discovery
st.subheader("3️⃣ Strategy Discovery")

# Test Container Strategies
try:
    container_strategies = container_service.get_available_strategies()
    st.success(f"✅ Container service found {len(container_strategies)} strategies:")

    for name, info in container_strategies.items():
        with st.expander(f"📋 Container: {info.display_name}"):
            st.write(f"**Name:** {name}")
            st.write(f"**Description:** {info.description}")
            st.write("**Parameters:**")
            st.json(info.parameters)

except Exception as e:
    st.error(f"❌ Container strategy discovery failed: {e}")
    st.code(traceback.format_exc())

# Test Worker Strategies (if available)
try:
    if 'worker_service' in locals() and worker_health.get('status') == 'healthy':
        worker_strategies = worker_service.get_available_strategies()
        st.success(f"🔗 Worker service found {len(worker_strategies)} existing strategies:")

        for name, info in worker_strategies.items():
            with st.expander(f"🚀 Worker: {info.display_name}"):
                st.write(f"**Name:** {name}")
                st.write(f"**Description:** {info.description}")
                st.write("**Parameters:**")
                st.json(info.parameters)
    else:
        st.info("ℹ️ Worker service not available for strategy discovery")

except Exception as e:
    st.warning(f"⚠️ Worker strategy discovery failed: {e}")
    st.code(traceback.format_exc())

# Test 4: Configuration Test
st.subheader("4️⃣ Configuration Test")
try:
    config = BacktestConfig(
        strategy_name='historical_signals',
        epic='CS.D.EURUSD.MINI.IP',
        days=3,
        timeframe='15m',
        parameters={'min_confidence': 0.7}
    )
    st.success(f"✅ Configuration created for: {config.strategy_name}")
    st.json({
        'strategy_name': config.strategy_name,
        'epic': config.epic,
        'days': config.days,
        'timeframe': config.timeframe,
        'parameters': config.parameters
    })
except Exception as e:
    st.error(f"❌ Configuration creation failed: {e}")
    st.code(traceback.format_exc())

# Test 5: Backtest Execution
st.subheader("5️⃣ Backtest Execution Test")

# Select service type
service_type = st.radio(
    "Select service to test:",
    ['container', 'worker'] if 'worker_service' in locals() and worker_health.get('status') == 'healthy' else ['container']
)

# Get strategies for selected service
if service_type == 'worker' and 'worker_strategies' in locals():
    available_strategies = worker_strategies
    test_service = worker_service
else:
    available_strategies = container_strategies
    test_service = container_service

strategy_to_test = st.selectbox(
    "Select strategy to test:",
    list(available_strategies.keys())
)

if st.button("🚀 Run Test Backtest"):
    try:
        with st.spinner("Running backtest..."):
            test_config = BacktestConfig(
                strategy_name=strategy_to_test,
                epic='CS.D.EURUSD.MINI.IP',
                days=3,
                timeframe='15m',
                parameters={'min_confidence': 0.7} if strategy_to_test == 'historical_signals' else {}
            )

            result = test_service.run_backtest(test_config)

        if result.success:
            st.success("✅ Backtest completed successfully!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Strategy", result.strategy_name)
            with col2:
                st.metric("Total Signals", result.total_signals)
            with col3:
                st.metric("Epic", result.epic.split('.')[-3] if '.' in result.epic else result.epic)
            with col4:
                st.metric("Execution Time", f"{result.execution_time:.2f}s")

            if result.performance_metrics:
                st.subheader("📊 Performance Metrics")
                st.json(result.performance_metrics)

            if result.signals:
                st.subheader("🎯 Sample Signals")
                st.write(f"Showing first 5 of {len(result.signals)} signals:")

                import pandas as pd
                signals_df = pd.DataFrame(result.signals[:5])
                st.dataframe(signals_df, use_container_width=True)

            if result.chart_data is not None and not result.chart_data.empty:
                st.subheader("📈 Chart Data Sample")
                st.write(f"Chart data shape: {result.chart_data.shape}")
                st.dataframe(result.chart_data.head(), use_container_width=True)

        else:
            st.error(f"❌ Backtest failed: {result.error_message}")

    except Exception as e:
        st.error(f"❌ Backtest execution failed: {e}")
        st.code(traceback.format_exc())

# Test 6: Database Connection Test
st.subheader("6️⃣ Database Connection Test")
try:
    import os
    from sqlalchemy import create_engine, text
    from services.data import get_epics

    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
    engine = create_engine(database_url)

    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        st.success("✅ Database connection successful")

    # Test epics
    epics = get_epics(engine)
    st.success(f"✅ Found {len(epics)} epics in database")

    if epics:
        st.write("Sample epics:")
        st.write(epics[:5])

except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.code(traceback.format_exc())

# System Information
st.subheader("ℹ️ System Information")
col1, col2 = st.columns(2)

with col1:
    st.write("**Environment Variables:**")
    import os
    env_vars = {
        'DATABASE_URL': os.getenv('DATABASE_URL', 'Not set'),
        'PYTHONPATH': os.getenv('PYTHONPATH', 'Not set')
    }
    st.json(env_vars)

with col2:
    st.write("**Python Path:**")
    import sys
    st.write(f"Python executable: {sys.executable}")
    st.write("Python path:")
    for path in sys.path[:5]:  # Show first 5 paths
        st.code(path)

st.success("🎉 Test page loaded successfully! The container-aware backtest system is ready.")