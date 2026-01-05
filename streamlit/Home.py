"""
Trading System Dashboard - Home
Main landing page with navigation to all dashboard pages.
"""
import streamlit as st

st.set_page_config(
    page_title="Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Trading System Dashboard")
st.markdown("---")

st.markdown("### Select a page from the sidebar to get started")

st.markdown("""
**Available Pages:**

- **Unified Analytics** - Main trading analytics dashboard with all tabs
- **Infrastructure Status** - Container monitoring and system health
- **System Status** - System overview and status checks
- **Stock Scanner** - Stock screening and analysis
- **TV Chart** - TradingView chart integration
- **Configuration Guide** - System configuration documentation
- **Forex Documentation** - Forex trading documentation
""")

st.markdown("---")
st.caption("Use the sidebar navigation to switch between pages.")
