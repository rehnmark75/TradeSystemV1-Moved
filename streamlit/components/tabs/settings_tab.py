"""
Settings & Debug Tab Component

Renders the settings and debug tab with:
- Database connection testing
- Manual connection override
- API operations
- System information
"""

import streamlit as st
import psycopg2
import requests
from datetime import datetime

from services.db_utils import get_psycopg2_connection


def render_settings_tab():
    """Render the settings and debug tab"""
    st.header("Settings & Debug")

    # Database connection status
    st.subheader("Database Connection")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Test Database Connection", key="test_db_conn"):
            conn = get_psycopg2_connection("trading")
            if conn:
                st.success("Database connection successful!")
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM trade_log")
                        count = cursor.fetchone()[0]
                        st.info(f"Found {count} records in trade_log table")
                except Exception as e:
                    st.warning(f"Connected but query failed: {e}")
                finally:
                    conn.close()
            else:
                st.error("Database connection failed!")

    with col2:
        if st.button("Clear Cache", key="clear_cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    # Connection debug information
    st.subheader("Connection Debug")

    if hasattr(st, 'secrets'):
        try:
            secrets_dict = st.secrets.to_dict()
            st.success(f"[+] Secret sections: {list(secrets_dict.keys())}")

            if 'database' in secrets_dict:
                db_keys = list(secrets_dict['database'].keys())
                st.success(f"[+] Database keys: {db_keys}")
            else:
                st.warning("[!] No [database] section found")

        except Exception as e:
            st.error(f"[-] Error reading secrets: {e}")
    else:
        st.error("[-] No secrets available")

    # Manual connection override
    st.subheader("Manual Connection Override")

    col1, col2 = st.columns(2)

    with col1:
        manual_host = st.text_input("Host", value="postgres", key="manual_host")
        manual_port = st.number_input("Port", value=5432, min_value=1, max_value=65535, key="manual_port")
        manual_db = st.text_input("Database", value="trading", key="manual_db")

    with col2:
        manual_user = st.text_input("Username", value="postgres", key="manual_user")
        manual_pass = st.text_input("Password", value="postgres", type="password", key="manual_pass")

    if st.button("Test Manual Settings", key="test_manual"):
        manual_conn_str = f"postgresql://{manual_user}:{manual_pass}@{manual_host}:{manual_port}/{manual_db}"
        try:
            test_conn = psycopg2.connect(manual_conn_str)
            test_conn.close()
            st.success("Manual connection successful!")
            st.info("Consider updating your secrets.toml with these settings")
        except Exception as e:
            st.error(f"Manual connection failed: {e}")

    # API Operations
    st.subheader("API Operations")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Calculate Complete P&L", key="calc_pnl_btn"):
            try:
                headers = {
                    "X-APIM-Gateway": "verified",
                    "X-API-KEY": "436abe054a074894a0517e5172f0e5b6",
                    "Content-Type": "application/json"
                }
                payload = {
                    "days_back": 7,
                    "update_trade_log": True,
                    "calculate_prices": True,
                    "include_detailed_results": False
                }
                response = requests.post(
                    "http://fastapi-dev:8000/api/trading/deals/calculate-complete-pnl",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("P&L calculation completed!")
                    st.json(result)
                else:
                    st.error(f"API call failed: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error calling API: {e}")

    with col2:
        if st.button("Correlate Activities", key="correlate_btn"):
            try:
                headers = {
                    "X-APIM-Gateway": "verified",
                    "X-API-KEY": "436abe054a074894a0517e5172f0e5b6",
                    "Content-Type": "application/json"
                }
                payload = {
                    "days_back": 7,
                    "update_trade_log": True,
                    "include_trade_lifecycles": False
                }
                response = requests.post(
                    "http://fastapi-dev:8000/api/trading/deals/correlate-activities",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("Activity correlation completed!")
                    st.json(result)
                else:
                    st.error(f"API call failed: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error calling API: {e}")

    # System information
    st.subheader("System Information")

    system_info = {
        "Streamlit Version": st.__version__,
        "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Session State Keys": len(st.session_state.keys())
    }

    if 'last_refresh' in st.session_state:
        system_info["Last Refresh"] = st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M:%S")

    for key, value in system_info.items():
        st.write(f"**{key}:** {value}")

    # Debug mode
    if st.checkbox("Show Debug Information", key="show_debug"):
        st.subheader("Session State")
        st.write(dict(st.session_state))
