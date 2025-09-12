import streamlit as st
import time

log_files = {
    "Trade Monitor (dev)": "/logs/dev/trade_monitor.log",
    "FastAPI (dev)": "/logs/dev/fastapi-dev.log",
    "FastAPI (stream)": "/logs/stream/fastapi-stream.log",
    "Trade Scan (worker)": "/logs/worker/trade_scan.log"
}

st.title("📜 Real-time Log Viewer (Latest at Top)")

selected_log = st.selectbox("Select log file to stream:", options=list(log_files.keys()))
selected_path = log_files[selected_log]

log_placeholder = st.empty()
num_lines = st.slider("Lines to show", 10, 100, 30)

def tail_log(path, num_lines=30):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            trimmed = lines[-num_lines:] if len(lines) > num_lines else lines
            return list(reversed(trimmed))  # Reverse *after* slicing
    except Exception as e:
        return [f"Error reading log: {e}"]

# Live update
while True:
    log_lines = tail_log(selected_path, num_lines)
    log_placeholder.text("\n".join(log_lines))  # Make sure newline separator is used
    time.sleep(1)
