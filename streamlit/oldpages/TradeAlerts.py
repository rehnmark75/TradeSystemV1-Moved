import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from datetime import datetime
import os

st.title("📈 Trade Alerts Viewer")

engine = create_engine(os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex"))

# Load alerts
df = pd.read_sql("SELECT * FROM alerts ORDER BY start_time DESC", engine)

# Filters
alert_types = df['alert_type'].unique()
selected_type = st.selectbox("Alert Type", alert_types)
df = df[df['alert_type'] == selected_type]

epic_options = df['epic'].unique()
selected_epic = st.selectbox("Select EPIC", epic_options)
filtered = df[df['epic'] == selected_epic]

# Plot
fig = px.scatter(
    filtered,
    x="start_time",
    y="price",
    color="direction",
    title=f"Alerts for {selected_epic}",
    labels={"start_time": "Time", "price": "Price"},
    symbol="direction"
)
st.plotly_chart(fig, use_container_width=True)
