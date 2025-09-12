from datetime import datetime
from typing import List, Dict
import plotly.graph_objects as go
from draggable_line_chart import draggable_line_chart
import pandas as pd

def plot_structure_signals(fig: go.Figure, signals: List[Dict]) -> go.Figure:
    """
    Plots BOS/CHoCH structure lines and labels in LuxAlgo style.

    Args:
        fig: The existing Plotly Figure object.
        signals: List of detected structure signals (BOS, CHoCH).

    Returns:
        The updated Plotly Figure.
    """
    for sig in signals:
        label = sig["label"]
        direction = sig["direction"]
        from_time = sig["from_time"]
        to_time = sig["to_time"]
        level = sig["price"]

        # Set visual style
        color = "green" if direction == "bullish" else "red"
        label_y_offset = 0.0003 * level  # offset to avoid overlap
        label_y = level + label_y_offset if direction == "bullish" else level - label_y_offset

        # Draw horizontal dotted structure line from from_time to to_time
        fig.add_shape(
            type="line",
            x0=from_time,
            x1=to_time,
            y0=level,
            y1=level,
            line=dict(color=color, width=1.5, dash="dot"),
            layer="above"
        )

        # Draw label (e.g., BOS or CHoCH)
        fig.add_trace(go.Scatter(
            x=[to_time],
            y=[label_y],
            text=[label],
            mode="text",
            textfont=dict(size=11, color=color),
            showlegend=False
        ))

    return fig


def plot_measurement(fig: go.Figure, df: pd.DataFrame, start_index: int, end_index: int) -> None:
    """
    Adds a measurement annotation to the chart showing:
    - price change (points and percentage)
    - number of candles (bars) and duration

    Args:
        fig (go.Figure): Plotly figure to annotate.
        df (pd.DataFrame): Candle dataframe with 'start_time' and 'close'.
        start_index (int): Index of starting candle.
        end_index (int): Index of ending candle.
    """
    if start_index >= end_index or start_index < 0 or end_index >= len(df):
        return

    # Enable interactive update of start and end price via draggable component
    price_series = pd.Series([df.loc[start_index, "close"], df.loc[end_index, "close"]], name="Selected Prices")
    updated_prices = draggable_line_chart(price_series)

    start_price = updated_prices.iloc[0]
    end_price = updated_prices.iloc[1]

    start_time = df.loc[start_index, "start_time"]
    end_time = df.loc[end_index, "start_time"]

    price_change = end_price - start_price
    price_pct = (price_change / start_price) * 100
    bar_count = end_index - start_index
    time_delta = end_time - start_time

    mid_price = (start_price + end_price) / 2
    mid_time = df.loc[(start_index + end_index) // 2, "start_time"]

    color = "green" if price_change > 0 else "red"

    # Draw arrow line
    fig.add_shape(
        type="line",
        x0=start_time,
        y0=start_price,
        x1=end_time,
        y1=end_price,
        line=dict(color=color, width=2),
        layer="above"
    )

    # Draw vertical box for visual range
    fig.add_shape(
        type="rect",
        x0=start_time,
        x1=end_time,
        y0=min(start_price, end_price),
        y1=max(start_price, end_price),
        line=dict(width=0),
        fillcolor="rgba(255,0,0,0.1)" if color == "red" else "rgba(0,255,0,0.1)",
        layer="below"
    )

    # Add annotation
    annotation = (
        f"{price_change:+.5f} ({price_pct:+.2f}%)<br>"
        f"{bar_count} bars, {time_delta}<br>"
    )
    fig.add_annotation(
        x=mid_time,
        y=mid_price,
        text=annotation,
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center",
        bordercolor=color,
        borderwidth=1,
        bgcolor=color,
        opacity=0.9
    )
