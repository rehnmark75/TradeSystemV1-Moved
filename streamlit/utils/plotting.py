import plotly.graph_objects as go
import pandas as pd


def build_candlestick_chart(df, trades_df, internal_swings, swing_swings, structure_signals,
                             trailing_extremes, zones, show_premium, show_discount, show_equilibrium,
                             indicators, ema1_period, ema2_period, show_swings, show_structure, bias,
                             zone_x0=None, zone_x1=None, show_swing_labels=True):

    fig = go.Figure()

    # --- Zones ---
    if zones:
        if show_premium and zones.get("premium_bottom") is not None and zones.get("premium_top") is not None:
            fig.add_shape(type="rect", x0=zone_x0, x1=zone_x1,
                          y0=zones["premium_bottom"], y1=zones["premium_top"],
                          fillcolor="rgba(255,0,0,0.08)", line=dict(width=0), layer="below")

        if show_discount and zones.get("discount_bottom") is not None and zones.get("discount_top") is not None:
            fig.add_shape(type="rect", x0=zone_x0, x1=zone_x1,
                          y0=zones["discount_bottom"], y1=zones["discount_top"],
                          fillcolor="rgba(0,255,0,0.08)", line=dict(width=0), layer="below")

        if show_equilibrium and zones.get("equilibrium_bottom") is not None and zones.get("equilibrium_top") is not None:
            fig.add_shape(type="rect", x0=zone_x0, x1=zone_x1,
                          y0=zones["equilibrium_bottom"], y1=zones["equilibrium_top"],
                          fillcolor="rgba(128,128,128,0.15)", line=dict(width=0), layer="below")

    # --- Candlesticks ---
    fig.add_trace(go.Candlestick(
        x=df["start_time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Candles",
        increasing_line_color="green", decreasing_line_color="red"
    ))

    # --- EMAs ---
    if "EMA1" in indicators:
        fig.add_trace(go.Scatter(x=df["start_time"], y=df["ema1"], mode="lines",
                                 name=f"EMA1 ({ema1_period})", line=dict(color="blue")))
    if "EMA2" in indicators:
        fig.add_trace(go.Scatter(x=df["start_time"], y=df["ema2"], mode="lines",
                                 name=f"EMA2 ({ema2_period})", line=dict(color="orange")))

    # --- Trades ---
    for _, trade in trades_df.iterrows():
        fig.add_trace(go.Scatter(x=[trade["timestamp"]], y=[trade["entry_price"]], mode="markers",
                                 name=f"{trade['direction']} Trade", showlegend=False,
                                 marker=dict(size=14,
                                             symbol="triangle-up" if trade["direction"] == "BUY" else "triangle-down",
                                             color="green" if trade["direction"] == "BUY" else "red",
                                             line=dict(width=1, color="black"))))

    # --- Swings + HH/LL Markers ---
    if show_swings:
        from services.smc_structure import convert_swings_to_plot_shapes
        for swing in convert_swings_to_plot_shapes(internal_swings + swing_swings):
            if show_swing_labels:
                fig.add_trace(go.Scatter(
                    x=[swing["x"]], y=[swing["y"]],
                    mode="markers+text",
                    marker=dict(symbol="circle", size=9, color=swing["label_color"]),
                    text=[swing["text"]],
                    textfont=dict(size=11, color=swing["label_color"]),
                    textposition="top center",
                    showlegend=False
                ))

            # Dotted vertical line for clarity
            fig.add_shape(
                type="line",
                x0=swing["x"], x1=swing["x"],
                y0=swing["y"], y1=swing["y"] + 0.0001,
                line=dict(color=swing["label_color"], width=1, dash="dot"),
                layer="above"
            )

    # --- BOS / CHoCH ---
    if show_structure:
        for signal in structure_signals:
            fig.add_trace(go.Scatter(
                x=[signal.get("confirmation_time", signal["time"])],
                y=[signal["price"]],
                mode="markers",
                marker=dict(symbol="diamond", size=12,
                            color="green" if signal["direction"] == "bullish" else "red",
                            line=dict(color="black", width=1)),
                name="BOS/CHoCH Confirm",
                showlegend=False
            ))

    # --- Annotate HH/LL sources ---
    if zones and hasattr(zones, "top_source"):
        fig.add_trace(go.Scatter(
            x=[zones.top_source["time"]],
            y=[zones.top_source["price"]],
            mode="markers+text",
            text=["↑ Zone HH"],
            marker=dict(symbol="star", color="red", size=10),
            textposition="top center",
            showlegend=False
        ))
    if zones and hasattr(zones, "bottom_source"):
        fig.add_trace(go.Scatter(
            x=[zones.bottom_source["time"]],
            y=[zones.bottom_source["price"]],
            mode="markers+text",
            text=["↓ Zone LL"],
            marker=dict(symbol="star", color="green", size=10),
            textposition="bottom center",
            showlegend=False
        ))

    # --- Y-axis range ---
    low = df["low"].min()
    high = df["high"].max()
    price_range = high - low
    if price_range < 1e-4 or pd.isna(price_range):
        center = low if pd.notna(low) else 1.0
        y_min, y_max = center * 0.999, center * 1.001
    else:
        padding = price_range * 0.05
        y_min, y_max = low - padding, high + padding

    if zones:
        extra_y = [
            zones.get("discount_bottom"), zones.get("discount_top"),
            zones.get("premium_bottom"), zones.get("premium_top"),
            zones.get("equilibrium_bottom"), zones.get("equilibrium_top")
        ]
        extra_y = [y for y in extra_y if y is not None]
        if extra_y:
            y_min = min(y_min, *extra_y)
            y_max = max(y_max, *extra_y)

    # --- Final Layout ---
    fig.update_layout(
        height=900,
        yaxis=dict(range=[y_min, y_max], autorange=False),
        xaxis=dict(
            type="date",
            autorange=False,
            tickformat="%H:%M\n%b %d",
            showgrid=True,
            rangeslider=dict(visible=False),
            rangebreaks=[dict(pattern="day of week", bounds=[6, 1])]
        ),
        xaxis_title="Time",
        yaxis_title="Price",
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="white"
    )

    return fig
