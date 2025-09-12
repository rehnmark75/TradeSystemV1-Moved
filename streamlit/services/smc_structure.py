# smc_structure.py (restored and patched to use strongest HH/LL for zones)

from typing import List, Dict, Optional
import pandas as pd

class Pivot:
    def __init__(self, index: int, time: pd.Timestamp, price: float, type_: str):
        self.index = index
        self.time = time
        self.price = price
        self.type = type_  # 'high' or 'low'

class TrailingExtremes:
    def __init__(self):
        self.top = None
        self.bottom = None
        self.bar_time = None
        self.bar_index = None
        self.last_top_time = None
        self.last_bottom_time = None


def detect_pivots(df: pd.DataFrame, lookback: int) -> List[Pivot]:
    df = df.reset_index(drop=True)
    pivots = []
    for i in range(lookback, len(df) - lookback):
        high = df.loc[i, "high"]
        low = df.loc[i, "low"]
        is_swing_high = all(high > df.loc[i - j, "high"] and high > df.loc[i + j, "high"] for j in range(1, lookback + 1))
        is_swing_low = all(low < df.loc[i - j, "low"] and low < df.loc[i + j, "low"] for j in range(1, lookback + 1))

        if is_swing_high:
            pivots.append(Pivot(i, df.loc[i, "start_time"], high, "high"))
        elif is_swing_low:
            pivots.append(Pivot(i, df.loc[i, "start_time"], low, "low"))
    return pivots


def classify_pivots(pivots: List[Pivot]) -> List[Dict]:
    classified = []
    last_high = None
    last_low = None

    for p in pivots:
        label = None
        if p.type == "high":
            if last_high is None:
                label = "HH"  # Treat first high as structure high
            elif p.price > last_high.price:
                label = "HH"
            else:
                label = "LH"
            last_high = p

        elif p.type == "low":
            if last_low is None:
                label = "LL"  # Fix: treat first low as structure low
            elif p.price < last_low.price:
                label = "LL"
            else:
                label = "HL"
            last_low = p

        classified.append({
            "index": p.index,
            "time": p.time,
            "price": p.price,
            "type": p.type,
            "label": label
        })

    return classified


def get_recent_trailing_extremes(swings: List[Dict], last_top=None, last_bottom=None) -> Optional[Dict]:
    # Always find the strongest ones in history as fallback
    strongest_hh = max((s for s in swings if s["label"] == "HH"), key=lambda x: x["price"], default=None)
    weakest_ll = min((s for s in swings if s["label"] == "LL"), key=lambda x: x["price"], default=None)

    if not strongest_hh or not weakest_ll:
        return None

    # Compare with last tracked
    should_update_top = last_top is None or strongest_hh["price"] > last_top["price"]
    should_update_bottom = last_bottom is None or weakest_ll["price"] < last_bottom["price"]

    new_top = strongest_hh if should_update_top else last_top
    new_bottom = weakest_ll if should_update_bottom else last_bottom

    return {
        "top": new_top["price"],
        "bottom": new_bottom["price"],
        "bar_time": max(new_top["time"], new_bottom["time"]),
        "top_source": new_top,
        "bottom_source": new_bottom
    }



def calculate_zones(top: float, bottom: float) -> Dict[str, float]:
    # Equilibrium levels
    equilibrium_top = 0.525 * top + 0.475 * bottom
    equilibrium_bottom = 0.525 * bottom + 0.475 * top

    return {
        "premium_top": top,
        "premium_bottom": equilibrium_top,
        "equilibrium_top": equilibrium_top,
        "equilibrium_bottom": equilibrium_bottom,
        "discount_top": equilibrium_bottom,
        "discount_bottom": bottom
    }


def convert_swings_to_plot_shapes(swings: List[Dict]) -> List[Dict]:
    shapes = []
    for s in swings:
        shapes.append({
            "x": s["time"],
            "y": s["price"],
            "text": s["label"],
            "type": s["type"],
            "label_color": "green" if s["type"] == "low" else "red"
        })
    return shapes


def determine_bias(signals: List[Dict]) -> Optional[int]:
    for s in reversed(signals):
        if s["label"] == "BOS":
            return 1 if s["direction"] == "bullish" else -1
        elif s["label"] == "CHoCH":
            return 1 if s["direction"] == "bullish" else -1
    return None


def detect_structure_signals_luxalgo(swings: List[Dict], df: pd.DataFrame) -> List[Dict]:
    signals = []
    trend_bias = None

    for i in range(1, len(swings)):
        prev = swings[i - 1]
        curr = swings[i]

        if prev["type"] != curr["type"]:
            continue

        segment = df[(df["start_time"] > prev["time"]) & (df["start_time"] <= curr["time"])]
        confirmation_row = None

        if curr["type"] == "high":
            confirmation_row = segment[segment["close"] > prev["price"]].head(1)
        elif curr["type"] == "low":
            confirmation_row = segment[segment["close"] < prev["price"]].head(1)

        if confirmation_row is None or confirmation_row.empty:
            continue

        confirm_time = confirmation_row.iloc[0]["start_time"]
        label = "BOS" if trend_bias != (-1 if curr["type"] == "high" else 1) else "CHoCH"
        direction = "bullish" if curr["type"] == "high" else "bearish"
        trend_bias = 1 if direction == "bullish" else -1

        signals.append({
            "label": label,
            "direction": direction,
            "price": prev["price"],
            "time": confirm_time,
            "from_time": prev["time"],
            "to_time": curr["time"],
            "confirmation_time": confirm_time
        })

    return signals
