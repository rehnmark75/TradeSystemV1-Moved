# smc_structure.py (restored and patched to use strongest HH/LL for zones)

from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import timedelta

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


def get_structure_signals(pivots):
    """
    Filters and returns only the BOS/CHoCH-confirmed swing pivots.
    Expects input as a list of pivot dictionaries with keys: 'price', 'time', 'type', 'label'.
    """
    confirmed = []
    
    for i, pivot in enumerate(pivots):
        if not isinstance(pivot, dict):
            continue  # skip if pivot is not a dictionary

        label = pivot.get("label")
        if label in ("HH", "LL"):
            confirmed.append(pivot)

    return confirmed


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


def get_recent_trailing_extremes(swings: List[Dict], window: int = 50) -> Optional[Dict[str, Any]]:
    if not swings:
        return None

    swings = sorted(swings, key=lambda x: x["time"])
    recent = swings[-window:] if len(swings) > window else swings

    hh_swings = [s for s in recent if s["label"] == "HH"]
    ll_swings = [s for s in recent if s["label"] == "LL"]

    if not hh_swings or not ll_swings:
        return None

    # Pick the highest HH and lowest LL, regardless of time
    top_source = max(hh_swings, key=lambda s: s["price"])
    bottom_source = min(ll_swings, key=lambda s: s["price"])

    return {
        "top": top_source["price"],
        "bottom": bottom_source["price"],
        "top_source": top_source,
        "bottom_source": bottom_source,
        "bar_time": bottom_source["time"]
    }




def calculate_zones(top: float, bottom: float) -> dict:
    """
    Calculate Premium, Discount, and Equilibrium zones based on top/bottom swing extremes.
    Zones are symmetric around the midpoint for visual consistency with the chart logic.
    """
    # Ensure correct ordering
    top, bottom = max(top, bottom), min(top, bottom)

    # Midpoint (equilibrium center)
    mid = (top + bottom) / 2

    # Zone width is 1/6 of the full range
    zone_width = (top - bottom) / 6

    return {
        "premium_top": top,
        "premium_bottom": mid + zone_width,
        "equilibrium_top": mid + zone_width,
        "equilibrium_bottom": mid - zone_width,
        "discount_top": mid - zone_width,
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

def detect_swing_pivots(df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    swings = []
    for i in range(lookback, len(df) - lookback):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        is_high = high == df["high"].iloc[i - lookback:i + lookback + 1].max()
        is_low = low == df["low"].iloc[i - lookback:i + lookback + 1].min()

        if is_high:
            swings.append({
                "type": "high",
                "label": "HH",  # <- This is what you want
                "price": high,
                "time": df["start_time"].iloc[i]
            })

        elif is_low:
            swings.append({
                "type": "low",
                "label": "LL",  # <- This is what you want
                "price": low,
                "time": df["start_time"].iloc[i]
            })

    return swings



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

def get_most_recent_confirmed_hh_ll(swings: list[dict], bar_limit: int = 96, debug: bool = False) -> Optional[dict]:
    if not swings:
        if debug: print("❌ No swings provided.")
        return None

    confirmed_swings = [s for s in swings if s.get("confirmed") and s["label"] in ("HH", "LL")]
    recent_swings = confirmed_swings[-bar_limit:]

    if debug:
        print(f"🔍 Found {len(recent_swings)} confirmed swings in last {bar_limit} bars")

    latest_ll = None
    for swing in reversed(recent_swings):
        if swing["label"] == "LL":
            latest_ll = swing
            break

    if not latest_ll:
        if debug: print("❌ No confirmed LL found in window.")
        return None

    latest_hh_before_ll = None
    for swing in reversed(recent_swings):
        if swing["time"] >= latest_ll["time"]:
            continue
        if swing["label"] == "HH" and swing["price"] > latest_ll["price"]:
            latest_hh_before_ll = swing
            break

    if latest_hh_before_ll and latest_ll:
        if debug: print("✅ Valid HH/LL pair found.")
        return {
            "top": latest_hh_before_ll["price"],
            "bottom": latest_ll["price"],
            "top_source": latest_hh_before_ll,
            "bottom_source": latest_ll
        }

    # 🔁 Fallback logic if no valid HH > LL found
    fallback_hh = None
    for swing in reversed(recent_swings):
        if swing["label"] == "HH" and swing["time"] < latest_ll["time"]:
            fallback_hh = swing
            break

    if fallback_hh:
        if debug:
            print("⚠️ Using fallback HH/LL pair (HH not > LL):")
            print(f"  HH: {fallback_hh['price']} @ {fallback_hh['time']}")
            print(f"  LL: {latest_ll['price']} @ {latest_ll['time']}")
        return {
            "top": fallback_hh["price"],
            "bottom": latest_ll["price"],
            "top_source": fallback_hh,
            "bottom_source": latest_ll
        }

    if debug: print("❌ No HH before LL found at all.")
    return None

def get_most_recent_hh_ll_from_swings(swings, bar_limit=96, debug=False):
    recent_swings = swings[-bar_limit:]

    hh = next((s for s in reversed(recent_swings) if s["label"] == "HH"), None)
    ll = next((s for s in reversed(recent_swings) if s["label"] == "LL"), None)

    if hh and ll:
        if debug:
            print("✅ Using raw HH/LL from swing pivots (not confirmed):")
            print(f"  HH: {hh['price']} @ {hh['time']}")
            print(f"  LL: {ll['price']} @ {ll['time']}")
        return {
            "top": hh["price"],
            "bottom": ll["price"],
            "top_source": hh,
            "bottom_source": ll,
            "is_fallback": True
        }
    else:
        if debug:
            print("❌ No usable raw HH/LL found in last swings.")
        return None

def mark_confirmed_swings(pivots, signals, price_tol=0.05, time_tol=pd.Timedelta(minutes=30)):
    for pivot in pivots:
        pivot_time = pd.to_datetime(pivot["time"])
        pivot_price = float(pivot["price"])
        pivot["confirmed"] = False

        for signal in signals:
            signal_price = float(signal["price"])
            from_time = pd.to_datetime(signal.get("from_time", signal["time"]))

            # Compare to signal.from_time instead of signal.time
            if pivot["label"] in ["HH", "LL"]:
                time_diff = abs(pivot_time - from_time)
                price_diff = abs(pivot_price - signal_price)

                if price_diff <= price_tol and time_diff <= time_tol:
                    pivot["confirmed"] = True
                    break
    return pivots

def calculate_premium_discount_zones(swings, lookback_bars=96, debug=False):
    """Calculate premium/discount/equilibrium zones from confirmed swings."""
    recent_swings = swings[-lookback_bars:]

    confirmed_swings = [s for s in recent_swings if s.get("confirmed") and s["label"] in ("HH", "LL")]
    if debug:
        print(f"🔍 Found {len(confirmed_swings)} confirmed swings in last {lookback_bars} bars")

    latest_ll = next((s for s in reversed(confirmed_swings) if s["label"] == "LL"), None)
    hh_before_ll = next((s for s in reversed(confirmed_swings) if s["label"] == "HH" and s["time"] < latest_ll["time"] and s["price"] > latest_ll["price"]), None) if latest_ll else None

    if not latest_ll or not hh_before_ll:
        if debug:
            print("⚠️ No valid HH > LL found — using fallback (last HH + LL regardless of price)")
        latest_ll = next((s for s in reversed(confirmed_swings) if s["label"] == "LL"), None)
        hh_before_ll = next((s for s in reversed(confirmed_swings) if s["label"] == "HH" and s["time"] < latest_ll["time"]), None)

    if not latest_ll or not hh_before_ll:
        if debug:
            print("❌ Unable to calculate zones — missing HH or LL.")
        return None

    top = hh_before_ll["price"]
    bottom = latest_ll["price"]

    if debug:
        print(f"📍 HH used for zone (top):\n  Time: {hh_before_ll['time']}\n  Price: {top}")
        print(f"📍 LL used for zone (bottom):\n  Time: {latest_ll['time']}\n  Price: {bottom}")

    premium = 0.95 * top + 0.05 * bottom
    discount = 0.95 * bottom + 0.05 * top
    equilibrium = (top + bottom) / 2

    return {
        "top": top,
        "bottom": bottom,
        "premium": premium,
        "discount": discount,
        "equilibrium": equilibrium,
        "hh": hh_before_ll,
        "ll": latest_ll
    }


def determine_internal_trend(signals, debug=False):
    """
    Return latest internal structure trend bias:
    +1 = bullish (internal BOS or CHoCH)
    -1 = bearish (internal BOS or CHoCH)
     0 = no internal structure signal
    """
    for s in reversed(signals):
        if s["label"] in ("BOS", "CHoCH") and s["from_time"] and s["to_time"]:
            if s["direction"] == "bullish":
                if debug:
                    print(f"🟢 Internal bullish structure: {s['label']} at {s['time']}")
                return 1
            elif s["direction"] == "bearish":
                if debug:
                    print(f"🔴 Internal bearish structure: {s['label']} at {s['time']}")
                return -1
    return 0


def select_recent_hh_ll_structure(swings, max_age_minutes=1440, bar_limit=96, now=None, debug=False):
    """
    Tries to return the most recent valid HH/LL pair within time and bar constraints.
    Falls back from confirmed to raw swings if needed.
    """
    now = now or pd.Timestamp.utcnow() 
    time_cutoff = now - timedelta(minutes=max_age_minutes)

    # Step 1: Try confirmed HH/LL within time + bar window
    confirmed = [s for s in swings if s.get("confirmed") and s["label"] in ("HH", "LL")]
    recent_confirmed = confirmed[-bar_limit:] if len(confirmed) > bar_limit else confirmed

    hh = next((s for s in reversed(recent_confirmed) if s["label"] == "HH" and s["time"] >= time_cutoff), None)
    ll = next((s for s in reversed(recent_confirmed) if s["label"] == "LL" and s["time"] >= time_cutoff), None)

    if hh and ll and hh["price"] > ll["price"]:
        if debug:
            print("✅ Using confirmed HH/LL within 24h:")
            print(f"  HH: {hh['price']} @ {hh['time']}")
            print(f"  LL: {ll['price']} @ {ll['time']}")
        return {
            "top": hh["price"],
            "bottom": ll["price"],
            "top_source": hh,
            "bottom_source": ll,
            "is_fallback": False
        }

    # Step 2: Try raw HH/LL within time window
    raw = [s for s in swings if s["label"] in ("HH", "LL")]
    recent_raw = raw[-bar_limit:] if len(raw) > bar_limit else raw

    hh_raw = next((s for s in reversed(recent_raw) if s["label"] == "HH" and s["time"] >= time_cutoff), None)
    ll_raw = next((s for s in reversed(recent_raw) if s["label"] == "LL" and s["time"] >= time_cutoff), None)

    if hh_raw and ll_raw and hh_raw["price"] > ll_raw["price"]:
        if debug:
            print("✅ Using raw HH/LL within 24h (no confirmation):")
            print(f"  HH: {hh_raw['price']} @ {hh_raw['time']}")
            print(f"  LL: {ll_raw['price']} @ {ll_raw['time']}")
        return {
            "top": hh_raw["price"],
            "bottom": ll_raw["price"],
            "top_source": hh_raw,
            "bottom_source": ll_raw,
            "is_fallback": True
        }

    if debug:
        print("❌ No valid HH > LL found in last 24h — skipping zone creation")
    return None