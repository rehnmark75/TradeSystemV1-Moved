import pandas as pd
import numpy as np
from datetime import timedelta

def detect_swings(df, lookback=2):
    swings = []
    for i in range(lookback, len(df) - lookback):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        
        is_swing_high = all(high > df["high"].iloc[i - j] and high > df["high"].iloc[i + j] for j in range(1, lookback + 1))
        is_swing_low = all(low < df["low"].iloc[i - j] and low < df["low"].iloc[i + j] for j in range(1, lookback + 1))

        if is_swing_high or is_swing_low:
            swings.append({
                "index": i,
                "time": df["start_time"].iloc[i],
                "price": high if is_swing_high else low,
                "type": "high" if is_swing_high else "low"
            })

    return swings

def classify_swings(swings):
    classified = []
    for i in range(1, len(swings)):
        prev = swings[i - 1]
        curr = swings[i]
        if curr["type"] == "high":
            label = "HH" if curr["price"] > prev["price"] else "LH"
        else:
            label = "HL" if curr["price"] > prev["price"] else "LL"
        classified.append({**curr, "label": label})
    return classified


def detect_fvg(df, threshold_percent=0.0, auto=False):
    """
    Detects Fair Value Gaps (FVGs) in a dataframe of candles.
    Based on LuxAlgo's 3-bar pattern logic.

    Parameters:
        df (pd.DataFrame): Must include ['start_time', 'open', 'high', 'low', 'close'] columns.
        threshold_percent (float): Manual threshold in percent. Used if auto=False.
        auto (bool): If True, uses cumulative volatility threshold.

    Returns:
        pd.DataFrame: FVGs with ['index', 'start_time', 'min', 'max', 'isbull']
    """
    df = df.copy().reset_index(drop=True)
    fvgs = []

    if auto:
        df["bar_volatility"] = (df["high"] - df["low"]) / df["low"]
        threshold = df["bar_volatility"].cumsum() / (df.index + 1)
    else:
        threshold = [threshold_percent / 100] * len(df)

    for i in range(2, len(df)):
        bull_condition = (
            df.loc[i, "low"] > df.loc[i - 2, "high"] and
            df.loc[i - 1, "close"] > df.loc[i - 2, "high"] and
            (df.loc[i, "low"] - df.loc[i - 2, "high"]) / df.loc[i - 2, "high"] > threshold[i]
        )

        bear_condition = (
            df.loc[i, "high"] < df.loc[i - 2, "low"] and
            df.loc[i - 1, "close"] < df.loc[i - 2, "low"] and
            (df.loc[i - 2, "low"] - df.loc[i, "high"]) / df.loc[i, "high"] > threshold[i]
        )

        if bull_condition:
            fvgs.append({
                "index": i,
                "start_time": df.loc[i, "start_time"],
                "min": df.loc[i, "low"],
                "max": df.loc[i - 2, "high"],
                "isbull": True
            })

        elif bear_condition:
            fvgs.append({
                "index": i,
                "start_time": df.loc[i, "start_time"],
                "min": df.loc[i - 2, "low"],
                "max": df.loc[i, "high"],
                "isbull": False
            })

    return pd.DataFrame(fvgs)


def get_recent_swing_bounds(df, swings, hours=4):
    end_time = df["start_time"].max()
    start_time = end_time - timedelta(hours=hours)

    recent_swings = [s for s in swings if start_time <= s["time"] <= end_time]

    highs = [s["price"] for s in recent_swings if s["type"] == "high"]
    lows  = [s["price"] for s in recent_swings if s["type"] == "low"]

    if not highs or not lows:
        return None  # Not enough swings in that range

    swing_high = max(highs)
    swing_low = min(lows)

    return {
        "high": swing_high,
        "low": swing_low,
        "equilibrium": (swing_high + swing_low) / 2,
        "premium": swing_low + 0.75 * (swing_high - swing_low),
        "discount": swing_low + 0.25 * (swing_high - swing_low),
        "start_time": start_time,
        "end_time": end_time
    }

def apply_indicators(df: pd.DataFrame, ema1_period: int = 21, ema2_period: int = 50, ema3_period: int = 200, indicators: list = None,
                     ema_short: int = None, ema_long: int = None, ema_trend: int = None, epic: str = None) -> pd.DataFrame:
    """
    Calculate EMAs and MACD with configurable periods.
    Supports both legacy static periods and new dynamic epic-specific periods.

    Args:
        df: DataFrame with OHLC data
        ema1_period: Legacy EMA1 period (default: 21)
        ema2_period: Legacy EMA2 period (default: 50)
        ema3_period: Legacy EMA3 period (default: 200)
        indicators: List of indicators to calculate
        ema_short: Dynamic short EMA period (overrides ema1_period if provided)
        ema_long: Dynamic long EMA period (overrides ema2_period if provided)
        ema_trend: Dynamic trend EMA period (overrides ema3_period if provided)
        epic: Epic code for dynamic configuration lookup

    Returns:
        DataFrame with calculated indicators
    """
    if indicators is None:
        indicators = ["EMA21", "EMA50", "EMA200"]

    # Use dynamic periods if provided, otherwise fall back to legacy periods
    # If epic is provided but dynamic periods are not, try to get them from the config service
    if epic and (ema_short is None or ema_long is None or ema_trend is None):
        try:
            # Use simple service directly since worker files aren't available in container
            from .ema_config_simple import get_ema_periods_for_chart_simple as get_ema_periods_for_chart

            dynamic_short, dynamic_long, dynamic_trend = get_ema_periods_for_chart(epic)
            ema_short = ema_short or dynamic_short
            ema_long = ema_long or dynamic_long
            ema_trend = ema_trend or dynamic_trend
        except Exception as e:
            # Fallback to legacy periods if dynamic lookup fails
            print(f"Warning: Could not get dynamic EMA periods for {epic}: {e}")
            pass

    # Final fallback to legacy parameters
    final_short = ema_short or ema1_period
    final_long = ema_long or ema2_period
    final_trend = ema_trend or ema3_period

    # Calculate EMA short (previously EMA21)
    if "EMA21" in indicators or "EMA_SHORT" in indicators:
        df["ema21"] = df["close"].ewm(span=final_short, adjust=False).mean()
        # Also create generic columns for dynamic usage
        df["ema_short"] = df["ema21"]

    # Calculate EMA long (previously EMA50)
    if "EMA50" in indicators or "EMA_LONG" in indicators:
        df["ema50"] = df["close"].ewm(span=final_long, adjust=False).mean()
        # Also create generic columns for dynamic usage
        df["ema_long"] = df["ema50"]

    # Calculate EMA trend (previously EMA200)
    if "EMA200" in indicators or "EMA_TREND" in indicators:
        df["ema200"] = df["close"].ewm(span=final_trend, adjust=False).mean()
        # Also create generic columns for dynamic usage
        df["ema_trend"] = df["ema200"]

    # Calculate MACD
    if "MACD" in indicators:
        # Standard MACD parameters: 12, 26, 9
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # Keep backward compatibility
    if "EMA1" in indicators and final_short:
        df["ema1"] = df["close"].ewm(span=final_short, adjust=False).mean()

    if "EMA2" in indicators and final_long:
        df["ema2"] = df["close"].ewm(span=final_long, adjust=False).mean()

    return df


def apply_dynamic_indicators(df: pd.DataFrame, epic: str, indicators: list = None) -> pd.DataFrame:
    """
    Convenience function to apply indicators using dynamic epic-specific configuration.

    Args:
        df: DataFrame with OHLC data
        epic: Epic code for configuration lookup
        indicators: List of indicators to calculate (optional)

    Returns:
        DataFrame with calculated indicators using epic-specific parameters
    """
    try:
        # Use simple service directly since worker files aren't available in container
        from .ema_config_simple import get_ema_periods_for_chart_simple as get_ema_periods_for_chart

        ema_short, ema_long, ema_trend = get_ema_periods_for_chart(epic)

        return apply_indicators(
            df=df,
            ema_short=ema_short,
            ema_long=ema_long,
            ema_trend=ema_trend,
            epic=epic,
            indicators=indicators
        )
    except Exception as e:
        # Fallback to legacy behavior
        print(f"Warning: Could not apply dynamic indicators for {epic}: {e}")
        return apply_indicators(df=df, indicators=indicators)


def calculate_two_pole_oscillator(df: pd.DataFrame, filter_length: int = 20, sma_length: int = 25, signal_delay: int = 4) -> pd.DataFrame:
    """
    Calculate Two-Pole Oscillator based on BigBeluga's implementation
    
    Args:
        df: DataFrame with 'close' price data
        filter_length: Length for the two-pole filter (default: 20)
        sma_length: Length for SMA calculation (default: 25)
        signal_delay: Delay for signal line (default: 4)
    
    Returns:
        DataFrame with two-pole oscillator columns added
    """
    if len(df) < max(sma_length * 2, filter_length * 2):
        # Insufficient data, return empty oscillator columns
        df['two_pole_osc'] = 0
        df['two_pole_osc_delayed'] = 0
        df['two_pole_is_green'] = False
        df['two_pole_is_purple'] = False
        return df
    
    # Step 1: Calculate SMA and normalize price deviation
    sma1 = df['close'].rolling(window=sma_length).mean()
    price_deviation = df['close'] - sma1
    
    # Calculate rolling mean and std of the deviation
    deviation_mean = price_deviation.rolling(window=sma_length).mean()
    deviation_std = price_deviation.rolling(window=sma_length).std()
    
    # Normalized deviation
    sma_n1 = (price_deviation - deviation_mean) / deviation_std.replace(0, 1)  # Avoid division by zero
    sma_n1 = sma_n1.fillna(0)
    
    # Step 2: Apply Two-Pole Smoothing Filter
    alpha = 2.0 / (filter_length + 1)
    
    # Initialize arrays for the two-pole filter
    smooth1 = np.full(len(df), np.nan)
    smooth2 = np.full(len(df), np.nan)
    
    # Apply the two-pole filter iteratively
    for i in range(len(df)):
        if i == 0:
            smooth1[i] = sma_n1.iloc[i] if not pd.isna(sma_n1.iloc[i]) else 0
            smooth2[i] = smooth1[i]
        else:
            if not pd.isna(sma_n1.iloc[i]):
                smooth1[i] = (1 - alpha) * smooth1[i-1] + alpha * sma_n1.iloc[i]
            else:
                smooth1[i] = smooth1[i-1]
            
            smooth2[i] = (1 - alpha) * smooth2[i-1] + alpha * smooth1[i]
    
    # Step 3: Create oscillator values
    df['two_pole_osc'] = smooth2
    df['two_pole_osc_delayed'] = df['two_pole_osc'].shift(signal_delay)
    
    # Step 4: Generate oscillator color/direction
    # Green when rising (osc > delayed), Purple when falling (osc <= delayed)
    df['two_pole_is_green'] = df['two_pole_osc'] > df['two_pole_osc_delayed']
    df['two_pole_is_purple'] = df['two_pole_osc'] <= df['two_pole_osc_delayed']
    
    return df


def calculate_zero_lag_ema(df: pd.DataFrame, length: int = 70, band_multiplier: float = 1.2) -> pd.DataFrame:
    """
    Calculate Zero Lag EMA and volatility bands
    
    Zero Lag EMA Formula: ZLEMA = EMA(src + (src - src[lag]), length)
    Where lag = (length - 1) / 2
    
    Args:
        df: DataFrame with price data
        length: Zero Lag EMA period (default: 21)
        band_multiplier: Volatility band multiplier (default: 2.0)
        
    Returns:
        DataFrame with Zero Lag indicators added
    """
    if len(df) < length:
        # Insufficient data
        df['zlema'] = np.nan
        df['zlema_upper'] = np.nan
        df['zlema_lower'] = np.nan
        df['zlema_trend'] = 'neutral'
        return df
    
    # Calculate Zero Lag EMA
    src = df['close']
    
    # Calculate lag as in Pine Script: floor((length - 1) / 2)
    lag = int((length - 1) // 2)
    
    # Zero Lag adjustment: src + (src - src[lag])
    lagged_src = src.shift(lag)
    momentum_adjustment = src - lagged_src
    zlema_input = src + momentum_adjustment
    
    # Calculate Zero Lag EMA using pandas ewm (matches Pine Script ta.ema())
    df['zlema'] = zlema_input.ewm(span=length, adjust=False).mean()
    
    # Calculate ATR for volatility bands
    prev_close = df['close'].shift(1)
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - prev_close).abs()
    low_prev_close = (df['low'] - prev_close).abs()
    
    # True Range
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # ATR using RMA (alpha = 1/length)
    atr_alpha = 1.0 / length
    atr = true_range.ewm(alpha=atr_alpha, adjust=False).mean()
    
    # Volatility bands: highest ATR over length*3 periods
    volatility_lookback = min(length * 3, len(df))
    volatility = atr.rolling(window=volatility_lookback, min_periods=1).max() * band_multiplier
    
    # Upper and lower bands
    df['zlema_upper'] = df['zlema'] + volatility
    df['zlema_lower'] = df['zlema'] - volatility
    
    # Determine trend based on price position relative to ZLEMA
    df['zlema_trend'] = 'neutral'
    df.loc[df['close'] > df['zlema'], 'zlema_trend'] = 'bullish'
    df.loc[df['close'] < df['zlema'], 'zlema_trend'] = 'bearish'
    
    # Add crossover signals
    df['zlema_crossover'] = (df['close'] > df['zlema']) & (df['close'].shift(1) <= df['zlema'].shift(1))
    df['zlema_crossunder'] = (df['close'] < df['zlema']) & (df['close'].shift(1) >= df['zlema'].shift(1))
    
    return df


def calculate_support_resistance(df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, 
                                volume_threshold: float = 20.0, max_levels: int = 5) -> dict:
    """
    Calculate Support and Resistance levels based on pivot highs/lows
    Based on LuxAlgo TradingView script logic
    
    Args:
        df: DataFrame with OHLC and volume data
        left_bars: Number of bars to look left for pivot detection (default: 15)
        right_bars: Number of bars to look right for pivot detection (default: 15)
        volume_threshold: Volume oscillator threshold for break detection (default: 20%)
        max_levels: Maximum number of levels to return per type (default: 5)
        
    Returns:
        Dictionary containing:
        - support_levels: List of support price levels
        - resistance_levels: List of resistance price levels
        - support_breaks: List of support break markers
        - resistance_breaks: List of resistance break markers
    """
    
    # Ensure we have enough data for pivot detection
    min_required = left_bars + right_bars + 1
    if len(df) < min_required:
        return {
            'support_levels': [],
            'resistance_levels': [],
            'support_breaks': [],
            'resistance_breaks': []
        }
    
    # Find pivot highs (resistance levels)
    pivot_highs = []
    for i in range(left_bars, len(df) - right_bars):
        current_high = df.iloc[i]['high']
        
        # Check if current high is highest in the window
        is_pivot_high = True
        
        # Check left bars
        for j in range(i - left_bars, i):
            if df.iloc[j]['high'] >= current_high:
                is_pivot_high = False
                break
        
        # Check right bars
        if is_pivot_high:
            for j in range(i + 1, i + right_bars + 1):
                if df.iloc[j]['high'] >= current_high:
                    is_pivot_high = False
                    break
        
        if is_pivot_high:
            pivot_highs.append({
                'index': i,
                'price': float(current_high),
                'time': df.iloc[i]['start_time']
            })
    
    # Find pivot lows (support levels)
    pivot_lows = []
    for i in range(left_bars, len(df) - right_bars):
        current_low = df.iloc[i]['low']
        
        # Check if current low is lowest in the window
        is_pivot_low = True
        
        # Check left bars
        for j in range(i - left_bars, i):
            if df.iloc[j]['low'] <= current_low:
                is_pivot_low = False
                break
        
        # Check right bars
        if is_pivot_low:
            for j in range(i + 1, i + right_bars + 1):
                if df.iloc[j]['low'] <= current_low:
                    is_pivot_low = False
                    break
        
        if is_pivot_low:
            pivot_lows.append({
                'index': i,
                'price': float(current_low),
                'time': df.iloc[i]['start_time']
            })
    
    # Get unique resistance levels (sorted descending)
    resistance_prices = sorted(list(set([p['price'] for p in pivot_highs])), reverse=True)
    resistance_levels = resistance_prices[:max_levels] if resistance_prices else []
    
    # Get unique support levels (sorted ascending)
    support_prices = sorted(list(set([p['price'] for p in pivot_lows])))
    support_levels = support_prices[:max_levels] if support_prices else []
    
    # Calculate volume oscillator for break detection
    support_breaks = []
    resistance_breaks = []
    
    if 'volume' in df.columns and len(df) >= 10:
        # Calculate volume oscillator: 100 * (short_ema - long_ema) / long_ema
        volume_short_ema = df['volume'].ewm(span=5, adjust=False).mean()
        volume_long_ema = df['volume'].ewm(span=10, adjust=False).mean()
        volume_osc = 100 * (volume_short_ema - volume_long_ema) / volume_long_ema.replace(0, 1)
        
        # Check for breaks with volume confirmation
        for i in range(1, len(df)):
            current_close = df.iloc[i]['close']
            prev_close = df.iloc[i-1]['close']
            current_osc = volume_osc.iloc[i] if not pd.isna(volume_osc.iloc[i]) else 0
            
            # Check for resistance breaks (upward)
            for resistance in resistance_levels:
                if prev_close <= resistance < current_close and current_osc > volume_threshold:
                    # Check for wick condition
                    is_bull_wick = (df.iloc[i]['open'] - df.iloc[i]['low']) > (df.iloc[i]['close'] - df.iloc[i]['open'])
                    resistance_breaks.append({
                        'time': df.iloc[i]['start_time'],
                        'price': resistance,
                        'type': 'bull_wick' if is_bull_wick else 'break',
                        'label': 'Bull Wick' if is_bull_wick else 'B'
                    })
            
            # Check for support breaks (downward)
            for support in support_levels:
                if prev_close >= support > current_close and current_osc > volume_threshold:
                    # Check for wick condition
                    is_bear_wick = (df.iloc[i]['open'] - df.iloc[i]['close']) < (df.iloc[i]['high'] - df.iloc[i]['open'])
                    support_breaks.append({
                        'time': df.iloc[i]['start_time'],
                        'price': support,
                        'type': 'bear_wick' if is_bear_wick else 'break',
                        'label': 'Bear Wick' if is_bear_wick else 'B'
                    })
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'support_breaks': support_breaks,
        'resistance_breaks': resistance_breaks,
        'pivot_highs': pivot_highs,
        'pivot_lows': pivot_lows
    }