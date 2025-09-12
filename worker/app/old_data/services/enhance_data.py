# enhance_data.py

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text

def fetch_candle_data(engine, epic, timeframe=15, lookback_hours=500):
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    source_tf = 5 if timeframe == 15 else timeframe
    
    query = text("""
        SELECT start_time, open, high, low, close, ltv
        FROM ig_candles
        WHERE epic = :epic
        AND timeframe = :timeframe
        AND start_time >= :since
        ORDER BY start_time ASC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {
            "epic": epic,
            "timeframe": source_tf,
            "since": since
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if df.empty:
        raise ValueError(f"No data returned for epic={epic}, timeframe={source_tf}")
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    if timeframe == 15 and source_tf == 5:
        df.set_index("start_time", inplace=True)
        df = df.resample("15min", label='right', closed='right').agg({
            'open': 'first', 
            'high': 'max', 
            'low': 'min', 
            'close': 'last',
            'ltv': 'sum'  # Sum volume over the resampled period
        }).dropna().reset_index()
    
    return df.reset_index(drop=True)

def find_support_resistance_levels(df, window=20, min_touches=2):
    """
    Find support and resistance levels using local extrema and clustering
    
    Args:
        df: DataFrame with OHLC data
        window: Period for finding local extrema
        min_touches: Minimum number of touches to confirm a level
    
    Returns:
        dict with support and resistance levels
    """
    # Find local maxima (potential resistance) and minima (potential support)
    highs = df['high'].values
    lows = df['low'].values
    
    # Find local extrema
    resistance_indices = argrelextrema(highs, np.greater, order=window)[0]
    support_indices = argrelextrema(lows, np.less, order=window)[0]
    
    resistance_levels = highs[resistance_indices]
    support_levels = lows[support_indices]
    
    # Cluster nearby levels (within 0.1% of price)
    def cluster_levels(levels, tolerance_pct=0.001):
        if len(levels) == 0:
            return []
        
        levels = np.sort(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance_pct:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    resistance_clusters = cluster_levels(resistance_levels)
    support_clusters = cluster_levels(support_levels)
    
    return {
        'resistance_levels': sorted(resistance_clusters, reverse=True),
        'support_levels': sorted(support_clusters, reverse=True)
    }

def calculate_weekly_daily_levels(df):
    """
    Calculate weekly and daily high/low levels
    """
    df_copy = df.copy()
    df_copy['date'] = df_copy['start_time'].dt.date
    df_copy['week'] = df_copy['start_time'].dt.isocalendar().week
    df_copy['year'] = df_copy['start_time'].dt.year
    
    # Daily levels
    daily_levels = df_copy.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).rename(columns={'high': 'daily_high', 'low': 'daily_low'})
    
    # Weekly levels
    weekly_levels = df_copy.groupby(['year', 'week']).agg({
        'high': 'max',
        'low': 'min'
    }).rename(columns={'high': 'weekly_high', 'low': 'weekly_low'})
    
    return daily_levels, weekly_levels

def add_support_resistance_to_df(df, epic_pair='EURUSD', volume_periods=[10, 20, 50]):
    """
    Add support/resistance levels, key price levels, and volume analysis to dataframe
    
    Args:
        df: DataFrame with OHLC data and ltv (volume) column
        epic_pair: Trading pair for pip calculation (default EURUSD)
        volume_periods: List of periods for volume moving averages (default [10, 20, 50])
    
    Returns:
        Enhanced DataFrame with support/resistance and volume data
    """
    df_enhanced = df.copy()
    
    # Calculate volume moving averages if ltv column exists
    if 'ltv' in df_enhanced.columns:
        # Handle NaN values in volume data
        df_enhanced['ltv'] = df_enhanced['ltv'].fillna(0)
        
        # Calculate moving averages for different periods
        for period in volume_periods:
            df_enhanced[f'volume_avg_{period}'] = df_enhanced['ltv'].rolling(window=period, min_periods=1).mean()
        
        # Calculate volume ratios (current volume vs averages)
        for period in volume_periods:
            avg_col = f'volume_avg_{period}'
            ratio_col = f'volume_ratio_{period}'
            df_enhanced[ratio_col] = df_enhanced['ltv'] / df_enhanced[avg_col].replace(0, np.nan)
        
        # Add volume percentile ranking (relative volume strength)
        df_enhanced['volume_percentile_50'] = df_enhanced['ltv'].rolling(window=50, min_periods=1).rank(pct=True) * 100
    
    # Determine pip value based on pair
    if 'JPY' in epic_pair.upper():
        pip_multiplier = 100  # JPY pairs: 1 pip = 0.01
    else:
        pip_multiplier = 10000  # Most pairs: 1 pip = 0.0001
    
    # Find support/resistance levels
    sr_levels = find_support_resistance_levels(df_enhanced)
    
    # Calculate daily/weekly levels
    daily_levels, weekly_levels = calculate_weekly_daily_levels(df_enhanced)
    
    # Prepare lists for new columns
    nearest_resistance = []
    nearest_support = []
    distance_to_resistance_pips = []
    distance_to_support_pips = []
    risk_reward_ratio = []
    weekly_highs = []
    weekly_lows = []
    daily_highs = []
    daily_lows = []
    
    for idx, row in df_enhanced.iterrows():
        current_price = row['close']
        current_date = row['start_time'].date()
        current_week = row['start_time'].isocalendar().week
        current_year = row['start_time'].year
        
        # Find nearest resistance (above current price)
        resistance_above = [r for r in sr_levels['resistance_levels'] if r > current_price]
        nearest_res = min(resistance_above) if resistance_above else None
        
        # Find nearest support (below current price)
        support_below = [s for s in sr_levels['support_levels'] if s < current_price]
        nearest_sup = max(support_below) if support_below else None
        
        # Calculate distances in pips
        if nearest_res:
            dist_to_res = (nearest_res - current_price) * pip_multiplier
        else:
            dist_to_res = np.nan
            
        if nearest_sup:
            dist_to_sup = (current_price - nearest_sup) * pip_multiplier
        else:
            dist_to_sup = np.nan
        
        # Calculate risk/reward ratio
        if not np.isnan(dist_to_res) and not np.isnan(dist_to_sup) and dist_to_sup != 0:
            rr_ratio = dist_to_res / dist_to_sup
        else:
            rr_ratio = np.nan
        
        # Get weekly levels
        try:
            week_high = weekly_levels.loc[(current_year, current_week), 'weekly_high']
            week_low = weekly_levels.loc[(current_year, current_week), 'weekly_low']
        except KeyError:
            week_high = np.nan
            week_low = np.nan
        
        # Get daily levels
        try:
            day_high = daily_levels.loc[current_date, 'daily_high']
            day_low = daily_levels.loc[current_date, 'daily_low']
        except KeyError:
            day_high = np.nan
            day_low = np.nan
        
        # Append to lists
        nearest_resistance.append(nearest_res)
        nearest_support.append(nearest_sup)
        distance_to_resistance_pips.append(dist_to_res)
        distance_to_support_pips.append(dist_to_sup)
        risk_reward_ratio.append(rr_ratio)
        weekly_highs.append(week_high)
        weekly_lows.append(week_low)
        daily_highs.append(day_high)
        daily_lows.append(day_low)
    
    # Add new columns to dataframe
    df_enhanced['nearest_resistance'] = nearest_resistance
    df_enhanced['nearest_support'] = nearest_support
    df_enhanced['distance_to_resistance_pips'] = distance_to_resistance_pips
    df_enhanced['distance_to_support_pips'] = distance_to_support_pips
    df_enhanced['risk_reward_ratio'] = risk_reward_ratio
    df_enhanced['weekly_high'] = weekly_highs
    df_enhanced['weekly_low'] = weekly_lows
    df_enhanced['daily_high'] = daily_highs
    df_enhanced['daily_low'] = daily_lows
    
    return df_enhanced

# Usage example:
def enhance_candle_data(engine, epic, epic_pair='EURUSD'):
    """
    Complete function to fetch and enhance candle data with support/resistance levels
    """
    # Fetch data
    df_5m = fetch_candle_data(engine, epic, 5, 1000)
    df_15m = fetch_candle_data(engine, epic, 15, 1000)
    df_1h = fetch_candle_data(engine, epic, 60, lookback_hours=200)
    
    # Add support/resistance analysis
    df_5m_enhanced = add_support_resistance_to_df(df_5m, epic_pair)
    df_15m_enhanced = add_support_resistance_to_df(df_15m, epic_pair)
    df_1h_enhanced = add_support_resistance_to_df(df_1h, epic_pair)
    
    return df_5m_enhanced, df_15m_enhanced, df_1h_enhanced

# Example usage:
# df_5m_enhanced, df_15m_enhanced, df_1h_enhanced = enhance_candle_data(engine, 'CS.D.EURUSD.MINI.IP', 'EURUSD')

def add_market_behavior_analysis(df_enhanced, epic_pair='EURUSD'):
    """
    Add market behavior and momentum analysis to the enhanced dataframe
    
    Args:
        df_enhanced: DataFrame that already has basic enhancement
        epic_pair: Trading pair for pip calculation
    
    Returns:
        DataFrame with additional market behavior indicators
    """
    df_behavior = df_enhanced.copy()
    
    # Determine pip value based on pair
    if 'JPY' in epic_pair.upper():
        pip_multiplier = 100  # JPY pairs: 1 pip = 0.01
    else:
        pip_multiplier = 10000  # Most pairs: 1 pip = 0.0001
    
    # Calculate price changes in pips for different timeframes
    # Note: These are approximate since we're using single timeframe data
    df_behavior['price_change_1_bar_pips'] = (df_behavior['close'] - df_behavior['close'].shift(1)) * pip_multiplier
    df_behavior['price_change_4_bars_pips'] = (df_behavior['close'] - df_behavior['close'].shift(4)) * pip_multiplier
    df_behavior['price_change_12_bars_pips'] = (df_behavior['close'] - df_behavior['close'].shift(12)) * pip_multiplier
    
    # Identify green and red candles
    df_behavior['is_green'] = df_behavior['close'] > df_behavior['open']
    df_behavior['is_red'] = df_behavior['close'] < df_behavior['open']
    df_behavior['is_doji'] = abs(df_behavior['close'] - df_behavior['open']) < (df_behavior['high'] - df_behavior['low']) * 0.1
    
    # Calculate consecutive candles
    def calculate_consecutive_candles(series):
        consecutive_counts = []
        current_count = 0
        current_type = None
        
        for value in series:
            if value == current_type:
                current_count += 1
            else:
                current_count = 1 if value else 0
                current_type = value
            consecutive_counts.append(current_count if value else 0)
        
        return consecutive_counts
    
    df_behavior['consecutive_green_candles'] = calculate_consecutive_candles(df_behavior['is_green'])
    df_behavior['consecutive_red_candles'] = calculate_consecutive_candles(df_behavior['is_red'])
    
    # Calculate rejection wicks (long upper/lower shadows)
    df_behavior['body_size'] = abs(df_behavior['close'] - df_behavior['open'])
    df_behavior['upper_wick'] = df_behavior['high'] - df_behavior[['open', 'close']].max(axis=1)
    df_behavior['lower_wick'] = df_behavior[['open', 'close']].min(axis=1) - df_behavior['low']
    df_behavior['total_range'] = df_behavior['high'] - df_behavior['low']
    
    # Define rejection wicks (wick > 50% of total range and > 2x body size)
    df_behavior['upper_rejection'] = (
        (df_behavior['upper_wick'] > df_behavior['total_range'] * 0.5) & 
        (df_behavior['upper_wick'] > df_behavior['body_size'] * 2)
    )
    df_behavior['lower_rejection'] = (
        (df_behavior['lower_wick'] > df_behavior['total_range'] * 0.5) & 
        (df_behavior['lower_wick'] > df_behavior['body_size'] * 2)
    )
    
    # Count rejection wicks in recent periods
    def rolling_rejection_count(df, window=10):
        rejection_counts = []
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            upper_rejections = df['upper_rejection'].iloc[start_idx:i+1].sum()
            lower_rejections = df['lower_rejection'].iloc[start_idx:i+1].sum()
            rejection_counts.append(upper_rejections + lower_rejections)
        return rejection_counts
    
    df_behavior['rejection_wicks_count'] = rolling_rejection_count(df_behavior)
    
    # Calculate consolidation range (high-low range over recent periods)
    def calculate_consolidation_range(df, window=20, pip_multiplier=10000):
        ranges = []
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            period_high = df['high'].iloc[start_idx:i+1].max()
            period_low = df['low'].iloc[start_idx:i+1].min()
            range_pips = (period_high - period_low) * pip_multiplier
            ranges.append(range_pips)
        return ranges
    
    df_behavior['consolidation_range_pips'] = calculate_consolidation_range(df_behavior, pip_multiplier=pip_multiplier)
    
    # Breakout detection (simplified)
    def detect_breakouts(df, window=20):
        breakout_bars = []
        for i in range(len(df)):
            if i < window:
                breakout_bars.append(0)
                continue
                
            # Look back to find recent range
            start_idx = max(0, i - window)
            recent_high = df['high'].iloc[start_idx:i].max()
            recent_low = df['low'].iloc[start_idx:i].min()
            
            # Check if current bar breaks out
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            if current_high > recent_high or current_low < recent_low:
                # This is a breakout bar, count bars since
                bars_since = 1
            else:
                # Count bars since last breakout
                bars_since = breakout_bars[-1] + 1 if breakout_bars else 1
                
            breakout_bars.append(bars_since)
        
        return breakout_bars
    
    df_behavior['bars_since_breakout'] = detect_breakouts(df_behavior)
    
    # Bars at current level (price staying within small range)
    def bars_at_level(df, tolerance_pips=5, pip_multiplier=10000):
        level_bars = []
        for i in range(len(df)):
            if i == 0:
                level_bars.append(1)
                continue
                
            current_price = df['close'].iloc[i]
            bars_at_current = 1
            
            # Look backwards to count bars within tolerance
            for j in range(i-1, -1, -1):
                past_price = df['close'].iloc[j]
                if abs(current_price - past_price) * pip_multiplier <= tolerance_pips:
                    bars_at_current += 1
                else:
                    break
                    
            level_bars.append(bars_at_current)
        
        return level_bars
    
    df_behavior['bars_at_current_level'] = bars_at_level(df_behavior, pip_multiplier=pip_multiplier)
    
def add_multi_timeframe_analysis(df_5m, df_15m, df_1h, epic_pair='EURUSD'):
    """
    Add multi-timeframe trend analysis and key level breaks to dataframes
    
    Args:
        df_5m, df_15m, df_1h: Enhanced dataframes for different timeframes
        epic_pair: Trading pair for analysis
    
    Returns:
        Tuple of enhanced dataframes with multi-timeframe analysis
    """
    
    # Validate input dataframes
    if df_5m is None:
        raise ValueError("df_5m is None - check your 5m data fetch")
    if df_15m is None:
        raise ValueError("df_15m is None - check your 15m data fetch")
    if df_1h is None:
        raise ValueError("df_1h is None - check your 1h data fetch")
    
    if len(df_5m) == 0:
        raise ValueError("df_5m is empty - no 5m data available")
    if len(df_15m) == 0:
        raise ValueError("df_15m is empty - no 15m data available")
    if len(df_1h) == 0:
        raise ValueError("df_1h is empty - no 1h data available")
    
    def determine_trend(df, period=20):
        """
        Determine trend direction using multiple indicators
        """
        if df is None or len(df) < 2:  # Need at least 2 bars
            return 'ranging'
        
        # Adjust period if dataframe is too small
        actual_period = min(period, len(df))
        if actual_period < 2:
            return 'ranging'
        
        # Use last 'actual_period' bars for trend analysis
        recent_data = df.tail(actual_period).copy()
        
        # Method 1: Price vs Moving Average
        ma_window_short = max(1, actual_period//4)
        ma_window_long = max(2, actual_period//2)
        
        recent_data['ma_short'] = recent_data['close'].rolling(window=ma_window_short, min_periods=1).mean()
        recent_data['ma_long'] = recent_data['close'].rolling(window=ma_window_long, min_periods=1).mean()
        
        current_price = recent_data['close'].iloc[-1]
        ma_short = recent_data['ma_short'].iloc[-1]
        ma_long = recent_data['ma_long'].iloc[-1]
        
        # Method 2: Higher highs/Lower lows
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Check if making higher highs and higher lows (bullish)
        lookback_recent = min(5, len(highs)//2)
        lookback_previous = min(10, len(highs))
        
        if len(highs) >= lookback_recent:
            recent_high = np.max(highs[-lookback_recent:])
            recent_low = np.min(lows[-lookback_recent:])
        else:
            recent_high = highs[-1]
            recent_low = lows[-1]
        
        if len(highs) >= lookback_previous:
            previous_high = np.max(highs[-lookback_previous:-lookback_recent]) if lookback_previous > lookback_recent else highs[0]
            previous_low = np.min(lows[-lookback_previous:-lookback_recent]) if lookback_previous > lookback_recent else lows[0]
        else:
            previous_high = highs[0]
            previous_low = lows[0]
        
        # Method 3: Slope of price action
        if len(recent_data) >= 3:
            x = np.arange(len(recent_data))
            slope = np.polyfit(x, recent_data['close'].values, 1)[0]
        else:
            slope = 0
        
        # Combine indicators for trend determination
        bullish_signals = 0
        bearish_signals = 0
        
        # MA signals
        if current_price > ma_short > ma_long:
            bullish_signals += 2
        elif current_price < ma_short < ma_long:
            bearish_signals += 2
        
        # Structure signals
        if recent_high > previous_high and recent_low > previous_low:
            bullish_signals += 2
        elif recent_high < previous_high and recent_low < previous_low:
            bearish_signals += 2
        
        # Slope signals (adjust threshold based on price level)
        price_threshold = current_price * 0.0001  # Dynamic threshold
        if slope > price_threshold:
            bullish_signals += 1
        elif slope < -price_threshold:
            bearish_signals += 1
        
        # Determine final trend
        if bullish_signals >= 3:
            return 'bullish'
        elif bearish_signals >= 3:
            return 'bearish'
        else:
            return 'ranging'
    
    def detect_level_breaks(df, lookback=20):
        """
        Detect if key levels have been broken or held
        """
        if len(df) < lookback:
            return {'broke_resistance': False, 'held_support': True, 'retesting_level': False}
        
        recent_data = df.tail(lookback).copy()
        
        # Find recent support/resistance levels
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        recent_range = recent_high - recent_low
        
        # Get current and previous bars
        current_bar = recent_data.iloc[-1]
        previous_bars = recent_data.iloc[-5:]  # Last 5 bars
        
        # Detect resistance break (price closes above recent resistance)
        resistance_level = recent_data['high'].rolling(window=10).max().iloc[-6]  # Resistance from 6+ bars ago
        broke_resistance = current_bar['close'] > resistance_level
        
        # Detect support hold (price stays above recent support)
        support_level = recent_data['low'].rolling(window=10).min().iloc[-6]  # Support from 6+ bars ago
        held_support = current_bar['low'] > support_level
        
        # Detect retesting (price comes back to test a level)
        # Check if price is within 20% of recent high or low
        distance_to_high = abs(current_bar['close'] - recent_high) / recent_range if recent_range > 0 else 0
        distance_to_low = abs(current_bar['close'] - recent_low) / recent_range if recent_range > 0 else 0
        
        retesting_level = distance_to_high < 0.2 or distance_to_low < 0.2
        
        return {
            'broke_resistance': broke_resistance,
            'held_support': held_support,
            'retesting_level': retesting_level
        }
    
    # Analyze trends for each timeframe
    trend_5m = determine_trend(df_5m, period=20)
    trend_15m = determine_trend(df_15m, period=20)
    trend_1h = determine_trend(df_1h, period=20)
    
    # For higher timeframes, we'll use longer periods on the 1h data
    trend_4h = determine_trend(df_1h, period=80)  # ~4h worth of 1h bars
    trend_daily = determine_trend(df_1h, period=200)  # ~Daily trend from 1h data
    
    # Since we don't have 1m data, we'll approximate from 5m
    trend_1m = trend_5m  # Approximate 1m trend from 5m
    
    # Detect level breaks for each timeframe
    breaks_5m = detect_level_breaks(df_5m)
    breaks_15m = detect_level_breaks(df_15m)
    breaks_1h = detect_level_breaks(df_1h)
    
    # Add multi-timeframe data to each dataframe
    def add_mtf_columns(df, timeframe_suffix=''):
        df_mtf = df.copy()
        
        # Add trend data
        df_mtf['trend_1m'] = trend_1m
        df_mtf['trend_5m'] = trend_5m
        df_mtf['trend_15m'] = trend_15m
        df_mtf['trend_1h'] = trend_1h
        df_mtf['trend_4h'] = trend_4h
        df_mtf['trend_daily'] = trend_daily
        
        # Add level break data (use the most relevant timeframe)
        if timeframe_suffix == '5m':
            df_mtf['broke_resistance_5m'] = breaks_5m['broke_resistance']
            df_mtf['held_support_5m'] = breaks_5m['held_support']
            df_mtf['retesting_level'] = breaks_5m['retesting_level']
        elif timeframe_suffix == '15m':
            df_mtf['broke_resistance_15m'] = breaks_15m['broke_resistance']
            df_mtf['held_support_15m'] = breaks_15m['held_support']
            df_mtf['retesting_level'] = breaks_15m['retesting_level']
        else:  # 1h
            df_mtf['broke_resistance_1h'] = breaks_1h['broke_resistance']
            df_mtf['held_support_1h'] = breaks_1h['held_support']
            df_mtf['retesting_level'] = breaks_1h['retesting_level']
        
        # Add trend alignment score (how many timeframes agree)
        trends = [trend_1m, trend_5m, trend_15m, trend_1h, trend_4h, trend_daily]
        bullish_count = trends.count('bullish')
        bearish_count = trends.count('bearish')
        ranging_count = trends.count('ranging')
        
        if bullish_count >= 4:
            df_mtf['trend_alignment'] = 'strong_bullish'
        elif bearish_count >= 4:
            df_mtf['trend_alignment'] = 'strong_bearish'
        elif bullish_count > bearish_count:
            df_mtf['trend_alignment'] = 'weak_bullish'
        elif bearish_count > bullish_count:
            df_mtf['trend_alignment'] = 'weak_bearish'
        else:
            df_mtf['trend_alignment'] = 'mixed'
        
        df_mtf['trend_strength_score'] = max(bullish_count, bearish_count) / len(trends)
        
        return df_mtf
    
    # Apply to all dataframes
    df_5m_mtf = add_mtf_columns(df_5m, '5m')
    df_15m_mtf = add_mtf_columns(df_15m, '15m')
    df_1h_mtf = add_mtf_columns(df_1h, '1h')
    
    return df_5m_mtf, df_15m_mtf, df_1h_mtf
    
def add_advanced_volume_analysis(df_enhanced):
    """
    Add advanced volume analysis indicators to the enhanced dataframe
    
    Args:
        df_enhanced: DataFrame that already has basic volume analysis
    
    Returns:
        DataFrame with additional volume indicators
    """
    if 'ltv' not in df_enhanced.columns:
        print("Warning: No volume data (ltv column) found. Skipping volume analysis.")
        return df_enhanced
    
    df_vol = df_enhanced.copy()
    
    # Volume-Price Trend (VPT) - similar to OBV but uses price change percentage
    df_vol['price_change_pct'] = df_vol['close'].pct_change()
    df_vol['vpt'] = (df_vol['ltv'] * df_vol['price_change_pct']).cumsum()
    
    # Volume Rate of Change (VROC)
    df_vol['volume_roc_10'] = df_vol['ltv'].pct_change(periods=10) * 100
    
    # High/Low Volume Detection
    df_vol['volume_high'] = df_vol['ltv'] > (df_vol['volume_avg_20'] * 1.5)  # 50% above average
    df_vol['volume_low'] = df_vol['ltv'] < (df_vol['volume_avg_20'] * 0.5)   # 50% below average
    
    # Volume Spike Detection (unusual volume activity)
    df_vol['volume_spike'] = df_vol['ltv'] > (df_vol['volume_avg_20'] * 2.0)  # 100% above average
    
    # Accumulation/Distribution approximation using volume and price position
    df_vol['money_flow_multiplier'] = ((df_vol['close'] - df_vol['low']) - (df_vol['high'] - df_vol['close'])) / (df_vol['high'] - df_vol['low'])
    df_vol['money_flow_multiplier'] = df_vol['money_flow_multiplier'].fillna(0)  # Handle division by zero
    df_vol['money_flow_volume'] = df_vol['money_flow_multiplier'] * df_vol['ltv']
    df_vol['accumulation_distribution'] = df_vol['money_flow_volume'].cumsum()
    
    return df_vol

def add_advanced_volume_analysis(df_enhanced):
    """
    Add advanced volume analysis indicators to the enhanced dataframe
    
    Args:
        df_enhanced: DataFrame that already has basic volume analysis
    
    Returns:
        DataFrame with additional volume indicators
    """
    if 'ltv' not in df_enhanced.columns:
        print("Warning: No volume data (ltv column) found. Skipping volume analysis.")
        return df_enhanced
    
    df_vol = df_enhanced.copy()
    
    # Volume-Price Trend (VPT) - similar to OBV but uses price change percentage
    df_vol['price_change_pct'] = df_vol['close'].pct_change()
    df_vol['vpt'] = (df_vol['ltv'] * df_vol['price_change_pct']).cumsum()
    
    # Volume Rate of Change (VROC)
    df_vol['volume_roc_10'] = df_vol['ltv'].pct_change(periods=10) * 100
    
    # High/Low Volume Detection
    df_vol['volume_high'] = df_vol['ltv'] > (df_vol['volume_avg_20'] * 1.5)  # 50% above average
    df_vol['volume_low'] = df_vol['ltv'] < (df_vol['volume_avg_20'] * 0.5)   # 50% below average
    
    # Volume Spike Detection (unusual volume activity)
    df_vol['volume_spike'] = df_vol['ltv'] > (df_vol['volume_avg_20'] * 2.0)  # 100% above average
    
    # Accumulation/Distribution approximation using volume and price position
    df_vol['money_flow_multiplier'] = ((df_vol['close'] - df_vol['low']) - (df_vol['high'] - df_vol['close'])) / (df_vol['high'] - df_vol['low'])
    df_vol['money_flow_multiplier'] = df_vol['money_flow_multiplier'].fillna(0)  # Handle division by zero
    df_vol['money_flow_volume'] = df_vol['money_flow_multiplier'] * df_vol['ltv']
    df_vol['accumulation_distribution'] = df_vol['money_flow_volume'].cumsum()
    
    return df_vol

def add_market_behavior_analysis(df_enhanced, epic_pair='EURUSD'):
    """
    Add market behavior and momentum analysis to the enhanced dataframe
    
    Args:
        df_enhanced: DataFrame that already has basic enhancement
        epic_pair: Trading pair for pip calculation
    
    Returns:
        DataFrame with additional market behavior indicators
    """
    try:
        if df_enhanced is None:
            print("Warning: df_enhanced is None in add_market_behavior_analysis")
            return None
            
        print(f"Starting market behavior analysis with {len(df_enhanced)} rows")
        df_behavior = df_enhanced.copy()
        
        # Determine pip value based on pair
        if 'JPY' in epic_pair.upper():
            pip_multiplier = 100  # JPY pairs: 1 pip = 0.01
        else:
            pip_multiplier = 10000  # Most pairs: 1 pip = 0.0001
        
        print("Calculating price changes...")
        # Calculate price changes in pips for different timeframes
        df_behavior['price_change_1_bar_pips'] = (df_behavior['close'] - df_behavior['close'].shift(1)) * pip_multiplier
        df_behavior['price_change_4_bars_pips'] = (df_behavior['close'] - df_behavior['close'].shift(4)) * pip_multiplier
        df_behavior['price_change_12_bars_pips'] = (df_behavior['close'] - df_behavior['close'].shift(12)) * pip_multiplier
        
        print("Identifying candle types...")
        # Identify green and red candles
        df_behavior['is_green'] = df_behavior['close'] > df_behavior['open']
        df_behavior['is_red'] = df_behavior['close'] < df_behavior['open']
        df_behavior['is_doji'] = abs(df_behavior['close'] - df_behavior['open']) < (df_behavior['high'] - df_behavior['low']) * 0.1
        
        print("Calculating consecutive candles...")
        # Calculate consecutive candles
        def calculate_consecutive_candles(series):
            consecutive_counts = []
            current_count = 0
            current_type = None
            
            for value in series:
                if value == current_type:
                    current_count += 1
                else:
                    current_count = 1 if value else 0
                    current_type = value
                consecutive_counts.append(current_count if value else 0)
            
            return consecutive_counts
        
        df_behavior['consecutive_green_candles'] = calculate_consecutive_candles(df_behavior['is_green'])
        df_behavior['consecutive_red_candles'] = calculate_consecutive_candles(df_behavior['is_red'])
        
        print("Calculating rejection wicks...")
        # Calculate rejection wicks (long upper/lower shadows)
        df_behavior['body_size'] = abs(df_behavior['close'] - df_behavior['open'])
        df_behavior['upper_wick'] = df_behavior['high'] - df_behavior[['open', 'close']].max(axis=1)
        df_behavior['lower_wick'] = df_behavior[['open', 'close']].min(axis=1) - df_behavior['low']
        df_behavior['total_range'] = df_behavior['high'] - df_behavior['low']
        
        # Define rejection wicks (wick > 50% of total range and > 2x body size)
        # Handle cases where total_range is 0 (avoid division by zero)
        mask_valid_range = df_behavior['total_range'] > 0
        df_behavior['upper_rejection'] = False
        df_behavior['lower_rejection'] = False
        
        # Only calculate for bars with valid range
        if mask_valid_range.any():
            df_behavior.loc[mask_valid_range, 'upper_rejection'] = (
                (df_behavior.loc[mask_valid_range, 'upper_wick'] > df_behavior.loc[mask_valid_range, 'total_range'] * 0.5) & 
                (df_behavior.loc[mask_valid_range, 'upper_wick'] > df_behavior.loc[mask_valid_range, 'body_size'] * 2)
            )
            df_behavior.loc[mask_valid_range, 'lower_rejection'] = (
                (df_behavior.loc[mask_valid_range, 'lower_wick'] > df_behavior.loc[mask_valid_range, 'total_range'] * 0.5) & 
                (df_behavior.loc[mask_valid_range, 'lower_wick'] > df_behavior.loc[mask_valid_range, 'body_size'] * 2)
            )
        
        print("Counting rejection wicks...")
        # Count rejection wicks in recent periods
        def rolling_rejection_count(df, window=10):
            rejection_counts = []
            for i in range(len(df)):
                start_idx = max(0, i - window + 1)
                upper_rejections = df['upper_rejection'].iloc[start_idx:i+1].sum()
                lower_rejections = df['lower_rejection'].iloc[start_idx:i+1].sum()
                rejection_counts.append(upper_rejections + lower_rejections)
            return rejection_counts
        
        df_behavior['rejection_wicks_count'] = rolling_rejection_count(df_behavior)
        
        print("Calculating consolidation range...")
        # Calculate consolidation range (high-low range over recent periods)
        def calculate_consolidation_range(df, window=20, pip_multiplier=10000):
            ranges = []
            for i in range(len(df)):
                start_idx = max(0, i - window + 1)
                period_high = df['high'].iloc[start_idx:i+1].max()
                period_low = df['low'].iloc[start_idx:i+1].min()
                range_pips = (period_high - period_low) * pip_multiplier
                ranges.append(range_pips)
            return ranges
        
        df_behavior['consolidation_range_pips'] = calculate_consolidation_range(df_behavior, pip_multiplier=pip_multiplier)
        
        print("Detecting breakouts...")
        # Breakout detection (simplified)
        def detect_breakouts(df, window=20):
            breakout_bars = []
            for i in range(len(df)):
                if i < window:
                    breakout_bars.append(0)
                    continue
                    
                # Look back to find recent range
                start_idx = max(0, i - window)
                recent_high = df['high'].iloc[start_idx:i].max()
                recent_low = df['low'].iloc[start_idx:i].min()
                
                # Check if current bar breaks out
                current_high = df['high'].iloc[i]
                current_low = df['low'].iloc[i]
                
                if current_high > recent_high or current_low < recent_low:
                    # This is a breakout bar, count bars since
                    bars_since = 1
                else:
                    # Count bars since last breakout
                    bars_since = breakout_bars[-1] + 1 if breakout_bars else 1
                    
                breakout_bars.append(bars_since)
            
            return breakout_bars
        
        df_behavior['bars_since_breakout'] = detect_breakouts(df_behavior)
        
        print("Calculating bars at current level...")
        # Bars at current level (price staying within small range)
        def bars_at_level(df, tolerance_pips=5, pip_multiplier=10000):
            level_bars = []
            for i in range(len(df)):
                if i == 0:
                    level_bars.append(1)
                    continue
                    
                current_price = df['close'].iloc[i]
                bars_at_current = 1
                
                # Look backwards to count bars within tolerance
                for j in range(i-1, -1, -1):
                    past_price = df['close'].iloc[j]
                    if abs(current_price - past_price) * pip_multiplier <= tolerance_pips:
                        bars_at_current += 1
                    else:
                        break
                        
                level_bars.append(bars_at_current)
            
            return level_bars
        
        df_behavior['bars_at_current_level'] = bars_at_level(df_behavior, pip_multiplier=pip_multiplier)
        
        print(f"Market behavior analysis complete, returning {len(df_behavior)} rows")
        return df_behavior
        
    except Exception as e:
        print(f"Error in add_market_behavior_analysis: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None

# Updated usage example with complete analysis:
def enhance_candle_data_complete(engine, epic, epic_pair='EURUSD'):
    """
    Complete function to fetch and enhance candle data with all analysis:
    - Support/Resistance levels
    - Volume analysis
    - Market behavior
    - Multi-timeframe trend analysis
    """
    print(f"Fetching data for {epic}...")
    
    # Fetch data with error handling
    try:
        df_5m = fetch_candle_data(engine, epic, 5, 1000)
        print(f"5m data: {len(df_5m) if df_5m is not None else 'None'} rows")
    except Exception as e:
        print(f"Error fetching 5m data: {e}")
        raise
    
    try:
        df_15m = fetch_candle_data(engine, epic, 15, 1000)
        print(f"15m data: {len(df_15m) if df_15m is not None else 'None'} rows")
    except Exception as e:
        print(f"Error fetching 15m data: {e}")
        raise
    
    try:
        df_1h = fetch_candle_data(engine, epic, 60, lookback_hours=200)
        print(f"1h data: {len(df_1h) if df_1h is not None else 'None'} rows")
    except Exception as e:
        print(f"Error fetching 1h data: {e}")
        raise
    
    # Validate data
    if df_5m is None or len(df_5m) == 0:
        raise ValueError("No 5m data available")
    if df_15m is None or len(df_15m) == 0:
        raise ValueError("No 15m data available") 
    if df_1h is None or len(df_1h) == 0:
        raise ValueError("No 1h data available")
    
    print("Adding support/resistance analysis...")
    # Add support/resistance analysis with volume
    df_5m_enhanced = add_support_resistance_to_df(df_5m, epic_pair)
    df_15m_enhanced = add_support_resistance_to_df(df_15m, epic_pair)
    df_1h_enhanced = add_support_resistance_to_df(df_1h, epic_pair)
    
    # Debug: Check if any dataframes became None
    print(f"After S/R analysis - 5m: {len(df_5m_enhanced) if df_5m_enhanced is not None else 'None'}")
    print(f"After S/R analysis - 15m: {len(df_15m_enhanced) if df_15m_enhanced is not None else 'None'}")
    print(f"After S/R analysis - 1h: {len(df_1h_enhanced) if df_1h_enhanced is not None else 'None'}")
    
    print("Adding volume analysis...")
    # Add advanced volume analysis
    df_5m_enhanced = add_advanced_volume_analysis(df_5m_enhanced)
    df_15m_enhanced = add_advanced_volume_analysis(df_15m_enhanced)
    df_1h_enhanced = add_advanced_volume_analysis(df_1h_enhanced)
    
    # Debug: Check if any dataframes became None
    print(f"After volume analysis - 5m: {len(df_5m_enhanced) if df_5m_enhanced is not None else 'None'}")
    print(f"After volume analysis - 15m: {len(df_15m_enhanced) if df_15m_enhanced is not None else 'None'}")
    print(f"After volume analysis - 1h: {len(df_1h_enhanced) if df_1h_enhanced is not None else 'None'}")
    
    print("Adding market behavior analysis...")
    # Add market behavior analysis
    df_5m_enhanced = add_market_behavior_analysis(df_5m_enhanced, epic_pair)
    df_15m_enhanced = add_market_behavior_analysis(df_15m_enhanced, epic_pair)
    df_1h_enhanced = add_market_behavior_analysis(df_1h_enhanced, epic_pair)
    
    # Debug: Check if any dataframes became None
    print(f"After behavior analysis - 5m: {len(df_5m_enhanced) if df_5m_enhanced is not None else 'None'}")
    print(f"After behavior analysis - 15m: {len(df_15m_enhanced) if df_15m_enhanced is not None else 'None'}")
    print(f"After behavior analysis - 1h: {len(df_1h_enhanced) if df_1h_enhanced is not None else 'None'}")
    
    print("Adding multi-timeframe analysis...")
    # Add multi-timeframe analysis
    df_5m_final, df_15m_final, df_1h_final = add_multi_timeframe_analysis(
        df_5m_enhanced, df_15m_enhanced, df_1h_enhanced, epic_pair
    )
    
    print("Enhancement complete!")
    return df_5m_final, df_15m_final, df_1h_final

# Keep the old function for backward compatibility
def enhance_candle_data_with_volume(engine, epic, epic_pair='EURUSD'):
    """
    Complete function to fetch and enhance candle data with support/resistance levels, volume analysis, and market behavior
    """
    return enhance_candle_data_complete(engine, epic, epic_pair)

# To view the enhanced data structure:
def print_enhanced_sample(df_enhanced):
    """Print a sample of the enhanced data including all analysis types"""
    latest_row = df_enhanced.iloc[-1]
    
    enhanced_data = {
        'timestamp': latest_row['start_time'].isoformat(),
        'price': latest_row['close'],
        'nearest_resistance': latest_row['nearest_resistance'],
        'nearest_support': latest_row['nearest_support'],
        'distance_to_resistance_pips': latest_row['distance_to_resistance_pips'],
        'distance_to_support_pips': latest_row['distance_to_support_pips'],
        'risk_reward_ratio': latest_row['risk_reward_ratio'],
        'weekly_high': latest_row['weekly_high'],
        'weekly_low': latest_row['weekly_low'],
        'daily_high': latest_row['daily_high'],
        'daily_low': latest_row['daily_low'],
    }
    
    # Add volume analysis if ltv column exists
    if 'ltv' in df_enhanced.columns:
        enhanced_data.update({
            # Volume analysis
            'volume_current': latest_row['ltv'],
            'volume_avg_10': latest_row.get('volume_avg_10', None),
            'volume_avg_20': latest_row.get('volume_avg_20', None),
            'volume_avg_50': latest_row.get('volume_avg_50', None),
            'volume_ratio_10': latest_row.get('volume_ratio_10', None),
            'volume_ratio_20': latest_row.get('volume_ratio_20', None),
            'volume_ratio_50': latest_row.get('volume_ratio_50', None),
            'volume_percentile_50': latest_row.get('volume_percentile_50', None),
        })
    
    # Add market behavior analysis
    if 'bars_since_breakout' in df_enhanced.columns:
        enhanced_data.update({
            # Recent behavior
            'bars_since_breakout': latest_row.get('bars_since_breakout', None),
            'bars_at_current_level': latest_row.get('bars_at_current_level', None),
            'rejection_wicks_count': latest_row.get('rejection_wicks_count', None),
            'consolidation_range_pips': latest_row.get('consolidation_range_pips', None),
            # Momentum context
            'price_change_1_bar_pips': latest_row.get('price_change_1_bar_pips', None),
            'price_change_4_bars_pips': latest_row.get('price_change_4_bars_pips', None),
            'price_change_12_bars_pips': latest_row.get('price_change_12_bars_pips', None),
            'consecutive_green_candles': latest_row.get('consecutive_green_candles', None),
            'consecutive_red_candles': latest_row.get('consecutive_red_candles', None),
        })
    
    # Add multi-timeframe analysis
    if 'trend_1m' in df_enhanced.columns:
        enhanced_data.update({
            # Structure across timeframes
            'trend_1m': latest_row.get('trend_1m', None),
            'trend_5m': latest_row.get('trend_5m', None),
            'trend_15m': latest_row.get('trend_15m', None),
            'trend_1h': latest_row.get('trend_1h', None),
            'trend_4h': latest_row.get('trend_4h', None),
            'trend_daily': latest_row.get('trend_daily', None),
            'trend_alignment': latest_row.get('trend_alignment', None),
            'trend_strength_score': latest_row.get('trend_strength_score', None),
        })
    
    # Add key level breaks (check which columns exist)
    level_break_columns = [col for col in df_enhanced.columns if 'broke_resistance' in col or 'held_support' in col or 'retesting_level' in col]
    if level_break_columns:
        level_breaks = {}
        for col in level_break_columns:
            level_breaks[col] = latest_row.get(col, None)
        enhanced_data.update(level_breaks)
    
    print("Enhanced data sample:")
    for key, value in enhanced_data.items():
        if isinstance(value, float) and not np.isnan(value):
            print(f"'{key}': {value:.4f},")
        else:
            print(f"'{key}': {value},")
    
    return enhanced_data