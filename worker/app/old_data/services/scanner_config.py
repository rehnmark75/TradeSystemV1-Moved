import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time

def add_ema_indicators(df, periods=[9, 21, 200]):
    """
    Add EMA indicators to the dataframe
    
    Args:
        df: Enhanced dataframe
        periods: List of EMA periods to calculate
    
    Returns:
        DataFrame with EMA columns added
    """
    df_ema = df.copy()
    
    for period in periods:
        # Calculate EMA using pandas ewm
        df_ema[f'ema_{period}'] = df_ema['close'].ewm(span=period, adjust=False).mean()
    
    return df_ema

def adjust_bid_prices_for_signals(df, spread_pips=1.5):
    """
    Convert BID prices to approximate MID prices for signal detection
    
    Args:
        df: DataFrame with BID prices
        spread_pips: Typical spread in pips (default 1.5)
    
    Returns:
        DataFrame with adjusted MID prices
    """
    spread = spread_pips / 10000  # Convert pips to decimal
    
    df_adjusted = df.copy()
    df_adjusted['open'] = df['open'] + spread/2
    df_adjusted['high'] = df['high'] + spread/2
    df_adjusted['low'] = df['low'] + spread/2
    df_adjusted['close'] = df['close'] + spread/2
    
    return df_adjusted

def detect_ema_signals(df, epic):
    """
    Detect bull and bear signals based on EMA conditions (for MID prices)
    
    Args:
        df: Enhanced dataframe with EMA indicators
        epic: Epic code being analyzed
    
    Returns:
        Dict with signal information
    """
    if len(df) < 200:  # Need enough data for 200 EMA
        return None
    
    # Get latest values
    latest = df.iloc[-1]
    previous = df.iloc[-2]  # Previous candle to check for new signals
    
    current_price = latest['close']
    prev_price = previous['close']
    
    ema_9_current = latest['ema_9']
    ema_21_current = latest['ema_21']
    ema_200_current = latest['ema_200']
    
    ema_9_prev = previous['ema_9']
    
    # Check for Bull Signal
    bull_conditions = {
        'price_above_ema9': current_price > ema_9_current,
        'ema9_above_ema21': ema_9_current > ema_21_current,
        'ema9_above_ema200': ema_9_current > ema_200_current,
        'ema21_above_ema200': ema_21_current > ema_200_current,
        'new_signal': prev_price <= ema_9_prev  # Was below EMA9, now above (new signal)
    }
    
    # Check for Bear Signal  
    bear_conditions = {
        'price_below_ema9': current_price < ema_9_current,
        'ema21_above_ema9': ema_21_current > ema_9_current,
        'ema200_above_ema9': ema_200_current > ema_9_current,
        'ema200_above_ema21': ema_200_current > ema_21_current,
        'new_signal': prev_price >= ema_9_prev  # Was above EMA9, now below (new signal)
    }
    
    # Determine signal type
    signal_type = None
    conditions_met = None
    confidence_score = 0
    
    if all(bull_conditions.values()):
        signal_type = 'BULL'
        conditions_met = bull_conditions
        # Calculate confidence based on EMA separation and volume
        ema_separation = (ema_9_current - ema_21_current) / current_price * 10000  # In pips
        volume_strength = latest.get('volume_ratio_20', 1.0)
        confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (volume_strength * 0.2))
        
    elif all(bear_conditions.values()):
        signal_type = 'BEAR'
        conditions_met = bear_conditions
        # Calculate confidence based on EMA separation and volume
        ema_separation = (ema_21_current - ema_9_current) / current_price * 10000  # In pips
        volume_strength = latest.get('volume_ratio_20', 1.0)
        confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (volume_strength * 0.2))
    
    if signal_type:
        return {
            'signal_type': signal_type,
            'epic': epic,
            'timeframe': '5m',
            'timestamp': latest['start_time'],
            'price': current_price,
            'ema_9': ema_9_current,
            'ema_21': ema_21_current,
            'ema_200': ema_200_current,
            'conditions_met': conditions_met,
            'confidence_score': confidence_score,
            'volume_ratio': latest.get('volume_ratio_20', None),
            'distance_to_support_pips': latest.get('distance_to_support_pips', None),
            'distance_to_resistance_pips': latest.get('distance_to_resistance_pips', None),
            'trend_alignment': latest.get('trend_alignment', None),
            'enhanced_data': df.tail(5).to_dict('records')  # Last 5 candles for context
        }
    
    return None

def detect_ema_signals_bid_adjusted(df, epic, spread_pips=1.5):
    """
    Detect EMA signals with BID price adjustment
    
    Args:
        df: Enhanced dataframe with BID prices
        epic: Epic code
        spread_pips: Spread adjustment in pips
    
    Returns:
        Signal dict with both BID and MID prices
    """
    if len(df) < 200:  # Need enough data for 200 EMA
        return None
    
    # Adjust BID prices to approximate MID prices for signal detection
    df_mid = adjust_bid_prices_for_signals(df, spread_pips)
    
    # Add EMA indicators to adjusted prices
    df_with_emas = add_ema_indicators(df_mid)
    
    # Get latest values
    latest = df_with_emas.iloc[-1]
    previous = df_with_emas.iloc[-2]
    original_latest = df.iloc[-1]  # Original BID prices
    
    current_price_mid = latest['close']
    prev_price_mid = previous['close']
    current_price_bid = original_latest['close']
    
    ema_9_current = latest['ema_9']
    ema_21_current = latest['ema_21']
    ema_200_current = latest['ema_200']
    
    ema_9_prev = previous['ema_9']
    
    # Check for Bull Signal
    bull_conditions = {
        'price_above_ema9': current_price_mid > ema_9_current,
        'ema9_above_ema21': ema_9_current > ema_21_current,
        'ema9_above_ema200': ema_9_current > ema_200_current,
        'ema21_above_ema200': ema_21_current > ema_200_current,
        'new_signal': prev_price_mid <= ema_9_prev  # Was below EMA9, now above
    }
    
    # Check for Bear Signal  
    bear_conditions = {
        'price_below_ema9': current_price_mid < ema_9_current,
        'ema21_above_ema9': ema_21_current > ema_9_current,
        'ema200_above_ema9': ema_200_current > ema_9_current,
        'ema200_above_ema21': ema_200_current > ema_21_current,
        'new_signal': prev_price_mid >= ema_9_prev  # Was above EMA9, now below
    }
    
    # Determine signal type
    signal_type = None
    conditions_met = None
    confidence_score = 0
    
    if all(bull_conditions.values()):
        signal_type = 'BULL'
        conditions_met = bull_conditions
        # Calculate confidence based on EMA separation and volume
        ema_separation = (ema_9_current - ema_21_current) / current_price_mid * 10000  # In pips
        volume_strength = latest.get('volume_ratio_20', 1.0)
        confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (volume_strength * 0.2))
        
    elif all(bear_conditions.values()):
        signal_type = 'BEAR'
        conditions_met = bear_conditions
        # Calculate confidence based on EMA separation and volume
        ema_separation = (ema_21_current - ema_9_current) / current_price_mid * 10000  # In pips
        volume_strength = latest.get('volume_ratio_20', 1.0)
        confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (volume_strength * 0.2))
    
    if signal_type:
        # Calculate execution prices
        spread = spread_pips / 10000
        if signal_type == 'BULL':
            execution_price = current_price_mid + (spread / 2)  # ASK price for buying
        else:  # BEAR
            execution_price = current_price_mid - (spread / 2)  # BID price for selling
        
        return {
            'signal_type': signal_type,
            'epic': epic,
            'timeframe': '5m',
            'timestamp': latest['start_time'],
            'price_mid': current_price_mid,  # MID price (for charts)
            'price_bid': current_price_bid,  # Original BID price
            'execution_price': execution_price,  # Actual trading price
            'spread_pips': spread_pips,
            'ema_9': ema_9_current,
            'ema_21': ema_21_current,
            'ema_200': ema_200_current,
            'conditions_met': conditions_met,
            'confidence_score': confidence_score,
            'volume_ratio': latest.get('volume_ratio_20', None),
            'distance_to_support_pips': latest.get('distance_to_support_pips', None),
            'distance_to_resistance_pips': latest.get('distance_to_resistance_pips', None),
            'trend_alignment': latest.get('trend_alignment', None),
            'enhanced_data': df_with_emas.tail(5).to_dict('records')  # Last 5 candles
        }
    
    return None

def backtest_ema_signals(df, epic, lookback_bars=1000, timeframe="5m"):
    """
    Backtest EMA signals on historical data (for MID prices)
    
    Args:
        df: Enhanced dataframe with EMA indicators
        epic: Epic code
        lookback_bars: How many bars to look back (default 1000)
        timeframe: Timeframe being analyzed (for display purposes)
    
    Returns:
        List of historical signals found
    """
    if df is None:
        print(f"❌ DataFrame is None for {epic}")
        return []
        
    if len(df) < 200:  # Need enough data for 200 EMA
        print(f"❌ Not enough data for {epic}: {len(df)} bars (need 200+)")
        return []
    
    signals = []
    start_idx = max(200, len(df) - lookback_bars)
    
    print(f"Analyzing {epic} from bar {start_idx} to {len(df)} ({len(df) - start_idx} bars)")
    
    for i in range(start_idx + 1, len(df)):  # +1 because we need previous bar
        try:
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            current_price = current['close']
            prev_price = previous['close']
            
            # Check if EMA columns exist
            if 'ema_9' not in current or 'ema_21' not in current or 'ema_200' not in current:
                continue
                
            ema_9_current = current['ema_9']
            ema_21_current = current['ema_21']
            ema_200_current = current['ema_200']
            ema_9_prev = previous['ema_9']
            
            # Skip if any EMA values are NaN
            if pd.isna(ema_9_current) or pd.isna(ema_21_current) or pd.isna(ema_200_current) or pd.isna(ema_9_prev):
                continue
            
            # Check for Bull Signal
            bull_conditions = {
                'price_above_ema9': current_price > ema_9_current,
                'ema9_above_ema21': ema_9_current > ema_21_current,
                'ema9_above_ema200': ema_9_current > ema_200_current,
                'ema21_above_ema200': ema_21_current > ema_200_current,
                'new_signal': prev_price <= ema_9_prev and current_price > ema_9_current  # Crossed above EMA9
            }
            
            # Check for Bear Signal  
            bear_conditions = {
                'price_below_ema9': current_price < ema_9_current,
                'ema21_above_ema9': ema_21_current > ema_9_current,
                'ema200_above_ema9': ema_200_current > ema_9_current,
                'ema200_above_ema21': ema_200_current > ema_21_current,
                'new_signal': prev_price >= ema_9_prev and current_price < ema_9_current  # Crossed below EMA9
            }
            
            signal_type = None
            confidence_score = 0
            
            if all(bull_conditions.values()):
                signal_type = 'BULL'
                ema_separation = abs(ema_9_current - ema_21_current) / current_price * 10000
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2))
                
            elif all(bear_conditions.values()):
                signal_type = 'BEAR'
                ema_separation = abs(ema_21_current - ema_9_current) / current_price * 10000
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2))
            
            if signal_type:
                # Calculate potential profit/loss for backtesting
                next_bars = df.iloc[i+1:i+21] if i+21 < len(df) else df.iloc[i+1:]  # Next 20 bars
                
                if len(next_bars) > 0:
                    if signal_type == 'BULL':
                        max_profit = (next_bars['high'].max() - current_price) * 10000  # Pips
                        max_loss = (current_price - next_bars['low'].min()) * 10000  # Pips
                    else:  # BEAR
                        max_profit = (current_price - next_bars['low'].min()) * 10000  # Pips
                        max_loss = (next_bars['high'].max() - current_price) * 10000  # Pips
                else:
                    max_profit = max_loss = 0
                
                signal_data = {
                    'signal_type': signal_type,
                    'epic': epic,
                    'timeframe': timeframe,
                    'timestamp': current['start_time'],
                    'price': current_price,
                    'ema_9': ema_9_current,
                    'ema_21': ema_21_current,
                    'ema_200': ema_200_current,
                    'confidence_score': confidence_score,
                    'volume_ratio': current.get('volume_ratio_20', None),
                    'distance_to_support_pips': current.get('distance_to_support_pips', None),
                    'distance_to_resistance_pips': current.get('distance_to_resistance_pips', None),
                    'trend_alignment': current.get('trend_alignment', None),
                    'max_profit_pips': max_profit,
                    'max_loss_pips': max_loss,
                    'risk_reward_potential': max_profit / max_loss if max_loss > 0 else 0
                }
                
                signals.append(signal_data)
                
        except Exception as e:
            continue
    
    return signals

def backtest_ema_signals_bid_adjusted(df, epic, lookback_bars=1000, timeframe="5m", spread_pips=1.5):
    """
    Backtest EMA signals with BID price adjustment
    """
    if df is None or len(df) < 200:
        return []
    
    # Adjust BID to MID prices
    df_mid = adjust_bid_prices_for_signals(df, spread_pips)
    df_with_emas = add_ema_indicators(df_mid)
    
    signals = []
    start_idx = max(200, len(df_with_emas) - lookback_bars)
    
    print(f"Backtesting {epic} with BID adjustment (spread: {spread_pips} pips)")
    
    for i in range(start_idx + 1, len(df_with_emas)):
        try:
            current = df_with_emas.iloc[i]
            previous = df_with_emas.iloc[i-1]
            original_current = df.iloc[i]  # Original BID prices
            
            current_price_mid = current['close']
            prev_price_mid = previous['close']
            
            # Check if EMA columns exist and are valid
            if any(pd.isna([current['ema_9'], current['ema_21'], current['ema_200'], previous['ema_9']])):
                continue
                
            ema_9_current = current['ema_9']
            ema_21_current = current['ema_21']
            ema_200_current = current['ema_200']
            ema_9_prev = previous['ema_9']
            
            # Check for Bull Signal
            bull_conditions = {
                'price_above_ema9': current_price_mid > ema_9_current,
                'ema9_above_ema21': ema_9_current > ema_21_current,
                'ema9_above_ema200': ema_9_current > ema_200_current,
                'ema21_above_ema200': ema_21_current > ema_200_current,
                'new_signal': prev_price_mid <= ema_9_prev and current_price_mid > ema_9_current
            }
            
            # Check for Bear Signal  
            bear_conditions = {
                'price_below_ema9': current_price_mid < ema_9_current,
                'ema21_above_ema9': ema_21_current > ema_9_current,
                'ema200_above_ema9': ema_200_current > ema_9_current,
                'ema200_above_ema21': ema_200_current > ema_21_current,
                'new_signal': prev_price_mid >= ema_9_prev and current_price_mid < ema_9_current
            }
            
            signal_type = None
            confidence_score = 0
            
            if all(bull_conditions.values()):
                signal_type = 'BULL'
                ema_separation = abs(ema_9_current - ema_21_current) / current_price_mid * 10000
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2))
                
            elif all(bear_conditions.values()):
                signal_type = 'BEAR'
                ema_separation = abs(ema_21_current - ema_9_current) / current_price_mid * 10000
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2))
            
            if signal_type:
                # Calculate potential profit/loss using MID prices
                next_bars = df_with_emas.iloc[i+1:i+21] if i+21 < len(df_with_emas) else df_with_emas.iloc[i+1:]
                
                if len(next_bars) > 0:
                    if signal_type == 'BULL':
                        max_profit = (next_bars['high'].max() - current_price_mid) * 10000
                        max_loss = (current_price_mid - next_bars['low'].min()) * 10000
                    else:
                        max_profit = (current_price_mid - next_bars['low'].min()) * 10000
                        max_loss = (next_bars['high'].max() - current_price_mid) * 10000
                else:
                    max_profit = max_loss = 0
                
                # Calculate execution price
                spread = spread_pips / 10000
                if signal_type == 'BULL':
                    execution_price = current_price_mid + (spread / 2)
                else:
                    execution_price = current_price_mid - (spread / 2)
                
                signal_data = {
                    'signal_type': signal_type,
                    'epic': epic,
                    'timeframe': timeframe,
                    'timestamp': current['start_time'],
                    'price_mid': current_price_mid,
                    'price_bid': original_current['close'],
                    'execution_price': execution_price,
                    'spread_pips': spread_pips,
                    'ema_9': ema_9_current,
                    'ema_21': ema_21_current,
                    'ema_200': ema_200_current,
                    'confidence_score': confidence_score,
                    'volume_ratio': current.get('volume_ratio_20', None),
                    'max_profit_pips': max_profit,
                    'max_loss_pips': max_loss,
                    'risk_reward_potential': max_profit / max_loss if max_loss > 0 else 0
                }
                
                signals.append(signal_data)
                
        except Exception as e:
            continue
    
    return signals

def extract_pair_from_epic(epic):
    """Extract currency pair from IG epic format"""
    parts = epic.split('.')
    if len(parts) >= 3:
        return parts[2]  # Usually the currency pair
    return 'EURUSD'  # Default fallback

def display_historical_signals(signals):
    """
    Display historical signals in a nice format - handles both old and new signal formats
    """
    if not signals:
        print("No historical signals found")
        return
    
    print(f"\n📊 LAST {len(signals)} EMA SIGNALS FOUND:")
    print("=" * 100)
    
    for i, signal in enumerate(signals, 1):
        print(f"\n#{i} - {signal['signal_type']} SIGNAL")
        print(f"Epic: {signal['epic']}")
        print(f"Time: {signal['timestamp']}")
        
        # Handle both old and new signal formats
        if 'price_mid' in signal:
            # NEW: BID-adjusted format
            print(f"MID Price: {signal['price_mid']:.5f}")
            print(f"BID Price: {signal['price_bid']:.5f}")
            print(f"Execution Price: {signal['execution_price']:.5f}")
            if 'spread_pips' in signal:
                print(f"Spread: {signal['spread_pips']:.1f} pips")
        elif 'price' in signal:
            # OLD: Original format
            print(f"Price: {signal['price']:.5f}")
        else:
            # FALLBACK: Try to find any price field
            price_fields = ['close', 'price_bid', 'execution_price']
            price_found = False
            for field in price_fields:
                if field in signal:
                    print(f"Price: {signal[field]:.5f}")
                    price_found = True
                    break
            if not price_found:
                print("Price: N/A")
        
        print(f"EMAs: 9={signal.get('ema_9', 0):.5f} | 21={signal.get('ema_21', 0):.5f} | 200={signal.get('ema_200', 0):.5f}")
        print(f"Confidence: {signal.get('confidence_score', 0):.1%}")
        
        if signal.get('volume_ratio'):
            print(f"Volume: {signal['volume_ratio']:.2f}x average")
        
        if signal.get('trend_alignment'):
            print(f"Trend: {signal['trend_alignment']}")
            
        # Backtesting results
        if signal.get('max_profit_pips') is not None and signal.get('max_loss_pips') is not None:
            print(f"📈 Max Profit: +{signal['max_profit_pips']:.1f} pips")
            print(f"📉 Max Loss: -{signal['max_loss_pips']:.1f} pips")
            if signal.get('risk_reward_potential', 0) > 0:
                print(f"🎯 R:R Potential: 1:{signal['risk_reward_potential']:.2f}")
        
        print("-" * 50)

def analyze_signal_performance(signals):
    """
    Analyze the performance of historical signals
    """
    if not signals:
        return
    
    bull_signals = [s for s in signals if s['signal_type'] == 'BULL']
    bear_signals = [s for s in signals if s['signal_type'] == 'BEAR']
    
    print(f"\n📊 SIGNAL PERFORMANCE ANALYSIS:")
    print(f"Total Signals: {len(signals)}")
    print(f"Bull Signals: {len(bull_signals)}")
    print(f"Bear Signals: {len(bear_signals)}")
    
    if signals:
        avg_confidence = sum(s['confidence_score'] for s in signals) / len(signals)
        avg_profit = sum(s['max_profit_pips'] for s in signals) / len(signals)
        avg_loss = sum(s['max_loss_pips'] for s in signals) / len(signals)
        
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Average Max Profit: {avg_profit:.1f} pips")
        print(f"Average Max Loss: {avg_loss:.1f} pips")
        
        # Win rate (assuming 20 pip profit target and 10 pip stop loss)
        profit_target = 20
        stop_loss = 10
        
        profitable_signals = len([s for s in signals if s['max_profit_pips'] >= profit_target])
        losing_signals = len([s for s in signals if s['max_loss_pips'] >= stop_loss])
        
        print(f"Signals reaching +{profit_target} pips: {profitable_signals}/{len(signals)} ({profitable_signals/len(signals)*100:.1f}%)")
        print(f"Signals hitting -{stop_loss} pips: {losing_signals}/{len(signals)} ({losing_signals/len(signals)*100:.1f}%)")

def test_single_epic_bid_adjusted(engine, epic, enhancement_function, spread_pips=1.5):
    """
    Test signal detection with BID adjustment
    """
    print(f"🔍 Testing {epic} with BID price adjustment (spread: {spread_pips} pips)...")
    
    try:
        # Get data
        df_5m, _, _ = enhancement_function(engine, epic, extract_pair_from_epic(epic))
        
        if df_5m is None:
            print(f"❌ No data returned for {epic}")
            return None
            
        print(f"✓ Got {len(df_5m)} 5-minute candles")
        
        # Show price adjustment effect
        latest_bid = df_5m.iloc[-1]
        df_mid = adjust_bid_prices_for_signals(df_5m.tail(3), spread_pips)
        latest_mid = df_mid.iloc[-1]
        
        print(f"\n📊 Price Adjustment Effect:")
        print(f"BID Close: {latest_bid['close']:.5f}")
        print(f"MID Close: {latest_mid['close']:.5f} (+{(latest_mid['close'] - latest_bid['close']) * 10000:.1f} pips)")
        
        # Test current signal
        signal = detect_ema_signals_bid_adjusted(df_5m, epic, spread_pips)
        
        if signal:
            print(f"\n🚨 CURRENT SIGNAL DETECTED:")
            print(f"Type: {signal['signal_type']}")
            print(f"MID Price: {signal['price_mid']:.5f}")
            print(f"BID Price: {signal['price_bid']:.5f}")
            print(f"Execution Price: {signal['execution_price']:.5f}")
            print(f"Confidence: {signal['confidence_score']:.1%}")
        else:
            print("\n✓ No current signals")
        
        # Get historical signals
        signals = backtest_ema_signals_bid_adjusted(df_5m, epic, lookback_bars=500, spread_pips=spread_pips)
        
        print(f"\n🎯 Found {len(signals)} historical signals")
        
        if signals:
            print("\n📋 Last 3 signals:")
            for signal in signals[-3:]:
                print(f"{signal['timestamp']} - {signal['signal_type']} at MID:{signal['price_mid']:.5f} EXEC:{signal['execution_price']:.5f} (Conf: {signal['confidence_score']:.1%})")
        
        return signals
        
    except Exception as e:
        print(f"❌ Error testing {epic}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_historical_analysis_bid_adjusted(engine, epic_list, enhancement_function, num_signals=10, spread_pips=1.5):
    """
    Run historical analysis with BID price adjustment
    """
    print(f"🔍 Running BID-Adjusted EMA Signal Analysis (spread: {spread_pips} pips)...")
    print(f"📋 Analyzing: {epic_list}")
    print("=" * 60)
    
    all_signals = []
    
    for epic in epic_list:
        try:
            print(f"Backtesting {epic}...")
            
            # Get enhanced data (5m)
            df_5m, _, _ = enhancement_function(engine, epic, extract_pair_from_epic(epic))
            
            if df_5m is None or len(df_5m) == 0:
                print(f"❌ No data available for {epic}")
                continue
                
            print(f"✓ Got {len(df_5m)} 5m candles")
            
            # Get signals with BID adjustment
            signals = backtest_ema_signals_bid_adjusted(df_5m, epic, lookback_bars=1000, spread_pips=spread_pips)
            all_signals.extend(signals)
            
            print(f"Found {len(signals)} signals for {epic}")
            
        except Exception as e:
            print(f"❌ Error backtesting {epic}: {e}")
            continue
    
    # Sort and display results
    all_signals.sort(key=lambda x: x['timestamp'], reverse=True)
    latest_signals = all_signals[:num_signals]
    
    display_historical_signals(latest_signals)
    analyze_signal_performance(latest_signals)
    
    return latest_signals

def get_latest_historical_signals(engine, epic_list, enhancement_function, num_signals=10, use_5min=True):
    """
    Get the latest historical signals across multiple epics
    
    Args:
        engine: Database engine
        epic_list: List of epics to analyze
        enhancement_function: Data enhancement function
        num_signals: Number of latest signals to return
        use_5min: If True, use 5-minute data; if False, use 15-minute data
    
    Returns:
        List of latest signals sorted by timestamp
    """
    all_signals = []
    
    for epic in epic_list:
        try:
            print(f"Backtesting {epic}...")
            
            # Get enhanced data
            df_5m_enhanced, df_15m_enhanced, df_1h_enhanced = enhancement_function(engine, epic, extract_pair_from_epic(epic))
            
            # Check if data was retrieved successfully
            if df_5m_enhanced is None or len(df_5m_enhanced) == 0:
                print(f"❌ No 5m data available for {epic}")
                continue
                
            if use_5min:
                # Use 5-minute data directly
                df_target = df_5m_enhanced
                timeframe = "5m"
            else:
                # Use 15-minute data
                if df_15m_enhanced is None or len(df_15m_enhanced) == 0:
                    print(f"❌ No 15m data available for {epic}")
                    continue
                df_target = df_15m_enhanced
                timeframe = "15m"
            
            print(f"✓ Got {len(df_target)} {timeframe} candles for {epic}")
            
            # Add EMA indicators
            df_with_emas = add_ema_indicators(df_target)
            
            # Ensure we have enough data for 200 EMA
            if len(df_with_emas) < 200:
                print(f"❌ Not enough data for 200 EMA on {epic} (need 200, got {len(df_with_emas)})")
                continue
            
            # Get historical signals
            signals = backtest_ema_signals(df_with_emas, epic, timeframe=timeframe)
            all_signals.extend(signals)
            
            print(f"Found {len(signals)} historical signals for {epic}")
            
        except Exception as e:
            print(f"❌ Error backtesting {epic}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Sort by timestamp and return latest signals
    all_signals.sort(key=lambda x: x['timestamp'], reverse=True)
    return all_signals[:num_signals]

def run_historical_analysis(engine, epic_list, enhancement_function, num_signals=10):
    """
    Run complete historical analysis of EMA signals (for MID prices)
    """
    print("🔍 Running Historical EMA Signal Analysis...")
    print(f"📋 Analyzing: {epic_list}")
    print(f"🎯 Looking for last {num_signals} signals")
    print("=" * 60)
    
    # Get historical signals
    signals = get_latest_historical_signals(engine, epic_list, enhancement_function, num_signals, use_5min=True)
    
    # Display results
    display_historical_signals(signals)
    
    # Performance analysis
    analyze_signal_performance(signals)
    
    return signals

def scan_epics_for_signals(engine, epic_list, enhancement_function, use_bid_adjustment=True, spread_pips=1.5):
    """
    Scan multiple epics for trading signals
    
    Args:
        engine: Database engine
        epic_list: List of epic codes to scan
        enhancement_function: Function to enhance candle data
        use_bid_adjustment: Whether to use BID price adjustment
        spread_pips: Spread in pips for BID adjustment
    
    Returns:
        List of detected signals
    """
    signals = []
    
    for epic in epic_list:
        try:
            print(f"Scanning {epic}...")
            
            # Get enhanced 5m data
            df_5m_enhanced, _, _ = enhancement_function(engine, epic, extract_pair_from_epic(epic))
            
            if df_5m_enhanced is None or len(df_5m_enhanced) == 0:
                print(f"❌ No data for {epic}")
                continue
            
            # Detect signals
            if use_bid_adjustment:
                signal = detect_ema_signals_bid_adjusted(df_5m_enhanced, epic, spread_pips)
            else:
                # Add EMA indicators for MID price detection
                df_with_emas = add_ema_indicators(df_5m_enhanced)
                signal = detect_ema_signals(df_with_emas, epic)
            
            if signal:
                signals.append(signal)
                print(f"🚨 {signal['signal_type']} signal detected for {epic}!")
            else:
                print(f"✓ {epic} - No signals")
                
        except Exception as e:
            print(f"❌ Error scanning {epic}: {e}")
            continue
    
    return signals

def send_alert_to_claude_api(signal, api_key=None):
    """
    Send trading signal to Claude API for final analysis
    
    Args:
        signal: Signal dictionary
        api_key: Anthropic API key
    
    Returns:
        Claude's analysis response
    """
    if not api_key:
        print("Warning: No API key provided for Claude analysis")
        return None
    
    # Determine price for display
    price_display = signal.get('price_mid', signal.get('price', 'N/A'))
    
    # Prepare the prompt for Claude
    prompt = f"""
    I have a {signal['signal_type']} trading signal that needs your analysis:
    
    📊 SIGNAL DETAILS:
    Epic: {signal['epic']}
    Signal: {signal['signal_type']}
    Price: {price_display}
    Timestamp: {signal['timestamp']}
    Confidence: {signal['confidence_score']:.2%}
    
    📈 EMA VALUES:
    EMA 9: {signal['ema_9']:.5f}
    EMA 21: {signal['ema_21']:.5f}
    EMA 200: {signal['ema_200']:.5f}
    
    📋 MARKET CONTEXT:
    Volume Ratio: {signal.get('volume_ratio', 'N/A')}
    Distance to Support: {signal.get('distance_to_support_pips', 'N/A')} pips
    Distance to Resistance: {signal.get('distance_to_resistance_pips', 'N/A')} pips
    Trend Alignment: {signal.get('trend_alignment', 'N/A')}
    
    Please analyze this signal and provide:
    1. Overall signal validity (VALID/INVALID/WEAK)
    2. Key strengths and weaknesses
    3. Risk assessment
    4. Entry/exit recommendations
    5. Overall confidence rating (1-10)
    
    Keep the analysis concise and actionable for a 5-minute timeframe trade.
    """
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': 'claude-3-sonnet-20240229',
            'max_tokens': 1000,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['content'][0]['text']
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error sending to Claude API: {e}")
        return None

def run_trading_scanner(engine, epic_list, enhancement_function, api_key=None, scan_interval=60, use_bid_adjustment=True, spread_pips=1.5):
    """
    Main trading scanner that runs continuously
    
    Args:
        engine: Database engine
        epic_list: List of epics to scan
        enhancement_function: Data enhancement function
        api_key: Claude API key for analysis
        scan_interval: Seconds between scans
        use_bid_adjustment: Whether to use BID price adjustment
        spread_pips: Spread in pips for BID adjustment
    """
    print(f"🚀 Starting 5-minute EMA trading scanner...")
    print(f"📋 Scanning {len(epic_list)} epics: {epic_list}")
    print(f"⏰ Scan interval: {scan_interval} seconds")
    print(f"🔧 BID Adjustment: {'ON' if use_bid_adjustment else 'OFF'}")
    if use_bid_adjustment:
        print(f"📊 Spread: {spread_pips} pips")
    print("=" * 60)
    
    last_signals = {}  # Track last signal time for each epic to avoid duplicates
    
    while True:
        try:
            print(f"\n🔍 Scanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Scan for signals
            signals = scan_epics_for_signals(engine, epic_list, enhancement_function, use_bid_adjustment, spread_pips)
            
            for signal in signals:
                epic = signal['epic']
                signal_time = signal['timestamp']
                
                # Check if this is a new signal (avoid duplicates)
                if epic not in last_signals or last_signals[epic] != signal_time:
                    last_signals[epic] = signal_time
                    
                    print(f"\n🚨 NEW {signal['signal_type']} ALERT for {epic}")
                    
                    # Display price info
                    if 'price_mid' in signal:
                        print(f"MID Price: {signal['price_mid']:.5f}, Execution: {signal['execution_price']:.5f}")
                    else:
                        print(f"Price: {signal.get('price', 'N/A')}")
                    
                    print(f"Confidence: {signal['confidence_score']:.2%}")
                    
                    # Send to Claude for analysis
                    if api_key:
                        print("📤 Sending to Claude for analysis...")
                        claude_analysis = send_alert_to_claude_api(signal, api_key)
                        
                        if claude_analysis:
                            print("🤖 Claude Analysis:")
                            print(claude_analysis)
                        else:
                            print("❌ Failed to get Claude analysis")
                    else:
                        print("⚠️ No API key - skipping Claude analysis")
                    
                    print("-" * 40)
            
            if not signals:
                print("✓ No new signals detected")
            
            # Wait for next scan
            time.sleep(scan_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped by user")
            break
        except Exception as e:
            print(f"❌ Scanner error: {e}")
            time.sleep(10)  # Wait 10 seconds before retrying

def smart_signal_detection(engine, epic_list, enhancement_function, cutoff_time=None):
    """
    Smart signal detection that handles both old BID and new MID data
    
    Args:
        engine: Database engine
        epic_list: List of epics to analyze
        enhancement_function: Data enhancement function
        cutoff_time: Datetime when new streaming started (default: current time - 2 hours)
    
    Returns:
        List of detected signals
    """
    if cutoff_time is None:
        cutoff_time = datetime.now() - timedelta(hours=2)  # Assume new data started 2 hours ago
    
    signals = []
    
    for epic in epic_list:
        try:
            # Get data
            df_5m, _, _ = enhancement_function(engine, epic, extract_pair_from_epic(epic))
            
            if df_5m is None or len(df_5m) == 0:
                print(f"❌ No data for {epic}")
                continue
            
            # Check data type based on timestamp
            latest_timestamp = df_5m['start_time'].iloc[-1]
            
            if latest_timestamp > cutoff_time:
                # Recent data - should be MID prices
                print(f"✅ {epic}: Using MID prices (new data)")
                df_with_emas = add_ema_indicators(df_5m)
                signal = detect_ema_signals(df_with_emas, epic)
            else:
                # Older data - use BID adjustment
                print(f"🔄 {epic}: Using BID adjustment (old data)")
                signal = detect_ema_signals_bid_adjusted(df_5m, epic)
                
            if signal:
                signals.append(signal)
                
        except Exception as e:
            print(f"❌ Error processing {epic}: {e}")
            continue
    
    return signals

# Example usage
if __name__ == "__main__":
    # Example epic list for major currency pairs
    EPIC_LIST = [
        'CS.D.EURUSD.MINI.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.AUDUSD.MINI.IP'
    ]
    
    # Set your Claude API key here
    CLAUDE_API_KEY = "your_anthropic_api_key_here"  # Replace with your actual API key
    
    # Example usage:
    # signals = run_historical_analysis_bid_adjusted(
    #     engine=engine,
    #     epic_list=EPIC_LIST,
    #     enhancement_function=enhance_candle_data_complete,
    #     num_signals=15,
    #     spread_pips=1.5
    # )
    
    # Run the live scanner:
    # run_trading_scanner(
    #     engine=engine,
    #     epic_list=EPIC_LIST,
    #     enhancement_function=enhance_candle_data_complete,
    #     api_key=CLAUDE_API_KEY,
    #     scan_interval=60,
    #     use_bid_adjustment=True,
    #     spread_pips=1.5
    # )