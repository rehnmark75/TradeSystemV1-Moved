# backtesting.py

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time

from services.data_utils import *
from services.ema_signals import *
from services.enhance_data import *

def get_epic_settings(epic):
    """
    Get epic-specific settings for accurate signal detection
    
    Args:
        epic: IG epic code
    
    Returns:
        dict with epic-specific settings
    """
    # Extract currency pair from epic
    pair = extract_pair_from_epic(epic)
    
    # Epic-specific settings based on typical IG Markets spreads
    settings = {
        'EURUSD': {'spread_pips': 1.2, 'pip_multiplier': 10000, 'min_move_pips': 0.5, 'volatility': 'medium'},
        'GBPUSD': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDJPY': {'spread_pips': 1.0, 'pip_multiplier': 100, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'USDCHF': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'low'},
        'AUDUSD': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDCAD': {'spread_pips': 2.0, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'medium'},
        'NZDUSD': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'high'},
        'EURGBP': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.6, 'volatility': 'low'},
        'EURJPY': {'spread_pips': 1.5, 'pip_multiplier': 100, 'min_move_pips': 1.5, 'volatility': 'medium'},
        'GBPJPY': {'spread_pips': 2.0, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'very_high'},
        'CHFJPY': {'spread_pips': 2.5, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'high'},
        'AUDCAD': {'spread_pips': 2.8, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'AUDCHF': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'AUDJPY': {'spread_pips': 2.2, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'high'},
        'CADCHF': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'low'},
        'CADJPY': {'spread_pips': 2.5, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'medium'},
        'EURAUD': {'spread_pips': 2.2, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'EURCAD': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'EURCHF': {'spread_pips': 2.0, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'low'},
        'EURNZD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.5, 'volatility': 'high'},
        'GBPAUD': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'high'},
        'GBPCAD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'high'},
        'GBPCHF': {'spread_pips': 2.8, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'GBPNZD': {'spread_pips': 4.0, 'pip_multiplier': 10000, 'min_move_pips': 1.5, 'volatility': 'very_high'},
        'NZDCAD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'medium'},
        'NZDCHF': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'medium'},
        'NZDJPY': {'spread_pips': 2.8, 'pip_multiplier': 100, 'min_move_pips': 2.5, 'volatility': 'high'},
    }
    
    # Default settings for unknown pairs
    default_settings = {
        'spread_pips': 2.0, 
        'pip_multiplier': 10000, 
        'min_move_pips': 1.0,
        'volatility': 'medium'
    }
    
    return settings.get(pair, default_settings)

def extract_pair_from_epic(epic):
    """Extract currency pair from IG epic format"""
    parts = epic.split('.')
    if len(parts) >= 3:
        return parts[2]  # Usually the currency pair
    return 'EURUSD'  # Default fallback

def backtest_ema_signals_bid_adjusted_dynamic(df, epic, lookback_bars=1000, timeframe="5m"):
    """
    Backtest EMA signals with dynamic epic-specific BID price adjustment
    
    Args:
        df: Enhanced dataframe with BID prices
        epic: Epic code 
        lookback_bars: How many bars to look back
        timeframe: Timeframe being analyzed
    
    Returns:
        List of historical signals with epic-specific settings
    """
    if df is None or len(df) < 200:
        return []
    
    # Get epic-specific settings
    settings = get_epic_settings(epic)
    spread_pips = settings['spread_pips']
    pip_multiplier = settings['pip_multiplier']
    min_move_pips = settings['min_move_pips']
    
    print(f"📊 {epic} Dynamic Settings:")
    print(f"   Spread: {spread_pips} pips")
    print(f"   Pip Multiplier: {pip_multiplier}")
    print(f"   Min Move: {min_move_pips} pips")
    print(f"   Volatility: {settings['volatility']}")
    
    # Adjust BID to MID prices using dynamic spread
    spread = spread_pips / pip_multiplier
    df_mid = df.copy()
    df_mid['open'] = df['open'] + spread/2
    df_mid['high'] = df['high'] + spread/2
    df_mid['low'] = df['low'] + spread/2
    df_mid['close'] = df['close'] + spread/2
    
    # Add EMA indicators
    from services.ema_signals import add_ema_indicators
    df_with_emas = add_ema_indicators(df_mid)
    
    signals = []
    start_idx = max(200, len(df_with_emas) - lookback_bars)
    
    print(f"Backtesting {epic} with dynamic settings (spread: {spread_pips} pips)")
    
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
                'new_signal': prev_price_mid <= ema_9_prev and current_price_mid > ema_9_current,
                'significant_move': abs((current_price_mid - prev_price_mid) * pip_multiplier) >= min_move_pips
            }
            
            # Check for Bear Signal  
            bear_conditions = {
                'price_below_ema9': current_price_mid < ema_9_current,
                'ema21_above_ema9': ema_21_current > ema_9_current,
                'ema200_above_ema9': ema_200_current > ema_9_current,
                'ema200_above_ema21': ema_200_current > ema_21_current,
                'new_signal': prev_price_mid >= ema_9_prev and current_price_mid < ema_9_current,
                'significant_move': abs((current_price_mid - prev_price_mid) * pip_multiplier) >= min_move_pips
            }
            
            signal_type = None
            confidence_score = 0
            
            if all(bull_conditions.values()):
                signal_type = 'BULL'
                ema_separation = abs(ema_9_current - ema_21_current) / current_price_mid * pip_multiplier
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                
                # Adjust confidence based on volatility
                volatility_bonus = {'low': 0.1, 'medium': 0.05, 'high': 0.0, 'very_high': -0.05}.get(settings['volatility'], 0)
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2) + volatility_bonus)
                
            elif all(bear_conditions.values()):
                signal_type = 'BEAR'
                ema_separation = abs(ema_21_current - ema_9_current) / current_price_mid * pip_multiplier
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                
                # Adjust confidence based on volatility
                volatility_bonus = {'low': 0.1, 'medium': 0.05, 'high': 0.0, 'very_high': -0.05}.get(settings['volatility'], 0)
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2) + volatility_bonus)
            
            if signal_type:
                # Calculate potential profit/loss using MID prices and dynamic pip multiplier
                next_bars = df_with_emas.iloc[i+1:i+21] if i+21 < len(df_with_emas) else df_with_emas.iloc[i+1:]
                
                if len(next_bars) > 0:
                    if signal_type == 'BULL':
                        max_profit = (next_bars['high'].max() - current_price_mid) * pip_multiplier
                        max_loss = (current_price_mid - next_bars['low'].min()) * pip_multiplier
                    else:
                        max_profit = (current_price_mid - next_bars['low'].min()) * pip_multiplier
                        max_loss = (next_bars['high'].max() - current_price_mid) * pip_multiplier
                else:
                    max_profit = max_loss = 0
                
                # Calculate execution price with dynamic spread
                if signal_type == 'BULL':
                    execution_price = current_price_mid + (spread / 2)  # ASK price
                else:
                    execution_price = current_price_mid - (spread / 2)  # BID price
                
                signal_data = {
                    'signal_type': signal_type,
                    'epic': epic,
                    'timeframe': timeframe,
                    'timestamp': current['start_time'],
                    'price_mid': current_price_mid,
                    'price_bid': original_current['close'],
                    'execution_price': execution_price,
                    'spread_pips': spread_pips,
                    'pip_multiplier': pip_multiplier,
                    'min_move_pips': min_move_pips,
                    'volatility_rating': settings['volatility'],
                    'ema_9': ema_9_current,
                    'ema_21': ema_21_current,
                    'ema_200': ema_200_current,
                    'confidence_score': confidence_score,
                    'volume_ratio': current.get('volume_ratio_20', None),
                    'max_profit_pips': max_profit,
                    'max_loss_pips': max_loss,
                    'risk_reward_potential': max_profit / max_loss if max_loss > 0 else 0,
                    'profit_after_spread': max_profit - spread_pips,  # Real profit after spread
                    'loss_including_spread': max_loss + spread_pips   # Real loss including spread
                }
                
                signals.append(signal_data)
                
        except Exception as e:
            continue
    
    return signals

def run_historical_analysis_bid_adjusted_dynamic(engine, epic_list, enhancement_function, num_signals=15):
    """
    Run historical analysis with dynamic epic-specific settings
    
    Args:
        engine: Database engine
        epic_list: List of epics to analyze
        enhancement_function: Data enhancement function
        num_signals: Number of latest signals to return
    
    Returns:
        List of signals with epic-specific analysis
    """
    print(f"🔍 Running Dynamic Epic-Specific BID-Adjusted Analysis...")
    print(f"📋 Analyzing {len(epic_list)} epics with individual settings")
    print("=" * 70)
    
    all_signals = []
    epic_summaries = []
    
    for epic in epic_list:
        try:
            print(f"\n🔍 Analyzing {epic}...")
            
            # Get epic-specific settings
            settings = get_epic_settings(epic)
            pair = extract_pair_from_epic(epic)
            
            # Get enhanced data
            df_5m, _, _ = enhancement_function(engine, epic, pair)
            
            if df_5m is None or len(df_5m) == 0:
                print(f"❌ No data available for {epic}")
                continue
                
            print(f"✓ Got {len(df_5m)} 5m candles")
            
            # Get signals with dynamic settings
            signals = backtest_ema_signals_bid_adjusted_dynamic(
                df_5m, epic, 
                lookback_bars=1000
            )
            
            # Add epic summary
            epic_summaries.append({
                'epic': epic,
                'pair': pair,
                'total_signals': len(signals),
                'bull_signals': len([s for s in signals if s['signal_type'] == 'BULL']),
                'bear_signals': len([s for s in signals if s['signal_type'] == 'BEAR']),
                'avg_confidence': sum(s['confidence_score'] for s in signals) / len(signals) if signals else 0,
                'avg_spread': settings['spread_pips'],
                'volatility': settings['volatility']
            })
            
            all_signals.extend(signals)
            print(f"Found {len(signals)} signals for {epic}")
            
        except Exception as e:
            print(f"❌ Error backtesting {epic}: {e}")
            continue
    
    # Sort by timestamp and get latest signals
    all_signals.sort(key=lambda x: x['timestamp'], reverse=True)
    latest_signals = all_signals[:num_signals]
    
    # Display results with epic-specific info
    display_dynamic_signals(latest_signals)
    display_epic_summaries(epic_summaries)
    analyze_dynamic_performance(latest_signals)
    
    return latest_signals

def display_dynamic_signals(signals):
    """
    Display signals with dynamic epic-specific information
    """
    if not signals:
        print("No historical signals found")
        return
    
    print(f"\n📊 LAST {len(signals)} DYNAMIC EMA SIGNALS:")
    print("=" * 100)
    
    for i, signal in enumerate(signals, 1):
        print(f"\n#{i} - {signal['signal_type']} SIGNAL")
        print(f"Epic: {signal['epic']} ({signal.get('volatility_rating', 'unknown')} volatility)")
        print(f"Time: {signal['timestamp']}")
        
        # Price information
        print(f"MID Price: {signal['price_mid']:.5f}")
        print(f"BID Price: {signal['price_bid']:.5f}")
        print(f"Execution Price: {signal['execution_price']:.5f}")
        print(f"Spread: {signal['spread_pips']:.1f} pips")
        print(f"Min Move: {signal.get('min_move_pips', 'N/A')} pips")
        
        # EMA and confidence
        print(f"EMAs: 9={signal.get('ema_9', 0):.5f} | 21={signal.get('ema_21', 0):.5f} | 200={signal.get('ema_200', 0):.5f}")
        print(f"Confidence: {signal.get('confidence_score', 0):.1%}")
        
        # Volume if available
        if signal.get('volume_ratio'):
            print(f"Volume: {signal['volume_ratio']:.2f}x average")
        
        # Realistic P&L (including spread)
        if signal.get('max_profit_pips') is not None:
            real_profit = signal.get('profit_after_spread', signal['max_profit_pips'])
            real_loss = signal.get('loss_including_spread', signal['max_loss_pips'])
            
            print(f"📈 Max Profit (after spread): +{real_profit:.1f} pips")
            print(f"📉 Max Loss (including spread): -{real_loss:.1f} pips")
            
            if real_loss > 0:
                real_rr = real_profit / real_loss
                print(f"🎯 Realistic R:R: 1:{real_rr:.2f}")
        
        print("-" * 50)

def display_epic_summaries(epic_summaries):
    """
    Display summary statistics for each epic
    """
    print(f"\n📋 EPIC PERFORMANCE SUMMARY:")
    print("=" * 80)
    print(f"{'Epic':<20} {'Signals':<8} {'Bull':<6} {'Bear':<6} {'Avg Conf':<10} {'Spread':<8} {'Volatility'}")
    print("-" * 80)
    
    for summary in epic_summaries:
        print(f"{summary['epic']:<20} "
              f"{summary['total_signals']:<8} "
              f"{summary['bull_signals']:<6} "
              f"{summary['bear_signals']:<6} "
              f"{summary['avg_confidence']:<10.1%} "
              f"{summary['avg_spread']:<8.1f} "
              f"{summary['volatility']}")

def analyze_dynamic_performance(signals):
    """
    Analyze performance with epic-specific considerations
    """
    if not signals:
        return
    
    print(f"\n📊 DYNAMIC PERFORMANCE ANALYSIS:")
    print("=" * 50)
    
    # Group by epic for analysis
    epic_groups = {}
    for signal in signals:
        epic = signal['epic']
        if epic not in epic_groups:
            epic_groups[epic] = []
        epic_groups[epic].append(signal)
    
    # Overall stats
    bull_signals = [s for s in signals if s['signal_type'] == 'BULL']
    bear_signals = [s for s in signals if s['signal_type'] == 'BEAR']
    
    print(f"Total Signals: {len(signals)}")
    print(f"Bull Signals: {len(bull_signals)}")
    print(f"Bear Signals: {len(bear_signals)}")
    
    if signals:
        avg_confidence = sum(s['confidence_score'] for s in signals) / len(signals)
        
        # Calculate realistic profits (after spread)
        realistic_profits = [s.get('profit_after_spread', s['max_profit_pips']) for s in signals]
        realistic_losses = [s.get('loss_including_spread', s['max_loss_pips']) for s in signals]
        
        avg_realistic_profit = sum(realistic_profits) / len(realistic_profits)
        avg_realistic_loss = sum(realistic_losses) / len(realistic_losses)
        
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Average Realistic Profit: {avg_realistic_profit:.1f} pips")
        print(f"Average Realistic Loss: {avg_realistic_loss:.1f} pips")
        
        # Win rate analysis with realistic spreads
        profit_target = 15  # Lower target due to spreads
        stop_loss = 10
        
        profitable_signals = len([s for s in signals if s.get('profit_after_spread', s['max_profit_pips']) >= profit_target])
        losing_signals = len([s for s in signals if s.get('loss_including_spread', s['max_loss_pips']) >= stop_loss])
        
        print(f"Signals reaching +{profit_target} pips (after spread): {profitable_signals}/{len(signals)} ({profitable_signals/len(signals)*100:.1f}%)")
        print(f"Signals hitting -{stop_loss} pips (including spread): {losing_signals}/{len(signals)} ({losing_signals/len(signals)*100:.1f}%)")
        
        # Epic-specific performance
        print(f"\n📈 PER-EPIC PERFORMANCE:")
        for epic, epic_signals in epic_groups.items():
            epic_profits = [s.get('profit_after_spread', s['max_profit_pips']) for s in epic_signals]
            epic_avg_profit = sum(epic_profits) / len(epic_profits) if epic_profits else 0
            epic_spread = epic_signals[0]['spread_pips'] if epic_signals else 0
            
            print(f"{epic}: {len(epic_signals)} signals, Avg profit: {epic_avg_profit:.1f} pips, Spread: {epic_spread:.1f} pips")

# Example usage function
def test_dynamic_backtesting(engine, epic_list, enhancement_function):
    """
    Test the dynamic backtesting system
    """
    print("🧪 Testing Dynamic Epic-Specific Backtesting...")
    
    signals = run_historical_analysis_bid_adjusted_dynamic(
        engine=engine,
        epic_list=epic_list,
        enhancement_function=enhancement_function,
        num_signals=20
    )
    
    return signals

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta

def get_epic_settings(epic):
    """
    Get epic-specific settings for accurate signal detection
    
    Args:
        epic: IG epic code
    
    Returns:
        dict with epic-specific settings
    """
    # Extract currency pair from epic
    pair = extract_pair_from_epic(epic)
    
    # Epic-specific settings based on typical IG Markets spreads
    settings = {
        'EURUSD': {'spread_pips': 1.2, 'pip_multiplier': 10000, 'min_move_pips': 0.5, 'volatility': 'medium'},
        'GBPUSD': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDJPY': {'spread_pips': 1.0, 'pip_multiplier': 100, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'USDCHF': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'low'},
        'AUDUSD': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDCAD': {'spread_pips': 2.0, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'medium'},
        'NZDUSD': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'high'},
        'EURGBP': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.6, 'volatility': 'low'},
        'EURJPY': {'spread_pips': 1.5, 'pip_multiplier': 100, 'min_move_pips': 1.5, 'volatility': 'medium'},
        'GBPJPY': {'spread_pips': 2.0, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'very_high'},
        'CHFJPY': {'spread_pips': 2.5, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'high'},
        'AUDCAD': {'spread_pips': 2.8, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'AUDCHF': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'AUDJPY': {'spread_pips': 2.2, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'high'},
        'CADCHF': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'low'},
        'CADJPY': {'spread_pips': 2.5, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'medium'},
        'EURAUD': {'spread_pips': 2.2, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'EURCAD': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'EURCHF': {'spread_pips': 2.0, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'low'},
        'EURNZD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.5, 'volatility': 'high'},
        'GBPAUD': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'high'},
        'GBPCAD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'high'},
        'GBPCHF': {'spread_pips': 2.8, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'GBPNZD': {'spread_pips': 4.0, 'pip_multiplier': 10000, 'min_move_pips': 1.5, 'volatility': 'very_high'},
        'NZDCAD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'medium'},
        'NZDCHF': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'medium'},
        'NZDJPY': {'spread_pips': 2.8, 'pip_multiplier': 100, 'min_move_pips': 2.5, 'volatility': 'high'},
    }
    
    # Default settings for unknown pairs
    default_settings = {
        'spread_pips': 2.0, 
        'pip_multiplier': 10000, 
        'min_move_pips': 1.0,
        'volatility': 'medium'
    }
    
    return settings.get(pair, default_settings)

def extract_pair_from_epic(epic):
    """Extract currency pair from IG epic format"""
    parts = epic.split('.')
    if len(parts) >= 3:
        return parts[2]  # Usually the currency pair
    return 'EURUSD'  # Default fallback

def get_market_session(local_time):
    """
    Determine market session based on local time
    
    Args:
        local_time: datetime object in local timezone
    
    Returns:
        str: Market session name
    """
    hour = local_time.hour
    
    if 0 <= hour <= 6:
        return "Asian Session (Late)"
    elif 7 <= hour <= 8:
        return "Asian/European Overlap" 
    elif 9 <= hour <= 12:
        return "European Session (Morning)"
    elif 13 <= hour <= 16:
        return "European/US Overlap"
    elif 17 <= hour <= 21:
        return "US Session"
    elif 22 <= hour <= 23:
        return "US Session (Late)"
    else:
        return "Quiet Hours"

def run_timezone_aware_backtesting(engine, epic_list, num_signals=20, user_timezone='Europe/Stockholm'):
    """
    Complete timezone-aware backtesting system with dynamic epic settings
    
    Args:
        engine: Database engine
        epic_list: List of epics to analyze
        num_signals: Number of latest signals to return
        user_timezone: User's timezone (default: Europe/Stockholm)
    
    Returns:
        List of signals with timezone and epic-specific analysis
    """
    
    def timezone_enhancement_wrapper(engine, epic, pair):
        """Wrapper to ensure timezone awareness in enhancement function"""
        # Import your enhancement function
        import services.enhance_data_support_resistance as sr
        
        # Call with timezone parameter (modify your function to accept this)
        try:
            # Try with timezone parameter if supported
            return sr.enhance_candle_data_complete(
                engine, epic, pair, 
                user_timezone=user_timezone
            )
        except TypeError:
            # Fallback if your function doesn't support timezone yet
            return sr.enhance_candle_data_complete(engine, epic, pair)
    
    print(f"🕒 TIMEZONE-AWARE DYNAMIC BACKTESTING SYSTEM")
    print(f"📍 User timezone: {user_timezone}")
    print(f"🗄️ Database timezone: UTC")
    print(f"📋 Analyzing {len(epic_list)} epics with individual settings")
    print("=" * 70)
    
    # Run dynamic backtesting with timezone wrapper
    signals = run_historical_analysis_bid_adjusted_dynamic(
        engine=engine,
        epic_list=epic_list,
        enhancement_function=timezone_enhancement_wrapper,
        num_signals=num_signals
    )
    
    # Add timezone information to signals
    signals_with_timezone = add_timezone_info_to_signals(signals, user_timezone)
    
    # Display with timezone info
    display_signals_with_timezone(signals_with_timezone, user_timezone)
    
    return signals_with_timezone

def add_timezone_info_to_signals(signals, user_timezone='Europe/Stockholm'):
    """
    Add timezone information to signal data
    
    Args:
        signals: List of signal dictionaries
        user_timezone: User's timezone
    
    Returns:
        List of signals with timezone information added
    """
    enhanced_signals = []
    local_tz = pytz.timezone(user_timezone)
    
    for signal in signals:
        # Create a copy to avoid modifying original
        enhanced_signal = signal.copy()
        
        # Convert UTC timestamp to local time
        utc_time = pd.to_datetime(signal['timestamp'])
        if utc_time.tz is None:
            utc_time = utc_time.tz_localize('UTC')
        
        local_time = utc_time.tz_convert(local_tz)
        
        # Add timezone fields
        enhanced_signal['timestamp_utc'] = utc_time.strftime('%Y-%m-%d %H:%M:%S UTC')
        enhanced_signal['timestamp_local'] = local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        enhanced_signal['local_hour'] = local_time.hour
        enhanced_signal['market_session'] = get_market_session(local_time)
        enhanced_signal['user_timezone'] = user_timezone
        
        # Add trading session quality rating
        session_quality = {
            "European/US Overlap": "excellent",
            "Asian/European Overlap": "good", 
            "European Session (Morning)": "good",
            "US Session": "good",
            "Asian Session (Late)": "fair",
            "US Session (Late)": "fair",
            "Quiet Hours": "poor"
        }
        enhanced_signal['session_quality'] = session_quality.get(enhanced_signal['market_session'], 'fair')
        
        enhanced_signals.append(enhanced_signal)
    
    return enhanced_signals

def display_signals_with_timezone(signals, user_timezone='Europe/Stockholm'):
    """
    Display signals with comprehensive timezone and session information
    """
    if not signals:
        print("No signals found")
        return
    
    print(f"\n📊 TIMEZONE-AWARE DYNAMIC SIGNALS ({user_timezone}):")
    print("=" * 100)
    
    for i, signal in enumerate(signals, 1):
        print(f"\n#{i} - {signal['signal_type']} SIGNAL")
        print(f"Epic: {signal['epic']} ({signal.get('volatility_rating', 'unknown')} volatility)")
        
        # Timezone information
        print(f"UTC Time: {signal.get('timestamp_utc', signal['timestamp'])}")
        print(f"Local Time: {signal.get('timestamp_local', 'N/A')}")
        print(f"Market Session: {signal.get('market_session', 'Unknown')} ({signal.get('session_quality', 'unknown')} quality)")
        
        # Price information with dynamic spread
        if 'price_mid' in signal:
            print(f"MID Price: {signal['price_mid']:.5f}")
            print(f"BID Price: {signal['price_bid']:.5f}")
            print(f"Execution Price: {signal['execution_price']:.5f}")
            print(f"Spread: {signal['spread_pips']:.1f} pips")
        else:
            print(f"Price: {signal.get('price', 'N/A'):.5f}")
        
        # EMA and confidence
        print(f"EMAs: 9={signal.get('ema_9', 0):.5f} | 21={signal.get('ema_21', 0):.5f} | 200={signal.get('ema_200', 0):.5f}")
        print(f"Confidence: {signal.get('confidence_score', 0):.1%}")
        
        # Volume if available
        if signal.get('volume_ratio'):
            print(f"Volume: {signal['volume_ratio']:.2f}x average")
        
        # Realistic P&L (including spread)
        if signal.get('max_profit_pips') is not None:
            real_profit = signal.get('profit_after_spread', signal['max_profit_pips'])
            real_loss = signal.get('loss_including_spread', signal['max_loss_pips'])
            
            print(f"📈 Max Profit (after spread): +{real_profit:.1f} pips")
            print(f"📉 Max Loss (including spread): -{real_loss:.1f} pips")
            
            if real_loss > 0:
                real_rr = real_profit / real_loss
                print(f"🎯 Realistic R:R: 1:{real_rr:.2f}")
        
        # Session trading recommendation
        session_quality = signal.get('session_quality', 'unknown')
        if session_quality == 'excellent':
            print("💡 Trading Recommendation: ✅ Excellent session for trading")
        elif session_quality == 'good':
            print("💡 Trading Recommendation: ✅ Good session for trading")
        elif session_quality == 'fair':
            print("💡 Trading Recommendation: ⚠️ Fair session - proceed with caution")
        else:
            print("💡 Trading Recommendation: ❌ Poor session - consider avoiding")
        
        print("-" * 50)

def analyze_timezone_performance(signals):
    """
    Analyze signal performance by timezone and market session
    """
    if not signals:
        return
    
    print(f"\n📊 TIMEZONE & SESSION PERFORMANCE ANALYSIS:")
    print("=" * 60)
    
    # Group by market session
    session_groups = {}
    for signal in signals:
        session = signal.get('market_session', 'Unknown')
        if session not in session_groups:
            session_groups[session] = []
        session_groups[session].append(signal)
    
    # Analyze by session
    print(f"{'Session':<25} {'Signals':<8} {'Avg Conf':<10} {'Avg Profit':<12}")
    print("-" * 60)
    
    for session, session_signals in session_groups.items():
        avg_conf = sum(s['confidence_score'] for s in session_signals) / len(session_signals)
        
        # Calculate average realistic profit
        profits = [s.get('profit_after_spread', s['max_profit_pips']) for s in session_signals]
        avg_profit = sum(profits) / len(profits) if profits else 0
        
        print(f"{session:<25} {len(session_signals):<8} {avg_conf:<10.1%} {avg_profit:<12.1f}")
    
    # Best trading hours
    hour_performance = {}
    for signal in signals:
        hour = signal.get('local_hour', 0)
        if hour not in hour_performance:
            hour_performance[hour] = []
        hour_performance[hour].append(signal)
    
    print(f"\n🕒 BEST TRADING HOURS (Local Time):")
    sorted_hours = sorted(hour_performance.items(), 
                         key=lambda x: sum(s.get('profit_after_spread', s['max_profit_pips']) for s in x[1]) / len(x[1]) if x[1] else 0, 
                         reverse=True)
    
    for hour, hour_signals in sorted_hours[:5]:  # Top 5 hours
        profits = [s.get('profit_after_spread', s['max_profit_pips']) for s in hour_signals]
        avg_profit = sum(profits) / len(profits) if profits else 0
        print(f"  {hour:02d}:00 - {len(hour_signals)} signals, avg profit: {avg_profit:.1f} pips")

# Enhanced version of the main backtesting function
def run_historical_analysis_bid_adjusted_dynamic(engine, epic_list, enhancement_function, num_signals=15):
    """
    Run historical analysis with dynamic epic-specific settings
    (Same as before but with timezone awareness built in)
    """
    print(f"🔍 Running Dynamic Epic-Specific BID-Adjusted Analysis...")
    print(f"📋 Analyzing {len(epic_list)} epics with individual settings")
    print("=" * 70)
    
    all_signals = []
    epic_summaries = []
    
    for epic in epic_list:
        try:
            print(f"\n🔍 Analyzing {epic}...")
            
            # Get epic-specific settings
            settings = get_epic_settings(epic)
            pair = extract_pair_from_epic(epic)
            
            # Get enhanced data (timezone-aware if supported)
            df_5m, _, _ = enhancement_function(engine, epic, pair)
            
            if df_5m is None or len(df_5m) == 0:
                print(f"❌ No data available for {epic}")
                continue
                
            print(f"✓ Got {len(df_5m)} 5m candles")
            
            # Get signals with dynamic settings
            signals = backtest_ema_signals_bid_adjusted_dynamic(
                df_5m, epic, 
                lookback_bars=1000
            )
            
            # Add epic summary
            epic_summaries.append({
                'epic': epic,
                'pair': pair,
                'total_signals': len(signals),
                'bull_signals': len([s for s in signals if s['signal_type'] == 'BULL']),
                'bear_signals': len([s for s in signals if s['signal_type'] == 'BEAR']),
                'avg_confidence': sum(s['confidence_score'] for s in signals) / len(signals) if signals else 0,
                'avg_spread': settings['spread_pips'],
                'volatility': settings['volatility']
            })
            
            all_signals.extend(signals)
            print(f"Found {len(signals)} signals for {epic}")
            
        except Exception as e:
            print(f"❌ Error backtesting {epic}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Sort by timestamp and get latest signals
    all_signals.sort(key=lambda x: x['timestamp'], reverse=True)
    latest_signals = all_signals[:num_signals]
    
    return latest_signals

def get_epic_settings(epic):
    """
    Get epic-specific settings for accurate signal detection
    
    Args:
        epic: IG epic code
    
    Returns:
        dict with epic-specific settings
    """
    # Extract currency pair from epic
    pair = extract_pair_from_epic(epic)
    
    # Epic-specific settings based on typical IG Markets spreads
    settings = {
        'EURUSD': {'spread_pips': 1.2, 'pip_multiplier': 10000, 'min_move_pips': 0.5, 'volatility': 'medium'},
        'GBPUSD': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDJPY': {'spread_pips': 1.0, 'pip_multiplier': 100, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'USDCHF': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'low'},
        'AUDUSD': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDCAD': {'spread_pips': 2.0, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'medium'},
        'NZDUSD': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'high'},
        'EURGBP': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.6, 'volatility': 'low'},
        'EURJPY': {'spread_pips': 1.5, 'pip_multiplier': 100, 'min_move_pips': 1.5, 'volatility': 'medium'},
        'GBPJPY': {'spread_pips': 2.0, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'very_high'},
        'CHFJPY': {'spread_pips': 2.5, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'high'},
        'AUDCAD': {'spread_pips': 2.8, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'AUDCHF': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'AUDJPY': {'spread_pips': 2.2, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'high'},
        'CADCHF': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'low'},
        'CADJPY': {'spread_pips': 2.5, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'medium'},
        'EURAUD': {'spread_pips': 2.2, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'EURCAD': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'EURCHF': {'spread_pips': 2.0, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'low'},
        'EURNZD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.5, 'volatility': 'high'},
        'GBPAUD': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'high'},
        'GBPCAD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'high'},
        'GBPCHF': {'spread_pips': 2.8, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'GBPNZD': {'spread_pips': 4.0, 'pip_multiplier': 10000, 'min_move_pips': 1.5, 'volatility': 'very_high'},
        'NZDCAD': {'spread_pips': 3.5, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'medium'},
        'NZDCHF': {'spread_pips': 3.0, 'pip_multiplier': 10000, 'min_move_pips': 1.2, 'volatility': 'medium'},
        'NZDJPY': {'spread_pips': 2.8, 'pip_multiplier': 100, 'min_move_pips': 2.5, 'volatility': 'high'},
    }
    
    # Default settings for unknown pairs
    default_settings = {
        'spread_pips': 2.0, 
        'pip_multiplier': 10000, 
        'min_move_pips': 1.0,
        'volatility': 'medium'
    }
    
    return settings.get(pair, default_settings)

def extract_pair_from_epic(epic):
    """Extract currency pair from IG epic format"""
    parts = epic.split('.')
    if len(parts) >= 3:
        return parts[2]  # Usually the currency pair
    return 'EURUSD'  # Default fallback

def backtest_ema_signals_bid_adjusted_dynamic(df, epic, lookback_bars=1000, timeframe="5m"):
    """
    Backtest EMA signals with dynamic epic-specific BID price adjustment
    
    Args:
        df: Enhanced dataframe with BID prices
        epic: Epic code 
        lookback_bars: How many bars to look back
        timeframe: Timeframe being analyzed
    
    Returns:
        List of historical signals with epic-specific settings
    """
    if df is None or len(df) < 200:
        return []
    
    # Get epic-specific settings
    settings = get_epic_settings(epic)
    spread_pips = settings['spread_pips']
    pip_multiplier = settings['pip_multiplier']
    min_move_pips = settings['min_move_pips']
    
    print(f"📊 {epic} Dynamic Settings:")
    print(f"   Spread: {spread_pips} pips")
    print(f"   Pip Multiplier: {pip_multiplier}")
    print(f"   Min Move: {min_move_pips} pips")
    print(f"   Volatility: {settings['volatility']}")
    
    # Adjust BID to MID prices using dynamic spread
    spread = spread_pips / pip_multiplier
    df_mid = df.copy()
    df_mid['open'] = df['open'] + spread/2
    df_mid['high'] = df['high'] + spread/2
    df_mid['low'] = df['low'] + spread/2
    df_mid['close'] = df['close'] + spread/2
    
    # Add EMA indicators
    from services.ema_signals import add_ema_indicators
    df_with_emas = add_ema_indicators(df_mid)
    
    signals = []
    start_idx = max(200, len(df_with_emas) - lookback_bars)
    
    print(f"Backtesting {epic} with dynamic settings (spread: {spread_pips} pips)")
    
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
                'new_signal': prev_price_mid <= ema_9_prev and current_price_mid > ema_9_current,
                'significant_move': abs((current_price_mid - prev_price_mid) * pip_multiplier) >= min_move_pips
            }
            
            # Check for Bear Signal  
            bear_conditions = {
                'price_below_ema9': current_price_mid < ema_9_current,
                'ema21_above_ema9': ema_21_current > ema_9_current,
                'ema200_above_ema9': ema_200_current > ema_9_current,
                'ema200_above_ema21': ema_200_current > ema_21_current,
                'new_signal': prev_price_mid >= ema_9_prev and current_price_mid < ema_9_current,
                'significant_move': abs((current_price_mid - prev_price_mid) * pip_multiplier) >= min_move_pips
            }
            
            signal_type = None
            confidence_score = 0
            
            if all(bull_conditions.values()):
                signal_type = 'BULL'
                ema_separation = abs(ema_9_current - ema_21_current) / current_price_mid * pip_multiplier
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                
                # Adjust confidence based on volatility
                volatility_bonus = {'low': 0.1, 'medium': 0.05, 'high': 0.0, 'very_high': -0.05}.get(settings['volatility'], 0)
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2) + volatility_bonus)
                
            elif all(bear_conditions.values()):
                signal_type = 'BEAR'
                ema_separation = abs(ema_21_current - ema_9_current) / current_price_mid * pip_multiplier
                volume_strength = current.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in current else 1.0
                
                # Adjust confidence based on volatility
                volatility_bonus = {'low': 0.1, 'medium': 0.05, 'high': 0.0, 'very_high': -0.05}.get(settings['volatility'], 0)
                confidence_score = min(0.95, 0.5 + (ema_separation * 0.05) + (max(0, volume_strength - 1) * 0.2) + volatility_bonus)
            
            if signal_type:
                # Calculate potential profit/loss using MID prices and dynamic pip multiplier
                next_bars = df_with_emas.iloc[i+1:i+21] if i+21 < len(df_with_emas) else df_with_emas.iloc[i+1:]
                
                if len(next_bars) > 0:
                    if signal_type == 'BULL':
                        max_profit = (next_bars['high'].max() - current_price_mid) * pip_multiplier
                        max_loss = (current_price_mid - next_bars['low'].min()) * pip_multiplier
                    else:
                        max_profit = (current_price_mid - next_bars['low'].min()) * pip_multiplier
                        max_loss = (next_bars['high'].max() - current_price_mid) * pip_multiplier
                else:
                    max_profit = max_loss = 0
                
                # Calculate execution price with dynamic spread
                if signal_type == 'BULL':
                    execution_price = current_price_mid + (spread / 2)  # ASK price
                else:
                    execution_price = current_price_mid - (spread / 2)  # BID price
                
                signal_data = {
                    'signal_type': signal_type,
                    'epic': epic,
                    'timeframe': timeframe,
                    'timestamp': current['start_time'],
                    'price_mid': current_price_mid,
                    'price_bid': original_current['close'],
                    'execution_price': execution_price,
                    'spread_pips': spread_pips,
                    'pip_multiplier': pip_multiplier,
                    'min_move_pips': min_move_pips,
                    'volatility_rating': settings['volatility'],
                    'ema_9': ema_9_current,
                    'ema_21': ema_21_current,
                    'ema_200': ema_200_current,
                    'confidence_score': confidence_score,
                    'volume_ratio': current.get('volume_ratio_20', None),
                    'max_profit_pips': max_profit,
                    'max_loss_pips': max_loss,
                    'risk_reward_potential': max_profit / max_loss if max_loss > 0 else 0,
                    'profit_after_spread': max_profit - spread_pips,  # Real profit after spread
                    'loss_including_spread': max_loss + spread_pips   # Real loss including spread
                }
                
                signals.append(signal_data)
                
        except Exception as e:
            continue
    
    return signals

def run_historical_analysis_bid_adjusted_dynamic(engine, epic_list, enhancement_function, num_signals=15):
    """
    Run historical analysis with dynamic epic-specific settings
    
    Args:
        engine: Database engine
        epic_list: List of epics to analyze
        enhancement_function: Data enhancement function
        num_signals: Number of latest signals to return
    
    Returns:
        List of signals with epic-specific analysis
    """
    print(f"🔍 Running Dynamic Epic-Specific BID-Adjusted Analysis...")
    print(f"📋 Analyzing {len(epic_list)} epics with individual settings")
    print("=" * 70)
    
    all_signals = []
    epic_summaries = []
    
    for epic in epic_list:
        try:
            print(f"\n🔍 Analyzing {epic}...")
            
            # Get epic-specific settings
            settings = get_epic_settings(epic)
            pair = extract_pair_from_epic(epic)
            
            # Get enhanced data
            df_5m, _, _ = enhancement_function(engine, epic, pair)
            
            if df_5m is None or len(df_5m) == 0:
                print(f"❌ No data available for {epic}")
                continue
                
            print(f"✓ Got {len(df_5m)} 5m candles")
            
            # Get signals with dynamic settings
            signals = backtest_ema_signals_bid_adjusted_dynamic(
                df_5m, epic, 
                lookback_bars=1000
            )
            
            # Add epic summary
            epic_summaries.append({
                'epic': epic,
                'pair': pair,
                'total_signals': len(signals),
                'bull_signals': len([s for s in signals if s['signal_type'] == 'BULL']),
                'bear_signals': len([s for s in signals if s['signal_type'] == 'BEAR']),
                'avg_confidence': sum(s['confidence_score'] for s in signals) / len(signals) if signals else 0,
                'avg_spread': settings['spread_pips'],
                'volatility': settings['volatility']
            })
            
            all_signals.extend(signals)
            print(f"Found {len(signals)} signals for {epic}")
            
        except Exception as e:
            print(f"❌ Error backtesting {epic}: {e}")
            continue
    
    # Sort by timestamp and get latest signals
    all_signals.sort(key=lambda x: x['timestamp'], reverse=True)
    latest_signals = all_signals[:num_signals]
    
    # Display results with epic-specific info
    display_dynamic_signals(latest_signals)
    display_epic_summaries(epic_summaries)
    analyze_dynamic_performance(latest_signals)
    
    return latest_signals

def display_dynamic_signals(signals):
    """
    Display signals with dynamic epic-specific information
    """
    if not signals:
        print("No historical signals found")
        return
    
    print(f"\n📊 LAST {len(signals)} DYNAMIC EMA SIGNALS:")
    print("=" * 100)
    
    for i, signal in enumerate(signals, 1):
        print(f"\n#{i} - {signal['signal_type']} SIGNAL")
        print(f"Epic: {signal['epic']} ({signal.get('volatility_rating', 'unknown')} volatility)")
        print(f"Time: {signal['timestamp']}")
        
        # Price information
        print(f"MID Price: {signal['price_mid']:.5f}")
        print(f"BID Price: {signal['price_bid']:.5f}")
        print(f"Execution Price: {signal['execution_price']:.5f}")
        print(f"Spread: {signal['spread_pips']:.1f} pips")
        print(f"Min Move: {signal.get('min_move_pips', 'N/A')} pips")
        
        # EMA and confidence
        print(f"EMAs: 9={signal.get('ema_9', 0):.5f} | 21={signal.get('ema_21', 0):.5f} | 200={signal.get('ema_200', 0):.5f}")
        print(f"Confidence: {signal.get('confidence_score', 0):.1%}")
        
        # Volume if available
        if signal.get('volume_ratio'):
            print(f"Volume: {signal['volume_ratio']:.2f}x average")
        
        # Realistic P&L (including spread)
        if signal.get('max_profit_pips') is not None:
            real_profit = signal.get('profit_after_spread', signal['max_profit_pips'])
            real_loss = signal.get('loss_including_spread', signal['max_loss_pips'])
            
            print(f"📈 Max Profit (after spread): +{real_profit:.1f} pips")
            print(f"📉 Max Loss (including spread): -{real_loss:.1f} pips")
            
            if real_loss > 0:
                real_rr = real_profit / real_loss
                print(f"🎯 Realistic R:R: 1:{real_rr:.2f}")
        
        print("-" * 50)

def display_epic_summaries(epic_summaries):
    """
    Display summary statistics for each epic
    """
    print(f"\n📋 EPIC PERFORMANCE SUMMARY:")
    print("=" * 80)
    print(f"{'Epic':<20} {'Signals':<8} {'Bull':<6} {'Bear':<6} {'Avg Conf':<10} {'Spread':<8} {'Volatility'}")
    print("-" * 80)
    
    for summary in epic_summaries:
        print(f"{summary['epic']:<20} "
              f"{summary['total_signals']:<8} "
              f"{summary['bull_signals']:<6} "
              f"{summary['bear_signals']:<6} "
              f"{summary['avg_confidence']:<10.1%} "
              f"{summary['avg_spread']:<8.1f} "
              f"{summary['volatility']}")

def analyze_dynamic_performance(signals):
    """
    Analyze performance with epic-specific considerations
    """
    if not signals:
        return
    
    print(f"\n📊 DYNAMIC PERFORMANCE ANALYSIS:")
    print("=" * 50)
    
    # Group by epic for analysis
    epic_groups = {}
    for signal in signals:
        epic = signal['epic']
        if epic not in epic_groups:
            epic_groups[epic] = []
        epic_groups[epic].append(signal)
    
    # Overall stats
    bull_signals = [s for s in signals if s['signal_type'] == 'BULL']
    bear_signals = [s for s in signals if s['signal_type'] == 'BEAR']
    
    print(f"Total Signals: {len(signals)}")
    print(f"Bull Signals: {len(bull_signals)}")
    print(f"Bear Signals: {len(bear_signals)}")
    
    if signals:
        avg_confidence = sum(s['confidence_score'] for s in signals) / len(signals)
        
        # Calculate realistic profits (after spread)
        realistic_profits = [s.get('profit_after_spread', s['max_profit_pips']) for s in signals]
        realistic_losses = [s.get('loss_including_spread', s['max_loss_pips']) for s in signals]
        
        avg_realistic_profit = sum(realistic_profits) / len(realistic_profits)
        avg_realistic_loss = sum(realistic_losses) / len(realistic_losses)
        
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Average Realistic Profit: {avg_realistic_profit:.1f} pips")
        print(f"Average Realistic Loss: {avg_realistic_loss:.1f} pips")
        
        # Win rate analysis with realistic spreads
        profit_target = 15  # Lower target due to spreads
        stop_loss = 10
        
        profitable_signals = len([s for s in signals if s.get('profit_after_spread', s['max_profit_pips']) >= profit_target])
        losing_signals = len([s for s in signals if s.get('loss_including_spread', s['max_loss_pips']) >= stop_loss])
        
        print(f"Signals reaching +{profit_target} pips (after spread): {profitable_signals}/{len(signals)} ({profitable_signals/len(signals)*100:.1f}%)")
        print(f"Signals hitting -{stop_loss} pips (including spread): {losing_signals}/{len(signals)} ({losing_signals/len(signals)*100:.1f}%)")
        
        # Epic-specific performance
        print(f"\n📈 PER-EPIC PERFORMANCE:")
        for epic, epic_signals in epic_groups.items():
            epic_profits = [s.get('profit_after_spread', s['max_profit_pips']) for s in epic_signals]
            epic_avg_profit = sum(epic_profits) / len(epic_profits) if epic_profits else 0
            epic_spread = epic_signals[0]['spread_pips'] if epic_signals else 0
            
            print(f"{epic}: {len(epic_signals)} signals, Avg profit: {epic_avg_profit:.1f} pips, Spread: {epic_spread:.1f} pips")

# Example usage function
def test_dynamic_backtesting(engine, epic_list, enhancement_function):
    """
    Test the dynamic backtesting system
    """
    print("🧪 Testing Dynamic Epic-Specific Backtesting...")
    
    signals = run_historical_analysis_bid_adjusted_dynamic(
        engine=engine,
        epic_list=epic_list,
        enhancement_function=enhancement_function,
        num_signals=20
    )
    
    return signals