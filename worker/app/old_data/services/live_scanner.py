# live_scanner.py
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time

from services.data_utils import *
from services.ema_signals import *

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

def smart_signal_detection_tz(engine, epic_list, enhancement_function, cutoff_time_local=None, local_timezone='Europe/Stockholm'):
    """
    Smart signal detection with proper timezone handling
    """
    if cutoff_time_local is None:
        # Default: 4 hours ago in local time
        local_tz = pytz.timezone(local_timezone)
        cutoff_time_local = datetime.now(local_tz) - timedelta(hours=4)
    
    # Convert to UTC for database comparison
    cutoff_utc = get_utc_cutoff_time(cutoff_time_local, local_timezone)
    
    print(f"🧠 Timezone-Aware Smart Signal Detection")
    print(f"🕒 Local cutoff: {cutoff_time_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"🕒 UTC cutoff: {cutoff_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    signals = []
    
    for epic in epic_list:
        try:
            # Get timezone-aware data
            df_5m, _, _ = enhancement_function(engine, epic, extract_pair_from_epic(epic), user_timezone=local_timezone)
            
            if df_5m is None or len(df_5m) == 0:
                print(f"❌ No data for {epic}")
                continue
            
            # Check most recent data timestamp against cutoff
            latest_timestamp = df_5m['start_time'].iloc[-1]
            
            if latest_timestamp > cutoff_utc:
                # Recent data - should be MID prices
                print(f"✅ {epic}: Using MID prices (new data)")
                df_with_emas = add_ema_indicators(df_5m)
                signal = detect_ema_signals(df_with_emas, epic)
            else:
                # Older data - use BID adjustment
                print(f"🔄 {epic}: Using BID adjustment (old data)")
                signal = detect_ema_signals_bid_adjusted(df_5m, epic)
            
            if signal:
                # Add timezone info to signal
                if 'start_time_local' in df_5m.columns:
                    signal_row = df_5m[df_5m['start_time'] == pd.to_datetime(signal['timestamp'])].iloc[0]
                    signal['timestamp_local'] = signal_row['start_time_local'].strftime('%Y-%m-%d %H:%M:%S %Z')
                
                signals.append(signal)
                print(f"🚨 {signal['signal_type']} signal detected!")
                
        except Exception as e:
            print(f"❌ Error processing {epic}: {e}")
            continue
    
    return signals
