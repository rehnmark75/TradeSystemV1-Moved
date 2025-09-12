#!/usr/bin/env python3
"""
Debug script to compare our Zero Lag calculations with TradingView values
"""

import sys
import os
sys.path.insert(0, '/app/forex_scanner')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.strategies.helpers.zero_lag_indicator_calculator import ZeroLagIndicatorCalculator
try:
    import config
except ImportError:
    from forex_scanner import config
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Known TradingView values from user feedback
TRADINGVIEW_VALUES = [
    {
        'timestamp': '2025-08-28 10:15:00',
        'upper_band': 1.16572,
        'lower_band': 1.16424,
        'zlema_basis': 1.16461,  # TradingView "Zero Lag Basis"
        'ribbon': 'RED'
    },
    {
        'timestamp': '2025-09-01 02:45:00',
        'upper_band': None,  # From screenshot, appears around 1.171+
        'lower_band': None,  # From screenshot, appears around 1.169+
        'zlema_basis': None,
        'ribbon': 'RED'
    }
]

def analyze_zero_lag_calculations():
    """Compare our calculations with TradingView"""
    
    # Setup
    db = DatabaseManager(config.DATABASE_URL)
    fetcher = DataFetcher(db, 'UTC')
    calc = ZeroLagIndicatorCalculator()
    
    # Parameters from config
    length = 70
    band_multiplier = 1.2
    
    logger.info("="*80)
    logger.info("ZERO LAG CALCULATION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Parameters: Length={length}, Band Multiplier={band_multiplier}")
    logger.info("")
    
    for tv_data in TRADINGVIEW_VALUES:
        timestamp = tv_data['timestamp']
        logger.info(f"\n📍 Analyzing timestamp: {timestamp}")
        logger.info("-"*60)
        
        # Get data around this timestamp
        target_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        start_dt = target_dt - timedelta(hours=24)
        end_dt = target_dt + timedelta(hours=1)
        
        # Fetch data using the correct method
        df = fetcher.get_enhanced_data(
            epic='CS.D.EURUSD.MINI.IP',
            start_time=start_dt,
            end_time=end_dt, 
            timeframe='15m',
            ensure_complete=False
        )
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {timestamp}")
            continue
        
        # Calculate indicators
        df = calc.ensure_zero_lag_indicators(df, length=length, band_multiplier=band_multiplier)
        
        # Find the specific row
        mask = df['start_time'].astype(str).str.contains(timestamp.replace(' ', ''))
        if not mask.any():
            logger.error(f"Timestamp {timestamp} not found in data")
            continue
            
        row = df[mask].iloc[0]
        idx = df[mask].index[0]
        
        # Our calculations
        our_close = row['close']
        our_open = row['open']
        our_zlema = row['zlema']
        our_upper = row['upper_band']
        our_lower = row['lower_band']
        our_volatility = row['volatility']
        our_trend = row['trend']
        
        # Candle color
        candle_color = "GREEN" if our_close > our_open else "RED" if our_close < our_open else "DOJI"
        
        logger.info("\n📊 OUR CALCULATIONS:")
        logger.info(f"  Close:       {our_close:.5f}")
        logger.info(f"  Open:        {our_open:.5f}")
        logger.info(f"  Candle:      {candle_color}")
        logger.info(f"  ZLEMA:       {our_zlema:.5f}")
        logger.info(f"  Upper Band:  {our_upper:.5f}")
        logger.info(f"  Lower Band:  {our_lower:.5f}")
        logger.info(f"  Volatility:  {our_volatility:.5f}")
        logger.info(f"  Trend:       {our_trend} ({'GREEN' if our_trend == 1 else 'RED' if our_trend == -1 else 'NEUTRAL'})")
        
        logger.info("\n📺 TRADINGVIEW VALUES:")
        if tv_data['zlema_basis']:
            logger.info(f"  ZLEMA Basis: {tv_data['zlema_basis']:.5f}")
        if tv_data['upper_band']:
            logger.info(f"  Upper Band:  {tv_data['upper_band']:.5f}")
        if tv_data['lower_band']:
            logger.info(f"  Lower Band:  {tv_data['lower_band']:.5f}")
        logger.info(f"  Ribbon:      {tv_data['ribbon']}")
        
        # Calculate differences
        logger.info("\n🔍 DIFFERENCES (Our - TradingView):")
        if tv_data['zlema_basis']:
            zlema_diff = our_zlema - tv_data['zlema_basis']
            zlema_pips = zlema_diff * 10000
            logger.info(f"  ZLEMA:       {zlema_diff:+.5f} ({zlema_pips:+.1f} pips)")
            
        if tv_data['upper_band']:
            upper_diff = our_upper - tv_data['upper_band']
            upper_pips = upper_diff * 10000
            logger.info(f"  Upper Band:  {upper_diff:+.5f} ({upper_pips:+.1f} pips)")
            
        if tv_data['lower_band']:
            lower_diff = our_lower - tv_data['lower_band']
            lower_pips = lower_diff * 10000
            logger.info(f"  Lower Band:  {lower_diff:+.5f} ({lower_pips:+.1f} pips)")
            
        # Band width comparison
        our_band_width = our_upper - our_lower
        if tv_data['upper_band'] and tv_data['lower_band']:
            tv_band_width = tv_data['upper_band'] - tv_data['lower_band']
            width_diff = our_band_width - tv_band_width
            logger.info(f"\n  Band Width:")
            logger.info(f"    Ours:        {our_band_width:.5f} ({our_band_width*10000:.1f} pips)")
            logger.info(f"    TradingView: {tv_band_width:.5f} ({tv_band_width*10000:.1f} pips)")
            logger.info(f"    Difference:  {width_diff:.5f} ({width_diff*10000:.1f} pips)")
        else:
            logger.info(f"\n  Our Band Width: {our_band_width:.5f} ({our_band_width*10000:.1f} pips)")
            
        # Debug the ZLEMA calculation components
        logger.info("\n🔧 ZLEMA CALCULATION DEBUG:")
        lag = (length - 1) // 2
        logger.info(f"  Lag: {lag}")
        
        # Get historical data for lag calculation
        if idx >= lag:
            lagged_close = df.iloc[idx - lag]['close']
            momentum_adj = our_close - lagged_close
            zlema_input = our_close + momentum_adj
            logger.info(f"  Current Close:     {our_close:.5f}")
            logger.info(f"  Lagged Close [{idx-lag}]: {lagged_close:.5f}")
            logger.info(f"  Momentum Adj:      {momentum_adj:.5f}")
            logger.info(f"  ZLEMA Input:       {zlema_input:.5f}")
            
        # Debug volatility calculation
        logger.info("\n🔧 VOLATILITY CALCULATION DEBUG:")
        logger.info(f"  ATR Length:        {length}")
        logger.info(f"  Volatility Window: {length * 3}")
        logger.info(f"  Band Multiplier:   {band_multiplier}")
        logger.info(f"  Volatility Value:  {our_volatility:.5f}")
        
        # Check surrounding bars for context
        logger.info("\n📈 SURROUNDING BARS:")
        for i in range(max(0, idx-2), min(len(df), idx+3)):
            bar = df.iloc[i]
            bar_trend = "🟢" if bar['trend'] == 1 else "🔴" if bar['trend'] == -1 else "⚪"
            is_current = " <-- CURRENT" if i == idx else ""
            logger.info(f"  [{i}] {bar['start_time']}: Close={bar['close']:.5f}, "
                       f"ZLEMA={bar['zlema']:.5f}, Upper={bar['upper_band']:.5f}, "
                       f"Trend={bar_trend}{is_current}")

if __name__ == "__main__":
    analyze_zero_lag_calculations()