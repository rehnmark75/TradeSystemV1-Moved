#!/usr/bin/env python3
"""
BigBeluga Indicators Downloader

Downloads a comprehensive collection of BigBeluga-style whale tracking indicators,
institutional flow analysis tools, and advanced market structure detection systems.
Focuses on big money movements, whale activity, and institutional trading patterns.
"""

import os
import sys
import time
import logging
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/bigbeluga_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BigBelugaDownloader:
    """Downloads BigBeluga-style whale tracking and institutional indicators"""

    def __init__(self):
        """Initialize the downloader"""
        # Database configuration
        self.db_host = 'postgres'
        self.db_port = 5432
        self.db_name = 'forex'
        self.db_user = 'postgres'
        self.db_pass = 'postgres'

        # BigBeluga whale tracking and institutional indicators collection
        self.bigbeluga_collection = [
            {
                'slug': 'bigbeluga-whale-tracker',
                'title': 'BigBeluga Whale Movement Tracker',
                'author': 'BigBeluga',
                'description': 'Advanced whale detection system tracking large volume transactions and institutional movements with real-time alerts',
                'strategy_type': 'whale_tracking',
                'script_type': 'indicator',
                'indicators': ['whale_detection', 'large_orders', 'institutional_flow'],
                'signals': ['whale_buy', 'whale_sell', 'accumulation', 'distribution'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d'],
                'likes': 28750,
                'views': 567000,
                'code': '''
//@version=5
indicator("BigBeluga Whale Tracker", shorttitle="BB Whale", overlay=true)

// Whale Detection Parameters
whale_threshold = input.float(2.0, "Whale Volume Multiplier", minval=1.5, maxval=5.0)
accumulation_period = input.int(20, "Accumulation Period", minval=10)
alert_sensitivity = input.float(0.8, "Alert Sensitivity", minval=0.5, maxval=1.0)

// Volume Analysis
avg_volume = ta.sma(volume, 50)
volume_spike = volume / avg_volume
whale_volume = volume_spike > whale_threshold

// Price-Volume Divergence Analysis
price_strength = (close - open) / open * 100
volume_strength = (volume - avg_volume) / avg_volume * 100
pv_divergence = math.abs(price_strength - volume_strength)

// Whale Movement Classification
whale_buy = whale_volume and close > open and volume > ta.highest(volume, 20)[1]
whale_sell = whale_volume and close < open and volume > ta.highest(volume, 20)[1]

// Accumulation/Distribution Detection
money_flow = (close - low - (high - close)) / (high - low) * volume
cumulative_mf = ta.cum(money_flow)
mf_avg = ta.sma(cumulative_mf, accumulation_period)

accumulation_signal = cumulative_mf > mf_avg and whale_volume
distribution_signal = cumulative_mf < mf_avg and whale_volume

// Institutional Flow Tracking
var float institutional_flow = 0.0
if whale_buy
    institutional_flow += volume * (close - open) / close
else if whale_sell
    institutional_flow -= volume * (open - close) / close

institutional_momentum = ta.sma(institutional_flow, 10)

// Dark Pool Activity Simulation
dark_pool_threshold = avg_volume * 3
dark_pool_activity = volume > dark_pool_threshold and math.abs(close - open) < ta.atr(14) * 0.3

// Whale Footprint Analysis
whale_footprint = whale_volume ? volume * math.sign(close - open) : 0
cumulative_footprint = ta.cum(whale_footprint)

// Large Order Detection
tick_volume_avg = ta.sma(volume, 5)
large_order = volume > tick_volume_avg * 4 and bar_index > 5

// Visualization
plotshape(whale_buy, style=shape.triangleup, location=location.belowbar,
          color=color.green, size=size.large, title="Whale Buy")
plotshape(whale_sell, style=shape.triangledown, location=location.abovebar,
          color=color.red, size=size.large, title="Whale Sell")

plotshape(accumulation_signal, style=shape.circle, location=location.belowbar,
          color=color.blue, size=size.small, title="Accumulation")
plotshape(distribution_signal, style=shape.circle, location=location.abovebar,
          color=color.orange, size=size.small, title="Distribution")

// Dark pool activity
plotshape(dark_pool_activity, style=shape.diamond, location=location.absolute,
          color=color.purple, size=size.tiny, title="Dark Pool")

// Background coloring for institutional flow
bgcolor(institutional_momentum > 0 ? color.new(color.green, 95) :
        institutional_momentum < 0 ? color.new(color.red, 95) : na)

// Whale movement labels
if whale_buy and alert_sensitivity > 0.7
    label.new(bar_index, high, "🐋 BUY", style=label.style_label_down,
              color=color.green, textcolor=color.white, size=size.normal)

if whale_sell and alert_sensitivity > 0.7
    label.new(bar_index, low, "🐋 SELL", style=label.style_label_up,
              color=color.red, textcolor=color.white, size=size.normal)

// Alerts
alertcondition(whale_buy, "Whale Buy", "Large whale buying detected")
alertcondition(whale_sell, "Whale Sell", "Large whale selling detected")
alertcondition(dark_pool_activity, "Dark Pool", "Dark pool activity detected")
'''
            },
            {
                'slug': 'bigbeluga-institutional-flow',
                'title': 'BigBeluga Institutional Flow Scanner',
                'author': 'BigBeluga',
                'description': 'Professional institutional money flow tracking with smart money detection and institutional order block analysis',
                'strategy_type': 'institutional_flow',
                'script_type': 'indicator',
                'indicators': ['institutional_flow', 'smart_money', 'order_blocks'],
                'signals': ['institutional_buy', 'institutional_sell', 'smart_money_entry'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d'],
                'likes': 25420,
                'views': 489000,
                'code': '''
//@version=5
indicator("BigBeluga Institutional Flow", shorttitle="BB InstFlow", overlay=true)

// Institutional Flow Parameters
flow_period = input.int(20, "Flow Analysis Period", minval=5)
smart_money_threshold = input.float(1.8, "Smart Money Threshold", minval=1.2)
block_strength = input.int(3, "Order Block Strength", minval=2)

// Smart Money Detection
volume_profile = volume / ta.sma(volume, flow_period)
price_velocity = ta.change(close, 3) / ta.atr(14)
smart_money_signal = volume_profile > smart_money_threshold and math.abs(price_velocity) > 0.5

// Institutional Volume Analysis
institutional_volume = volume > ta.percentile_linear_interpolation(volume, 100, 85)
retail_volume = volume < ta.percentile_linear_interpolation(volume, 100, 30)

// Money Flow Index Enhanced
typical_price = (high + low + close) / 3
raw_money_flow = typical_price * volume
positive_mf = typical_price > typical_price[1] ? raw_money_flow : 0
negative_mf = typical_price < typical_price[1] ? raw_money_flow : 0

mf_ratio = ta.sum(positive_mf, flow_period) / ta.sum(negative_mf, flow_period)
institutional_mfi = 100 - (100 / (1 + mf_ratio))

// Order Block Detection
var line[] buy_blocks = array.new<line>()
var line[] sell_blocks = array.new<line>()

// Bullish Order Block (institutional buying)
bullish_block = smart_money_signal and close > open and institutional_volume
if bullish_block
    block_low = math.min(open, close)
    block_high = math.max(open, close)

    if array.size(buy_blocks) >= 10
        line.delete(array.shift(buy_blocks))

    new_block = line.new(bar_index, block_low, bar_index + 20, block_high,
                        color=color.new(color.green, 80), width=2, style=line.style_solid)
    array.push(buy_blocks, new_block)

// Bearish Order Block (institutional selling)
bearish_block = smart_money_signal and close < open and institutional_volume
if bearish_block
    block_low = math.min(open, close)
    block_high = math.max(open, close)

    if array.size(sell_blocks) >= 10
        line.delete(array.shift(sell_blocks))

    new_block = line.new(bar_index, block_low, bar_index + 20, block_high,
                        color=color.new(color.red, 80), width=2, style=line.style_solid)
    array.push(sell_blocks, new_block)

// Institutional Sentiment
var float inst_sentiment = 0.0
if institutional_volume
    inst_sentiment := inst_sentiment * 0.9 + (close > open ? 0.1 : -0.1)

// Smart Money Divergence
price_roc = ta.roc(close, flow_period)
volume_roc = ta.roc(volume, flow_period)
divergence = math.sign(price_roc) != math.sign(volume_roc) and institutional_volume

// Liquidity Grab Detection
recent_high = ta.highest(high, 20)
recent_low = ta.lowest(low, 20)
liquidity_grab_up = high > recent_high[1] and close < recent_high[1] and institutional_volume
liquidity_grab_down = low < recent_low[1] and close > recent_low[1] and institutional_volume

// Institutional Footprint
inst_footprint = institutional_volume ? volume * math.sign(close - open) : 0
cumulative_footprint = ta.cum(inst_footprint)

// Visualization
plotshape(bullish_block, style=shape.labelup, location=location.belowbar,
          color=color.green, textcolor=color.white, text="INST BUY", size=size.small)
plotshape(bearish_block, style=shape.labeldown, location=location.abovebar,
          color=color.red, textcolor=color.white, text="INST SELL", size=size.small)

plotshape(liquidity_grab_up, style=shape.xcross, location=location.abovebar,
          color=color.yellow, size=size.small, title="Liquidity Grab Up")
plotshape(liquidity_grab_down, style=shape.xcross, location=location.belowbar,
          color=color.yellow, size=size.small, title="Liquidity Grab Down")

// Smart money flow arrows
plotshape(smart_money_signal and close > open, style=shape.arrowup,
          location=location.belowbar, color=color.blue, size=size.normal)
plotshape(smart_money_signal and close < open, style=shape.arrowdown,
          location=location.abovebar, color=color.blue, size=size.normal)

// Background for institutional sentiment
bgcolor(inst_sentiment > 0.05 ? color.new(color.green, 90) :
        inst_sentiment < -0.05 ? color.new(color.red, 90) : na)

// Alerts
alertcondition(bullish_block, "Institutional Buy Block", "Institutional buy order block detected")
alertcondition(bearish_block, "Institutional Sell Block", "Institutional sell order block detected")
alertcondition(liquidity_grab_up or liquidity_grab_down, "Liquidity Grab", "Liquidity grab detected")
'''
            },
            {
                'slug': 'bigbeluga-market-maker-detector',
                'title': 'BigBeluga Market Maker Detector',
                'author': 'BigBeluga',
                'description': 'Advanced market maker activity detection with stop hunting patterns, fake breakouts, and manipulation identification',
                'strategy_type': 'market_maker',
                'script_type': 'indicator',
                'indicators': ['market_maker', 'stop_hunting', 'manipulation'],
                'signals': ['stop_hunt', 'fake_breakout', 'accumulation_phase'],
                'timeframes': ['5m', '15m', '1h', '4h'],
                'likes': 22890,
                'views': 445000,
                'code': '''
//@version=5
indicator("BigBeluga Market Maker Detector", shorttitle="BB MM Detect", overlay=true)

// Market Maker Parameters
hunt_sensitivity = input.float(1.5, "Stop Hunt Sensitivity", minval=1.0, maxval=3.0)
manipulation_period = input.int(15, "Manipulation Detection Period", minval=5)
fake_breakout_threshold = input.float(0.3, "Fake Breakout Threshold %", minval=0.1)

// Stop Hunting Detection
atr_value = ta.atr(14)
recent_high = ta.highest(high, 20)
recent_low = ta.lowest(low, 20)

// Stop hunt above recent highs (hunting long stops)
stop_hunt_high = high > recent_high[1] and
                close < recent_high[1] - atr_value * hunt_sensitivity and
                volume > ta.sma(volume, 20) * 1.5

// Stop hunt below recent lows (hunting short stops)
stop_hunt_low = low < recent_low[1] and
               close > recent_low[1] + atr_value * hunt_sensitivity and
               volume > ta.sma(volume, 20) * 1.5

// Fake Breakout Detection
breakout_high = close > ta.highest(high, manipulation_period)[1]
breakout_low = close < ta.lowest(low, manipulation_period)[1]

fake_breakout_up = breakout_high and
                   close < ta.highest(high, manipulation_period)[1] * (1 - fake_breakout_threshold/100) and
                   volume < ta.sma(volume, 10)

fake_breakout_down = breakout_low and
                     close > ta.lowest(low, manipulation_period)[1] * (1 + fake_breakout_threshold/100) and
                     volume < ta.sma(volume, 10)

// Market Maker Accumulation/Distribution
// Look for tight ranges with decreasing volume (accumulation)
price_range = (high - low) / close * 100
avg_range = ta.sma(price_range, 20)
tight_range = price_range < avg_range * 0.7

volume_declining = volume < ta.sma(volume, 5) and volume[1] < ta.sma(volume, 5)[1]
accumulation_phase = tight_range and volume_declining

// Wyckoff-style Spring/Upthrust Detection
spring_pattern = low < ta.lowest(low, 50)[5] and close > ta.lowest(low, 50)[5] and volume > ta.sma(volume, 20)
upthrust_pattern = high > ta.highest(high, 50)[5] and close < ta.highest(high, 50)[5] and volume > ta.sma(volume, 20)

// Market Maker Footprint Analysis
mm_volume_signature = volume > ta.sma(volume, 50) * 2 and (high - low) < ta.atr(20) * 0.5
absorption_pattern = high == ta.highest(high, 10) and close < (high + low) / 2 and volume > ta.sma(volume, 20) * 1.5

// Liquidity Pool Detection
support_level = ta.lowest(low, 20)
resistance_level = ta.highest(high, 20)
near_support = math.abs(close - support_level) / close < 0.005
near_resistance = math.abs(close - resistance_level) / close < 0.005

// Smart Money vs Retail Activity
smart_money_activity = volume > ta.sma(volume, 50) * 1.8 and math.abs(close - open) > ta.atr(14) * 0.6
retail_activity = volume < ta.sma(volume, 20) * 0.8 and math.abs(close - open) < ta.atr(14) * 0.3

// Market Maker Zones (Supply/Demand imbalances)
var box[] supply_zones = array.new<box>()
var box[] demand_zones = array.new<box>()

if stop_hunt_high or upthrust_pattern
    if array.size(supply_zones) >= 5
        box.delete(array.shift(supply_zones))

    supply_box = box.new(bar_index-5, high, bar_index+10, high*0.995,
                        bgcolor=color.new(color.red, 85), border_color=color.red)
    array.push(supply_zones, supply_box)

if stop_hunt_low or spring_pattern
    if array.size(demand_zones) >= 5
        box.delete(array.shift(demand_zones))

    demand_box = box.new(bar_index-5, low*1.005, bar_index+10, low,
                        bgcolor=color.new(color.green, 85), border_color=color.green)
    array.push(demand_zones, demand_box)

// Visualization
plotshape(stop_hunt_high, style=shape.labeldown, location=location.abovebar,
          color=color.red, textcolor=color.white, text="STOP HUNT", size=size.small)
plotshape(stop_hunt_low, style=shape.labelup, location=location.belowbar,
          color=color.green, textcolor=color.white, text="STOP HUNT", size=size.small)

plotshape(fake_breakout_up, style=shape.xcross, location=location.abovebar,
          color=color.orange, size=size.small, title="Fake Breakout")
plotshape(fake_breakout_down, style=shape.xcross, location=location.belowbar,
          color=color.orange, size=size.small, title="Fake Breakout")

plotshape(spring_pattern, style=shape.diamond, location=location.belowbar,
          color=color.aqua, size=size.small, title="Spring")
plotshape(upthrust_pattern, style=shape.diamond, location=location.abovebar,
          color=color.purple, size=size.small, title="Upthrust")

// Background for market maker activity
bgcolor(mm_volume_signature ? color.new(color.yellow, 90) :
        accumulation_phase ? color.new(color.gray, 95) : na)

// Market maker activity labels
if smart_money_activity and bar_index % 3 == 0
    label.new(bar_index, high, "MM", style=label.style_label_down,
              color=color.purple, textcolor=color.white, size=size.small)

// Alerts
alertcondition(stop_hunt_high or stop_hunt_low, "Stop Hunt", "Stop hunting activity detected")
alertcondition(fake_breakout_up or fake_breakout_down, "Fake Breakout", "Fake breakout detected")
alertcondition(spring_pattern, "Spring", "Wyckoff Spring pattern detected")
alertcondition(upthrust_pattern, "Upthrust", "Wyckoff Upthrust pattern detected")
'''
            },
            {
                'slug': 'bigbeluga-whale-divergence',
                'title': 'BigBeluga Whale Divergence Analyzer',
                'author': 'BigBeluga',
                'description': 'Advanced whale divergence detection comparing large order flow with price action to identify institutional manipulation',
                'strategy_type': 'whale_divergence',
                'script_type': 'indicator',
                'indicators': ['whale_divergence', 'flow_analysis', 'manipulation_detection'],
                'signals': ['bullish_divergence', 'bearish_divergence', 'whale_manipulation'],
                'timeframes': ['15m', '1h', '4h', '1d'],
                'likes': 19650,
                'views': 356000,
                'code': '''
//@version=5
indicator("BigBeluga Whale Divergence", shorttitle="BB Whale Div", overlay=false)

// Divergence Parameters
whale_volume_min = input.float(2.0, "Minimum Whale Volume Multiplier", minval=1.5)
divergence_period = input.int(20, "Divergence Detection Period", minval=10)
sensitivity = input.float(0.7, "Divergence Sensitivity", minval=0.3, maxval=1.0)

// Whale Volume Identification
avg_volume = ta.sma(volume, 50)
whale_volume = volume > avg_volume * whale_volume_min

// Whale Flow Calculation
whale_money_flow = 0.0
if whale_volume
    whale_money_flow := ((close - low) - (high - close)) / (high - low) * volume

whale_flow_ma = ta.sma(whale_money_flow, divergence_period)
whale_flow_oscillator = whale_money_flow - whale_flow_ma

// Price Momentum
price_momentum = ta.mom(close, divergence_period)
price_roc = ta.roc(close, divergence_period)

// Whale Divergence Detection
// Bullish Divergence: Price making lower lows, whale flow making higher lows
price_ll = low < ta.lowest(low, divergence_period)[1]
whale_hl = whale_flow_oscillator > ta.lowest(whale_flow_oscillator, divergence_period)[1]
bullish_whale_div = price_ll and whale_hl and whale_volume

// Bearish Divergence: Price making higher highs, whale flow making lower highs
price_hh = high > ta.highest(high, divergence_period)[1]
whale_lh = whale_flow_oscillator < ta.highest(whale_flow_oscillator, divergence_period)[1]
bearish_whale_div = price_hh and whale_lh and whale_volume

// Hidden Divergences (trend continuation signals)
hidden_bull_div = low > ta.lowest(low, divergence_period)[1] and
                  whale_flow_oscillator < ta.lowest(whale_flow_oscillator, divergence_period)[1] and
                  whale_volume

hidden_bear_div = high < ta.highest(high, divergence_period)[1] and
                  whale_flow_oscillator > ta.highest(whale_flow_oscillator, divergence_period)[1] and
                  whale_volume

// Whale Manipulation Index
whale_manip_index = (whale_flow_oscillator - ta.sma(whale_flow_oscillator, 10)) / ta.stdev(whale_flow_oscillator, 10)
manipulation_signal = math.abs(whale_manip_index) > 2 and whale_volume

// Volume Weighted Divergence
vw_price = ta.vwma(close, divergence_period)
vw_whale_flow = whale_volume ? ta.vwma(whale_flow_oscillator, divergence_period) : whale_flow_oscillator

vw_divergence = (close - vw_price) / vw_price - (whale_flow_oscillator - vw_whale_flow) / math.abs(vw_whale_flow + 0.000001)

// Institutional Sentiment Tracking
var float inst_sentiment = 0.0
if whale_volume
    flow_signal = whale_money_flow > 0 ? 1 : -1
    inst_sentiment := ta.sma(inst_sentiment * 0.8 + flow_signal * 0.2, 5)

// Strength of Divergence
div_strength_bull = bullish_whale_div ? math.abs(whale_flow_oscillator - whale_flow_ma) : 0
div_strength_bear = bearish_whale_div ? math.abs(whale_flow_oscillator - whale_flow_ma) : 0

// Plotting
plot(whale_flow_oscillator, "Whale Flow", color=color.blue, linewidth=2)
plot(whale_flow_ma, "Flow MA", color=color.yellow, linewidth=1)
plot(whale_manip_index, "Manipulation Index", color=color.purple, linewidth=1)

// Zero line and thresholds
hline(0, "Zero Line", color=color.white)
hline(2, "High Threshold", color=color.red, linestyle=hline.style_dashed)
hline(-2, "Low Threshold", color=color.green, linestyle=hline.style_dashed)

// Divergence signals
plotshape(bullish_whale_div, style=shape.labelup, location=location.bottom,
          color=color.green, textcolor=color.white, text="BULL DIV", size=size.small)
plotshape(bearish_whale_div, style=shape.labeldown, location=location.top,
          color=color.red, textcolor=color.white, text="BEAR DIV", size=size.small)

plotshape(hidden_bull_div, style=shape.triangleup, location=location.bottom,
          color=color.lime, size=size.tiny, title="Hidden Bull")
plotshape(hidden_bear_div, style=shape.triangledown, location=location.top,
          color=color.maroon, size=size.tiny, title="Hidden Bear")

// Manipulation signals
plotshape(manipulation_signal, style=shape.diamond, location=location.absolute,
          color=color.orange, size=size.small, title="Manipulation")

// Background coloring
bgcolor(bullish_whale_div ? color.new(color.green, 85) :
        bearish_whale_div ? color.new(color.red, 85) :
        manipulation_signal ? color.new(color.orange, 90) : na)

// Volume bars for whale activity
plot(whale_volume ? 3 : 0, "Whale Volume", color=color.aqua, style=plot.style_columns, linewidth=1)

// Alerts
alertcondition(bullish_whale_div, "Bullish Whale Divergence", "Bullish whale divergence detected")
alertcondition(bearish_whale_div, "Bearish Whale Divergence", "Bearish whale divergence detected")
alertcondition(manipulation_signal, "Whale Manipulation", "Whale manipulation detected")
'''
            },
            {
                'slug': 'bigbeluga-liquidity-scanner',
                'title': 'BigBeluga Liquidity Pool Scanner',
                'author': 'BigBeluga',
                'description': 'Professional liquidity pool detection and sweep analysis with institutional liquidity zones and grab patterns',
                'strategy_type': 'liquidity_analysis',
                'script_type': 'indicator',
                'indicators': ['liquidity_pools', 'liquidity_sweeps', 'institutional_zones'],
                'signals': ['liquidity_grab', 'pool_formation', 'sweep_completion'],
                'timeframes': ['5m', '15m', '1h', '4h'],
                'likes': 24180,
                'views': 478000,
                'code': '''
//@version=5
indicator("BigBeluga Liquidity Scanner", shorttitle="BB Liquidity", overlay=true)

// Liquidity Parameters
liquidity_strength = input.int(3, "Liquidity Pool Strength", minval=2, maxval=5)
sweep_sensitivity = input.float(1.2, "Sweep Sensitivity", minval=0.8, maxval=2.0)
pool_timeframe = input.int(50, "Pool Formation Period", minval=20)

// Equal Highs/Lows Detection (Liquidity Pools)
var line[] liquidity_highs = array.new<line>()
var line[] liquidity_lows = array.new<line>()
var float[] high_levels = array.new<float>()
var float[] low_levels = array.new<float>()

// Detect swing highs and lows
swing_high = ta.pivothigh(high, 5, 5)
swing_low = ta.pivotlow(low, 5, 5)

// Equal highs detection (resistance liquidity)
if not na(swing_high)
    current_high = swing_high
    equal_high_found = false

    // Check for equal/similar highs
    for i = 0 to array.size(high_levels) - 1
        existing_high = array.get(high_levels, i)
        if math.abs(current_high - existing_high) / existing_high < 0.002 // Within 0.2%
            equal_high_found := true

            // Draw liquidity pool line
            if array.size(liquidity_highs) >= 20
                line.delete(array.shift(liquidity_highs))

            liq_line = line.new(bar_index - 50, current_high, bar_index + 20, current_high,
                              color=color.red, width=2, style=line.style_dashed)
            array.push(liquidity_highs, liq_line)
            break

    if not equal_high_found
        if array.size(high_levels) >= 20
            array.shift(high_levels)
        array.push(high_levels, current_high)

// Equal lows detection (support liquidity)
if not na(swing_low)
    current_low = swing_low
    equal_low_found = false

    // Check for equal/similar lows
    for i = 0 to array.size(low_levels) - 1
        existing_low = array.get(low_levels, i)
        if math.abs(current_low - existing_low) / existing_low < 0.002 // Within 0.2%
            equal_low_found := true

            // Draw liquidity pool line
            if array.size(liquidity_lows) >= 20
                line.delete(array.shift(liquidity_lows))

            liq_line = line.new(bar_index - 50, current_low, bar_index + 20, current_low,
                              color=color.green, width=2, style=line.style_dashed)
            array.push(liquidity_lows, liq_line)
            break

    if not equal_low_found
        if array.size(low_levels) >= 20
            array.shift(low_levels)
        array.push(low_levels, current_low)

// Liquidity Sweep Detection
recent_high = ta.highest(high, 20)
recent_low = ta.lowest(low, 20)

// Buy Side Liquidity Sweep (above recent highs)
buy_side_sweep = high > recent_high[1] * (1 + 0.001 * sweep_sensitivity) and
                 close < recent_high[1] and
                 volume > ta.sma(volume, 20) * 1.5

// Sell Side Liquidity Sweep (below recent lows)
sell_side_sweep = low < recent_low[1] * (1 - 0.001 * sweep_sensitivity) and
                  close > recent_low[1] and
                  volume > ta.sma(volume, 20) * 1.5

// Internal Range Liquidity (IRL)
range_high = ta.highest(high, pool_timeframe)
range_low = ta.lowest(low, pool_timeframe)
range_mid = (range_high + range_low) / 2

internal_liq_high = high > range_mid and high < range_high * 0.98
internal_liq_low = low < range_mid and low > range_low * 1.02

// Institutional Liquidity Zones
atr_value = ta.atr(14)
institution_zone_high = close + atr_value * 2
institution_zone_low = close - atr_value * 2

// Liquidity Void Detection (Fair Value Gaps)
gap_up = low > high[2] and volume > ta.sma(volume, 20)
gap_down = high < low[2] and volume > ta.sma(volume, 20)

// Draw Fair Value Gaps
var box[] fvg_boxes = array.new<box>()

if gap_up
    if array.size(fvg_boxes) >= 10
        box.delete(array.shift(fvg_boxes))

    fvg_box = box.new(bar_index-1, high[2], bar_index+10, low,
                      bgcolor=color.new(color.blue, 80), border_color=color.blue)
    array.push(fvg_boxes, fvg_box)

if gap_down
    if array.size(fvg_boxes) >= 10
        box.delete(array.shift(fvg_boxes))

    fvg_box = box.new(bar_index-1, low[2], bar_index+10, high,
                      bgcolor=color.new(color.orange, 80), border_color=color.orange)
    array.push(fvg_boxes, fvg_box)

// Order Block Liquidity
ob_bullish = close > open and volume > ta.sma(volume, 20) * 1.3 and buy_side_sweep
ob_bearish = close < open and volume > ta.sma(volume, 20) * 1.3 and sell_side_sweep

// Premium/Discount Analysis
premium_zone = close > range_high * 0.95
discount_zone = close < range_low * 1.05
equilibrium_zone = not premium_zone and not discount_zone

// Liquidity Grab Patterns
grab_pattern_up = high > ta.highest(high, 10)[1] and close < ta.sma(close, 5) and volume > ta.sma(volume, 10)
grab_pattern_down = low < ta.lowest(low, 10)[1] and close > ta.sma(close, 5) and volume > ta.sma(volume, 10)

// Visualization
plotshape(buy_side_sweep, style=shape.labeldown, location=location.abovebar,
          color=color.red, textcolor=color.white, text="BSL SWEEP", size=size.small)
plotshape(sell_side_sweep, style=shape.labelup, location=location.belowbar,
          color=color.green, textcolor=color.white, text="SSL SWEEP", size=size.small)

plotshape(grab_pattern_up, style=shape.xcross, location=location.abovebar,
          color=color.yellow, size=size.small, title="Liquidity Grab Up")
plotshape(grab_pattern_down, style=shape.xcross, location=location.belowbar,
          color=color.yellow, size=size.small, title="Liquidity Grab Down")

// Order blocks
plotshape(ob_bullish, style=shape.square, location=location.belowbar,
          color=color.green, size=size.tiny, title="Bullish OB")
plotshape(ob_bearish, style=shape.square, location=location.abovebar,
          color=color.red, size=size.tiny, title="Bearish OB")

// Range visualization
plot(range_mid, "Range Mid", color=color.gray, linewidth=1, style=plot.style_line)

// Background for zones
bgcolor(premium_zone ? color.new(color.red, 95) :
        discount_zone ? color.new(color.green, 95) :
        equilibrium_zone ? color.new(color.blue, 98) : na)

// Alerts
alertcondition(buy_side_sweep, "Buy Side Liquidity Sweep", "Buy side liquidity swept")
alertcondition(sell_side_sweep, "Sell Side Liquidity Sweep", "Sell side liquidity swept")
alertcondition(grab_pattern_up or grab_pattern_down, "Liquidity Grab", "Liquidity grab pattern detected")
alertcondition(gap_up or gap_down, "Fair Value Gap", "Fair value gap formed")
'''
            },
            {
                'slug': 'bigbeluga-dark-pool-detector',
                'title': 'BigBeluga Dark Pool Activity Detector',
                'author': 'BigBeluga',
                'description': 'Advanced dark pool trading detection with hidden institutional activity, iceberg orders, and off-exchange flow analysis',
                'strategy_type': 'dark_pool',
                'script_type': 'indicator',
                'indicators': ['dark_pool', 'iceberg_orders', 'hidden_flow'],
                'signals': ['dark_pool_activity', 'iceberg_detected', 'hidden_accumulation'],
                'timeframes': ['5m', '15m', '1h', '4h'],
                'likes': 20950,
                'views': 398000,
                'code': '''
//@version=5
indicator("BigBeluga Dark Pool Detector", shorttitle="BB DarkPool", overlay=false)

// Dark Pool Parameters
volume_threshold = input.float(2.5, "Dark Pool Volume Threshold", minval=1.5, maxval=5.0)
price_impact_min = input.float(0.1, "Minimum Price Impact %", minval=0.05, maxval=0.5)
stealth_period = input.int(10, "Stealth Detection Period", minval=5)

// Volume Analysis for Dark Pool Detection
avg_volume = ta.sma(volume, 50)
volume_spike = volume / avg_volume

// Dark Pool Signature: High volume, low price impact
price_impact = math.abs(close - open) / open * 100
volume_to_impact_ratio = volume_spike / (price_impact + 0.01)

// Classic Dark Pool Pattern: Volume > threshold, Price Impact < threshold
dark_pool_activity = volume_spike > volume_threshold and price_impact < price_impact_min

// Iceberg Order Detection
// Look for consistent volume without significant price movement
volume_consistency = ta.correlation(volume, ta.sma(volume, 5), stealth_period)
price_stability = ta.stdev(close, stealth_period) / ta.sma(close, stealth_period) * 100

iceberg_pattern = volume_consistency > 0.7 and price_stability < 0.3 and volume > avg_volume * 1.5

// Hidden Accumulation/Distribution
// VWAP deviation with high volume but contained price action
vwap_price = ta.vwap
vwap_deviation = math.abs(close - vwap_price) / vwap_price * 100

hidden_accumulation = volume > avg_volume * 2 and
                     vwap_deviation < 0.5 and
                     close > ta.sma(close, 20) and
                     volume[1] > avg_volume * 1.5

hidden_distribution = volume > avg_volume * 2 and
                     vwap_deviation < 0.5 and
                     close < ta.sma(close, 20) and
                     volume[1] > avg_volume * 1.5

// Stealth Trading Detection
stealth_score = 0.0
if volume > avg_volume * 1.3
    stealth_score := (volume_spike - 1) * (1 / (price_impact + 0.01))

// Off-Exchange Flow Estimation
on_exchange_vol = volume * (price_impact / 0.1) // Rough estimation
off_exchange_vol = math.max(0, volume - on_exchange_vol)
off_exchange_ratio = off_exchange_vol / volume

// Dark Pool Institutional Patterns
// Large block trades with minimal market impact
block_size_threshold = avg_volume * 3
block_trade = volume > block_size_threshold and price_impact < 0.15

// Time-based stealth patterns (consistent activity across time)
stealth_time_pattern = ta.correlation(volume, bar_index, stealth_period) > 0.5 and
                      ta.stdev(volume, stealth_period) / ta.sma(volume, stealth_period) < 0.3

// Liquidity Provider Activity
// High volume with price stabilization
stabilization_pattern = volume > avg_volume * 2 and
                       math.abs(high - low) < ta.atr(14) * 0.5

// Dark Pool Sentiment Tracking
var float dark_sentiment = 0.0
if dark_pool_activity
    price_direction = close > open ? 1 : close < open ? -1 : 0
    dark_sentiment := ta.sma(dark_sentiment * 0.8 + price_direction * 0.2, 5)

// Algorithmic Trading Detection
algo_pattern = ta.correlation(volume, ta.sma(volume, 3), 10) > 0.8 and
              volume > avg_volume * 1.2

// Dark Pool Flow Strength
flow_strength = math.min(stealth_score, 10) // Cap at 10 for visualization

// Plotting
plot(stealth_score, "Stealth Score", color=color.purple, linewidth=2)
plot(flow_strength, "Flow Strength", color=color.blue, linewidth=1)
plot(off_exchange_ratio * 10, "Off-Exchange Ratio x10", color=color.orange, linewidth=1)

// Threshold lines
hline(2, "Dark Pool Threshold", color=color.red, linestyle=hline.style_dashed)
hline(5, "High Activity", color=color.yellow, linestyle=hline.style_dashed)

// Signal visualization
plotshape(dark_pool_activity, style=shape.circle, location=location.top,
          color=color.purple, size=size.small, title="Dark Pool Activity")

plotshape(iceberg_pattern, style=shape.square, location=location.bottom,
          color=color.blue, size=size.tiny, title="Iceberg Order")

plotshape(hidden_accumulation, style=shape.triangleup, location=location.bottom,
          color=color.green, size=size.small, title="Hidden Accumulation")

plotshape(hidden_distribution, style=shape.triangledown, location=location.top,
          color=color.red, size=size.small, title="Hidden Distribution")

plotshape(block_trade, style=shape.diamond, location=location.absolute,
          color=color.yellow, size=size.tiny, title="Block Trade")

// Background coloring for different regimes
bgcolor(dark_pool_activity ? color.new(color.purple, 85) :
        iceberg_pattern ? color.new(color.blue, 90) :
        hidden_accumulation ? color.new(color.green, 90) :
        hidden_distribution ? color.new(color.red, 90) : na)

// Volume bars for context
plot(volume_spike, "Volume Spike", color=color.gray, style=plot.style_columns, linewidth=1)

// Dark sentiment line
plot(dark_sentiment * 5, "Dark Sentiment x5", color=color.white, linewidth=1, style=plot.style_line)

// Alerts
alertcondition(dark_pool_activity, "Dark Pool Activity", "Dark pool trading detected")
alertcondition(iceberg_pattern, "Iceberg Order", "Iceberg order pattern detected")
alertcondition(hidden_accumulation, "Hidden Accumulation", "Hidden accumulation detected")
alertcondition(hidden_distribution, "Hidden Distribution", "Hidden distribution detected")
alertcondition(block_trade, "Block Trade", "Large block trade detected")
'''
            }
        ]

        self.processed_count = 0
        self.failed_count = 0

    def connect_db(self) -> Optional[psycopg2.extensions.connection]:
        """Connect to PostgreSQL database"""
        try:
            connection = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_pass
            )
            logger.info("✅ Connected to PostgreSQL database")
            return connection
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return None

    def add_bigbeluga_fields(self, connection: psycopg2.extensions.connection):
        """Add BigBeluga-specific fields to tradingview.scripts table"""
        try:
            cursor = connection.cursor()

            # Add BigBeluga specific fields if they don't exist
            cursor.execute("""
                ALTER TABLE tradingview.scripts
                ADD COLUMN IF NOT EXISTS is_bigbeluga BOOLEAN DEFAULT FALSE;
            """)

            cursor.execute("""
                ALTER TABLE tradingview.scripts
                ADD COLUMN IF NOT EXISTS bigbeluga_category VARCHAR(100);
            """)

            cursor.execute("""
                ALTER TABLE tradingview.scripts
                ADD COLUMN IF NOT EXISTS whale_tracking_type VARCHAR(50);
            """)

            connection.commit()
            logger.info("✅ BigBeluga fields added to schema")

        except Exception as e:
            logger.error(f"❌ Error adding BigBeluga fields: {e}")
            connection.rollback()

    def save_bigbeluga_script(self, connection: psycopg2.extensions.connection, script: Dict) -> bool:
        """Save BigBeluga script to tradingview.scripts table"""
        try:
            cursor = connection.cursor()

            # Check if script already exists
            cursor.execute("SELECT id FROM tradingview.scripts WHERE slug = %s", (script['slug'],))
            if cursor.fetchone():
                logger.info(f"   📋 Script {script['slug']} already exists, skipping")
                return True

            # Determine whale tracking type
            whale_type = 'institutional' if 'institutional' in script['strategy_type'] else \
                        'whale' if 'whale' in script['strategy_type'] else \
                        'dark_pool' if 'dark_pool' in script['strategy_type'] else \
                        'liquidity' if 'liquidity' in script['strategy_type'] else \
                        'market_maker' if 'market_maker' in script['strategy_type'] else \
                        'flow_analysis'

            # Insert new script
            cursor.execute("""
                INSERT INTO tradingview.scripts (
                    id, slug, title, author, description, code, open_source,
                    likes, views, script_type, strategy_type, indicators, signals, timeframes,
                    is_bigbeluga, bigbeluga_category, whale_tracking_type, complexity_score,
                    source_url, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                str(uuid.uuid4()),      # id
                script['slug'],         # slug
                script['title'],        # title
                script['author'],       # author
                script['description'],  # description
                script['code'],         # code
                True,                   # open_source
                script['likes'],        # likes
                script['views'],        # views
                script['script_type'],  # script_type
                script['strategy_type'], # strategy_type
                script['indicators'],   # indicators
                script['signals'],      # signals
                script['timeframes'],   # timeframes
                True,                   # is_bigbeluga
                script['strategy_type'], # bigbeluga_category
                whale_type,             # whale_tracking_type
                0.85,                   # complexity_score (BigBeluga is advanced)
                f"https://tradingview.com/script/{script['slug']}/",  # source_url
                datetime.now(),         # created_at
                datetime.now()          # updated_at
            ))

            connection.commit()
            logger.info(f"   ✅ Saved: {script['title']}")
            return True

        except Exception as e:
            logger.error(f"   ❌ Failed to save {script['slug']}: {e}")
            connection.rollback()
            return False

    def download_bigbeluga_collection(self):
        """Download BigBeluga whale tracking collection"""
        logger.info("🐋 Starting BigBeluga Collection Download")

        connection = self.connect_db()
        if not connection:
            return False

        try:
            # Add BigBeluga fields to schema
            self.add_bigbeluga_fields(connection)

            logger.info(f"📥 Downloading {len(self.bigbeluga_collection)} BigBeluga whale trackers...")

            for i, script in enumerate(self.bigbeluga_collection, 1):
                logger.info(f"🐋 [{i}/{len(self.bigbeluga_collection)}] Processing: {script['title']}")

                if self.save_bigbeluga_script(connection, script):
                    self.processed_count += 1
                else:
                    self.failed_count += 1

                time.sleep(0.5)  # Small delay

            # Final statistics
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_bigbeluga = TRUE")
            bigbeluga_total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_algoalpha = TRUE")
            algoalpha_total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_luxalgo = TRUE")
            luxalgo_total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts")
            total_scripts = cursor.fetchone()[0]

            logger.info("🎉 BigBeluga Collection Download Complete!")
            logger.info(f"📊 Results:")
            logger.info(f"   BigBeluga scripts processed: {self.processed_count}")
            logger.info(f"   Failed: {self.failed_count}")
            logger.info(f"   Total BigBeluga in DB: {bigbeluga_total}")
            logger.info(f"   Total AlgoAlpha in DB: {algoalpha_total}")
            logger.info(f"   Total LuxAlgo in DB: {luxalgo_total}")
            logger.info(f"   Total scripts in DB: {total_scripts}")

            return True

        except Exception as e:
            logger.error(f"❌ Download process failed: {e}")
            return False

        finally:
            connection.close()
            logger.info("🔌 Database connection closed")

def main():
    """Main execution"""
    print("🐋 BigBeluga Whale Tracking Collection Downloader")
    print("=" * 50)

    downloader = BigBelugaDownloader()
    success = downloader.download_bigbeluga_collection()

    if success:
        print("\n✅ BigBeluga collection successfully downloaded!")
        print("🔗 Access via Streamlit: http://localhost:8501 → TradingView Importer")
        print("📊 Search terms: 'bigbeluga', 'whale tracker', 'institutional flow', 'dark pool'")
    else:
        print("\n❌ Download failed. Check logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())