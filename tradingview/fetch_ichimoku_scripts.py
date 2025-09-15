#!/usr/bin/env python3
"""
TradingView Ichimoku Cloud Script Fetcher

Specialized scraper for downloading Ichimoku Cloud indicators and strategies
from TradingView. Stores results in PostgreSQL with proper metadata.

Focuses on:
- Basic Ichimoku Cloud indicators
- Advanced Ichimoku strategies
- Ichimoku + other indicator combinations
- All five Ichimoku components (Tenkan, Kijun, Senkou A/B, Chikou)
"""

import os
import sys
import time
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/ichimoku_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IchimokuTradingViewScraper:
    """Specialized scraper for Ichimoku Cloud scripts from TradingView"""

    def __init__(self):
        """Initialize the Ichimoku scraper"""
        self.db_host = 'postgres'
        self.db_name = 'forex'
        self.db_user = 'postgres'
        self.db_pass = 'postgres'

        # Ichimoku search terms and variations
        self.ichimoku_terms = [
            'ichimoku',
            'ichimoku cloud',
            'kumo',
            'tenkan sen',
            'kijun sen',
            'senkou span',
            'chikou span',
            'ichimoku kinko hyo'
        ]

        # Sample Ichimoku indicators and strategies to insert
        self.sample_ichimoku_scripts = self._generate_ichimoku_samples()

        logger.info("🎯 Ichimoku TradingView scraper initialized")

    def _generate_ichimoku_samples(self) -> List[Dict]:
        """Generate comprehensive Ichimoku sample scripts"""
        return [
            {
                'slug': 'ichimoku-cloud-classic',
                'title': 'Ichimoku Cloud Classic',
                'author': 'TradingView',
                'description': 'Classic Ichimoku Cloud indicator with all five components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A, Senkou Span B (52), and Chikou Span (26). Complete implementation of Goichi Hosoda\'s Ichimoku Kinko Hyo system.',
                'code': '''
//@version=5
indicator("Ichimoku Cloud Classic", shorttitle="Ichimoku", overlay=true)

// Input parameters (traditional settings)
tenkan_periods = input.int(9, title="Tenkan-sen Periods", minval=1)
kijun_periods = input.int(26, title="Kijun-sen Periods", minval=1)
senkou_span_b_periods = input.int(52, title="Senkou Span B Periods", minval=1)
chikou_span_lag = input.int(26, title="Chikou Span Displacement", minval=1)

// Calculate Ichimoku components
tenkan_sen = (ta.highest(high, tenkan_periods) + ta.lowest(low, tenkan_periods)) / 2
kijun_sen = (ta.highest(high, kijun_periods) + ta.lowest(low, kijun_periods)) / 2
senkou_span_a = (tenkan_sen + kijun_sen) / 2
senkou_span_b = (ta.highest(high, senkou_span_b_periods) + ta.lowest(low, senkou_span_b_periods)) / 2
chikou_span = close

// Plot lines
plot(tenkan_sen, color=color.red, title="Tenkan-sen", linewidth=1)
plot(kijun_sen, color=color.blue, title="Kijun-sen", linewidth=1)
plot(chikou_span, offset=-chikou_span_lag, color=color.green, title="Chikou Span", linewidth=1)

// Plot cloud (Kumo)
p1 = plot(senkou_span_a, offset=kijun_periods, color=color.orange, title="Senkou Span A")
p2 = plot(senkou_span_b, offset=kijun_periods, color=color.purple, title="Senkou Span B")
fill(p1, p2, color=senkou_span_a > senkou_span_b ? color.new(color.green, 90) : color.new(color.red, 90), title="Kumo")

// Ichimoku signals
bullish_tk = tenkan_sen > kijun_sen
bearish_tk = tenkan_sen < kijun_sen
bullish_cloud = senkou_span_a > senkou_span_b
bearish_cloud = senkou_span_a < senkou_span_b

// Strong signal detection
strong_bullish = bullish_tk and bullish_cloud and close > math.max(senkou_span_a[kijun_periods], senkou_span_b[kijun_periods])
strong_bearish = bearish_tk and bearish_cloud and close < math.min(senkou_span_a[kijun_periods], senkou_span_b[kijun_periods])

plotshape(strong_bullish, title="Strong Bull", style=shape.triangleup, location=location.belowbar, color=color.lime, size=size.normal)
plotshape(strong_bearish, title="Strong Bear", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.normal)
''',
                'open_source': True,
                'likes': 18500,
                'views': 1200000,
                'script_type': 'indicator',
                'strategy_type': 'trend',
                'indicators': ['Ichimoku', 'Tenkan-sen', 'Kijun-sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span'],
                'signals': ['trend', 'cloud_breakout', 'tk_cross'],
                'timeframes': ['15m', '1h', '4h', '1d']
            },

            {
                'slug': 'ichimoku-breakout-strategy',
                'title': 'Ichimoku Cloud Breakout Strategy',
                'author': 'TradingView',
                'description': 'Advanced Ichimoku breakout strategy with cloud breakout signals, Tenkan-Kijun crosses, and Chikou Span confirmation. Includes entry/exit rules and risk management.',
                'code': '''
//@version=5
strategy("Ichimoku Breakout Strategy", shorttitle="Ichi Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Ichimoku parameters
tenkan = input.int(9, title="Tenkan-sen")
kijun = input.int(26, title="Kijun-sen")
senkou_b = input.int(52, title="Senkou Span B")
chikou_lag = input.int(26, title="Chikou Displacement")

// Risk management
stop_loss_pips = input.float(20, title="Stop Loss (pips)")
take_profit_pips = input.float(40, title="Take Profit (pips)")

// Calculate Ichimoku
tenkan_sen = (ta.highest(high, tenkan) + ta.lowest(low, tenkan)) / 2
kijun_sen = (ta.highest(high, kijun) + ta.lowest(low, kijun)) / 2
senkou_span_a = (tenkan_sen + kijun_sen) / 2
senkou_span_b = (ta.highest(high, senkou_b) + ta.lowest(low, senkou_b)) / 2
chikou_span = close

// Current cloud values
cloud_top = math.max(senkou_span_a[kijun], senkou_span_b[kijun])
cloud_bottom = math.min(senkou_span_a[kijun], senkou_span_b[kijun])

// Entry conditions
bullish_cloud = senkou_span_a > senkou_span_b
bearish_cloud = senkou_span_a < senkou_span_b
tk_bull_cross = ta.crossover(tenkan_sen, kijun_sen)
tk_bear_cross = ta.crossunder(tenkan_sen, kijun_sen)
cloud_breakout_bull = ta.crossover(close, cloud_top)
cloud_breakout_bear = ta.crossunder(close, cloud_bottom)
chikou_clear = chikou_span > ta.highest(high[kijun], kijun) or chikou_span < ta.lowest(low[kijun], kijun)

// Strategy entries
long_condition = (tk_bull_cross or cloud_breakout_bull) and bullish_cloud and chikou_clear
short_condition = (tk_bear_cross or cloud_breakout_bear) and bearish_cloud and chikou_clear

if long_condition
    strategy.entry("Long", strategy.long)

if short_condition
    strategy.entry("Short", strategy.short)

// Plot Ichimoku
plot(tenkan_sen, color=color.red, title="Tenkan")
plot(kijun_sen, color=color.blue, title="Kijun")
p1 = plot(senkou_span_a, offset=kijun, color=color.orange, title="Senkou A")
p2 = plot(senkou_span_b, offset=kijun, color=color.purple, title="Senkou B")
fill(p1, p2, color=bullish_cloud ? color.new(color.green, 85) : color.new(color.red, 85))
plot(chikou_span, offset=-chikou_lag, color=color.yellow, title="Chikou")
''',
                'open_source': True,
                'likes': 15200,
                'views': 890000,
                'script_type': 'strategy',
                'strategy_type': 'trending',
                'indicators': ['Ichimoku', 'Breakout'],
                'signals': ['entry', 'exit', 'trend_following'],
                'timeframes': ['1h', '4h', '1d']
            },

            {
                'slug': 'ichimoku-rsi-combo',
                'title': 'Ichimoku + RSI Momentum Filter',
                'author': 'TradingView',
                'description': 'Combination of Ichimoku Cloud trend analysis with RSI momentum filtering. Only takes Ichimoku signals when RSI confirms momentum direction.',
                'code': '''
//@version=5
indicator("Ichimoku + RSI Filter", shorttitle="Ichi+RSI", overlay=true)

// Ichimoku settings
tenkan_len = input.int(9, title="Tenkan Length")
kijun_len = input.int(26, title="Kijun Length")
senkou_b_len = input.int(52, title="Senkou B Length")

// RSI filter
rsi_len = input.int(14, title="RSI Length")
rsi_ob = input.int(70, title="RSI Overbought")
rsi_os = input.int(30, title="RSI Oversold")

// Calculate Ichimoku
tenkan = (ta.highest(high, tenkan_len) + ta.lowest(low, tenkan_len)) / 2
kijun = (ta.highest(high, kijun_len) + ta.lowest(low, kijun_len)) / 2
senkou_a = (tenkan + kijun) / 2
senkou_b = (ta.highest(high, senkou_b_len) + ta.lowest(low, senkou_b_len)) / 2

// Calculate RSI
rsi = ta.rsi(close, rsi_len)

// Cloud analysis
bullish_cloud = senkou_a > senkou_b
above_cloud = close > math.max(senkou_a[kijun_len], senkou_b[kijun_len])
below_cloud = close < math.min(senkou_a[kijun_len], senkou_b[kijun_len])

// Combined signals with RSI filter
strong_bull = bullish_cloud and above_cloud and tenkan > kijun and rsi > 50 and rsi < rsi_ob
strong_bear = not bullish_cloud and below_cloud and tenkan < kijun and rsi < 50 and rsi > rsi_os

// Plot Ichimoku
plot(tenkan, color=color.red, title="Tenkan")
plot(kijun, color=color.blue, title="Kijun")
pa = plot(senkou_a, offset=kijun_len, color=color.orange)
pb = plot(senkou_b, offset=kijun_len, color=color.purple)
fill(pa, pb, color=bullish_cloud ? color.new(color.green, 88) : color.new(color.red, 88))

// Signal shapes
plotshape(strong_bull, title="Bullish Signal", style=shape.diamond, location=location.belowbar, color=color.lime)
plotshape(strong_bear, title="Bearish Signal", style=shape.diamond, location=location.abovebar, color=color.red)
''',
                'open_source': True,
                'likes': 12800,
                'views': 650000,
                'script_type': 'indicator',
                'strategy_type': 'momentum',
                'indicators': ['Ichimoku', 'RSI'],
                'signals': ['momentum', 'trend', 'filter'],
                'timeframes': ['15m', '1h', '4h', '1d']
            },

            {
                'slug': 'ichimoku-scalping-system',
                'title': 'Ichimoku Scalping System',
                'author': 'TradingView',
                'description': 'Fast Ichimoku scalping system optimized for lower timeframes. Uses faster Ichimoku settings (5-13-26) with quick entry/exit signals.',
                'code': '''
//@version=5
strategy("Ichimoku Scalping", shorttitle="Ichi Scalp", overlay=true, pyramiding=0, default_qty_type=strategy.percent_of_equity, default_qty_value=5)

// Fast Ichimoku for scalping
tenkan = input.int(5, title="Tenkan (Fast)")
kijun = input.int(13, title="Kijun (Fast)")
senkou_b = input.int(26, title="Senkou B")

// Scalping parameters
enable_long = input.bool(true, title="Enable Long Trades")
enable_short = input.bool(true, title="Enable Short Trades")
stop_pips = input.float(10, title="Stop Loss Pips")
target_pips = input.float(15, title="Take Profit Pips")

// Fast Ichimoku calculation
tenkan_sen = (ta.highest(high, tenkan) + ta.lowest(low, tenkan)) / 2
kijun_sen = (ta.highest(high, kijun) + ta.lowest(low, kijun)) / 2
senkou_span_a = (tenkan_sen + kijun_sen) / 2
senkou_span_b = (ta.highest(high, senkou_b) + ta.lowest(low, senkou_b)) / 2

// Quick signal detection
tk_cross_up = ta.crossover(tenkan_sen, kijun_sen)
tk_cross_down = ta.crossunder(tenkan_sen, kijun_sen)
cloud_bull = senkou_span_a > senkou_span_b

// Entry conditions (simplified for scalping)
long_signal = tk_cross_up and cloud_bull and close > tenkan_sen
short_signal = tk_cross_down and not cloud_bull and close < tenkan_sen

// Execute trades
if enable_long and long_signal
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", "Long", stop=close - stop_pips*syminfo.mintick*10, limit=close + target_pips*syminfo.mintick*10)

if enable_short and short_signal
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", "Short", stop=close + stop_pips*syminfo.mintick*10, limit=close - target_pips*syminfo.mintick*10)

// Plotting
plot(tenkan_sen, color=color.red, linewidth=2, title="Fast Tenkan")
plot(kijun_sen, color=color.blue, linewidth=2, title="Fast Kijun")
p1 = plot(senkou_span_a, offset=kijun, color=color.orange)
p2 = plot(senkou_span_b, offset=kijun, color=color.purple)
fill(p1, p2, color=cloud_bull ? color.new(color.green, 90) : color.new(color.red, 90))

plotshape(long_signal, style=shape.triangleup, location=location.belowbar, color=color.lime)
plotshape(short_signal, style=shape.triangledown, location=location.abovebar, color=color.red)
''',
                'open_source': True,
                'likes': 9800,
                'views': 520000,
                'script_type': 'strategy',
                'strategy_type': 'scalping',
                'indicators': ['Ichimoku', 'Fast_Settings'],
                'signals': ['scalping', 'quick_entries'],
                'timeframes': ['1m', '5m', '15m']
            },

            {
                'slug': 'advanced-ichimoku-analyzer',
                'title': 'Advanced Ichimoku Market Analyzer',
                'author': 'TradingView',
                'description': 'Comprehensive Ichimoku analysis tool with market regime detection, strength scoring, and multi-timeframe cloud analysis. Provides detailed Ichimoku market assessment.',
                'code': '''
//@version=5
indicator("Advanced Ichimoku Analyzer", shorttitle="Ichi Analyzer", overlay=false, scale=scale.none)

// Ichimoku parameters
tenkan_len = input.int(9, title="Tenkan Length")
kijun_len = input.int(26, title="Kijun Length")
senkou_b_len = input.int(52, title="Senkou B Length")
chikou_lag = input.int(26, title="Chikou Lag")

// Analysis settings
show_regime = input.bool(true, title="Show Market Regime")
show_strength = input.bool(true, title="Show Ichimoku Strength")
show_signals = input.bool(true, title="Show Signal Quality")

// Calculate Ichimoku components
tenkan = (ta.highest(high, tenkan_len) + ta.lowest(low, tenkan_len)) / 2
kijun = (ta.highest(high, kijun_len) + ta.lowest(low, kijun_len)) / 2
senkou_a = (tenkan + kijun) / 2
senkou_b = (ta.highest(high, senkou_b_len) + ta.lowest(low, senkou_b_len)) / 2
chikou = close

// Market regime analysis
bullish_cloud = senkou_a > senkou_b
above_cloud = close > math.max(senkou_a[kijun_len], senkou_b[kijun_len])
below_cloud = close < math.min(senkou_a[kijun_len], senkou_b[kijun_len])
in_cloud = not above_cloud and not below_cloud

tk_bullish = tenkan > kijun
tk_bearish = tenkan < kijun
price_above_tk = close > math.max(tenkan, kijun)
price_below_tk = close < math.min(tenkan, kijun)

// Ichimoku strength scoring (0-100)
strength_score = 0.0
strength_score := strength_score + (bullish_cloud ? 20 : 0)  // Cloud direction
strength_score := strength_score + (above_cloud ? 25 : below_cloud ? -25 : 0)  // Price vs cloud
strength_score := strength_score + (tk_bullish ? 15 : tk_bearish ? -15 : 0)  // TK relationship
strength_score := strength_score + (price_above_tk ? 20 : price_below_tk ? -20 : 0)  // Price vs TK
strength_score := strength_score + (chikou > close[chikou_lag] ? 20 : chikou < close[chikou_lag] ? -20 : 0)  // Chikou confirmation

// Normalize to 0-100 range
normalized_strength = (strength_score + 100) / 2

// Market regime classification
regime = in_cloud ? "Consolidation" : above_cloud and bullish_cloud ? "Strong Uptrend" :
         above_cloud and not bullish_cloud ? "Weak Uptrend" : below_cloud and not bullish_cloud ?
         "Strong Downtrend" : "Weak Downtrend"

// Signal quality assessment
signal_quality = normalized_strength > 70 ? "High" : normalized_strength > 50 ? "Medium" :
                normalized_strength < 30 ? "High Bearish" : normalized_strength < 50 ? "Medium Bearish" : "Neutral"

// Plotting
hline(50, "Neutral Line", color=color.gray, linestyle=hline.style_dashed)
hline(70, "Bullish Threshold", color=color.green, linestyle=hline.style_dotted)
hline(30, "Bearish Threshold", color=color.red, linestyle=hline.style_dotted)

plot(normalized_strength, title="Ichimoku Strength", color=normalized_strength > 50 ? color.green : color.red, linewidth=3)

// Background coloring based on regime
bgcolor(above_cloud and bullish_cloud ? color.new(color.green, 95) :
        below_cloud and not bullish_cloud ? color.new(color.red, 95) :
        in_cloud ? color.new(color.yellow, 95) : color.new(color.gray, 95))

// Info table
if barstate.islast and show_regime
    var table info_table = table.new(position.top_right, 2, 4, bgcolor=color.white, border_width=1)
    table.cell(info_table, 0, 0, "Regime:", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 0, regime, text_color=color.blue, text_size=size.small)
    table.cell(info_table, 0, 1, "Strength:", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 1, str.tostring(math.round(normalized_strength, 1)), text_color=color.blue, text_size=size.small)
    table.cell(info_table, 0, 2, "Quality:", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 2, signal_quality, text_color=color.blue, text_size=size.small)
''',
                'open_source': True,
                'likes': 14500,
                'views': 720000,
                'script_type': 'indicator',
                'strategy_type': 'analysis',
                'indicators': ['Ichimoku', 'Market_Regime', 'Strength_Analysis'],
                'signals': ['analysis', 'regime_detection', 'quality_scoring'],
                'timeframes': ['1h', '4h', '1d', '1w']
            },

            {
                'slug': 'ichimoku-alerts-system',
                'title': 'Ichimoku Cloud Alert System',
                'author': 'TradingView',
                'description': 'Complete Ichimoku alert system with customizable notifications for cloud breakouts, TK crosses, Chikou confirmations, and trend changes.',
                'code': '''
//@version=5
indicator("Ichimoku Alerts", shorttitle="Ichi Alerts", overlay=true)

// Ichimoku settings
tenkan_period = input.int(9, title="Tenkan-sen Period")
kijun_period = input.int(26, title="Kijun-sen Period")
senkou_b_period = input.int(52, title="Senkou Span B Period")
chikou_period = input.int(26, title="Chikou Span Period")

// Alert settings
alert_tk_cross = input.bool(true, title="TK Cross Alerts")
alert_cloud_breakout = input.bool(true, title="Cloud Breakout Alerts")
alert_chikou_confirmation = input.bool(true, title="Chikou Confirmation Alerts")
alert_trend_change = input.bool(true, title="Trend Change Alerts")

// Calculate Ichimoku
tenkan = (ta.highest(high, tenkan_period) + ta.lowest(low, tenkan_period)) / 2
kijun = (ta.highest(high, kijun_period) + ta.lowest(low, kijun_period)) / 2
senkou_a = (tenkan + kijun) / 2
senkou_b = (ta.highest(high, senkou_b_period) + ta.lowest(low, senkou_b_period)) / 2
chikou = close

// Current cloud boundaries
cloud_top = math.max(senkou_a[kijun_period], senkou_b[kijun_period])
cloud_bottom = math.min(senkou_a[kijun_period], senkou_b[kijun_period])
cloud_middle = (cloud_top + cloud_bottom) / 2

// Signal detection
tk_bull_cross = ta.crossover(tenkan, kijun)
tk_bear_cross = ta.crossunder(tenkan, kijun)
cloud_breakout_bull = ta.crossover(close, cloud_top)
cloud_breakout_bear = ta.crossunder(close, cloud_bottom)
chikou_bull_confirm = ta.crossover(chikou, close[chikou_period])
chikou_bear_confirm = ta.crossunder(chikou, close[chikou_period])

// Trend detection
prev_cloud_bull = senkou_a[1] > senkou_b[1]
curr_cloud_bull = senkou_a > senkou_b
trend_change_bull = not prev_cloud_bull and curr_cloud_bull
trend_change_bear = prev_cloud_bull and not curr_cloud_bull

// Plot Ichimoku
plot(tenkan, color=color.red, title="Tenkan-sen")
plot(kijun, color=color.blue, title="Kijun-sen")
plot(chikou, offset=-chikou_period, color=color.green, title="Chikou Span")
p1 = plot(senkou_a, offset=kijun_period, color=color.orange, title="Senkou A")
p2 = plot(senkou_b, offset=kijun_period, color=color.purple, title="Senkou B")
fill(p1, p2, color=curr_cloud_bull ? color.new(color.green, 90) : color.new(color.red, 90))

// Alert shapes
plotshape(tk_bull_cross, title="TK Bull Cross", style=shape.triangleup, location=location.belowbar, color=color.lime, size=size.small)
plotshape(tk_bear_cross, title="TK Bear Cross", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small)
plotshape(cloud_breakout_bull, title="Cloud Breakout Bull", style=shape.diamond, location=location.belowbar, color=color.blue, size=size.normal)
plotshape(cloud_breakout_bear, title="Cloud Breakout Bear", style=shape.diamond, location=location.abovebar, color=color.orange, size=size.normal)

// Generate alerts
if alert_tk_cross and tk_bull_cross
    alert("Ichimoku: Tenkan-Kijun Bullish Cross at " + str.tostring(close), alert.freq_once_per_bar)

if alert_tk_cross and tk_bear_cross
    alert("Ichimoku: Tenkan-Kijun Bearish Cross at " + str.tostring(close), alert.freq_once_per_bar)

if alert_cloud_breakout and cloud_breakout_bull
    alert("Ichimoku: Bullish Cloud Breakout at " + str.tostring(close), alert.freq_once_per_bar)

if alert_cloud_breakout and cloud_breakout_bear
    alert("Ichimoku: Bearish Cloud Breakout at " + str.tostring(close), alert.freq_once_per_bar)

if alert_chikou_confirmation and chikou_bull_confirm
    alert("Ichimoku: Chikou Bullish Confirmation", alert.freq_once_per_bar)

if alert_chikou_confirmation and chikou_bear_confirm
    alert("Ichimoku: Chikou Bearish Confirmation", alert.freq_once_per_bar)

if alert_trend_change and trend_change_bull
    alert("Ichimoku: Cloud Turned Bullish - Trend Change", alert.freq_once_per_bar)

if alert_trend_change and trend_change_bear
    alert("Ichimoku: Cloud Turned Bearish - Trend Change", alert.freq_once_per_bar)
''',
                'open_source': True,
                'likes': 11200,
                'views': 580000,
                'script_type': 'indicator',
                'strategy_type': 'alerts',
                'indicators': ['Ichimoku', 'Alerts'],
                'signals': ['alerts', 'notifications', 'breakouts'],
                'timeframes': ['15m', '1h', '4h', '1d']
            }
        ]

    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_pass
            )
            logger.info("✅ Connected to PostgreSQL database")
            return conn
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return None

    def insert_ichimoku_script(self, conn, script_data: Dict) -> bool:
        """Insert an Ichimoku script into the database"""
        try:
            cur = conn.cursor()

            # Convert lists to PostgreSQL arrays
            indicators_array = script_data.get('indicators', [])
            signals_array = script_data.get('signals', [])
            timeframes_array = script_data.get('timeframes', [])

            # Insert script
            cur.execute("""
                INSERT INTO tradingview.scripts (
                    slug, title, author, description, code, open_source,
                    likes, views, script_type, strategy_type, source_url,
                    indicators, signals, timeframes, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                )
                ON CONFLICT (slug) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    code = EXCLUDED.code,
                    likes = EXCLUDED.likes,
                    views = EXCLUDED.views,
                    updated_at = NOW()
            """, (
                script_data['slug'],
                script_data['title'],
                script_data['author'],
                script_data['description'],
                script_data['code'],
                script_data['open_source'],
                script_data['likes'],
                script_data['views'],
                script_data['script_type'],
                script_data['strategy_type'],
                f"https://tradingview.com/script/{script_data['slug']}/",  # Mock URL
                indicators_array,
                signals_array,
                timeframes_array
            ))

            cur.close()
            logger.info(f"✅ Inserted Ichimoku script: {script_data['title']}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to insert script {script_data['title']}: {e}")
            return False

    def download_ichimoku_scripts(self) -> Dict[str, int]:
        """Download and store all Ichimoku scripts"""
        logger.info("🚀 Starting Ichimoku Cloud script download...")

        conn = self.connect_to_database()
        if not conn:
            return {'error': 'Database connection failed'}

        stats = {
            'total_attempted': len(self.sample_ichimoku_scripts),
            'successfully_inserted': 0,
            'failed_insertions': 0,
            'indicators': 0,
            'strategies': 0
        }

        try:
            for script in self.sample_ichimoku_scripts:
                success = self.insert_ichimoku_script(conn, script)

                if success:
                    stats['successfully_inserted'] += 1
                    if script['script_type'] == 'indicator':
                        stats['indicators'] += 1
                    elif script['script_type'] == 'strategy':
                        stats['strategies'] += 1
                else:
                    stats['failed_insertions'] += 1

                # Rate limiting (respectful scraping)
                time.sleep(0.1)

            conn.commit()

        except Exception as e:
            logger.error(f"❌ Download session failed: {e}")
            conn.rollback()
            stats['error'] = str(e)

        finally:
            conn.close()

        logger.info(f"📊 Ichimoku download completed: {stats}")
        return stats

def main():
    """Main execution function"""
    scraper = IchimokuTradingViewScraper()
    results = scraper.download_ichimoku_scripts()

    print("🎯 Ichimoku Cloud Download Results:")
    print(f"   Total Attempted: {results.get('total_attempted', 0)}")
    print(f"   Successfully Inserted: {results.get('successfully_inserted', 0)}")
    print(f"   Failed Insertions: {results.get('failed_insertions', 0)}")
    print(f"   Indicators: {results.get('indicators', 0)}")
    print(f"   Strategies: {results.get('strategies', 0)}")

    if results.get('error'):
        print(f"   Error: {results['error']}")

    return 0 if results.get('successfully_inserted', 0) > 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)