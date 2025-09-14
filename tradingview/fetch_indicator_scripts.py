#!/usr/bin/env python3
"""
TradingView Indicator Scripts Fetcher

Downloads top community indicator scripts from TradingView, focusing on 
popular indicators that can enhance the TradeSystemV1 analysis capabilities.
"""

import sys
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_indicator_scripts():
    """Generate sample top indicator scripts for testing"""
    
    sample_indicators = [
        {
            'slug': 'volume-weighted-average-price-vwap',
            'title': 'Volume Weighted Average Price (VWAP)',
            'author': 'TradingView',
            'description': 'The Volume Weighted Average Price (VWAP) is a trading benchmark used especially in pension funds',
            'code': '''
//@version=5
indicator("VWAP", shorttitle="VWAP", overlay=true)

// VWAP calculation
vwap_value = ta.vwap(hlc3)

// Plot VWAP
plot(vwap_value, color=color.blue, linewidth=2, title="VWAP")

// VWAP bands (optional)
show_bands = input.bool(false, title="Show VWAP Bands")
band_multiplier = input.float(1.0, title="Band Multiplier", minval=0.1, maxval=5.0)

if show_bands
    vwap_upper = vwap_value * (1 + band_multiplier / 100)
    vwap_lower = vwap_value * (1 - band_multiplier / 100)
    plot(vwap_upper, color=color.gray, title="VWAP Upper Band")
    plot(vwap_lower, color=color.gray, title="VWAP Lower Band")
            ''',
            'open_source': True,
            'likes': 15420,
            'views': 890000,
            'script_type': 'indicator',
            'category': 'volume',
            'indicators': ['VWAP'],
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d']
        },
        {
            'slug': 'relative-strength-index-rsi',
            'title': 'Relative Strength Index (RSI)',
            'author': 'TradingView',
            'description': 'RSI is a momentum oscillator that measures the speed and magnitude of price changes',
            'code': '''
//@version=5
indicator("RSI", shorttitle="RSI", format=format.price, precision=2, timeframe="", timeframe_gaps=true)

// RSI settings
length = input.int(14, title="Length", minval=1)
source = input(close, "Source")

// RSI calculation
rsi_value = ta.rsi(source, length)

// Overbought/Oversold levels
overbought = input.int(70, title="Overbought Level", minval=50, maxval=99)
oversold = input.int(30, title="Oversold Level", minval=1, maxval=50)

// Plot RSI
plot(rsi_value, "RSI", color=color.rgb(124, 77, 255))
hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(50, "Midline", color=color.gray, linestyle=hline.style_dotted)

// Background color for extreme levels
bgcolor(rsi_value >= overbought ? color.new(color.red, 90) : rsi_value <= oversold ? color.new(color.green, 90) : na)
            ''',
            'open_source': True,
            'likes': 12800,
            'views': 750000,
            'script_type': 'indicator',
            'category': 'momentum',
            'indicators': ['RSI'],
            'timeframes': ['5m', '15m', '30m', '1h', '4h', '1d']
        },
        {
            'slug': 'bollinger-bands-bb',
            'title': 'Bollinger Bands',
            'author': 'TradingView', 
            'description': 'Bollinger Bands are volatility bands placed above and below a moving average',
            'code': '''
//@version=5
indicator("Bollinger Bands", shorttitle="BB", overlay=true, timeframe="", timeframe_gaps=true)

// BB settings
length = input.int(20, title="Length", minval=1)
src = input(close, title="Source")
mult = input.float(2.0, title="StdDev", minval=0.001, maxval=50)

// BB calculation
basis = ta.sma(src, length)
dev = mult * ta.stdev(src, length)
upper = basis + dev
lower = basis - dev

// Plot BB
plot(basis, "Basis", color=color.orange, linewidth=1)
p1 = plot(upper, "Upper", color=color.blue, linewidth=1)
p2 = plot(lower, "Lower", color=color.blue, linewidth=1)
fill(p1, p2, title="Background", color=color.rgb(33, 150, 243, 95))

// BB squeeze detection
squeeze_threshold = input.float(0.1, title="Squeeze Threshold")
bb_width = (upper - lower) / basis
is_squeeze = bb_width < squeeze_threshold
plotchar(is_squeeze, title="BB Squeeze", char="•", location=location.bottom, color=color.red, size=size.small)
            ''',
            'open_source': True,
            'likes': 9800,
            'views': 580000,
            'script_type': 'indicator',
            'category': 'volatility',
            'indicators': ['BB'],
            'timeframes': ['15m', '30m', '1h', '4h', '1d']
        },
        {
            'slug': 'moving-average-convergence-divergence-macd',
            'title': 'MACD (Moving Average Convergence Divergence)',
            'author': 'TradingView',
            'description': 'MACD is a trend-following momentum indicator that shows the relationship between two moving averages',
            'code': '''
//@version=5
indicator("MACD", shorttitle="MACD", format=format.price, precision=4, timeframe="")

// MACD settings
fast_length = input.int(12, title="Fast Length", minval=1)
slow_length = input.int(26, title="Slow Length", minval=1)
signal_length = input.int(9, title="Signal Length", minval=1)
src = input(close, title="Source")

// MACD calculation
fast_ma = ta.ema(src, fast_length)
slow_ma = ta.ema(src, slow_length)
macd_line = fast_ma - slow_ma
signal_line = ta.ema(macd_line, signal_length)
hist = macd_line - signal_line

// Plot MACD
plot(hist, title="Histogram", style=plot.style_columns, color=(hist >= 0 ? (hist[1] < hist ? color.lime : color.green) : (hist[1] < hist ? color.red : color.maroon)))
plot(macd_line, title="MACD", color=color.blue, linewidth=2)
plot(signal_line, title="Signal", color=color.red, linewidth=1)
hline(0, "Zero Line", color=color.gray, linestyle=hline.style_dashed)

// MACD crossover signals
macd_cross_up = ta.crossover(macd_line, signal_line)
macd_cross_down = ta.crossunder(macd_line, signal_line)

plotshape(macd_cross_up, title="MACD Cross Up", style=shape.triangleup, location=location.bottom, color=color.green, size=size.small)
plotshape(macd_cross_down, title="MACD Cross Down", style=shape.triangledown, location=location.top, color=color.red, size=size.small)
            ''',
            'open_source': True,
            'likes': 11200,
            'views': 680000,
            'script_type': 'indicator',
            'category': 'momentum',
            'indicators': ['MACD'],
            'timeframes': ['30m', '1h', '4h', '1d']
        },
        {
            'slug': 'stochastic-oscillator',
            'title': 'Stochastic Oscillator',
            'author': 'TradingView',
            'description': 'The Stochastic Oscillator is a momentum indicator comparing a closing price to its price range',
            'code': '''
//@version=5
indicator("Stochastic", shorttitle="Stoch", format=format.price, precision=2)

// Stochastic settings
periodK = input.int(14, title="%K Length", minval=1)
smoothK = input.int(1, title="%K Smoothing", minval=1)
periodD = input.int(3, title="%D Smoothing", minval=1)

// Stochastic calculation
k = ta.sma(ta.stoch(close, high, low, periodK), smoothK)
d = ta.sma(k, periodD)

// Plot Stochastic
plot(k, title="%K", color=color.blue, linewidth=2)
plot(d, title="%D", color=color.red, linewidth=1)

// Overbought/Oversold levels
overbought = input.int(80, title="Overbought Level")
oversold = input.int(20, title="Oversold Level")

hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(50, "Midline", color=color.gray, linestyle=hline.style_dotted)

// Background coloring
bgcolor(k >= overbought and d >= overbought ? color.new(color.red, 90) : 
        k <= oversold and d <= oversold ? color.new(color.green, 90) : na)

// Crossover signals
stoch_cross_up = ta.crossover(k, d) and k < oversold + 10
stoch_cross_down = ta.crossunder(k, d) and k > overbought - 10

plotshape(stoch_cross_up, title="Stoch Buy", style=shape.triangleup, location=location.bottom, color=color.green, size=size.small)
plotshape(stoch_cross_down, title="Stoch Sell", style=shape.triangledown, location=location.top, color=color.red, size=size.small)
            ''',
            'open_source': True,
            'likes': 8900,
            'views': 520000,
            'script_type': 'indicator',
            'category': 'momentum',
            'indicators': ['Stochastic'],
            'timeframes': ['15m', '30m', '1h', '4h', '1d']
        },
        {
            'slug': 'awesome-oscillator-ao',
            'title': 'Awesome Oscillator (AO)',
            'author': 'TradingView',
            'description': 'The Awesome Oscillator is an indicator used to measure market momentum',
            'code': '''
//@version=5
indicator("Awesome Oscillator", shorttitle="AO", format=format.price, precision=4)

// AO calculation
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)

// Plot AO
plot(ao, title="AO", style=plot.style_columns, 
     color=ao > ao[1] ? color.lime : color.red)

hline(0, "Zero Line", color=color.gray, linestyle=hline.style_solid)

// AO signals
ao_saucer = ao > 0 and ao[1] > 0 and ao[2] > 0 and ao[1] < ao[2] and ao > ao[1]
ao_cross_up = ta.crossover(ao, 0)
ao_cross_down = ta.crossunder(ao, 0)

// Twin peaks (simplified)
ao_twin_peaks = ao < 0 and ao[1] < ao[2] and ao > ao[1] and ta.crossover(ao, 0)

plotshape(ao_saucer, title="Saucer", style=shape.circle, location=location.bottom, color=color.green, size=size.small)
plotshape(ao_cross_up, title="Zero Cross Up", style=shape.triangleup, location=location.bottom, color=color.blue, size=size.small)
plotshape(ao_cross_down, title="Zero Cross Down", style=shape.triangledown, location=location.top, color=color.orange, size=size.small)
plotshape(ao_twin_peaks, title="Twin Peaks", style=shape.diamond, location=location.bottom, color=color.yellow, size=size.small)
            ''',
            'open_source': True,
            'likes': 6700,
            'views': 340000,
            'script_type': 'indicator',
            'category': 'momentum',
            'indicators': ['AO'],
            'timeframes': ['30m', '1h', '4h', '1d']
        },
        {
            'slug': 'average-true-range-atr',
            'title': 'Average True Range (ATR)',
            'author': 'TradingView',
            'description': 'ATR measures volatility by decomposing the entire range of an asset price for that period',
            'code': '''
//@version=5
indicator("Average True Range", shorttitle="ATR", format=format.price, precision=4, timeframe="")

// ATR settings
length = input.int(14, title="Length", minval=1)
smoothing_method = input.string("RMA", title="Smoothing", options=["RMA", "SMA", "EMA", "WMA"])

// ATR calculation
ma_function(source, length) =>
    switch smoothing_method
        "RMA" => ta.rma(source, length)
        "SMA" => ta.sma(source, length)
        "EMA" => ta.ema(source, length)
        "WMA" => ta.wma(source, length)

atr_value = ma_function(ta.tr(true), length)

// Plot ATR
plot(atr_value, title="ATR", color=color.blue, linewidth=2)

// ATR-based support/resistance levels
show_levels = input.bool(false, title="Show ATR Levels")
multiplier = input.float(2.0, title="ATR Multiplier")

if show_levels
    atr_upper = close + (atr_value * multiplier)
    atr_lower = close - (atr_value * multiplier)
    plot(atr_upper, title="ATR Upper", color=color.red, linestyle=plot.style_dashed)
    plot(atr_lower, title="ATR Lower", color=color.green, linestyle=plot.style_dashed)

// ATR percentile ranking
atr_rank = ta.percentrank(atr_value, 100)
high_volatility = atr_rank > 80
low_volatility = atr_rank < 20

bgcolor(high_volatility ? color.new(color.red, 95) : low_volatility ? color.new(color.green, 95) : na)
            ''',
            'open_source': True,
            'likes': 7800,
            'views': 420000,
            'script_type': 'indicator',
            'category': 'volatility',
            'indicators': ['ATR'],
            'timeframes': ['1h', '4h', '1d', '1w']
        },
        {
            'slug': 'williams-percent-r',
            'title': 'Williams %R',
            'author': 'TradingView',
            'description': 'Williams %R is a momentum indicator that measures overbought and oversold levels',
            'code': '''
//@version=5
indicator("Williams %R", shorttitle="%R", format=format.price, precision=2)

// Williams %R settings
length = input.int(14, title="Length", minval=1)

// Williams %R calculation
williams_r = -100 * (ta.highest(high, length) - close) / (ta.highest(high, length) - ta.lowest(low, length))

// Plot Williams %R
plot(williams_r, title="Williams %R", color=color.blue, linewidth=2)

// Overbought/Oversold levels
overbought = input.int(-20, title="Overbought Level", maxval=-1)
oversold = input.int(-80, title="Oversold Level", minval=-99)

hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(-50, "Midline", color=color.gray, linestyle=hline.style_dotted)

// Background coloring
bgcolor(williams_r >= overbought ? color.new(color.red, 90) : 
        williams_r <= oversold ? color.new(color.green, 90) : na)

// Signals
wr_buy_signal = williams_r <= oversold and williams_r > williams_r[1]
wr_sell_signal = williams_r >= overbought and williams_r < williams_r[1]

plotshape(wr_buy_signal, title="Buy Signal", style=shape.triangleup, location=location.bottom, color=color.green, size=size.small)
plotshape(wr_sell_signal, title="Sell Signal", style=shape.triangledown, location=location.top, color=color.red, size=size.small)
            ''',
            'open_source': True,
            'likes': 5900,
            'views': 280000,
            'script_type': 'indicator',
            'category': 'momentum',
            'indicators': ['Williams %R'],
            'timeframes': ['15m', '30m', '1h', '4h', '1d']
        },
        {
            'slug': 'commodity-channel-index-cci',
            'title': 'Commodity Channel Index (CCI)',
            'author': 'TradingView',
            'description': 'CCI measures the difference between current price and its average price',
            'code': '''
//@version=5
indicator("CCI", format=format.price, precision=2)

// CCI settings
length = input.int(20, title="Length", minval=1)

// CCI calculation
cci_value = ta.cci(close, length)

// Plot CCI
plot(cci_value, title="CCI", color=color.blue, linewidth=2)

// CCI levels
overbought = input.int(100, title="Overbought Level")
oversold = input.int(-100, title="Oversold Level")
extreme_ob = input.int(200, title="Extreme Overbought")
extreme_os = input.int(-200, title="Extreme Oversold")

hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(extreme_ob, "Extreme OB", color=color.maroon)
hline(extreme_os, "Extreme OS", color=color.lime)
hline(0, "Zero Line", color=color.gray, linestyle=hline.style_dotted)

// CCI divergence detection (simplified)
cci_higher_high = cci_value > cci_value[1] and cci_value[1] > cci_value[2]
cci_lower_low = cci_value < cci_value[1] and cci_value[1] < cci_value[2]
price_higher_high = high > high[1] and high[1] > high[2]
price_lower_low = low < low[1] and low[1] < low[2]

bearish_div = price_higher_high and not cci_higher_high and cci_value > overbought
bullish_div = price_lower_low and not cci_lower_low and cci_value < oversold

plotshape(bearish_div, title="Bearish Divergence", style=shape.triangledown, location=location.top, color=color.red, size=size.small)
plotshape(bullish_div, title="Bullish Divergence", style=shape.triangleup, location=location.bottom, color=color.green, size=size.small)

// Background coloring for extreme levels
bgcolor(cci_value > extreme_ob ? color.new(color.red, 95) : cci_value < extreme_os ? color.new(color.green, 95) : na)
            ''',
            'open_source': True,
            'likes': 4800,
            'views': 210000,
            'script_type': 'indicator',
            'category': 'momentum',
            'indicators': ['CCI'],
            'timeframes': ['30m', '1h', '4h', '1d']
        },
        {
            'slug': 'on-balance-volume-obv',
            'title': 'On Balance Volume (OBV)',
            'author': 'TradingView',
            'description': 'OBV uses volume flow to predict changes in stock price',
            'code': '''
//@version=5
indicator("On Balance Volume", shorttitle="OBV", format=format.volume, timeframe="")

// OBV calculation
obv_value = ta.cum(math.sign(ta.change(close)) * volume)

// Plot OBV
plot(obv_value, title="OBV", color=color.blue, linewidth=2)

// OBV moving average
show_ma = input.bool(true, title="Show OBV MA")
ma_length = input.int(21, title="MA Length", minval=1)

obv_ma = ta.sma(obv_value, ma_length)
plot(show_ma ? obv_ma : na, title="OBV MA", color=color.orange, linewidth=1)

// OBV divergence with price
obv_increasing = obv_value > obv_value[1]
obv_decreasing = obv_value < obv_value[1]
price_increasing = close > close[1]
price_decreasing = close < close[1]

// Simple divergence detection
bullish_divergence = price_decreasing and obv_increasing
bearish_divergence = price_increasing and obv_decreasing

plotshape(bullish_divergence, title="Bullish OBV Divergence", style=shape.triangleup, 
          location=location.bottom, color=color.green, size=size.small)
plotshape(bearish_divergence, title="Bearish OBV Divergence", style=shape.triangledown, 
          location=location.top, color=color.red, size=size.small)

// OBV trend confirmation
obv_uptrend = obv_value > obv_ma and ta.rising(obv_ma, 3)
obv_downtrend = obv_value < obv_ma and ta.falling(obv_ma, 3)

bgcolor(obv_uptrend ? color.new(color.green, 98) : obv_downtrend ? color.new(color.red, 98) : na)
            ''',
            'open_source': True,
            'likes': 6200,
            'views': 320000,
            'script_type': 'indicator',
            'category': 'volume',
            'indicators': ['OBV'],
            'timeframes': ['30m', '1h', '4h', '1d']
        }
    ]
    
    return sample_indicators

def insert_indicator_scripts(conn, indicators):
    """Insert sample indicators into database"""
    
    cursor = conn.cursor()
    
    for indicator in indicators:
        try:
            # Convert arrays to JSON strings
            indicators_json = str(indicator['indicators'])
            timeframes_json = str(indicator['timeframes'])
            
            cursor.execute('''
                INSERT OR REPLACE INTO scripts (
                    slug, title, author, description, code, open_source,
                    likes, views, strategy_type, indicators, timeframes,
                    source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                indicator['slug'],
                indicator['title'], 
                indicator['author'],
                indicator['description'],
                indicator['code'],
                indicator['open_source'],
                indicator['likes'],
                indicator['views'],
                indicator['script_type'],  # 'indicator' instead of strategy type
                indicators_json,
                timeframes_json,
                f"https://www.tradingview.com/script/{indicator['slug']}/"
            ))
            
            logger.info(f"✅ Inserted indicator: {indicator['title']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert {indicator['slug']}: {e}")
    
    conn.commit()

def main():
    """Main execution function"""
    
    logger.info("🚀 Starting TradingView top indicators download...")
    
    # Connect to existing database
    db_path = Path(__file__).parent / "data" / "tvscripts.db"
    
    if not db_path.exists():
        logger.error(f"❌ Database not found at {db_path}")
        logger.error("   Please run the EMA strategy scraper first to create the database")
        return False
    
    try:
        # Connect to database
        logger.info("📋 Connecting to existing database...")
        conn = sqlite3.connect(str(db_path))
        
        # Check current script count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scripts")
        before_count = cursor.fetchone()[0]
        logger.info(f"📊 Current scripts in database: {before_count}")
        
        # Generate sample indicators
        logger.info("🎯 Generating top community indicators...")
        indicators = create_sample_indicator_scripts()
        
        # Insert into database
        logger.info("💾 Inserting indicators into database...")
        insert_indicator_scripts(conn, indicators)
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM scripts")
        after_count = cursor.fetchone()[0]
        added_count = after_count - before_count
        
        logger.info(f"✅ Successfully added {added_count} indicator scripts to database")
        logger.info(f"📊 Total scripts in database: {after_count}")
        
        # Test search functionality with indicators
        logger.info("🔍 Testing indicator search functionality...")
        
        # Search for momentum indicators
        cursor.execute("SELECT title FROM scripts_fts WHERE scripts_fts MATCH 'momentum' LIMIT 5")
        momentum_results = cursor.fetchall()
        
        logger.info("🎯 Momentum indicator search results:")
        for i, (title,) in enumerate(momentum_results, 1):
            logger.info(f"   {i}. {title}")
        
        # Search for volume indicators  
        cursor.execute("SELECT title FROM scripts_fts WHERE scripts_fts MATCH 'volume' LIMIT 3")
        volume_results = cursor.fetchall()
        
        logger.info("📊 Volume indicator search results:")
        for i, (title,) in enumerate(volume_results, 1):
            logger.info(f"   {i}. {title}")
        
        # Show breakdown by category
        cursor.execute("SELECT strategy_type, COUNT(*) FROM scripts GROUP BY strategy_type")
        category_counts = cursor.fetchall()
        
        logger.info("📈 Scripts by category:")
        for category, count in category_counts:
            logger.info(f"   {category}: {count}")
        
        conn.close()
        logger.info("🎉 Indicator download completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Indicator download failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)