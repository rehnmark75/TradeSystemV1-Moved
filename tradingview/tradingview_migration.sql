-- TradingView Scripts Migration
-- Run this against your PostgreSQL database

-- Create extension and schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE SCHEMA IF NOT EXISTS tradingview;


-- Create scripts table
CREATE TABLE IF NOT EXISTS tradingview.scripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(255) NOT NULL,
    description TEXT,
    code TEXT,
    open_source BOOLEAN DEFAULT TRUE,
    likes INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_url TEXT,
    script_type VARCHAR(50) DEFAULT 'strategy',
    strategy_type VARCHAR(50),
    indicators TEXT[],
    signals TEXT[],
    timeframes TEXT[],
    parameters JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_scripts_slug ON tradingview.scripts(slug);
CREATE INDEX IF NOT EXISTS idx_scripts_strategy_type ON tradingview.scripts(strategy_type);
CREATE INDEX IF NOT EXISTS idx_scripts_likes ON tradingview.scripts(likes DESC);
CREATE INDEX IF NOT EXISTS idx_scripts_title_fts ON tradingview.scripts USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_scripts_description_fts ON tradingview.scripts USING gin(to_tsvector('english', description));


-- Insert TradingView scripts
INSERT INTO tradingview.scripts (
    id, slug, title, author, description, code, open_source,
    likes, views, script_type, strategy_type, indicators,
    signals, timeframes, source_url, parameters, metadata
) VALUES
    (
        '5662e6c8-3d42-4599-be59-8a86256ab17a',
        'volume-weighted-average-price-vwap',
        'Volume Weighted Average Price (VWAP)',
        'TradingView',
        'The Volume Weighted Average Price (VWAP) is a trading benchmark used especially in pension funds',
        '
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
            ',
        True,
        15420,
        890000,
        'indicator',
        'indicator',
        ARRAY['VWAP'],
        ARRAY[]::TEXT[],
        ARRAY['1m','5m','15m','1h','4h','1d'],
        'https://www.tradingview.com/script/volume-weighted-average-price-vwap/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        'adefd591-3723-4d8b-949b-de85040139bc',
        'relative-strength-index-rsi',
        'Relative Strength Index (RSI)',
        'TradingView',
        'RSI is a momentum oscillator that measures the speed and magnitude of price changes',
        '
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
            ',
        True,
        12800,
        750000,
        'indicator',
        'indicator',
        ARRAY['RSI'],
        ARRAY[]::TEXT[],
        ARRAY['5m','15m','30m','1h','4h','1d'],
        'https://www.tradingview.com/script/relative-strength-index-rsi/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '2c4f0cb8-622f-4499-991e-001ab959d1f8',
        'moving-average-convergence-divergence-macd',
        'MACD (Moving Average Convergence Divergence)',
        'TradingView',
        'MACD is a trend-following momentum indicator that shows the relationship between two moving averages',
        '
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
            ',
        True,
        11200,
        680000,
        'indicator',
        'indicator',
        ARRAY['MACD'],
        ARRAY[]::TEXT[],
        ARRAY['30m','1h','4h','1d'],
        'https://www.tradingview.com/script/moving-average-convergence-divergence-macd/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        'd0e410ad-f3e6-4a36-8163-180992103390',
        'bollinger-bands-bb',
        'Bollinger Bands',
        'TradingView',
        'Bollinger Bands are volatility bands placed above and below a moving average',
        '
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
            ',
        True,
        9800,
        580000,
        'indicator',
        'indicator',
        ARRAY['BB'],
        ARRAY[]::TEXT[],
        ARRAY['15m','30m','1h','4h','1d'],
        'https://www.tradingview.com/script/bollinger-bands-bb/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '38dcf049-fd85-4ea2-924b-a6d83997fc36',
        'stochastic-oscillator',
        'Stochastic Oscillator',
        'TradingView',
        'The Stochastic Oscillator is a momentum indicator comparing a closing price to its price range',
        '
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
            ',
        True,
        8900,
        520000,
        'indicator',
        'indicator',
        ARRAY['Stochastic'],
        ARRAY[]::TEXT[],
        ARRAY['15m','30m','1h','4h','1d'],
        'https://www.tradingview.com/script/stochastic-oscillator/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '2f5ea6f5-9844-4c58-9aad-996c61729722',
        'average-true-range-atr',
        'Average True Range (ATR)',
        'TradingView',
        'ATR measures volatility by decomposing the entire range of an asset price for that period',
        '
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
            ',
        True,
        7800,
        420000,
        'indicator',
        'indicator',
        ARRAY['ATR'],
        ARRAY[]::TEXT[],
        ARRAY['1h','4h','1d','1w'],
        'https://www.tradingview.com/script/average-true-range-atr/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '2f9122f9-0298-4ec4-b7bc-6b95c3e20109',
        'awesome-oscillator-ao',
        'Awesome Oscillator (AO)',
        'TradingView',
        'The Awesome Oscillator is an indicator used to measure market momentum',
        '
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
            ',
        True,
        6700,
        340000,
        'indicator',
        'indicator',
        ARRAY['AO'],
        ARRAY[]::TEXT[],
        ARRAY['30m','1h','4h','1d'],
        'https://www.tradingview.com/script/awesome-oscillator-ao/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        'ebc8441b-7ed3-4758-bd8f-3a9d92a05209',
        'on-balance-volume-obv',
        'On Balance Volume (OBV)',
        'TradingView',
        'OBV uses volume flow to predict changes in stock price',
        '
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
            ',
        True,
        6200,
        320000,
        'indicator',
        'indicator',
        ARRAY['OBV'],
        ARRAY[]::TEXT[],
        ARRAY['30m','1h','4h','1d'],
        'https://www.tradingview.com/script/on-balance-volume-obv/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '2ee45623-ef4a-4353-81f1-2f40c181083d',
        'williams-percent-r',
        'Williams %R',
        'TradingView',
        'Williams %R is a momentum indicator that measures overbought and oversold levels',
        '
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
            ',
        True,
        5900,
        280000,
        'indicator',
        'indicator',
        ARRAY['Williams %R'],
        ARRAY[]::TEXT[],
        ARRAY['15m','30m','1h','4h','1d'],
        'https://www.tradingview.com/script/williams-percent-r/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        'dd2b5d51-0b6d-4b94-a997-01ac39e07dd9',
        'commodity-channel-index-cci',
        'Commodity Channel Index (CCI)',
        'TradingView',
        'CCI measures the difference between current price and its average price',
        '
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
            ',
        True,
        4800,
        210000,
        'indicator',
        'indicator',
        ARRAY['CCI'],
        ARRAY[]::TEXT[],
        ARRAY['30m','1h','4h','1d'],
        'https://www.tradingview.com/script/commodity-channel-index-cci/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        'c872590c-a893-4f83-847e-c8373db0747a',
        'ema-trend-following',
        'EMA Trend Following System',
        'TrendTrader',
        'Long-term trend following using multiple EMA timeframes',
        '
//@version=5
strategy("EMA Trend Following", overlay=true)

// EMA parameters for different timeframes
short_ema = input.int(10, title="Short EMA")
medium_ema = input.int(50, title="Medium EMA") 
long_ema = input.int(200, title="Long EMA")

// Calculate EMAs
ema10 = ta.ema(close, short_ema)
ema50 = ta.ema(close, medium_ema)
ema200 = ta.ema(close, long_ema)

// Trend identification
strong_uptrend = ema10 > ema50 and ema50 > ema200 and close > ema10
strong_downtrend = ema10 < ema50 and ema50 < ema200 and close < ema10

// Entry conditions
long_condition = ta.crossover(close, ema10) and strong_uptrend
short_condition = ta.crossunder(close, ema10) and strong_downtrend

// Exit conditions
long_exit = ta.crossunder(close, ema50)
short_exit = ta.crossover(close, ema50)

// Execute trades
if long_condition
    strategy.entry("Long", strategy.long)
if long_exit
    strategy.close("Long")
    
if short_condition
    strategy.entry("Short", strategy.short)
if short_exit
    strategy.close("Short")

// Plot EMAs
plot(ema10, color=color.blue, title="EMA 10")
plot(ema50, color=color.orange, title="EMA 50")  
plot(ema200, color=color.red, linewidth=2, title="EMA 200")
            ',
        True,
        520,
        12000,
        'strategy',
        'trending',
        ARRAY['EMA'],
        ARRAY['trend_following','crossover'],
        ARRAY['4h','1d','1w'],
        'https://www.tradingview.com/script/ema-trend-following/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '55a82f95-0b81-404a-bed8-efbda08d1654',
        'triple-ema-system',
        'Triple EMA System',
        'EMAExpert',
        'Advanced triple EMA system with trend confirmation',
        '
//@version=5
strategy("Triple EMA System", overlay=true)

// Input parameters
ema1_length = input.int(8, title="EMA 1 Length")
ema2_length = input.int(21, title="EMA 2 Length")
ema3_length = input.int(55, title="EMA 3 Length")

// Calculate EMAs
ema1 = ta.ema(close, ema1_length)
ema2 = ta.ema(close, ema2_length)
ema3 = ta.ema(close, ema3_length)

// Trend conditions
uptrend = ema1 > ema2 and ema2 > ema3
downtrend = ema1 < ema2 and ema2 < ema3

// Entry conditions
long_condition = ta.crossover(close, ema1) and uptrend
short_condition = ta.crossunder(close, ema1) and downtrend

// Execute trades
if long_condition
    strategy.entry("Long", strategy.long)
    
if short_condition
    strategy.entry("Short", strategy.short)

// Plot EMAs
plot(ema1, color=color.blue, title="Fast EMA")
plot(ema2, color=color.orange, title="Medium EMA")
plot(ema3, color=color.red, title="Slow EMA")
            ',
        True,
        420,
        8500,
        'strategy',
        'trending',
        ARRAY['EMA'],
        ARRAY['crossover','trend_confirmation'],
        ARRAY['15m','1h','4h'],
        'https://www.tradingview.com/script/triple-ema-system/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '62764d93-afc0-4f15-808c-3400a908737c',
        'ema-rsi-combo',
        'EMA + RSI Combo Strategy',
        'StrategyMaster',
        'EMA crossover with RSI confirmation for better entries',
        '
//@version=5
strategy("EMA + RSI Combo", overlay=true)

// EMA parameters
fast_ema_length = input.int(12, title="Fast EMA Length")
slow_ema_length = input.int(26, title="Slow EMA Length")

// RSI parameters
rsi_length = input.int(14, title="RSI Length")
rsi_oversold = input.int(30, title="RSI Oversold Level")
rsi_overbought = input.int(70, title="RSI Overbought Level")

// Calculate indicators
fast_ema = ta.ema(close, fast_ema_length)
slow_ema = ta.ema(close, slow_ema_length)
rsi = ta.rsi(close, rsi_length)

// Entry conditions
long_condition = ta.crossover(fast_ema, slow_ema) and rsi < rsi_overbought
short_condition = ta.crossunder(fast_ema, slow_ema) and rsi > rsi_oversold

// Execute trades
if long_condition
    strategy.entry("Long", strategy.long)
    
if short_condition
    strategy.entry("Short", strategy.short)

// Plot indicators
plot(fast_ema, color=color.blue, title="Fast EMA")
plot(slow_ema, color=color.red, title="Slow EMA")
            ',
        True,
        380,
        7200,
        'strategy',
        'trending',
        ARRAY['EMA','RSI'],
        ARRAY['crossover','momentum_confirmation'],
        ARRAY['30m','1h','2h'],
        'https://www.tradingview.com/script/ema-rsi-combo/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        '379b5aee-b35f-4393-aacd-cebd5ce3dd5f',
        'ema-crossover-basic',
        'EMA Crossover Basic Strategy',
        'TradingViewUser1',
        'Simple EMA crossover strategy using 9 and 21 period EMAs',
        '
//@version=5
strategy("EMA Crossover Basic", overlay=true)

// Input parameters
fast_ema_length = input.int(9, title="Fast EMA Length")
slow_ema_length = input.int(21, title="Slow EMA Length")

// Calculate EMAs
fast_ema = ta.ema(close, fast_ema_length)
slow_ema = ta.ema(close, slow_ema_length)

// Entry conditions
long_condition = ta.crossover(fast_ema, slow_ema)
short_condition = ta.crossunder(fast_ema, slow_ema)

// Execute trades
if long_condition
    strategy.entry("Long", strategy.long)
    
if short_condition
    strategy.entry("Short", strategy.short)

// Plot EMAs
plot(fast_ema, color=color.blue, title="Fast EMA")
plot(slow_ema, color=color.red, title="Slow EMA")
            ',
        True,
        250,
        5000,
        'strategy',
        'trending',
        ARRAY['EMA'],
        ARRAY['crossover'],
        ARRAY['1h','4h','1d'],
        'https://www.tradingview.com/script/ema-crossover-basic/',
        '{}',
        '{"migrated_from": "sqlite"}'
    ),
    (
        'd7b09bbf-4356-4a0d-bd4d-aa70ca35eab2',
        'ema-bounce-scalping',
        'EMA Bounce Scalping',
        'ScalpingPro',
        'Scalping strategy based on EMA bounce with tight stops',
        '
//@version=5
strategy("EMA Bounce Scalping", overlay=true)

// EMA parameters
ema_length = input.int(20, title="EMA Length")
bounce_threshold = input.float(0.1, title="Bounce Threshold %")

// Risk management
stop_loss_pct = input.float(0.5, title="Stop Loss %")
take_profit_pct = input.float(1.0, title="Take Profit %")

// Calculate EMA
ema = ta.ema(close, ema_length)

// Bounce conditions
price_near_ema = math.abs((close - ema) / ema * 100) < bounce_threshold
bullish_bounce = close < ema and close > low and ta.change(close) > 0
bearish_bounce = close > ema and close < high and ta.change(close) < 0

// Entry conditions
long_condition = bullish_bounce and price_near_ema
short_condition = bearish_bounce and price_near_ema

// Execute trades with risk management
if long_condition
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", "Long", stop=close * (1 - stop_loss_pct/100), limit=close * (1 + take_profit_pct/100))
    
if short_condition
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", "Short", stop=close * (1 + stop_loss_pct/100), limit=close * (1 - take_profit_pct/100))

// Plot EMA
plot(ema, color=color.yellow, linewidth=2, title="EMA")
            ',
        True,
        180,
        3500,
        'strategy',
        'scalping',
        ARRAY['EMA'],
        ARRAY['bounce','mean_reversion'],
        ARRAY['1m','5m','15m'],
        'https://www.tradingview.com/script/ema-bounce-scalping/',
        '{}',
        '{"migrated_from": "sqlite"}'
    );

-- Verify migration
SELECT COUNT(*) as total_scripts FROM tradingview.scripts;
SELECT script_type, COUNT(*) FROM tradingview.scripts GROUP BY script_type;