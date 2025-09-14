#!/usr/bin/env python3
"""
Run TradingView scraper in fallback mode (no dependencies required)
Generates sample EMA strategies for testing the integration.
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

def create_sample_ema_strategies():
    """Generate sample EMA strategies for testing"""
    
    sample_strategies = [
        {
            'slug': 'ema-crossover-basic',
            'title': 'EMA Crossover Basic Strategy',
            'author': 'TradingViewUser1',
            'description': 'Simple EMA crossover strategy using 9 and 21 period EMAs',
            'code': '''
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
            ''',
            'open_source': True,
            'likes': 250,
            'views': 5000,
            'strategy_type': 'trending',
            'indicators': ['EMA'],
            'signals': ['crossover'],
            'timeframes': ['1h', '4h', '1d']
        },
        {
            'slug': 'triple-ema-system',
            'title': 'Triple EMA System',
            'author': 'EMAExpert',
            'description': 'Advanced triple EMA system with trend confirmation',
            'code': '''
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
            ''',
            'open_source': True,
            'likes': 420,
            'views': 8500,
            'strategy_type': 'trending',
            'indicators': ['EMA'],
            'signals': ['crossover', 'trend_confirmation'],
            'timeframes': ['15m', '1h', '4h']
        },
        {
            'slug': 'ema-rsi-combo',
            'title': 'EMA + RSI Combo Strategy',
            'author': 'StrategyMaster',
            'description': 'EMA crossover with RSI confirmation for better entries',
            'code': '''
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
            ''',
            'open_source': True,
            'likes': 380,
            'views': 7200,
            'strategy_type': 'trending',
            'indicators': ['EMA', 'RSI'],
            'signals': ['crossover', 'momentum_confirmation'],
            'timeframes': ['30m', '1h', '2h']
        },
        {
            'slug': 'ema-bounce-scalping',
            'title': 'EMA Bounce Scalping',
            'author': 'ScalpingPro',
            'description': 'Scalping strategy based on EMA bounce with tight stops',
            'code': '''
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
            ''',
            'open_source': True,
            'likes': 180,
            'views': 3500,
            'strategy_type': 'scalping',
            'indicators': ['EMA'],
            'signals': ['bounce', 'mean_reversion'],
            'timeframes': ['1m', '5m', '15m']
        },
        {
            'slug': 'ema-trend-following',
            'title': 'EMA Trend Following System',
            'author': 'TrendTrader',
            'description': 'Long-term trend following using multiple EMA timeframes',
            'code': '''
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
            ''',
            'open_source': True,
            'likes': 520,
            'views': 12000,
            'strategy_type': 'trending',
            'indicators': ['EMA'],
            'signals': ['trend_following', 'crossover'],
            'timeframes': ['4h', '1d', '1w']
        }
    ]
    
    return sample_strategies

def create_database(db_path: str):
    """Create SQLite database with FTS5 search"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create main scripts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            description TEXT,
            code TEXT,
            open_source BOOLEAN DEFAULT TRUE,
            likes INTEGER DEFAULT 0,
            views INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_url TEXT,
            strategy_type TEXT,
            indicators TEXT, -- JSON array
            signals TEXT,    -- JSON array  
            timeframes TEXT  -- JSON array
        )
    ''')
    
    # Create FTS5 search table
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS scripts_fts USING fts5(
            title, 
            description, 
            code,
            author,
            indicators,
            signals,
            content='scripts',
            content_rowid='id'
        )
    ''')
    
    # Create triggers to keep FTS5 in sync
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS scripts_fts_insert AFTER INSERT ON scripts BEGIN
            INSERT INTO scripts_fts(rowid, title, description, code, author, indicators, signals)
            VALUES (new.id, new.title, new.description, new.code, new.author, new.indicators, new.signals);
        END
    ''')
    
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS scripts_fts_delete AFTER DELETE ON scripts BEGIN
            INSERT INTO scripts_fts(scripts_fts, rowid, title, description, code, author, indicators, signals)
            VALUES ('delete', old.id, old.title, old.description, old.code, old.author, old.indicators, old.signals);
        END
    ''')
    
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS scripts_fts_update AFTER UPDATE ON scripts BEGIN
            INSERT INTO scripts_fts(scripts_fts, rowid, title, description, code, author, indicators, signals)
            VALUES ('delete', old.id, old.title, old.description, old.code, old.author, old.indicators, old.signals);
            INSERT INTO scripts_fts(rowid, title, description, code, author, indicators, signals)
            VALUES (new.id, new.title, new.description, new.code, new.author, new.indicators, new.signals);
        END
    ''')
    
    conn.commit()
    return conn

def insert_sample_strategies(conn, strategies):
    """Insert sample strategies into database"""
    
    cursor = conn.cursor()
    
    for strategy in strategies:
        try:
            # Convert arrays to JSON strings
            indicators_json = str(strategy['indicators'])
            signals_json = str(strategy['signals'])
            timeframes_json = str(strategy['timeframes'])
            
            cursor.execute('''
                INSERT OR REPLACE INTO scripts (
                    slug, title, author, description, code, open_source,
                    likes, views, strategy_type, indicators, signals, timeframes,
                    source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy['slug'],
                strategy['title'], 
                strategy['author'],
                strategy['description'],
                strategy['code'],
                strategy['open_source'],
                strategy['likes'],
                strategy['views'],
                strategy['strategy_type'],
                indicators_json,
                signals_json,
                timeframes_json,
                f"https://www.tradingview.com/script/{strategy['slug']}/"
            ))
            
            logger.info(f"✅ Inserted: {strategy['title']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert {strategy['slug']}: {e}")
    
    conn.commit()

def main():
    """Main execution function"""
    
    logger.info("🚀 Starting TradingView scraper in fallback mode...")
    
    # Create database directory
    db_dir = Path(__file__).parent / "data"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "tvscripts.db"
    
    try:
        # Create database
        logger.info("📋 Creating database schema...")
        conn = create_database(str(db_path))
        
        # Generate sample strategies
        logger.info("🎯 Generating sample EMA strategies...")
        strategies = create_sample_ema_strategies()
        
        # Insert into database
        logger.info("💾 Inserting strategies into database...")
        insert_sample_strategies(conn, strategies)
        
        # Verify insertion
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scripts")
        count = cursor.fetchone()[0]
        
        logger.info(f"✅ Successfully added {count} EMA strategies to database")
        logger.info(f"📍 Database location: {db_path}")
        
        # Test search functionality
        logger.info("🔍 Testing search functionality...")
        cursor.execute("SELECT title FROM scripts_fts WHERE scripts_fts MATCH 'EMA' LIMIT 3")
        results = cursor.fetchall()
        
        logger.info("🎯 Search test results:")
        for i, (title,) in enumerate(results, 1):
            logger.info(f"   {i}. {title}")
        
        conn.close()
        logger.info("🎉 Scraper completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Scraper failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)