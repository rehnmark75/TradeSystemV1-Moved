# Market Intelligence System

This document provides comprehensive guidance for the Market Intelligence system that delivers real-time market analysis and trade context evaluation in the Streamlit TradingView chart interface.

## System Overview

The Market Intelligence system provides sophisticated market analysis capabilities that work independently of the forex_scanner module. It uses simplified analysis algorithms when the full MarketIntelligenceEngine is not available, ensuring robust operation across different deployment scenarios.

### Key Capabilities

- **Real-time Market Regime Detection**: Identifies trending vs ranging market conditions
- **Trading Session Analysis**: Tracks global trading sessions and volatility patterns
- **Trade Context Evaluation**: AI-powered assessment of trade quality and timing
- **Performance Analytics**: Historical analysis of regime-based performance
- **Intelligent Recommendations**: Actionable suggestions for trade improvement

## Architecture Components

### 1. StreamlitDataFetcher (`streamlit/services/data_fetcher_adapter.py`)

**Core Responsibilities:**
- Adapts Streamlit's database connection for intelligence analysis
- Maps timeframes correctly (5m=5, 15m=15, 1h=60 as integers)
- Enhances data with technical indicators (EMAs, ATR, volatility)
- Optimizes SQL queries for ig_candles table structure

**Key Features:**
```python
class StreamlitDataFetcher:
    def __init__(self, engine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        
    def get_enhanced_data(self, epic: str, timeframe: str, lookback_hours: int = 48):
        """Fetch and enhance market data with technical indicators"""
        # Map string timeframes to database integers
        timeframe_map = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        tf_minutes = timeframe_map.get(timeframe, 15)
        
        # Optimized SQL query
        query = """
            SELECT start_time, epic, open_price_mid as open, high_price_mid as high,
                   low_price_mid as low, close_price_mid as close, volume, ltv
            FROM ig_candles 
            WHERE epic = %s AND timeframe = %s 
            AND start_time >= NOW() - INTERVAL '%s hours'
            ORDER BY start_time ASC
        """
        
        df = pd.read_sql_query(query, self.engine, params=[epic, tf_minutes, lookback_hours])
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        return df
```

**Technical Indicator Enhancement:**
```python
def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add essential technical indicators for market analysis"""
    if len(df) < 21:
        return df
        
    # EMA calculations for trend analysis
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()  
    df['ema_200'] = df['close'].ewm(span=200).mean()
    
    # ATR for volatility measurement
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['atr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1).rolling(14).mean()
    
    # Volatility percentile ranking
    df['volatility'] = df['close'].rolling(20).std()
    df['volatility_percentile'] = df['volatility'].rolling(100).rank(pct=True)
    
    return df
```

### 2. MarketIntelligenceService (`streamlit/services/market_intelligence_service.py`)

**Core Responsibilities:**
- Main service wrapper for market intelligence operations
- Provides both full and simplified analysis modes
- Implements 5-minute caching mechanism for performance
- Handles graceful fallback when forex_scanner module unavailable

**Service Architecture:**
```python
class MarketIntelligenceService:
    def __init__(self, data_fetcher=None):
        self.data_fetcher = data_fetcher
        self.cache_duration = timedelta(minutes=5)
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
    def get_regime_for_timeframe(self, epic: str, timeframe: str) -> Dict:
        """Get market regime with caching for performance"""
        cache_key = f"regime_{epic}_{timeframe}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
            
        try:
            # Try full analysis first
            regime_data = self._get_full_regime_analysis(epic, timeframe)
        except Exception as e:
            self.logger.warning(f"Full analysis failed: {e}, using simplified mode")
            regime_data = self._get_simplified_regime_analysis(epic, timeframe)
            
        self.cache[cache_key] = regime_data
        return regime_data
```

**Caching Implementation:**
```python
def _is_cache_valid(self, cache_key: str) -> bool:
    """Check if cached data is still valid"""
    if cache_key not in self.cache:
        return False
        
    cache_time = self.cache.get(f"{cache_key}_timestamp")
    if not cache_time:
        return False
        
    return datetime.now() - cache_time < self.cache_duration
```

### 3. TradingView Chart Integration (`streamlit/pages/tvchart.py`)

**Core Responsibilities:**
- Sidebar controls for intelligence features
- Session and regime visualization above chart
- Comprehensive trade context analysis with tabbed interface
- Proper handling of simplified data structures

**User Interface Integration:**
```python
# Sidebar controls for intelligence features
show_market_regime = st.sidebar.checkbox("Show Market Regime", value=False)
show_session_analysis = st.sidebar.checkbox("Show Session Info", value=True)
show_trade_context = st.sidebar.checkbox("Analyze Trade Context", value=False)

# Market regime display
if show_market_regime:
    regime_data = intelligence_service.get_regime_for_timeframe(epic, '15m')
    if regime_data:
        confidence = regime_data.get('confidence', 0)
        regime_type = regime_data.get('regime', 'unknown')
        description = regime_data.get('description', 'No description')
        
        # Format display with appropriate emoji
        regime_emoji = {'trending_up': '📈', 'trending_down': '📉', 
                       'ranging': '📊', 'breakout': '🚀'}.get(regime_type, '❓')
        
        st.info(f"{regime_emoji} {regime_type.upper()} | {confidence:.0%} confidence | {description}")
```

## Market Analysis Features

### Market Regime Analysis

**Detection Methods:**
- EMA-based trend analysis using 21/50/200 moving averages
- Price action pattern recognition
- Volatility-adjusted regime classification
- Multi-timeframe confirmation

**Regime Types:**
```python
REGIME_TYPES = {
    'trending_up': {
        'description': 'Strong upward momentum',
        'criteria': 'Price > EMA21 > EMA50 > EMA200',
        'confidence_weight': 0.8
    },
    'trending_down': {
        'description': 'Strong downward momentum', 
        'criteria': 'Price < EMA21 < EMA50 < EMA200',
        'confidence_weight': 0.8
    },
    'ranging': {
        'description': 'Sideways consolidation',
        'criteria': 'EMAs converged, low volatility',
        'confidence_weight': 0.6
    },
    'breakout': {
        'description': 'Volatility expansion phase',
        'criteria': 'High volatility, EMA divergence',
        'confidence_weight': 0.7
    }
}
```

**Confidence Scoring Algorithm:**
```python
def calculate_regime_confidence(self, df: pd.DataFrame) -> float:
    """Calculate confidence score for regime classification"""
    if len(df) < 50:
        return 0.5  # Insufficient data
        
    latest = df.iloc[-1]
    
    # EMA alignment score (0.0 to 1.0)
    ema_alignment = self._calculate_ema_alignment(latest)
    
    # Trend consistency score (0.0 to 1.0)
    trend_consistency = self._calculate_trend_consistency(df.tail(20))
    
    # Volatility confirmation score (0.0 to 1.0)
    volatility_confirmation = self._calculate_volatility_confirmation(df)
    
    # Weighted average
    confidence = (
        ema_alignment * 0.4 + 
        trend_consistency * 0.4 + 
        volatility_confirmation * 0.2
    )
    
    return max(0.6, min(0.95, confidence))  # Clamp between 60-95%
```

### Trading Session Analysis

**Global Trading Sessions:**
- **Asian Session**: 22:00-08:00 UTC (Tokyo, Sydney)
- **London Session**: 08:00-16:00 UTC (London, Frankfurt)
- **New York Session**: 13:00-22:00 UTC (New York, Chicago)
- **Overlap Periods**: London/New York (13:00-16:00 UTC)

**Session Characteristics:**
```python
SESSION_CHARACTERISTICS = {
    'asian': {
        'volatility': 'low',
        'characteristics': 'Range-bound trading, lower volume',
        'active_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD'],
        'peak_hours': '00:00-04:00 UTC'
    },
    'london': {
        'volatility': 'high', 
        'characteristics': 'Trend establishment, high volume',
        'active_pairs': ['EURUSD', 'GBPUSD', 'EURGBP'],
        'peak_hours': '08:00-11:00 UTC'
    },
    'new_york': {
        'volatility': 'high',
        'characteristics': 'Trend continuation, news impact',
        'active_pairs': ['EURUSD', 'GBPUSD', 'USDCAD'],
        'peak_hours': '13:00-17:00 UTC'
    },
    'overlap': {
        'volatility': 'peak',
        'characteristics': 'Maximum liquidity and volatility',
        'active_pairs': ['EURUSD', 'GBPUSD', 'USDCHF'],
        'peak_hours': '13:00-16:00 UTC'
    }
}
```

**Volatility Mapping Implementation:**
```python
def get_session_analysis(self) -> Dict:
    """Analyze current trading session and characteristics"""
    current_utc = datetime.now(timezone.utc)
    hour = current_utc.hour
    
    # Determine active session
    if 22 <= hour or hour < 8:
        session = 'asian'
        session_name = 'ASIAN SESSION'
        emoji = '🌏'
    elif 8 <= hour < 13:
        session = 'london'
        session_name = 'LONDON SESSION' 
        emoji = '🌍'
    elif 13 <= hour < 16:
        session = 'overlap'
        session_name = 'LONDON/NY OVERLAP'
        emoji = '🟢'
    elif 16 <= hour < 22:
        session = 'new_york'
        session_name = 'NEW YORK SESSION'
        emoji = '🌎'
    else:
        session = 'off_hours'
        session_name = 'OFF HOURS'
        emoji = '🌙'
        
    session_info = SESSION_CHARACTERISTICS.get(session, {})
    
    return {
        'session': session,
        'session_name': session_name,
        'emoji': emoji,
        'volatility_level': session_info.get('volatility', 'minimal'),
        'characteristics': session_info.get('characteristics', 'Low activity period'),
        'active_pairs': session_info.get('active_pairs', []),
        'current_time_utc': current_utc.strftime('%H:%M UTC')
    }
```

### Trade Context Analysis

**Quality Scoring System:**
- **Combined Regime (60%) and Session (40%) alignment**
- **Context-aware scoring based on trade direction and market conditions**
- **Success factor identification and improvement suggestions**

**Regime Alignment Scoring:**
```python
def calculate_regime_alignment_score(self, trade_direction: str, regime_data: Dict) -> float:
    """Calculate how well trade aligns with market regime"""
    regime_type = regime_data.get('regime', 'ranging')
    confidence = regime_data.get('confidence', 0.5)
    
    # Base alignment scores
    alignment_scores = {
        'trending_up': {'BUY': 0.8, 'SELL': 0.2},
        'trending_down': {'BUY': 0.2, 'SELL': 0.8}, 
        'ranging': {'BUY': 0.4, 'SELL': 0.4},
        'breakout': {'BUY': 0.7, 'SELL': 0.7}
    }
    
    base_score = alignment_scores.get(regime_type, {}).get(trade_direction, 0.4)
    
    # Adjust by confidence level
    confidence_adjusted = base_score * confidence + 0.4 * (1 - confidence)
    
    return {
        'score': confidence_adjusted,
        'regime': regime_type,
        'confidence': confidence,
        'alignment': 'Good' if confidence_adjusted > 0.6 else 'Poor'
    }
```

**Session Scoring Implementation:**
```python
def calculate_session_score(self, session_data: Dict) -> float:
    """Calculate session quality score for trading"""
    volatility = session_data.get('volatility_level', 'minimal')
    
    session_scores = {
        'peak': 0.9,      # London/NY overlap
        'high': 0.7,      # London or NY single session
        'medium': 0.5,    # Transition periods
        'low': 0.4,       # Asian session
        'minimal': 0.2    # Off hours
    }
    
    return session_scores.get(volatility, 0.4)
```

**Trade Quality Assessment:**
```python
def analyze_trade_context(self, trade_data: Dict, epic: str) -> Dict:
    """Comprehensive trade context analysis"""
    
    # Get market regime and session data
    regime_data = self.get_regime_for_timeframe(epic, '15m')
    session_data = self.get_session_analysis()
    
    # Calculate component scores
    regime_alignment = self.calculate_regime_alignment_score(
        trade_data.get('direction', 'BUY'), regime_data
    )
    session_score = self.calculate_session_score(session_data)
    
    # Overall quality score (weighted average)
    overall_score = (regime_alignment['score'] * 0.6) + (session_score * 0.4)
    
    # Generate success factors
    success_factors = self._generate_success_factors(
        regime_data, session_data, trade_data, overall_score
    )
    
    # Generate improvement suggestions
    improvements = self._generate_improvement_suggestions(
        regime_alignment, session_score, trade_data
    )
    
    return {
        'overall_score': overall_score,
        'quality_rating': self._get_quality_rating(overall_score),
        'regime_alignment': regime_alignment,
        'session_score': session_score,
        'session_info': session_data,
        'success_factors': success_factors,
        'improvement_suggestions': improvements,
        'analysis_timestamp': datetime.now().isoformat()
    }
```

## Database Schema Requirements

The system expects the following ig_candles table structure:

```sql
CREATE TABLE ig_candles (
    start_time TIMESTAMP,
    epic VARCHAR(50),
    timeframe INTEGER,           -- 5, 15, 60, 240, 1440 (minutes)
    open_price_mid DECIMAL(10,5),
    high_price_mid DECIMAL(10,5),
    low_price_mid DECIMAL(10,5),
    close_price_mid DECIMAL(10,5),
    volume INTEGER,
    ltv DECIMAL(15,2),
    
    PRIMARY KEY (start_time, epic, timeframe)
);

-- Optimized indexes for intelligence queries
CREATE INDEX idx_ig_candles_epic_timeframe_time ON ig_candles (epic, timeframe, start_time);
CREATE INDEX idx_ig_candles_time_desc ON ig_candles (start_time DESC);
```

## Usage Examples

### Basic Intelligence Integration

```python
from services.market_intelligence_service import get_intelligence_service

# Initialize service
intelligence_service = get_intelligence_service()
intelligence_service.data_fetcher.engine = engine  # Pass DB engine

# Get market analyses
regime_data = intelligence_service.get_regime_for_timeframe('CS.D.EURUSD.MINI.IP', '15m')
session_data = intelligence_service.get_session_analysis()

# Analyze trade context
trade_data = {
    'direction': 'BUY',
    'entry_price': 1.0950,
    'stop_loss': 1.0920,
    'take_profit': 1.0980,
    'confidence': 0.75
}

trade_context = intelligence_service.analyze_trade_context(trade_data, 'CS.D.EURUSD.MINI.IP')
print(f"Trade Quality: {trade_context['quality_rating']} ({trade_context['overall_score']:.1%})")
```

### Streamlit Dashboard Integration

```python
# In Streamlit app
if show_session_analysis:
    session_data = intelligence_service.get_session_analysis()
    session_display = (
        f"{session_data['emoji']} {session_data['session_name']} | "
        f"{session_data['volatility_level'].title()} Volatility | "
        f"{session_data['characteristics']}"
    )
    st.info(session_display)

if show_trade_context and selected_trades:
    st.subheader("🎯 Trade Context Analysis")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Individual Trades", "Regime Performance", "Trade Summary"])
    
    with tab1:
        for trade in selected_trades:
            context = intelligence_service.analyze_trade_context(trade, epic)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**Trade #{trade['id']}** - {trade['direction']}")
                st.write(f"Quality: {context['quality_rating']}")
            with col2:
                st.metric("Overall Score", f"{context['overall_score']:.1%}")
            with col3:
                st.metric("Regime Alignment", context['regime_alignment']['alignment'])
```

## Simplified Analysis Mode

When the forex_scanner module is not available (Streamlit container), the system automatically uses simplified analysis:

### Fallback Implementations

```python
def _get_simplified_regime_analysis(self, epic: str, timeframe: str) -> Dict:
    """Simplified regime analysis without forex_scanner dependencies"""
    try:
        df = self.data_fetcher.get_enhanced_data(epic, timeframe, lookback_hours=48)
        
        if df.empty or len(df) < 21:
            return self._get_default_regime()
            
        latest = df.iloc[-1]
        
        # Simple EMA-based regime detection
        if pd.notna(latest['ema_21']) and pd.notna(latest['ema_50']) and pd.notna(latest['ema_200']):
            price = latest['close']
            ema21 = latest['ema_21']
            ema50 = latest['ema_50'] 
            ema200 = latest['ema_200']
            
            # Determine regime based on EMA alignment
            if price > ema21 > ema50 > ema200:
                regime = 'trending_up'
                confidence = 0.7
                description = 'Strong upward momentum'
            elif price < ema21 < ema50 < ema200:
                regime = 'trending_down'
                confidence = 0.7 
                description = 'Strong downward momentum'
            elif abs(ema21 - ema50) / price < 0.001:  # EMAs close together
                regime = 'ranging'
                confidence = 0.6
                description = 'Sideways consolidation'
            else:
                regime = 'breakout'
                confidence = 0.65
                description = 'Mixed signals, potential breakout'
        else:
            return self._get_default_regime()
            
        return {
            'regime': regime,
            'confidence': confidence,
            'description': description,
            'mode': 'simplified',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        self.logger.error(f"Simplified analysis failed: {e}")
        return self._get_default_regime()
```

## Performance Considerations

### Optimization Strategies

- **5-minute cache duration** for all analyses
- **Limit to 5 pairs** for batch regime analysis  
- **48-hour default lookback** for data fetching
- **Fallback to 5m data** when 15m unavailable
- **Lazy loading** of technical indicators
- **Connection pooling** for database queries

### Memory Management

```python
class PerformanceOptimizer:
    def __init__(self):
        self.max_cache_size = 100
        self.cache_cleanup_interval = 300  # 5 minutes
        
    def cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry['timestamp'] > self.cache_duration:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.cache[key]
            
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        # Convert to optimal data types
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        # Remove unnecessary columns
        essential_cols = ['start_time', 'open', 'high', 'low', 'close', 'volume']
        df = df[essential_cols + [col for col in df.columns if col.startswith('ema_')]]
        
        return df
```

## Common Issues and Solutions

### 1. "Unknown Volatility" in Session Display

**Issue**: Session display shows "Unknown Volatility" instead of expected levels
**Solution**: Check session_characteristics field instead of session_config

```python
# ❌ WRONG
volatility = session_data.get('session_config', {}).get('volatility')

# ✅ CORRECT  
volatility = session_data.get('volatility_level', 'minimal')
```

### 2. SQL Column Errors

**Issue**: Column names differ from expected (open vs open_price_mid)
**Solution**: Use correct column names in data_fetcher_adapter.py

```python
# ✅ CORRECT column mapping
query = """
    SELECT start_time, epic, 
           open_price_mid as open, high_price_mid as high,
           low_price_mid as low, close_price_mid as close,
           volume, ltv
    FROM ig_candles 
    WHERE epic = %s AND timeframe = %s
"""
```

### 3. Timeframe Type Errors

**Issue**: Database stores timeframes as integers (5, 60) not strings ('5m', '1h')
**Solution**: Map strings to integers in data_fetcher_adapter.py

```python
# ✅ CORRECT timeframe mapping
timeframe_map = {
    '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
}
tf_minutes = timeframe_map.get(timeframe, 15)
```

### 4. Missing Fields in Trade Context

**Issue**: UI expects specific fields (entry_price, regime_alignment.score, etc.)
**Solution**: Ensure all fields provided in _get_simplified_trade_context()

```python
# ✅ COMPLETE field structure
return {
    'overall_score': float,
    'quality_rating': str,
    'regime_alignment': {
        'score': float,
        'regime': str,
        'confidence': float,
        'alignment': str
    },
    'session_score': float,
    'session_info': dict,
    'success_factors': list,
    'improvement_suggestions': list,
    'analysis_timestamp': str
}
```

## Testing and Validation

### Development Testing

```bash
# Restart Streamlit to load changes
docker-compose restart streamlit

# Check logs for errors
docker-compose logs --tail=30 streamlit | grep -E "(Error|error|❌|⚠️)"

# Clear Python cache if needed
docker-compose exec -T streamlit find /app -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Test database connectivity
docker exec streamlit python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    result = conn.execute('SELECT COUNT(*) FROM ig_candles')
    print(f'✅ Database OK: {result.fetchone()[0]} candles')
"
```

### Performance Testing

```python
def test_intelligence_performance():
    """Test market intelligence system performance"""
    import time
    
    # Test regime analysis speed
    start_time = time.time()
    regime_data = intelligence_service.get_regime_for_timeframe('CS.D.EURUSD.MINI.IP', '15m')
    regime_time = time.time() - start_time
    
    # Test session analysis speed  
    start_time = time.time()
    session_data = intelligence_service.get_session_analysis()
    session_time = time.time() - start_time
    
    # Test trade context analysis speed
    start_time = time.time()
    trade_context = intelligence_service.analyze_trade_context(sample_trade, 'CS.D.EURUSD.MINI.IP')
    context_time = time.time() - start_time
    
    print(f"Performance Results:")
    print(f"  Regime Analysis: {regime_time:.3f}s")
    print(f"  Session Analysis: {session_time:.3f}s") 
    print(f"  Trade Context: {context_time:.3f}s")
    
    # Performance assertions
    assert regime_time < 1.0, f"Regime analysis too slow: {regime_time:.3f}s"
    assert session_time < 0.1, f"Session analysis too slow: {session_time:.3f}s"
    assert context_time < 2.0, f"Trade context too slow: {context_time:.3f}s"
```

For architectural context, see [System Architecture](claude-architecture.md).
For command usage, see [Commands & CLI](claude-commands.md).
For development patterns, see [Development Best Practices](claude-development.md).