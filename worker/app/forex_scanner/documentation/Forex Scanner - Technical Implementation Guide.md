# Forex Scanner - Technical Implementation Guide
## 🔧 Developer Reference & API Documentation

---

## 📋 Table of Contents

1. [Core API Reference](#core-api-reference)
2. [Database Schema Details](#database-schema-details)
3. [Strategy Development Guide](#strategy-development-guide)
4. [Integration Patterns](#integration-patterns)
5. [Testing Framework](#testing-framework)
6. [Deployment Guide](#deployment-guide)
7. [Performance Tuning](#performance-tuning)
8. [Security Considerations](#security-considerations)

---

## 🔌 Core API Reference

### SignalDetector Class

```python
from core.signal_detector import SignalDetector

class SignalDetector:
    def __init__(self, db_manager, user_timezone='Europe/Stockholm'):
        """Initialize signal detector with database connection and timezone"""
        
    def detect_signals(self, epic_list: List[str], timeframe: str = '15m') -> List[Dict]:
        """
        Detect signals across multiple currency pairs
        
        Args:
            epic_list: List of currency pair EPICs
            timeframe: Trading timeframe ('15m', '1h', '4h', '1d')
            
        Returns:
            List of signal dictionaries
        """
        
    def get_crossover_state(self, epic: str, strategy: str) -> Dict:
        """Get current crossover state for epic/strategy combination"""
        
    def save_crossover_state(self, epic: str, strategy: str, state: Dict):
        """Save crossover state to prevent duplicate signals"""
```

### Strategy Base Class

```python
from core.strategies.base_strategy import BaseStrategy

class CustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__('custom_strategy')
    
    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> Optional[Dict]:
        """
        Implement signal detection logic
        
        Args:
            df: OHLCV data with technical indicators
            epic: Currency pair EPIC
            spread_pips: Current spread in pips
            timeframe: Trading timeframe
            
        Returns:
            Signal dictionary or None
        """
        pass
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['ema_9', 'ema_21', 'ema_200']
    
    def calculate_confidence(self, signal_data: Dict) -> float:
        """Calculate confidence score (0.0 to 1.0)"""
        return 0.8
```

### DatabaseManager Class

```python
from core.database import DatabaseManager

class DatabaseManager:
    def __init__(self, database_url: str):
        """Initialize with PostgreSQL connection string"""
        
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters for safe execution
            
        Returns:
            pandas DataFrame with results
        """
        
    def get_connection(self):
        """Get raw database connection for advanced operations"""
        
    def health_check(self) -> bool:
        """Check database connectivity and performance"""
```

### ClaudeAnalyzer Class

```python
from alerts.claude_api import ClaudeAnalyzer

class ClaudeAnalyzer:
    def __init__(self, api_key: str):
        """Initialize with Claude API key"""
        
    def analyze_signal(self, signal: Dict) -> Dict:
        """
        Analyze signal using Claude AI
        
        Args:
            signal: Signal dictionary with OHLCV and indicator data
            
        Returns:
            Analysis result with recommendation and confidence
        """
        
    def test_connection(self) -> bool:
        """Test Claude API connectivity"""
```

---

## 🗄️ Database Schema Details

### Table Relationships

```mermaid
erDiagram
    alert_history {
        int id PK
        timestamp alert_timestamp
        string epic
        string strategy
        decimal confidence_score
        json strategy_config
        text claude_analysis
    }
    
    strategy_summary {
        int id PK
        date date_tracked
        string epic
        string strategy
        int total_alerts
        decimal avg_confidence
    }
    
    strategy_configs {
        int id PK
        string strategy
        string config_hash
        json config_data
    }
    
    ig_candles {
        int id PK
        string epic
        timestamp start_time
        decimal open
        decimal high
        decimal low
        decimal close
        decimal volume
    }
    
    alert_history ||--o{ strategy_summary : "aggregated_in"
    strategy_configs ||--o{ alert_history : "configured_by"
    ig_candles ||--o{ alert_history : "based_on"
```

### Key Indexes

```sql
-- Performance indexes
CREATE INDEX idx_alert_history_epic_timestamp ON alert_history(epic, alert_timestamp);
CREATE INDEX idx_alert_history_strategy ON alert_history(strategy);
CREATE INDEX idx_alert_history_signal_hash ON alert_history(signal_hash);

-- Market data indexes
CREATE INDEX idx_ig_candles_epic_time ON ig_candles(epic, start_time);
CREATE INDEX idx_ig_candles_timeframe ON ig_candles(timeframe);

-- Analysis indexes
CREATE INDEX idx_strategy_summary_date_epic ON strategy_summary(date_tracked, epic);
```

### Data Types and Constraints

```sql
-- Confidence scores: 0.0000 to 1.0000
DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1)

-- Price data: 5 decimal places for forex precision
DECIMAL(10,5) 

-- JSON fields for flexible data storage
JSON NOT NULL DEFAULT '{}'

-- Enum-like constraints
CHECK (signal_type IN ('BUY', 'SELL', 'BULL', 'BEAR'))
CHECK (timeframe IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d'))
```

---

## 📈 Strategy Development Guide

### Creating a New Strategy

1. **Inherit from BaseStrategy**
```python
from core.strategies.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__('my_custom_strategy')
        self.required_indicators = ['rsi', 'bb_upper', 'bb_lower']
```

2. **Implement Detection Logic**
```python
def detect_signal(self, df, epic, spread_pips, timeframe):
    if len(df) < 50:  # Minimum data requirement
        return None
    
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Example: RSI oversold with Bollinger Band touch
    if (current['rsi'] < 30 and 
        current['close'] <= current['bb_lower'] and
        previous['close'] > previous['bb_lower']):
        
        return {
            'epic': epic,
            'signal_type': 'BUY',
            'strategy': self.name,
            'confidence_score': self.calculate_confidence({
                'rsi': current['rsi'],
                'bb_distance': abs(current['close'] - current['bb_lower'])
            }),
            'entry_price': current['close'],
            'timeframe': timeframe,
            'strategy_config': {
                'rsi_threshold': 30,
                'bb_period': 20
            }
        }
    
    return None
```

3. **Register Strategy**
```python
# In core/signal_detector.py
def _initialize_strategies(self):
    if getattr(config, 'MY_CUSTOM_STRATEGY', False):
        self.my_custom_strategy = MyCustomStrategy()
```

### Strategy Testing Framework

```python
# Test your strategy
def test_my_custom_strategy():
    strategy = MyCustomStrategy()
    
    # Create test data
    test_df = pd.DataFrame({
        'close': [1.1000, 1.0950, 1.0900],
        'rsi': [35, 28, 25],
        'bb_lower': [1.0905, 1.0905, 1.0905]
    })
    
    signal = strategy.detect_signal(test_df, 'CS.D.EURUSD.MINI.IP', 1.5, '15m')
    
    assert signal is not None
    assert signal['signal_type'] == 'BUY'
    assert signal['confidence_score'] > 0.5
```

---

## 🔄 Integration Patterns

### Event-Driven Architecture

```python
class SignalEventHandler:
    def __init__(self):
        self.subscribers = []
    
    def subscribe(self, callback):
        self.subscribers.append(callback)
    
    def emit_signal(self, signal):
        for callback in self.subscribers:
            try:
                callback(signal)
            except Exception as e:
                logging.error(f"Signal handler error: {e}")

# Usage
signal_handler = SignalEventHandler()
signal_handler.subscribe(save_to_database)
signal_handler.subscribe(send_notification)
signal_handler.subscribe(claude_analysis)
```

### Plugin Architecture

```python
# plugins/custom_indicator.py
class CustomIndicator:
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate custom technical indicator"""
        return df['close'].rolling(20).mean()
    
    def get_name(self) -> str:
        return 'custom_ma_20'

# Register plugin
from plugins.custom_indicator import CustomIndicator
indicator_registry['custom_ma_20'] = CustomIndicator()
```

### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncSignalProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_signals_async(self, epic_list):
        """Process multiple EPICs concurrently"""
        tasks = []
        for epic in epic_list:
            task = asyncio.create_task(self.process_epic_async(epic))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    async def process_epic_async(self, epic):
        """Process single EPIC asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.signal_detector.detect_signals, 
            [epic]
        )
```

---

## 🧪 Testing Framework

### Unit Testing Structure

```python
# tests/test_strategies.py
import unittest
import pandas as pd
from core.strategies.ema_strategy import EMAStrategy

class TestEMAStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = EMAStrategy()
        self.test_data = self._create_test_data()
    
    def _create_test_data(self):
        """Create realistic test data with EMA crossover"""
        dates = pd.date_range('2025-01-01', periods=100, freq='15min')
        
        # Simulate price movement with EMA crossover
        prices = [1.1000]
        for i in range(1, 100):
            change = 0.0001 if i > 50 else -0.0001  # Trend change at midpoint
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 0.0005 for p in prices],
            'low': [p - 0.0005 for p in prices],
            'close': prices,
            'volume': [1000] * 100
        })
        
        # Add EMA indicators
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        return df
    
    def test_bullish_crossover_detection(self):
        """Test detection of bullish EMA crossover"""
        # Find crossover point
        crossover_idx = None
        for i in range(1, len(self.test_data)):
            if (self.test_data.iloc[i]['ema_9'] > self.test_data.iloc[i]['ema_21'] and
                self.test_data.iloc[i-1]['ema_9'] <= self.test_data.iloc[i-1]['ema_21']):
                crossover_idx = i
                break
        
        self.assertIsNotNone(crossover_idx, "No crossover found in test data")
        
        # Test signal detection at crossover
        test_df = self.test_data.iloc[:crossover_idx+1]
        signal = self.strategy.detect_signal(test_df, 'CS.D.EURUSD.MINI.IP', 1.5, '15m')
        
        if signal:  # Crossover detected
            self.assertEqual(signal['signal_type'], 'BULL')
            self.assertGreater(signal['confidence_score'], 0.5)
    
    def test_insufficient_data_handling(self):
        """Test behavior with insufficient data"""
        small_df = self.test_data.head(10)
        signal = self.strategy.detect_signal(small_df, 'CS.D.EURUSD.MINI.IP', 1.5, '15m')
        self.assertIsNone(signal, "Should return None for insufficient data")
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        signal_data = {
            'ema_separation': 0.0010,
            'trend_strength': 0.8,
            'volume_confirmation': True
        }
        
        confidence = self.strategy.calculate_confidence(signal_data)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

# tests/test_database.py
class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager(config.TEST_DATABASE_URL)
    
    def test_connection(self):
        """Test database connectivity"""
        self.assertTrue(self.db.health_check())
    
    def test_signal_storage(self):
        """Test signal storage and retrieval"""
        test_signal = {
            'epic': 'TEST.EPIC',
            'signal_type': 'BUY',
            'strategy': 'test_strategy',
            'confidence_score': 0.75,
            'price': 1.1000
        }
        
        # Store signal
        from alerts.alert_history import AlertHistoryManager
        alert_mgr = AlertHistoryManager(self.db)
        alert_id = alert_mgr.save_alert(test_signal, "Test message")
        
        self.assertIsNotNone(alert_id)
        
        # Retrieve signal
        stored_signal = alert_mgr.get_alert_by_id(alert_id)
        self.assertEqual(stored_signal['epic'], 'TEST.EPIC')
        self.assertEqual(stored_signal['signal_type'], 'BUY')
```

### Integration Testing

```python
# tests/test_integration.py
class TestSignalPipeline(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager(config.TEST_DATABASE_URL)
        self.signal_detector = SignalDetector(self.db)
        self.claude_analyzer = ClaudeAnalyzer(config.CLAUDE_API_KEY)
    
    def test_end_to_end_signal_processing(self):
        """Test complete signal processing pipeline"""
        # 1. Detect signals
        signals = self.signal_detector.detect_signals(['CS.D.EURUSD.MINI.IP'])
        
        if signals:
            # 2. Process with Claude
            for signal in signals:
                claude_result = self.claude_analyzer.analyze_signal(signal)
                signal.update(claude_result)
            
            # 3. Store in database
            alert_mgr = AlertHistoryManager(self.db)
            for signal in signals:
                alert_id = alert_mgr.save_alert(signal, "Integration test")
                self.assertIsNotNone(alert_id)
            
            # 4. Verify storage
            recent_alerts = alert_mgr.get_recent_alerts(limit=len(signals))
            self.assertEqual(len(recent_alerts), len(signals))

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Performance Testing

```python
# tests/test_performance.py
import time
import cProfile
import pstats

class TestPerformance(unittest.TestCase):
    def test_signal_detection_speed(self):
        """Test signal detection performance"""
        start_time = time.time()
        
        signals = self.signal_detector.detect_signals(config.EPIC_LIST)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process signals in under 10 seconds
        self.assertLess(processing_time, 10.0)
        print(f"Signal detection took {processing_time:.2f} seconds")
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process signals
        for _ in range(10):
            signals = self.signal_detector.detect_signals(['CS.D.EURUSD.MINI.IP'])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        self.assertLess(memory_increase, 100)
        print(f"Memory increase: {memory_increase:.2f} MB")
    
    def profile_signal_detection(self):
        """Profile signal detection for optimization"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run signal detection
        signals = self.signal_detector.detect_signals(config.EPIC_LIST)
        
        profiler.disable()
        
        # Analyze results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
```

---

## 🚀 Deployment Guide

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/forex_scanner
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from core.database import DatabaseManager; import config; DatabaseManager(config.DATABASE_URL).health_check()" || exit 1

# Default command
CMD ["python", "trade_scan.py", "docker"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  forex-scanner:
    build: .
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/forex
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - LOG_LEVEL=INFO
    depends_on:
      - postgres
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=forex
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forex-scanner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: forex-scanner
  template:
    metadata:
      labels:
        app: forex-scanner
    spec:
      containers:
      - name: forex-scanner
        image: forex-scanner:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: forex-secrets
              key: database-url
        - name: CLAUDE_API_KEY
          valueFrom:
            secretKeyRef:
              name: forex-secrets
              key: claude-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from core.database import DatabaseManager; import config; DatabaseManager(config.DATABASE_URL).health_check()"
          initialDelaySeconds: 30
          periodSeconds: 60
```

### Environment Configuration

```bash
# .env.production
DATABASE_URL=postgresql://user:pass@host:5432/forex
CLAUDE_API_KEY=sk-ant-api03-...
LOG_LEVEL=INFO
ENABLE_NOTIFICATIONS=true
SCAN_INTERVAL=60
MIN_CONFIDENCE=0.7

# Performance settings
MAX_WORKERS=4
CONNECTION_POOL_SIZE=10
QUERY_TIMEOUT=30

# Feature flags
ENABLE_CLAUDE_ANALYSIS=true
ENABLE_BACKTESTING=true
ENABLE_LIVE_TRADING=false
```

---

## ⚡ Performance Tuning

### Database Optimization

```sql
-- Optimize alert_history table
VACUUM ANALYZE alert_history;

-- Update table statistics
ANALYZE alert_history;

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_alert_history_strategy_timestamp 
ON alert_history(strategy, alert_timestamp DESC);

CREATE INDEX CONCURRENTLY idx_alert_history_epic_confidence 
ON alert_history(epic, confidence_score DESC);

-- Partition large tables by date
CREATE TABLE alert_history_2025_07 PARTITION OF alert_history
FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
```

### Connection Pooling

```python
# core/database.py - Enhanced connection pooling
from psycopg2 import pool

class DatabaseManager:
    def __init__(self, database_url, min_conn=1, max_conn=10):
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn, database_url
        )
    
    def get_connection(self):
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        self.connection_pool.putconn(conn)
    
    def execute_query_pooled(self, query, params=None):
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(result, columns=columns)
            else:
                conn.commit()
                return cursor.rowcount
                
        finally:
            if conn:
                cursor.close()
                self.return_connection(conn)
```

### Caching Strategy

```python
# utils/cache.py
import redis
import json
from functools import wraps
from datetime import timedelta

class CacheManager:
    def __init__(self, redis_url='redis://localhost:6379'):
        self.redis_client = redis.from_url(redis_url)
    
    def cache_result(self, expire_seconds=300):
        """Decorator to cache function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.redis_client.setex(
                    cache_key, 
                    expire_seconds, 
                    json.dumps(result, default=str)
                )
                
                return result
            return wrapper
        return decorator

# Usage
cache = CacheManager()

@cache.cache_result(expire_seconds=60)
def get_market_data(epic, timeframe):
    """Cached market data retrieval"""
    return fetch_market_data(epic, timeframe)
```

### Async Processing

```python
# core/async_processor.py
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncSignalProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process_epics_parallel(self, epic_list):
        """Process multiple EPICs in parallel"""
        tasks = []
        
        for epic in epic_list:
            task = asyncio.create_task(self.process_epic(epic))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Epic processing failed: {result}")
            else:
                successful_results.extend(result)
        
        return successful_results
    
    async def process_epic(self, epic):
        """Process single EPIC asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive signal detection in thread pool
        signals = await loop.run_in_executor(
            self.executor,
            self.detect_signals_sync,
            epic
        )
        
        # Process Claude analysis concurrently if signals found
        if signals:
            claude_tasks = []
            for signal in signals:
                task = self.analyze_with_claude_async(signal)
                claude_tasks.append(task)
            
            claude_results = await asyncio.gather(*claude_tasks)
            
            # Merge Claude results with signals
            for signal, claude_result in zip(signals, claude_results):
                signal.update(claude_result)
        
        return signals
    
    async def analyze_with_claude_async(self, signal):
        """Async Claude analysis"""
        # Use aiohttp for async HTTP requests to Claude API
        async with self.session.post(
            'https://api.anthropic.com/v1/messages',
            headers={'Authorization': f'Bearer {config.CLAUDE_API_KEY}'},
            json=self.prepare_claude_request(signal)
        ) as response:
            result = await response.json()
            return self.parse_claude_response(result)
```

---

## 🔒 Security Considerations

### API Key Management

```python
# config/security.py
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.cipher_key = os.environ.get('ENCRYPTION_KEY')
        if self.cipher_key:
            self.cipher = Fernet(self.cipher_key.encode())
    
    def encrypt_api_key(self, api_key):
        """Encrypt API key for storage"""
        if self.cipher:
            return self.cipher.encrypt(api_key.encode()).decode()
        return api_key
    
    def decrypt_api_key(self, encrypted_key):
        """Decrypt API key for use"""
        if self.cipher:
            return self.cipher.decrypt(encrypted_key.encode()).decode()
        return encrypted_key
    
    @property
    def claude_api_key(self):
        """Securely retrieve Claude API key"""
        encrypted_key = os.environ.get('CLAUDE_API_KEY_ENCRYPTED')
        if encrypted_key:
            return self.decrypt_api_key(encrypted_key)
        return os.environ.get('CLAUDE_API_KEY')

# Usage
secure_config = SecureConfig()
claude_key = secure_config.claude_api_key
```

### Database Security

```python
# core/secure_database.py
import logging
from psycopg2.extras import RealDictCursor

class SecureDatabaseManager(DatabaseManager):
    def __init__(self, database_url):
        super().__init__(database_url)
        self.setup_security_logging()
    
    def setup_security_logging(self):
        """Setup security event logging"""
        self.security_logger = logging.getLogger('security')
        handler = logging.FileHandler('/var/log/forex_scanner_security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
    
    def execute_query(self, query, params=None):
        """Execute query with security logging"""
        # Log potentially dangerous operations
        if any(keyword in query.upper() for keyword in ['DROP', 'DELETE', 'TRUNCATE']):
            self.security_logger.warning(
                f"Dangerous SQL operation: {query[:100]}..."
            )
        
        # Validate parameters to prevent SQL injection
        if params and not isinstance(params, (tuple, list)):
            raise ValueError("Parameters must be tuple or list")
        
        return super().execute_query(query, params)
    
    def sanitize_input(self, input_value):
        """Sanitize user input"""
        if isinstance(input_value, str):
            # Remove potentially dangerous characters
            dangerous_chars = [';', '--', '/*', '*/', 'xp_']
            for char in dangerous_chars:
                input_value = input_value.replace(char, '')
        
        return input_value
```

### Rate Limiting

```python
# utils/rate_limiter.py
import time
from collections import defaultdict, deque

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
    
    def is_allowed(self, key, max_requests=60, window_seconds=60):
        """Check if request is within rate limits"""
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old requests
        request_times = self.requests[key]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check if under limit
        if len(request_times) < max_requests:
            request_times.append(now)
            return True
        
        return False
    
    def get_reset_time(self, key, window_seconds=60):
        """Get time until rate limit resets"""
        request_times = self.requests[key]
        if not request_times:
            return 0
        
        return max(0, request_times[0] + window_seconds - time.time())

# Usage in Claude API calls
rate_limiter = RateLimiter()

def claude_analyze_with_rate_limit(signal):
    if not rate_limiter.is_allowed('claude_api', max_requests=50, window_seconds=60):
        wait_time = rate_limiter.get_reset_time('claude_api')
        raise Exception(f"Rate limit exceeded. Wait {wait_time:.1f} seconds")
    
    return claude_analyzer.analyze_signal(signal)
```

---

## 📊 Monitoring & Alerting

### Health Check Endpoints

```python
# monitoring/health.py
from flask import Flask, jsonify
import psutil
import time

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

@app.route('/health/detailed')
def detailed_health_check():
    """Detailed system health check"""
    try:
        # Database connectivity
        db_healthy = DatabaseManager(config.DATABASE_URL).health_check()
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Recent signal activity
        recent_signals = get_recent_signal_count()
        
        return jsonify({
            'status': 'healthy' if db_healthy else 'unhealthy',
            'database': {
                'connected': db_healthy,
                'response_time_ms': measure_db_response_time()
            },
            'system': {
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'cpu_usage_percent': psutil.cpu_percent()
            },
            'application': {
                'recent_signals_1h': recent_signals,
                'uptime_seconds': time.time() - start_time
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Metrics Collection

```python
# monitoring/metrics.py
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
SIGNALS_DETECTED = Counter('forex_signals_detected_total', 'Total signals detected', ['epic', 'strategy'])
SIGNAL_PROCESSING_TIME = Histogram('forex_signal_processing_seconds', 'Time spent processing signals')
CLAUDE_API_CALLS = Counter('claude_api_calls_total', 'Total Claude API calls', ['status'])
DATABASE_QUERY_TIME = Histogram('database_query_seconds', 'Database query execution time')
ACTIVE_EPICS = Gauge('forex_active_epics', 'Number of active currency pairs being monitored')

class MetricsCollector:
    def __init__(self):
        # Start metrics server
        start_http_server(8000)
    
    def record_signal_detected(self, epic, strategy):
        """Record signal detection"""
        SIGNALS_DETECTED.labels(epic=epic, strategy=strategy).inc()
    
    def record_processing_time(self, processing_time):
        """Record signal processing time"""
        SIGNAL_PROCESSING_TIME.observe(processing_time)
    
    def record_claude_call(self, success=True):
        """Record Claude API call"""
        status = 'success' if success else 'error'
        CLAUDE_API_CALLS.labels(status=status).inc()
    
    def record_db_query_time(self, query_time):
        """Record database query time"""
        DATABASE_QUERY_TIME.observe(query_time)
    
    def update_active_epics(self, count):
        """Update active epics count"""
        ACTIVE_EPICS.set(count)

# Usage
metrics = MetricsCollector()

def process_signal_with_metrics(signal):
    start_time = time.time()
    
    try:
        # Process signal
        result = process_signal(signal)
        
        # Record metrics
        metrics.record_signal_detected(
            signal['epic'], 
            signal['strategy']
        )
        
        return result
    
    finally:
        processing_time = time.time() - start_time
        metrics.record_processing_time(processing_time)
```

---

This completes the comprehensive technical implementation guide for the Forex Scanner project. The documentation covers all major aspects of development, deployment, and maintenance of the system.

Key highlights:
- **Complete API reference** for all core components
- **Database schema** with relationships and optimization tips
- **Strategy development framework** with testing examples
- **Production deployment** configurations for Docker and Kubernetes
- **Performance optimization** techniques and caching strategies
- **Security best practices** for API keys and database access
- **Monitoring and alerting** setup with metrics collection

The documentation should be saved in your `documentation/` folder and updated as the project evolves.