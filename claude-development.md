# Development Best Practices

This document provides comprehensive development patterns, testing approaches, and best practices for maintaining and extending TradeSystemV1's codebase.

## Development Setup

### Local Development Environment

```bash
# Clone and setup repository
git clone <repository-url>
cd TradeSystemV1

# Start development services
docker-compose up -d postgres
docker-compose up -d fastapi-dev
docker-compose up -d streamlit

# Install dependencies for local development
cd dev-app && pip install -r requirements.txt
cd ../streamlit && pip install -r requirements.txt
cd ../worker/app && pip install -r requirements.txt
```

### Development Workflow

```bash
# Development cycle
1. Start services: docker-compose up -d
2. Make code changes in your IDE
3. Test changes: docker exec task-worker python -m pytest tests/
4. Run backtests: docker exec task-worker python forex_scanner/backtests/backtest_ema.py
5. Commit changes: git add . && git commit -m "feature: description"
6. Push to remote: git push origin feature-branch
```

### Environment Configuration

```bash
# Required environment variables
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/forex"
export IG_API_KEY="your-ig-api-key"
export AZURE_KEY_VAULT_URL="your-key-vault-url"

# Development vs Production
export ENVIRONMENT="development"  # or "production"
export DEBUG="true"               # Enable debug logging
export LOG_LEVEL="DEBUG"          # Detailed logging
```

## Code Organization Patterns

### Service Communication

**Internal Services (Docker Network):**
```python
# ✅ CORRECT: Use service names for internal communication
DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/forex"
API_BASE_URL = "http://fastapi-dev:8001"
STREAMLIT_URL = "http://streamlit:8501"

# ❌ WRONG: Hardcoded localhost (breaks in containers)
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/forex"
```

**External Integration:**
```python
# ✅ CORRECT: Robust error handling with retries
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_ig_api(endpoint: str, data: dict) -> dict:
    """Call IG Markets API with automatic retry"""
    try:
        response = requests.post(f"{IG_BASE_URL}{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"IG API call failed: {e}")
        raise
```

### Error Handling Patterns

**Comprehensive Logging:**
```python
import logging
from logging.handlers import RotatingFileHandler

# ✅ STANDARD logging configuration
def setup_logging(service_name: str, log_level: str = "INFO"):
    """Setup comprehensive logging for service"""
    
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f'logs/{service_name}.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```

**Database Transaction Management:**
```python
from contextlib import contextmanager

@contextmanager
def database_transaction(db_manager):
    """Context manager for database transactions with proper rollback"""
    conn = None
    try:
        conn = db_manager.get_connection()
        conn.begin()
        yield conn
        conn.commit()
        logger.debug("Database transaction committed successfully")
    except Exception as e:
        if conn:
            conn.rollback()
            logger.error(f"Database transaction rolled back due to error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Usage
with database_transaction(db_manager) as conn:
    # Multiple database operations
    conn.execute("INSERT INTO trades ...")
    conn.execute("UPDATE positions ...")
    # Automatic commit or rollback
```

**API Error Handling:**
```python
from enum import Enum
from dataclasses import dataclass

class ErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    DATABASE_ERROR = "database_error"
    EXTERNAL_API_ERROR = "external_api_error"
    CONFIGURATION_ERROR = "configuration_error"

@dataclass
class ServiceError:
    error_type: ErrorType
    message: str
    details: dict = None
    
def handle_service_error(func):
    """Decorator for standardized error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            raise ServiceError(ErrorType.VALIDATION_ERROR, str(e), {'field': e.field})
        except DatabaseError as e:
            raise ServiceError(ErrorType.DATABASE_ERROR, str(e), {'query': e.query})
        except APIError as e:
            raise ServiceError(ErrorType.EXTERNAL_API_ERROR, str(e), {'endpoint': e.endpoint})
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}")
            raise ServiceError(ErrorType.CONFIGURATION_ERROR, str(e))
    return wrapper
```

### Configuration Management

**Environment-based Configuration:**
```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

@dataclass
class TradingConfig:
    ig_api_key: str
    ig_base_url: str
    max_position_size: float
    risk_per_trade: float = 0.02

@dataclass
class AppConfig:
    environment: str
    debug: bool
    log_level: str
    database: DatabaseConfig
    trading: TradingConfig
    
    @classmethod
    def from_environment(cls) -> 'AppConfig':
        """Load configuration from environment variables"""
        return cls(
            environment=os.getenv('ENVIRONMENT', 'development'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            database=DatabaseConfig(
                url=os.getenv('DATABASE_URL'),
                pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            ),
            trading=TradingConfig(
                ig_api_key=os.getenv('IG_API_KEY'),
                ig_base_url=os.getenv('IG_BASE_URL', 'https://demo-api.ig.com'),
                max_position_size=float(os.getenv('MAX_POSITION_SIZE', '10000')),
            )
        )
    
    def validate(self) -> None:
        """Validate configuration completeness"""
        if not self.database.url:
            raise ValueError("DATABASE_URL is required")
        if not self.trading.ig_api_key:
            raise ValueError("IG_API_KEY is required")
```

## Testing Approach

### Backtesting as Primary Validation

TradeSystemV1 uses **backtesting** as the primary validation method rather than traditional unit tests:

```python
# ✅ COMPREHENSIVE backtest validation
class BacktestValidator:
    def __init__(self, strategy_class, epic: str, days: int = 30):
        self.strategy_class = strategy_class
        self.epic = epic
        self.days = days
        
    def run_validation_suite(self) -> dict:
        """Run comprehensive validation suite"""
        results = {}
        
        # 1. Parameter validation
        results['parameter_validation'] = self.validate_parameters()
        
        # 2. Signal generation validation
        results['signal_validation'] = self.validate_signal_generation()
        
        # 3. Performance validation
        results['performance_validation'] = self.validate_performance()
        
        # 4. Risk management validation
        results['risk_validation'] = self.validate_risk_management()
        
        return results
    
    def validate_parameters(self) -> dict:
        """Validate strategy parameters are reasonable"""
        strategy = self.strategy_class(epic=self.epic, use_optimal_parameters=True)
        
        # Get strategy parameters
        params = strategy._get_strategy_periods()
        
        # Validation rules
        validations = {
            'has_required_params': all(key in params for key in ['short', 'long', 'trend']),
            'logical_order': params['short'] < params['long'] < params['trend'],
            'reasonable_values': 1 <= params['short'] <= 50 and params['trend'] <= 500
        }
        
        return {
            'passed': all(validations.values()),
            'details': validations,
            'parameters': params
        }
    
    def validate_performance(self) -> dict:
        """Validate strategy performance meets minimum standards"""
        backtest_results = self.run_backtest()
        
        # Performance criteria
        min_win_rate = 0.35
        min_profit_factor = 1.1
        max_drawdown = 0.15
        
        performance_check = {
            'win_rate_acceptable': backtest_results['win_rate'] >= min_win_rate,
            'profit_factor_positive': backtest_results['profit_factor'] >= min_profit_factor,
            'drawdown_controlled': backtest_results['max_drawdown'] <= max_drawdown,
            'sufficient_signals': backtest_results['total_signals'] >= 5
        }
        
        return {
            'passed': all(performance_check.values()),
            'details': performance_check,
            'metrics': backtest_results
        }
```

### Strategy Testing Framework

```python
# ✅ STANDARDIZED strategy testing
class StrategyTestFramework:
    def __init__(self):
        self.test_epics = [
            'CS.D.EURUSD.MINI.IP',
            'CS.D.GBPUSD.MINI.IP', 
            'CS.D.AUDUSD.MINI.IP'
        ]
        
    def test_strategy_across_epics(self, strategy_class) -> dict:
        """Test strategy performance across multiple epics"""
        results = {}
        
        for epic in self.test_epics:
            try:
                validator = BacktestValidator(strategy_class, epic, days=30)
                epic_results = validator.run_validation_suite()
                results[epic] = epic_results
            except Exception as e:
                results[epic] = {'error': str(e), 'passed': False}
                
        # Aggregate results
        overall_passed = all(
            result.get('performance_validation', {}).get('passed', False) 
            for result in results.values() 
            if 'error' not in result
        )
        
        return {
            'overall_passed': overall_passed,
            'epic_results': results,
            'summary': self._generate_summary(results)
        }
        
    def _generate_summary(self, results: dict) -> dict:
        """Generate test summary statistics"""
        passed_count = sum(1 for r in results.values() if r.get('performance_validation', {}).get('passed', False))
        total_count = len([r for r in results.values() if 'error' not in r])
        
        return {
            'pass_rate': passed_count / total_count if total_count > 0 else 0,
            'epics_tested': total_count,
            'epics_passed': passed_count,
            'epics_failed': total_count - passed_count
        }
```

### Database Testing

```python
# ✅ DATABASE integration testing
class DatabaseTestSuite:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def test_optimization_data_integrity(self) -> dict:
        """Test optimization data integrity and completeness"""
        tests = {}
        
        with self.db_manager.get_connection() as conn:
            # Test 1: EMA optimization data completeness
            ema_count = conn.execute(
                "SELECT COUNT(DISTINCT epic) FROM ema_best_parameters"
            ).fetchone()[0]
            tests['ema_data_complete'] = ema_count >= 10
            
            # Test 2: MACD optimization data completeness  
            macd_count = conn.execute(
                "SELECT COUNT(DISTINCT epic) FROM macd_best_parameters"
            ).fetchone()[0]
            tests['macd_data_complete'] = macd_count >= 10
            
            # Test 3: Data consistency
            ema_results_count = conn.execute(
                "SELECT COUNT(*) FROM ema_optimization_results"
            ).fetchone()[0]
            tests['sufficient_test_data'] = ema_results_count >= 100000
            
            # Test 4: Performance scores reasonable
            avg_score = conn.execute(
                "SELECT AVG(best_composite_score) FROM ema_best_parameters"
            ).fetchone()[0]
            tests['reasonable_scores'] = 0.1 <= float(avg_score) <= 10.0
            
        return {
            'passed': all(tests.values()),
            'details': tests,
            'statistics': {
                'ema_epics': ema_count,
                'macd_epics': macd_count,
                'optimization_results': ema_results_count,
                'average_score': float(avg_score)
            }
        }
        
    def test_data_fetcher_performance(self) -> dict:
        """Test data fetcher performance and reliability"""
        from forex_scanner.core.enhanced_data_fetcher import EnhancedDataFetcher
        
        fetcher = EnhancedDataFetcher(self.db_manager)
        epic = 'CS.D.EURUSD.MINI.IP'
        
        # Performance test
        import time
        start_time = time.time()
        data = fetcher.get_enhanced_data(epic, 'EURUSD', '15m', lookback_hours=48)
        fetch_time = time.time() - start_time
        
        tests = {
            'fetch_speed_acceptable': fetch_time < 5.0,  # Under 5 seconds
            'data_returned': data is not None and not data.empty,
            'sufficient_data': len(data) >= 100 if data is not None else False,
            'required_columns': all(col in data.columns for col in ['ema_21', 'ema_50', 'ema_200']) if data is not None else False
        }
        
        return {
            'passed': all(tests.values()),
            'details': tests,
            'performance': {
                'fetch_time_seconds': fetch_time,
                'rows_returned': len(data) if data is not None else 0
            }
        }
```

## Security Best Practices

### Credential Management

```python
# ✅ SECURE credential handling
import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class SecureCredentialManager:
    def __init__(self):
        self.key_vault_url = os.getenv('AZURE_KEY_VAULT_URL')
        if self.key_vault_url:
            credential = DefaultAzureCredential()
            self.client = SecretClient(vault_url=self.key_vault_url, credential=credential)
        else:
            self.client = None
            
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault or environment fallback"""
        if self.client:
            try:
                secret = self.client.get_secret(secret_name)
                return secret.value
            except Exception as e:
                logger.warning(f"Key Vault access failed for {secret_name}: {e}")
                
        # Fallback to environment variable
        env_value = os.getenv(secret_name.upper())
        if not env_value:
            raise ValueError(f"Secret {secret_name} not found in Key Vault or environment")
            
        return env_value

# Usage
credential_manager = SecureCredentialManager()
ig_api_key = credential_manager.get_secret('ig-api-key')
database_password = credential_manager.get_secret('database-password')
```

### Input Validation

```python
# ✅ COMPREHENSIVE input validation
from typing import Union, List
import re

class TradingInputValidator:
    @staticmethod
    def validate_epic(epic: str) -> bool:
        """Validate epic format"""
        # Expected format: CS.D.EURUSD.MINI.IP
        pattern = r'^CS\.D\.[A-Z]{6}\.MINI\.IP$'
        return bool(re.match(pattern, epic))
    
    @staticmethod
    def validate_position_size(size: float, max_size: float = 100000) -> bool:
        """Validate position size is reasonable"""
        return 0 < size <= max_size
    
    @staticmethod
    def validate_price(price: float, epic: str) -> bool:
        """Validate price is reasonable for currency pair"""
        if 'JPY' in epic:
            return 50 <= price <= 200  # JPY pairs typically 50-200
        else:
            return 0.5 <= price <= 5.0  # Major pairs typically 0.5-5.0
            
    @staticmethod 
    def validate_timeframe(timeframe: str) -> bool:
        """Validate timeframe is supported"""
        valid_timeframes = ['5m', '15m', '1h', '4h', '1d']
        return timeframe in valid_timeframes
    
    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """Validate confidence score range"""
        return 0.0 <= confidence <= 1.0

# Usage in API endpoints
def validate_trade_request(request_data: dict) -> List[str]:
    """Validate trade request data"""
    errors = []
    
    epic = request_data.get('epic')
    if not epic or not TradingInputValidator.validate_epic(epic):
        errors.append("Invalid epic format")
        
    size = request_data.get('size')
    if not size or not TradingInputValidator.validate_position_size(float(size)):
        errors.append("Invalid position size")
        
    return errors
```

### API Security

```python
# ✅ API security patterns
from functools import wraps
import jwt
from datetime import datetime, timedelta

def require_api_key(f):
    """Decorator to require valid API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return {'error': 'Invalid API key'}, 401
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(max_requests: int = 100, window_minutes: int = 60):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            if is_rate_limited(client_ip, max_requests, window_minutes):
                return {'error': 'Rate limit exceeded'}, 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage
@app.route('/api/trade', methods=['POST'])
@require_api_key
@rate_limit(max_requests=50, window_minutes=60)
def create_trade():
    """Create new trade with security controls"""
    try:
        # Validate input
        errors = validate_trade_request(request.json)
        if errors:
            return {'error': 'Validation failed', 'details': errors}, 400
            
        # Process trade
        result = trading_service.create_trade(request.json)
        return {'success': True, 'trade_id': result['id']}
        
    except Exception as e:
        logger.error(f"Trade creation failed: {e}")
        return {'error': 'Internal server error'}, 500
```

## Performance Optimization

### Database Query Optimization

```python
# ✅ OPTIMIZED database queries
class OptimizedDataQueries:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def get_candle_data_optimized(self, epic: str, timeframe: int, hours: int = 48) -> pd.DataFrame:
        """Optimized candle data query with proper indexing"""
        
        # Use parameterized query with index-friendly WHERE clause
        query = """
            SELECT start_time, open_price_mid, high_price_mid, low_price_mid, close_price_mid, volume
            FROM ig_candles 
            WHERE epic = %(epic)s 
            AND timeframe = %(timeframe)s 
            AND start_time >= NOW() - INTERVAL '%(hours)s hours'
            ORDER BY start_time ASC
        """
        
        with self.db_manager.get_connection() as conn:
            # Use pandas read_sql for efficient data loading
            df = pd.read_sql_query(
                query, 
                conn, 
                params={'epic': epic, 'timeframe': timeframe, 'hours': hours},
                parse_dates=['start_time']
            )
            
        return df
    
    def get_optimization_results_batch(self, epics: List[str]) -> dict:
        """Batch fetch optimization results for multiple epics"""
        
        # Single query with IN clause instead of multiple queries
        query = """
            SELECT epic, best_ema_config, best_confidence_threshold, 
                   optimal_stop_loss_pips, optimal_take_profit_pips,
                   best_win_rate, best_composite_score
            FROM ema_best_parameters 
            WHERE epic = ANY(%(epics)s)
        """
        
        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params={'epics': epics})
            
        # Convert to dictionary for easy lookup
        return df.set_index('epic').to_dict('index')
```

### Memory Management

```python
# ✅ EFFICIENT memory usage
class MemoryEfficientDataProcessor:
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        
    def process_large_dataset(self, data_source) -> dict:
        """Process large datasets in chunks to manage memory"""
        results = {
            'total_processed': 0,
            'summary_stats': {}
        }
        
        # Process in chunks
        for chunk in self.get_data_chunks(data_source):
            # Process chunk
            chunk_results = self.process_chunk(chunk)
            
            # Update results
            results['total_processed'] += len(chunk)
            self.update_summary_stats(results['summary_stats'], chunk_results)
            
            # Free memory
            del chunk
            del chunk_results
            
        return results
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        # Convert to optimal dtypes
        for col in df.columns:
            if df[col].dtype == 'float64':
                # Check if can be downcast to float32
                if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                    df[col] = df[col].astype('float32')
                    
            elif df[col].dtype == 'int64':
                # Check if can be downcast to smaller int
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
                    
        return df
```

### Caching Strategies

```python
# ✅ INTELLIGENT caching
from functools import lru_cache
import redis
import pickle
from datetime import datetime, timedelta

class TradingDataCache:
    def __init__(self, redis_url: str = None):
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.redis_client = None
        self.memory_cache = {}
        
    def cache_market_data(self, epic: str, timeframe: str, data: pd.DataFrame, ttl_minutes: int = 5):
        """Cache market data with appropriate TTL"""
        cache_key = f"market_data:{epic}:{timeframe}"
        
        if self.redis_client:
            # Use Redis for distributed caching
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(cache_key, ttl_minutes * 60, serialized_data)
        else:
            # Use memory cache for single instance
            expiry = datetime.now() + timedelta(minutes=ttl_minutes)
            self.memory_cache[cache_key] = {
                'data': data,
                'expiry': expiry
            }
    
    def get_cached_market_data(self, epic: str, timeframe: str) -> pd.DataFrame:
        """Retrieve cached market data"""
        cache_key = f"market_data:{epic}:{timeframe}"
        
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
        else:
            cached = self.memory_cache.get(cache_key)
            if cached and datetime.now() < cached['expiry']:
                return cached['data']
                
        return None

    @lru_cache(maxsize=128)
    def get_optimization_parameters(self, epic: str, strategy: str) -> dict:
        """Cache optimization parameters (rarely change)"""
        # This will be cached in memory for the lifetime of the process
        from optimization.optimal_parameter_service import get_epic_optimal_parameters
        return get_epic_optimal_parameters(epic)
```

## Monitoring and Debugging

### Performance Monitoring

```python
# ✅ COMPREHENSIVE monitoring
import time
import psutil
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    cpu_percent: float
    memory_mb: float
    execution_time_ms: float
    database_queries: int
    cache_hits: int
    cache_misses: int

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start_monitoring(self, operation_name: str):
        """Start monitoring an operation"""
        self.start_time = time.time()
        self.metrics[operation_name] = {
            'start_cpu': psutil.cpu_percent(),
            'start_memory': psutil.virtual_memory().used / 1024 / 1024,
            'db_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def end_monitoring(self, operation_name: str) -> PerformanceMetrics:
        """End monitoring and return metrics"""
        if operation_name not in self.metrics:
            raise ValueError(f"No monitoring started for {operation_name}")
            
        end_time = time.time()
        start_metrics = self.metrics[operation_name]
        
        return PerformanceMetrics(
            cpu_percent=psutil.cpu_percent() - start_metrics['start_cpu'],
            memory_mb=psutil.virtual_memory().used / 1024 / 1024 - start_metrics['start_memory'],
            execution_time_ms=(end_time - self.start_time) * 1000,
            database_queries=start_metrics['db_queries'],
            cache_hits=start_metrics['cache_hits'],
            cache_misses=start_metrics['cache_misses']
        )

# Usage with decorator
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            monitor.start_monitoring(operation_name)
            
            try:
                result = func(*args, **kwargs)
                metrics = monitor.end_monitoring(operation_name)
                
                # Log performance metrics
                logger.info(f"Performance - {operation_name}: "
                           f"{metrics.execution_time_ms:.1f}ms, "
                           f"CPU: {metrics.cpu_percent:.1f}%, "
                           f"Memory: {metrics.memory_mb:.1f}MB")
                
                return result
            except Exception as e:
                logger.error(f"Operation {operation_name} failed: {e}")
                raise
                
        return wrapper
    return decorator
```

### Debugging Tools

```python
# ✅ ADVANCED debugging utilities
class TradingSystemDebugger:
    def __init__(self, logger):
        self.logger = logger
        
    def debug_signal_generation(self, strategy, epic: str, timeframe: str):
        """Debug signal generation step by step"""
        self.logger.debug(f"🔍 DEBUGGING signal generation for {epic} ({timeframe})")
        
        # 1. Data fetching
        data_fetcher = strategy.data_fetcher
        df = data_fetcher.get_enhanced_data(epic, epic.split('.')[2], timeframe)
        self.logger.debug(f"   📊 Data points: {len(df)}")
        self.logger.debug(f"   📅 Date range: {df.index[0]} to {df.index[-1]}")
        
        # 2. Technical indicators
        latest = df.iloc[-1]
        self.logger.debug(f"   📈 Latest prices: O:{latest['open']:.5f} H:{latest['high']:.5f} L:{latest['low']:.5f} C:{latest['close']:.5f}")
        self.logger.debug(f"   📊 EMAs: 21:{latest.get('ema_21', 'N/A'):.5f} 50:{latest.get('ema_50', 'N/A'):.5f} 200:{latest.get('ema_200', 'N/A'):.5f}")
        
        # 3. Strategy-specific indicators
        if hasattr(strategy, '_debug_indicators'):
            strategy._debug_indicators(df)
            
        # 4. Signal validation
        signal = strategy.detect_signal(df, epic, timeframe=timeframe)
        if signal:
            self.logger.debug(f"   ✅ Signal generated: {signal['type']} at {signal['confidence']:.1%} confidence")
        else:
            self.logger.debug(f"   ❌ No signal generated")
            
        return signal
    
    def debug_optimization_parameters(self, epic: str, strategy_type: str):
        """Debug optimization parameter loading"""
        self.logger.debug(f"🔍 DEBUGGING optimization parameters for {epic} ({strategy_type})")
        
        try:
            if strategy_type == 'ema':
                from optimization.optimal_parameter_service import get_epic_optimal_parameters
                params = get_epic_optimal_parameters(epic)
            elif strategy_type == 'macd':
                from optimization.optimal_parameter_service import get_macd_optimal_parameters
                params = get_macd_optimal_parameters(epic, '15m')
            else:
                self.logger.debug(f"   ❌ Unknown strategy type: {strategy_type}")
                return None
                
            self.logger.debug(f"   ✅ Parameters loaded: {params}")
            return params
            
        except Exception as e:
            self.logger.debug(f"   ❌ Parameter loading failed: {e}")
            return None
    
    def debug_database_queries(self, db_manager, epic: str):
        """Debug database queries and data availability"""
        self.logger.debug(f"🔍 DEBUGGING database queries for {epic}")
        
        with db_manager.get_connection() as conn:
            # Check candle data availability
            candle_count = conn.execute(
                "SELECT COUNT(*) FROM ig_candles WHERE epic = %s", (epic,)
            ).fetchone()[0]
            self.logger.debug(f"   📊 Candle records: {candle_count}")
            
            # Check optimization data
            ema_params = conn.execute(
                "SELECT * FROM ema_best_parameters WHERE epic = %s", (epic,)
            ).fetchone()
            self.logger.debug(f"   🎯 EMA optimization: {'Available' if ema_params else 'Missing'}")
            
            macd_params = conn.execute(
                "SELECT * FROM macd_best_parameters WHERE epic = %s", (epic,)
            ).fetchone()
            self.logger.debug(f"   🎯 MACD optimization: {'Available' if macd_params else 'Missing'}")
```

## Quality Assurance

### Code Review Checklist

```markdown
## Code Review Checklist

### Security
- [ ] No hardcoded credentials or API keys
- [ ] Input validation implemented for all user inputs
- [ ] SQL injection prevention (parameterized queries)
- [ ] Error messages don't expose sensitive information
- [ ] Rate limiting implemented for API endpoints

### Performance  
- [ ] Database queries optimized with appropriate indexes
- [ ] Caching implemented where appropriate
- [ ] Large datasets processed in chunks
- [ ] Memory usage optimized (no memory leaks)
- [ ] Response times under acceptable limits

### Reliability
- [ ] Comprehensive error handling with proper logging
- [ ] Database transactions with rollback capability
- [ ] Retry mechanisms for external API calls
- [ ] Graceful degradation when services unavailable
- [ ] Circuit breaker pattern for external dependencies

### Testing
- [ ] Backtest validation for strategy changes
- [ ] Performance benchmarks run and documented
- [ ] Edge cases identified and tested
- [ ] Database migrations tested
- [ ] Integration tests pass

### Documentation
- [ ] Code changes documented in relevant .md files
- [ ] API changes documented with examples
- [ ] Configuration changes noted
- [ ] Breaking changes highlighted
```

### Deployment Checklist

```bash
# ✅ DEPLOYMENT validation checklist

# 1. Pre-deployment checks
docker-compose config --quiet                    # Validate compose file
docker exec task-worker python -m pytest tests/ # Run test suite
docker exec task-worker python forex_scanner/optimization/test_dynamic_integration.py

# 2. Database validation
docker exec postgres psql -U postgres -d forex -c "SELECT COUNT(*) FROM ig_candles;"
docker exec postgres psql -U postgres -d forex -c "SELECT COUNT(*) FROM ema_best_parameters;"

# 3. Service health checks
curl -f http://localhost:8001/health || echo "❌ API health check failed"
curl -f http://localhost:8501 || echo "❌ Streamlit health check failed"

# 4. Performance validation
docker exec task-worker python -c "
import time
start = time.time()
from optimization.optimal_parameter_service import get_epic_optimal_parameters
params = get_epic_optimal_parameters('CS.D.EURUSD.CEEM.IP')
duration = time.time() - start
print(f'Parameter service: {duration:.3f}s')
assert duration < 2.0, 'Performance regression detected'
"

# 5. Configuration validation
docker exec task-worker python -c "
from configdata.strategies import validate_strategy_configs
results = validate_strategy_configs()
assert results['overall_valid'], 'Configuration validation failed'
print('✅ All strategy configurations valid')
"
```

For command usage, see [Commands & CLI](claude-commands.md).
For architectural context, see [System Architecture](claude-architecture.md).
For strategy patterns, see [Strategy Development](claude-strategies.md).