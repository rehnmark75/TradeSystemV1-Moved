# Chart Stream Data Validation System

## Overview

The Chart Stream Data Validation System is a comprehensive solution for maintaining data integrity in real-time forex trading streams. It provides multi-layered validation, quality scoring, and automated correction mechanisms to ensure accurate price data for trading systems.

## Architecture Components

### 1. Chart Streamer (`chart_streamer.py`)

**Purpose**: Primary streaming component that receives real-time candle data from IG Markets via Lightstreamer.

**Key Components**:
- **CandleData**: Structured data container with built-in validation
- **CandleBuffer**: Manages partial updates and detects candle completion
- **IGChartCandleListener**: Processes streaming updates and stores validated data
- **StreamManager**: Handles connection lifecycle and reconnection logic

**Data Flow**:
```
Lightstreamer → CandleBuffer → Validation → Database Storage → API Validation Queue
```

### 2. Stream Validator (`stream_validator.py`)

**Purpose**: Cross-validates streamed data against IG REST API to ensure accuracy.

**Key Features**:
- Asynchronous validation queue processing
- Rate limiting (80 requests/minute)
- Configurable validation frequency and delays
- Confidence scoring and discrepancy classification
- Automated price correction for critical discrepancies

**Validation Workflow**:
```
Stream Data → Queue → Rate Limiting → API Fetch → Compare → Log/Correct → Statistics
```

### 3. Database Validation Function (`validate_price_data`)

**Purpose**: PostgreSQL function that performs real-time validation checks on incoming candle data.

**Validation Rules**:
- **Price Movement**: Flags movements >10 pips from recent candle
- **Suspicious Activity**: Flags movements >30 pips as potentially erroneous
- **Data Staleness**: Flags data >30 minutes old during market hours
- **Quality Scoring**: Assigns confidence scores based on validation results

## Data Validation Layers

### Layer 1: Structural Validation
- **OHLC Relationships**: Ensures High ≥ Open/Close ≥ Low
- **Spread Validation**: Ensures Offer prices ≥ Bid prices
- **NaN/Inf Detection**: Filters out invalid numerical values
- **Field Completeness**: Ensures all required fields are present

### Layer 2: Database Validation (`validate_price_data` function)
- **Price Continuity**: Compares with recent historical prices
- **Movement Thresholds**: Configurable pip-based thresholds
- **Temporal Validation**: Checks data freshness
- **Quality Scoring**: Calculates confidence scores (0.0-1.0)

### Layer 3: API Cross-Validation (`stream_validator.py`)
- **Independent Verification**: Compares stream vs REST API data
- **Discrepancy Classification**: NONE/MINOR/MODERATE/MAJOR/CRITICAL
- **Automated Correction**: Updates database for critical discrepancies
- **Performance Monitoring**: Tracks validation statistics

## Quality Scoring System

Quality scores range from 0.0 (unreliable) to 1.0 (perfect):

- **1.0**: Perfect data with no validation issues
- **0.8-0.99**: Minor issues (e.g., small price movements)
- **0.5-0.79**: Moderate issues (e.g., larger movements, stale data)
- **0.3-0.49**: Major issues (multiple validation flags)
- **0.0-0.29**: Critical issues (suspicious activity, API errors)

## Configuration Parameters

### Stream Validator Settings
```python
STREAM_VALIDATION_DELAY_SECONDS = 45      # Delay before validation
ENABLE_STREAM_API_VALIDATION = True       # Enable/disable validation
STREAM_VALIDATION_FREQUENCY = 5           # Validate every 5th candle
```

### Database Function Thresholds
```sql
p_threshold_pips = 10.0                   -- Price movement threshold
SUSPICIOUS_MOVE_THRESHOLD = 30.0          -- Suspicious activity threshold  
STALE_DATA_THRESHOLD = '30 minutes'       -- Data freshness requirement
```

## Logging and Monitoring

### Log Levels
- **DEBUG**: Minor validation issues (quality_score ≥ 0.5)
- **WARNING**: Critical validation issues (quality_score < 0.5)
- **ERROR**: API validation failures, connection issues
- **CRITICAL**: System-level failures, database errors

### Key Metrics
- **Total Validations**: Count of all validation attempts
- **Success Rate**: Percentage of successful API validations
- **Discrepancy Rate**: Frequency of validation issues
- **Correction Count**: Number of automated price corrections

## Error Handling and Recovery

### Connection Management
- **Automatic Reconnection**: Exponential backoff up to 10 attempts
- **Health Monitoring**: 30-second connection health checks
- **Graceful Degradation**: Continue operation without API validation if needed

### Data Integrity Protection
- **Transaction Safety**: Database operations use proper transaction management
- **Rollback Capability**: Failed operations don't corrupt existing data
- **Audit Trails**: All validation results logged to `price_validation_log` table

### Rate Limiting Protection
- **API Limits**: Respects IG Markets rate limits (80 req/min)
- **Queue Management**: Prevents validation queue overflow (max 1000 items)
- **Priority Processing**: High-priority validations processed first

## Database Schema

### Main Tables
```sql
-- Primary candle storage
ig_candles (
    start_time TIMESTAMP,
    epic VARCHAR,
    timeframe INTEGER,
    open/high/low/close DECIMAL,
    quality_score DECIMAL,
    validation_flags TEXT[]
)

-- Validation audit trail
price_validation_log (
    epic VARCHAR,
    timeframe INTEGER,
    candle_time TIMESTAMP,
    validation_type VARCHAR,
    severity VARCHAR,
    message TEXT,
    price_difference_pips DECIMAL
)
```

## Usage Examples

### Starting Chart Stream
```python
from igstream.chart_streamer import stream_chart_candles
from igstream.ig_auth_prod import ig_login

# Get authentication headers
headers = ig_login()

# Start streaming for EURUSD
await stream_chart_candles("CS.D.EURUSD.MINI.IP", headers)
```

### Monitoring Validation Statistics
```python
# Access validation stats
stats = stream_manager.validator.get_validation_stats()
print(f"Success Rate: {stats['success_rate_pct']}%")
print(f"Corrections Made: {stats['corrections_made']}")
```

### Querying Validation Logs
```sql
-- Recent validation issues
SELECT * FROM price_validation_log 
WHERE severity IN ('WARNING', 'CRITICAL')
AND created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;
```

## Performance Optimization

### Stream Processing
- **Asynchronous Operations**: Non-blocking validation and database writes
- **Connection Pooling**: Efficient database connection management
- **Batch Processing**: Grouped database operations where possible

### Memory Management
- **Bounded Queues**: Prevent memory leaks from queue overflow
- **Data Cleanup**: Regular cleanup of old validation records
- **Resource Limits**: Configurable timeouts and connection limits

## Troubleshooting Guide

### Common Issues
1. **High Validation Warnings**: Adjust thresholds in `validate_price_data` function
2. **API Rate Limiting**: Reduce validation frequency or increase delays
3. **Connection Drops**: Check network stability and authentication tokens
4. **Database Locks**: Monitor concurrent access patterns

### Diagnostic Commands
```bash
# Check streaming service logs
docker-compose logs -f fastapi-stream

# Monitor validation statistics
docker exec postgres psql -U postgres -d forex -c "SELECT * FROM price_validation_log ORDER BY created_at DESC LIMIT 10;"

# Check connection health
curl http://localhost:8003/health
```

## Security Considerations

### Authentication
- **Token Management**: Secure storage and refresh of IG API tokens
- **Connection Security**: SSL/TLS for all external communications
- **Access Control**: Database access limited to service accounts

### Data Protection
- **Input Validation**: All external data validated before processing
- **SQL Injection Prevention**: Parameterized queries throughout
- **Audit Logging**: Complete audit trail of all data modifications

## Future Enhancements

### Planned Features
- **Machine Learning Validation**: AI-powered anomaly detection
- **Multi-Source Validation**: Cross-validation with multiple data providers
- **Real-time Alerting**: Immediate notifications for critical issues
- **Advanced Analytics**: Trend analysis and prediction capabilities

### Scalability Improvements
- **Horizontal Scaling**: Support for multiple stream instances
- **Load Balancing**: Distribute validation load across nodes
- **Caching Layer**: Reduce database load with intelligent caching
- **Microservice Architecture**: Further decomposition of components

---

*This documentation covers the Chart Stream Data Validation System as of August 2025. For updates and modifications, refer to the git commit history and deployment logs.*