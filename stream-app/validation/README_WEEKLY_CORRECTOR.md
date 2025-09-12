# Weekly Close Price Corrector

## Overview
This script fixes corrupted close price data from streaming services by replacing it with accurate data from the IG REST API. It targets the systematic +8 pip error found in historical streaming data and corrects it for the past week.

## Key Features
- ✅ **Safe Operation**: Dry-run mode by default
- ✅ **Comprehensive Coverage**: All 13 forex pairs × 3 timeframes (5m, 15m, 60m)
- ✅ **Rate Limited**: Respects API limits with 500ms delays
- ✅ **Smart Targeting**: Only corrects `data_source = 'chart_streamer'` entries
- ✅ **Audit Trail**: Marks corrections with `data_source = 'api_backfill_fixed'`
- ✅ **Progress Tracking**: Detailed logging and final reports

## Files Created
- `weekly_close_price_corrector.py` - Main correction script
- `test_corrector.py` - Validation test script
- `README_WEEKLY_CORRECTOR.md` - This documentation

## Usage Examples

### 1. Test First (Recommended)
```bash
# Run the test script to validate functionality
cd /datadrive/Trader/TradeSystemV1/stream-app
python test_corrector.py
```

### 2. Dry Run (Safe Preview)
```bash
# See what would be corrected without making changes
python weekly_close_price_corrector.py --dry-run
```

### 3. Test Single Epic
```bash
# Test with just EURUSD 5m to verify accuracy
python weekly_close_price_corrector.py --dry-run --epic CS.D.EURUSD.CEEM.IP --timeframe 5
```

### 4. Live Correction (After Testing)
```bash
# Apply corrections to all data
python weekly_close_price_corrector.py --live
```

## Safety Considerations

### Before Running
1. **Database Backup**: Consider backing up the `ig_candles` table
2. **Off-Hours**: Run during non-trading hours to minimize impact
3. **Test First**: Always run dry-run mode first
4. **Monitor Resources**: Watch database performance during execution

### Data Selection
- **Target**: Only `data_source = 'chart_streamer'` (corrupted streaming data)
- **Preserve**: Leaves recent streaming data and existing backfilled data intact
- **Period**: 1 week back from today 20:00 UTC (customizable)
- **Threshold**: Only corrects differences >2 pips

## Expected Results

### Correction Statistics
- **Period**: 2025-08-26 20:00 UTC to 2025-09-02 20:00 UTC (1 week)
- **Scope**: 13 forex pairs × 3 timeframes = 39 data streams
- **Estimate**: ~2,000-5,000 corrupted entries expected
- **Corrections**: Only entries with >2 pip error will be updated

### Output Files
- `weekly_correction.log` - Detailed execution log
- `correction_report_YYYYMMDD_HHMMSS.log` - Final statistics report

## Technical Details

### Forex Pairs Processed
```python
# Major USD pairs
'CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.AUDUSD.MINI.IP',
'CS.D.NZDUSD.MINI.IP', 'CS.D.USDCHF.MINI.IP', 'CS.D.USDCAD.MINI.IP',

# JPY pairs  
'CS.D.USDJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP', 'CS.D.GBPJPY.MINI.IP',
'CS.D.AUDJPY.MINI.IP', 'CS.D.CADJPY.MINI.IP', 'CS.D.CHFJPY.MINI.IP',
'CS.D.NZDJPY.MINI.IP'
```

### Timeframes
- **5m**: Most critical for short-term strategies
- **15m**: Important for trend confirmation
- **60m**: Long-term trend analysis

### API Rate Limiting
- **Delay**: 500ms between API calls
- **Batch Size**: Max 100 candles per request
- **Timeout**: 30 seconds per request
- **Retry**: Built-in error handling with logging

## Troubleshooting

### Authentication Issues
```bash
# Check Azure Key Vault secrets
demoapikey - IG API key
demopwd - IG password
```

### Database Connection
- Ensure PostgreSQL is running
- Check `DATABASE_URL` environment variable
- Verify `ig_candles` table exists

### API Rate Limits
- Script includes built-in rate limiting
- If errors persist, increase `api_call_delay` in script
- Consider running in smaller batches

### Memory/Performance
- Script processes data in batches to manage memory
- Monitor database performance during execution
- Can be interrupted and resumed (idempotent design)

## Command Line Arguments

```bash
python weekly_close_price_corrector.py [OPTIONS]

Options:
  --dry-run          Run in preview mode (default: True)
  --live             Apply corrections (overrides dry-run)
  --epic EPIC        Process single epic only (testing)
  --timeframe {5,15,60}  Process single timeframe only (testing)
  -h, --help         Show help message
```

## Validation

After running corrections, validate results by:
1. **Check Logs**: Review correction statistics in output
2. **Database Query**: Verify entries have `data_source = 'api_backfill_fixed'`
3. **Spot Check**: Compare a few corrected timestamps with IG Charts
4. **Strategy Test**: Run backtests to ensure improved accuracy

## Post-Correction

Once corrections are complete:
1. **Monitor Trading**: Watch for improved signal accuracy
2. **Backtest Validation**: Run strategy backtests on corrected data
3. **Performance**: Monitor database performance impact
4. **Documentation**: Update any analysis that relied on old corrupted data

## Support

For issues or questions:
1. Check the execution logs first
2. Review this documentation
3. Test with single epic/timeframe to isolate issues
4. Consider running during off-peak hours if performance is impacted