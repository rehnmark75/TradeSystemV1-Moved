# Stream-App Validation Scripts

This directory contains testing, debugging, and validation scripts that are not part of the core FastAPI application.

## Script Categories

### Data Validation & Analysis
- `analyze_why_missed.py` - Analyze missed candle data
- `check_actual_corruption.py` - Check for data corruption in candles
- `check_chart_streamer_entries.py` - Validate chart streamer entries

### Backfill & Data Correction
- `backfill_eurusd_sept8.py` - Specific EURUSD backfill for September 8th
- `backfill_eurusd_specific.py` - Targeted EURUSD backfill operations
- `fill_specific_gap.py` - Fill specific data gaps
- `fix_eurusd_scaling.py` - Fix EURUSD price scaling issues
- `fix_remaining_eurusd_scaling.py` - Additional EURUSD scaling fixes

### Price Correction Tools
- `close_price_corrector.py` - Correct close price discrepancies
- `correct_specific_entries.py` - Correct specific data entries
- `selective_backfill_corrector.py` - Selective backfill corrections
- `weekly_close_price_corrector.py` - Weekly close price corrections

### Testing Scripts
- `test_corrector.py` - Test price correction functionality
- `test_corrector_on_chart_streamer.py` - Test corrector on chart streamer
- `test_db_update.py` - Test database update operations
- `test_stream_validation.py` - Test stream validation logic

### Debug & Utility
- `debug_batch_update.py` - Debug batch update operations
- `debug_weekly_corrector_issue.py` - Debug weekly corrector issues
- `patch_backfill_logging.py` - Patch backfill logging functionality
- `price_field_fix_validation.py` - Validate price field fixes

## Usage

These scripts are meant to be run manually for:
- Data validation and debugging
- Historical data correction
- Testing specific functionality
- One-off maintenance tasks

## Core Application Files

The core FastAPI application files remain in the parent directory:
- `main.py` - Main FastAPI application
- `config.py` - Application configuration
- `routers/` - API routing
- `services/` - Core services
- `igstream/` - Streaming functionality