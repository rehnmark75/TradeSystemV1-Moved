# Old Services - Legacy Components

This directory contains legacy service files that have been replaced by the new `igstream/` module architecture.

## Moved Files and Replacement Status

### Authentication Services
- **`ig_auth.py`** - Old IG authentication module
  - **Replaced by**: `igstream/ig_auth_prod.py`
  - **Status**: Legacy - new system uses production auth module

### Stream Management (Legacy Architecture)
- **`stream_controller.py`** - Old stream controller
  - **Replaced by**: `igstream/sync_manager.py` 
  - **Status**: Legacy - new system uses sync_manager for stream orchestration

- **`stream_manager.py`** - Old stream manager with trade tracking
  - **Replaced by**: `igstream/sync_manager.py` + `igstream/chart_streamer.py`
  - **Status**: Legacy - still referenced by router but not used by main app
  - **Dependencies**: Uses `tradelistner.py` and `trade_utils.py`

### Trade Management (Legacy)
- **`tradelistner.py`** - Trade listener for monitoring positions
  - **Status**: Legacy - only used by old `stream_manager.py`
  - **Functionality**: Trade monitoring and position tracking

- **`trade_utils.py`** - Trade utility functions
  - **Status**: Legacy - only used by old `stream_manager.py` 
  - **Functionality**: Stop loss modification and trade utilities

## Current Active Services

The following services remain active in the parent `/services` directory:

### Core Infrastructure
- **`db.py`** - Database connections and session management
- **`models.py`** - SQLAlchemy database models 
- **`keyvault.py`** - Azure Key Vault integration for secrets

### Monitoring & Operations
- **`alert_manager.py`** - Real-time alert management system
- **`log_parser.py`** - Log file parsing and analysis
- **`operation_tracker.py`** - Gap detection and backfill operation tracking

## Migration Notes

- The current system uses `igstream/sync_manager.py` for all stream management
- Authentication is handled by `igstream/ig_auth_prod.py`  
- Chart data streaming is managed by `igstream/chart_streamer.py`
- Auto backfill uses `igstream/auto_backfill.py` with gap detection
- The router (`routers/stream_router.py`) may still reference some old services but main app does not

## Future Cleanup

These files can potentially be removed entirely once:
1. Router endpoints are updated to not reference legacy stream management
2. Any remaining trade tracking functionality is migrated to new architecture
3. Thorough testing confirms no dependencies remain