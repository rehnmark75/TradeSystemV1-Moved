# Unused Services

This folder contains service files that are no longer actively used by the dev-app system.

## Files

### ig_streamer.py
- **Purpose**: Lightstreamer integration for real-time price streaming
- **Status**: Unused - no imports found in codebase
- **Reason**: Streaming functionality has been moved to a separate container (fastapi-stream)
- **Date moved**: 2024-09-10
- **Dependencies**: lightstreamer.client, asyncio

## Notes

Files in this directory were moved here to clean up the active services directory while preserving the code for potential future reference or restoration if needed.

Before permanently deleting any files from this directory, verify that:
1. No other services or containers reference them
2. The functionality has been completely replaced or is no longer needed
3. The code doesn't contain any unique business logic that might be needed elsewhere