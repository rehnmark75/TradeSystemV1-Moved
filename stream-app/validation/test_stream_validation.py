#!/usr/bin/env python3
"""
Test script for stream validation system
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from igstream.stream_validator import StreamValidatorService, ValidationRequest
from igstream.ig_auth_prod import ig_login
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_stream_validation():
    """Test the stream validation system"""
    
    logger.info("🧪 Starting stream validation test...")
    
    try:
        # Login to get headers
        logger.info("🔐 Logging in to IG...")
        headers = await ig_login()
        
        if not headers:
            logger.error("❌ Failed to get IG headers")
            return False
            
        # Create validator
        logger.info("🔍 Initializing stream validator...")
        validator = StreamValidatorService(headers)
        
        # Create some test validation requests
        test_candles = [
            {
                'epic': 'CS.D.EURUSD.CEEM.IP',
                'timeframe': 5,
                'timestamp': datetime.utcnow() - timedelta(minutes=10),
                'streamed_ohlc': {
                    'open': 1.08500,
                    'high': 1.08550, 
                    'low': 1.08480,
                    'close': 1.08520
                }
            },
            {
                'epic': 'CS.D.GBPUSD.MINI.IP', 
                'timeframe': 5,
                'timestamp': datetime.utcnow() - timedelta(minutes=15),
                'streamed_ohlc': {
                    'open': 1.27200,
                    'high': 1.27250,
                    'low': 1.27180,
                    'close': 1.27230
                }
            }
        ]
        
        # Queue test candles for validation
        logger.info("📋 Queuing test candles for validation...")
        for candle in test_candles:
            validator.queue_candle_for_validation(
                epic=candle['epic'],
                timeframe=candle['timeframe'],
                timestamp=candle['timestamp'],
                streamed_ohlc=candle['streamed_ohlc'],
                data_source='test_client',
                priority=2  # High priority for testing
            )
        
        # Start validation worker
        logger.info("⚙️ Starting validation worker...")
        validator_task = asyncio.create_task(validator.start_validation_worker())
        
        # Run for a limited time
        await asyncio.sleep(30)  # Let it process for 30 seconds
        
        # Stop validator
        logger.info("🛑 Stopping validator...")
        validator.stop()
        validator_task.cancel()
        
        # Get final stats
        stats = validator.get_validation_stats()
        logger.info(f"📊 Final validation stats: {stats}")
        
        logger.info("✅ Stream validation test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_stream_validation())
    sys.exit(0 if success else 1)