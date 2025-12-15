#!/usr/bin/env python3
"""Test script for Claude Vision analysis with chart saving"""

import sys
sys.path.insert(0, '/app/forex_scanner')
import logging
import os

# Setup logging to see output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

from alerts.claude_analyzer import ClaudeAnalyzer
from core.data_fetcher import DataFetcher
import config

print("\n" + "="*60)
print("CLAUDE VISION ANALYSIS TEST")
print("="*60)

# Initialize data fetcher (needs db_manager)
from core.database import DatabaseManager
db_manager = DatabaseManager(config.DATABASE_URL)
data_fetcher = DataFetcher(db_manager=db_manager)

# Initialize Claude analyzer with data fetcher
analyzer = ClaudeAnalyzer(
    api_key=config.CLAUDE_API_KEY,
    auto_save=True,
    save_directory="claude_analysis",
    data_fetcher=data_fetcher
)

print(f"\n=== Claude Analyzer Status ===")
print(f"Chart generator: {analyzer.chart_generator}")
print(f"Data fetcher available: {analyzer.data_fetcher is not None}")
print(f"Vision API enabled: {analyzer.use_vision_api}")
print(f"Vision strategies: {analyzer.vision_strategies}")
print(f"Save vision artifacts: {analyzer.save_vision_artifacts}")

# Create a test signal mimicking the AUDJPY signal from alert 6464
test_signal = {
    'epic': 'CS.D.AUDJPY.MINI.IP',
    'signal_type': 'BEAR',
    'signal': 'BEAR',
    'strategy': 'SMC_SIMPLE',
    'price': 103.05,
    'entry_price': 103.05,
    'stop_loss': 103.25,
    'take_profit': 102.56,
    'confidence_score': 0.75,
    'rr_ratio': 2.45,
    'timestamp': '2025-12-15T16:12:41',
    'strategy_indicators': {
        'tier1_ema': {'bias': 'bearish'},
        'tier2_swing': {'break_confirmed': True},
        'tier3_entry': {'entry_quality': 'good'}
    }
}

print(f"\n=== Testing Vision Analysis ===")
print(f"Testing with signal: {test_signal['epic']} {test_signal['signal_type']}")

# Run vision analysis
result = analyzer.analyze_signal_with_vision(
    signal=test_signal,
    candles=None,  # Will be fetched by analyzer
    alert_id=6464,
    save_to_file=True
)

if result:
    print(f"\n=== Analysis Result ===")
    print(f"Score: {result.get('score')}")
    print(f"Decision: {result.get('decision')}")
    print(f"Vision used: {result.get('vision_used')}")
    print(f"Chart generated: {result.get('chart_generated')}")
    print(f"Reason: {result.get('reason')}")
else:
    print("Analysis returned None!")

# Check if files were saved
vision_dir = "/app/claude_analysis/vision_analysis"
if os.path.exists(vision_dir):
    files = os.listdir(vision_dir)
    print(f"\n=== Vision Analysis Files ===")
    print(f"Directory: {vision_dir}")
    if files:
        for f in files:
            filepath = os.path.join(vision_dir, f)
            size = os.path.getsize(filepath)
            print(f"  - {f} ({size} bytes)")
    else:
        print("  (no files)")
else:
    print(f"\nVision directory does not exist: {vision_dir}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
