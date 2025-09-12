#!/usr/bin/env python3
"""
Quick patch to add debug logging to understand why gaps aren't being filtered
"""

import sys
import os

# Read the auto_backfill.py file
file_path = "/app/igstream/auto_backfill.py"

with open(file_path, 'r') as f:
    content = f.read()

# Add logging after gap detection
old_detection = '            logger.info(f"Found {len(prioritized_gaps)} gaps to backfill")'
new_detection = '''            logger.info(f"Found {len(prioritized_gaps)} gaps to backfill")
            
            # Log details of gaps for debugging
            for gap in prioritized_gaps[:5]:  # Show first 5 gaps
                logger.info(f"  Gap: {gap['epic']} {gap['timeframe']}m at {gap['gap_start']} (weekday: {gap['gap_start'].weekday()}, hour: {gap['gap_start'].hour})")'''

content = content.replace(old_detection, new_detection)

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print("✅ Added debug logging to auto_backfill.py")