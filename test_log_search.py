#!/usr/bin/env python3
"""
Test script for the improved log search functionality
"""

import sys
import os
from datetime import datetime, timedelta

# Add services to path
sys.path.append('streamlit/services')

from simple_log_intelligence import SimpleLogParser

def test_log_discovery():
    """Test log file discovery"""
    print("="*80)
    print("TESTING LOG FILE DISCOVERY")
    print("="*80)

    parser = SimpleLogParser()

    print(f"\n📂 Base log directory: {parser.base_log_dir if parser.base_log_dir else '/logs (container)'}")
    print(f"\n📋 Discovered log files:\n")

    for log_type, files in parser.log_files.items():
        print(f"\n{log_type}:")
        print(f"  Count: {len(files)}")
        for file in files:
            if parser.base_log_dir == "":
                file_path = file
            else:
                file_path = os.path.join(parser.base_log_dir, file)

            exists = "✅" if os.path.exists(file_path) else "❌"
            size_mb = os.path.getsize(file_path) / 1024 / 1024 if os.path.exists(file_path) else 0
            print(f"  {exists} {file} ({size_mb:.2f} MB)")

def test_log_info():
    """Test log file info retrieval"""
    print("\n" + "="*80)
    print("TESTING LOG FILE INFO")
    print("="*80)

    parser = SimpleLogParser()
    info = parser.get_log_file_info()

    for log_type, data in info.items():
        print(f"\n{log_type}:")
        print(f"  Total files: {data['count']}")
        print(f"  Total size: {data['total_size'] / 1024 / 1024:.2f} MB")

        if data['files']:
            print(f"  Files:")
            for file_info in data['files'][:3]:  # Show first 3
                if file_info['exists']:
                    print(f"    ✅ {file_info['path']}")
                    print(f"       Size: {file_info['size'] / 1024:.1f} KB")
                    print(f"       Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"    ❌ {file_info['path']} - NOT FOUND")

def test_signal_data():
    """Test signal data retrieval"""
    print("\n" + "="*80)
    print("TESTING SIGNAL DATA RETRIEVAL (Last 4 hours)")
    print("="*80)

    parser = SimpleLogParser()
    signal_data = parser.get_recent_signal_data(hours_back=4)

    print(f"\n📊 Signal Statistics:")
    print(f"  Total signals: {signal_data['total_signals']}")
    print(f"  Detected: {signal_data['signals_detected']}")
    print(f"  Rejected: {signal_data['signals_rejected']}")
    print(f"  Avg confidence: {signal_data['avg_confidence']:.1%}")
    print(f"  Success rate: {signal_data['success_rate']:.1%}")
    print(f"  Top epic: {signal_data['top_epic']}")
    print(f"  Active pairs: {signal_data['active_pairs']}")

def test_recent_activity():
    """Test recent activity retrieval"""
    print("\n" + "="*80)
    print("TESTING RECENT ACTIVITY (Last 2 hours, max 10 entries)")
    print("="*80)

    parser = SimpleLogParser()
    activities = parser.get_recent_activity(hours_back=2, max_entries=10)

    print(f"\n🚀 Recent Activities: {len(activities)} found\n")

    for activity in activities[:5]:  # Show first 5
        timestamp = activity['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        activity_type = activity['type']
        epic = activity.get('epic', 'N/A')
        print(f"  [{timestamp}] {activity_type} - {epic}")

def main():
    """Run all tests"""
    try:
        test_log_discovery()
        test_log_info()
        test_signal_data()
        test_recent_activity()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
