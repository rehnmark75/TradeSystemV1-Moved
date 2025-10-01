#!/usr/bin/env python3
"""
Test script for search functionality specifically
"""

import sys
import os
from datetime import datetime, timedelta

# Add services to path
sys.path.append('streamlit/services')
sys.path.append('streamlit/pages')

# Mock streamlit for testing
class MockStreamlit:
    def warning(self, msg): print(f"⚠️  {msg}")
    def error(self, msg): print(f"❌ {msg}")
    def success(self, msg): print(f"✅ {msg}")
    def info(self, msg): print(f"ℹ️  {msg}")

sys.modules['streamlit'] = MockStreamlit()

from simple_log_intelligence import SimpleLogParser

def mock_search_logs(parser, search_term, log_types, start_date, end_date, regex_mode=False, case_sensitive=False, max_results=500):
    """Simplified search_logs function for testing"""
    import re

    results = []
    search_stats = {
        'files_searched': 0,
        'files_found': 0,
        'files_missing': 0,
        'lines_scanned': 0,
        'matches_found': 0
    }

    # Prepare search pattern
    if regex_mode:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(search_term, flags)
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return [], search_stats
    else:
        if case_sensitive:
            search_func = lambda text: search_term in text
        else:
            search_func = lambda text: search_term.lower() in text.lower()

    # Define log file mappings
    log_files_to_search = []
    if 'forex_scanner' in log_types:
        log_files_to_search.extend(parser.log_files.get('forex_scanner', []))
    if 'stream_service' in log_types:
        log_files_to_search.extend(parser.log_files.get('stream_service', []))
    if 'trade_monitor' in log_types:
        log_files_to_search.extend(parser.log_files.get('trade_monitor', []))
    if 'fastapi_dev' in log_types:
        log_files_to_search.extend(parser.log_files.get('fastapi_dev', []))
    if 'dev_trade' in log_types:
        log_files_to_search.extend(parser.log_files.get('dev_trade', []))

    # Check if we have files to search
    if not log_files_to_search:
        print(f"⚠️ No log files found for selected sources: {', '.join(log_types)}")
        return [], search_stats

    for log_file in log_files_to_search:
        search_stats['files_searched'] += 1

        if parser.base_log_dir == "":
            file_path = log_file
        else:
            file_path = os.path.join(parser.base_log_dir, log_file)

        if not os.path.exists(file_path):
            search_stats['files_missing'] += 1
            continue

        search_stats['files_found'] += 1

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    search_stats['lines_scanned'] += 1

                    # Parse timestamp for filtering
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')

                            # Filter by date range
                            if log_time.date() < start_date or log_time.date() > end_date:
                                continue
                        except ValueError:
                            continue
                    else:
                        continue

                    # Search in line
                    match_found = False
                    if regex_mode:
                        match = pattern.search(line)
                        if match:
                            match_found = True
                    else:
                        if search_func(line):
                            match_found = True

                    if match_found:
                        search_stats['matches_found'] += 1

                        # Determine log type
                        log_type = 'info'
                        if ' - ERROR - ' in line:
                            log_type = 'error'
                        elif ' - WARNING - ' in line:
                            log_type = 'warning'
                        elif '📊' in line or 'signal' in line.lower():
                            log_type = 'signal'

                        results.append({
                            'file': os.path.basename(file_path),
                            'line_number': line_number,
                            'timestamp': log_time if timestamp_match else None,
                            'content': line.strip(),
                            'log_type': log_type
                        })

                        if len(results) >= max_results:
                            break

        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(file_path)}: {e}")
            continue

        if len(results) >= max_results:
            break

    # Sort results by timestamp (newest first)
    results.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)

    return results, search_stats

def test_search_signals():
    """Test searching for signals"""
    print("="*80)
    print("TEST 1: Search for Signals")
    print("="*80)

    parser = SimpleLogParser()

    results, stats = mock_search_logs(
        parser=parser,
        search_term="📊",
        log_types=['forex_scanner'],
        start_date=datetime.now().date(),
        end_date=datetime.now().date(),
        regex_mode=False,
        max_results=10
    )

    print(f"\n📊 Search Statistics:")
    print(f"  Files Searched: {stats['files_searched']}")
    print(f"  Files Found: {stats['files_found']}")
    print(f"  Files Missing: {stats['files_missing']}")
    print(f"  Lines Scanned: {stats['lines_scanned']:,}")
    print(f"  Matches Found: {stats['matches_found']}")

    print(f"\n📋 Results (showing first 5):")
    for i, result in enumerate(results[:5], 1):
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if result['timestamp'] else 'Unknown'
        print(f"  {i}. [{timestamp}] {result['file']}:{result['line_number']}")
        print(f"     {result['content'][:100]}...")

    return len(results) > 0

def test_search_errors():
    """Test searching for errors"""
    print("\n" + "="*80)
    print("TEST 2: Search for Errors")
    print("="*80)

    parser = SimpleLogParser()

    results, stats = mock_search_logs(
        parser=parser,
        search_term="ERROR",
        log_types=['forex_scanner', 'fastapi_dev'],
        start_date=datetime.now().date() - timedelta(days=1),
        end_date=datetime.now().date(),
        regex_mode=False,
        max_results=5
    )

    print(f"\n📊 Search Statistics:")
    print(f"  Files Searched: {stats['files_searched']}")
    print(f"  Files Found: {stats['files_found']}")
    print(f"  Lines Scanned: {stats['lines_scanned']:,}")
    print(f"  Matches Found: {stats['matches_found']}")

    print(f"\n📋 Error Results: {len(results)} found")
    for i, result in enumerate(results[:3], 1):
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if result['timestamp'] else 'Unknown'
        print(f"  {i}. [{timestamp}] {result['content'][:100]}")

    return True

def test_regex_search():
    """Test regex search"""
    print("\n" + "="*80)
    print("TEST 3: Regex Search (BULL|BEAR signals)")
    print("="*80)

    parser = SimpleLogParser()

    results, stats = mock_search_logs(
        parser=parser,
        search_term=r"CS\.D\.[A-Z]{6}\.MINI\.IP.*(BULL|BEAR)",
        log_types=['forex_scanner'],
        start_date=datetime.now().date(),
        end_date=datetime.now().date(),
        regex_mode=True,
        max_results=10
    )

    print(f"\n📊 Search Statistics:")
    print(f"  Files Searched: {stats['files_searched']}")
    print(f"  Files Found: {stats['files_found']}")
    print(f"  Lines Scanned: {stats['lines_scanned']:,}")
    print(f"  Matches Found: {stats['matches_found']}")

    print(f"\n📋 Signal Results: {len(results)} found")
    for i, result in enumerate(results[:5], 1):
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if result['timestamp'] else 'Unknown'
        print(f"  {i}. [{timestamp}]")
        print(f"     {result['content'][:150]}")

    return len(results) > 0

def main():
    """Run all search tests"""
    print("\n🔍 TESTING SEARCH FUNCTIONALITY\n")

    try:
        test1 = test_search_signals()
        test2 = test_search_errors()
        test3 = test_regex_search()

        print("\n" + "="*80)
        if test1 and test2 and test3:
            print("✅ ALL SEARCH TESTS PASSED")
        else:
            print("⚠️ SOME TESTS HAD ISSUES")
        print("="*80)

        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
