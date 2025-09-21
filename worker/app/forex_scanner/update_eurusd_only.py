#!/usr/bin/env python3
"""
Update only EURUSD epic references from MINI to CEEM
All other pairs (EURJPY, EURGBP, etc.) should keep MINI format
"""

import os
import re
import sys

def update_file(file_path):
    """Update EURUSD epic references in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Only replace EURUSD MINI with CEEM
        # This pattern specifically targets EURUSD and not other EUR pairs
        content = re.sub(r'CS\.D\.EURUSD\.MINI\.IP', 'CS.D.EURUSD.CEEM.IP', content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to update files"""
    base_dir = "/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner"

    # Find all Python files that might contain EURUSD MINI references
    files_to_check = []

    for root, dirs, files in os.walk(base_dir):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue

        for file in files:
            if file.endswith(('.py', '.sql', '.md', '.txt', '.yml', '.yaml', '.json')):
                file_path = os.path.join(root, file)
                files_to_check.append(file_path)

    updated_files = []

    for file_path in files_to_check:
        # Check if file contains EURUSD MINI pattern
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'CS.D.EURUSD.CEEM.IP' in content:
                    if update_file(file_path):
                        updated_files.append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Updated {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"  - {file_path}")

    return len(updated_files)

if __name__ == "__main__":
    count = main()
    print(f"\nCompleted: Updated EURUSD MINI -> CEEM in {count} files")
    print("All other EUR pairs (EURJPY, EURGBP, etc.) remain with MINI format")