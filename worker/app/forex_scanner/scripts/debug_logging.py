#!/usr/bin/env python3
"""
Force file logging to work by bypassing any existing handlers
"""

import os
import logging
import time
from logging.handlers import RotatingFileHandler

def force_file_logging_now():
    """Force setup file logging immediately, bypassing any existing setup"""
    
    print("🔧 FORCING file logging setup...")
    
    # Create logs directory
    log_dir = "/app/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger and FORCE clear all handlers
    root_logger = logging.getLogger()
    
    print(f"📊 Current handlers before cleanup: {len(root_logger.handlers)}")
    for i, handler in enumerate(root_logger.handlers):
        print(f"   {i}: {type(handler).__name__}")
    
    # FORCE remove all handlers
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
            root_logger.removeHandler(handler)
        except:
            pass
    
    print(f"📊 Handlers after cleanup: {len(root_logger.handlers)}")
    
    # Set level
    root_logger.setLevel(logging.DEBUG)
    
    # Create formatter with local time
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter.converter = time.localtime
    
    # FORCE create file handler
    log_file = os.path.join(log_dir, "forex-scanner.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # FORCE create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    print(f"📊 Handlers after setup: {len(root_logger.handlers)}")
    for i, handler in enumerate(root_logger.handlers):
        print(f"   {i}: {type(handler).__name__}")
    
    # Test logging immediately
    logger = logging.getLogger('force_test')
    logger.info("🚀 FORCED file logging test message")
    logger.error("❌ FORCED error test message")
    logger.warning("⚠️ FORCED warning test message")
    
    # Force flush
    for handler in root_logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    
    # Check if file was created
    if os.path.exists(log_file):
        file_size = os.path.getsize(log_file)
        print(f"✅ Log file created: {log_file} ({file_size} bytes)")
        
        if file_size > 0:
            with open(log_file, 'r') as f:
                content = f.read()
                print("📄 File contents:")
                for line in content.strip().split('\n'):
                    print(f"   {line}")
        else:
            print("❌ Log file created but empty")
    else:
        print(f"❌ Log file NOT created: {log_file}")
    
    return log_file

def patch_trade_scan_logging():
    """Create a patch file to fix trade_scan.py logging"""
    
    patch_code = '''
# FORCE LOGGING PATCH - Add this at the very beginning of trade_scan.py main() function

import os
import logging
import time
from logging.handlers import RotatingFileHandler

def force_setup_file_logging():
    """FORCE file logging setup"""
    log_dir = "/app/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    root_logger = logging.getLogger()
    
    # Only setup if no file handlers exist
    has_file_handler = any(isinstance(h, (RotatingFileHandler, logging.FileHandler)) for h in root_logger.handlers)
    
    if not has_file_handler:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter.converter = time.localtime
        
        log_file = os.path.join(log_dir, "forex-scanner.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        
        # Test it
        logging.getLogger('forced_patch').info("🔧 FORCED file logging patch applied")

# Call this BEFORE any other logging happens
force_setup_file_logging()
'''
    
    print("🔧 Creating logging patch...")
    print("Add this code to the very beginning of your trade_scan.py main() function:")
    print("=" * 60)
    print(patch_code)
    print("=" * 60)

def quick_fix_current_session():
    """Apply quick fix to current Python session"""
    print("🔧 Applying quick fix to current session...")
    
    # Force setup logging right now
    log_file = force_file_logging_now()
    
    # Test with some trading system messages
    logger = logging.getLogger('forex_scanner_test')
    logger.info("🚀 Forex Scanner file logging test")
    logger.info("📊 This should appear in the log file")
    logger.info("✅ File logging test completed")
    
    return log_file

def main():
    print("🔧 FORCE FILE LOGGING UTILITY")
    print("=" * 40)
    
    # Test 1: Force file logging now
    log_file = quick_fix_current_session()
    
    # Test 2: Show patch code
    patch_trade_scan_logging()
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Forced file logging setup applied")
    print(f"📄 Check log file: {log_file}")
    print(f"🔧 If this works, apply the patch to trade_scan.py")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Check: ls -la /app/logs/")
    print("2. Read: cat /app/logs/forex-scanner.log")
    print("3. If it works, the issue is handler setup timing in trade_scan.py")

if __name__ == "__main__":
    main()