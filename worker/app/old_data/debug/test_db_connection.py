#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

try:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner import config
    
    print('🔍 Testing database connection...')
    
    # Get database URL from config
    db_url = getattr(config, 'DATABASE_URL', None)
    if not db_url:
        print('❌ DATABASE_URL not found in config')
        sys.exit(1)
        
    print(f'📊 Database URL configured: {db_url[:30]}...')
    
    # Test DatabaseManager
    db_manager = DatabaseManager(db_url)
    print('✅ DatabaseManager initialized successfully')
    
    # Test engine connection
    with db_manager.get_engine().connect() as conn:
        result = conn.execute(text("SELECT 1 as test_value"))
        row = result.fetchone()
        print(f'✅ Engine connection test passed: {row[0]}')
    
    # Test basic query
    from sqlalchemy import text
    with db_manager.get_engine().connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) as count FROM ig_candles LIMIT 1"))
        row = result.fetchone()
        print(f'✅ Database query test passed: {row[0]} total candles')
    
    print('🎉 Database connection is working correctly!')
    
except Exception as e:
    print(f'❌ Database connection test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
