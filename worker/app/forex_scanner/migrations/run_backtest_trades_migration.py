#!/usr/bin/env python3
"""
Run Backtest Trades Table Migration
Creates the backtest_trades table and supporting structures
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from core.database import DatabaseManager
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner import config


def run_migration():
    """Run the backtest_trades table migration"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting backtest_trades table migration...")

    try:
        # Initialize database manager
        db_manager = DatabaseManager(config.DATABASE_URL)
        logger.info("✅ Database connection established")

        # Read the migration SQL file
        migration_file = script_dir / "create_backtest_trades_table.sql"

        if not migration_file.exists():
            logger.error(f"❌ Migration file not found: {migration_file}")
            return False

        with open(migration_file, 'r') as f:
            sql_content = f.read()

        logger.info("📄 Migration SQL file loaded")

        # Execute the migration
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                logger.info("🔄 Executing migration...")

                # Split by semicolon and execute each statement separately
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

                for i, statement in enumerate(statements, 1):
                    # Skip comments and empty statements
                    if statement.startswith('--') or not statement:
                        continue

                    try:
                        cursor.execute(statement)
                        logger.debug(f"   ✅ Statement {i} executed successfully")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Statement {i} warning: {e}")
                        # Continue with other statements
                        continue

                # Commit all changes
                conn.commit()
                logger.info("✅ Migration committed to database")

        # Verify the table was created
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name = 'backtest_trades'
                    AND table_schema = 'public'
                """)
                result = cursor.fetchone()

                if result:
                    logger.info("✅ backtest_trades table verified")

                    # Check if indexes were created
                    cursor.execute("""
                        SELECT indexname
                        FROM pg_indexes
                        WHERE tablename = 'backtest_trades'
                    """)
                    indexes = cursor.fetchall()
                    logger.info(f"✅ {len(indexes)} indexes created")

                    # Check if views were created
                    cursor.execute("""
                        SELECT viewname
                        FROM pg_views
                        WHERE viewname IN ('backtest_trade_summary', 'backtest_decision_analysis')
                    """)
                    views = cursor.fetchall()
                    logger.info(f"✅ {len(views)} views created")

                else:
                    logger.error("❌ backtest_trades table not found after migration")
                    return False

        logger.info("🎉 Migration completed successfully!")
        logger.info("📊 Backtest trades table is ready for historical scanner pipeline")
        return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)