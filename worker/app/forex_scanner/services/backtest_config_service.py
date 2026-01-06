# services/backtest_config_service.py
"""
Backtest Config Service - Manages config snapshots for backtest parameter isolation

Provides CRUD operations for named parameter configurations that can be:
1. Created from current config with overrides
2. Used across multiple backtest runs
3. Compared side-by-side
4. Promoted to live trading after validation
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner import config
except ImportError:
    from core.database import DatabaseManager
    import config


@dataclass
class ConfigSnapshot:
    """Represents a saved parameter configuration snapshot"""
    id: int
    snapshot_name: str
    description: Optional[str]
    parameter_overrides: Dict[str, Any]
    base_config_version: Optional[str]
    created_at: datetime
    updated_at: datetime
    created_by: str
    last_tested_at: Optional[datetime]
    test_results: Optional[Dict]
    test_count: int
    is_promoted: bool
    is_active: bool
    tags: List[str] = field(default_factory=list)


class BacktestConfigService:
    """
    Service for managing backtest configuration snapshots

    Provides:
    - Create snapshots with parameter overrides
    - List and retrieve snapshots
    - Update snapshot metadata after tests
    - Promote snapshots to live configuration
    - Delete/deactivate snapshots
    """

    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager(config.DATABASE_URL)
        self.logger = logging.getLogger(__name__)
        self._config_db_url = getattr(config, 'STRATEGY_CONFIG_DATABASE_URL', config.DATABASE_URL)

    def _get_config_db_connection(self):
        """Get connection to strategy_config database"""
        import psycopg2
        return psycopg2.connect(self._config_db_url)

    def create_snapshot(
        self,
        name: str,
        parameter_overrides: Dict[str, Any],
        description: Optional[str] = None,
        created_by: str = "cli",
        tags: List[str] = None
    ) -> Optional[int]:
        """
        Create a new configuration snapshot

        Args:
            name: Unique name for the snapshot
            parameter_overrides: Dict of parameter overrides
            description: Optional description
            created_by: Who created this snapshot
            tags: Optional list of tags for organization

        Returns:
            Snapshot ID if created, None if failed
        """
        conn = None
        try:
            conn = self._get_config_db_connection()
            cursor = conn.cursor()

            # Get current base config version (optional - don't fail if table doesn't exist)
            base_version = None
            try:
                cursor.execute("""
                    SELECT version FROM smc_simple_global_config
                    WHERE is_active = TRUE LIMIT 1
                """)
                result = cursor.fetchone()
                base_version = result[0] if result else None
            except Exception:
                # Table may not exist - that's OK, rollback the failed query and continue
                conn.rollback()

            cursor.execute("""
                INSERT INTO smc_backtest_snapshots (
                    snapshot_name, description, parameter_overrides,
                    base_config_version, created_by, tags
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                name,
                description,
                json.dumps(parameter_overrides),
                base_version,
                created_by,
                tags or []
            ))

            snapshot_id = cursor.fetchone()[0]
            conn.commit()

            self.logger.info(f"✅ Created snapshot '{name}' (ID: {snapshot_id}) with {len(parameter_overrides)} overrides")
            return snapshot_id

        except Exception as e:
            if conn:
                conn.rollback()
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                self.logger.error(f"❌ Snapshot '{name}' already exists")
            else:
                self.logger.error(f"❌ Error creating snapshot: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def get_snapshot(self, name: str) -> Optional[ConfigSnapshot]:
        """
        Get a snapshot by name

        Args:
            name: Snapshot name

        Returns:
            ConfigSnapshot if found, None otherwise
        """
        conn = None
        try:
            conn = self._get_config_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    id, snapshot_name, description, parameter_overrides,
                    base_config_version, created_at, updated_at, created_by,
                    last_tested_at, test_results, test_count,
                    is_promoted, is_active, tags
                FROM smc_backtest_snapshots
                WHERE snapshot_name = %s AND is_active = TRUE
            """, (name,))

            row = cursor.fetchone()
            if not row:
                return None

            return ConfigSnapshot(
                id=row[0],
                snapshot_name=row[1],
                description=row[2],
                parameter_overrides=row[3] if isinstance(row[3], dict) else json.loads(row[3] or '{}'),
                base_config_version=row[4],
                created_at=row[5],
                updated_at=row[6],
                created_by=row[7],
                last_tested_at=row[8],
                test_results=row[9] if isinstance(row[9], dict) else json.loads(row[9] or '{}') if row[9] else None,
                test_count=row[10] or 0,
                is_promoted=row[11] or False,
                is_active=row[12],
                tags=row[13] or []
            )

        except Exception as e:
            self.logger.error(f"❌ Error getting snapshot '{name}': {e}")
            return None
        finally:
            if conn:
                conn.close()

    def get_snapshot_overrides(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get just the parameter overrides for a snapshot (for use in backtests)

        Args:
            name: Snapshot name

        Returns:
            Dict of parameter overrides if found, None otherwise
        """
        snapshot = self.get_snapshot(name)
        return snapshot.parameter_overrides if snapshot else None

    def list_snapshots(
        self,
        include_inactive: bool = False,
        tags: List[str] = None,
        limit: int = 50
    ) -> List[ConfigSnapshot]:
        """
        List all snapshots

        Args:
            include_inactive: Include deactivated snapshots
            tags: Filter by tags (any match)
            limit: Maximum number of results

        Returns:
            List of ConfigSnapshot objects
        """
        conn = None
        try:
            conn = self._get_config_db_connection()
            cursor = conn.cursor()

            query = """
                SELECT
                    id, snapshot_name, description, parameter_overrides,
                    base_config_version, created_at, updated_at, created_by,
                    last_tested_at, test_results, test_count,
                    is_promoted, is_active, tags
                FROM smc_backtest_snapshots
                WHERE 1=1
            """
            params = []

            if not include_inactive:
                query += " AND is_active = TRUE"

            if tags:
                query += " AND tags && %s"
                params.append(tags)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)

            snapshots = []
            for row in cursor.fetchall():
                snapshots.append(ConfigSnapshot(
                    id=row[0],
                    snapshot_name=row[1],
                    description=row[2],
                    parameter_overrides=row[3] if isinstance(row[3], dict) else json.loads(row[3] or '{}'),
                    base_config_version=row[4],
                    created_at=row[5],
                    updated_at=row[6],
                    created_by=row[7],
                    last_tested_at=row[8],
                    test_results=row[9] if isinstance(row[9], dict) else json.loads(row[9] or '{}') if row[9] else None,
                    test_count=row[10] or 0,
                    is_promoted=row[11] or False,
                    is_active=row[12],
                    tags=row[13] or []
                ))

            return snapshots

        except Exception as e:
            self.logger.error(f"❌ Error listing snapshots: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def update_test_results(
        self,
        name: str,
        execution_id: int,
        results: Dict[str, Any],
        epic_tested: str = None,
        days_tested: int = None
    ) -> bool:
        """
        Update snapshot with test results after a backtest run

        Args:
            name: Snapshot name
            execution_id: Backtest execution ID
            results: Test results dict (win_rate, profit_factor, etc.)
            epic_tested: Epic that was tested
            days_tested: Number of days tested

        Returns:
            True if updated, False otherwise
        """
        conn = None
        try:
            conn = self._get_config_db_connection()
            cursor = conn.cursor()

            # Update snapshot main record
            cursor.execute("""
                UPDATE smc_backtest_snapshots
                SET last_tested_at = NOW(),
                    last_test_execution_id = %s,
                    test_results = %s,
                    test_count = test_count + 1
                WHERE snapshot_name = %s AND is_active = TRUE
                RETURNING id
            """, (
                execution_id,
                json.dumps(results),
                name
            ))

            result = cursor.fetchone()
            if not result:
                self.logger.error(f"❌ Snapshot '{name}' not found")
                return False

            snapshot_id = result[0]

            # Insert into test history
            cursor.execute("""
                INSERT INTO smc_snapshot_test_history (
                    snapshot_id, execution_id, epic_tested, days_tested,
                    total_signals, win_rate, profit_factor, total_pips,
                    avg_profit_pips, avg_loss_pips
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                snapshot_id,
                execution_id,
                epic_tested,
                days_tested,
                results.get('total_signals', 0),
                results.get('win_rate'),
                results.get('profit_factor'),
                results.get('total_pips'),
                results.get('avg_profit_pips'),
                results.get('avg_loss_pips')
            ))

            conn.commit()
            self.logger.info(f"✅ Updated test results for snapshot '{name}'")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"❌ Error updating test results: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def delete_snapshot(self, name: str, hard_delete: bool = False) -> bool:
        """
        Delete or deactivate a snapshot

        Args:
            name: Snapshot name
            hard_delete: If True, permanently delete; otherwise soft delete

        Returns:
            True if deleted, False otherwise
        """
        conn = None
        try:
            conn = self._get_config_db_connection()
            cursor = conn.cursor()

            if hard_delete:
                cursor.execute("""
                    DELETE FROM smc_backtest_snapshots
                    WHERE snapshot_name = %s
                    RETURNING id
                """, (name,))
            else:
                cursor.execute("""
                    UPDATE smc_backtest_snapshots
                    SET is_active = FALSE, updated_at = NOW()
                    WHERE snapshot_name = %s AND is_active = TRUE
                    RETURNING id
                """, (name,))

            result = cursor.fetchone()
            if not result:
                self.logger.error(f"❌ Snapshot '{name}' not found")
                return False

            conn.commit()
            action = "Deleted" if hard_delete else "Deactivated"
            self.logger.info(f"✅ {action} snapshot '{name}'")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"❌ Error deleting snapshot: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def promote_to_live(
        self,
        name: str,
        promoted_by: str = "cli",
        notes: str = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Promote snapshot parameters to live configuration

        Args:
            name: Snapshot name
            promoted_by: Who is promoting
            notes: Promotion notes
            dry_run: If True, only show what would change

        Returns:
            Dict with promotion status and changes
        """
        snapshot = self.get_snapshot(name)
        if not snapshot:
            return {'success': False, 'error': f"Snapshot '{name}' not found"}

        if snapshot.test_count < 1:
            return {
                'success': False,
                'error': f"Snapshot '{name}' has never been tested. Run at least one backtest first."
            }

        overrides = snapshot.parameter_overrides
        if not overrides:
            return {'success': False, 'error': "Snapshot has no parameter overrides"}

        result = {
            'snapshot_name': name,
            'parameters_to_change': list(overrides.keys()),
            'overrides': overrides,
            'test_count': snapshot.test_count,
            'last_test_results': snapshot.test_results,
            'dry_run': dry_run
        }

        if dry_run:
            result['message'] = "DRY RUN: No changes made. Remove --dry-run to apply."
            return result

        # Actually apply changes to live config
        conn = None
        try:
            conn = self._get_config_db_connection()
            cursor = conn.cursor()

            # Update each parameter in smc_simple_global_config
            for param, value in overrides.items():
                cursor.execute(f"""
                    UPDATE smc_simple_global_config
                    SET {param} = %s, updated_at = NOW()
                    WHERE is_active = TRUE
                """, (value,))

            # Mark snapshot as promoted
            cursor.execute("""
                UPDATE smc_backtest_snapshots
                SET is_promoted = TRUE,
                    promoted_to_live_at = NOW(),
                    promoted_by = %s,
                    promotion_notes = %s
                WHERE snapshot_name = %s
            """, (promoted_by, notes, name))

            conn.commit()

            result['success'] = True
            result['message'] = f"✅ Promoted {len(overrides)} parameters to live configuration"
            self.logger.info(result['message'])
            return result

        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"❌ Error promoting snapshot: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            if conn:
                conn.close()

    def get_test_history(self, name: str, limit: int = 20) -> List[Dict]:
        """
        Get test history for a snapshot

        Args:
            name: Snapshot name
            limit: Maximum results

        Returns:
            List of test result dicts
        """
        conn = None
        try:
            conn = self._get_config_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    h.execution_id, h.epic_tested, h.days_tested,
                    h.total_signals, h.win_rate, h.profit_factor,
                    h.total_pips, h.tested_at
                FROM smc_snapshot_test_history h
                JOIN smc_backtest_snapshots s ON h.snapshot_id = s.id
                WHERE s.snapshot_name = %s
                ORDER BY h.tested_at DESC
                LIMIT %s
            """, (name, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'execution_id': row[0],
                    'epic_tested': row[1],
                    'days_tested': row[2],
                    'total_signals': row[3],
                    'win_rate': float(row[4]) if row[4] else None,
                    'profit_factor': float(row[5]) if row[5] else None,
                    'total_pips': float(row[6]) if row[6] else None,
                    'tested_at': row[7]
                })

            return results

        except Exception as e:
            self.logger.error(f"❌ Error getting test history: {e}")
            return []
        finally:
            if conn:
                conn.close()


# Singleton instance
_service_instance = None


def get_backtest_config_service() -> BacktestConfigService:
    """Get singleton instance of BacktestConfigService"""
    global _service_instance
    if _service_instance is None:
        _service_instance = BacktestConfigService()
    return _service_instance
