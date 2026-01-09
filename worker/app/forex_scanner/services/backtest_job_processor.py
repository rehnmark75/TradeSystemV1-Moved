"""
Backtest Job Processor

Polls the backtest_job_queue table for pending jobs and executes them.
Run as a background process in the task-worker container.
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestJobProcessor:
    """
    Processes backtest jobs from the database queue.

    Polls backtest_job_queue for pending jobs and executes them
    using the backtest CLI.
    """

    def __init__(self,
                 db_host: str = 'postgres',
                 db_name: str = 'forex',
                 db_user: str = 'postgres',
                 db_password: str = 'postgres',
                 poll_interval: int = 5):
        """
        Initialize processor.

        Args:
            db_host: Database host
            db_name: Database name
            db_user: Database user
            db_password: Database password
            poll_interval: Seconds between queue polls
        """
        self.db_config = {
            'host': db_host,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }
        self.poll_interval = poll_interval
        self.running = False
        self._conn = None

    def _get_connection(self):
        """Get database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.db_config)
        return self._conn

    def _claim_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Claim the next pending job from the queue.

        Uses row-level locking to prevent race conditions.

        Returns:
            Job dict or None if no pending jobs
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Lock and claim the next pending job
                cur.execute("""
                    UPDATE backtest_job_queue
                    SET status = 'running', started_at = NOW()
                    WHERE id = (
                        SELECT id FROM backtest_job_queue
                        WHERE status = 'pending'
                        ORDER BY priority, submitted_at
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING *
                """)
                job = cur.fetchone()
                conn.commit()
                return dict(job) if job else None
        except Exception as e:
            logger.error(f"Error claiming job: {e}")
            conn.rollback()
            return None

    def _complete_job(self, job_id: int, execution_id: Optional[int] = None,
                      error: Optional[str] = None):
        """Mark job as completed or failed."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if error:
                    cur.execute("""
                        UPDATE backtest_job_queue
                        SET status = 'failed',
                            completed_at = NOW(),
                            error_message = %s
                        WHERE id = %s
                    """, (error, job_id))
                else:
                    cur.execute("""
                        UPDATE backtest_job_queue
                        SET status = 'completed',
                            completed_at = NOW(),
                            execution_id = %s
                        WHERE id = %s
                    """, (execution_id, job_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error completing job: {e}")
            conn.rollback()

    def _build_command(self, job: Dict[str, Any]) -> list:
        """Build CLI command from job configuration."""
        cmd = ['python', '/app/forex_scanner/bt.py']

        # Core params
        cmd.extend([job['epic'], str(job['days'])])

        # Strategy
        if job.get('strategy'):
            cmd.append(job['strategy'])

        # Timeframe
        if job.get('timeframe'):
            cmd.extend(['--timeframe', job['timeframe']])

        # Date range
        if job.get('start_date'):
            cmd.extend(['--start-date', str(job['start_date'])])
        if job.get('end_date'):
            cmd.extend(['--end-date', str(job['end_date'])])

        # Parallel execution
        if job.get('parallel'):
            cmd.append('--parallel')
            if job.get('workers'):
                cmd.extend(['--workers', str(job['workers'])])
            if job.get('chunk_days'):
                cmd.extend(['--chunk-days', str(job['chunk_days'])])

        # Chart generation
        if job.get('generate_chart', True):
            cmd.append('--chart')

        # Pipeline mode
        if job.get('pipeline_mode'):
            cmd.append('--pipeline')

        # Parameter overrides
        overrides = job.get('parameter_overrides', {})
        if isinstance(overrides, str):
            try:
                overrides = json.loads(overrides)
            except json.JSONDecodeError:
                overrides = {}

        for key, value in overrides.items():
            if value is not None:
                cmd.extend(['--override', f'{key}={value}'])

        # Snapshot
        if job.get('snapshot_name'):
            cmd.extend(['--snapshot', job['snapshot_name']])

        return cmd

    def _extract_execution_id(self, output: str) -> Optional[int]:
        """Extract execution_id from CLI output."""
        # Look for patterns like "Execution ID: 1234" or "execution_id=1234"
        import re

        patterns = [
            r'Execution ID:\s*(\d+)',
            r'execution_id[=:]\s*(\d+)',
            r'Saved execution #(\d+)',
            r'backtest_executions.*id=(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _get_latest_execution_id(self, epic: str) -> Optional[int]:
        """Get the most recent execution_id for this epic."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM backtest_executions
                    WHERE epics_tested::text LIKE %s
                    ORDER BY start_time DESC
                    LIMIT 1
                """, (f'%{epic}%',))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Error getting execution ID: {e}")
            return None

    def process_job(self, job: Dict[str, Any]) -> bool:
        """
        Process a single job.

        Args:
            job: Job dict from database

        Returns:
            True if successful, False otherwise
        """
        job_id = job['id']
        job_uid = job['job_id']
        epic = job['epic']

        logger.info(f"Processing job {job_uid}: {epic} {job['days']} days")

        try:
            cmd = self._build_command(job)
            logger.info(f"Command: {' '.join(cmd)}")

            # Execute backtest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Log output
            if result.stdout:
                logger.info(f"Job {job_uid} stdout:\n{result.stdout[-2000:]}")  # Last 2000 chars
            if result.stderr:
                logger.warning(f"Job {job_uid} stderr:\n{result.stderr[-1000:]}")

            # Check result
            if result.returncode == 0:
                # Try to extract execution_id
                execution_id = self._extract_execution_id(result.stdout)
                if not execution_id:
                    execution_id = self._get_latest_execution_id(epic)

                self._complete_job(job_id, execution_id=execution_id)
                logger.info(f"Job {job_uid} completed successfully (execution_id: {execution_id})")
                return True
            else:
                error = result.stderr or f"Exit code: {result.returncode}"
                self._complete_job(job_id, error=error[:1000])
                logger.error(f"Job {job_uid} failed: {error[:200]}")
                return False

        except subprocess.TimeoutExpired:
            self._complete_job(job_id, error="Job timed out after 1 hour")
            logger.error(f"Job {job_uid} timed out")
            return False

        except Exception as e:
            self._complete_job(job_id, error=str(e)[:1000])
            logger.error(f"Job {job_uid} error: {e}")
            return False

    def run(self):
        """Run the job processor loop."""
        logger.info("Starting backtest job processor...")
        self.running = True

        while self.running:
            try:
                # Check for pending jobs
                job = self._claim_next_job()

                if job:
                    self.process_job(job)
                else:
                    # No jobs, wait before checking again
                    time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                self.running = False

            except Exception as e:
                logger.error(f"Error in processor loop: {e}")
                time.sleep(self.poll_interval)

        logger.info("Job processor stopped")

    def stop(self):
        """Stop the processor."""
        self.running = False


def main():
    """Main entry point."""
    if psycopg2 is None:
        logger.error("psycopg2 not available")
        sys.exit(1)

    processor = BacktestJobProcessor(
        poll_interval=5  # Check every 5 seconds
    )
    processor.run()


if __name__ == '__main__':
    main()
