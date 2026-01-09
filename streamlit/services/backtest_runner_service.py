"""
Backtest Runner Service

Provides backtest submission and status monitoring.
Executes backtests directly via docker exec to task-worker container.
"""

import os
import subprocess
import re
import uuid
import threading
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime

try:
    from sqlalchemy import create_engine, text
except ImportError:
    create_engine = None
    text = None


# Store for background backtest results (in-memory fallback)
_background_results: Dict[str, Dict] = {}
_background_lock = threading.Lock()

# Store for real-time output capture
_background_output: Dict[str, List[str]] = {}
_output_lock = threading.Lock()

# File-based state for cross-process persistence
import json
import tempfile
import os

_STATE_DIR = os.path.join(tempfile.gettempdir(), 'backtest_jobs')
os.makedirs(_STATE_DIR, exist_ok=True)


def _save_job_state(job_id: str, state: dict):
    """Save job state to file for cross-process access"""
    filepath = os.path.join(_STATE_DIR, f'{job_id}.json')
    try:
        with open(filepath, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass  # Silent fail for state persistence


def _load_job_state(job_id: str) -> Optional[dict]:
    """Load job state from file"""
    filepath = os.path.join(_STATE_DIR, f'{job_id}.json')
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _save_job_output(job_id: str, lines: List[str]):
    """Save recent output to file"""
    filepath = os.path.join(_STATE_DIR, f'{job_id}_output.json')
    try:
        with open(filepath, 'w') as f:
            json.dump(lines[-50:], f)  # Keep last 50 lines
    except Exception:
        pass


def _load_job_output(job_id: str) -> List[str]:
    """Load recent output from file"""
    filepath = os.path.join(_STATE_DIR, f'{job_id}_output.json')
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _cleanup_job_files(job_id: str):
    """Remove job state files after completion"""
    for suffix in ['.json', '_output.json']:
        filepath = os.path.join(_STATE_DIR, f'{job_id}{suffix}')
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


class BacktestRunnerService:
    """Service for submitting and monitoring backtest runs from Streamlit"""

    # Available currency pairs (shortname, full epic)
    EPIC_OPTIONS: List[Tuple[str, str]] = [
        ('EURUSD', 'CS.D.EURUSD.MINI.IP'),
        ('GBPUSD', 'CS.D.GBPUSD.MINI.IP'),
        ('USDJPY', 'CS.D.USDJPY.MINI.IP'),
        ('AUDUSD', 'CS.D.AUDUSD.MINI.IP'),
        ('USDCHF', 'CS.D.USDCHF.MINI.IP'),
        ('USDCAD', 'CS.D.USDCAD.MINI.IP'),
        ('NZDUSD', 'CS.D.NZDUSD.MINI.IP'),
        ('EURJPY', 'CS.D.EURJPY.MINI.IP'),
        ('AUDJPY', 'CS.D.AUDJPY.MINI.IP'),
        ('GBPJPY', 'CS.D.GBPJPY.MINI.IP'),
    ]

    # Strategy options
    STRATEGY_OPTIONS = ['SMC_SIMPLE']

    # Timeframe options
    TIMEFRAME_OPTIONS = ['5m', '15m', '30m', '1h', '4h']

    def __init__(self):
        self.database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/forex'
        )
        self._engine = None

    @property
    def engine(self):
        """Lazy-load database engine."""
        if self._engine is None and create_engine:
            self._engine = create_engine(self.database_url)
        return self._engine

    def build_command(self, config: dict) -> list:
        """
        Build CLI command from configuration.

        Args:
            config: Dictionary with backtest configuration

        Returns:
            List of command arguments
        """
        # Use -u for unbuffered Python output to get real-time progress
        cmd = ['docker', 'exec', 'task-worker', 'python', '-u', '/app/forex_scanner/bt.py']

        # Core params (required)
        cmd.extend([config['epic'], str(config['days'])])

        # Strategy (optional, positional)
        if config.get('strategy'):
            cmd.append(config['strategy'])

        # Timeframe
        if config.get('timeframe'):
            cmd.extend(['--timeframe', config['timeframe']])

        # Date range (alternative to days)
        if config.get('start_date'):
            cmd.extend(['--start-date', config['start_date']])
        if config.get('end_date'):
            cmd.extend(['--end-date', config['end_date']])

        # Parallel execution
        if config.get('parallel'):
            cmd.append('--parallel')
            cmd.extend(['--workers', str(config.get('workers', 4))])
            if config.get('chunk_days'):
                cmd.extend(['--chunk-days', str(config['chunk_days'])])

        # Chart generation (enabled by default)
        if config.get('chart', True):
            cmd.append('--chart')

        # Pipeline mode
        if config.get('pipeline'):
            cmd.append('--pipeline')

        # Parameter overrides
        overrides = config.get('overrides', {})
        for key, value in overrides.items():
            if value is not None:
                cmd.extend(['--override', f'{key}={value}'])

        # Snapshot
        if config.get('snapshot') and config['snapshot'] != 'None':
            cmd.extend(['--snapshot', config['snapshot']])

        # Historical intelligence (default is OFF, so only add flag when enabled)
        if config.get('use_historical_intelligence'):
            cmd.append('--use-historical-intelligence')

        # Parameter variation mode
        variation = config.get('variation', {})
        if variation.get('enabled') and variation.get('param_grid'):
            param_grid = variation['param_grid']

            # Convert param_grid to JSON format
            import json
            vary_json = json.dumps(param_grid)
            cmd.extend(['--vary-json', vary_json])

            # Add variation options
            if variation.get('workers'):
                cmd.extend(['--vary-workers', str(variation['workers'])])
            if variation.get('rank_by'):
                cmd.extend(['--rank-by', variation['rank_by']])
            if variation.get('top_n'):
                cmd.extend(['--top-n', str(variation['top_n'])])

        return cmd

    def _parse_backtest_output(self, stdout: str) -> Dict[str, Any]:
        """Parse backtest CLI output to extract results"""
        result = {
            'signal_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0.0,
            'total_pips': 0.0,
            'execution_id': None,
            'chart_url': None,
        }

        lines = stdout.split('\n')

        for line in lines:
            line_lower = line.lower()

            # Look for signal count
            if 'signal' in line_lower and ('total' in line_lower or 'count' in line_lower or 'generated' in line_lower):
                match = re.search(r'(\d+)\s*signal', line_lower)
                if match:
                    result['signal_count'] = int(match.group(1))

            # Look for win rate
            if 'win rate' in line_lower or 'winrate' in line_lower:
                match = re.search(r'(\d+\.?\d*)\s*%', line)
                if match:
                    result['win_rate'] = float(match.group(1))

            # Look for total pips
            if 'total pips' in line_lower or 'pips:' in line_lower:
                match = re.search(r'[-+]?(\d+\.?\d*)', line)
                if match:
                    result['total_pips'] = float(match.group(1))

            # Look for execution ID
            if 'execution' in line_lower and 'id' in line_lower:
                match = re.search(r'id[:\s]+(\d+)', line_lower)
                if match:
                    result['execution_id'] = int(match.group(1))

            # Look for chart URL
            if 'chart' in line_lower and ('url' in line_lower or 'http' in line_lower or 'minio' in line_lower):
                match = re.search(r'(https?://\S+|/backtest-charts/\S+)', line)
                if match:
                    result['chart_url'] = match.group(1)

            # Look for wins/losses
            if 'win' in line_lower:
                match = re.search(r'(\d+)\s*win', line_lower)
                if match:
                    result['win_count'] = int(match.group(1))

            if 'loss' in line_lower:
                match = re.search(r'(\d+)\s*loss', line_lower)
                if match:
                    result['loss_count'] = int(match.group(1))

        # Calculate win rate if not found but we have wins/losses
        if result['win_rate'] == 0.0 and (result['win_count'] > 0 or result['loss_count'] > 0):
            total = result['win_count'] + result['loss_count']
            if total > 0:
                result['win_rate'] = (result['win_count'] / total) * 100

        return result

    def _run_backtest_in_background(self, job_id: str, config: dict):
        """Run backtest in a background thread with real-time output capture"""
        started_at = datetime.now().isoformat()
        try:
            start_time = datetime.now()
            cmd = self.build_command(config)

            # Initialize output buffer
            with _output_lock:
                _background_output[job_id] = []

            # Use Popen for real-time output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )

            stdout_lines = []
            stderr_lines = []

            # Read output in real-time
            while True:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    stdout_lines.append(line)

                    # Update progress in shared state
                    with _output_lock:
                        _background_output[job_id].append(line)
                        # Keep only last 50 lines to save memory
                        if len(_background_output[job_id]) > 50:
                            _background_output[job_id] = _background_output[job_id][-50:]
                        # Also save to file for cross-process access
                        _save_job_output(job_id, _background_output[job_id])

                    # Update progress state
                    self._update_progress(job_id, line, start_time, config, started_at)

                elif process.poll() is not None:
                    break

            # Get any remaining stderr
            stderr = process.stderr.read()
            if stderr:
                stderr_lines = stderr.strip().split('\n')

            returncode = process.returncode
            duration = (datetime.now() - start_time).total_seconds()

            # Parse final output
            full_stdout = '\n'.join(stdout_lines)
            parsed = self._parse_backtest_output(full_stdout)

            final_state = {
                'status': 'completed' if returncode == 0 else 'failed',
                'started_at': started_at,
                'completed_at': datetime.now().isoformat(),
                'config': config,
                'result': {
                    'success': returncode == 0,
                    'duration_seconds': duration,
                    'stdout': full_stdout if returncode != 0 else None,
                    'error': '\n'.join(stderr_lines) if returncode != 0 else None,
                    **parsed
                }
            }

            with _background_lock:
                _background_results[job_id] = final_state

            # Save final state to file
            _save_job_state(job_id, final_state)

            # Cleanup output buffer after completion
            with _output_lock:
                if job_id in _background_output:
                    del _background_output[job_id]

        except Exception as e:
            error_state = {
                'status': 'failed',
                'started_at': started_at,
                'completed_at': datetime.now().isoformat(),
                'config': config,
                'result': {
                    'success': False,
                    'error': str(e)
                }
            }
            with _background_lock:
                _background_results[job_id] = error_state
            _save_job_state(job_id, error_state)

    def _update_progress(self, job_id: str, line: str, start_time: datetime, config: dict = None, started_at: str = None):
        """Update job progress based on output line"""
        elapsed = (datetime.now() - start_time).total_seconds()

        # Parse progress indicators from output
        progress_info = {
            'elapsed_seconds': elapsed,
            'last_activity': line[:100] if line else None,  # Truncate long lines
        }

        # Look for specific progress patterns
        if 'Loading' in line or 'Fetching' in line:
            progress_info['phase'] = 'loading_data'
        elif 'Processing' in line:
            progress_info['phase'] = 'processing'
        elif 'variation' in line.lower() or '[' in line and '/' in line:
            # Parse variation progress like "[3/12]"
            match = re.search(r'\[(\d+)/(\d+)\]', line)
            if match:
                progress_info['phase'] = 'running_variations'
                progress_info['current'] = int(match.group(1))
                progress_info['total'] = int(match.group(2))
        elif 'signal' in line.lower():
            progress_info['phase'] = 'analyzing_signals'
        elif 'chart' in line.lower():
            progress_info['phase'] = 'generating_chart'
        elif '✅' in line or '✓' in line:
            progress_info['phase'] = 'completing'

        # Build current state
        current_state = {
            'status': 'running',
            'started_at': started_at or datetime.now().isoformat(),
            'config': config or {},
            'progress': progress_info
        }

        # Update progress in result (in-memory)
        with _background_lock:
            if job_id in _background_results:
                _background_results[job_id]['progress'] = progress_info
            else:
                _background_results[job_id] = current_state

        # Also save to file for cross-process access
        _save_job_state(job_id, current_state)

    def submit_backtest(self, config: dict, async_mode: bool = True) -> dict:
        """
        Submit backtest for execution.

        Args:
            config: Dictionary with backtest configuration
            async_mode: If True, runs in background; if False, waits for completion

        Returns:
            Dict with success status and job details
        """
        try:
            # Build command
            cmd = self.build_command(config)

            if async_mode:
                # Generate job ID and run in background
                job_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

                with _background_lock:
                    _background_results[job_id] = {
                        'status': 'running',
                        'started_at': datetime.now().isoformat(),
                        'config': config,
                        'result': None
                    }

                # Start background thread
                thread = threading.Thread(
                    target=self._run_backtest_in_background,
                    args=(job_id, config),
                    daemon=True
                )
                thread.start()

                return {
                    'success': True,
                    'job_id': job_id,
                    'status': 'running',
                    'command': ' '.join(cmd),
                    'message': 'Backtest started. Use Check Status to monitor progress.'
                }
            else:
                # Synchronous execution
                start_time = datetime.now()

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )

                duration = (datetime.now() - start_time).total_seconds()
                parsed = self._parse_backtest_output(result.stdout)

                return {
                    'success': result.returncode == 0,
                    'job_id': None,
                    'result': {
                        'success': result.returncode == 0,
                        'duration_seconds': duration,
                        'stdout': result.stdout if result.returncode != 0 else None,
                        'error': result.stderr if result.returncode != 0 else None,
                        **parsed
                    },
                    'command': ' '.join(cmd),
                    'message': 'Backtest completed.'
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Backtest timed out after 10 minutes'
            }

        except FileNotFoundError:
            return {
                'success': False,
                'error': 'Docker not found. Make sure docker is installed and accessible.'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_job_status(self, job_id: str) -> dict:
        """
        Get status of a backtest job.

        Args:
            job_id: Job ID from submit_backtest

        Returns:
            Dict with job status details
        """
        # First try in-memory store
        with _background_lock:
            if job_id in _background_results:
                result = _background_results[job_id].copy()

                # Add recent output if job is still running
                if result.get('status') == 'running':
                    with _output_lock:
                        if job_id in _background_output:
                            result['recent_output'] = _background_output[job_id][-10:]  # Last 10 lines
                        else:
                            # Try loading from file
                            result['recent_output'] = _load_job_output(job_id)[-10:]

                return result

        # Try loading from file (cross-process access)
        file_state = _load_job_state(job_id)
        if file_state:
            # Add recent output if running
            if file_state.get('status') == 'running':
                file_state['recent_output'] = _load_job_output(job_id)[-10:]
            return file_state

        return {'error': f'Job {job_id} not found'}

    def get_pending_jobs(self) -> pd.DataFrame:
        """Get all pending/running jobs."""
        with _background_lock:
            running = [
                {'job_id': k, **v}
                for k, v in _background_results.items()
                if v.get('status') == 'running'
            ]
        if running:
            return pd.DataFrame(running)
        return pd.DataFrame()

    def get_backtest_status(self, execution_id: int) -> dict:
        """
        Get detailed status of a backtest execution from database.

        Args:
            execution_id: Backtest execution ID

        Returns:
            Dict with status details or error
        """
        if not self.engine:
            return {'error': 'Database not available'}

        query = """
        SELECT
            be.id,
            be.status,
            be.start_time,
            be.end_time,
            be.total_candles_processed,
            be.execution_duration_seconds,
            be.chart_url,
            be.epics_tested,
            be.strategy_name,
            be.data_start_date,
            be.data_end_date,
            COALESCE(bs.signal_count, 0) as signal_count,
            COALESCE(bs.win_count, 0) as win_count,
            COALESCE(bs.loss_count, 0) as loss_count,
            COALESCE(bs.total_pips, 0) as total_pips
        FROM backtest_executions be
        LEFT JOIN (
            SELECT
                execution_id,
                COUNT(*) as signal_count,
                SUM(CASE WHEN pips_gained > 0 THEN 1 ELSE 0 END) as win_count,
                SUM(CASE WHEN pips_gained <= 0 THEN 1 ELSE 0 END) as loss_count,
                COALESCE(SUM(pips_gained), 0) as total_pips
            FROM backtest_signals
            GROUP BY execution_id
        ) bs ON be.id = bs.execution_id
        WHERE be.id = :exec_id
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'exec_id': execution_id})
                row = result.fetchone()
                if row:
                    status = dict(row._mapping)
                    # Calculate win rate
                    if status['signal_count'] > 0:
                        status['win_rate'] = (status['win_count'] / status['signal_count']) * 100
                    else:
                        status['win_rate'] = 0
                    return status
                return {'error': f'Execution {execution_id} not found'}
        except Exception as e:
            return {'error': str(e)}

    def get_recent_backtests(self, limit: int = 5) -> pd.DataFrame:
        """
        Get recent backtest executions.

        Args:
            limit: Maximum number of results

        Returns:
            DataFrame with recent backtests
        """
        if not self.engine:
            return pd.DataFrame()

        query = """
        SELECT
            id,
            status,
            strategy_name,
            epics_tested,
            start_time,
            execution_duration_seconds,
            data_start_date,
            data_end_date
        FROM backtest_executions
        ORDER BY start_time DESC
        LIMIT :limit
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'limit': limit})
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception:
            return pd.DataFrame()

    def get_available_snapshots(self) -> List[str]:
        """
        Get list of available configuration snapshots.

        Returns:
            List of snapshot names
        """
        # TODO: Query from config_snapshots table when implemented
        return []

    @staticmethod
    def get_default_overrides() -> Dict[str, Any]:
        """
        Get default parameter override values.

        Returns:
            Dict with default values for each override parameter
        """
        return {
            # Risk Management
            'fixed_stop_loss_pips': 9.0,
            'fixed_take_profit_pips': 15.0,
            'sl_buffer_pips': 1.5,
            'min_risk_reward': 1.5,
            'max_position_size': 1.0,
            'use_atr_stop_loss': False,

            # Entry Filters
            'min_confidence': 0.48,
            'max_confidence': 0.90,
            'ema_period': 50,
            'swing_lookback_bars': 20,
            'macd_filter_enabled': True,
            'volume_filter_enabled': False,
            'require_ema_alignment': True,
            'trend_filter_enabled': True,

            # Session Filters
            'block_asian_session': True,
            'london_open_hour': 7,
            'ny_open_hour': 13,
            'weekend_filter_enabled': True,
            'session_end_buffer_minutes': 30,
            'high_impact_news_filter': False,

            # Advanced
            'signal_cooldown_minutes': 60,
            'max_daily_signals': 5,
            'require_liquidity_sweep': False,
            'fvg_minimum_size_pips': 3.0,
            'displacement_atr_multiplier': 1.5,
        }

    def check_fastapi_health(self) -> dict:
        """Check if docker and task-worker are available."""
        try:
            # Check if task-worker container is running
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=task-worker', '--format', '{{.Status}}'],
                capture_output=True,
                text=True,
                timeout=5
            )

            container_running = 'Up' in result.stdout

            with _background_lock:
                pending_count = len([j for j in _background_results.values() if j['status'] == 'running'])

            return {
                'status': 'healthy' if container_running else 'degraded',
                'task_worker_running': container_running,
                'pending_jobs': pending_count
            }

        except FileNotFoundError:
            return {
                'status': 'unhealthy',
                'error': 'Docker not found'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
