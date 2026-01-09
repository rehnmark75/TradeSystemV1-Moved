"""
Backtest Router - FastAPI endpoints for triggering backtests directly

Provides synchronous backtest execution via HTTP POST.
Replaces the database queue approach for immediate execution.
"""

import logging
import subprocess
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


class BacktestConfig(BaseModel):
    """Configuration for a backtest run"""
    epic: str = Field(..., description="Currency pair (e.g., EURUSD, GBPUSD)")
    days: int = Field(default=14, ge=1, le=365, description="Number of days to backtest")
    strategy: str = Field(default="SMC_SIMPLE", description="Strategy name")
    timeframe: str = Field(default="15m", description="Timeframe (5m, 15m, 30m, 1h, 4h)")

    # Execution options
    parallel: bool = Field(default=False, description="Enable parallel execution")
    workers: Optional[int] = Field(default=4, ge=2, le=8, description="Worker count for parallel")
    chunk_days: Optional[int] = Field(default=7, ge=1, le=30, description="Chunk size in days")
    generate_chart: bool = Field(default=True, description="Generate chart")
    pipeline_mode: bool = Field(default=False, description="Full pipeline mode")

    # Parameter overrides
    overrides: Optional[Dict[str, Any]] = Field(default=None, description="Strategy parameter overrides")

    # Snapshot
    snapshot: Optional[str] = Field(default=None, description="Snapshot name to load")

    # Date range (alternative to days)
    start_date: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")

    # Historical intelligence
    use_historical_intelligence: bool = Field(
        default=False,
        description="Use stored market intelligence from database (default: False)"
    )


class BacktestResult(BaseModel):
    """Result of a backtest execution"""
    success: bool
    execution_id: Optional[int] = None
    signal_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    duration_seconds: float = 0.0
    chart_url: Optional[str] = None
    error: Optional[str] = None
    stdout: Optional[str] = None


# Epic mapping from shortname to full IG epic
EPIC_MAPPING = {
    'EURUSD': 'CS.D.EURUSD.MINI.IP',
    'GBPUSD': 'CS.D.GBPUSD.MINI.IP',
    'USDJPY': 'CS.D.USDJPY.MINI.IP',
    'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
    'USDCHF': 'CS.D.USDCHF.MINI.IP',
    'USDCAD': 'CS.D.USDCAD.MINI.IP',
    'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
    'EURJPY': 'CS.D.EURJPY.MINI.IP',
    'AUDJPY': 'CS.D.AUDJPY.MINI.IP',
    'GBPJPY': 'CS.D.GBPJPY.MINI.IP',
}


def _build_backtest_command(config: BacktestConfig) -> List[str]:
    """Build the docker exec command for backtest"""

    # Map epic shortname to full epic if needed
    epic = config.epic
    if epic in EPIC_MAPPING:
        epic = EPIC_MAPPING[epic]
    elif not epic.startswith('CS.D.'):
        # Try to construct full epic
        epic = f'CS.D.{epic}.MINI.IP'

    cmd = [
        'docker', 'exec', 'task-worker',
        'python', '/app/forex_scanner/bt.py',
        epic.split('.')[2] if '.' in epic else epic,  # Extract pair name
        str(config.days)
    ]

    # Strategy
    if config.strategy:
        cmd.append(config.strategy)

    # Timeframe
    if config.timeframe:
        cmd.extend(['--timeframe', config.timeframe])

    # Date range
    if config.start_date:
        cmd.extend(['--start-date', config.start_date])
    if config.end_date:
        cmd.extend(['--end-date', config.end_date])

    # Parallel execution
    if config.parallel:
        cmd.append('--parallel')
        if config.workers:
            cmd.extend(['--workers', str(config.workers)])
        if config.chunk_days:
            cmd.extend(['--chunk-days', str(config.chunk_days)])

    # Chart generation
    if config.generate_chart:
        cmd.append('--chart')

    # Pipeline mode
    if config.pipeline_mode:
        cmd.append('--pipeline')

    # Parameter overrides
    if config.overrides:
        for key, value in config.overrides.items():
            if value is not None:
                cmd.extend(['--override', f'{key}={value}'])

    # Snapshot
    if config.snapshot:
        cmd.extend(['--snapshot', config.snapshot])

    # Historical intelligence (default is OFF, so only add flag when enabled)
    if config.use_historical_intelligence:
        cmd.append('--use-historical-intelligence')

    return cmd


def _parse_backtest_output(stdout: str) -> Dict[str, Any]:
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


@router.post("/run", response_model=BacktestResult)
async def run_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run a backtest synchronously and return results.

    This executes the backtest directly in the task-worker container
    and waits for completion before returning results.
    """
    start_time = datetime.now()

    try:
        # Build command
        cmd = _build_backtest_command(config)
        logger.info(f"Running backtest: {' '.join(cmd)}")

        # Execute backtest
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Parse output
        parsed = _parse_backtest_output(result.stdout)

        if result.returncode != 0:
            logger.error(f"Backtest failed: {result.stderr}")
            return BacktestResult(
                success=False,
                error=result.stderr or "Backtest execution failed",
                stdout=result.stdout,
                duration_seconds=duration,
                **parsed
            )

        logger.info(f"Backtest completed in {duration:.1f}s: {parsed['signal_count']} signals")

        return BacktestResult(
            success=True,
            duration_seconds=duration,
            stdout=result.stdout,
            **parsed
        )

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("Backtest timed out after 10 minutes")
        return BacktestResult(
            success=False,
            error="Backtest timed out after 10 minutes",
            duration_seconds=duration
        )

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.exception(f"Backtest error: {e}")
        return BacktestResult(
            success=False,
            error=str(e),
            duration_seconds=duration
        )


# Store for background backtest results
_background_results: Dict[str, Dict] = {}


@router.post("/run-async")
async def run_backtest_async(config: BacktestConfig, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Start a backtest in the background and return immediately.

    Use GET /api/backtest/status/{job_id} to check progress.
    """
    import uuid

    job_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Initialize job status
    _background_results[job_id] = {
        'status': 'running',
        'started_at': datetime.now().isoformat(),
        'config': config.dict(),
        'result': None
    }

    def run_in_background(job_id: str, config: BacktestConfig):
        """Background task to run backtest"""
        try:
            start_time = datetime.now()
            cmd = _build_backtest_command(config)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            duration = (datetime.now() - start_time).total_seconds()
            parsed = _parse_backtest_output(result.stdout)

            _background_results[job_id] = {
                'status': 'completed' if result.returncode == 0 else 'failed',
                'started_at': _background_results[job_id]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'config': config.dict(),
                'result': BacktestResult(
                    success=result.returncode == 0,
                    duration_seconds=duration,
                    stdout=result.stdout if result.returncode != 0 else None,
                    error=result.stderr if result.returncode != 0 else None,
                    **parsed
                ).dict()
            }

        except Exception as e:
            _background_results[job_id] = {
                'status': 'failed',
                'started_at': _background_results[job_id]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'config': config.dict(),
                'result': BacktestResult(
                    success=False,
                    error=str(e)
                ).dict()
            }

    background_tasks.add_task(run_in_background, job_id, config)

    return {
        'job_id': job_id,
        'status': 'running',
        'message': 'Backtest started. Use GET /api/backtest/status/{job_id} to check progress.'
    }


@router.get("/status/{job_id}")
async def get_backtest_status(job_id: str) -> Dict[str, Any]:
    """Get the status of a background backtest"""
    if job_id not in _background_results:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _background_results[job_id]


@router.get("/health")
async def backtest_health() -> Dict[str, Any]:
    """Check if backtest service is available"""
    try:
        # Check if task-worker container is running
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=task-worker', '--format', '{{.Status}}'],
            capture_output=True,
            text=True,
            timeout=5
        )

        container_running = 'Up' in result.stdout

        return {
            'status': 'healthy' if container_running else 'degraded',
            'task_worker_running': container_running,
            'pending_jobs': len([j for j in _background_results.values() if j['status'] == 'running'])
        }

    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


@router.get("/epics")
async def get_available_epics() -> List[Dict[str, str]]:
    """Get list of available currency pairs for backtesting"""
    return [
        {'shortname': k, 'epic': v}
        for k, v in EPIC_MAPPING.items()
    ]


class VariationConfig(BaseModel):
    """Configuration for parameter variation testing"""
    epic: str = Field(..., description="Currency pair (e.g., EURUSD)")
    days: int = Field(default=14, ge=1, le=365, description="Days to backtest")
    strategy: str = Field(default="SMC_SIMPLE", description="Strategy name")
    timeframe: str = Field(default="15m", description="Timeframe")

    # Parameter grid - dict of param names to lists of values
    param_grid: Dict[str, List[Any]] = Field(
        ...,
        description="Parameter grid. Example: {'fixed_stop_loss_pips': [8, 10, 12]}"
    )

    # Execution options
    max_workers: int = Field(default=4, ge=1, le=8, description="Parallel workers")
    rank_by: str = Field(
        default="composite_score",
        description="Metric to rank results by"
    )
    top_n: int = Field(default=10, ge=1, le=100, description="Number of top results")


class VariationResultItem(BaseModel):
    """Single variation result"""
    rank: int
    params: Dict[str, Any]
    signal_count: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    profit_factor: float = 0.0
    composite_score: float = 0.0
    status: str = "completed"


class VariationResults(BaseModel):
    """Results of parameter variation testing"""
    success: bool
    total_variations: int = 0
    completed: int = 0
    failed: int = 0
    duration_seconds: float = 0.0
    results: List[VariationResultItem] = []
    best_params: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _build_variation_command(config: VariationConfig) -> List[str]:
    """Build docker exec command for variation testing"""
    import json

    # Map epic shortname to full epic if needed
    epic = config.epic
    if epic in EPIC_MAPPING:
        epic = EPIC_MAPPING[epic]
    elif not epic.startswith('CS.D.'):
        epic = f'CS.D.{epic}.MINI.IP'

    cmd = [
        'docker', 'exec', 'task-worker',
        'python', '/app/forex_scanner/bt.py',
        epic.split('.')[2] if '.' in epic else epic,
        str(config.days)
    ]

    # Strategy
    if config.strategy:
        cmd.append(config.strategy)

    # Timeframe
    if config.timeframe:
        cmd.extend(['--timeframe', config.timeframe])

    # Parameter grid as JSON
    vary_json = json.dumps(config.param_grid)
    cmd.extend(['--vary-json', vary_json])

    # Variation options
    cmd.extend(['--vary-workers', str(config.max_workers)])
    cmd.extend(['--rank-by', config.rank_by])
    cmd.extend(['--top-n', str(config.top_n)])

    return cmd


@router.post("/run-variations", response_model=VariationResults)
async def run_parameter_variations(config: VariationConfig) -> VariationResults:
    """
    Run parameter variation testing.

    Tests multiple parameter combinations in parallel and returns ranked results.
    """
    start_time = datetime.now()

    try:
        cmd = _build_variation_command(config)
        logger.info(f"Running variation testing: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout for variations
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode != 0:
            logger.error(f"Variation testing failed: {result.stderr}")
            return VariationResults(
                success=False,
                error=result.stderr or "Variation testing failed",
                duration_seconds=duration
            )

        # Parse results from stdout (basic parsing)
        # The detailed results would ideally be returned via a structured format
        parsed_results = _parse_variation_output(result.stdout)

        return VariationResults(
            success=True,
            duration_seconds=duration,
            **parsed_results
        )

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        return VariationResults(
            success=False,
            error="Variation testing timed out after 30 minutes",
            duration_seconds=duration
        )
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.exception(f"Variation testing error: {e}")
        return VariationResults(
            success=False,
            error=str(e),
            duration_seconds=duration
        )


def _parse_variation_output(stdout: str) -> Dict[str, Any]:
    """Parse variation testing output"""
    result = {
        'total_variations': 0,
        'completed': 0,
        'failed': 0,
        'results': [],
        'best_params': None
    }

    lines = stdout.split('\n')

    for line in lines:
        # Look for combination count
        if 'combinations' in line.lower():
            match = re.search(r'(\d+)\s*combination', line.lower())
            if match:
                result['total_variations'] = int(match.group(1))

        # Look for completed count
        if 'successful' in line.lower() or 'completed' in line.lower():
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                result['completed'] = int(match.group(1))
                result['total_variations'] = int(match.group(2))

    result['failed'] = result['total_variations'] - result['completed']

    return result
