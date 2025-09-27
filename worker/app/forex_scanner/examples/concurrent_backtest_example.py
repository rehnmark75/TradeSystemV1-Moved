#!/usr/bin/env python3
# examples/concurrent_backtest_example.py
"""
Complete Example: High-Performance Concurrent Backtest Execution

This example demonstrates how to use the concurrent execution system to run
multiple backtests in parallel while maintaining live scanner performance.

Usage:
    python concurrent_backtest_example.py [--mode concurrent|standalone] [--jobs 5]
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from core.concurrent_execution import (
        create_concurrent_execution_system,
        create_optimized_backtest_config,
        ExecutionMode,
        JobPriority,
        check_system_readiness,
        PerformanceProfiler
    )
    from core.database import DatabaseManager
    import config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the correct directory")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('concurrent_backtest_example.log')
    ]
)

logger = logging.getLogger('ConcurrentBacktestExample')


class ConcurrentBacktestDemo:
    """
    Demonstration of concurrent backtest execution system

    Shows how to:
    1. Initialize the concurrent execution system
    2. Submit multiple backtest jobs with different priorities
    3. Monitor system performance and resource usage
    4. Handle job completion and results
    5. Gracefully shutdown the system
    """

    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.CONCURRENT):
        self.execution_mode = execution_mode
        self.execution_manager = None
        self.submitted_jobs: List[str] = []
        self.completed_jobs: Dict[str, Any] = {}
        self.start_time = None

    async def initialize_system(self, max_concurrent_backtests: int = 4):
        """Initialize the concurrent execution system"""

        logger.info("🚀 Initializing Concurrent Backtest Execution System")
        logger.info(f"   Mode: {self.execution_mode.value}")
        logger.info(f"   Max concurrent backtests: {max_concurrent_backtests}")

        # Check system readiness
        is_ready, message = check_system_readiness()
        logger.info(f"📊 System Status: {message}")

        if not is_ready:
            logger.warning("⚠️ System may not be optimal for concurrent execution")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False

        # Create database manager (optional for demo)
        try:
            db_manager = DatabaseManager(config.DATABASE_URL)
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.warning(f"⚠️ Database not available, using mock mode: {e}")
            db_manager = None

        # Create execution manager with optimized configuration
        self.execution_manager = create_concurrent_execution_system(
            max_concurrent_backtests=max_concurrent_backtests,
            max_memory_usage_gb=6.0,
            enable_memory_pools=True,
            enable_process_isolation=True,
            db_manager=db_manager,
            logger=logger
        )

        # Start the system
        await self.execution_manager.start(self.execution_mode)
        logger.info("✅ Concurrent execution system started successfully")

        return True

    def create_sample_backtest_configs(self, num_jobs: int = 5) -> List[Dict[str, Any]]:
        """Create sample backtest configurations for demonstration"""

        strategies = ['MACD', 'EMA_Crossover', 'RSI_Divergence', 'Bollinger_Bands', 'Momentum']
        currency_pairs = [
            ['EUR/USD', 'GBP/USD'],
            ['USD/JPY', 'EUR/JPY'],
            ['AUD/USD', 'NZD/USD'],
            ['USD/CHF', 'EUR/CHF'],
            ['GBP/JPY', 'EUR/GBP']
        ]
        timeframes = ['5m', '15m', '30m', '1h']
        priorities = [JobPriority.HIGH, JobPriority.NORMAL, JobPriority.NORMAL, JobPriority.LOW, JobPriority.BULK]

        configs = []
        end_date = datetime.now()

        for i in range(min(num_jobs, len(strategies))):
            # Create different time ranges for variety
            days_back = 30 + (i * 15)  # 30, 45, 60, 75, 90 days
            start_date = end_date - timedelta(days=days_back)

            config = create_optimized_backtest_config(
                strategy_name=strategies[i],
                start_date=start_date,
                end_date=end_date - timedelta(days=i),  # Slightly different end dates
                epics=currency_pairs[i],
                timeframe=timeframes[i % len(timeframes)],
                priority=priorities[i]
            )

            # Add demo-specific metadata
            config.update({
                'demo_job_id': f"demo_job_{i+1}",
                'estimated_duration_minutes': 5 + (i * 2),  # Simulate different durations
                'description': f"Demo backtest for {strategies[i]} strategy on {len(currency_pairs[i])} pairs"
            })

            configs.append(config)

        return configs

    async def submit_backtest_jobs(self, configs: List[Dict[str, Any]]) -> List[str]:
        """Submit backtest jobs to the execution system"""

        logger.info(f"📝 Submitting {len(configs)} backtest jobs...")

        submitted_job_ids = []

        for i, config in enumerate(configs, 1):
            try:
                # Add job completion callback
                def job_completion_callback(job, results):
                    self._handle_job_completion(job, results)

                job_id = self.execution_manager.submit_backtest(
                    config,
                    priority=config['priority'],
                    callback=job_completion_callback
                )

                submitted_job_ids.append(job_id)
                logger.info(f"   ✅ Job {i}/{len(configs)} submitted: {job_id}")
                logger.info(f"      Strategy: {config['strategy_name']}")
                logger.info(f"      Priority: {config['priority'].name}")
                logger.info(f"      Time Range: {config['start_date'].date()} to {config['end_date'].date()}")

                # Small delay between submissions to demonstrate queueing
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Failed to submit job {i}: {e}")

        self.submitted_jobs = submitted_job_ids
        logger.info(f"📋 Total jobs submitted: {len(submitted_job_ids)}")

        return submitted_job_ids

    async def monitor_execution(self, monitoring_duration_minutes: int = 10):
        """Monitor the execution system performance and job progress"""

        logger.info(f"📊 Starting execution monitoring for {monitoring_duration_minutes} minutes...")

        self.start_time = datetime.now()
        monitoring_start = datetime.now()
        last_stats_time = datetime.now()

        while (datetime.now() - monitoring_start).total_seconds() < monitoring_duration_minutes * 60:
            try:
                # Get system statistics every 10 seconds
                if (datetime.now() - last_stats_time).total_seconds() >= 10:
                    await self._log_system_statistics()
                    last_stats_time = datetime.now()

                # Check job statuses
                await self._check_job_statuses()

                # Check if all jobs completed
                if len(self.completed_jobs) >= len(self.submitted_jobs):
                    logger.info("✅ All jobs completed!")
                    break

                await asyncio.sleep(2.0)

            except KeyboardInterrupt:
                logger.info("⚠️ Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _log_system_statistics(self):
        """Log comprehensive system statistics"""

        try:
            # Execution statistics
            exec_stats = self.execution_manager.get_execution_stats()

            # Live scanner performance (if in concurrent mode)
            live_perf = self.execution_manager.get_live_scanner_performance()

            logger.info("📊 System Statistics:")
            logger.info(f"   ⚡ Memory Usage: {exec_stats.memory_usage_gb:.1f}GB")
            logger.info(f"   🖥️  CPU Usage: {exec_stats.cpu_usage_percent:.1f}%")
            logger.info(f"   📋 Jobs - Queued: {exec_stats.backtests_queued}, Running: {exec_stats.backtests_running}")
            logger.info(f"   ✅ Jobs - Completed: {exec_stats.backtests_completed}, Failed: {exec_stats.backtests_failed}")

            if self.execution_mode == ExecutionMode.CONCURRENT:
                logger.info(f"   🔍 Live Scanner Latency: {exec_stats.live_scanner_latency_ms:.1f}ms")

            # Resource monitor status
            if hasattr(self.execution_manager, 'resource_monitor'):
                alert_level = self.execution_manager.resource_monitor.get_alert_level()
                circuit_state = self.execution_manager.resource_monitor.get_circuit_breaker_state()
                logger.info(f"   🚨 Alert Level: {alert_level.value}")
                logger.info(f"   ⚡ Circuit Breaker: {circuit_state.value}")

        except Exception as e:
            logger.error(f"❌ Error logging system statistics: {e}")

    async def _check_job_statuses(self):
        """Check and log job statuses"""

        try:
            for job_id in self.submitted_jobs:
                if job_id not in self.completed_jobs:
                    status = self.execution_manager.get_job_status(job_id)

                    if status.get('status') in ['completed', 'failed', 'cancelled']:
                        logger.info(f"🎯 Job {job_id} {status['status']}")
                        self.completed_jobs[job_id] = status

        except Exception as e:
            logger.error(f"❌ Error checking job statuses: {e}")

    def _handle_job_completion(self, job, results):
        """Handle job completion callback"""

        try:
            logger.info(f"✅ Job completed: {job.job_id}")
            logger.info(f"   Strategy: {job.backtest_config.get('strategy_name', 'Unknown')}")

            if hasattr(job, 'end_time') and hasattr(job, 'start_time'):
                duration = (job.end_time - job.start_time).total_seconds()
                logger.info(f"   Duration: {duration:.1f}s")

            # Log key results (if available)
            if isinstance(results, dict):
                signals_detected = results.get('execution_stats', {}).get('total_signals_detected', 0)
                logger.info(f"   Signals detected: {signals_detected}")

        except Exception as e:
            logger.error(f"❌ Error handling job completion: {e}")

    async def generate_final_report(self):
        """Generate final execution report"""

        if not self.start_time:
            return

        total_duration = datetime.now() - self.start_time

        logger.info("\n" + "="*60)
        logger.info("📊 FINAL EXECUTION REPORT")
        logger.info("="*60)

        logger.info(f"🕐 Total Execution Time: {total_duration.total_seconds():.1f}s")
        logger.info(f"📝 Jobs Submitted: {len(self.submitted_jobs)}")
        logger.info(f"✅ Jobs Completed: {len(self.completed_jobs)}")

        # Calculate success rate
        successful_jobs = len([job for job in self.completed_jobs.values()
                              if job.get('status') == 'completed'])

        if len(self.submitted_jobs) > 0:
            success_rate = (successful_jobs / len(self.submitted_jobs)) * 100
            logger.info(f"🎯 Success Rate: {success_rate:.1f}%")

        # System performance summary
        try:
            final_stats = self.execution_manager.get_execution_stats()
            logger.info(f"🚀 Peak Memory Usage: {final_stats.memory_usage_gb:.1f}GB")
            logger.info(f"⚡ Circuit Breaker Trips: {final_stats.circuit_breaker_trips}")
            logger.info(f"🔄 Worker Restarts: {final_stats.worker_restarts}")

            if final_stats.live_scanner_latency_ms > 0:
                logger.info(f"🔍 Final Live Scanner Latency: {final_stats.live_scanner_latency_ms:.1f}ms")

        except Exception as e:
            logger.error(f"❌ Error generating performance summary: {e}")

        logger.info("="*60)

    async def cleanup_and_shutdown(self):
        """Clean up resources and shutdown the system"""

        logger.info("🛑 Initiating system shutdown...")

        if self.execution_manager:
            # Cancel any remaining jobs
            for job_id in self.submitted_jobs:
                if job_id not in self.completed_jobs:
                    try:
                        cancelled = self.execution_manager.cancel_job(job_id)
                        if cancelled:
                            logger.info(f"❌ Cancelled job: {job_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to cancel job {job_id}: {e}")

            # Shutdown execution manager
            await self.execution_manager.shutdown(timeout_seconds=30.0)

        logger.info("✅ System shutdown completed")

    async def run_complete_demo(self, num_jobs: int = 5, monitoring_minutes: int = 10):
        """Run the complete demonstration"""

        logger.info("🎬 Starting Concurrent Backtest Execution Demo")
        logger.info("="*60)

        try:
            # Initialize system
            if not await self.initialize_system():
                logger.error("❌ System initialization failed")
                return False

            # Create sample configurations
            configs = self.create_sample_backtest_configs(num_jobs)
            logger.info(f"📋 Created {len(configs)} sample backtest configurations")

            # Submit jobs
            submitted_jobs = await self.submit_backtest_jobs(configs)
            if not submitted_jobs:
                logger.error("❌ No jobs submitted successfully")
                return False

            # Monitor execution
            await self.monitor_execution(monitoring_minutes)

            # Generate final report
            await self.generate_final_report()

            return True

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False

        finally:
            # Always cleanup
            await self.cleanup_and_shutdown()


async def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Concurrent Backtest Execution System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python concurrent_backtest_example.py --mode concurrent --jobs 3 --monitor 5
  python concurrent_backtest_example.py --mode standalone --jobs 10
        """
    )

    parser.add_argument(
        '--mode',
        choices=['concurrent', 'standalone'],
        default='concurrent',
        help='Execution mode (default: concurrent)'
    )

    parser.add_argument(
        '--jobs',
        type=int,
        default=5,
        help='Number of backtest jobs to submit (default: 5)'
    )

    parser.add_argument(
        '--monitor',
        type=int,
        default=10,
        help='Monitoring duration in minutes (default: 10)'
    )

    args = parser.parse_args()

    # Convert mode string to enum
    execution_mode = ExecutionMode.CONCURRENT if args.mode == 'concurrent' else ExecutionMode.STANDALONE

    # Create and run demo
    demo = ConcurrentBacktestDemo(execution_mode)

    logger.info(f"🎯 Demo Configuration:")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Jobs: {args.jobs}")
    logger.info(f"   Monitor Duration: {args.monitor} minutes")

    success = await demo.run_complete_demo(
        num_jobs=args.jobs,
        monitoring_minutes=args.monitor
    )

    if success:
        logger.info("🎉 Demo completed successfully!")
        return 0
    else:
        logger.error("💥 Demo failed!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("⚠️ Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)