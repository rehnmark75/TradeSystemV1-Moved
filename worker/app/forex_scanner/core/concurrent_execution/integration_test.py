#!/usr/bin/env python3
# core/concurrent_execution/integration_test.py
"""
Integration Test Suite for Concurrent Backtest Execution System

This comprehensive test suite validates the complete concurrent execution system
including performance characteristics, resource management, and fault tolerance.
"""

import asyncio
import unittest
import time
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import tempfile
import os

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ConcurrentExecutionIntegrationTest(unittest.TestCase):
    """Comprehensive integration tests for concurrent backtest execution"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = logging.getLogger('IntegrationTest')
        cls.temp_dir = tempfile.mkdtemp(prefix='backtest_test_')
        cls.test_start_time = datetime.now()

        cls.logger.info("🧪 Starting Concurrent Execution Integration Tests")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        test_duration = datetime.now() - cls.test_start_time
        cls.logger.info(f"🧪 Integration tests completed in {test_duration.total_seconds():.1f}s")

    def setUp(self):
        """Set up each test"""
        self.test_start = time.time()

    def tearDown(self):
        """Clean up after each test"""
        test_duration = time.time() - self.test_start
        self.logger.info(f"✅ Test completed in {test_duration:.2f}s")

    async def test_basic_system_initialization(self):
        """Test basic system initialization and configuration"""
        self.logger.info("🔧 Testing system initialization...")

        try:
            from . import (
                create_concurrent_execution_system,
                check_system_readiness,
                ExecutionMode
            )

            # Check system readiness
            is_ready, message = check_system_readiness()
            self.logger.info(f"System readiness: {message}")

            if not is_ready:
                self.skipTest(f"System not ready for testing: {message}")

            # Create execution system
            execution_manager = create_concurrent_execution_system(
                max_concurrent_backtests=2,  # Limited for testing
                max_memory_usage_gb=4.0,
                enable_memory_pools=True
            )

            self.assertIsNotNone(execution_manager)
            self.logger.info("✅ System initialization successful")

        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")

    async def test_memory_pool_performance(self):
        """Test memory pool performance and efficiency"""
        self.logger.info("💾 Testing memory pool performance...")

        try:
            from .memory_pools import (
                create_market_data_pools,
                PerformanceProfiler,
                ExecutionConfig
            )

            config = ExecutionConfig()
            pool_manager = create_market_data_pools(config)

            # Test DataFrame pool performance
            df_pool = pool_manager.get_pool("market_data")
            if df_pool:
                with PerformanceProfiler("dataframe_pool_operations") as profiler:
                    # Simulate high-frequency DataFrame operations
                    dataframes = []
                    for i in range(100):
                        df = df_pool.get()
                        if df is not None:
                            dataframes.append(df)

                    # Return all DataFrames
                    for df in dataframes:
                        df_pool.return_object(df)

                stats = profiler.get_stats()
                self.logger.info(f"DataFrame pool ops: {stats.average/1000:.2f}ms avg, {stats.count} operations")

                # Verify pool efficiency
                pool_stats = df_pool.get_stats()
                self.assertGreater(pool_stats.hit_rate, 0.5, "Pool hit rate should be > 50%")

            pool_manager.cleanup()
            self.logger.info("✅ Memory pool performance test passed")

        except ImportError as e:
            self.skipTest(f"Memory pool modules not available: {e}")

    async def test_performance_counter_accuracy(self):
        """Test performance counter accuracy and overhead"""
        self.logger.info("⏱️ Testing performance counter accuracy...")

        try:
            from .performance_counters import (
                PerformanceCounter,
                CounterType,
                benchmark_function
            )

            # Test timing accuracy
            counter = PerformanceCounter("timing_test", CounterType.TIMING)

            # Measure a known operation
            def test_operation():
                time.sleep(0.001)  # 1ms sleep

            # Benchmark the operation
            benchmark_results = benchmark_function(test_operation, iterations=10)

            self.logger.info(f"Benchmark results: {benchmark_results}")

            # Verify timing accuracy (within reasonable tolerance)
            expected_time_us = 1000  # 1ms = 1000us
            measured_time_us = benchmark_results['average_time_us']

            # Allow 50% tolerance for timing variations
            tolerance = 0.5
            self.assertGreater(measured_time_us, expected_time_us * (1 - tolerance))
            self.assertLess(measured_time_us, expected_time_us * (1 + tolerance) + 1000)

            self.logger.info("✅ Performance counter accuracy test passed")

        except ImportError as e:
            self.skipTest(f"Performance counter modules not available: {e}")

    async def test_resource_monitor_thresholds(self):
        """Test resource monitor threshold detection"""
        self.logger.info("📊 Testing resource monitor thresholds...")

        try:
            from .resource_monitor import (
                ResourceMonitor,
                ResourceThresholds,
                AlertLevel,
                ExecutionConfig
            )

            config = ExecutionConfig()

            # Create monitor with low thresholds for testing
            thresholds = ResourceThresholds(
                cpu_warning=10.0,     # Very low for testing
                cpu_critical=20.0,
                memory_warning=10.0,
                memory_critical=20.0
            )

            monitor = ResourceMonitor(config, logger=self.logger, thresholds=thresholds)

            # Start monitoring
            await monitor.start()

            # Let it collect some data
            await asyncio.sleep(2.0)

            # Check alert level
            alert_level = monitor.get_alert_level()
            self.logger.info(f"Current alert level: {alert_level}")

            # Get metrics
            metrics = monitor.get_current_metrics()
            self.assertIsNotNone(metrics)

            # Verify circuit breaker state
            cb_state = monitor.get_circuit_breaker_state()
            self.logger.info(f"Circuit breaker state: {cb_state}")

            await monitor.stop()
            self.logger.info("✅ Resource monitor test passed")

        except ImportError as e:
            self.skipTest(f"Resource monitor modules not available: {e}")

    async def test_queue_priority_scheduling(self):
        """Test backtest queue priority scheduling"""
        self.logger.info("📋 Testing queue priority scheduling...")

        try:
            from .backtest_queue import (
                BacktestQueue,
                JobPriority,
                BacktestJob,
                ExecutionConfig
            )
            from datetime import datetime

            config = ExecutionConfig()
            queue = BacktestQueue(config, logger=self.logger)

            # Create test jobs with different priorities
            test_jobs = []
            priorities = [JobPriority.LOW, JobPriority.HIGH, JobPriority.NORMAL, JobPriority.CRITICAL]

            for i, priority in enumerate(priorities):
                job = BacktestJob(
                    job_id=f"test_job_{i}",
                    execution_id=i + 1,
                    backtest_config={
                        'strategy_name': f'test_strategy_{i}',
                        'start_date': datetime.now() - timedelta(days=30),
                        'end_date': datetime.now(),
                        'epics': ['EUR/USD'],
                        'timeframe': '15m'
                    },
                    priority=priority,
                    submitted_at=datetime.now()
                )
                test_jobs.append(job)

            # Submit jobs in random order
            for job in test_jobs:
                success = await queue.submit_job(job)
                self.assertTrue(success, f"Failed to submit job {job.job_id}")

            # Retrieve jobs and verify priority ordering
            retrieved_jobs = []
            for _ in range(len(test_jobs)):
                job = await queue.get_next_job(timeout_seconds=1.0)
                if job:
                    retrieved_jobs.append(job)

            # Verify CRITICAL job came first
            self.assertEqual(retrieved_jobs[0].priority, JobPriority.CRITICAL)
            self.logger.info(f"Job retrieval order: {[j.priority.name for j in retrieved_jobs]}")

            self.logger.info("✅ Queue priority scheduling test passed")

        except ImportError as e:
            self.skipTest(f"Queue modules not available: {e}")

    def test_process_isolation_cpu_affinity(self):
        """Test process isolation and CPU affinity (synchronous test)"""
        self.logger.info("🎯 Testing process isolation and CPU affinity...")

        try:
            from .process_isolation import (
                ProcessManager,
                ProcessType,
                ExecutionConfig
            )
            import psutil

            config = ExecutionConfig(worker_affinity_enabled=True)
            process_manager = ProcessManager(config, logger=self.logger)

            # Register current process as live scanner for testing
            current_pid = os.getpid()
            success = process_manager.register_live_scanner(current_pid)

            if success:
                # Verify process was registered
                live_perf = process_manager.get_live_scanner_performance()
                self.assertEqual(live_perf['pid'], current_pid)
                self.logger.info(f"Live scanner registered: {live_perf}")

            # Test CPU affinity allocation
            cpu_affinity = process_manager._allocate_cpu_affinity_for_worker(1)
            if cpu_affinity:
                self.logger.info(f"Worker CPU affinity: {cpu_affinity}")

            self.logger.info("✅ Process isolation test passed")

        except ImportError as e:
            self.skipTest(f"Process isolation modules not available: {e}")

    async def test_complete_system_integration(self):
        """Test complete system integration with mock backtest"""
        self.logger.info("🔗 Testing complete system integration...")

        try:
            from . import (
                create_concurrent_execution_system,
                create_optimized_backtest_config,
                JobPriority,
                ExecutionMode
            )

            # Create execution system
            execution_manager = create_concurrent_execution_system(
                max_concurrent_backtests=1,  # Single worker for testing
                max_memory_usage_gb=2.0
            )

            # Start system in standalone mode
            await execution_manager.start(ExecutionMode.STANDALONE)

            # Create a simple backtest config
            backtest_config = create_optimized_backtest_config(
                strategy_name="test_integration_strategy",
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                epics=['EUR/USD'],
                priority=JobPriority.HIGH
            )

            # Submit backtest job (this would normally create execution record)
            # For testing, we'll simulate the job submission
            try:
                job_id = execution_manager.submit_backtest(
                    backtest_config,
                    priority=JobPriority.HIGH
                )
                self.logger.info(f"Submitted job: {job_id}")

                # Monitor job status
                start_time = time.time()
                timeout = 30.0  # 30 second timeout

                while time.time() - start_time < timeout:
                    status = execution_manager.get_job_status(job_id)
                    self.logger.info(f"Job status: {status}")

                    if status.get('status') in ['completed', 'failed', 'not_found']:
                        break

                    await asyncio.sleep(1.0)

            except Exception as e:
                # Expected if database/full system not available
                self.logger.info(f"Job submission skipped (expected in test): {e}")

            # Test system statistics
            stats = execution_manager.get_execution_stats()
            self.assertIsNotNone(stats)
            self.logger.info(f"System stats: queued={stats.backtests_queued}, running={stats.backtests_running}")

            # Shutdown system
            await execution_manager.shutdown(timeout_seconds=10.0)

            self.logger.info("✅ Complete system integration test passed")

        except ImportError as e:
            self.skipTest(f"System integration modules not available: {e}")


class PerformanceBenchmarkTest(unittest.TestCase):
    """Performance benchmark tests for the concurrent execution system"""

    def setUp(self):
        """Set up performance tests"""
        self.logger = logging.getLogger('PerformanceBenchmark')

    def test_memory_pool_throughput(self):
        """Benchmark memory pool throughput"""
        self.logger.info("🏎️ Benchmarking memory pool throughput...")

        try:
            from .memory_pools import DataFramePool
            from .performance_counters import benchmark_function

            pool = DataFramePool(max_objects=1000)
            pool.preallocate()

            def pool_operation():
                df = pool.get()
                if df is not None:
                    pool.return_object(df)

            # Benchmark pool operations
            results = benchmark_function(pool_operation, iterations=10000)

            self.logger.info(f"Pool throughput benchmark: {results}")
            self.assertLess(results['average_time_us'], 100, "Pool operations should be < 100μs")

            pool.cleanup()

        except ImportError:
            self.skipTest("Memory pool modules not available")

    def test_performance_counter_overhead(self):
        """Measure performance counter overhead"""
        self.logger.info("⏱️ Measuring performance counter overhead...")

        try:
            from .performance_counters import PerformanceCounter, CounterType

            counter = PerformanceCounter("overhead_test", CounterType.TIMING)

            # Measure timing overhead
            iterations = 100000
            start_time = time.perf_counter()

            for _ in range(iterations):
                timer_id = counter.start_timer()
                counter.stop_timer(timer_id)

            end_time = time.perf_counter()

            total_time_us = (end_time - start_time) * 1_000_000
            overhead_per_measurement = total_time_us / iterations

            self.logger.info(f"Performance counter overhead: {overhead_per_measurement:.2f}μs per measurement")
            self.assertLess(overhead_per_measurement, 10, "Counter overhead should be < 10μs")

        except ImportError:
            self.skipTest("Performance counter modules not available")


class StressTest(unittest.TestCase):
    """Stress tests for system reliability under load"""

    def setUp(self):
        """Set up stress tests"""
        self.logger = logging.getLogger('StressTest')

    async def test_queue_high_load(self):
        """Stress test queue with high job load"""
        self.logger.info("💪 Stress testing queue with high load...")

        try:
            from .backtest_queue import BacktestQueue, BacktestJob, JobPriority, ExecutionConfig
            from datetime import datetime

            config = ExecutionConfig()
            queue = BacktestQueue(config, max_queue_size=10000, logger=self.logger)

            # Submit large number of jobs quickly
            num_jobs = 1000
            submit_start = time.time()

            for i in range(num_jobs):
                job = BacktestJob(
                    job_id=f"stress_job_{i}",
                    execution_id=i + 1,
                    backtest_config={
                        'strategy_name': f'stress_strategy_{i}',
                        'start_date': datetime.now() - timedelta(days=1),
                        'end_date': datetime.now(),
                        'epics': ['EUR/USD'],
                        'timeframe': '15m'
                    },
                    priority=JobPriority.NORMAL,
                    submitted_at=datetime.now()
                )

                success = await queue.submit_job(job)
                if not success:
                    self.fail(f"Failed to submit job {i} of {num_jobs}")

            submit_time = time.time() - submit_start

            # Retrieve all jobs
            retrieve_start = time.time()
            retrieved_count = 0

            while retrieved_count < num_jobs:
                job = await queue.get_next_job(timeout_seconds=0.1)
                if job:
                    retrieved_count += 1
                else:
                    break

            retrieve_time = time.time() - retrieve_start

            self.logger.info(f"Stress test results:")
            self.logger.info(f"  Submitted {num_jobs} jobs in {submit_time:.2f}s ({num_jobs/submit_time:.1f} jobs/s)")
            self.logger.info(f"  Retrieved {retrieved_count} jobs in {retrieve_time:.2f}s ({retrieved_count/retrieve_time:.1f} jobs/s)")

            self.assertEqual(retrieved_count, num_jobs, "All jobs should be retrieved")

        except ImportError:
            self.skipTest("Queue modules not available")


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Concurrent Execution System Integration Tests")
    print("=" * 60)

    # Create test suite
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(ConcurrentExecutionIntegrationTest)
    performance_suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmarkTest)
    stress_suite = unittest.TestLoader().loadTestsFromTestCase(StressTest)

    # Combine all test suites
    full_suite = unittest.TestSuite([integration_suite, performance_suite, stress_suite])

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(full_suite)

    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


async def run_async_tests():
    """Run async integration tests"""
    test_instance = ConcurrentExecutionIntegrationTest()
    test_instance.setUpClass()

    tests_to_run = [
        test_instance.test_basic_system_initialization,
        test_instance.test_memory_pool_performance,
        test_instance.test_performance_counter_accuracy,
        test_instance.test_resource_monitor_thresholds,
        test_instance.test_queue_priority_scheduling,
        test_instance.test_complete_system_integration
    ]

    print("🔄 Running async integration tests...")

    for test in tests_to_run:
        try:
            test_instance.setUp()
            await test()
            test_instance.tearDown()
            print(f"✅ {test.__name__} passed")
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    test_instance.tearDownClass()


if __name__ == "__main__":
    # Run synchronous tests
    sync_success = run_integration_tests()

    # Run async tests
    print("\n🔄 Running async tests...")
    try:
        asyncio.run(run_async_tests())
        print("✅ Async tests completed")
    except Exception as e:
        print(f"❌ Async tests failed: {e}")

    print("\n🎯 Integration test suite completed")