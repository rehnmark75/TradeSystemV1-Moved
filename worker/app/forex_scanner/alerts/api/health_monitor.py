"""
Health Monitor - API Health Monitoring Module
Monitors API health, tracks performance metrics, and provides status reporting
Extracted from claude_api.py for better modularity
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum


class HealthStatus(Enum):
    """API health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class HealthMetrics:
    """Health metrics for API monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    rate_limited_requests: int = 0
    average_response_time: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    uptime_percentage: float = 100.0


@dataclass
class HealthCheck:
    """Single health check result"""
    timestamp: datetime
    status: HealthStatus
    response_time: float
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    success: bool = True


class HealthMonitor:
    """
    Monitors API health and provides comprehensive status reporting
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.health_history: List[HealthCheck] = []
        self.metrics = HealthMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Thresholds for health determination
        self.healthy_response_time_threshold = 5.0  # seconds
        self.degraded_response_time_threshold = 10.0  # seconds
        self.max_consecutive_failures = 3
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = 0
    
    def record_request(self, 
                      success: bool,
                      response_time: float,
                      status_code: Optional[int] = None,
                      error_message: Optional[str] = None,
                      is_timeout: bool = False,
                      is_rate_limited: bool = False) -> None:
        """
        Record the result of an API request
        """
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.now()
        else:
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = datetime.now()
            
            if is_timeout:
                self.metrics.timeout_requests += 1
            if is_rate_limited:
                self.metrics.rate_limited_requests += 1
        
        # Update average response time (rolling average)
        if self.metrics.total_requests == 1:
            self.metrics.average_response_time = response_time
        else:
            # Weighted rolling average (more weight to recent requests)
            weight = 0.1
            self.metrics.average_response_time = (
                (1 - weight) * self.metrics.average_response_time + 
                weight * response_time
            )
        
        # Update uptime percentage
        if self.metrics.total_requests > 0:
            self.metrics.uptime_percentage = (
                self.metrics.successful_requests / self.metrics.total_requests * 100
            )
        
        # Record health check
        status = self._determine_status(success, response_time, is_rate_limited)
        health_check = HealthCheck(
            timestamp=datetime.now(),
            status=status,
            response_time=response_time,
            error_message=error_message,
            status_code=status_code,
            success=success
        )
        
        self._add_health_check(health_check)
        
        # Log significant events
        if not success and self.metrics.consecutive_failures == 1:
            self.logger.warning(f"⚠️ First API failure: {error_message}")
        elif not success and self.metrics.consecutive_failures >= self.max_consecutive_failures:
            self.logger.error(f"🚨 {self.metrics.consecutive_failures} consecutive API failures")
        elif success and self.metrics.consecutive_successes == 1 and self.metrics.failed_requests > 0:
            self.logger.info(f"✅ API recovered after {self.metrics.consecutive_failures} failures")
    
    def perform_health_check(self, health_check_func: callable) -> HealthCheck:
        """
        Perform a dedicated health check using provided function
        """
        start_time = time.time()
        
        try:
            # Execute health check function
            result = health_check_func()
            response_time = time.time() - start_time
            
            # Determine if health check was successful
            if isinstance(result, dict):
                success = result.get('success', True)
                error_message = result.get('error')
                status_code = result.get('status_code')
            else:
                success = bool(result)
                error_message = None
                status_code = None
            
            self.record_request(
                success=success,
                response_time=response_time,
                status_code=status_code,
                error_message=error_message
            )
            
            return self.health_history[-1] if self.health_history else None
            
        except Exception as e:
            response_time = time.time() - start_time
            error_message = str(e)
            
            self.record_request(
                success=False,
                response_time=response_time,
                error_message=error_message
            )
            
            return self.health_history[-1] if self.health_history else None
    
    def get_current_status(self) -> Dict:
        """
        Get comprehensive current health status
        """
        current_status = self._get_overall_status()
        
        # Get recent health checks (last 10)
        recent_checks = self.health_history[-10:] if self.health_history else []
        
        # Calculate recent success rate
        if recent_checks:
            recent_successes = sum(1 for check in recent_checks if check.success)
            recent_success_rate = recent_successes / len(recent_checks) * 100
        else:
            recent_success_rate = 0.0
        
        # Time since last success/failure
        time_since_last_success = None
        time_since_last_failure = None
        
        if self.metrics.last_success_time:
            time_since_last_success = (datetime.now() - self.metrics.last_success_time).total_seconds()
        
        if self.metrics.last_failure_time:
            time_since_last_failure = (datetime.now() - self.metrics.last_failure_time).total_seconds()
        
        return {
            'status': current_status.value,
            'recommendation': self._get_recommendation(current_status),
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'success_rate': self.metrics.uptime_percentage,
                'recent_success_rate': recent_success_rate,
                'average_response_time': self.metrics.average_response_time,
                'consecutive_failures': self.metrics.consecutive_failures,
                'consecutive_successes': self.metrics.consecutive_successes,
                'timeout_rate': (self.metrics.timeout_requests / max(self.metrics.total_requests, 1)) * 100,
                'rate_limit_rate': (self.metrics.rate_limited_requests / max(self.metrics.total_requests, 1)) * 100
            },
            'timing': {
                'time_since_last_success': time_since_last_success,
                'time_since_last_failure': time_since_last_failure,
                'last_check': self.health_history[-1].timestamp.isoformat() if self.health_history else None
            },
            'recent_checks': [
                {
                    'timestamp': check.timestamp.isoformat(),
                    'status': check.status.value,
                    'response_time': check.response_time,
                    'success': check.success,
                    'error': check.error_message
                }
                for check in recent_checks
            ]
        }
    
    def get_health_summary(self) -> Dict:
        """
        Get a simplified health summary
        """
        status = self._get_overall_status()
        
        return {
            'status': status.value,
            'healthy': status == HealthStatus.HEALTHY,
            'success_rate': self.metrics.uptime_percentage,
            'average_response_time': self.metrics.average_response_time,
            'total_requests': self.metrics.total_requests,
            'consecutive_failures': self.metrics.consecutive_failures,
            'recommendation': self._get_recommendation(status)
        }
    
    def reset_metrics(self):
        """Reset all metrics and history"""
        self.health_history.clear()
        self.metrics = HealthMetrics()
        self.logger.info("📊 Health metrics reset")
    
    def _add_health_check(self, health_check: HealthCheck):
        """Add health check to history with size limit"""
        self.health_history.append(health_check)
        
        # Maintain max history size
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
    
    def _determine_status(self, 
                         success: bool, 
                         response_time: float, 
                         is_rate_limited: bool) -> HealthStatus:
        """
        Determine health status based on request result
        """
        if not success:
            if is_rate_limited:
                return HealthStatus.DEGRADED
            elif self.metrics.consecutive_failures >= self.max_consecutive_failures:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.DEGRADED
        
        # Success case
        if response_time > self.degraded_response_time_threshold:
            return HealthStatus.DEGRADED
        elif response_time > self.healthy_response_time_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _get_overall_status(self) -> HealthStatus:
        """
        Get overall health status based on recent history and metrics
        """
        if self.metrics.total_requests == 0:
            return HealthStatus.UNAVAILABLE
        
        # Check for critical failures
        if self.metrics.consecutive_failures >= self.max_consecutive_failures:
            return HealthStatus.UNHEALTHY
        
        # Check recent performance
        recent_checks = self.health_history[-10:] if len(self.health_history) >= 10 else self.health_history
        
        if not recent_checks:
            return HealthStatus.UNAVAILABLE
        
        # Count status types in recent checks
        status_counts = {}
        for check in recent_checks:
            status_counts[check.status] = status_counts.get(check.status, 0) + 1
        
        total_recent = len(recent_checks)
        
        # Determine overall status
        if status_counts.get(HealthStatus.HEALTHY, 0) / total_recent >= 0.8:
            return HealthStatus.HEALTHY
        elif status_counts.get(HealthStatus.UNHEALTHY, 0) / total_recent >= 0.5:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED
    
    def _get_recommendation(self, status: HealthStatus) -> str:
        """
        Get recommendation based on health status
        """
        recommendations = {
            HealthStatus.HEALTHY: "Continue normal operations",
            HealthStatus.DEGRADED: "Monitor closely, consider fallback mode if performance continues to degrade",
            HealthStatus.UNHEALTHY: "Use fallback analysis mode, check API status and credentials",
            HealthStatus.UNAVAILABLE: "Configure API key and test connectivity",
            HealthStatus.ERROR: "Check logs for detailed error information"
        }
        
        base_recommendation = recommendations.get(status, "Monitor system status")
        
        # Add specific recommendations based on metrics
        if self.metrics.rate_limited_requests > 0:
            base_recommendation += ". Consider reducing request frequency."
        
        if self.metrics.timeout_requests > self.metrics.total_requests * 0.1:
            base_recommendation += ". High timeout rate detected, check network connectivity."
        
        if self.metrics.average_response_time > self.degraded_response_time_threshold:
            base_recommendation += ". Slow response times detected."
        
        return base_recommendation


# Factory function for Claude API health monitoring
def create_claude_health_monitor() -> HealthMonitor:
    """Create health monitor optimized for Claude API"""
    monitor = HealthMonitor(max_history=100)
    
    # Claude-specific thresholds
    monitor.healthy_response_time_threshold = 3.0  # seconds
    monitor.degraded_response_time_threshold = 8.0  # seconds
    monitor.max_consecutive_failures = 3
    
    return monitor


# Usage example
if __name__ == "__main__":
    monitor = create_claude_health_monitor()
    
    # Simulate some API calls
    import random
    
    for i in range(10):
        success = random.choice([True, True, True, False])  # 75% success rate
        response_time = random.uniform(1.0, 6.0)
        
        monitor.record_request(
            success=success,
            response_time=response_time,
            status_code=200 if success else 500,
            error_message=None if success else "Simulated error"
        )
    
    # Get health status
    status = monitor.get_current_status()
    print("🏥 Health Status:")
    print(f"  Status: {status['status']}")
    print(f"  Success Rate: {status['metrics']['success_rate']:.1f}%")
    print(f"  Avg Response Time: {status['metrics']['average_response_time']:.2f}s")
    print(f"  Recommendation: {status['recommendation']}")