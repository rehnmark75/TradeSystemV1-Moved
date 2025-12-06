"""
Health checker service - performs HTTP and TCP health checks on services.
"""
import httpx
import socket
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from ..config import settings, HEALTH_ENDPOINTS, TCP_CHECK_CONTAINERS, HEALTH_HEADERS
from ..models import HealthStatus

logger = logging.getLogger(__name__)


class HealthChecker:
    """Service for checking health of individual services."""

    def __init__(self):
        self._health_state: Dict[str, Dict] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=settings.health_check_timeout)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def check_http_health(self, name: str, url: str) -> Tuple[HealthStatus, float, Optional[str]]:
        """
        Check HTTP health endpoint.
        Returns: (status, response_time_ms, error_message)
        """
        start_time = datetime.now()
        try:
            client = await self._get_client()
            # Get any custom headers for this service
            headers = HEALTH_HEADERS.get(name, {})
            response = await client.get(url, headers=headers)
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            if response.status_code == 200:
                self._update_health_state(name, True)
                return HealthStatus.HEALTHY, response_time, None
            else:
                self._update_health_state(name, False)
                return HealthStatus.UNHEALTHY, response_time, f"HTTP {response.status_code}"

        except httpx.TimeoutException:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_state(name, False)
            return HealthStatus.UNHEALTHY, response_time, "Timeout"
        except httpx.ConnectError as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_state(name, False)
            return HealthStatus.UNHEALTHY, response_time, f"Connection failed: {e}"
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_state(name, False)
            logger.error(f"Health check failed for {name}: {e}")
            return HealthStatus.UNKNOWN, response_time, str(e)

    async def check_tcp_health(self, name: str, host: str, port: int) -> Tuple[HealthStatus, float, Optional[str]]:
        """
        Check TCP port connectivity.
        Returns: (status, response_time_ms, error_message)
        """
        start_time = datetime.now()
        try:
            # Use asyncio to check TCP connection
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=settings.health_check_timeout
            )
            writer.close()
            await writer.wait_closed()

            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_state(name, True)
            return HealthStatus.HEALTHY, response_time, None

        except asyncio.TimeoutError:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_state(name, False)
            return HealthStatus.UNHEALTHY, response_time, "Connection timeout"
        except ConnectionRefusedError:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_state(name, False)
            return HealthStatus.UNHEALTHY, response_time, "Connection refused"
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_state(name, False)
            logger.error(f"TCP health check failed for {name}: {e}")
            return HealthStatus.UNKNOWN, response_time, str(e)

    def _update_health_state(self, name: str, is_healthy: bool):
        """Update health state tracking for a service."""
        if name not in self._health_state:
            self._health_state[name] = {
                "consecutive_failures": 0,
                "consecutive_successes": 0,
                "last_check": None,
                "last_status": None,
            }

        state = self._health_state[name]
        state["last_check"] = datetime.now(timezone.utc)

        if is_healthy:
            state["consecutive_failures"] = 0
            state["consecutive_successes"] += 1
            state["last_status"] = HealthStatus.HEALTHY
        else:
            state["consecutive_successes"] = 0
            state["consecutive_failures"] += 1
            state["last_status"] = HealthStatus.UNHEALTHY

    def get_health_state(self, name: str) -> Dict:
        """Get health state for a service."""
        return self._health_state.get(name, {
            "consecutive_failures": 0,
            "consecutive_successes": 0,
            "last_check": None,
            "last_status": None,
        })

    def is_failing(self, name: str) -> bool:
        """Check if a service has exceeded the failure threshold."""
        state = self.get_health_state(name)
        return state["consecutive_failures"] >= settings.health_check_failures_threshold

    async def check_service(self, name: str) -> Tuple[HealthStatus, float, Optional[str]]:
        """
        Check health for a named service using appropriate method.
        Returns: (status, response_time_ms, error_message)
        """
        # Check if it's a TCP-based service
        if name in TCP_CHECK_CONTAINERS:
            host, port = TCP_CHECK_CONTAINERS[name]
            return await self.check_tcp_health(name, host, port)

        # Check if it has an HTTP health endpoint
        if name in HEALTH_ENDPOINTS:
            url = HEALTH_ENDPOINTS[name]
            return await self.check_http_health(name, url)

        # No health check configured - return unknown
        return HealthStatus.NONE, 0.0, "No health check configured"

    async def check_all_services(self) -> Dict[str, Dict]:
        """
        Check health for all configured services.
        Returns dict mapping service name to health info.
        """
        results = {}

        # Gather all health checks concurrently
        tasks = []
        service_names = list(HEALTH_ENDPOINTS.keys()) + list(TCP_CHECK_CONTAINERS.keys())

        for name in service_names:
            tasks.append(self.check_service(name))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for name, response in zip(service_names, responses):
            if isinstance(response, Exception):
                results[name] = {
                    "status": HealthStatus.UNKNOWN,
                    "response_time_ms": 0.0,
                    "error": str(response),
                    "state": self.get_health_state(name),
                }
            else:
                status, response_time, error = response
                results[name] = {
                    "status": status,
                    "response_time_ms": response_time,
                    "error": error,
                    "state": self.get_health_state(name),
                }

        return results

    async def check_database_health(self) -> Tuple[HealthStatus, Optional[str]]:
        """
        Check PostgreSQL database health with a simple query.
        Returns: (status, error_message)
        """
        try:
            import psycopg2

            conn = psycopg2.connect(
                settings.database_url,
                connect_timeout=settings.health_check_timeout
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()

            self._update_health_state("postgres_query", True)
            return HealthStatus.HEALTHY, None

        except Exception as e:
            self._update_health_state("postgres_query", False)
            logger.error(f"Database health check failed: {e}")
            return HealthStatus.UNHEALTHY, str(e)
