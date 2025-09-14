"""
TradingView Scripts MCP Client

Provides high-level interface for communicating with the TradingView Scripts MCP Server.
Supports both synchronous and asynchronous operations for different use cases.
"""

import json
import logging
import subprocess
import asyncio
import threading
import time
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import queue

logger = logging.getLogger(__name__)

class TVScriptsClient:
    """Synchronous client for TradingView Scripts MCP Server"""
    
    def __init__(self, server_cmd: Optional[List[str]] = None, timeout: int = 30):
        """
        Initialize MCP client
        
        Args:
            server_cmd: Command to start MCP server (default: auto-detect)
            timeout: Timeout for operations in seconds
        """
        self.timeout = timeout
        self.proc = None
        self.request_id = 0
        self._setup_server_command(server_cmd)
        
    def _setup_server_command(self, server_cmd: Optional[List[str]]):
        """Setup server command with auto-detection"""
        if server_cmd:
            self.server_cmd = server_cmd
        else:
            # Auto-detect server location
            current_dir = Path(__file__).parent.parent
            server_path = current_dir / "tvscripts_server" / "server.py"
            
            if server_path.exists():
                self.server_cmd = ["python3", "-m", "mcp.tvscripts_server.server"]
            else:
                self.server_cmd = ["python3", str(server_path)]
    
    def start(self) -> bool:
        """Start MCP server connection"""
        try:
            self.proc = subprocess.Popen(
                self.server_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Test connection with initialize
            response = self._call("initialize", {
                "capabilities": {},
                "clientInfo": {"name": "TradeSystemV1", "version": "1.0.0"}
            })
            
            if response and not response.get("error"):
                logger.info("MCP server connection established")
                return True
            else:
                logger.error(f"Failed to initialize MCP server: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    def stop(self):
        """Stop MCP server connection"""
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            finally:
                self.proc = None
                logger.info("MCP server connection closed")
    
    def _call(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make MCP call to server"""
        if not self.proc:
            logger.error("MCP server not started")
            return None
        
        try:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params
            }
            
            # Send request
            request_line = json.dumps(request) + "\n"
            self.proc.stdin.write(request_line)
            self.proc.stdin.flush()
            
            # Read response with timeout
            response_line = self.proc.stdout.readline()
            if not response_line:
                logger.error("No response from MCP server")
                return None
            
            response = json.loads(response_line.strip())
            
            if "error" in response:
                logger.error(f"MCP server error: {response['error']}")
                return response
            
            return response.get("result", response)
            
        except Exception as e:
            logger.error(f"MCP call failed: {e}")
            return None
    
    def search(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search TradingView scripts
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters
            
        Returns:
            List of script dictionaries
        """
        try:
            params = {"q": query, "limit": limit}
            if filters:
                params["filters"] = filters
            
            response = self._call("tools/call", {
                "name": "search_scripts",
                "arguments": params
            })
            
            if response and "content" in response:
                content = response["content"][0]["text"]
                result = json.loads(content)
                return result.get("results", [])
            
            return []
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_script(self, slug: str) -> Optional[Dict]:
        """
        Get script by slug
        
        Args:
            slug: Script slug identifier
            
        Returns:
            Script dictionary or None
        """
        try:
            response = self._call("tools/call", {
                "name": "get_script",
                "arguments": {"slug": slug}
            })
            
            if response and "content" in response:
                content = response["content"][0]["text"]
                return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.error(f"Get script failed: {e}")
            return None
    
    def analyze_script(self, slug: Optional[str] = None, code: Optional[str] = None) -> Optional[Dict]:
        """
        Analyze Pine Script code
        
        Args:
            slug: Script slug to analyze
            code: Pine Script code to analyze directly
            
        Returns:
            Analysis results or None
        """
        try:
            params = {}
            if slug:
                params["slug"] = slug
            if code:
                params["code"] = code
            
            if not params:
                logger.error("Either slug or code must be provided")
                return None
            
            response = self._call("tools/call", {
                "name": "analyze_pine_script",
                "arguments": params
            })
            
            if response and "content" in response:
                content = response["content"][0]["text"]
                return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.error(f"Analyze script failed: {e}")
            return None
    
    def generate_config(self, slug: str, strategy_name: str = "ImportedFromTV") -> Optional[Dict]:
        """
        Generate TradeSystemV1 configuration from script
        
        Args:
            slug: Script slug
            strategy_name: Name for generated strategy
            
        Returns:
            Generated configuration or None
        """
        try:
            response = self._call("tools/call", {
                "name": "generate_strategy_config",
                "arguments": {
                    "slug": slug,
                    "strategy_name": strategy_name
                }
            })
            
            if response and "content" in response:
                content = response["content"][0]["text"]
                result = json.loads(content)
                return result.get("config")
            
            return None
            
        except Exception as e:
            logger.error(f"Generate config failed: {e}")
            return None
    
    def get_stats(self) -> Optional[Dict]:
        """
        Get database statistics
        
        Returns:
            Statistics dictionary or None
        """
        try:
            response = self._call("tools/call", {
                "name": "get_database_stats",
                "arguments": {}
            })
            
            if response and "content" in response:
                content = response["content"][0]["text"]
                return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            return None
    
    def import_best(self, query: str, strategy_name: Optional[str] = None) -> Optional[Dict]:
        """
        Import best matching script as strategy configuration
        
        Args:
            query: Search query
            strategy_name: Name for imported strategy
            
        Returns:
            Import result with configuration
        """
        try:
            # Search for scripts
            results = self.search(query, limit=1)
            if not results:
                return {"error": "No scripts found for query"}
            
            best_script = results[0]
            slug = best_script["slug"]
            
            if not strategy_name:
                strategy_name = f"TV_{best_script['title'].replace(' ', '_')}"
            
            # Get script details
            script = self.get_script(slug)
            if not script or not script.get("open_source"):
                return {"error": "Script not available or not open-source"}
            
            # Generate configuration
            config = self.generate_config(slug, strategy_name)
            if not config:
                return {"error": "Failed to generate configuration"}
            
            # Add script metadata
            config["provenance"].update({
                "slug": slug,
                "url": script.get("url"),
                "title": script.get("title"),
                "author": script.get("author")
            })
            
            return {
                "success": True,
                "config": config,
                "script_info": best_script
            }
            
        except Exception as e:
            logger.error(f"Import best failed: {e}")
            return {"error": str(e)}
    
    def __enter__(self):
        """Context manager entry"""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start MCP client")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

class AsyncTVScriptsClient:
    """Asynchronous client for TradingView Scripts MCP Server"""
    
    def __init__(self, server_cmd: Optional[List[str]] = None, timeout: int = 30):
        """Initialize async MCP client"""
        self.timeout = timeout
        self.proc = None
        self.request_id = 0
        self.response_queue = asyncio.Queue()
        self._setup_server_command(server_cmd)
    
    def _setup_server_command(self, server_cmd: Optional[List[str]]):
        """Setup server command with auto-detection"""
        if server_cmd:
            self.server_cmd = server_cmd
        else:
            current_dir = Path(__file__).parent.parent
            server_path = current_dir / "tvscripts_server" / "server.py"
            self.server_cmd = ["python3", str(server_path)]
    
    async def start(self) -> bool:
        """Start async MCP server connection"""
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.server_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Test connection
            response = await self._call("initialize", {
                "capabilities": {},
                "clientInfo": {"name": "TradeSystemV1-Async", "version": "1.0.0"}
            })
            
            if response and not response.get("error"):
                logger.info("Async MCP server connection established")
                return True
            else:
                logger.error(f"Failed to initialize async MCP server: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start async MCP server: {e}")
            return False
    
    async def stop(self):
        """Stop async MCP server connection"""
        if self.proc:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
            finally:
                self.proc = None
                logger.info("Async MCP server connection closed")
    
    async def _call(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make async MCP call to server"""
        if not self.proc:
            logger.error("Async MCP server not started")
            return None
        
        try:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params
            }
            
            # Send request
            request_line = json.dumps(request) + "\n"
            self.proc.stdin.write(request_line.encode())
            await self.proc.stdin.drain()
            
            # Read response
            response_line = await asyncio.wait_for(
                self.proc.stdout.readline(),
                timeout=self.timeout
            )
            
            if not response_line:
                logger.error("No response from async MCP server")
                return None
            
            response = json.loads(response_line.decode().strip())
            
            if "error" in response:
                logger.error(f"Async MCP server error: {response['error']}")
                return response
            
            return response.get("result", response)
            
        except Exception as e:
            logger.error(f"Async MCP call failed: {e}")
            return None
    
    async def search(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[Dict]:
        """Async search TradingView scripts"""
        try:
            params = {"q": query, "limit": limit}
            if filters:
                params["filters"] = filters
            
            response = await self._call("tools/call", {
                "name": "search_scripts",
                "arguments": params
            })
            
            if response and "content" in response:
                content = response["content"][0]["text"]
                result = json.loads(content)
                return result.get("results", [])
            
            return []
            
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return []
    
    async def get_script(self, slug: str) -> Optional[Dict]:
        """Async get script by slug"""
        try:
            response = await self._call("tools/call", {
                "name": "get_script",
                "arguments": {"slug": slug}
            })
            
            if response and "content" in response:
                content = response["content"][0]["text"]
                return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.error(f"Async get script failed: {e}")
            return None
    
    async def __aenter__(self):
        """Async context manager entry"""
        if await self.start():
            return self
        else:
            raise RuntimeError("Failed to start async MCP client")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

class TVScriptsClientPool:
    """Connection pool for MCP clients"""
    
    def __init__(self, pool_size: int = 3, server_cmd: Optional[List[str]] = None):
        """Initialize client pool"""
        self.pool_size = pool_size
        self.server_cmd = server_cmd
        self.available_clients = queue.Queue()
        self.all_clients = []
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize client pool"""
        for _ in range(self.pool_size):
            client = TVScriptsClient(self.server_cmd)
            self.all_clients.append(client)
            self.available_clients.put(client)
    
    def get_client(self, timeout: int = 10) -> Optional[TVScriptsClient]:
        """Get client from pool"""
        try:
            client = self.available_clients.get(timeout=timeout)
            if not client.proc:
                client.start()
            return client
        except queue.Empty:
            logger.warning("No available clients in pool")
            return None
    
    def return_client(self, client: TVScriptsClient):
        """Return client to pool"""
        if client in self.all_clients:
            self.available_clients.put(client)
    
    def close_all(self):
        """Close all clients in pool"""
        while not self.available_clients.empty():
            try:
                client = self.available_clients.get_nowait()
                client.stop()
            except queue.Empty:
                break
        
        for client in self.all_clients:
            client.stop()

# Convenience functions for quick operations
def search_scripts(query: str, limit: int = 20, **kwargs) -> List[Dict]:
    """Quick search function"""
    with TVScriptsClient() as client:
        return client.search(query, limit, kwargs.get('filters'))

def get_script_by_slug(slug: str) -> Optional[Dict]:
    """Quick get script function"""
    with TVScriptsClient() as client:
        return client.get_script(slug)

def import_strategy(query: str, strategy_name: Optional[str] = None) -> Optional[Dict]:
    """Quick import function"""
    with TVScriptsClient() as client:
        return client.import_best(query, strategy_name)