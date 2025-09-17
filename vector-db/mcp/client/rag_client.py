"""
Trading RAG MCP Client
=====================

Client for communicating with the Trading RAG MCP Server.
Provides easy-to-use interface for Claude Code and other applications.
"""

import json
import logging
import subprocess
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class TradingRAGClient:
    """Client for Trading RAG MCP Server"""

    def __init__(self, server_cmd: Optional[List[str]] = None, timeout: int = 30):
        """
        Initialize RAG MCP client

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
            server_path = current_dir / "rag_server" / "server.py"

            if server_path.exists():
                self.server_cmd = ["python3", str(server_path)]
            else:
                self.server_cmd = ["python3", "-m", "vector-db.mcp.rag_server.server"]

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
                "clientInfo": {"name": "TradeSystemV1-RAG", "version": "1.0.0"}
            })

            if response and not response.get("error"):
                logger.info("RAG MCP server connection established")
                return True
            else:
                logger.error(f"Failed to initialize RAG MCP server: {response}")
                return False

        except Exception as e:
            logger.error(f"Failed to start RAG MCP server: {e}")
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
                logger.info("RAG MCP server connection closed")

    def _call(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make MCP call to server"""
        if not self.proc:
            logger.error("RAG MCP server not started")
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
                logger.error("No response from RAG MCP server")
                return None

            response = json.loads(response_line.strip())

            if "error" in response:
                logger.error(f"RAG MCP server error: {response['error']}")
                return response

            return response.get("result", response)

        except Exception as e:
            logger.error(f"RAG MCP call failed: {e}")
            return None

    def search_indicators(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for trading indicators

        Args:
            query: Natural language search query
            n_results: Number of results to return
            filters: Optional filters

        Returns:
            List of indicator dictionaries
        """
        try:
            params = {"query": query, "n_results": n_results}
            if filters:
                params["filters"] = filters

            response = self._call("tools/call", {
                "name": "search_indicators",
                "arguments": params
            })

            if response and "content" in response:
                content = response["content"][0]["text"]
                result = json.loads(content)
                return result.get("results", [])

            return []

        except Exception as e:
            logger.error(f"Indicator search failed: {e}")
            return []

    def search_strategies(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for strategy templates

        Args:
            query: Search query for strategies
            n_results: Number of results to return

        Returns:
            List of strategy dictionaries
        """
        try:
            response = self._call("tools/call", {
                "name": "search_strategies",
                "arguments": {"query": query, "n_results": n_results}
            })

            if response and "content" in response:
                content = response["content"][0]["text"]
                result = json.loads(content)
                return result.get("results", [])

            return []

        except Exception as e:
            logger.error(f"Strategy search failed: {e}")
            return []

    def find_similar_indicators(self, indicator_id: str, n_results: int = 3) -> List[Dict]:
        """
        Find similar indicators

        Args:
            indicator_id: Reference indicator ID
            n_results: Number of similar indicators to find

        Returns:
            List of similar indicators
        """
        try:
            response = self._call("tools/call", {
                "name": "find_similar_indicators",
                "arguments": {"indicator_id": indicator_id, "n_results": n_results}
            })

            if response and "content" in response:
                content = response["content"][0]["text"]
                result = json.loads(content)
                return result.get("similar_indicators", [])

            return []

        except Exception as e:
            logger.error(f"Similar indicators search failed: {e}")
            return []

    def ask_trading_question(self, question: str, include_examples: bool = True, max_indicators: int = 5) -> Dict:
        """
        Ask an intelligent trading question

        Args:
            question: Your trading question
            include_examples: Include practical examples
            max_indicators: Maximum indicators to reference

        Returns:
            AI-generated answer with recommendations
        """
        try:
            response = self._call("tools/call", {
                "name": "ask_trading_question",
                "arguments": {
                    "question": question,
                    "include_examples": include_examples,
                    "max_indicators": max_indicators
                }
            })

            if response and "content" in response:
                content = response["content"][0]["text"]
                return json.loads(content)

            return {"error": "No response received"}

        except Exception as e:
            logger.error(f"Trading question failed: {e}")
            return {"error": str(e)}

    def get_rag_stats(self) -> Dict:
        """
        Get RAG system statistics

        Returns:
            System statistics and health info
        """
        try:
            response = self._call("tools/call", {
                "name": "get_rag_stats",
                "arguments": {}
            })

            if response and "content" in response:
                content = response["content"][0]["text"]
                return json.loads(content)

            return {"error": "No response received"}

        except Exception as e:
            logger.error(f"RAG stats failed: {e}")
            return {"error": str(e)}

    def quick_search(self, query: str) -> Dict:
        """
        Quick search combining indicators and strategies

        Args:
            query: Search query

        Returns:
            Combined results from indicators and strategies
        """
        indicators = self.search_indicators(query, n_results=3)
        strategies = self.search_strategies(query, n_results=2)

        return {
            "query": query,
            "indicators": indicators,
            "strategies": strategies,
            "total_results": len(indicators) + len(strategies)
        }

    def __enter__(self):
        """Context manager entry"""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start RAG MCP client")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

# Convenience functions for quick operations
def search_trading_indicators(query: str, n_results: int = 5) -> List[Dict]:
    """Quick indicator search function"""
    with TradingRAGClient() as client:
        return client.search_indicators(query, n_results)

def search_trading_strategies(query: str, n_results: int = 5) -> List[Dict]:
    """Quick strategy search function"""
    with TradingRAGClient() as client:
        return client.search_strategies(query, n_results)

def ask_trading_ai(question: str) -> Dict:
    """Quick AI question function"""
    with TradingRAGClient() as client:
        return client.ask_trading_question(question)

def get_trading_stats() -> Dict:
    """Quick stats function"""
    with TradingRAGClient() as client:
        return client.get_rag_stats()