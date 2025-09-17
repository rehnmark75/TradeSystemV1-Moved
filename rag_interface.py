#!/usr/bin/env python3
"""
RAG System Interface for Claude Code
===================================

This script provides a simple interface to interact with the TradeSystemV1 RAG system,
making it accessible to Claude Code for intelligent analysis and recommendations.
"""

import requests
import json
import sys
from typing import Dict, List, Optional

class RAGInterface:
    def __init__(self, base_url: str = "http://localhost:8090"):
        self.base_url = base_url

    def health_check(self) -> Dict:
        """Check if RAG system is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}

    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_recommendations(self, query: str) -> Dict:
        """Get recommendations for a trading query"""
        try:
            response = requests.get(f"{self.base_url}/recommendations/{query}")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def search_indicators(self, query: str, limit: int = 5) -> Dict:
        """Search for indicators using semantic search"""
        try:
            payload = {"query": query, "limit": limit}
            response = requests.post(
                f"{self.base_url}/search/indicators",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def search_templates(self, query: str, limit: int = 5) -> Dict:
        """Search for strategy templates using semantic search"""
        try:
            payload = {"query": query, "limit": limit}
            response = requests.post(
                f"{self.base_url}/search/templates",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def get_similar_indicators(self, indicator_id: str, limit: int = 5) -> Dict:
        """Find indicators similar to a given indicator"""
        try:
            params = {"limit": limit}
            response = requests.get(f"{self.base_url}/similar/{indicator_id}", params=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def compose_strategy(self, description: str, market_condition: str = "trending",
                        trading_style: str = "day_trading", complexity_level: str = "intermediate") -> Dict:
        """Compose a trading strategy based on requirements"""
        try:
            payload = {
                "description": description,
                "market_condition": market_condition,
                "trading_style": trading_style,
                "complexity_level": complexity_level
            }
            response = requests.post(
                f"{self.base_url}/compose/strategy",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def enhanced_search(self, query: str, n_results: int = 5, include_performance: bool = True,
                       user_context: Optional[Dict] = None) -> Dict:
        """Enhanced search with intelligent processing and performance weighting"""
        try:
            payload = {
                "query": query,
                "n_results": n_results,
                "include_performance": include_performance,
                "user_context": user_context
            }
            response = requests.post(
                f"{self.base_url}/search/enhanced",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def analyze_query(self, query: str) -> Dict:
        """Analyze query with intelligent processing"""
        try:
            response = requests.post(
                f"{self.base_url}/query/analyze",
                params={"query": query}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def get_market_regime(self, epic: str = "CS.D.EURUSD.MINI.IP") -> Dict:
        """Get current market regime analysis"""
        try:
            response = requests.get(f"{self.base_url}/market/regime", params={"epic": epic})
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def analyze_compatibility(self, indicators: List[str]) -> Dict:
        """Analyze compatibility between indicators"""
        try:
            response = requests.post(
                f"{self.base_url}/indicators/compatibility",
                json=indicators,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def enhance_database(self) -> Dict:
        """Trigger enhanced database processing"""
        try:
            response = requests.post(f"{self.base_url}/data/enhance")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

def main():
    """CLI interface for RAG system"""
    if len(sys.argv) < 2:
        print("Usage: python rag_interface.py <command> [args]")
        print("Commands:")
        print("  health                           - Check system health")
        print("  stats                           - Get database statistics")
        print("  recommend <query>               - Get recommendations")
        print("  search-indicators <query>       - Search indicators")
        print("  search-templates <query>        - Search templates")
        print("  similar <indicator_id>          - Find similar indicators")
        print("  compose <requirements>          - Compose strategy")
        print("")
        print("Enhanced Commands:")
        print("  enhanced-search <query>         - Enhanced semantic search with performance")
        print("  analyze-query <query>           - Analyze query with AI processing")
        print("  market-regime [epic]            - Get current market regime")
        print("  compatibility <ind1,ind2,...>   - Analyze indicator compatibility")
        print("  enhance                         - Trigger enhanced data processing")
        return

    rag = RAGInterface()
    command = sys.argv[1].lower()

    if command == "health":
        result = rag.health_check()
    elif command == "stats":
        result = rag.get_stats()
    elif command == "recommend" and len(sys.argv) > 2:
        result = rag.get_recommendations(sys.argv[2])
    elif command == "search-indicators" and len(sys.argv) > 2:
        result = rag.search_indicators(sys.argv[2])
    elif command == "search-templates" and len(sys.argv) > 2:
        result = rag.search_templates(sys.argv[2])
    elif command == "similar" and len(sys.argv) > 2:
        result = rag.get_similar_indicators(sys.argv[2])
    elif command == "compose" and len(sys.argv) > 2:
        result = rag.compose_strategy(sys.argv[2])
    elif command == "enhanced-search" and len(sys.argv) > 2:
        result = rag.enhanced_search(sys.argv[2])
    elif command == "analyze-query" and len(sys.argv) > 2:
        result = rag.analyze_query(sys.argv[2])
    elif command == "market-regime":
        epic = sys.argv[2] if len(sys.argv) > 2 else "CS.D.EURUSD.MINI.IP"
        result = rag.get_market_regime(epic)
    elif command == "compatibility" and len(sys.argv) > 2:
        indicators = sys.argv[2].split(',')
        result = rag.analyze_compatibility(indicators)
    elif command == "enhance":
        result = rag.enhance_database()
    else:
        print(f"Unknown command or missing arguments: {command}")
        return

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()