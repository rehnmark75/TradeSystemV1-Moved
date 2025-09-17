#!/usr/bin/env python3
"""
Simple RAG Query Script for Claude Code
=======================================

This script provides easy ways to query the RAG system from Claude Code.
"""

import requests
import json
import sys
from typing import Dict, Any

class SimpleRAGQuery:
    def __init__(self, base_url: str = "http://localhost:8090"):
        self.base_url = base_url

    def health_check(self) -> Dict:
        """Check if RAG system is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def search_with_curl(self, query: str, search_type: str = "indicators") -> str:
        """Generate curl command for searching"""
        if search_type == "indicators":
            endpoint = "/search/indicators"
        elif search_type == "templates":
            endpoint = "/search/templates"
        else:
            endpoint = "/search/indicators"

        curl_cmd = f'''curl -s -X POST "{self.base_url}{endpoint}" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "{query}", "limit": 5}}' | jq .'''

        return curl_cmd

    def query_similar(self, indicator_id: str) -> str:
        """Generate curl command for finding similar indicators"""
        curl_cmd = f'''curl -s "{self.base_url}/similar/{indicator_id}?n_results=3" | jq .'''
        return curl_cmd

    def compose_strategy_curl(self, description: str) -> str:
        """Generate curl command for strategy composition"""
        curl_cmd = f'''curl -s -X POST "{self.base_url}/compose/strategy" \\
  -H "Content-Type: application/json" \\
  -d '{{"description": "{description}", "market_condition": "trending", "trading_style": "day_trading"}}' | jq .'''

        return curl_cmd

def print_usage():
    """Print usage instructions"""
    print("RAG System Query Methods")
    print("=" * 40)
    print()
    print("1. HEALTH CHECK:")
    print("   python3 query_rag_system.py health")
    print()
    print("2. DATABASE STATS:")
    print("   python3 query_rag_system.py stats")
    print()
    print("3. SEARCH INDICATORS:")
    print("   python3 query_rag_system.py search 'RSI momentum'")
    print()
    print("4. SEARCH TEMPLATES:")
    print("   python3 query_rag_system.py templates 'scalping strategy'")
    print()
    print("5. FIND SIMILAR:")
    print("   python3 query_rag_system.py similar 'indicator-id'")
    print()
    print("6. COMPOSE STRATEGY:")
    print("   python3 query_rag_system.py compose 'momentum strategy for EURUSD'")
    print()
    print("7. DIRECT CURL COMMANDS:")
    print("   python3 query_rag_system.py curl-search 'trend indicators'")

def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    rag = SimpleRAGQuery()
    command = sys.argv[1].lower()

    if command == "health":
        result = rag.health_check()
        print(json.dumps(result, indent=2))

    elif command == "stats":
        result = rag.get_stats()
        print(json.dumps(result, indent=2))

    elif command == "search" and len(sys.argv) > 2:
        query = sys.argv[2]
        curl_cmd = rag.search_with_curl(query, "indicators")
        print("Execute this curl command:")
        print(curl_cmd)

    elif command == "templates" and len(sys.argv) > 2:
        query = sys.argv[2]
        curl_cmd = rag.search_with_curl(query, "templates")
        print("Execute this curl command:")
        print(curl_cmd)

    elif command == "similar" and len(sys.argv) > 2:
        indicator_id = sys.argv[2]
        curl_cmd = rag.query_similar(indicator_id)
        print("Execute this curl command:")
        print(curl_cmd)

    elif command == "compose" and len(sys.argv) > 2:
        description = sys.argv[2]
        curl_cmd = rag.compose_strategy_curl(description)
        print("Execute this curl command:")
        print(curl_cmd)

    elif command == "curl-search" and len(sys.argv) > 2:
        query = sys.argv[2]
        curl_cmd = rag.search_with_curl(query)
        print(curl_cmd)

    else:
        print_usage()

if __name__ == "__main__":
    main()