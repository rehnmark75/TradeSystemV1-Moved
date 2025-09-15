#!/usr/bin/env python3
"""
Claude RAG Helper - Simple interface for Claude Code to use the RAG system
"""

import json
import subprocess
import sys

def query_rag(command: str, *args) -> str:
    """
    Simple function for Claude Code to query the RAG system

    Usage examples:
    - query_rag("health") - Check system health
    - query_rag("stats") - Get database statistics
    - query_rag("recommend", "trend analysis") - Get recommendations
    - query_rag("search-indicators", "moving average") - Search indicators
    - query_rag("search-templates", "scalping") - Search templates
    """
    try:
        cmd = ["python3", "rag_interface.py", command] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/hr/Projects/TradeSystemV1")

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 claude_rag_helper.py <command> [args...]")
        sys.exit(1)

    result = query_rag(*sys.argv[1:])
    print(result)