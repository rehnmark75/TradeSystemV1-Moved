#!/usr/bin/env python3
"""
Simple Direct RAG Queries for Claude Code
==========================================

Working queries that bypass the async issues in the current vector-db service.
"""

import requests
import json

def query_tradingview_database():
    """Query TradingView database directly"""
    print("=== TradingView Database Query ===")

    # Example queries you can use in Claude Code
    examples = [
        {
            "description": "Search for LuxAlgo indicators",
            "command": "docker exec postgres psql -U postgres -d forex -c \"SELECT slug, title, author FROM tradingview.scripts WHERE is_luxalgo = true LIMIT 5;\""
        },
        {
            "description": "Find RSI-related scripts",
            "command": "docker exec postgres psql -U postgres -d forex -c \"SELECT slug, title, 'RSI' as indicator FROM tradingview.scripts WHERE 'RSI' = ANY(indicators) LIMIT 3;\""
        },
        {
            "description": "Search by complexity",
            "command": "docker exec postgres psql -U postgres -d forex -c \"SELECT slug, title, complexity_score FROM tradingview.scripts WHERE complexity_score > 0.7 ORDER BY complexity_score DESC LIMIT 5;\""
        },
        {
            "description": "Find indicators by collection",
            "command": "docker exec postgres psql -U postgres -d forex -c \"SELECT COUNT(*) as count, CASE WHEN is_luxalgo THEN 'LuxAlgo' WHEN is_zeiierman THEN 'Zeiierman' WHEN is_lazybear THEN 'LazyBear' ELSE 'Other' END as collection FROM tradingview.scripts GROUP BY collection;\""
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")

    print("\n=== Performance Data Queries ===")
    performance_examples = [
        {
            "description": "Best EMA strategies",
            "command": "docker exec postgres psql -U postgres -d forex -c \"SELECT epic, best_win_rate, best_net_pips FROM ema_best_parameters ORDER BY best_net_pips DESC LIMIT 5;\""
        },
        {
            "description": "MACD performance",
            "command": "docker exec postgres psql -U postgres -d forex -c \"SELECT epic, best_composite_score, best_win_rate FROM macd_best_parameters ORDER BY best_composite_score DESC LIMIT 5;\""
        }
    ]

    for i, example in enumerate(performance_examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")

def working_api_examples():
    """Show working API examples"""
    print("\n=== Working API Queries ===")

    # Test basic endpoints
    try:
        # Health check (always works)
        health = requests.get("http://localhost:8090/health").json()
        print("✅ Health Check:", json.dumps(health, indent=2))

        # Stats (always works)
        stats = requests.get("http://localhost:8090/stats").json()
        print("\n✅ Database Stats:", json.dumps(stats, indent=2))

    except Exception as e:
        print(f"❌ API Error: {e}")

def claude_code_examples():
    """Examples specifically for Claude Code usage"""
    print("\n=== Claude Code Examples ===")
    print("""
# 1. QUICK DATABASE QUERIES
# Find all LuxAlgo indicators:
docker exec postgres psql -U postgres -d forex -c "SELECT title, complexity_score FROM tradingview.scripts WHERE is_luxalgo = true ORDER BY complexity_score DESC;"

# 2. PERFORMANCE ANALYSIS
# Best performing EMA strategies:
docker exec postgres psql -U postgres -d forex -c "SELECT epic, best_win_rate, best_net_pips, best_timeframe FROM ema_best_parameters WHERE best_net_pips > 100 ORDER BY best_net_pips DESC;"

# 3. SEARCH BY INDICATORS
# Find all scripts using MACD:
docker exec postgres psql -U postgres -d forex -c "SELECT title, author, complexity_score FROM tradingview.scripts WHERE 'MACD' = ANY(indicators);"

# 4. COMPLEXITY ANALYSIS
# Find advanced indicators:
docker exec postgres psql -U postgres -d forex -c "SELECT title, complexity_score, CASE WHEN is_luxalgo THEN 'LuxAlgo' WHEN is_zeiierman THEN 'Zeiierman' ELSE 'Other' END as collection FROM tradingview.scripts WHERE complexity_score > 0.8;"

# 5. RAG SYSTEM STATUS
python3 query_rag_system.py health
python3 query_rag_system.py stats
""")

if __name__ == "__main__":
    print("Simple RAG Queries for Claude Code")
    print("=" * 50)

    # Show working API queries
    working_api_examples()

    # Show database queries
    query_tradingview_database()

    # Show Claude Code specific examples
    claude_code_examples()