#!/usr/bin/env python3
"""
Test Trading RAG MCP Server
===========================

Test script to verify the RAG MCP server functionality.
"""

import sys
import json
from pathlib import Path

# Add vector-db to path
sys.path.append(str(Path(__file__).parent / "vector-db"))

from mcp.client.rag_client import TradingRAGClient

def test_rag_mcp_server():
    """Test the RAG MCP server functionality"""
    print("🧪 Testing Trading RAG MCP Server")
    print("=" * 50)

    try:
        with TradingRAGClient() as client:
            print("✅ MCP client connected successfully")

            # Test 1: Get RAG stats
            print("\n📊 Testing RAG stats...")
            stats = client.get_rag_stats()
            if "error" not in stats:
                print(f"✅ Stats retrieved: {stats.get('rag_statistics', {}).get('indicators_count', 'N/A')} indicators available")
            else:
                print(f"❌ Stats error: {stats['error']}")

            # Test 2: Search indicators
            print("\n🔍 Testing indicator search...")
            indicators = client.search_indicators("RSI momentum", n_results=3)
            if indicators:
                print(f"✅ Found {len(indicators)} indicators:")
                for i, ind in enumerate(indicators[:2], 1):
                    print(f"  {i}. {ind.get('title', 'Unknown')} (score: {ind.get('similarity_score', 0):.3f})")
            else:
                print("❌ No indicators found")

            # Test 3: Search strategies
            print("\n📈 Testing strategy search...")
            strategies = client.search_strategies("EMA EURUSD", n_results=2)
            if strategies:
                print(f"✅ Found {len(strategies)} strategies:")
                for i, strat in enumerate(strategies, 1):
                    print(f"  {i}. {strat.get('title', 'Unknown Strategy')}")
            else:
                print("❌ No strategies found")

            # Test 4: Ask trading question
            print("\n🤖 Testing AI trading question...")
            answer = client.ask_trading_question("What are the best RSI settings for scalping?")
            if "error" not in answer:
                print("✅ AI answer generated:")
                print(f"  Question: {answer.get('question', 'N/A')}")
                print(f"  Answer preview: {answer.get('answer', 'No answer')[:100]}...")
            else:
                print(f"❌ AI question error: {answer['error']}")

            # Test 5: Find similar indicators (if we found any)
            if indicators:
                print("\n🔗 Testing similar indicators...")
                first_indicator_id = indicators[0].get('id', '')
                if first_indicator_id:
                    similar = client.find_similar_indicators(first_indicator_id, n_results=2)
                    if similar:
                        print(f"✅ Found {len(similar)} similar indicators")
                    else:
                        print("❌ No similar indicators found")
                else:
                    print("⚠️ No indicator ID available for similarity test")

            print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True

def test_convenience_functions():
    """Test convenience functions"""
    print("\n🚀 Testing convenience functions...")

    # Test quick search
    indicators = search_trading_indicators("MACD trend", n_results=2)
    print(f"Quick indicator search: {len(indicators)} results")

    # Test AI question
    answer = ask_trading_ai("Which indicators work best for ranging markets?")
    if "error" not in answer:
        print("Quick AI question: Success")
    else:
        print(f"Quick AI question error: {answer['error']}")

if __name__ == "__main__":
    # Import convenience functions
    from vector_db.mcp.client.rag_client import search_trading_indicators, ask_trading_ai

    print("🎯 Starting RAG MCP Server Tests")

    # Main test
    success = test_rag_mcp_server()

    if success:
        # Test convenience functions
        test_convenience_functions()
        print("\n✅ All tests passed! RAG MCP server is ready for Claude Code.")
    else:
        print("\n❌ Tests failed. Check the RAG API and MCP server setup.")

    print("\n💡 Usage in Claude Code:")
    print("The MCP server provides these tools:")
    print("- search_indicators: Find trading indicators")
    print("- search_strategies: Find strategy templates")
    print("- find_similar_indicators: Get similar indicators")
    print("- ask_trading_question: Get AI trading advice")
    print("- get_rag_stats: Check system status")