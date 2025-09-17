"""
Trading RAG MCP Server
=====================

MCP server providing intelligent trading strategy search and recommendations.
Integrates with the enhanced RAG system for semantic indicator and strategy search.
"""

import json
import asyncio
import logging
import httpx
from typing import Any, Sequence, Optional, Dict, List
from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    CallToolRequest,
    ListResourcesRequest,
    ReadResourceRequest,
    Resource
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingRAGServer:
    """Trading RAG MCP Server for intelligent strategy search"""

    def __init__(self, rag_url: str = "http://localhost:8090"):
        self.rag_url = rag_url
        self.server = Server("trading-rag")
        self._setup_tools()
        self._setup_resources()

    def _setup_tools(self):
        """Setup MCP tools for RAG operations"""

        @self.server.call_tool()
        async def search_indicators(arguments: dict) -> Sequence[TextContent]:
            """Search for trading indicators using semantic RAG search

            Search the enhanced RAG database for trading indicators based on:
            - Natural language queries (e.g. "RSI momentum for scalping")
            - Technical analysis concepts (e.g. "trend following", "mean reversion")
            - Market conditions (e.g. "ranging markets", "high volatility")
            - Trading styles (e.g. "scalping", "swing trading", "day trading")

            Args:
                query (str): Natural language search query
                n_results (int, optional): Number of results to return (default: 5, max: 20)
                filters (dict, optional): Additional filters (collection, complexity_level, etc.)

            Returns:
                List of matched indicators with similarity scores and metadata
            """
            try:
                query = arguments.get("query", "")
                n_results = min(int(arguments.get("n_results", 5)), 20)
                filters = arguments.get("filters", {})

                if not query:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Query parameter is required"})
                    )]

                # Call RAG API
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.rag_url}/search/indicators",
                        json={
                            "query": query,
                            "n_results": n_results,
                            "filters": filters
                        },
                        timeout=30.0
                    )

                if response.status_code == 200:
                    results = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "query": query,
                            "results": results,
                            "total": len(results),
                            "search_type": "indicators"
                        }, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"RAG API error: {response.text}"})
                    )]

            except Exception as e:
                logger.error(f"Indicator search failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]

        @self.server.call_tool()
        async def search_strategies(arguments: dict) -> Sequence[TextContent]:
            """Search for trading strategy templates

            Find optimized trading strategy templates based on:
            - Market pairs (e.g. "EURUSD", "GBPJPY")
            - Strategy types (e.g. "EMA", "MACD", "SMC")
            - Performance criteria (e.g. "high win rate", "profitable")
            - Timeframes (e.g. "1h", "4h", "1d")

            Args:
                query (str): Search query for strategy templates
                n_results (int, optional): Number of results (default: 5, max: 15)

            Returns:
                List of strategy templates with performance data
            """
            try:
                query = arguments.get("query", "")
                n_results = min(int(arguments.get("n_results", 5)), 15)

                if not query:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Query parameter is required"})
                    )]

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.rag_url}/search/templates",
                        json={
                            "query": query,
                            "n_results": n_results
                        },
                        timeout=30.0
                    )

                if response.status_code == 200:
                    results = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "query": query,
                            "results": results,
                            "total": len(results),
                            "search_type": "strategy_templates"
                        }, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"RAG API error: {response.text}"})
                    )]

            except Exception as e:
                logger.error(f"Strategy search failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]

        @self.server.call_tool()
        async def find_similar_indicators(arguments: dict) -> Sequence[TextContent]:
            """Find indicators similar to a given indicator

            Discover indicators with similar functionality or characteristics.
            Useful for finding alternatives or building indicator combinations.

            Args:
                indicator_id (str): ID of the reference indicator
                n_results (int, optional): Number of similar indicators (default: 3, max: 10)

            Returns:
                List of similar indicators ranked by similarity
            """
            try:
                indicator_id = arguments.get("indicator_id", "")
                n_results = min(int(arguments.get("n_results", 3)), 10)

                if not indicator_id:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "indicator_id parameter is required"})
                    )]

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.rag_url}/similar/{indicator_id}",
                        params={"n_results": n_results},
                        timeout=30.0
                    )

                if response.status_code == 200:
                    results = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "reference_indicator": indicator_id,
                            "similar_indicators": results,
                            "total": len(results)
                        }, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"RAG API error: {response.text}"})
                    )]

            except Exception as e:
                logger.error(f"Similar indicators search failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]

        @self.server.call_tool()
        async def ask_trading_question(arguments: dict) -> Sequence[TextContent]:
            """Ask intelligent trading questions to the RAG system

            Get AI-powered answers to trading questions using the enhanced RAG system.
            Combines semantic search with trading knowledge for comprehensive responses.

            Examples:
            - "What are the best RSI settings for scalping EURUSD?"
            - "Which indicators work well in ranging markets?"
            - "How to combine MACD with EMA for trend following?"
            - "What's the difference between LuxAlgo and Zeiierman indicators?"

            Args:
                question (str): Your trading question in natural language
                include_examples (bool, optional): Include practical examples (default: true)
                max_indicators (int, optional): Max indicators to reference (default: 5)

            Returns:
                AI-generated answer with relevant indicator recommendations
            """
            try:
                question = arguments.get("question", "")
                include_examples = arguments.get("include_examples", True)
                max_indicators = min(int(arguments.get("max_indicators", 5)), 10)

                if not question:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Question parameter is required"})
                    )]

                # First, search for relevant indicators
                async with httpx.AsyncClient() as client:
                    search_response = await client.post(
                        f"{self.rag_url}/search/indicators",
                        json={
                            "query": question,
                            "n_results": max_indicators
                        },
                        timeout=30.0
                    )

                if search_response.status_code != 200:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"Search failed: {search_response.text}"})
                    )]

                indicators = search_response.json()

                # Generate intelligent response
                response_data = {
                    "question": question,
                    "answer": self._generate_trading_answer(question, indicators, include_examples),
                    "relevant_indicators": indicators[:3],  # Top 3 most relevant
                    "additional_suggestions": indicators[3:max_indicators] if len(indicators) > 3 else []
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )]

            except Exception as e:
                logger.error(f"Trading question failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]

        @self.server.call_tool()
        async def get_rag_stats(arguments: dict) -> Sequence[TextContent]:
            """Get RAG system statistics and health status

            Returns current status of the RAG database including:
            - Number of indicators and templates available
            - System health and performance metrics
            - Recent sync status and data freshness

            Returns:
                RAG system statistics and status information
            """
            try:
                async with httpx.AsyncClient() as client:
                    # Get both health and stats
                    health_response = await client.get(f"{self.rag_url}/health", timeout=10.0)
                    stats_response = await client.get(f"{self.rag_url}/stats", timeout=10.0)

                health_data = health_response.json() if health_response.status_code == 200 else {"error": "Health check failed"}
                stats_data = stats_response.json() if stats_response.status_code == 200 else {"error": "Stats unavailable"}

                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "rag_health": health_data,
                        "rag_statistics": stats_data,
                        "api_status": "operational" if health_response.status_code == 200 else "degraded"
                    }, indent=2)
                )]

            except Exception as e:
                logger.error(f"RAG stats failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]

    def _setup_resources(self):
        """Setup MCP resources for RAG data"""

        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            """List available RAG resources"""
            return [
                Resource(
                    uri="rag://indicators/collections",
                    name="Indicator Collections",
                    description="Available indicator collections (LuxAlgo, Zeiierman, etc.)",
                    mimeType="application/json"
                ),
                Resource(
                    uri="rag://strategies/templates",
                    name="Strategy Templates",
                    description="Optimized strategy templates with performance data",
                    mimeType="application/json"
                ),
                Resource(
                    uri="rag://help/usage",
                    name="RAG Usage Guide",
                    description="How to effectively use the RAG system for trading research",
                    mimeType="text/markdown"
                )
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read RAG resource content"""
            if uri == "rag://help/usage":
                return """# Trading RAG System Usage Guide

## Available Tools

### 1. search_indicators
Search for trading indicators using natural language:
- "RSI momentum for scalping"
- "trend following indicators"
- "LuxAlgo volatility tools"

### 2. search_strategies
Find optimized strategy templates:
- "EURUSD EMA strategy"
- "high win rate MACD setups"
- "4h timeframe strategies"

### 3. find_similar_indicators
Discover similar indicators:
- Reference an indicator ID to find alternatives
- Build complementary indicator combinations

### 4. ask_trading_question
Get AI-powered trading advice:
- "Best RSI settings for scalping?"
- "How to combine MACD with EMA?"
- "Which indicators for ranging markets?"

### 5. get_rag_stats
Check system status and available data

## Tips for Better Results

1. **Be Specific**: Include trading style, timeframe, market conditions
2. **Use Trading Terms**: RSI, MACD, EMA, scalping, swing trading, etc.
3. **Ask Questions**: The system understands natural language questions
4. **Explore Collections**: Try LuxAlgo, Zeiierman, LazyBear, ChrisMoody
5. **Combine Searches**: Use multiple tools to build comprehensive strategies

## Example Workflows

### Finding a Scalping Strategy
1. `search_indicators` with "scalping EURUSD RSI"
2. `find_similar_indicators` for alternatives
3. `search_strategies` for "EURUSD scalping templates"
4. `ask_trading_question` about "best scalping timeframes"

### Building a Trend Following System
1. `search_indicators` with "trend following EMA MACD"
2. `ask_trading_question` about "combining trend indicators"
3. `search_strategies` for proven trend strategies
"""

            elif uri == "rag://indicators/collections":
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{self.rag_url}/stats")
                        if response.status_code == 200:
                            return json.dumps(response.json(), indent=2)
                except:
                    pass
                return json.dumps({"error": "Unable to fetch collections data"})

            elif uri == "rag://strategies/templates":
                return json.dumps({
                    "info": "Strategy templates with optimization data",
                    "usage": "Use search_strategies tool to find specific templates",
                    "data_source": "EMA, MACD, and SMC optimization results"
                }, indent=2)

            else:
                raise ValueError(f"Unknown resource: {uri}")

    def _generate_trading_answer(self, question: str, indicators: List[Dict], include_examples: bool = True) -> str:
        """Generate intelligent trading answer based on question and relevant indicators"""

        if not indicators:
            return f"I couldn't find specific indicators related to your question: '{question}'. Try rephrasing your question or using more general trading terms like 'RSI', 'MACD', 'EMA', etc."

        # Extract key information from indicators
        top_indicator = indicators[0]
        indicator_names = [ind.get('title', 'Unknown') for ind in indicators[:3]]
        collections = list(set([ind.get('metadata', {}).get('collection', 'Unknown') for ind in indicators[:3]]))

        # Generate contextual answer based on question type
        answer_parts = []

        # Introduction
        answer_parts.append(f"Based on your question about '{question}', I found {len(indicators)} relevant indicators in the database.")

        # Top recommendation
        answer_parts.append(f"\n**Top Recommendation:** {top_indicator.get('title', 'Unknown')}")
        answer_parts.append(f"- Collection: {top_indicator.get('metadata', {}).get('collection', 'Unknown')}")
        answer_parts.append(f"- Complexity: {top_indicator.get('metadata', {}).get('complexity_level', 'Unknown')}")
        answer_parts.append(f"- Similarity Score: {top_indicator.get('similarity_score', 0):.2f}")

        # Additional options
        if len(indicators) > 1:
            answer_parts.append(f"\n**Other Options:** {', '.join(indicator_names[1:])}")

        # Collections summary
        if collections:
            answer_parts.append(f"\n**Available Collections:** {', '.join(collections)}")

        # Practical advice based on question keywords
        if any(word in question.lower() for word in ['scalping', 'scalp']):
            answer_parts.append(f"\n**Scalping Tips:** Look for indicators with fast signal generation and low lag. Consider shorter timeframes (1m, 5m) and focus on high-frequency signals.")

        elif any(word in question.lower() for word in ['swing', 'position']):
            answer_parts.append(f"\n**Swing Trading Tips:** Use indicators that work well on 4h and daily timeframes. Focus on trend-following indicators with good signal quality over noise.")

        elif any(word in question.lower() for word in ['ranging', 'sideways']):
            answer_parts.append(f"\n**Ranging Market Tips:** Look for oscillators like RSI, Stochastic, or Williams %R. Avoid pure trend-following indicators in ranging conditions.")

        elif any(word in question.lower() for word in ['trend', 'trending']):
            answer_parts.append(f"\n**Trend Following Tips:** Combine moving averages (EMA/SMA) with momentum indicators (MACD, RSI). Use multiple timeframe confirmation.")

        # Usage recommendation
        answer_parts.append(f"\n**Next Steps:** Use the 'find_similar_indicators' tool with '{top_indicator.get('id', '')}' to explore alternatives, or search for strategy templates that use these indicators.")

        return "".join(answer_parts)

    async def run(self):
        """Run the RAG MCP server"""
        logger.info("🚀 Starting Trading RAG MCP Server...")

        # Test RAG API connection
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.rag_url}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info("✅ RAG API connection successful")
                else:
                    logger.warning(f"⚠️ RAG API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to RAG API: {e}")

        # Start MCP server
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

# Create server instance
rag_server = TradingRAGServer()

if __name__ == "__main__":
    asyncio.run(rag_server.run())