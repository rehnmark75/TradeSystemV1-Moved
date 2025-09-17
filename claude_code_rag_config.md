# Trading RAG MCP Server - Claude Code Configuration

## 🚀 **MCP Server Ready for Claude Code!**

Your Trading RAG MCP server is now set up and ready to use in Claude Code conversations. This provides natural language access to your enhanced RAG system for intelligent trading strategy searches.

## 📋 **Claude Code Configuration**

### Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "trading-rag": {
      "command": "python3",
      "args": ["/home/hr/Projects/TradeSystemV1/vector-db/mcp/rag_server/server.py"],
      "env": {
        "RAG_URL": "http://localhost:8090"
      }
    }
  }
}
```

## 🎯 **Available Tools in Claude Code**

Once configured, you'll have these natural language tools available:

### 1. **search_indicators**
```
"Find RSI indicators for scalping EURUSD"
"Show me LuxAlgo trend following tools"
"What momentum indicators work in ranging markets?"
```

### 2. **search_strategies**
```
"Find profitable EMA strategies for GBPJPY"
"Show me high win rate MACD setups"
"What are the best 4h timeframe strategies?"
```

### 3. **find_similar_indicators**
```
"Find indicators similar to chrismoody-rsi-ema-divergence-signal"
"What alternatives exist for this indicator?"
```

### 4. **ask_trading_question**
```
"What are the best RSI settings for scalping?"
"How do I combine MACD with EMA for trend following?"
"Which indicators work best in ranging markets?"
"What's the difference between LuxAlgo and Zeiierman tools?"
```

### 5. **get_rag_stats**
```
"Show me RAG system status"
"How many indicators are available?"
"What's the health of the trading database?"
```

## 🌟 **Example Claude Code Conversations**

### Conversation 1: Finding Scalping Tools
```
You: "I need RSI indicators optimized for scalping EURUSD"
Claude: [Uses search_indicators tool automatically]
Result: Shows 5 relevant RSI indicators with similarity scores, metadata, and descriptions
```

### Conversation 2: Strategy Development
```
You: "Help me build a trend following strategy using EMA and MACD"
Claude: [Uses search_indicators + ask_trading_question tools]
Result: Finds relevant indicators + provides AI-generated strategy advice
```

### Conversation 3: Market Analysis
```
You: "What indicators perform best in high volatility markets?"
Claude: [Uses ask_trading_question tool]
Result: AI-powered analysis with specific indicator recommendations
```

## ⚡ **Quick Start Commands**

### Test the MCP server:
```bash
# Test the server directly (once MCP package is installed)
cd /home/hr/Projects/TradeSystemV1
python3 test_rag_mcp.py
```

### Start the RAG API (if not running):
```bash
docker start vector-db
```

### Check RAG API health:
```bash
curl http://localhost:8090/health
```

## 🎛️ **Features Available**

### Intelligence Features:
- **Semantic Search**: Understands trading concepts and terminology
- **Performance Awareness**: Integrates optimization data for recommendations
- **Market Context**: Considers market conditions and trading styles
- **AI Answers**: Generates intelligent responses to trading questions

### Data Sources:
- **53 Trading Indicators**: LuxAlgo, Zeiierman, LazyBear, ChrisMoody collections
- **24 Strategy Templates**: EMA, MACD, SMC optimization results
- **Real-time Performance Data**: Win rates, profit factors, drawdown metrics

### Search Capabilities:
- **Natural Language**: "Find profitable RSI for scalping"
- **Technical Terms**: RSI, MACD, EMA, Bollinger Bands, etc.
- **Market Conditions**: trending, ranging, volatile, low volatility
- **Trading Styles**: scalping, swing trading, position trading

## 🔧 **Troubleshooting**

### If MCP server fails to start:
1. Ensure the RAG API is running: `docker ps | grep vector-db`
2. Check RAG API health: `curl http://localhost:8090/health`
3. Verify Python path: `python3 --version`
4. Install missing dependencies: `pip3 install mcp httpx`

### If searches return no results:
1. Check RAG API stats: `curl http://localhost:8090/stats`
2. Verify data is loaded: Should show 53+ indicators
3. Try simpler queries: "RSI", "MACD", "EMA"

### If Claude Code doesn't recognize the tools:
1. Restart Claude Code after adding MCP configuration
2. Check MCP server logs for errors
3. Verify the server.py path is correct

## 📈 **Next Steps**

1. **Add to Claude Code**: Configure the MCP server in your Claude Code settings
2. **Test Integration**: Try the example conversations above
3. **Explore Collections**: Ask about LuxAlgo, Zeiierman, LazyBear tools
4. **Build Strategies**: Use the AI to help design custom trading systems
5. **Performance Analysis**: Leverage optimization data for strategy selection

## 💡 **Pro Tips**

- **Be Specific**: Include trading style, timeframe, and market pair for better results
- **Ask Questions**: The AI understands natural language trading questions
- **Explore Collections**: Each collection (LuxAlgo, Zeiierman, etc.) has unique strengths
- **Combine Tools**: Use multiple tools together for comprehensive analysis
- **Check Performance**: Look for indicators with high similarity scores and good performance data

---

**🎉 Your enhanced RAG system is now ready for intelligent trading conversations in Claude Code!**