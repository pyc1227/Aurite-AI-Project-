# 🚀 **Setup Guide for Aurite AI Project**

## **Quick Installation**

### **1. Install Dependencies**
```bash
# Install required packages
pip install mcp[server] yfinance openai pandas loguru scikit-learn langchain langchain-openai

# Or install from requirements.txt
pip install -r requirements.txt
```

### **2. Set API Keys (Optional)**
```bash
# Set OpenAI API key for LLM features (optional)
export OPENAI_API_KEY="your_openai_api_key_here"

# Set FRED API key for real economic data (optional)
export FRED_API_KEY="your_fred_api_key_here"
```

### **3. Run the MCP Server**
```bash
# Start the FastMCP server
python "MCP Server/agent2_analysis_mcp_server.py"
```

### **4. Test the Installation**
```bash
# Test the server functionality
python test_agent2_integration.py
```

---

## **🔧 Troubleshooting Common Issues**

### **Issue: "MCPServer is not present. Please use FastMCP"**

**✅ Solution:** This project uses the **FastMCP** framework, not the old `MCPServer` class.

**What happened:**
- Older MCP versions used `MCPServer` class
- This project has been updated to use `FastMCP` (simpler, decorator-based)
- Your MCP installation might be outdated

**Fix:**
```bash
# Update to latest MCP version
pip install --upgrade mcp[server]

# Verify installation
python -c "from mcp.server import FastMCP; print('✅ FastMCP available')"
```

### **Issue: Import Errors**

**Missing packages:**
```bash
# Install missing packages individually
pip install mcp[server]      # MCP framework
pip install yfinance         # Stock data
pip install openai           # LLM integration  
pip install pandas           # Data processing
pip install loguru           # Logging
pip install scikit-learn     # ML models
pip install langchain        # LLM chains
pip install langchain-openai # OpenAI integration
```

### **Issue: API Key Warnings**

**Expected behavior:** The system works without API keys using sample data.

**To enable full functionality:**
```bash
# Get free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY="your_fred_api_key"

# Get OpenAI API key: https://platform.openai.com/api-keys  
export OPENAI_API_KEY="your_openai_api_key"
```

---

## **📊 How to Use**

### **Run Analysis**
```bash
# Start the MCP server
cd "/path/to/Aurite-AI-Project-"
python "MCP Server/agent2_analysis_mcp_server.py"
```

### **Expected Output**
```
🚀 Starting Agent 2 Portfolio Analysis MCP Server...
✅ MacroAnalysisAgent initialized
✅ BondAnalysisAgent initialized  
✅ GoldAnalysisAgent initialized
✅ StockAnalysisAgent initialized with 100 NASDAQ-100 stocks
✅ Agent 2 MCP Server initialized successfully
🚀 Starting Agent 2 Portfolio Analysis MCP Server...
```

### **Test Individual Components**
```bash
# Test stock analysis
python -c "
import asyncio
import sys
sys.path.append('MCP Server')
from agent2_analysis_mcp_server import get_analysis

async def test():
    result = await get_analysis('stock', [], 'Get top 5 stocks for Q3')
    print(f'Stock analysis: {result.get(\"status\", \"failed\")}')

asyncio.run(test())
"
```

---

## **🎯 System Architecture**

**This project uses:**
- ✅ **FastMCP** (not MCPServer)
- ✅ **4 Analysis Agents** (Macro, Stock, Bond, Gold)
- ✅ **API-based data** (not database)
- ✅ **Top 5 picks** per asset class
- ✅ **Q3 2024 horizon**
- ✅ **JSON output files**

**MCP Tools Available:**
- `analyze_market_with_prompt` - Complete portfolio analysis
- `get_analysis` - Individual asset class analysis

---

## **💡 Quick Verification**

```bash
# Verify FastMCP installation
python -c "from mcp.server import FastMCP; print('✅ FastMCP ready')"

# Test basic functionality  
python test_agent2_integration.py

# View available analysis agents
python -c "
import sys
sys.path.append('MCP Server')
from agent2_analysis_mcp_server import ANALYSIS_AGENTS
print('Available agents:', list(ANALYSIS_AGENTS.keys()))
"
```

**Expected agents:** `['bond', 'gold', 'stock']`

---

## **📞 Support**

If you still encounter issues:

1. **Check Python version:** Requires Python 3.8+
2. **Update pip:** `pip install --upgrade pip`
3. **Clean install:** `pip uninstall mcp && pip install mcp[server]`
4. **Test step by step:** Run each command individually

**Common working setup:**
```bash
pip install mcp[server]==1.0.0 yfinance openai pandas loguru
python "MCP Server/agent2_analysis_mcp_server.py"
``` 