# 🚀 **Aurite AI - Advanced Financial Portfolio Analysis System**

> **FastMCP-powered multi-agent system for comprehensive stock, bond, and gold analysis with ML-enhanced macro integration**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-1.0+-green.svg)](https://github.com/modelcontextprotocol/python)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 **System Overview**

Aurite AI is a sophisticated financial analysis platform that combines **Machine Learning macro analysis** with **multi-asset portfolio recommendations**. The system uses a **FastMCP server architecture** to orchestrate specialized analysis agents for stocks (NASDAQ-100), bonds, and gold investments.

### **🏗️ Architecture Highlights**
- **FastMCP Server**: Modern decorator-based MCP implementation
- **4 Specialized Agents**: Macro, Stock, Bond, and Gold analysis
- **ML-Enhanced Macro**: 135+ time series features with logistic regression
- **Real-time Data**: Yahoo Finance + FRED API integration
- **Structured Output**: JSON files with confidence scores and rankings

---

## 📊 **Core Features**

### **🧠 Enhanced Macro Analysis**
- **ML Model**: Logistic regression with 135+ time series features
- **Real-time Data**: FRED API integration for economic indicators
- **Structured Output**: JSON signals for asset class allocation
- **Confidence Scoring**: Probability-based predictions
- **GPT-4 Integration**: Advanced LLM analysis with institutional-grade prompts

### **📈 NASDAQ-100 Stock Analysis**
- **Comprehensive Coverage**: All 100 NASDAQ stocks analyzed
- **Technical Indicators**: RSI, MACD, Bollinger Bands, momentum
- **Fundamental Metrics**: P/E, P/B, debt ratios, growth rates
- **Macro Integration**: Enhanced predictions with economic context
- **GPT-4 Analysis**: Advanced financial modeling and risk assessment

### **🏛️ Bond Analysis**
- **Duration Strategy**: Short, intermediate, and long-term bonds
- **Credit Quality**: Treasury, corporate, high-yield, municipal
- **Interest Rate Sensitivity**: Macro-adjusted duration analysis
- **Yield Curve Positioning**: Slope and inversion analysis
- **GPT-4 Integration**: Advanced duration modeling and credit analysis

### **🥇 Gold Analysis**
- **Asset Types**: ETFs, futures, miners, physical gold
- **Inflation Hedge**: Real rates and dollar strength analysis
- **Volatility Protection**: VIX and geopolitical risk integration
- **Safe Haven**: Flight-to-quality during market stress
- **GPT-4 Analysis**: Advanced supply-demand and geopolitical risk modeling

### **🎯 Portfolio Integration**
- **Top 5 Picks**: Per asset class (15 total recommendations)
- **Q3 2024 Focus**: Next quarter investment horizon
- **Risk-Adjusted Returns**: Confidence scores and macro context
- **Unified Rankings**: Cross-asset class comparison

---

## 🏗️ **Project Structure**

```
Aurite-AI-Project-/
├── 📂 ai_agent/                    # Core AI components
│   ├── __init__.py                 # Package initialization
│   ├── agent.py                    # Macro analysis agent
│   ├── api_client.py               # FRED + Yahoo Finance API
│   ├── config.py                   # Configuration management
│   ├── feature_engineer.py         # 135+ time series features
│   ├── model_manager.py            # ML model management
│   └── openai_client.py            # LLM integration
├── 📂 MCP Server/                  # FastMCP server
│   └── agent2_analysis_mcp_server.py # Main server (549 lines)
├── 📂 analysis_outputs/            # Generated JSON files
│   ├── stock_analysis_YYYYMMDD_HHMMSS.json
│   ├── bond_analysis_YYYYMMDD_HHMMSS.json
│   └── gold_analysis_YYYYMMDD_HHMMSS.json
├── 📂 data/                        # Sample data
├── 📂 models/                      # Trained ML models
├── 📊 stock_analysis_agent.py      # NASDAQ-100 analysis (1834 lines)
├── 🏛️ etf_analysis_agent.py        # Bond analysis (1505 lines)
├── 🥇 gold_analysis_agent.py       # Gold analysis (883 lines)
├── 🧠 enhanced_macro_analysis.py   # ML macro analysis (589 lines)
├── 🚀 train_unified_model.py       # Model training (334 lines)
├── 📋 requirements.txt             # Dependencies
├── 📖 README.md                    # This file
├── 🛠️ SETUP_GUIDE.md               # Installation guide
└── ⚙️ .gitignore                   # Git ignore rules
```

---

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/Aurite-AI-Project-.git
cd Aurite-AI-Project-

# Install dependencies
pip install -r requirements.txt

# Set API keys (optional)
export OPENAI_API_KEY="your_openai_key"
export FRED_API_KEY="your_fred_key"
```

### **2. Run the MCP Server**
```bash
# Start the FastMCP server
python3 "MCP Server/agent2_analysis_mcp_server.py"
```

### **3. Expected Output**
```
🚀 Starting Agent 2 Portfolio Analysis MCP Server...
✅ MacroAnalysisAgent initialized
✅ BondAnalysisAgent initialized  
✅ GoldAnalysisAgent initialized
✅ StockAnalysisAgent initialized with 100 NASDAQ-100 stocks
✅ Agent 2 MCP Server initialized successfully
```

---

## 📊 **Usage Examples**

### **🎯 Complete Portfolio Analysis**
```python
# The MCP server automatically:
# 1. Analyzes 100 NASDAQ stocks
# 2. Evaluates 8 bond types
# 3. Assesses 5 gold assets
# 4. Integrates macro context
# 5. Generates top 5 picks per class
# 6. Outputs 4 JSON files
```

### **📁 Generated Output Files**
```
analysis_outputs/
├── stock_analysis_20250806_215052.json    # Top 5 NASDAQ picks
├── bond_analysis_20250806_215052.json     # Top 5 bond picks
├── gold_analysis_20250806_215052.json     # Top 5 gold picks
└── macro_analysis_20250806_215052.json    # Macro context
```

### **📊 Sample Output Structure**
```json
{
  "analysis_timestamp": "2025-08-06T21:50:52",
  "total_assets_analyzed": 113,
  "top_picks_per_class": 5,
  "investment_horizon": "next_quarter",
  "portfolio_summary": {
    "stocks": {"count": 5, "avg_expected_return": 0.082, "risk_level": "moderate"},
    "bonds": {"count": 5, "avg_expected_return": 0.045, "risk_level": "low"},
    "gold": {"count": 5, "avg_expected_return": 0.058, "risk_level": "moderate"}
  }
}
```

---

## 🔧 **Configuration**

### **📋 Dependencies**
```txt
# Core ML & Data
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0

# MCP & API
mcp[server]>=1.0.0
openai>=1.0.0
langchain>=0.1.0
langchain-openai>=0.1.0

# Utilities
loguru>=0.7.0
pydantic>=2.0.0
typing-extensions>=4.0.0
```

### **🔑 API Keys (Optional)**
```bash
# For enhanced functionality
export OPENAI_API_KEY="sk-..."      # LLM commentary
export FRED_API_KEY="..."           # Real economic data
```

---

## 🎯 **Analysis Outputs**

### **📈 Stock Analysis** (Top 5 from NASDAQ-100 for Q3 with Macro Context)
- **Technology Leaders**: NVDA, AMD, AAPL, MSFT (from 100 NASDAQ stocks analyzed)
- **Growth Sectors**: AI, Cloud, Consumer Tech (economic cycle: expansion)
- **Expected Returns**: 5.2% - 8.5% (macro-adjusted for Q3 conditions)
- **Risk-Adjusted Returns**: Confidence scores and macro-adjusted expectations

### **🏛️ Bond Analysis** (Top 5 for Q3 with Macro Context)
- **Duration Strategy**: Short, intermediate, long-term allocation
- **Credit Quality**: Treasury, corporate, municipal bonds
- **Interest Rate Sensitivity**: Macro-adjusted duration analysis
- **Expected Returns**: 3.5% - 4.2% (rate environment dependent)

### **🥇 Gold Analysis** (Top 5 for Q3 with Macro Context)
- **Asset Types**: ETFs, miners, physical gold exposure
- **Inflation Hedge**: Real rates and dollar strength analysis
- **Safe Haven**: Volatility and geopolitical risk protection
- **Expected Returns**: 5.8% - 8.5% (inflation and risk dependent)

---

## 🛠️ **Development**

### **🧠 Model Training**
```bash
# Train the unified macro model
python3 train_unified_model.py
```

### **📊 Feature Engineering**
The system includes 135+ time series features:
- **Lag Features**: Historical price and volume patterns
- **Autoregressive Features**: Self-predicting patterns
- **Trend Features**: Moving averages and momentum
- **Cyclical Features**: Seasonal and business cycle patterns
- **Cross-lag Features**: Multi-asset correlations
- **Stationarity Features**: Statistical stability measures

### **🔧 Customization**
```python
# Modify analysis parameters
ANALYSIS_CONFIG = {
    'horizon': 'next_quarter',  # Investment horizon
    'top_picks': 5,            # Picks per asset class
    'total_target': 15         # Total portfolio size
}
```

---

## 📈 **Performance Metrics**

### **🎯 Model Accuracy**
- **Macro Prediction**: ML model with 135+ features
- **Stock Analysis**: 100 NASDAQ stocks with technical + fundamental
- **Bond Analysis**: 8 bond types with duration + credit analysis
- **Gold Analysis**: 5 gold assets with inflation + volatility analysis

### **⚡ Processing Speed**
- **Parallel Analysis**: All asset classes analyzed simultaneously
- **Intelligent Sampling**: Large datasets processed efficiently
- **Caching**: API responses cached for performance
- **Real-time Data**: Live market data integration

---

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes
4. **Push** to the branch
5. **Create** a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 **Support**

- **Documentation**: See [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions

---

## 🎉 **Acknowledgments**

- **FastMCP**: Modern MCP server implementation
- **Yahoo Finance**: Real-time market data
- **FRED**: Economic indicators
- **Scikit-learn**: Machine learning framework
- **OpenAI**: LLM integration

---

**🚀 Ready to analyze your portfolio with AI-powered insights!** 