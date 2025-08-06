# agent2_analysis_mcp_server.py
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache
import warnings

# Add parent directory to path to import ai_agent modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from mcp.server import MCPServer
    from mcp.types import Tool, Resource
    import yfinance as yf
    import openai
    import pandas as pd
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MCP dependencies not available: {e}")
    MCP_AVAILABLE = False

# Import enhanced macro analysis
try:
    from enhanced_macro_analysis import EnhancedMacroAnalyzer
    ENHANCED_MACRO_AVAILABLE = True
except ImportError:
    print("⚠️  Enhanced macro analysis not available")
    ENHANCED_MACRO_AVAILABLE = False

# Import macro analysis components
from ai_agent.agent import MacroAnalysisAgent
from ai_agent.api_client import MacroAPIClient, APIConfig
from ai_agent.feature_engineer import FeatureEngineer
from ai_agent.model_manager import ModelManager

# Import analysis agents with unified error handling
ANALYSIS_AGENTS = {}

try:
    from etf_analysis_agent import BondAnalysisAgent, AgentConfig, LLMConfig, BondPrediction, MarketConditions
    ANALYSIS_AGENTS['bond'] = {
        'class': BondAnalysisAgent,
        'config_class': AgentConfig,
        'llm_config_class': LLMConfig,
        'name': 'Bond Analysis Agent',
        'system_prompt': "You are an expert bond and fixed income analyst specializing in comprehensive bond analysis.",
        'symbols': ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "TIPS"]
    }
except ImportError:
    print("⚠️  Bond analysis agent not available")

try:
    from gold_analysis_agent import GoldAnalysisAgent, AgentConfig, LLMConfig, GoldPrediction, GoldMarketConditions
    ANALYSIS_AGENTS['gold'] = {
        'class': GoldAnalysisAgent,
        'config_class': AgentConfig,
        'llm_config_class': LLMConfig,
        'name': 'Gold Analysis Agent',
        'system_prompt': "You are an expert gold and precious metals analyst specializing in comprehensive gold analysis.",
        'symbols': ["GLD", "IAU", "GC=F", "GDX", "SGOL"]
    }
except ImportError:
    print("⚠️  Gold analysis agent not available")

try:
    from stock_analysis_agent import StockAnalysisAgent, AgentConfig as StockAgentConfig, LLMConfig as StockLLMConfig, StockPrediction, StockMarketConditions
    ANALYSIS_AGENTS['stock'] = {
        'class': StockAnalysisAgent,
        'config_class': StockAgentConfig,
        'llm_config_class': StockLLMConfig,
        'name': 'Stock Analysis Agent',
        'system_prompt': "You are an expert stock market analyst specializing in comprehensive stock analysis.",
        'symbols': ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    }
except ImportError:
    print("⚠️  Stock analysis agent not available")

warnings.filterwarnings('ignore')

class Agent2MCPServer:
    def __init__(self):
        if not MCP_AVAILABLE:
            raise ImportError("MCP dependencies not available")
            
        self.server = MCPServer("agent2-data-analysis")
        self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Optimized data structures
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        self.output_dir = Path("./analysis_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Unified agent management
        self.agents = {}
        self.macro_agent = None
        self.api_client = None
        self.feature_engineer = None
        self.model_manager = None
        
        self.setup_mcp_tools()
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize all analysis agents with unified approach"""
        print("[Agent 2] Initializing analysis agents...")
        
        # Initialize macro analysis components
        try:
            api_config = APIConfig()
            self.api_client = MacroAPIClient(api_config)
            self.feature_engineer = FeatureEngineer()
            self.model_manager = ModelManager()
            
            self.macro_agent = MacroAnalysisAgent()
            self.macro_agent.initialize()
            print("✅ Macro analysis agent initialized")
            
        except Exception as e:
            print(f"⚠️  Macro agent initialization failed: {e}")
        
        # Initialize analysis agents using unified approach
        for agent_type, agent_info in ANALYSIS_AGENTS.items():
            try:
                config = agent_info['config_class'](
                    name=agent_info['name'],
                    system_prompt=agent_info['system_prompt'],
                    llm_config=agent_info['llm_config_class'](
                        model="gpt-3.5-turbo",
                        temperature=0.6,
                        max_tokens=1000,
                        api_key=os.getenv("OPENAI_API_KEY")
                    ),
                    analysis_depth="comprehensive",
                    enable_llm_commentary=True
                )
                
                self.agents[agent_type] = agent_info['class'](config)
                print(f"✅ {agent_info['name']} initialized")
                
            except Exception as e:
                print(f"⚠️  {agent_info['name']} initialization failed: {e}")
    
    def setup_mcp_tools(self):
        """Setup MCP tools with optimized structure"""
        
        @self.server.tool("analyze_market_with_prompt")
        async def analyze_market_with_prompt(user_prompt: str, analysis_depth: str = "comprehensive") -> dict:
            """Main tool for complete market analysis"""
            try:
                print(f"[Agent 2] Received user prompt: {user_prompt}")
                
                # Step 1: Macro Analysis
                macro_prediction = await self.perform_macro_analysis(user_prompt)
                
                # Step 2: Market Data Collection (with caching)
                market_data = await self.get_cached_market_data()
                
                # Step 3: LLM Macro Analysis
                llm_macro_analysis = await self.perform_llm_macro_analysis(
                    market_data["macro_indicators"], 
                    user_prompt,
                    analysis_depth
                )
                
                # Step 4: Parallel Analysis Execution
                analysis_tasks = []
                for agent_type in self.agents.keys():
                    task = self.perform_analysis_with_macro_context(
                        agent_type, user_prompt, market_data, llm_macro_analysis
                    )
                    analysis_tasks.append(task)
                
                # Execute all analyses in parallel
                analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                # Step 5: Save Results
                saved_files = await self.save_all_analyses(analysis_results, llm_macro_analysis)
                
                # Step 6: Return Complete Results
                return self.build_complete_results(
                    user_prompt, macro_prediction, llm_macro_analysis, 
                    analysis_results, saved_files, market_data
                )
                
            except Exception as e:
                return self.build_error_response(str(e))
        
        # Unified tool for getting analysis results
        @self.server.tool("get_analysis")
        async def get_analysis(analysis_type: str, symbols: List[str] = None, 
                             use_macro_context: bool = False) -> dict:
            """Unified tool for getting any type of analysis"""
            try:
                if analysis_type not in self.agents:
                    return {"error": f"{analysis_type} agent not available"}
                
                if not symbols:
                    symbols = ANALYSIS_AGENTS[analysis_type]['symbols']
                
                # Get macro context if requested
                macro_analysis = None
                if use_macro_context:
                    macro_analysis = await self.get_macro_context()
                
                # Perform analysis
                if macro_analysis:
                    analysis = await self.perform_analysis_with_macro_context(
                        analysis_type, "", {}, macro_analysis
                    )
                else:
                    analysis = await self.agents[analysis_type].analyze_bonds(symbols)
                
                return {
                    "status": "success",
                    "analysis": analysis,
                    "macro_context_used": macro_analysis is not None,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {"error": str(e)}
        
        @self.server.tool("get_macro_prediction")
        async def get_macro_prediction() -> dict:
            """Get latest macro prediction"""
            try:
                if not self.macro_agent:
                    return {"error": "Macro agent not initialized"}
                
                prediction = self.macro_agent.predict_next_quarter()
                return {
                    "status": "success",
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.server.tool("health_check")
        async def health_check() -> dict:
            """Check health of all components"""
            return await self.perform_health_check()
        
        @self.server.resource("analysis_outputs")
        async def analysis_outputs_resource() -> str:
            """Access to all analysis output files"""
            return await self.get_analysis_outputs()
    
    async def get_cached_market_data(self) -> Dict[str, Any]:
        """Get market data with caching"""
        cache_key = "market_data"
        current_time = datetime.now()
        
        if cache_key in self.data_cache:
            cached_data, cache_time = self.data_cache[cache_key]
            if (current_time - cache_time).seconds < self.cache_ttl:
                return cached_data
        
        # Collect fresh data
        market_data = await self.collect_comprehensive_market_data()
        self.data_cache[cache_key] = (market_data, current_time)
        return market_data
    
    async def perform_analysis_with_macro_context(self, agent_type: str, user_prompt: str, 
                                                market_data: Dict, macro_analysis: Dict) -> Dict[str, Any]:
        """Unified method for performing analysis with macro context"""
        try:
            agent = self.agents[agent_type]
            symbols = self.extract_symbols_from_prompt(user_prompt, agent_type)
            
            if not symbols:
                symbols = ANALYSIS_AGENTS[agent_type]['symbols']
            
            # Perform analysis with macro context
            if hasattr(agent, 'analyze_bonds'):
                analysis = await agent.analyze_bonds(symbols, macro_context=macro_analysis)
            elif hasattr(agent, 'analyze_gold'):
                analysis = await agent.analyze_gold(symbols, macro_context=macro_analysis)
            elif hasattr(agent, 'analyze_stocks'):
                analysis = await agent.analyze_stocks(symbols, macro_context=macro_analysis)
            else:
                analysis = await agent.analyze_bonds(symbols)
            
            # Get top picks
            top_picks = self.get_top_picks(agent, analysis, agent_type)
            
            return {
                "analysis": analysis,
                "top_picks": top_picks,
                "analyzed_symbols": symbols,
                "macro_context_used": macro_analysis is not None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in {agent_type} analysis: {e}")
            return {
                "error": str(e),
                "fallback": {"analysis": {}, "timestamp": datetime.now().isoformat()}
            }
    
    def get_top_picks(self, agent, analysis: Dict, agent_type: str) -> List[Dict]:
        """Get top picks based on agent type"""
        try:
            if agent_type == 'bond':
                return agent.get_top_bond_picks(analysis, horizon='next_quarter', top_n=10)
            elif agent_type == 'gold':
                return agent.get_top_gold_picks(analysis, horizon='next_quarter', top_n=10)
            elif agent_type == 'stock':
                return agent.get_top_stock_picks(analysis, horizon='next_quarter', top_n=10)
            else:
                return []
        except Exception as e:
            print(f"Error getting top picks for {agent_type}: {e}")
            return []
    
    def extract_symbols_from_prompt(self, user_prompt: str, agent_type: str) -> List[str]:
        """Extract symbols from prompt based on agent type"""
        if not user_prompt:
            return []
        
        # Common symbols for each agent type
        symbol_maps = {
            'bond': ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "TIPS", "VCIT", "VCSH"],
            'gold': ["GLD", "IAU", "SGOL", "GLDM", "BAR", "GC=F", "XAUUSD=X", "GDX", "GDXJ"],
            'stock': ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "AMD", "NFLX"]
        }
        
        symbols = symbol_maps.get(agent_type, [])
        found_symbols = []
        prompt_upper = user_prompt.upper()
        
        for symbol in symbols:
            if symbol in prompt_upper:
                found_symbols.append(symbol)
        
        return found_symbols
    
    async def load_enhanced_macro_context(self) -> Optional[Dict]:
        """Load enhanced macro analysis JSON file as context."""
        try:
            # Look for the most recent enhanced macro analysis file
            macro_files = list(Path('.').glob('macro_signals_*.json'))
            
            if not macro_files:
                print("No enhanced macro analysis files found")
                return None
            
            # Get the most recent file
            latest_file = max(macro_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                macro_context = json.load(f)
            
            print(f"✅ Loaded enhanced macro context from: {latest_file}")
            return macro_context
            
        except Exception as e:
            print(f"⚠️ Could not load enhanced macro context: {e}")
            return None
    
    async def generate_enhanced_macro_signals(self) -> Optional[Dict]:
        """Generate fresh enhanced macro analysis signals."""
        if not ENHANCED_MACRO_AVAILABLE:
            return None
            
        try:
            print("🔍 Generating enhanced macro analysis...")
            analyzer = EnhancedMacroAnalyzer()
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"macro_signals_{timestamp}.json"
            
            # Export macro signals
            macro_signals = analyzer.export_macro_signals_json(filepath)
            
            if macro_signals:
                print(f"✅ Enhanced macro signals generated: {filepath}")
                return macro_signals
            else:
                print("⚠️ Enhanced macro analysis failed")
                return None
                
        except Exception as e:
            print(f"❌ Enhanced macro analysis error: {e}")
            return None

    async def get_macro_context(self) -> Optional[Dict]:
        """Get macro context for analysis - prioritize enhanced analysis."""
        # First try to load existing enhanced macro analysis
        enhanced_context = await self.load_enhanced_macro_context()
        
        if enhanced_context:
            return enhanced_context
        
        # If no enhanced analysis available, generate fresh one
        fresh_context = await self.generate_enhanced_macro_signals()
        
        if fresh_context:
            return fresh_context
        
        # Fallback to basic macro prediction
        try:
            macro_pred = await self.get_macro_prediction()
            if macro_pred.get('status') == 'success':
                return macro_pred
        except Exception as e:
            print(f"Warning: Could not get macro context: {e}")
        
        return None
    
    async def save_all_analyses(self, analysis_results: List, llm_macro_analysis: Dict) -> Dict[str, str]:
        """Save all analysis results to files"""
        saved_files = {}
        
        # Save macro analysis
        macro_file = await self.save_analysis(llm_macro_analysis, "macro_analysis")
        saved_files["macro_analysis_json"] = str(macro_file)
        
        # Save other analyses
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                continue
            
            agent_type = list(self.agents.keys())[i]
            if result and 'analysis' in result:
                file_path = await self.save_analysis(result, f"{agent_type}_analysis")
                saved_files[f"{agent_type}_analysis_json"] = str(file_path)
        
        return saved_files
    
    async def save_analysis(self, analysis: Dict, analysis_type: str) -> Path:
        """Save analysis to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{analysis_type}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"[Agent 2] {analysis_type} saved to: {filepath}")
        return filepath
    
    def build_complete_results(self, user_prompt: str, macro_prediction: Dict, 
                             llm_macro_analysis: Dict, analysis_results: List, 
                             saved_files: Dict, market_data: Dict) -> Dict[str, Any]:
        """Build complete results dictionary"""
        results = {
            "status": "success",
            "user_prompt": user_prompt,
            "analysis_timestamp": datetime.now().isoformat(),
            "macro_prediction": macro_prediction,
            "llm_macro_analysis": llm_macro_analysis,
            "market_data_summary": self.summarize_market_data(market_data),
            "output_files": saved_files,
            "data_quality": self.assess_data_quality(market_data),
            "next_agent_ready": True
        }
        
        # Add analysis results
        for i, result in enumerate(analysis_results):
            if not isinstance(result, Exception):
                agent_type = list(self.agents.keys())[i]
                results[f"{agent_type}_analysis"] = result
        
        return results
    
    def build_error_response(self, error: str) -> Dict[str, Any]:
        """Build standardized error response"""
        return {
            "status": "error",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "macro_agent": False,
            "api_client": False,
            "model_manager": False,
            "openai_client": False,
            "analysis_agents": {}
        }
        
        # Check macro agent
        if self.macro_agent:
            try:
                health = self.macro_agent.health_check()
                health_status["macro_agent"] = health.get("status") == "healthy"
            except:
                pass
        
        # Check API client
        if self.api_client:
            try:
                health = self.api_client.health_check()
                health_status["api_client"] = any(health.values())
            except:
                pass
        
        # Check model manager
        if self.model_manager:
            health_status["model_manager"] = self.model_manager.is_model_loaded()
        
        # Check OpenAI client
        try:
            await self.openai_client.models.list()
            health_status["openai_client"] = True
        except:
            pass
        
        # Check analysis agents
        for agent_type, agent in self.agents.items():
            health_status["analysis_agents"][agent_type] = agent is not None
        
        return health_status
    
    async def get_analysis_outputs(self) -> str:
        """Get analysis output files information"""
        files = list(self.output_dir.glob("*.json"))
        file_info = []
        
        for file in files:
            stat = file.stat()
            file_info.append({
                "filename": file.name,
                "path": str(file),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return json.dumps({"analysis_files": file_info})
    
    # Keep existing methods for backward compatibility
    async def perform_macro_analysis(self, user_prompt: str) -> Dict[str, Any]:
        """Perform macro analysis using the macro analysis agent"""
        print("[Agent 2] Performing macro analysis...")
        
        try:
            if not self.macro_agent:
                return {"error": "Macro agent not initialized"}
            
            prediction = self.macro_agent.predict_next_quarter()
            market_analysis = self.macro_agent.get_market_analysis()
            prediction_summary = self.macro_agent.get_prediction_summary()
            
            return {
                "quarterly_prediction": prediction,
                "market_analysis": market_analysis,
                "prediction_summary": prediction_summary,
                "agent_status": self.macro_agent.get_agent_status(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in macro analysis: {e}")
            return {
                "error": str(e),
                "fallback": {
                    "quarterly_prediction": {"prediction": "neutral", "confidence": 50},
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    async def collect_comprehensive_market_data(self) -> Dict[str, Any]:
        """Collect comprehensive market data for analysis"""
        print("[Agent 2] Collecting market data...")
        
        # Define assets to analyze
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        bonds = ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "TIPS"]
        rare_metals = ["GLD", "SLV", "PALL", "PPLT"]
        market_etfs = ["SPY", "QQQ", "IWM", "VTI"]
        
        market_data = {
            "collection_timestamp": datetime.now().isoformat(),
            "stocks": {},
            "bonds": {},
            "rare_metals": {},
            "market_etfs": {},
            "macro_indicators": {}
        }
        
        # Collect data in parallel
        tasks = []
        for symbol in stocks:
            tasks.append(self.collect_asset_data(symbol, "stocks"))
        for symbol in bonds:
            tasks.append(self.collect_asset_data(symbol, "bonds"))
        for symbol in rare_metals:
            tasks.append(self.collect_asset_data(symbol, "rare_metals"))
        
        # Execute all data collection tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, dict) and 'symbol' in result:
                category = result['category']
                symbol = result['symbol']
                market_data[category][symbol] = result['data']
        
        # Collect macro indicators
        try:
            market_data["macro_indicators"] = await self.collect_macro_indicators()
        except Exception as e:
            print(f"Error collecting macro indicators: {e}")
        
        return market_data
    
    async def collect_asset_data(self, symbol: str, category: str) -> Dict[str, Any]:
        """Collect data for a single asset"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            info = ticker.info
            
            data = {
                "current_price": float(hist['Close'].iloc[-1]),
                "price_change_1m": float(((hist['Close'].iloc[-1] - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]) * 100),
                "price_change_3m": float(((hist['Close'].iloc[-1] - hist['Close'].iloc[-66]) / hist['Close'].iloc[-66]) * 100),
                "volatility": float(hist['Close'].pct_change().std() * 100)
            }
            
            # Add category-specific data
            if category == "stocks":
                data.update({
                    "volume_avg": float(hist['Volume'].tail(20).mean()),
                    "market_cap": info.get('marketCap', 0),
                    "pe_ratio": info.get('trailingPE', None),
                    "sector": info.get('sector', 'Unknown')
                })
            
            return {
                "symbol": symbol,
                "category": category,
                "data": data
            }
            
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
            return {"symbol": symbol, "category": category, "data": {}}
    
    async def collect_macro_indicators(self) -> Dict[str, Any]:
        """Collect macro indicators"""
        try:
            # Collect macro data in parallel
            vix_task = self.collect_ticker_data("^VIX", "vix")
            dxy_task = self.collect_ticker_data("DX-Y.NYB", "dollar_index")
            tnx_task = self.collect_ticker_data("^TNX", "treasury_10y")
            
            results = await asyncio.gather(vix_task, dxy_task, tnx_task, return_exceptions=True)
            
            macro_indicators = {}
            
            if not isinstance(results[0], Exception):
                macro_indicators["vix"] = {
                    "current": results[0]["current"],
                    "avg_1m": results[0]["avg_1m"],
                    "interpretation": "fear_gauge"
                }
            
            if not isinstance(results[1], Exception):
                macro_indicators["dollar_index"] = {
                    "current": results[1]["current"],
                    "change_3m": results[1]["change_3m"],
                    "interpretation": "dollar_strength"
                }
            
            if not isinstance(results[2], Exception):
                macro_indicators["treasury_10y"] = {
                    "current": results[2]["current"],
                    "change_3m": results[2]["change_3m"],
                    "interpretation": "interest_rate_proxy"
                }
            
            return macro_indicators
            
        except Exception as e:
            print(f"Error collecting macro indicators: {e}")
            return {}
    
    async def collect_ticker_data(self, symbol: str, data_type: str) -> Dict[str, Any]:
        """Collect data for a specific ticker"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if data_type == "vix":
                return {
                    "current": float(hist['Close'].iloc[-1]),
                    "avg_1m": float(hist['Close'].mean())
                }
            else:
                return {
                    "current": float(hist['Close'].iloc[-1]),
                    "change_3m": float(hist['Close'].iloc[-1] - hist['Close'].iloc[0])
                }
                
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
            return {}
    
    async def perform_llm_macro_analysis(self, macro_data: Dict, user_prompt: str, analysis_depth: str) -> Dict[str, Any]:
        """Use OpenAI to perform macro economic analysis"""
        print("[Agent 2] Performing LLM macro analysis...")
        
        system_prompt = """You are a senior macroeconomic analyst with 20+ years of experience. 
        Analyze the provided market data and provide a comprehensive macroeconomic assessment.
        
        Your analysis should include:
        1. Overall market sentiment (bullish/bearish)
        2. Probability assessment (0-100%)
        3. Key risk factors
        4. Economic cycle assessment
        5. Specific recommendations for different asset classes
        
        Respond ONLY with valid JSON format."""
        
        user_analysis_prompt = f"""
        Based on the following macro indicators and user context, provide a detailed analysis:
        
        USER CONTEXT: {user_prompt}
        
        MACRO DATA:
        {json.dumps(macro_data, indent=2)}
        
        Please provide analysis in this exact JSON format:
        {{
            "macro_sentiment": {{
                "overall_direction": "bullish" or "bearish",
                "confidence_probability": number between 0-100,
                "strength": "weak", "moderate", or "strong"
            }},
            "economic_environment": {{
                "cycle_phase": "expansion", "peak", "contraction", or "trough",
                "inflation_outlook": "deflationary", "stable", or "inflationary",
                "interest_rate_trend": "rising", "stable", or "falling"
            }},
            "asset_class_outlook": {{
                "stocks": {{
                    "recommendation": "overweight", "neutral", or "underweight",
                    "probability": number 0-100,
                    "reasoning": "brief explanation"
                }},
                "bonds": {{
                    "recommendation": "overweight", "neutral", or "underweight", 
                    "probability": number 0-100,
                    "reasoning": "brief explanation"
                }},
                "rare_metals": {{
                    "recommendation": "overweight", "neutral", or "underweight",
                    "probability": number 0-100,
                    "reasoning": "brief explanation"
                }}
            }},
            "key_risks": [
                "list of 3-5 key risk factors"
            ],
            "time_horizon": "short-term (1-3 months), medium-term (3-12 months), long-term (1+ years)",
            "analysis_confidence": number 0-100,
            "last_updated": "{datetime.now().isoformat()}"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            macro_analysis = json.loads(analysis_text)
            
            macro_analysis["llm_model"] = "gpt-4o"
            macro_analysis["prompt_tokens"] = response.usage.prompt_tokens
            macro_analysis["completion_tokens"] = response.usage.completion_tokens
            
            return macro_analysis
            
        except Exception as e:
            print(f"Error in LLM macro analysis: {e}")
            return {
                "error": str(e),
                "fallback_analysis": {
                    "macro_sentiment": {"overall_direction": "neutral", "confidence_probability": 50},
                    "analysis_confidence": 25
                }
            }
    
    def summarize_market_data(self, market_data: Dict) -> Dict[str, Any]:
        """Create a summary of collected market data"""
        summary = {
            "total_assets": 0,
            "asset_categories": {},
            "macro_indicators_count": len(market_data.get("macro_indicators", {})),
            "data_freshness": market_data.get("collection_timestamp", "unknown")
        }
        
        for category in ["stocks", "bonds", "rare_metals", "market_etfs"]:
            count = len(market_data.get(category, {}))
            summary["asset_categories"][category] = count
            summary["total_assets"] += count
        
        return summary
    
    def assess_data_quality(self, market_data: Dict) -> Dict[str, Any]:
        """Assess the quality and completeness of collected data"""
        total_expected = {
            "stocks": 8,
            "bonds": 8, 
            "rare_metals": 4,
            "market_etfs": 4
        }
        
        quality_score = {}
        for category, expected_count in total_expected.items():
            actual_count = len(market_data.get(category, {}))
            quality_score[category] = {
                "expected": expected_count,
                "actual": actual_count,
                "completeness": (actual_count / expected_count) * 100
            }
        
        overall_quality = sum(q["completeness"] for q in quality_score.values()) / len(quality_score)
        
        return {
            "overall_quality_score": round(overall_quality, 2),
            "category_breakdown": quality_score,
            "macro_indicators_available": len(market_data.get("macro_indicators", {})),
            "data_freshness": "current" if overall_quality > 80 else "partial"
        }
    
    async def run(self):
        """Start the Agent 2 MCP server"""
        print("Starting Agent 2 MCP Server on port 8002...")
        print(f"✅ Macro Analysis Agent: {'Available' if self.macro_agent else 'Not Available'}")
        
        for agent_type, agent in self.agents.items():
            print(f"✅ {ANALYSIS_AGENTS[agent_type]['name']}: {'Available' if agent else 'Not Available'}")
        
        await self.server.run(port=8002)