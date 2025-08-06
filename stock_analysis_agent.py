"""
Stock Analysis Agent - Stock Market Analysis & Prediction System with Macro Integration
Analyzes stocks and stock ETFs with macro context integration
Outputs stock market analysis and investment recommendations
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import asyncio
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    try:
        import yfinance
    except ImportError:
        missing_deps.append("yfinance")
    try:
        from openai import OpenAI
    except ImportError:
        missing_deps.append("openai")
    
    if missing_deps:
        raise ImportError(f"Missing required dependencies: {', '.join(missing_deps)}")
    return True

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.6
    max_tokens: int = 1000
    api_key: Optional[str] = None

@dataclass
class AgentConfig:
    """Agent configuration with system prompts"""
    name: str
    system_prompt: str
    llm_config: LLMConfig
    analysis_depth: str = "comprehensive"
    enable_llm_commentary: bool = True

@dataclass
class StockPrediction:
    """Individual stock prediction result"""
    ticker: str
    stock_type: str
    sector: str
    market_cap: str
    label: str
    expected_return: float
    rank: int
    sentiment: str
    confidence: float
    summary: str
    pe_ratio: float
    dividend_yield: float
    beta: float

@dataclass
class StockMarketConditions:
    """Current market conditions for stock analysis"""
    vix_level: float
    market_momentum: float
    sector_rotation: str
    earnings_growth: float
    interest_rate_environment: float
    economic_growth: float
    market_volatility: float

class LLMCommentaryEngine:
    """LLM-powered stock commentary and analysis engine"""
    
    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self.client = None
        
        if llm_config.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=llm_config.api_key)
                logger.info("✅ OpenAI client initialized for stock analysis")
            except Exception as e:
                logger.warning(f"⚠️  OpenAI client initialization failed: {e}")
    
    async def generate_stock_commentary(self, symbol: str, stock_type: str, sector: str,
                                      technical_data: Dict, fundamental_data: Dict,
                                      market_conditions: StockMarketConditions, horizon: str) -> Tuple[float, float, str, str]:
        """Generate LLM-powered stock analysis commentary"""
        if not self.client:
            return self._fallback_stock_analysis(symbol, stock_type, sector, technical_data, market_conditions, horizon)
        
        try:
            system_prompt = f"""You are an expert stock market analyst. 
            Analyze the provided stock data and market conditions to provide:
            1. Expected return (as a decimal, e.g., 0.08 for 8%)
            2. Confidence level (0.0 to 1.0)
            3. Sentiment (bullish/bearish/neutral)
            4. Brief summary (1-2 sentences)
            
            Respond ONLY with valid JSON format:
            {{
                "expected_return": 0.08,
                "confidence": 0.85,
                "sentiment": "bullish",
                "summary": "Stock shows strong fundamentals with positive technical momentum."
            }}"""
            
            user_prompt = f"""
            Analyze this stock investment:
            
            Symbol: {symbol}
            Type: {stock_type}
            Sector: {sector}
            Horizon: {horizon}
            
            Technical Data: {json.dumps(technical_data, indent=2)}
            Fundamental Data: {json.dumps(fundamental_data, indent=2)}
            Market Conditions: {json.dumps(asdict(market_conditions), indent=2)}
            
            Provide analysis in the exact JSON format specified.
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return self._parse_llm_response_safe(response.choices[0].message.content, symbol)
            
        except Exception as e:
            logger.error(f"Error in LLM stock commentary: {e}")
            return self._fallback_stock_analysis(symbol, stock_type, sector, technical_data, market_conditions, horizon)
    
    def _parse_llm_response_safe(self, response: str, symbol: str) -> Tuple[float, float, str, str]:
        """Safely parse LLM response for stock analysis"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                expected_return = float(data.get('expected_return', 0.06))
                confidence = float(data.get('confidence', 0.8))
                sentiment = str(data.get('sentiment', 'neutral'))
                summary = str(data.get('summary', f'Analysis for {symbol}'))
                
                return expected_return, confidence, sentiment, summary
            else:
                return self._fallback_stock_values(symbol)
        except Exception as e:
            logger.error(f"Error parsing LLM response for {symbol}: {e}")
            return self._fallback_stock_values(symbol)
    
    def _fallback_stock_values(self, symbol: str) -> Tuple[float, float, str, str]:
        """Fallback values for stock analysis"""
        return 0.06, 0.7, "neutral", f"Standard analysis for {symbol}"
    
    def _fallback_stock_analysis(self, symbol: str, stock_type: str, sector: str, technical_data: Dict, 
                               market_conditions: StockMarketConditions, horizon: str) -> Tuple[float, float, str, str]:
        """Fallback stock analysis when LLM is not available"""
        # Simple rule-based analysis
        base_return = 0.06  # 6% baseline for stocks
        
        # Adjust based on market conditions
        if market_conditions.market_momentum > 0.6:
            base_return += 0.02  # Strong momentum helps stocks
        if market_conditions.earnings_growth > 0.05:
            base_return += 0.015  # Strong earnings help stocks
        if market_conditions.interest_rate_environment < 0.03:
            base_return += 0.01  # Low rates help stocks
        
        confidence = 0.75
        sentiment = "bullish" if base_return > 0.08 else "neutral" if base_return > 0.06 else "bearish"
        summary = f"Stock analysis for {symbol}: {sentiment} outlook based on market conditions."
        
        return base_return, confidence, sentiment, summary

class StockAnalysisAgent:
    """Stock market analysis agent with macro integration"""
    
    def __init__(self, config: AgentConfig = None):
        """Initialize the stock analysis agent"""
        self.config = config or self._create_default_config()
        self.llm_engine = None
        
        if self.config.enable_llm_commentary:
            self.llm_engine = LLMCommentaryEngine(self.config.llm_config)
        
        # Default stock symbols by category
        self.default_symbols = {
            "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B"],
            "mid_cap": ["AMD", "NFLX", "CRM", "ADBE", "PYPL", "INTC", "ORCL", "IBM"],
            "small_cap": ["SNAP", "TWTR", "UBER", "LYFT", "PINS", "SQ", "ZM", "ROKU"],
            "etfs": ["SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "EFA"]
        }
        
        logger.info(f"🤖 {self.config.name} initialized")
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration for stock analysis"""
        return AgentConfig(
            name="Stock Analysis Agent",
            system_prompt="You are an expert stock market analyst specializing in comprehensive stock analysis.",
            llm_config=LLMConfig(
                model="gpt-3.5-turbo",
                temperature=0.6,
                max_tokens=1000,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            analysis_depth="comprehensive",
            enable_llm_commentary=True
        )
    
    async def classify_stock_with_llm(self, symbol: str, info: Dict) -> Tuple[str, str, str]:
        """Classify stock type, sector, and market cap using LLM"""
        if not self.llm_engine or not self.llm_engine.client:
            return "Common Stock", "Technology", "Large Cap"
        
        try:
            system_prompt = "Classify the stock type, sector, and market cap. Return JSON: {\"type\": \"Common Stock/ETF/REIT\", \"sector\": \"Technology/Healthcare/etc\", \"market_cap\": \"Large/Mid/Small Cap\"}"
            
            user_prompt = f"""
            Classify this stock:
            Symbol: {symbol}
            Info: {json.dumps(info, indent=2)}
            
            Return classification in JSON format.
            """
            
            response = await asyncio.to_thread(
                self.llm_engine.client.chat.completions.create,
                model=self.config.llm_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            content = response.choices[0].message.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                data = json.loads(content[json_start:json_end])
                return data.get('type', 'Common Stock'), data.get('sector', 'Technology'), data.get('market_cap', 'Large Cap')
            
        except Exception as e:
            logger.error(f"Error classifying stock {symbol}: {e}")
        
        return "Common Stock", "Technology", "Large Cap"
    
    async def analyze_stocks(self, symbols: List[str] = None, analysis_date: str = None, macro_context: Dict = None) -> Dict:
        """Main method to analyze stocks with optional macro context"""
        if symbols is None:
            # Use default symbols from all categories
            symbols = []
            for category_symbols in self.default_symbols.values():
                symbols.extend(category_symbols)
            
        if analysis_date is None:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting stock analysis for {len(symbols)} symbols on {analysis_date}")
        
        # Fetch market data
        market_data = await self._fetch_stock_data(symbols)
        
        if not market_data:
            logger.error("No stock data available")
            return {"error": "No stock data available", "prediction_date": analysis_date}
        
        # Get market conditions (enhanced with macro context if available)
        market_conditions = await self._get_stock_market_conditions(macro_context)
        
        # Analyze each horizon
        horizons = ['next_quarter', 'next_year', 'long_term']
        horizon_results = {}
        
        for horizon in horizons:
            predictions = await self._analyze_horizon(market_data, market_conditions, horizon)
            
            # Sort by expected return and add ranks
            predictions.sort(key=lambda x: x.expected_return, reverse=True)
            for i, prediction in enumerate(predictions):
                prediction.rank = i + 1
            
            # Convert to dict format
            horizon_results[horizon] = [asdict(p) for p in predictions]
        
        # Create final results
        final_results = {
            'prediction_date': analysis_date,
            'analysis_type': 'LLM-enhanced stock analysis with macro context' if macro_context else 'Quantitative stock analysis',
            'total_stock_investments': len(market_data),
            'horizons': horizon_results,
            'market_conditions': asdict(market_conditions),
            'macro_context_used': macro_context is not None
        }
        
        # Add macro context if available
        if macro_context:
            final_results['macro_context'] = self._extract_macro_context(macro_context)
        
        logger.info("Stock analysis completed successfully")
        return final_results
    
    async def _fetch_stock_data(self, symbols: List[str]) -> Dict:
        """Fetch historical stock data with improved error handling"""
        logger.info(f"Fetching stock data for {len(symbols)} symbols...")
        
        start_date = "2020-01-01"  # 5 years of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        def fetch_stock_data(symbol):
            try:
                ticker = yf.Ticker(symbol)
                ticker.session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                })
                
                # Try different periods if full history fails
                hist_data = None
                for period_years in [5, 3, 2, 1]:
                    try:
                        period_start = (datetime.now() - timedelta(days=365*period_years)).strftime("%Y-%m-%d")
                        hist_data = ticker.history(start=period_start, end=end_date)
                        if len(hist_data) > 50:  # Need sufficient data
                            break
                    except:
                        continue
                
                if hist_data is None or hist_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    return symbol, None
                
                # Get info with fallback
                try:
                    info = ticker.info
                except:
                    info = {}
                
                # Generate label
                label = info.get('longName', info.get('shortName', symbol))
                if len(label) > 35:
                    label = label[:32] + "..."
                
                return symbol, {
                    'price_data': hist_data,
                    'info': info,
                    'label': label
                }
            except Exception as e:
                logger.error(f"Error fetching stock data for {symbol}: {e}")
                return symbol, None
        
        market_data = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_symbol = {executor.submit(fetch_stock_data, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, data = future.result()
                    if data:
                        # Classify stock type, sector, and market cap
                        stock_type, sector, market_cap = await self.classify_stock_with_llm(symbol, data['info'])
                        data['stock_type'] = stock_type
                        data['sector'] = sector
                        data['market_cap'] = market_cap
                        market_data[symbol] = data
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        return market_data
    
    async def _get_stock_market_conditions(self, macro_context: Dict = None) -> StockMarketConditions:
        """Analyze current stock market conditions with macro context"""
        try:
            def fetch_indicator(symbol, default_value):
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="30d")
                    return data['Close'].iloc[-1] if not data.empty else default_value
                except:
                    return default_value
            
            # Market indicators
            vix_current = fetch_indicator("^VIX", 20.0)
            vix_level = min(1.0, vix_current / 40.0)
            
            # S&P 500 for market momentum
            spy_current = fetch_indicator("SPY", 400.0)
            spy_30d_ago = fetch_indicator("SPY", 380.0)  # Approximate
            market_momentum = (spy_current - spy_30d_ago) / spy_30d_ago if spy_30d_ago > 0 else 0.02
            
            # Base market conditions
            conditions = StockMarketConditions(
                vix_level=vix_level,
                market_momentum=market_momentum,
                sector_rotation="technology",  # Default
                earnings_growth=0.05,  # 5% baseline
                interest_rate_environment=0.04,  # 4% baseline
                economic_growth=0.025,  # 2.5% baseline
                market_volatility=vix_level
            )
            
            # Enhance with macro context if available
            if macro_context:
                conditions = self._enhance_market_conditions_with_macro(conditions, macro_context)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error fetching stock market conditions: {e}")
            return StockMarketConditions(0.5, 0.02, "technology", 0.05, 0.04, 0.025, 0.5)
    
    def _enhance_market_conditions_with_macro(self, conditions: StockMarketConditions, macro_context: Dict) -> StockMarketConditions:
        """Enhance market conditions with macro context"""
        try:
            # Extract macro insights
            macro_insights = self._extract_macro_insights(macro_context)
            
            # Adjust earnings growth
            if macro_insights.get('economic_cycle') == 'expansion':
                conditions.earnings_growth = 0.08
            elif macro_insights.get('economic_cycle') == 'contraction':
                conditions.earnings_growth = 0.02
            
            # Adjust interest rate environment
            if macro_insights.get('interest_rate_trend') == 'rising':
                conditions.interest_rate_environment = 0.06
            elif macro_insights.get('interest_rate_trend') == 'falling':
                conditions.interest_rate_environment = 0.02
            
            # Adjust economic growth
            if macro_insights.get('economic_cycle') == 'expansion':
                conditions.economic_growth = 0.035
            elif macro_insights.get('economic_cycle') == 'contraction':
                conditions.economic_growth = 0.015
            
            # Adjust market momentum based on sentiment
            if macro_insights.get('sentiment') == 'bullish':
                conditions.market_momentum += 0.02
            elif macro_insights.get('sentiment') == 'bearish':
                conditions.market_momentum -= 0.02
            
        except Exception as e:
            logger.error(f"Error enhancing market conditions with macro context: {e}")
        
        return conditions
    
    def _extract_macro_insights(self, macro_context: Dict) -> Dict[str, Any]:
        """Extract key insights from macro context"""
        insights = {
            'sentiment': 'neutral',
            'economic_cycle': 'unknown',
            'interest_rate_trend': 'stable'
        }
        
        try:
            # Extract from macro prediction
            if 'macro_prediction' in macro_context:
                pred = macro_context['macro_prediction']
                if 'quarterly_prediction' in pred:
                    q_pred = pred['quarterly_prediction']
                    insights['sentiment'] = q_pred.get('prediction', 'neutral')
            
            # Extract from LLM macro analysis
            if 'llm_macro_analysis' in macro_context:
                llm_macro = macro_context['llm_macro_analysis']
                
                if 'economic_environment' in llm_macro:
                    econ_env = llm_macro['economic_environment']
                    insights['economic_cycle'] = econ_env.get('cycle_phase', 'unknown')
                    insights['interest_rate_trend'] = econ_env.get('interest_rate_trend', 'stable')
            
        except Exception as e:
            logger.error(f"Error extracting macro insights: {e}")
        
        return insights
    
    def _extract_macro_context(self, macro_context: Dict) -> Dict[str, Any]:
        """Extract and format macro context for stock analysis"""
        return {
            'macro_sentiment': self._extract_macro_insights(macro_context).get('sentiment', 'neutral'),
            'economic_cycle': self._extract_macro_insights(macro_context).get('economic_cycle', 'unknown'),
            'interest_rate_trend': self._extract_macro_insights(macro_context).get('interest_rate_trend', 'stable'),
            'macro_analysis_timestamp': macro_context.get('timestamp', datetime.now().isoformat())
        }
    
    async def _analyze_horizon(self, market_data: Dict, market_conditions: StockMarketConditions, 
                             horizon: str) -> List[StockPrediction]:
        """Analyze stocks for specific time horizon"""
        predictions = []
        
        for symbol, data in market_data.items():
            try:
                prediction = await self._analyze_single_stock(
                    symbol, data, market_conditions, horizon
                )
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for {horizon}: {e}")
        
        return predictions
    
    async def _analyze_single_stock(self, symbol: str, data: Dict, 
                                 market_conditions: StockMarketConditions, horizon: str) -> StockPrediction:
        """Perform comprehensive analysis on single stock"""
        price_data = data['price_data']
        stock_type = data['stock_type']
        sector = data['sector']
        market_cap = data['market_cap']
        label = data['label']
        info = data['info']
        
        # Calculate technical indicators
        technical = self._calculate_technical_indicators(price_data)
        
        # Calculate fundamental metrics
        fundamental = self._calculate_fundamental_metrics(info, price_data)
        
        # Use LLM for prediction if available
        if self.llm_engine and self.llm_engine.client:
            expected_return, confidence, sentiment, summary = await self.llm_engine.generate_stock_commentary(
                symbol, stock_type, sector, technical, fundamental, market_conditions, horizon
            )
        else:
            expected_return, confidence, sentiment, summary = await self._predict_stock_performance_traditional(
                symbol, stock_type, sector, technical, fundamental, market_conditions, horizon
            )
        
        return StockPrediction(
            ticker=symbol,
            stock_type=stock_type,
            sector=sector,
            market_cap=market_cap,
            label=label,
            expected_return=expected_return,
            rank=0,  # Will be set later
            sentiment=sentiment,
            confidence=confidence,
            summary=summary,
            pe_ratio=fundamental.get('pe_ratio', 0),
            dividend_yield=fundamental.get('dividend_yield', 0),
            beta=fundamental.get('beta', 1.0)
        )
    
    async def _predict_stock_performance_traditional(self, symbol: str, stock_type: str, sector: str,
                                                  technical: Dict, fundamental: Dict, 
                                                  market_conditions: StockMarketConditions,
                                                  horizon: str) -> Tuple[float, float, str, str]:
        """Traditional rule-based stock prediction engine"""
        
        base_return = 0.06  # 6% baseline for stocks
        confidence = 0.80
        reasoning = []
        
        # Market momentum impact
        if market_conditions.market_momentum > 0.05:
            base_return += 0.02
            reasoning.append("Strong market momentum")
        
        # Earnings growth impact
        if market_conditions.earnings_growth > 0.05:
            base_return += 0.015
            reasoning.append("Strong earnings growth")
        
        # Interest rate environment impact
        if market_conditions.interest_rate_environment < 0.03:
            base_return += 0.01
            reasoning.append("Low interest rate environment")
        
        # VIX impact (fear trade)
        if market_conditions.vix_level > 0.6:
            base_return -= 0.01
            reasoning.append("High market volatility")
        
        # Technical indicators
        if technical.get('rsi', 50) < 30:
            base_return += 0.005
            reasoning.append("Oversold conditions")
        elif technical.get('rsi', 50) > 70:
            base_return -= 0.005
            reasoning.append("Overbought conditions")
        
        # Fundamental metrics
        pe_ratio = fundamental.get('pe_ratio', 20)
        if pe_ratio < 15:
            base_return += 0.005
            reasoning.append("Attractive valuation")
        elif pe_ratio > 30:
            base_return -= 0.005
            reasoning.append("Expensive valuation")
        
        # Determine sentiment
        if base_return > 0.08:
            sentiment = "bullish"
        elif base_return < 0.04:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Create summary
        summary = f"Stock analysis for {symbol}: {sentiment} outlook. " + "; ".join(reasoning[:3])
        
        return base_return, confidence, sentiment, summary
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for stocks"""
        try:
            close_prices = price_data['Close']
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # Moving averages
            ma_20 = close_prices.rolling(20).mean()
            ma_50 = close_prices.rolling(50).mean()
            
            # Volatility
            volatility = close_prices.pct_change().rolling(20).std()
            
            return {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0,
                'ma_20': float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else 0,
                'ma_50': float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else 0,
                'volatility': float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.02,
                'current_price': float(close_prices.iloc[-1]) if not pd.isna(close_prices.iloc[-1]) else 0
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {'rsi': 50, 'macd': 0, 'macd_signal': 0, 'ma_20': 0, 'ma_50': 0, 'volatility': 0.02, 'current_price': 0}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for stock prices"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_fundamental_metrics(self, info: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate fundamental metrics for stocks"""
        try:
            close_prices = price_data['Close']
            volume = price_data['Volume'] if 'Volume' in price_data.columns else pd.Series([0] * len(close_prices))
            
            # Price metrics
            current_price = float(close_prices.iloc[-1]) if not close_prices.empty else 0
            price_change_1m = float(((close_prices.iloc[-1] - close_prices.iloc[-22]) / close_prices.iloc[-22]) * 100) if len(close_prices) > 22 else 0
            price_change_3m = float(((close_prices.iloc[-1] - close_prices.iloc[-66]) / close_prices.iloc[-66]) * 100) if len(close_prices) > 66 else 0
            
            # Volume metrics
            avg_volume = float(volume.tail(20).mean()) if not volume.empty else 0
            current_volume = float(volume.iloc[-1]) if not volume.empty else 0
            
            return {
                'current_price': current_price,
                'price_change_1m': price_change_1m,
                'price_change_3m': price_change_3m,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 20.0),
                'dividend_yield': info.get('dividendYield', 0.0) if info.get('dividendYield') else 0.0,
                'beta': info.get('beta', 1.0),
                'price_to_book': info.get('priceToBook', 0.0),
                'debt_to_equity': info.get('debtToEquity', 0.0)
            }
        except Exception as e:
            logger.error(f"Error calculating fundamental metrics: {e}")
            return {'current_price': 0, 'price_change_1m': 0, 'price_change_3m': 0, 'avg_volume': 0, 'current_volume': 0, 'volume_ratio': 1.0, 'pe_ratio': 20.0, 'dividend_yield': 0.0, 'beta': 1.0}
    
    def save_predictions(self, predictions: Dict, filename: str = None) -> str:
        """Save stock predictions to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_predictions_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"✅ Stock predictions saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Error saving stock predictions: {e}")
            return ""
    
    def print_summary(self, predictions: Dict):
        """Print a summary of stock predictions"""
        if 'error' in predictions:
            print(f"❌ Error in stock analysis: {predictions['error']}")
            return
        
        print(f"\n📈 STOCK ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"📅 Analysis Date: {predictions.get('prediction_date', 'N/A')}")
        print(f"📊 Analysis Type: {predictions.get('analysis_type', 'N/A')}")
        print(f"🏆 Total Investments: {predictions.get('total_stock_investments', 0)}")
        print(f"🔗 Macro Context Used: {predictions.get('macro_context_used', False)}")
        
        # Show top picks for next year
        if 'horizons' in predictions and 'next_year' in predictions['horizons']:
            next_year_picks = predictions['horizons']['next_year']
            print(f"\n🏆 TOP STOCK PICKS (Next Year):")
            for i, pick in enumerate(next_year_picks[:5], 1):
                print(f"{i}. {pick['ticker']}: {pick['expected_return']:.2%} return, {pick['sentiment']} sentiment")
    
    def get_top_stock_picks(self, predictions: Dict, horizon: str = 'next_year', 
                          stock_type: str = None, sector: str = None, top_n: int = 10) -> List[Dict]:
        """Get top stock picks based on criteria"""
        if 'error' in predictions or 'horizons' not in predictions:
            return []
        
        horizon_picks = predictions['horizons'].get(horizon, [])
        
        # Filter by criteria if specified
        if stock_type:
            horizon_picks = [pick for pick in horizon_picks if pick.get('stock_type') == stock_type]
        if sector:
            horizon_picks = [pick for pick in horizon_picks if pick.get('sector') == sector]
        
        # Sort by expected return and return top N
        horizon_picks.sort(key=lambda x: x.get('expected_return', 0), reverse=True)
        return horizon_picks[:top_n]
    
    def analyze_stock_market_sentiment(self, predictions: Dict) -> Dict:
        """Analyze overall stock market sentiment"""
        if 'error' in predictions or 'horizons' not in predictions:
            return {'overall_sentiment': 'neutral', 'confidence': 0.5}
        
        all_predictions = []
        for horizon_picks in predictions['horizons'].values():
            all_predictions.extend(horizon_picks)
        
        if not all_predictions:
            return {'overall_sentiment': 'neutral', 'confidence': 0.5}
        
        # Calculate sentiment distribution
        sentiment_counts = {}
        total_confidence = 0
        weighted_sentiment = 0
        
        for pred in all_predictions:
            sentiment = pred.get('sentiment', 'neutral')
            confidence = pred.get('confidence', 0.5)
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_confidence += confidence
            
            # Weight sentiment by confidence
            sentiment_value = 1 if sentiment == 'bullish' else -1 if sentiment == 'bearish' else 0
            weighted_sentiment += sentiment_value * confidence
        
        # Determine overall sentiment
        if weighted_sentiment > 0.5:
            overall_sentiment = 'bullish'
        elif weighted_sentiment < -0.5:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        avg_confidence = total_confidence / len(all_predictions) if all_predictions else 0.5
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'sentiment_distribution': sentiment_counts,
            'weighted_sentiment_score': weighted_sentiment,
            'total_analyzed': len(all_predictions)
        }

# Convenience functions
async def quick_stock_analysis(symbols: List[str] = None, api_key: str = None, macro_context: Dict = None) -> Dict:
    """Quick stock analysis with optional macro context"""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    config = AgentConfig(
        name="Quick Stock Analyzer",
        system_prompt="Quick stock market analysis",
        llm_config=LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.6,
            max_tokens=1000,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    )
    
    agent = StockAnalysisAgent(config)
    return await agent.analyze_stocks(symbols, macro_context=macro_context)

def create_custom_stock_agent(openai_api_key: str = None, model: str = "gpt-3.5-turbo") -> StockAnalysisAgent:
    """Create a custom stock analysis agent"""
    config = AgentConfig(
        name="Custom Stock Analyzer",
        system_prompt="Custom stock market analysis agent",
        llm_config=LLMConfig(
            model=model,
            temperature=0.6,
            max_tokens=1000,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    )
    
    return StockAnalysisAgent(config)

def get_sample_stock_symbols() -> Dict[str, List[str]]:
    """Get sample stock symbols by category"""
    return {
        "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B"],
        "mid_cap": ["AMD", "NFLX", "CRM", "ADBE", "PYPL", "INTC", "ORCL", "IBM"],
        "small_cap": ["SNAP", "TWTR", "UBER", "LYFT", "PINS", "SQ", "ZM", "ROKU"],
        "etfs": ["SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "EFA"]
    }

# Example usage
async def main():
    """Example usage of the stock analysis agent"""
    print("📈 Stock Analysis Agent Example")
    print("=" * 40)
    
    # Create agent
    agent = StockAnalysisAgent()
    
    # Analyze stocks
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    results = await agent.analyze_stocks(symbols)
    
    # Print summary
    agent.print_summary(results)
    
    # Get top picks
    top_picks = agent.get_top_stock_picks(results, horizon='next_year', top_n=10)
    print(f"\n🏆 Top 10 Stock Picks:")
    for pick in top_picks:
        print(f"  - {pick['ticker']}: {pick['expected_return']:.2%} return")

if __name__ == "__main__":
    asyncio.run(main())