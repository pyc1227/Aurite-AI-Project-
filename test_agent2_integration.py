#!/usr/bin/env python3
"""
Test script for Agent 2 MCP Server Integration
Tests both macro analysis and bond analysis capabilities
"""

import asyncio
import json
import os
from pathlib import Path

# Add MCP Server directory to path
import sys
sys.path.append(str(Path(__file__).parent / "MCP Server"))

async def test_agent2_integration():
    """Test the integrated Agent 2 MCP server"""
    
    print("🧪 Testing Agent 2 MCP Server Integration")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "MCP Server/agent2_analysis_mcp_server.py",
        "etf_analysis_agent.py",
        "ai_agent/agent.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            return
    
    # Check environment variables
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY found")
    else:
        print("⚠️  OPENAI_API_KEY not set - some features may not work")
    
    # Test importing components
    print("\n📦 Testing Component Imports...")
    
    try:
        from MCP Server.agent2_analysis_mcp_server import Agent2MCPServer
        print("✅ Agent2MCPServer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Agent2MCPServer: {e}")
        return
    
    try:
        from ai_agent.agent import MacroAnalysisAgent
        print("✅ MacroAnalysisAgent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MacroAnalysisAgent: {e}")
    
    try:
        from etf_analysis_agent import BondAnalysisAgent
        print("✅ BondAnalysisAgent imported successfully")
    except ImportError as e:
        print(f"⚠️  Failed to import BondAnalysisAgent: {e}")
    
    # Test server initialization
    print("\n🚀 Testing Server Initialization...")
    
    try:
        server = Agent2MCPServer()
        print("✅ Agent2MCPServer initialized successfully")
        
        # Test health check
        print("\n🏥 Testing Health Check...")
        health_status = await server.health_check()
        print(f"Health Status: {json.dumps(health_status, indent=2)}")
        
        # Test macro prediction
        print("\n📊 Testing Macro Prediction...")
        if server.macro_agent:
            try:
                prediction = await server.get_macro_prediction()
                print(f"Macro Prediction: {json.dumps(prediction, indent=2)}")
            except Exception as e:
                print(f"⚠️  Macro prediction failed: {e}")
        
        # Test bond analysis
        print("\n📈 Testing Bond Analysis...")
        if server.bond_agent:
            try:
                bond_result = await server.get_bond_analysis(["TLT", "IEF", "AGG", "BND"])
                print(f"Bond Analysis: {json.dumps(bond_result, indent=2)}")
            except Exception as e:
                print(f"⚠️  Bond analysis failed: {e}")
        
        # Test full analysis workflow
        print("\n🔄 Testing Full Analysis Workflow...")
        test_prompt = "I want to invest $10,000 with moderate risk tolerance, focusing on bonds and fixed income"
        
        try:
            full_analysis = await server.analyze_market_with_prompt(test_prompt, "comprehensive")
            print("✅ Full analysis workflow completed")
            print(f"Analysis Status: {full_analysis.get('status', 'unknown')}")
            
            if full_analysis.get('status') == 'success':
                print("📁 Output Files:")
                for file_type, file_path in full_analysis.get('output_files', {}).items():
                    if file_path:
                        print(f"  - {file_type}: {file_path}")
            
        except Exception as e:
            print(f"⚠️  Full analysis workflow failed: {e}")
        
        print("\n✅ Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return

def test_standalone_components():
    """Test individual components without MCP server"""
    
    print("\n🔧 Testing Standalone Components...")
    print("=" * 40)
    
    # Test macro analysis agent
    print("\n📊 Testing Macro Analysis Agent...")
    try:
        from ai_agent.agent import MacroAnalysisAgent
        agent = MacroAnalysisAgent()
        agent.initialize()
        
        prediction = agent.predict_next_quarter()
        print(f"✅ Macro prediction: {prediction}")
        
    except Exception as e:
        print(f"❌ Macro agent test failed: {e}")
    
    # Test bond analysis agent
    print("\n📈 Testing Bond Analysis Agent...")
    try:
        from etf_analysis_agent import BondAnalysisAgent, AgentConfig, LLMConfig
        
        config = AgentConfig(
            name="Test Bond Agent",
            system_prompt="You are a test bond analyst.",
            llm_config=LLMConfig(
                model="gpt-3.5-turbo",
                temperature=0.6,
                max_tokens=1000,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )
        
        bond_agent = BondAnalysisAgent(config)
        print("✅ Bond agent created successfully")
        
    except Exception as e:
        print(f"❌ Bond agent test failed: {e}")

if __name__ == "__main__":
    print("🧪 Agent 2 MCP Server Integration Test")
    print("=" * 50)
    
    # Test standalone components first
    test_standalone_components()
    
    # Test full integration
    asyncio.run(test_agent2_integration())
    
    print("\n🎉 All tests completed!") 