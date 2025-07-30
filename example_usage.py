#!/usr/bin/env python3
"""
Example usage of the AI Macro Analysis Agent.
Demonstrates how to use the agent for quarterly NASDAQ 100 predictions.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_agent import MacroAnalysisAgent
from loguru import logger

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def main():
    """Main example function."""

    print("""
🤖 AI MACRO ANALYSIS AGENT - ENHANCED TIME SERIES EXAMPLE
=========================================================
    
This example demonstrates the ENHANCED MacroAnalysisAgent with 
advanced time series features for quarterly NASDAQ 100 predictions.

🚀 NEW ENHANCED FEATURES:
   • 135+ Time Series Features (vs. 14 basic features)
   • Lag Features: Previous quarter values & changes
   • Autoregressive: Temporal dependencies & mean reversion
   • Trend Features: Long-term patterns & accelerations
   • Cyclical Features: Business cycle & recession indicators
   • Cross-lag Features: Economic relationship delays
   • Stationarity Features: Change-based & regime detection
    """)

    try:
        # Step 1: Initialize the agent
        print("\n🚀 STEP 1: Initializing Agent...")
        print("-" * 40)

        agent = MacroAnalysisAgent()

        # Step 2: Perform health check
        print("\n🔍 STEP 2: Health Check...")
        print("-" * 40)

        health = agent.health_check()
        print("Health Status:")
        for component, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}: {status}")

        if not all(health.values()):
            print("\n⚠️ WARNING: Some components failed health check.")
            print("   Make sure your .env file is configured correctly.")
            print("   Check that your Neon DB contains macro data.")
            print("   Ensure you have a trained model in the models/ directory.")
            return

        # Step 3: Initialize agent (load models)
        print("\n📚 STEP 3: Loading Models...")
        print("-" * 40)

        if not agent.initialize():
            print("❌ Agent initialization failed. Check logs above.")
            return

        # Step 4: Get agent status
        print("\n📊 STEP 4: Agent Status...")
        print("-" * 40)

        status = agent.get_agent_status()
        print(f"Agent Name: {status['agent_name']}")
        print(f"Initialized: {status['initialized']}")
        print(f"Model Type: {status['model_info']['model_type']}")
        print(f"Features Available: {status['data_status']['feature_count']}")
        print(f"Macro Records: {status['data_status']['macro_records']}")

        # Step 5: Generate prediction
        print("\n🎯 STEP 5: Generating Prediction...")
        print("-" * 40)

        # Optional: Add market context
        market_context = """
        Current market environment shows elevated uncertainty due to:
        - Federal Reserve policy decisions
        - Geopolitical tensions
        - Technology sector volatility
        - Inflation concerns
        """

        prediction_result = agent.predict_next_quarter(
            include_openai_analysis=True, market_context=market_context.strip()
        )

        # Step 6: Display results
        print("\n📋 STEP 6: Prediction Results...")
        print("-" * 40)

        # Basic prediction info
        pred = prediction_result["prediction"]
        print(f"🎯 PREDICTION: {pred['prediction'].upper()}")
        print(f"📊 CONFIDENCE: {pred['confidence']:.1%}")
        print(f"📅 TARGET QUARTER: {prediction_result['target_quarter']}")
        print(f"🤖 MODEL: {pred['model_name']}")

        # Probability breakdown
        print("\n📈 PROBABILITIES:")
        print(f"   • Bullish: {pred['probabilities']['bullish']:.1%}")
        print(f"   • Bearish: {pred['probabilities']['bearish']:.1%}")

        # Feature importance
        if prediction_result.get("feature_importance"):
            print("\n🔍 TOP 5 INFLUENTIAL FEATURES:")
            for i, (feature, importance) in enumerate(
                prediction_result["feature_importance"][:5], 1
            ):
                value = prediction_result["features"].get(feature, 0.0)
                print(f"   {i}. {feature}: {value:.3f} (importance: {importance:.3f})")

        # Step 7: Get formatted summary
        print("\n📝 STEP 7: Formatted Summary...")
        print("-" * 40)

        summary = agent.get_prediction_summary(include_technical=False)
        print(summary)

        # Step 8: OpenAI Analysis (if available)
        if (
            prediction_result.get("openai_analysis")
            and "prediction_report" in prediction_result["openai_analysis"]
        ):
            print("\n🧠 STEP 8: AI-Generated Analysis...")
            print("-" * 40)
            print(prediction_result["openai_analysis"]["prediction_report"])

            if "factor_explanation" in prediction_result["openai_analysis"]:
                print("\n🔍 Factor Explanation:")
                print(prediction_result["openai_analysis"]["factor_explanation"])

        # Step 9: Market Analysis
        print("\n📊 STEP 9: Market Analysis...")
        print("-" * 40)

        market_analysis = agent.get_market_analysis()
        print(market_analysis)

        print("\n✅ Example completed successfully!")
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("• Review the prediction and analysis")
        print("• Consider additional risk factors")
        print("• Implement appropriate position sizing")
        print("• Monitor model performance over time")
        print("• Retrain models as new data becomes available")

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        print(f"\n❌ ERROR: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check your .env file configuration")
        print("2. Ensure Neon DB is accessible and contains data")
        print("3. Verify OpenAI API key is valid")
        print("4. Make sure you have trained models in models/ directory")
        print("5. Check that all required packages are installed")

    finally:
        # Cleanup
        try:
            agent.close()
        except:
            pass


def demo_quick_prediction():
    """Quick demonstration of making a prediction."""

    print("\n🚀 QUICK PREDICTION DEMO")
    print("=" * 30)

    try:
        # Create and initialize agent
        agent = MacroAnalysisAgent()

        if not agent.initialize():
            print("❌ Quick demo failed - agent initialization error")
            return

        # Make prediction (without OpenAI to be faster)
        result = agent.predict_next_quarter(include_openai_analysis=False)

        # Show results
        pred = result["prediction"]
        print(
            f"🎯 QUICK PREDICTION: {pred['prediction'].upper()} ({pred['confidence']:.1%})"
        )
        print(f"📅 Target Quarter: {result['target_quarter']}")
        print(f"🤖 Model: {pred['model_name']}")

        print("✅ Quick demo complete!")

    except Exception as e:
        print(f"❌ Quick demo failed: {e}")

    finally:
        try:
            agent.close()
        except:
            pass


def check_environment():
    """Check if environment is properly configured."""

    print("\n🔍 ENVIRONMENT CHECK")
    print("=" * 25)

    required_vars = [
        "NEON_DB_URL",
        "NEON_DB_HOST",
        "NEON_DB_NAME",
        "NEON_DB_USER",
        "NEON_DB_PASSWORD",
        "OPENAI_API_KEY",
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n💡 Create a .env file based on config_example.env")
        return False
    else:
        print("✅ All required environment variables are set")
        return True


def demonstrate_enhanced_features():
    """Demonstrate the enhanced time series features without full prediction."""

    print("\n🚀 ENHANCED TIME SERIES FEATURES DEMO")
    print("=" * 40)

    try:
        # Just initialize agent to show feature capabilities
        agent = MacroAnalysisAgent()

        print("✅ Enhanced Time Series Agent Initialized!")
        print("\n📊 NEW FEATURE CATEGORIES AVAILABLE:")
        print("   📅 Lag Features: vix_lag_1q, unemployment_rate_lag_2q, etc.")
        print("   🔄 Autoregressive: vix_mean_reversion, fed_funds_rate_ar1, etc.")
        print(
            "   📈 Trend Features: vix_trend_slope_8q, treasury_10y_acceleration, etc."
        )
        print("   🌊 Cyclical: business_cycle_6y, recession_indicator, etc.")
        print("   🔗 Cross-lag: fed_lag2q_x_unemployment_rate, etc.")
        print("   ⚖️ Stationarity: vix_diff_1q, unemployment_rate_zscore, etc.")

        print("\n🎯 ENHANCEMENT SUMMARY:")
        print("   • From ~14 basic features → 135+ time series features")
        print("   • Captures quarterly economic relationships")
        print("   • Models business cycle patterns (6-8 year cycles)")
        print("   • Includes Fed policy lags (2-4 quarter delays)")
        print("   • Detects regime changes and trend reversals")

        agent.close()

    except Exception as e:
        print(f"⚠️ Feature demo error: {e}")


if __name__ == "__main__":
    print("🤖 AI MACRO ANALYSIS AGENT - ENHANCED")
    print("=" * 50)

    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed. Please configure your .env file.")
        sys.exit(1)

    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Full demonstration (with OpenAI analysis)")
    print("2. Quick prediction (without OpenAI)")
    print("3. Enhanced features demo (showcase new capabilities)")
    print("4. Environment check only")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        demo_quick_prediction()
    elif choice == "3":
        demonstrate_enhanced_features()
    elif choice == "4":
        print("✅ Environment check already completed above.")
    else:
        print("❌ Invalid choice. Running enhanced features demo...")
        demonstrate_enhanced_features()
