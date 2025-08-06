#!/usr/bin/env python3
"""
Simple Prediction Interface
Asks user if they want to predict next quarter and outputs bullish/bearish with probability.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_agent.agent import MacroAnalysisAgent
from loguru import logger

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stdout, 
    level="INFO", 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def main():
    """Simple prediction interface."""
    
    print("🤖 QUARTERLY PREDICTION SYSTEM")
    print("=" * 40)
    print("Using trained unified model with API data")
    print()
    
    try:
        # Initialize the agent
        print("🚀 Initializing prediction system...")
        agent = MacroAnalysisAgent()
        
        if not agent.initialize():
            print("❌ Failed to initialize prediction system")
            print("💡 Try running: python train_unified_model.py")
            return
        
        print("✅ Prediction system ready!")
        print()
        
        while True:
            # Ask user if they want to predict
            print("🎯 Do you want to predict the next quarter?")
            print("1. Yes - Make prediction")
            print("2. No - Exit")
            
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                make_prediction(agent)
            elif choice == "2":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Application error: {e}")


def make_prediction(agent):
    """Make a prediction and display results."""
    print("\n🎯 Generating prediction...")
    print("(Fetching latest data from APIs)")
    
    try:
        # Generate prediction
        result = agent.predict_next_quarter(include_openai_analysis=False)
        
        if result:
            print("\n" + "=" * 50)
            print("📊 PREDICTION RESULTS")
            print("=" * 50)
            
            # Get prediction direction and probability
            direction = result['prediction'].upper()
            confidence = result['confidence']
            target_quarter = result['target_quarter']
            model_used = result['agent_info']['model_used']
            
            # Display results
            print(f"🎯 Direction: {direction}")
            print(f"📊 Probability: {confidence:.1%}")
            print(f"📅 Target Quarter: {target_quarter}")
            print(f"🤖 Model: {model_used}")
            
            # Add interpretation
            if direction == "BULLISH":
                print(f"📈 Interpretation: Market expected to rise with {confidence:.1%} confidence")
            else:
                print(f"📉 Interpretation: Market expected to fall with {confidence:.1%} confidence")
            
            print("=" * 50)
            
        else:
            print("❌ Prediction failed. Check logs for details.")
    
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        logger.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main() 