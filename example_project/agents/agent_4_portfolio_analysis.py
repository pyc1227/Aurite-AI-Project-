import argparse
import asyncio
import json
import logging
import os

from aurite import Aurite
from termcolor import colored
from utils.aurite_instance import AuriteSingleton, shutdown_aurite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Portfolio Analysis Schema ---
portfolio_analysis_schema = {
    "type": "object",
    "properties": {
        "analysis_summary": {
            "type": "string",
            "description": "A high-level summary of the portfolio analysis.",
        },
        "investment_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {"type": "string"},
                    "recommendation": {
                        "type": "string",
                        "enum": ["BUY", "SELL", "HOLD"],
                    },
                    "justification": {"type": "string"},
                },
                "required": ["ticker_symbol", "recommendation", "justification"],
            },
        },
        "risk_assessment": {
            "type": "string",
            "enum": ["Low", "Medium", "High"],
        },
    },
    "required": ["analysis_summary", "investment_recommendations", "risk_assessment"],
}


async def run_portfolio_analysis_agent(aurite: Aurite, user_id: str):
    """
    Runs the portfolio analysis agent for a specific user.
    """
    print(
        colored("\n--- Running Portfolio Analysis Agent ---", "yellow", attrs=["bold"])
    )

    # 1. Load user preferences
    try:
        script_dir = os.path.dirname(__file__)
        user_prefs_path = os.path.join(
            script_dir, "..", "data", "user_preferences", f"user_{user_id}.json"
        )
        with open(user_prefs_path, "r") as f:
            user_prefs = json.load(f)
    except FileNotFoundError:
        print(colored(f"Error: Preferences for user '{user_id}' not found.", "red"))
        return

    # 2. Load financial data
    try:
        script_dir = os.path.dirname(__file__)
        financial_data_path = os.path.join(
            script_dir, "..", "data", "financial_analysis", "latest_analysis.json"
        )
        with open(financial_data_path, "r") as f:
            financial_data = json.load(f)
    except FileNotFoundError:
        print(colored("Error: Financial analysis data not found.", "red"))
        return

    # 3. Construct user message
    user_message = f"""
    Analyze the following financial data based on the provided user investment preferences.

    User Preferences:
    {json.dumps(user_prefs, indent=2)}

    Financial Data:
    {json.dumps(financial_data, indent=2)}
    """

    # 4. Run agent
    agent_result = await aurite.run_agent(
        agent_name="portfolio_agent", user_message=user_message
    )

    if not (agent_result and hasattr(agent_result, "primary_text")):
        print(colored("Error: Did not receive a valid response from the agent.", "red"))
        return

    # 5. Process and save response
    try:
        portfolio_analysis = json.loads(agent_result.primary_text)
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, "..", "data", "portfolio_analysis")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"portfolio_{user_id}.json")

        with open(output_path, "w") as f:
            json.dump(portfolio_analysis, f, indent=4)

        print(
            colored(
                f"\nSuccessfully saved portfolio analysis to: {output_path}", "green"
            )
        )
        print(colored(json.dumps(portfolio_analysis, indent=4), "cyan"))

    except json.JSONDecodeError:
        print(colored("Error: Failed to parse agent response as JSON.", "red"))
        print(f"Raw response: {agent_result.primary_text}")
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}", exc_info=True)


async def main():
    """Main function to set up Aurite and run the agent test."""
    parser = argparse.ArgumentParser(description="Run the portfolio analysis agent.")
    parser.add_argument(
        "--user",
        type=str,
        help="The user ID to run the analysis for. If not provided, the first user found will be used.",
    )
    args = parser.parse_args()

    aurite = await AuriteSingleton.get_instance()
    try:
        user_id = args.user
        if not user_id:
            script_dir = os.path.dirname(__file__)
            user_prefs_dir = os.path.join(script_dir, "..", "data", "user_preferences")
            user_files = [f for f in os.listdir(user_prefs_dir) if f.endswith(".json")]
            if not user_files:
                print(
                    colored("No user preference files found. Run agent 1 first.", "red")
                )
                return
            user_id = user_files[0].replace("user_", "").replace(".json", "")

        await run_portfolio_analysis_agent(aurite, user_id)
    finally:
        await shutdown_aurite()


if __name__ == "__main__":
    asyncio.run(main())
