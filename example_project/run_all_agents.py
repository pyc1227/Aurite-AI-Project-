import argparse
import asyncio
import logging

from agents.agent_1_user_prefs import run_user_prefs_agent
from agents.agent_2_3_finance_data import run_finance_data_workflow
from agents.agent_4_portfolio_analysis import run_portfolio_analysis_agent
from termcolor import colored
from utils.aurite_instance import AuriteSingleton, shutdown_aurite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Runs the complete financial agent workflow using a shared Aurite instance.
    """
    parser = argparse.ArgumentParser(
        description="Run the full financial agent workflow."
    )
    parser.add_argument(
        "--message",
        type=str,
        default="I'm a cautious investor looking for long-term growth. I'm interested in healthcare and renewable energy.",
        help="The user's message describing their investment preferences.",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="The user ID to run the analysis for. If not provided, the most recently created user will be used.",
    )
    args = parser.parse_args()

    aurite = await AuriteSingleton.get_instance()
    try:
        # --- Agent 1: User Preferences ---
        user_id = await run_user_prefs_agent(aurite, args.message, args.user)

        if not user_id:
            print(colored("Could not determine user ID. Aborting.", "red"))
            return

        # --- Agents 2 & 3: Financial Data Workflow ---
        await run_finance_data_workflow(aurite)

        # --- Agent 4: Portfolio Analysis ---
        await run_portfolio_analysis_agent(aurite, user_id)

    finally:
        await shutdown_aurite()


if __name__ == "__main__":
    asyncio.run(main())
