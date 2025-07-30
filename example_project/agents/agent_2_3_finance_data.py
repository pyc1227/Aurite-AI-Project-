import asyncio
import json
import logging
import os

from aurite import Aurite
from termcolor import colored
from utils.aurite_instance import AuriteSingleton, shutdown_aurite
from utils.financial_model import run_financial_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_finance_data_workflow(aurite: Aurite):
    """
    Runs a workflow to collect and process financial data.
    """
    print(colored("\n--- Running Finance Data Workflow ---", "yellow", attrs=["bold"]))

    # Run the workflow
    workflow_result = await aurite.run_workflow(
        "finance_data_workflow", "Generate financial data."
    )

    macro_data = {}
    micro_data = {}
    for step_result in workflow_result.step_results:
        if step_result.step_name == "macro_economic_agent":
            macro_data = json.loads(step_result.result.primary_text)
        elif step_result.step_name == "micro_economic_agent":
            micro_data = json.loads(step_result.result.primary_text)

    # Run the financial model
    model_output = run_financial_model(macro_data, micro_data)

    # Save the model output
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "..", "data", "financial_analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "latest_analysis.json")

    with open(output_path, "w") as f:
        json.dump(model_output, f, indent=4)

    print(
        colored(f"\nSuccessfully saved financial analysis to: {output_path}", "green")
    )
    print(colored(json.dumps(model_output, indent=4), "cyan"))


async def main():
    """Main function to set up Aurite and run the agent test."""
    aurite = await AuriteSingleton.get_instance()
    try:
        await run_finance_data_workflow(aurite)
    finally:
        await shutdown_aurite()


if __name__ == "__main__":
    asyncio.run(main())
