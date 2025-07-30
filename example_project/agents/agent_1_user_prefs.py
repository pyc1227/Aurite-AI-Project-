import argparse
import asyncio
import json
import logging
import os
import uuid

from aurite import Aurite
from termcolor import colored
from utils.aurite_instance import AuriteSingleton, shutdown_aurite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- User Preferences Schema ---
user_preferences_schema = {
    "type": "object",
    "properties": {
        "user_id": {
            "type": "string",
            "description": "A unique identifier for the user.",
        },
        "risk_tolerance": {
            "type": "string",
            "enum": ["Low", "Medium", "High"],
            "description": "The user's tolerance for investment risk.",
        },
        "investment_goals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of the user's primary investment goals (e.g., 'Long-term Growth', 'Retirement Savings', 'Income Generation').",
        },
        "preferred_sectors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of industries or sectors the user is interested in (e.g., 'Technology', 'Healthcare', 'Renewable Energy').",
        },
    },
    "required": ["user_id", "risk_tolerance", "investment_goals", "preferred_sectors"],
    "additionalProperties": False,
}


async def run_user_prefs_agent(
    aurite: Aurite, user_input: str, user_id_arg: str = None
):
    """
    Runs the user preferences agent and saves the structured output to a file.
    """
    print(colored("\n--- Running User Preferences Agent ---", "yellow", attrs=["bold"]))
    print(f'User Input: "{user_input}"')

    agent_result = await aurite.run_agent(
        agent_name="user_prefs_agent", user_message=user_input
    )

    if not (agent_result and hasattr(agent_result, "primary_text")):
        print(colored("Error: Did not receive a valid response from the agent.", "red"))
        return

    try:
        preferences = json.loads(agent_result.primary_text)

        # Determine the user_id
        if user_id_arg:
            user_id = user_id_arg
            print(colored(f"Using provided user_id: {user_id}", "yellow"))
        else:
            user_id = preferences.get("user_id")
            if not user_id:
                # If agent doesn't generate a user_id, create one.
                user_id = str(uuid.uuid4())
                print(
                    colored(
                        f"Agent did not provide a user_id. Generated new one: {user_id}",
                        "yellow",
                    )
                )

        preferences["user_id"] = user_id

        # Define the output path and ensure the directory exists
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, "..", "data", "user_preferences")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"user_{user_id}.json")

        # Save the structured data to the file
        with open(output_path, "w") as f:
            json.dump(preferences, f, indent=4)

        print(
            colored(f"\nSuccessfully saved user preferences to: {output_path}", "green")
        )
        print(colored(json.dumps(preferences, indent=4), "cyan"))

        return user_id

    except json.JSONDecodeError:
        print(colored("Error: Failed to parse agent response as JSON.", "red"))
        print(f"Raw response: {agent_result.primary_text}")
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}", exc_info=True)

    return None


async def main():
    """Main function to set up Aurite and run the agent test."""
    parser = argparse.ArgumentParser(description="Run the user preferences agent.")
    parser.add_argument(
        "--message",
        type=str,
        default="Hi, I'm a new investor. I don't like a lot of risk, so let's keep it low. I'm mainly saving up for retirement and maybe a new car. I'm interested in tech and green energy companies.",
        help="The user's message describing their investment preferences.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="The user ID to assign to the preferences. If not provided, a new one will be generated.",
    )
    args = parser.parse_args()

    aurite = await AuriteSingleton.get_instance()
    try:
        await run_user_prefs_agent(aurite, args.message, args.user)
    finally:
        await shutdown_aurite()


if __name__ == "__main__":
    asyncio.run(main())
