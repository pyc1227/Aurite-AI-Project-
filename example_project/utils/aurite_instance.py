import logging

from aurite import Aurite
from aurite.config.config_models import AgentConfig, LLMConfig, WorkflowConfig
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuriteSingleton:
    _instance = None

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = Aurite()
            await cls._instance.initialize()
            await setup_components(cls._instance)
        return cls._instance


async def setup_components(aurite: Aurite):
    """Registers all LLMs and Agents with the Aurite instance."""
    load_dotenv()

    # ---LLM---
    llm_config = LLMConfig(
        llm_id="gpt4_turbo",
        provider="openai",
        model_name="gpt-4-turbo-preview",
        max_tokens=1000,
    )
    await aurite.register_llm_config(llm_config)

    # ---Agent Schemas---
    from agents.agent_1_user_prefs import user_preferences_schema
    from agents.agent_4_portfolio_analysis import portfolio_analysis_schema

    macro_economic_schema = {
        "type": "object",
        "properties": {
            "gdp_growth_rate": {"type": "number"},
            "inflation_rate": {"type": "number"},
            "unemployment_rate": {"type": "number"},
        },
        "required": ["gdp_growth_rate", "inflation_rate", "unemployment_rate"],
    }

    micro_economic_schema = {
        "type": "object",
        "properties": {
            "nasdaq_composite_index": {"type": "number"},
            "average_pe_ratio": {"type": "number"},
        },
        "required": ["nasdaq_composite_index", "average_pe_ratio"],
    }

    # ---Agent Configs---
    user_prefs_agent_config = AgentConfig(
        name="user_prefs_agent",
        system_prompt="""You are a helpful assistant designed to capture a user's investment preferences.
Your task is to understand the user's input and extract their preferences into a structured JSON format.

**Instructions:**
1.  Read the user's message to understand their risk tolerance, goals, and interests.
2.  Generate a unique `user_id` for them using a UUID format.
3.  Map their description to the appropriate fields in the JSON schema.
    -   `risk_tolerance` must be one of 'Low', 'Medium', or 'High'.
    -   `investment_goals` and `preferred_sectors` should be lists of strings.
4.  Your final output MUST be ONLY the JSON object that adheres to the required schema. Do not include any other text or explanations.
""",
        mcp_servers=[],
        include_history=False,
        config_validation_schema=user_preferences_schema,
        llm_config_id="gpt4_turbo",
    )

    macro_economic_agent_config = AgentConfig(
        name="macro_economic_agent",
        system_prompt="Generate mock macroeconomic data including GDP growth, inflation, and unemployment rates.",
        config_validation_schema=macro_economic_schema,
        llm_config_id="gpt4_turbo",
    )

    micro_economic_agent_config = AgentConfig(
        name="micro_economic_agent",
        system_prompt="Generate mock NASDAQ microeconomic data including the composite index and average P/E ratio.",
        config_validation_schema=micro_economic_schema,
        llm_config_id="gpt4_turbo",
    )

    portfolio_agent_config = AgentConfig(
        name="portfolio_agent",
        system_prompt="""You are a financial portfolio analyst. Your task is to analyze financial data based on a user's specific investment preferences and generate a clear, structured portfolio recommendation.

You will receive a user message containing two JSON objects: 'User Preferences' and 'Financial Data'.

**Instructions:**
1.  **Analyze User Preferences:** Carefully review the user's risk tolerance, investment goals, and any specified interests or exclusions.
2.  **Analyze Financial Data:** Examine the provided financial reports, news summaries, and market trends.
3.  **Synthesize and Recommend:** Based on your analysis of both datasets, create a set of investment recommendations.
    -   Your recommendations must directly align with the user's preferences. For example, if the user has a 'Low' risk tolerance, do not recommend high-risk assets.
    -   For each recommendation (BUY, SELL, HOLD), provide a concise but specific justification that references both the user's goals and the financial data.
4.  **Format Output:** Your final output MUST be a single JSON object that strictly adheres to the required schema. Do not include any extra text, explanations, or markdown formatting outside of the JSON object itself.
""",
        mcp_servers=[],
        include_history=False,
        config_validation_schema=portfolio_analysis_schema,
        llm_config_id="gpt4_turbo",
    )

    await aurite.register_agent(user_prefs_agent_config)
    await aurite.register_agent(macro_economic_agent_config)
    await aurite.register_agent(micro_economic_agent_config)
    await aurite.register_agent(portfolio_agent_config)

    # ---Workflow---
    finance_workflow_config = WorkflowConfig(
        name="finance_data_workflow",
        steps=["macro_economic_agent", "micro_economic_agent"],
        description="A workflow to collect macroeconomic and microeconomic data.",
    )
    await aurite.register_workflow(finance_workflow_config)


async def shutdown_aurite():
    """Shuts down the Aurite instance if it exists."""
    instance = await AuriteSingleton.get_instance()
    if instance:
        await instance.shutdown()
        AuriteSingleton._instance = None
        logger.info("Aurite shutdown complete.")
