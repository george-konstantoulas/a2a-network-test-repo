from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents import Agent
from os import getenv
from truth_agent.auth import a2a_httpx_client

ZERO_SHOT_URL = getenv("ZERO_SHOT_AGENT_URL","http://localhost:8001")
FINE_TUNED_URL = getenv("FINE_TUNED_AGENT_URL","http://localhost:8002")
EXPLAINER_URL = getenv("EXPLAINER_AGENT_URL","http://localhost:8003")
FINE_TUNING_URL = getenv("FINE_TUNING_AGENT_URL","http://localhost:8004")
AGENT_CARD_PATH = "/.well-known/agent-card.json"

zero_shot_prediction_agent = RemoteA2aAgent(
    "zero_shot_prediction_agent",
    agent_card=f"{ZERO_SHOT_URL}{AGENT_CARD_PATH}",
    httpx_client=a2a_httpx_client(ZERO_SHOT_URL),
    description="""
    Invokes 'dataset_info_tool' to summarize data and `zero_shot_predict_tool`
    to return predicted labels and if possible, classification metrics.
    """
)

fine_tuned_prediction_agent = RemoteA2aAgent(
    "fine_tuned_prediction_agent",
    agent_card=f"{FINE_TUNED_URL}{AGENT_CARD_PATH}",
    httpx_client=a2a_httpx_client(FINE_TUNED_URL),
    description="""
    Invokes 'dataset_info_tool' to summarize data and `fine_tuned_predict_tool`
    to return predicted labels and if possible, classification metrics.
    """
)

explainer_agent = RemoteA2aAgent(
    "explainer_agent",
    agent_card=f"{EXPLAINER_URL}{AGENT_CARD_PATH}",
    httpx_client=a2a_httpx_client(EXPLAINER_URL),
    description="""
    Invokes 'dataset_info_tool' to summarize data and `explain_tool`
    to return predicted labels, explanations, and if possible, classification metrics.
    """
)

fine_tuning_agent = RemoteA2aAgent(
    "fine_tuning_agent",
    agent_card=f"{FINE_TUNING_URL}{AGENT_CARD_PATH}",
    httpx_client=a2a_httpx_client(FINE_TUNING_URL),
    description="""
    Invokes 'dataset_info_tool' to summarize data and `fine_tune_tool` to fine tune the model and return the endpoint URL.
    """
)
orchestrator_agent = Agent(
    name="orchestrator_agent",
    model="gemini-3.1-flash-lite",
    description="""
        Agent-to-Agent Network Coordinator. Receives user queries and delegates tasks
        accordingly.""",
    instruction="""
        Analyze incoming requests and delegate tasks:
        - Zero-shot predictions -> zero_shot_prediction agent
        - Fine-tuned predictions -> fine_tuned_prediction_agent
        - Fine tuning tasks -> fine_tuning_agent
        - Providing explanations -> explainer_agent

        - When the user requests a model comparison or predictions from both models (e.g. consensus, aggregation),
        delegate to both predictors. Declare the predictor with better metric performance as the most credible.

        - Explanation also carries out predictions, so there is no need to delegete
        an explanation task to a prediction agent before delegating to explainer agent.
        In either prediction or explanation tasks, assume the zero-shot prediction model
        as default, unless the user specifies otherwise.
        """,
    sub_agents=[
        zero_shot_prediction_agent,
        fine_tuned_prediction_agent,
        explainer_agent,
        fine_tuning_agent
    ]
)