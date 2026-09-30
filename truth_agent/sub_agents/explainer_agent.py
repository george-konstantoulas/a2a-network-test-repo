import os

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from truth_agent.auth import mcp_httpx_client_factory

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8080/sse")
MCP_CLIENT_FACTORY = mcp_httpx_client_factory(MCP_SERVER_URL)

explainer_agent = Agent(
    name="explainer_agent",
    model="gemini-3.1-flash-lite",
    description="""
        Invokes the explanation tool to make predictions on input data
        and provide explanations for said predictions. Also, if possible,
        calculates classification metrics""",
    instruction="""
        You are an agent that invokes the explanation tool. You also make predictions,
        so if the user asks for explanations and metrics, do not delegate the task
        to the prediction agents.
        YOUR WORKFLOW:
        1. Invoke the `dataset_info_tool` on the provided path to confirm the dataset is valid and not how many statements it contains.
        2. Use the zero-shot predictor by default. If the user specifies
        his chosen model, use that for predictions.
        4. Invoke the `explain_tool` to make predictions on input data and provide explanations
        for the decisions.
        5. If provided with ground-truth labels, calculate classification metrics.
        6. Return (predictions,explanations,metrics).""",
    tools = [McpToolset(
        connection_params=SseConnectionParams(
            url=MCP_SERVER_URL,
            httpx_client_factory=MCP_CLIENT_FACTORY
        ),
        tool_filter = ["explain_tool","dataset_info_tool"]
    )]
)