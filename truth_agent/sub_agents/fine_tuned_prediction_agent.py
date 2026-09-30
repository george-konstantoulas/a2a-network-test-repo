import os

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from truth_agent.auth import mcp_httpx_client_factory

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8080/sse")
MCP_CLIENT_FACTORY = mcp_httpx_client_factory(MCP_SERVER_URL)

fine_tuned_prediction_agent = Agent(
    name="fine_tuned_prediction_agent",
    model="gemini-3.1-flash-lite",
    description="""
        Invokes fine-tuned classification tool using
        the FineTunedPredictor instance.Returns predicted labels
        and if possible, classification metrics.""",
        instruction="""
        You are an agent that invokes the fine-tuned prediction_tool.
        YOUR WORKFLOW:
        1. Invoke the `dataset_info_tool` on the provided path to confirm the dataset is valid and not how many statements it contains.
        2. Invoke the `fine_tuned_predict_tool` to make predictions on input data.
        3. If provided with ground-truth labels, calculate classification metrics.
        4. Return (predictions,metrics).""",
    tools=[McpToolset(
        connection_params=SseConnectionParams(
            url=MCP_SERVER_URL,
            httpx_client_factory=MCP_CLIENT_FACTORY
        ),
        tool_filter = ["fine_tuned_predict_tool","dataset_info_tool"]
    )]
)