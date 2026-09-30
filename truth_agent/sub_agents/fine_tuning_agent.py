import os

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from truth_agent.auth import mcp_httpx_client_factory

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8080/sse")
MCP_CLIENT_FACTORY = mcp_httpx_client_factory(MCP_SERVER_URL)

fine_tuning_agent = Agent(
    name="fine_tuning_agent",
    model="gemini-3.1-flash-lite",
    description="""
        Invokes fine tuning model to carry out supervised fine tuning
        of a zero-shot prediction model.""",
    instruction="""
        You are an agent that invokes the fine tuning tool.
        YOUR WORKFLOW:
        1. Invoke the `dataset_info_tool` on the provided path to confirm the dataset is valid and not how many statements it contains.
        2. Split dataset into train-test-validation sets.
        3. Keep test set in persistent memory, according to specified
        output path.
        4. Convert train and validation sets to model appropriate format.
        5. Upload them to specified cloud location.
        6. Start the fine tuning job.
        7. Once complete, store endpoint URL to a json file.
        8. Print the endpoint URL in the console.
        If the model already has an allocated endpoint, ask the user to proceed further.""",
    tools=[McpToolset(
        connection_params=SseConnectionParams(
            url=MCP_SERVER_URL,
            httpx_client_factory=MCP_CLIENT_FACTORY
        ),
        tool_filter = ["fine_tune_tool","dataset_info_tool"]
    )]
)