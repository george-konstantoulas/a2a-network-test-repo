import os
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from truth_agent.sub_agents.explainer_agent import explainer_agent
from truth_agent.telemetry import instrument_asgi, setup_telemetry

setup_telemetry("explainer-agent-ntoulas")
app = instrument_asgi(to_a2a(
    explainer_agent,
    host=os.getenv("A2A_ADVERTISED_HOST", "localhost"),
    port=int(os.getenv("A2A_ADVERTISED_PORT", "8003")),
    protocol=os.getenv("A2A_ADVERTISED_PROTOCOL","http")
))
