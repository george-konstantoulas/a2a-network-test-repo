import os
from google.adk.cli.fast_api import get_fast_api_app
from truth_agent.auth import install_api_key_auth
from truth_agent.telemetry import setup_telemetry

AGENTS_DIR = os.getenv("ADK_AGENTS_DIR", "/app")
SERVE_WEB_UI = os.getenv("ADK_SERVE_WEB_UI", "true").strip().lower() in (
    "1", "true", "yes"
)

setup_telemetry("orchestrator-agent-ntoulas")
app = get_fast_api_app(agents_dir=AGENTS_DIR, web=SERVE_WEB_UI)

@app.get("/health")
def health():
    return {"status": "ok"}


install_api_key_auth(app, exempt_paths=("/health",))
