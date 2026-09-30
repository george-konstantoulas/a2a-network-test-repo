from fastmcp import FastMCP
from mcp_package.tools.predictor_tools import zero_shot_predict_tool,fine_tuned_predict_tool
from mcp_package.tools.fine_tune_tools import fine_tune_tool
from mcp_package.tools.explainer_tools import explain_tool
from mcp_package.tools.shared_tools import dataset_info_tool
import uvicorn
from starlette.middleware import Middleware
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from truth_agent.telemetry import cloud_telemetry_enabled,setup_telemetry

mcp = FastMCP("Truthfulness A2A Network Tools")

mcp.add_tool(zero_shot_predict_tool)
mcp.add_tool(fine_tuned_predict_tool)
mcp.add_tool(fine_tune_tool)
mcp.add_tool(explain_tool)
mcp.add_tool(dataset_info_tool)

def main():

    setup_telemetry("mcp-server-ntoulas")
    middleware = []
    if cloud_telemetry_enabled():
        middleware.append(Middleware(OpenTelemetryMiddleware))

    app = mcp.http_app(transport="sse",middleware=middleware)
    uvicorn.run(app,host="0.0.0.0",port=8080)

if __name__ == "__main__":
    main()