from fastmcp import FastMCP
from mcp_package.tools.predictor_tools import zero_shot_predict_tool,fine_tuned_predict_tool
from mcp_package.tools.fine_tune_tools import fine_tune_tool
from mcp_package.tools.explainer_tools import explain_tool
from mcp_package.tools.shared_tools import dataset_info_tool

mcp = FastMCP("Truthfulness A2A Network Tools")

mcp.add_tool(zero_shot_predict_tool)
mcp.add_tool(fine_tuned_predict_tool)
mcp.add_tool(fine_tune_tool)
mcp.add_tool(explain_tool)
mcp.add_tool(dataset_info_tool)

def main():

    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()