import os
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
from helpers import ArgumentsHelper, register_opentelemetry_through_langfuse
register_opentelemetry_through_langfuse()

server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.13", **os.environ},
)

with ToolCollection.from_mcp(server_parameters) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=ArgumentsHelper().getModel(), add_base_tools=True)
    agent.run("Please find a remedy for hangover.")