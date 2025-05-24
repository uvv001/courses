from smolagents import CodeAgent, DuckDuckGoSearchTool
from helpers import ArgumentsHelper, register_opentelemetry_through_langfuse
register_opentelemetry_through_langfuse()

search_tool = DuckDuckGoSearchTool()
agent = CodeAgent(
    model=ArgumentsHelper().getModel(),
    tools=[search_tool],
)

response = agent.run(
    "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
)
print(response)