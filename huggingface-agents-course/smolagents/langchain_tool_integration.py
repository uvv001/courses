from langchain.agents import load_tools
from smolagents import CodeAgent, Tool
from helpers import ArgumentsHelper, register_opentelemetry_through_langfuse
register_opentelemetry_through_langfuse()

search_tool = Tool.from_langchain(load_tools(["ddg-search"])[0])

agent = CodeAgent(tools=[search_tool], model=ArgumentsHelper().getModel())

agent.run("Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences.")